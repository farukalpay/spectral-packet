from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn.functional as F

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.domain import InfiniteWell1D
from spectral_packet_engine.projector import ProjectionConfig, StateProjector
from spectral_packet_engine.simulation import SimulationRecord, simulate
from spectral_packet_engine.state import GaussianPacketParameters, make_truncated_gaussian_packet
from spectral_packet_engine.dynamics import SpectralPropagator

ObservationMode = Literal["field", "density"]
ObservationOperator = Callable[[SimulationRecord], torch.Tensor]


def _softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    clamped = torch.clamp(value, min=torch.finfo(value.dtype).tiny)
    return clamped + torch.log(-torch.expm1(-clamped))


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    parameters: GaussianPacketParameters
    final_loss: float
    history: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EstimationConfig:
    steps: int = 200
    learning_rate: float = 0.05
    sigma_floor: float = 1e-6

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.sigma_floor <= 0:
            raise ValueError("sigma_floor must be positive")


class GaussianPacketEstimator:
    def __init__(
        self,
        domain: InfiniteWell1D,
        *,
        basis: InfiniteWellBasis | None = None,
        num_modes: int | None = None,
        projection_config: ProjectionConfig = ProjectionConfig(quadrature_points=2048),
        estimation_config: EstimationConfig = EstimationConfig(),
    ) -> None:
        if basis is None:
            if num_modes is None:
                raise ValueError("either basis or num_modes must be provided")
            basis = InfiniteWellBasis(domain, num_modes)
        self.domain = domain
        self.basis = basis
        self.projector = StateProjector(basis, projection_config)
        self.propagator = SpectralPropagator(basis)
        self.estimation_config = estimation_config

    def _coerce_single_packet(self, parameters: GaussianPacketParameters) -> GaussianPacketParameters:
        single = parameters.to(dtype=self.domain.real_dtype, device=self.domain.device)
        if single.packet_count != 1:
            raise ValueError("the estimator currently fits a single Gaussian packet")
        return single

    def _encode_parameters(
        self,
        parameters: GaussianPacketParameters,
        *,
        optimize_phase: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single = self._coerce_single_packet(parameters)
        eps = torch.finfo(self.domain.real_dtype).eps
        scaled_center = torch.clamp(
            (single.center[0] - self.domain.left) / self.domain.length,
            min=eps,
            max=1 - eps,
        )
        center_raw = torch.logit(scaled_center)
        width_raw = _softplus_inverse(
            torch.clamp(
                single.width[0] - self.estimation_config.sigma_floor,
                min=torch.finfo(self.domain.real_dtype).tiny,
            )
        )

        if optimize_phase:
            raw = torch.stack([center_raw, width_raw, single.wavenumber[0], single.phase[0]])
            fixed_phase = torch.zeros((), dtype=self.domain.real_dtype, device=self.domain.device)
        else:
            raw = torch.stack([center_raw, width_raw, single.wavenumber[0]])
            fixed_phase = single.phase[0]
        return raw, fixed_phase

    def _decode_parameters(
        self,
        raw: torch.Tensor,
        *,
        optimize_phase: bool,
        fixed_phase: torch.Tensor,
    ) -> GaussianPacketParameters:
        center = self.domain.left + self.domain.length * torch.sigmoid(raw[0])
        width = F.softplus(raw[1]) + self.estimation_config.sigma_floor
        wavenumber = raw[2]
        phase = raw[3] if optimize_phase else fixed_phase
        return GaussianPacketParameters.single(
            center=center,
            width=width,
            wavenumber=wavenumber,
            phase=phase,
            dtype=self.domain.real_dtype,
            device=self.domain.device,
        )

    def predict_record(
        self,
        parameters: GaussianPacketParameters,
        *,
        observation_grid,
        times,
    ) -> SimulationRecord:
        packet_parameters = self._coerce_single_packet(parameters)
        packet = make_truncated_gaussian_packet(
            self.domain,
            center=packet_parameters.center[0],
            width=packet_parameters.width[0],
            wavenumber=packet_parameters.wavenumber[0],
            phase=packet_parameters.phase[0],
        )
        return simulate(
            packet,
            times,
            projector=self.projector,
            propagator=self.propagator,
            grid=observation_grid,
        )

    def predict(
        self,
        parameters: GaussianPacketParameters,
        *,
        observation_grid,
        times,
        observation_mode: ObservationMode = "density",
        observation_operator: ObservationOperator | None = None,
    ) -> torch.Tensor:
        record = self.predict_record(parameters, observation_grid=observation_grid, times=times)
        if observation_operator is not None:
            return observation_operator(record)
        if observation_mode == "field":
            if record.wavefunctions is None:
                raise ValueError("prediction requires a reconstruction grid")
            return record.wavefunctions
        if observation_mode == "density":
            if record.densities is None:
                raise ValueError("prediction requires a reconstruction grid")
            return record.densities
        raise ValueError(f"unsupported observation_mode: {observation_mode}")

    def loss(
        self,
        parameters: GaussianPacketParameters,
        *,
        observation_grid,
        times,
        target,
        observation_mode: ObservationMode = "density",
        observation_operator: ObservationOperator | None = None,
    ) -> torch.Tensor:
        prediction = self.predict(
            parameters,
            observation_grid=observation_grid,
            times=times,
            observation_mode=observation_mode,
            observation_operator=observation_operator,
        )
        target_tensor = torch.as_tensor(target, device=prediction.device)
        if torch.is_complex(prediction):
            target_tensor = target_tensor.to(dtype=prediction.dtype)
            residual = prediction - target_tensor
            return torch.mean(torch.abs(residual) ** 2)
        target_tensor = target_tensor.to(dtype=prediction.dtype)
        residual = prediction - target_tensor
        return torch.mean(residual**2)

    def fit(
        self,
        *,
        observation_grid,
        times,
        target,
        initial_guess: GaussianPacketParameters,
        observation_mode: ObservationMode = "density",
        observation_operator: ObservationOperator | None = None,
        steps: int | None = None,
        learning_rate: float | None = None,
    ) -> ReconstructionResult:
        optimize_phase = observation_mode == "field" or observation_operator is not None
        raw_initial, fixed_phase = self._encode_parameters(initial_guess, optimize_phase=optimize_phase)
        raw = torch.nn.Parameter(raw_initial.clone())
        optimizer = torch.optim.Adam(
            [raw],
            lr=learning_rate or self.estimation_config.learning_rate,
        )
        history: list[float] = []

        total_steps = steps or self.estimation_config.steps
        for _ in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            parameters = self._decode_parameters(
                raw,
                optimize_phase=optimize_phase,
                fixed_phase=fixed_phase,
            )
            loss = self.loss(
                parameters,
                observation_grid=observation_grid,
                times=times,
                target=target,
                observation_mode=observation_mode,
                observation_operator=observation_operator,
            )
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))

        final_parameters = self._decode_parameters(
            raw.detach(),
            optimize_phase=optimize_phase,
            fixed_phase=fixed_phase,
        )
        return ReconstructionResult(
            parameters=final_parameters,
            final_loss=history[-1],
            history=tuple(history),
        )


PacketDomain = InfiniteWell1D
PacketParameters = GaussianPacketParameters
SpectralPacketInverseModel = GaussianPacketEstimator


__all__ = [
    "EstimationConfig",
    "GaussianPacketEstimator",
    "ObservationMode",
    "PacketDomain",
    "PacketParameters",
    "ReconstructionResult",
    "SpectralPacketInverseModel",
]
