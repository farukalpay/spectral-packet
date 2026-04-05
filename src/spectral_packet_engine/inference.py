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


def softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    clamped = torch.clamp(value, min=torch.finfo(value.dtype).tiny)
    return clamped + torch.log(-torch.expm1(-clamped))


def real_view(value: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(value):
        return torch.stack((value.real, value.imag), dim=-1)
    return value


def flatten_real_view(value: torch.Tensor) -> torch.Tensor:
    return real_view(value).reshape(-1)


def _central_normal_quantile(
    confidence_level: float,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    level = torch.as_tensor(confidence_level, dtype=dtype, device=device)
    sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    return sqrt_two * torch.erfinv(level)


def covariance_to_correlation(covariance: torch.Tensor) -> torch.Tensor:
    standard_deviation = torch.sqrt(torch.clamp(torch.diagonal(covariance), min=0.0))
    denominator = standard_deviation[:, None] * standard_deviation[None, :]
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    correlation = covariance / safe_denominator
    return torch.where(denominator > 0, correlation, torch.zeros_like(correlation))


def complex_magnitude_standard_deviation(mean: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(mean):
        return torch.sqrt(torch.clamp(torch.diagonal(covariance), min=0.0)).reshape(mean.shape)
    flat_mean = mean.reshape(-1)
    std_values: list[torch.Tensor] = []
    tiny = torch.finfo(mean.real.dtype).tiny
    for index, coefficient in enumerate(flat_mean):
        block = covariance[2 * index : 2 * index + 2, 2 * index : 2 * index + 2]
        magnitude = torch.abs(coefficient)
        if magnitude <= tiny:
            direction = torch.zeros(2, dtype=mean.real.dtype, device=mean.device)
        else:
            direction = torch.stack((coefficient.real / magnitude, coefficient.imag / magnitude))
        variance = direction @ block @ direction
        std_values.append(torch.sqrt(torch.clamp(variance, min=0.0)))
    return torch.stack(std_values).reshape(mean.shape)


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


@dataclass(frozen=True, slots=True)
class PosteriorConfig:
    enabled: bool = True
    noise_scale: float | None = None
    noise_floor: float = 1e-6
    fisher_damping: float = 1e-6
    confidence_level: float = 0.95
    compute_coefficient_posterior: bool = True
    compute_sensitivity: bool = True

    def __post_init__(self) -> None:
        if self.noise_scale is not None and self.noise_scale <= 0:
            raise ValueError("noise_scale must be positive when provided")
        if self.noise_floor <= 0:
            raise ValueError("noise_floor must be positive")
        if self.fisher_damping < 0:
            raise ValueError("fisher_damping must be non-negative")
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("confidence_level must lie strictly between 0 and 1")


@dataclass(frozen=True, slots=True)
class ParameterPosteriorSummary:
    parameter_names: tuple[str, ...]
    mean: torch.Tensor
    standard_deviation: torch.Tensor
    confidence_level: float
    confidence_interval_low: torch.Tensor
    confidence_interval_high: torch.Tensor
    covariance: torch.Tensor
    correlation: torch.Tensor
    fisher_information: torch.Tensor
    information_eigenvalues: torch.Tensor
    effective_rank: int
    condition_number: float
    identifiability_score: float
    noise_scale: float
    residual_rms: float


@dataclass(frozen=True, slots=True)
class CoefficientPosteriorSummary:
    mean: torch.Tensor
    real_standard_deviation: torch.Tensor
    imag_standard_deviation: torch.Tensor
    magnitude_standard_deviation: torch.Tensor


@dataclass(frozen=True, slots=True)
class SensitivityMapSummary:
    parameter_names: tuple[str, ...]
    observation_shape: tuple[int, ...]
    gradient: torch.Tensor
    one_sigma_effect: torch.Tensor
    rms_one_sigma_effect: torch.Tensor
    peak_one_sigma_effect: torch.Tensor


@dataclass(frozen=True, slots=True)
class PhysicalInferenceSummary:
    observation_mode: ObservationMode
    residual: torch.Tensor
    parameter_posterior: ParameterPosteriorSummary
    coefficient_posterior: CoefficientPosteriorSummary | None
    sensitivity: SensitivityMapSummary | None


def summarize_parameter_posterior(
    *,
    parameter_names: tuple[str, ...],
    mean: torch.Tensor,
    residual_vector: torch.Tensor,
    observation_jacobian: torch.Tensor,
    posterior_config: PosteriorConfig,
) -> ParameterPosteriorSummary:
    mean_vector = torch.as_tensor(mean).reshape(-1)
    residual_flat = torch.as_tensor(
        residual_vector,
        dtype=mean_vector.dtype,
        device=mean_vector.device,
    ).reshape(-1)
    jacobian = torch.as_tensor(
        observation_jacobian,
        dtype=mean_vector.dtype,
        device=mean_vector.device,
    ).reshape(-1, mean_vector.shape[0])
    residual_rms_tensor = torch.sqrt(torch.mean(residual_flat**2))
    if posterior_config.noise_scale is None:
        noise_scale = torch.clamp(
            residual_rms_tensor.detach(),
            min=posterior_config.noise_floor,
        )
    else:
        noise_scale = torch.as_tensor(
            posterior_config.noise_scale,
            dtype=mean_vector.dtype,
            device=mean_vector.device,
        )

    fisher_information = (jacobian.transpose(0, 1) @ jacobian) / (noise_scale**2)
    fisher_information = 0.5 * (fisher_information + fisher_information.transpose(0, 1))
    identity = torch.eye(
        mean_vector.shape[0],
        dtype=fisher_information.dtype,
        device=fisher_information.device,
    )
    precision = fisher_information + posterior_config.fisher_damping * identity
    covariance = torch.linalg.pinv(precision, hermitian=True)
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    standard_deviation = torch.sqrt(torch.clamp(torch.diagonal(covariance), min=0.0))
    z_score = _central_normal_quantile(
        posterior_config.confidence_level,
        dtype=standard_deviation.dtype,
        device=standard_deviation.device,
    )
    confidence_interval_low = mean_vector - z_score * standard_deviation
    confidence_interval_high = mean_vector + z_score * standard_deviation
    information_eigenvalues = torch.linalg.eigvalsh(fisher_information).detach()
    tolerance = (
        torch.finfo(information_eigenvalues.dtype).eps
        * max(float(torch.max(torch.abs(information_eigenvalues)).item()), 1.0)
        * max(int(mean_vector.numel()), 1)
    )
    positive_eigenvalues = information_eigenvalues[information_eigenvalues > tolerance]
    effective_rank = int(positive_eigenvalues.numel())
    if effective_rank != int(mean_vector.numel()) or effective_rank == 0:
        condition_number = float("inf")
        identifiability_score = 0.0
    else:
        max_eigenvalue = float(torch.max(positive_eigenvalues).item())
        min_eigenvalue = float(torch.min(positive_eigenvalues).item())
        condition_number = max_eigenvalue / min_eigenvalue
        identifiability_score = min_eigenvalue / max_eigenvalue

    return ParameterPosteriorSummary(
        parameter_names=parameter_names,
        mean=mean_vector.detach(),
        standard_deviation=standard_deviation.detach(),
        confidence_level=posterior_config.confidence_level,
        confidence_interval_low=confidence_interval_low.detach(),
        confidence_interval_high=confidence_interval_high.detach(),
        covariance=covariance.detach(),
        correlation=covariance_to_correlation(covariance).detach(),
        fisher_information=fisher_information.detach(),
        information_eigenvalues=information_eigenvalues,
        effective_rank=effective_rank,
        condition_number=condition_number,
        identifiability_score=float(identifiability_score),
        noise_scale=float(noise_scale.detach()),
        residual_rms=float(residual_rms_tensor.detach()),
    )


def linearized_covariance(
    output_jacobian: torch.Tensor,
    parameter_covariance: torch.Tensor,
) -> torch.Tensor:
    jacobian = torch.as_tensor(output_jacobian)
    covariance = jacobian @ parameter_covariance @ jacobian.transpose(0, 1)
    return 0.5 * (covariance + covariance.transpose(0, 1))


def summarize_sensitivity_map(
    *,
    parameter_names: tuple[str, ...],
    observation: torch.Tensor,
    observation_jacobian: torch.Tensor,
    parameter_standard_deviation: torch.Tensor,
) -> SensitivityMapSummary:
    observation_shape = tuple(int(size) for size in real_view(observation).shape)
    jacobian = torch.as_tensor(observation_jacobian).reshape(-1, len(parameter_names))
    standard_deviation = torch.as_tensor(
        parameter_standard_deviation,
        dtype=jacobian.dtype,
        device=jacobian.device,
    ).reshape(len(parameter_names))
    gradient = jacobian.transpose(0, 1).reshape(
        len(parameter_names),
        *observation_shape,
    )
    broadcast_shape = (len(parameter_names),) + (1,) * len(observation_shape)
    one_sigma_effect = gradient * standard_deviation.reshape(broadcast_shape)
    flattened_effect = torch.abs(one_sigma_effect).reshape(len(parameter_names), -1)
    return SensitivityMapSummary(
        parameter_names=parameter_names,
        observation_shape=observation_shape,
        gradient=gradient.detach(),
        one_sigma_effect=one_sigma_effect.detach(),
        rms_one_sigma_effect=torch.sqrt(torch.mean(flattened_effect**2, dim=-1)).detach(),
        peak_one_sigma_effect=torch.max(flattened_effect, dim=-1).values.detach(),
    )


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
        width_raw = softplus_inverse(
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

    def _parameter_names(self, *, optimize_phase: bool) -> tuple[str, ...]:
        names = ["center", "width", "wavenumber"]
        if optimize_phase:
            names.append("phase")
        return tuple(names)

    def _parameter_vector(
        self,
        parameters: GaussianPacketParameters,
        *,
        optimize_phase: bool,
    ) -> torch.Tensor:
        single = self._coerce_single_packet(parameters)
        values = [single.center[0], single.width[0], single.wavenumber[0]]
        if optimize_phase:
            values.append(single.phase[0])
        return torch.stack(values)

    def _parameters_from_vector(
        self,
        vector: torch.Tensor,
        *,
        optimize_phase: bool,
        fixed_phase: torch.Tensor,
    ) -> GaussianPacketParameters:
        phase = vector[3] if optimize_phase else fixed_phase
        return GaussianPacketParameters.single(
            center=vector[0],
            width=vector[1],
            wavenumber=vector[2],
            phase=phase,
            dtype=self.domain.real_dtype,
            device=self.domain.device,
        )

    def _predict_from_vector(
        self,
        vector: torch.Tensor,
        *,
        optimize_phase: bool,
        fixed_phase: torch.Tensor,
        observation_grid,
        times,
        observation_mode: ObservationMode,
        observation_operator: ObservationOperator | None,
    ) -> torch.Tensor:
        return self.predict(
            self._parameters_from_vector(
                vector,
                optimize_phase=optimize_phase,
                fixed_phase=fixed_phase,
            ),
            observation_grid=observation_grid,
            times=times,
            observation_mode=observation_mode,
            observation_operator=observation_operator,
        )

    def _coefficients_from_vector(
        self,
        vector: torch.Tensor,
        *,
        optimize_phase: bool,
        fixed_phase: torch.Tensor,
    ) -> torch.Tensor:
        parameters = self._parameters_from_vector(
            vector,
            optimize_phase=optimize_phase,
            fixed_phase=fixed_phase,
        )
        packet = make_truncated_gaussian_packet(
            self.domain,
            center=parameters.center[0],
            width=parameters.width[0],
            wavenumber=parameters.wavenumber[0],
            phase=parameters.phase[0],
        )
        return self.projector.project_packet(packet).coefficients

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

    def infer(
        self,
        parameters: GaussianPacketParameters,
        *,
        observation_grid,
        times,
        target,
        observation_mode: ObservationMode = "density",
        observation_operator: ObservationOperator | None = None,
        posterior_config: PosteriorConfig = PosteriorConfig(),
    ) -> PhysicalInferenceSummary:
        if not posterior_config.enabled:
            raise ValueError("posterior inference is disabled for this request")

        optimize_phase = observation_mode == "field" or observation_operator is not None
        fitted_parameters = self._coerce_single_packet(parameters)
        fixed_phase = fitted_parameters.phase[0]
        mean_vector = self._parameter_vector(
            fitted_parameters,
            optimize_phase=optimize_phase,
        ).detach()
        differentiable_mean = mean_vector.clone().requires_grad_(True)

        prediction = self._predict_from_vector(
            differentiable_mean,
            optimize_phase=optimize_phase,
            fixed_phase=fixed_phase,
            observation_grid=observation_grid,
            times=times,
            observation_mode=observation_mode,
            observation_operator=observation_operator,
        )
        target_tensor = torch.as_tensor(target, device=prediction.device).to(dtype=prediction.dtype)
        residual = prediction - target_tensor
        residual_vector = flatten_real_view(residual)

        def observation_vector_fn(vector: torch.Tensor) -> torch.Tensor:
            prediction_vector = self._predict_from_vector(
                vector,
                optimize_phase=optimize_phase,
                fixed_phase=fixed_phase,
                observation_grid=observation_grid,
                times=times,
                observation_mode=observation_mode,
                observation_operator=observation_operator,
            )
            return flatten_real_view(prediction_vector)

        observation_jacobian = torch.autograd.functional.jacobian(observation_vector_fn, differentiable_mean)
        observation_jacobian = observation_jacobian.reshape(-1, differentiable_mean.shape[0]).detach()

        parameter_names = self._parameter_names(optimize_phase=optimize_phase)
        parameter_posterior = summarize_parameter_posterior(
            parameter_names=parameter_names,
            mean=mean_vector,
            residual_vector=residual_vector,
            observation_jacobian=observation_jacobian,
            posterior_config=posterior_config,
        )

        coefficient_posterior: CoefficientPosteriorSummary | None = None
        if posterior_config.compute_coefficient_posterior:
            coefficient_mean = self._coefficients_from_vector(
                mean_vector,
                optimize_phase=optimize_phase,
                fixed_phase=fixed_phase,
            ).detach()

            def coefficient_vector_fn(vector: torch.Tensor) -> torch.Tensor:
                return flatten_real_view(
                    self._coefficients_from_vector(
                        vector,
                        optimize_phase=optimize_phase,
                        fixed_phase=fixed_phase,
                    )
                )

            coefficient_jacobian = torch.autograd.functional.jacobian(coefficient_vector_fn, differentiable_mean)
            coefficient_jacobian = coefficient_jacobian.reshape(-1, differentiable_mean.shape[0]).detach()
            coefficient_covariance = linearized_covariance(
                coefficient_jacobian,
                parameter_posterior.covariance,
            )
            coefficient_std_view = torch.sqrt(torch.clamp(torch.diagonal(coefficient_covariance), min=0.0)).reshape(
                real_view(coefficient_mean).shape
            )
            if torch.is_complex(coefficient_mean):
                real_std = coefficient_std_view[..., 0]
                imag_std = coefficient_std_view[..., 1]
            else:
                real_std = coefficient_std_view
                imag_std = torch.zeros_like(real_std)
            coefficient_posterior = CoefficientPosteriorSummary(
                mean=coefficient_mean,
                real_standard_deviation=real_std.detach(),
                imag_standard_deviation=imag_std.detach(),
                magnitude_standard_deviation=complex_magnitude_standard_deviation(
                    coefficient_mean,
                    coefficient_covariance,
                ).detach(),
            )

        sensitivity: SensitivityMapSummary | None = None
        if posterior_config.compute_sensitivity:
            sensitivity = summarize_sensitivity_map(
                parameter_names=parameter_names,
                observation=prediction,
                observation_jacobian=observation_jacobian,
                parameter_standard_deviation=parameter_posterior.standard_deviation,
            )

        return PhysicalInferenceSummary(
            observation_mode=observation_mode,
            residual=residual.detach(),
            parameter_posterior=parameter_posterior,
            coefficient_posterior=coefficient_posterior,
            sensitivity=sensitivity,
        )


PacketDomain = InfiniteWell1D
PacketParameters = GaussianPacketParameters
SpectralPacketInverseModel = GaussianPacketEstimator


__all__ = [
    "CoefficientPosteriorSummary",
    "EstimationConfig",
    "complex_magnitude_standard_deviation",
    "covariance_to_correlation",
    "flatten_real_view",
    "GaussianPacketEstimator",
    "linearized_covariance",
    "ObservationMode",
    "PacketDomain",
    "PacketParameters",
    "ParameterPosteriorSummary",
    "PhysicalInferenceSummary",
    "PosteriorConfig",
    "ReconstructionResult",
    "real_view",
    "SensitivityMapSummary",
    "softplus_inverse",
    "SpectralPacketInverseModel",
    "summarize_parameter_posterior",
    "summarize_sensitivity_map",
]
