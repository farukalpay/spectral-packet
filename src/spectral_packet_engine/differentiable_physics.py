from __future__ import annotations

"""Differentiable physics workflows built directly on the spectral core."""

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.domain import InfiniteWell1D
from spectral_packet_engine.dynamics import SpectralPropagator
from spectral_packet_engine.eigensolver import solve_eigenproblem
from spectral_packet_engine.inference import (
    ObservationMode,
    ParameterPosteriorSummary,
    PosteriorConfig,
    SensitivityMapSummary,
    flatten_real_view,
    softplus_inverse,
    summarize_parameter_posterior,
    summarize_sensitivity_map,
)
from spectral_packet_engine.observables import expectation_position
from spectral_packet_engine.parametric_potentials import PotentialFamilyDefinition, PotentialParameterSpec, resolve_potential_family
from spectral_packet_engine.projector import ProjectionConfig, StateProjector
from spectral_packet_engine.runtime import TorchRuntime, inspect_torch_runtime
from spectral_packet_engine.simulation import simulate
from spectral_packet_engine.state import GaussianPacketParameters, make_truncated_gaussian_packet

Tensor = torch.Tensor
ControlObjective = Literal["target_position", "target_interval_probability"]


@dataclass(frozen=True, slots=True)
class GradientOptimizationConfig:
    steps: int = 200
    learning_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True, slots=True)
class PotentialCalibrationSummary:
    family: str
    parameter_names: tuple[str, ...]
    estimated_parameters: dict[str, float]
    target_eigenvalues: Tensor
    predicted_eigenvalues: Tensor
    transition_energies: Tensor
    final_loss: float
    history: tuple[float, ...]
    parameter_posterior: ParameterPosteriorSummary | None
    sensitivity: SensitivityMapSummary | None
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransitionDesignSummary:
    family: str
    parameter_names: tuple[str, ...]
    optimized_parameters: dict[str, float]
    target_transition: float
    achieved_transition: float
    transition_indices: tuple[int, int]
    predicted_eigenvalues: Tensor
    final_loss: float
    history: tuple[float, ...]
    gradient: Tensor
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObservableGradientSummary:
    parameter_names: tuple[str, ...]
    parameter_values: Tensor
    objective_name: str
    objective_value: float
    gradient: Tensor


@dataclass(frozen=True, slots=True)
class PacketControlOptimizationSummary:
    objective: ControlObjective
    optimized_parameters: GaussianPacketParameters
    target_value: float
    achieved_value: float
    final_loss: float
    history: tuple[float, ...]
    gradient_summary: ObservableGradientSummary
    runtime: TorchRuntime
    observation_grid: Tensor
    final_time: float
    final_density: Tensor
    final_expectation_position: float
    final_interval_probability: float | None
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ScalarParameterSpec:
    name: str
    lower_bound: float | None = None
    upper_bound: float | None = None


class _ConstrainedParameterization:
    def __init__(
        self,
        specs: Sequence[PotentialParameterSpec | _ScalarParameterSpec],
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        domain: InfiniteWell1D | None = None,
    ) -> None:
        self.specs = tuple(specs)
        self.dtype = dtype
        self.device = device
        self.domain = domain

    def _bounds(self, spec: PotentialParameterSpec | _ScalarParameterSpec) -> tuple[Tensor | None, Tensor | None]:
        if isinstance(spec, PotentialParameterSpec):
            lower, upper = spec.resolved_bounds(self.domain)
        else:
            lower = None if spec.lower_bound is None else torch.as_tensor(spec.lower_bound)
            upper = None if spec.upper_bound is None else torch.as_tensor(spec.upper_bound)
        if lower is not None:
            lower = lower.to(dtype=self.dtype, device=self.device)
        if upper is not None:
            upper = upper.to(dtype=self.dtype, device=self.device)
        return lower, upper

    def encode(self, parameters: Mapping[str, float | Tensor]) -> Tensor:
        raw_values: list[Tensor] = []
        eps = torch.finfo(self.dtype).eps
        for spec in self.specs:
            if spec.name not in parameters:
                raise KeyError(f"missing parameter '{spec.name}'")
            value = torch.as_tensor(parameters[spec.name], dtype=self.dtype, device=self.device).reshape(())
            lower, upper = self._bounds(spec)
            if lower is not None and upper is not None:
                scaled = torch.clamp((value - lower) / (upper - lower), min=eps, max=1.0 - eps)
                raw_values.append(torch.logit(scaled))
            elif lower is not None:
                raw_values.append(softplus_inverse(torch.clamp(value - lower, min=torch.finfo(self.dtype).tiny)))
            elif upper is not None:
                raw_values.append(softplus_inverse(torch.clamp(upper - value, min=torch.finfo(self.dtype).tiny)))
            else:
                raw_values.append(value)
        return torch.stack(raw_values)

    def decode(self, raw: Tensor) -> Tensor:
        values: list[Tensor] = []
        for index, spec in enumerate(self.specs):
            lower, upper = self._bounds(spec)
            if lower is not None and upper is not None:
                values.append(lower + (upper - lower) * torch.sigmoid(raw[index]))
            elif lower is not None:
                values.append(lower + torch.nn.functional.softplus(raw[index]))
            elif upper is not None:
                values.append(upper - torch.nn.functional.softplus(raw[index]))
            else:
                values.append(raw[index])
        return torch.stack(values)

    def mapping_from_vector(self, vector: Tensor) -> dict[str, Tensor]:
        physical = torch.as_tensor(vector, dtype=self.dtype, device=self.device).reshape(len(self.specs))
        return {spec.name: physical[index].reshape(()) for index, spec in enumerate(self.specs)}


def _run_optimization(
    *,
    raw_initial: Tensor,
    config: GradientOptimizationConfig,
    objective_fn,
) -> tuple[Tensor, tuple[float, ...]]:
    raw = torch.nn.Parameter(raw_initial.clone())
    optimizer = torch.optim.Adam([raw], lr=config.learning_rate)
    history: list[float] = []
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = objective_fn(raw)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return raw.detach(), tuple(history)


def _potential_domain(
    *,
    domain_length: float,
    left: float,
    mass: float,
    hbar: float,
    device: str | torch.device,
) -> tuple[TorchRuntime, InfiniteWell1D]:
    runtime = inspect_torch_runtime(device)
    domain = InfiniteWell1D.from_length(
        domain_length,
        left=left,
        mass=mass,
        hbar=hbar,
        dtype=runtime.preferred_real_dtype,
        device=runtime.device,
    )
    return runtime, domain


def _solve_family_spectrum(
    family: PotentialFamilyDefinition,
    domain: InfiniteWell1D,
    parameter_vector: Tensor,
    *,
    num_points: int,
    num_states: int,
) -> Tensor:
    result = solve_eigenproblem(
        family.build_from_vector(domain, parameter_vector),
        domain,
        num_points=num_points,
        num_states=num_states,
    )
    return result.eigenvalues


def calibrate_potential_from_spectrum(
    *,
    family: str,
    target_eigenvalues,
    initial_guess: Mapping[str, float],
    domain_length: float = 1.0,
    left: float = 0.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    num_points: int = 128,
    optimization_config: GradientOptimizationConfig = GradientOptimizationConfig(),
    posterior_config: PosteriorConfig | None = PosteriorConfig(),
    device: str | torch.device = "auto",
) -> PotentialCalibrationSummary:
    runtime, domain = _potential_domain(
        domain_length=domain_length,
        left=left,
        mass=mass,
        hbar=hbar,
        device=device,
    )
    family_def = resolve_potential_family(family)
    target = torch.as_tensor(target_eigenvalues, dtype=domain.real_dtype, device=domain.device).reshape(-1)
    parameterization = _ConstrainedParameterization(
        family_def.parameter_specs,
        dtype=domain.real_dtype,
        device=domain.device,
        domain=domain,
    )
    raw_initial = parameterization.encode(initial_guess)

    def loss_fn(raw_vector: Tensor) -> Tensor:
        parameter_vector = parameterization.decode(raw_vector)
        predicted = _solve_family_spectrum(
            family_def,
            domain,
            parameter_vector,
            num_points=num_points,
            num_states=int(target.shape[0]),
        )
        return torch.mean((predicted - target) ** 2)

    raw_final, history = _run_optimization(
        raw_initial=raw_initial,
        config=optimization_config,
        objective_fn=loss_fn,
    )
    parameter_vector = parameterization.decode(raw_final)
    predicted = _solve_family_spectrum(
        family_def,
        domain,
        parameter_vector,
        num_points=num_points,
        num_states=int(target.shape[0]),
    )
    transition_energies = predicted[1:] - predicted[0] if predicted.shape[0] > 1 else torch.empty(0, dtype=predicted.dtype)

    parameter_posterior: ParameterPosteriorSummary | None = None
    sensitivity: SensitivityMapSummary | None = None
    resolved_posterior = posterior_config
    if resolved_posterior is not None and resolved_posterior.enabled:
        differentiable_vector = parameter_vector.detach().clone().requires_grad_(True)

        def observation_fn(vector: Tensor) -> Tensor:
            return _solve_family_spectrum(
                family_def,
                domain,
                vector,
                num_points=num_points,
                num_states=int(target.shape[0]),
            )

        observation_jacobian = torch.autograd.functional.jacobian(observation_fn, differentiable_vector)
        observation_jacobian = observation_jacobian.reshape(-1, differentiable_vector.shape[0]).detach()
        residual_vector = flatten_real_view(predicted - target)
        parameter_posterior = summarize_parameter_posterior(
            parameter_names=family_def.parameter_names,
            mean=parameter_vector,
            residual_vector=residual_vector,
            observation_jacobian=observation_jacobian,
            posterior_config=resolved_posterior,
        )
        sensitivity = summarize_sensitivity_map(
            parameter_names=family_def.parameter_names,
            observation=predicted,
            observation_jacobian=observation_jacobian,
            parameter_standard_deviation=parameter_posterior.standard_deviation,
        )

    return PotentialCalibrationSummary(
        family=family_def.name,
        parameter_names=family_def.parameter_names,
        estimated_parameters={
            name: float(value.item())
            for name, value in parameterization.mapping_from_vector(parameter_vector).items()
        },
        target_eigenvalues=target.detach(),
        predicted_eigenvalues=predicted.detach(),
        transition_energies=transition_energies.detach(),
        final_loss=history[-1],
        history=history,
        parameter_posterior=parameter_posterior,
        sensitivity=sensitivity,
        assumptions=(
            "The target data are matched by differentiating through the bounded-domain spectral eigensolver.",
            "Posterior uncertainty is a local Laplace approximation around the calibrated parameter vector, not a global Bayesian posterior over model families.",
        ),
    )


def design_potential_for_target_transition(
    *,
    family: str,
    target_transition: float,
    transition_indices: tuple[int, int] = (0, 1),
    initial_guess: Mapping[str, float],
    domain_length: float = 1.0,
    left: float = 0.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    num_points: int = 128,
    num_states: int = 4,
    optimization_config: GradientOptimizationConfig = GradientOptimizationConfig(),
    device: str | torch.device = "auto",
) -> TransitionDesignSummary:
    lower_state, upper_state = transition_indices
    if upper_state <= lower_state:
        raise ValueError("transition_indices must satisfy upper > lower")

    _, domain = _potential_domain(
        domain_length=domain_length,
        left=left,
        mass=mass,
        hbar=hbar,
        device=device,
    )
    family_def = resolve_potential_family(family)
    parameterization = _ConstrainedParameterization(
        family_def.parameter_specs,
        dtype=domain.real_dtype,
        device=domain.device,
        domain=domain,
    )
    raw_initial = parameterization.encode(initial_guess)
    target_transition_tensor = torch.as_tensor(target_transition, dtype=domain.real_dtype, device=domain.device)

    def transition_fn(vector: Tensor) -> Tensor:
        spectrum = _solve_family_spectrum(
            family_def,
            domain,
            vector,
            num_points=num_points,
            num_states=max(num_states, upper_state + 1),
        )
        return spectrum[upper_state] - spectrum[lower_state]

    def loss_fn(raw_vector: Tensor) -> Tensor:
        parameter_vector = parameterization.decode(raw_vector)
        achieved = transition_fn(parameter_vector)
        return (achieved - target_transition_tensor) ** 2

    raw_final, history = _run_optimization(
        raw_initial=raw_initial,
        config=optimization_config,
        objective_fn=loss_fn,
    )
    parameter_vector = parameterization.decode(raw_final)
    predicted_eigenvalues = _solve_family_spectrum(
        family_def,
        domain,
        parameter_vector,
        num_points=num_points,
        num_states=max(num_states, upper_state + 1),
    )
    achieved_transition = predicted_eigenvalues[upper_state] - predicted_eigenvalues[lower_state]
    differentiable_vector = parameter_vector.detach().clone().requires_grad_(True)
    gradient = torch.autograd.functional.jacobian(transition_fn, differentiable_vector).reshape(-1).detach()

    return TransitionDesignSummary(
        family=family_def.name,
        parameter_names=family_def.parameter_names,
        optimized_parameters={
            name: float(value.item())
            for name, value in parameterization.mapping_from_vector(parameter_vector).items()
        },
        target_transition=float(target_transition),
        achieved_transition=float(achieved_transition.detach()),
        transition_indices=transition_indices,
        predicted_eigenvalues=predicted_eigenvalues.detach(),
        final_loss=history[-1],
        history=history,
        gradient=gradient,
        assumptions=(
            "This is gradient-based inverse design over a restricted parameterized potential family.",
            "The optimized transition is computed from the bounded-domain spectrum; it is not a claim about unrestricted control over arbitrary multidimensional systems.",
        ),
    )


def _packet_parameterization(domain: InfiniteWell1D) -> _ConstrainedParameterization:
    return _ConstrainedParameterization(
        (
            _ScalarParameterSpec("center", lower_bound=float(domain.left), upper_bound=float(domain.right)),
            _ScalarParameterSpec("width", lower_bound=1e-6),
            _ScalarParameterSpec("wavenumber"),
            _ScalarParameterSpec("phase"),
        ),
        dtype=domain.real_dtype,
        device=domain.device,
    )


def _packet_observable(
    parameter_vector: Tensor,
    *,
    domain: InfiniteWell1D,
    projector: StateProjector,
    propagator: SpectralPropagator,
    observation_grid: Tensor,
    final_time: Tensor,
    objective: ControlObjective,
    target_value: Tensor,
    interval: tuple[float, float] | None,
) -> tuple[Tensor, torch.Tensor, torch.Tensor]:
    packet = make_truncated_gaussian_packet(
        domain,
        center=parameter_vector[0],
        width=parameter_vector[1],
        wavenumber=parameter_vector[2],
        phase=parameter_vector[3],
    )
    record = simulate(
        packet,
        final_time.reshape(1),
        projector=projector,
        propagator=propagator,
        grid=observation_grid,
    )
    if record.wavefunctions is None or record.densities is None:
        raise ValueError("packet control requires a reconstruction grid")
    final_wavefunction = record.wavefunctions[0]
    final_density = record.densities[0]
    final_expectation_position = expectation_position(final_wavefunction, observation_grid)
    if objective == "target_position":
        loss = (final_expectation_position - target_value) ** 2
        objective_value = final_expectation_position
    elif objective == "target_interval_probability":
        if interval is None:
            raise ValueError("interval must be provided for target_interval_probability")
        interval_probability = record.interval_probability(interval[0], interval[1])[0]
        loss = (interval_probability - target_value) ** 2
        objective_value = interval_probability
    else:
        raise ValueError(f"unsupported control objective: {objective}")
    return loss, objective_value, final_density


def compute_packet_observable_gradient(
    *,
    initial_guess: Mapping[str, float],
    objective: ControlObjective,
    target_value: float,
    final_time: float,
    interval: tuple[float, float] | None = None,
    num_modes: int = 96,
    quadrature_points: int = 2048,
    grid_points: int = 128,
    device: str | torch.device = "auto",
) -> ObservableGradientSummary:
    runtime = inspect_torch_runtime(device)
    domain = InfiniteWell1D.from_length(1.0, dtype=runtime.preferred_real_dtype, device=runtime.device)
    basis = InfiniteWellBasis(domain, num_modes)
    projector = StateProjector(basis, ProjectionConfig(quadrature_points=quadrature_points))
    propagator = SpectralPropagator(basis)
    observation_grid = domain.grid(grid_points)
    parameterization = _packet_parameterization(domain)
    parameter_vector = parameterization.decode(parameterization.encode(initial_guess)).detach().clone().requires_grad_(True)
    target_tensor = torch.as_tensor(target_value, dtype=domain.real_dtype, device=domain.device)
    final_time_tensor = torch.as_tensor(final_time, dtype=domain.real_dtype, device=domain.device)
    loss, objective_value, _ = _packet_observable(
        parameter_vector,
        domain=domain,
        projector=projector,
        propagator=propagator,
        observation_grid=observation_grid,
        final_time=final_time_tensor,
        objective=objective,
        target_value=target_tensor,
        interval=interval,
    )
    gradient = torch.autograd.grad(loss, parameter_vector)[0].detach()
    return ObservableGradientSummary(
        parameter_names=("center", "width", "wavenumber", "phase"),
        parameter_values=parameter_vector.detach(),
        objective_name=objective,
        objective_value=float(objective_value.detach()),
        gradient=gradient,
    )


def optimize_packet_control(
    *,
    initial_guess: Mapping[str, float],
    objective: ControlObjective,
    target_value: float,
    final_time: float,
    interval: tuple[float, float] | None = None,
    num_modes: int = 96,
    quadrature_points: int = 2048,
    grid_points: int = 128,
    optimization_config: GradientOptimizationConfig = GradientOptimizationConfig(),
    device: str | torch.device = "auto",
) -> PacketControlOptimizationSummary:
    runtime = inspect_torch_runtime(device)
    domain = InfiniteWell1D.from_length(1.0, dtype=runtime.preferred_real_dtype, device=runtime.device)
    basis = InfiniteWellBasis(domain, num_modes)
    projector = StateProjector(basis, ProjectionConfig(quadrature_points=quadrature_points))
    propagator = SpectralPropagator(basis)
    observation_grid = domain.grid(grid_points)
    target_tensor = torch.as_tensor(target_value, dtype=domain.real_dtype, device=domain.device)
    final_time_tensor = torch.as_tensor(final_time, dtype=domain.real_dtype, device=domain.device)
    parameterization = _packet_parameterization(domain)
    raw_initial = parameterization.encode(initial_guess)

    def loss_fn(raw_vector: Tensor) -> Tensor:
        parameter_vector = parameterization.decode(raw_vector)
        loss, _, _ = _packet_observable(
            parameter_vector,
            domain=domain,
            projector=projector,
            propagator=propagator,
            observation_grid=observation_grid,
            final_time=final_time_tensor,
            objective=objective,
            target_value=target_tensor,
            interval=interval,
        )
        return loss

    raw_final, history = _run_optimization(
        raw_initial=raw_initial,
        config=optimization_config,
        objective_fn=loss_fn,
    )
    parameter_vector = parameterization.decode(raw_final)
    loss, objective_value, final_density = _packet_observable(
        parameter_vector,
        domain=domain,
        projector=projector,
        propagator=propagator,
        observation_grid=observation_grid,
        final_time=final_time_tensor,
        objective=objective,
        target_value=target_tensor,
        interval=interval,
    )
    packet = make_truncated_gaussian_packet(
        domain,
        center=parameter_vector[0],
        width=parameter_vector[1],
        wavenumber=parameter_vector[2],
        phase=parameter_vector[3],
    )
    record = simulate(
        packet,
        final_time_tensor.reshape(1),
        projector=projector,
        propagator=propagator,
        grid=observation_grid,
    )
    if record.wavefunctions is None:
        raise ValueError("packet control requires a reconstruction grid")
    final_expectation_position = float(expectation_position(record.wavefunctions[0], observation_grid))
    final_interval_probability = None
    if interval is not None:
        final_interval_probability = float(record.interval_probability(interval[0], interval[1])[0])

    gradient_summary = compute_packet_observable_gradient(
        initial_guess={
            "center": float(parameter_vector[0].item()),
            "width": float(parameter_vector[1].item()),
            "wavenumber": float(parameter_vector[2].item()),
            "phase": float(parameter_vector[3].item()),
        },
        objective=objective,
        target_value=target_value,
        final_time=final_time,
        interval=interval,
        num_modes=num_modes,
        quadrature_points=quadrature_points,
        grid_points=grid_points,
        device=runtime.device,
    )

    return PacketControlOptimizationSummary(
        objective=objective,
        optimized_parameters=GaussianPacketParameters.single(
            center=parameter_vector[0],
            width=parameter_vector[1],
            wavenumber=parameter_vector[2],
            phase=parameter_vector[3],
            dtype=domain.real_dtype,
            device=domain.device,
        ),
        target_value=float(target_value),
        achieved_value=float(objective_value.detach()),
        final_loss=float(loss.detach()),
        history=history,
        gradient_summary=gradient_summary,
        runtime=runtime,
        observation_grid=observation_grid.detach(),
        final_time=float(final_time),
        final_density=final_density.detach(),
        final_expectation_position=final_expectation_position,
        final_interval_probability=final_interval_probability,
        assumptions=(
            "This workflow optimizes initial packet preparation parameters, not an arbitrary time-dependent control pulse.",
            "Gradients are computed through the bounded-domain spectral projection and propagation stack implemented in PyTorch.",
        ),
    )


__all__ = [
    "ControlObjective",
    "GradientOptimizationConfig",
    "ObservableGradientSummary",
    "PacketControlOptimizationSummary",
    "PotentialCalibrationSummary",
    "TransitionDesignSummary",
    "calibrate_potential_from_spectrum",
    "compute_packet_observable_gradient",
    "design_potential_for_target_transition",
    "optimize_packet_control",
]
