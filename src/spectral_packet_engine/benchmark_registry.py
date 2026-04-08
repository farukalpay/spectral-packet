from __future__ import annotations

"""Official benchmark registry for the bounded-domain spectral engine.

The registry is intentionally a product-level measurement suite, not a
workflow router.  Each case declares the physical contract, mode budget, and
honest limits it measures, while the runner reports the same error, runtime,
memory, identifiability, and backend-comparison shape for every case.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import math
from pathlib import Path
import time
import tracemalloc
from typing import Any, Literal

import numpy as np
import torch

from spectral_packet_engine.differentiable_physics import (
    GradientOptimizationConfig,
    calibrate_potential_from_spectrum,
)
from spectral_packet_engine.domain import InfiniteWell1D
from spectral_packet_engine.eigensolver import EigensolverResult, solve_eigenproblem
from spectral_packet_engine.parametric_potentials import resolve_potential_family
from spectral_packet_engine.physics_contracts import BasisSpec, build_hamiltonian_operator
from spectral_packet_engine.profiles import compress_profiles, normalize_profiles, relative_l2_error
from spectral_packet_engine.reduced_models import analyze_separable_tensor_product_spectrum
from spectral_packet_engine.runtime import TorchRuntime, inspect_torch_runtime
from spectral_packet_engine.scattering import rectangular_barrier, scattering_spectrum
from spectral_packet_engine.uq import PosteriorConfig
from spectral_packet_engine.workflows import simulate_gaussian_packet

Tensor = torch.Tensor
BenchmarkStatus = Literal["passed", "failed", "skipped"]


@dataclass(frozen=True, slots=True)
class BenchmarkCaseDefinition:
    case_id: str
    label: str
    category: str
    objective: str
    physical_contract: str
    default_mode_budget: dict[str, Any]
    reported_metrics: tuple[str, ...]
    honest_limits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkCaseMetrics:
    score: float
    error: dict[str, float]
    timing: dict[str, float]
    memory: dict[str, Any]
    mode_budget: dict[str, Any]
    identifiability: dict[str, Any]
    backend: dict[str, Any]
    backend_comparison: dict[str, Any]
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BenchmarkCaseResult:
    definition: BenchmarkCaseDefinition
    status: BenchmarkStatus
    metrics: BenchmarkCaseMetrics | None
    assumptions: tuple[str, ...]
    message: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkRegistryReport:
    suite_name: str
    suite_version: str
    runtime: TorchRuntime
    case_results: tuple[BenchmarkCaseResult, ...]
    summary: dict[str, Any]
    assumptions: tuple[str, ...]

    def write_artifacts(
        self,
        output_dir: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ):
        from spectral_packet_engine.artifacts import (
            inspect_artifact_directory,
            write_benchmark_registry_artifacts,
        )

        write_benchmark_registry_artifacts(output_dir, self, metadata=metadata)
        return inspect_artifact_directory(output_dir)


@dataclass(frozen=True, slots=True)
class _CasePayload:
    score: float
    error: dict[str, float]
    mode_budget: dict[str, Any]
    identifiability: dict[str, Any]
    details: dict[str, Any]
    assumptions: tuple[str, ...]
    timing: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class _ResourceUsage:
    elapsed_seconds: float
    python_current_bytes: int
    python_peak_bytes: int
    accelerator_peak_bytes: int | None
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _BenchmarkCase:
    definition: BenchmarkCaseDefinition
    runner: Callable[[TorchRuntime], _CasePayload]


def _float(value: Any) -> float:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.numel() != 1:
        raise ValueError("expected a scalar value")
    return float(tensor.item())


def _tensor_list(values: Tensor) -> list[float]:
    return [float(item) for item in torch.as_tensor(values).detach().cpu().reshape(-1).tolist()]


def _domain_for_runtime(runtime: TorchRuntime, *, length: float = 1.0, left: float = 0.0) -> InfiniteWell1D:
    return InfiniteWell1D.from_length(
        length,
        left=left,
        dtype=runtime.preferred_real_dtype,
        device=runtime.device,
    )


def _rms_relative_error(values: Tensor, reference: Tensor) -> Tensor:
    values = torch.as_tensor(values)
    reference = torch.as_tensor(reference, dtype=values.dtype, device=values.device)
    common = min(int(values.numel()), int(reference.numel()))
    if common == 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    values = values.reshape(-1)[:common]
    reference = reference.reshape(-1)[:common]
    denominator = torch.clamp(torch.abs(reference), min=1.0)
    return torch.sqrt(torch.mean(((values - reference) / denominator) ** 2))


def _solve_contract_spectrum(
    runtime: TorchRuntime,
    *,
    family: str,
    parameters: Mapping[str, float],
    modes: int,
    quadrature_points: int,
    states: int,
) -> EigensolverResult:
    domain = _domain_for_runtime(runtime)
    operator = build_hamiltonian_operator(
        family,
        domain=domain,
        parameters=parameters,
        basis_spec=BasisSpec(num_modes=modes, quadrature_points=quadrature_points),
    )
    return operator.solve(num_states=states)


def _summarize_jacobian_identifiability(
    jacobian: Tensor,
    *,
    parameter_names: tuple[str, ...],
    method: str,
    notes: tuple[str, ...] = (),
) -> dict[str, Any]:
    matrix = torch.as_tensor(jacobian).detach()
    if matrix.ndim != 2:
        matrix = matrix.reshape(-1, len(parameter_names))
    singular_values = torch.linalg.svdvals(matrix)
    max_singular = float(torch.max(torch.abs(singular_values)).item()) if singular_values.numel() else 0.0
    tolerance = torch.finfo(singular_values.dtype).eps * max(max_singular, 1.0) * max(matrix.shape)
    positive = singular_values[singular_values > tolerance]
    effective_rank = int(positive.numel())
    if positive.numel() == len(parameter_names) and positive.numel() > 0:
        min_singular = float(torch.min(positive).item())
        condition_number = max_singular / max(min_singular, torch.finfo(singular_values.dtype).tiny)
        identifiability_score = min_singular / max(max_singular, torch.finfo(singular_values.dtype).tiny)
    else:
        condition_number = float("inf")
        identifiability_score = 0.0
    return {
        "method": method,
        "parameter_names": list(parameter_names),
        "parameter_count": len(parameter_names),
        "observation_count": int(matrix.shape[0]),
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "identifiability_score": float(identifiability_score),
        "singular_values": _tensor_list(singular_values),
        "notes": list(notes),
    }


def _spectrum_identifiability(
    runtime: TorchRuntime,
    *,
    family: str,
    parameters: Mapping[str, float],
    modes: int,
    states: int,
) -> dict[str, Any]:
    family_def = resolve_potential_family(family)
    domain = _domain_for_runtime(runtime)
    vector = family_def.vector_from_mapping(
        parameters,
        domain=domain,
        dtype=domain.real_dtype,
        device=domain.device,
    ).detach()

    def observation_fn(parameter_vector: Tensor) -> Tensor:
        return solve_eigenproblem(
            family_def.build_from_vector(domain, parameter_vector),
            domain,
            num_points=modes,
            num_states=states,
        ).eigenvalues

    try:
        jacobian = torch.autograd.functional.jacobian(
            observation_fn,
            vector.clone().requires_grad_(True),
        ).reshape(states, len(family_def.parameter_names))
    except RuntimeError as exc:
        return {
            "method": "local spectrum Jacobian",
            "status": "failed",
            "parameter_names": list(family_def.parameter_names),
            "parameter_count": len(family_def.parameter_names),
            "observation_count": int(states),
            "message": str(exc),
        }
    return _summarize_jacobian_identifiability(
        jacobian,
        parameter_names=family_def.parameter_names,
        method="local spectrum Jacobian",
        notes=("Computed at the benchmark parameter point; this is local identifiability, not global posterior geometry.",),
    )


def _finite_difference_identifiability(
    *,
    parameter_names: tuple[str, ...],
    values: Sequence[float],
    observation_fn: Callable[[Tensor], Tensor],
    relative_step: float = 1e-4,
) -> dict[str, Any]:
    base = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    base_observation = observation_fn(base).detach().to(dtype=torch.float64).reshape(-1)
    columns: list[Tensor] = []
    for index, value in enumerate(base):
        step = relative_step * max(abs(float(value.item())), 1.0)
        upper = base.clone()
        lower = base.clone()
        upper[index] = upper[index] + step
        lower[index] = lower[index] - step
        column = (observation_fn(upper) - observation_fn(lower)).detach().to(dtype=torch.float64).reshape(-1)
        columns.append(column / (2.0 * step))
    jacobian = torch.stack(columns, dim=1) if columns else torch.empty(base_observation.numel(), 0)
    return _summarize_jacobian_identifiability(
        jacobian,
        parameter_names=parameter_names,
        method="central finite-difference observation Jacobian",
        notes=(f"relative_step={relative_step:g}", "Used because this scattering benchmark is transfer-matrix based rather than autograd based."),
    )


def _linear_condition_identifiability(
    matrix: Tensor,
    *,
    parameter_count: int,
    observation_count: int,
    method: str,
) -> dict[str, Any]:
    singular_values = torch.linalg.svdvals(torch.as_tensor(matrix).detach())
    max_singular = float(torch.max(singular_values).item()) if singular_values.numel() else 0.0
    tolerance = torch.finfo(singular_values.dtype).eps * max(max_singular, 1.0) * max(parameter_count, observation_count)
    positive = singular_values[singular_values > tolerance]
    effective_rank = int(positive.numel())
    if positive.numel() >= parameter_count and positive.numel() > 0:
        min_singular = float(torch.min(positive).item())
        condition_number = max_singular / max(min_singular, torch.finfo(singular_values.dtype).tiny)
        score = min_singular / max(max_singular, torch.finfo(singular_values.dtype).tiny)
    else:
        condition_number = float("inf")
        score = 0.0
    return {
        "method": method,
        "parameter_count": int(parameter_count),
        "observation_count": int(observation_count),
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "identifiability_score": float(score),
        "singular_values": _tensor_list(singular_values),
    }


def _posterior_identifiability(calibration: Any) -> dict[str, Any]:
    posterior = calibration.parameter_posterior
    if posterior is None:
        return {
            "method": "local posterior Fisher information",
            "status": "unavailable",
            "parameter_names": list(calibration.parameter_names),
        }
    return {
        "method": "local posterior Fisher information",
        "parameter_names": list(posterior.parameter_names),
        "parameter_count": len(posterior.parameter_names),
        "observation_count": int(calibration.target_eigenvalues.numel()),
        "effective_rank": posterior.effective_rank,
        "condition_number": posterior.condition_number,
        "identifiability_score": posterior.identifiability_score,
        "information_eigenvalues": _tensor_list(posterior.information_eigenvalues),
        "residual_rms": posterior.residual_rms,
        "noise_scale": posterior.noise_scale,
    }


def _runtime_payload(runtime: TorchRuntime) -> dict[str, Any]:
    return {
        "requested_device": runtime.requested_device,
        "device": str(runtime.device),
        "backend": runtime.backend,
        "accelerator": runtime.accelerator,
        "available_backends": list(runtime.available_backends),
        "preferred_real_dtype": str(runtime.preferred_real_dtype).replace("torch.", ""),
        "supports_float64": runtime.supports_float64,
        "num_threads": runtime.num_threads,
        "notes": list(runtime.notes),
    }


def _measure_payload(fn: Callable[[], _CasePayload], runtime: TorchRuntime) -> tuple[_CasePayload, _ResourceUsage]:
    notes: list[str] = ["Python allocations are measured with tracemalloc; native BLAS/GPU allocator accounting can be backend-specific."]
    accelerator_peak_bytes: int | None = None
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
        torch.cuda.reset_peak_memory_stats(runtime.device)
    tracing_started = not tracemalloc.is_tracing()
    if tracing_started:
        tracemalloc.start()
    start = time.perf_counter()
    try:
        payload = fn()
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
            accelerator_peak_bytes = int(torch.cuda.max_memory_allocated(runtime.device))
    finally:
        elapsed = time.perf_counter() - start
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        if tracing_started:
            tracemalloc.stop()
    if runtime.device.type == "mps":
        notes.append("MPS allocator memory is not included in the Python peak byte count.")
    return payload, _ResourceUsage(
        elapsed_seconds=elapsed,
        python_current_bytes=int(current_bytes),
        python_peak_bytes=int(peak_bytes),
        accelerator_peak_bytes=accelerator_peak_bytes,
        notes=tuple(notes),
    )


def _run_harmonic_oscillator(runtime: TorchRuntime) -> _CasePayload:
    family = "harmonic"
    parameters = {"omega": 5.0}
    states = 4
    coarse_modes = 18
    reference_modes = 40
    coarse = _solve_contract_spectrum(
        runtime,
        family=family,
        parameters=parameters,
        modes=coarse_modes,
        quadrature_points=144,
        states=states,
    )
    reference = _solve_contract_spectrum(
        runtime,
        family=family,
        parameters=parameters,
        modes=reference_modes,
        quadrature_points=320,
        states=states,
    )
    absolute_error = torch.abs(coarse.eigenvalues - reference.eigenvalues)
    rms_relative = _rms_relative_error(coarse.eigenvalues, reference.eigenvalues)
    return _CasePayload(
        score=_float(rms_relative),
        error={
            "eigenvalue_rms_relative_error": _float(rms_relative),
            "eigenvalue_max_absolute_error": _float(torch.max(absolute_error)),
        },
        mode_budget={
            "family": family,
            "states": states,
            "solve_modes": coarse_modes,
            "solve_quadrature_points": 144,
            "reference_modes": reference_modes,
            "reference_quadrature_points": 320,
        },
        identifiability=_spectrum_identifiability(
            runtime,
            family=family,
            parameters=parameters,
            modes=coarse_modes,
            states=states,
        ),
        details={
            "parameters": parameters,
            "coarse_eigenvalues": _tensor_list(coarse.eigenvalues),
            "reference_eigenvalues": _tensor_list(reference.eigenvalues),
        },
        assumptions=(
            "The reference is a higher-budget bounded-domain spectral solve, not an unbounded analytic harmonic oscillator spectrum.",
        ),
    )


def _run_double_well(runtime: TorchRuntime) -> _CasePayload:
    family = "double-well"
    parameters = {"a_param": 18.0, "b_param": 6.0}
    states = 5
    coarse_modes = 20
    reference_modes = 44
    coarse = _solve_contract_spectrum(
        runtime,
        family=family,
        parameters=parameters,
        modes=coarse_modes,
        quadrature_points=160,
        states=states,
    )
    reference = _solve_contract_spectrum(
        runtime,
        family=family,
        parameters=parameters,
        modes=reference_modes,
        quadrature_points=352,
        states=states,
    )
    coarse_split = coarse.eigenvalues[1] - coarse.eigenvalues[0]
    reference_split = reference.eigenvalues[1] - reference.eigenvalues[0]
    rms_relative = _rms_relative_error(coarse.eigenvalues, reference.eigenvalues)
    split_error = torch.abs(coarse_split - reference_split)
    return _CasePayload(
        score=max(_float(rms_relative), _float(split_error / torch.clamp(torch.abs(reference_split), min=1.0))),
        error={
            "eigenvalue_rms_relative_error": _float(rms_relative),
            "tunnel_splitting_absolute_error": _float(split_error),
        },
        mode_budget={
            "family": family,
            "states": states,
            "solve_modes": coarse_modes,
            "solve_quadrature_points": 160,
            "reference_modes": reference_modes,
            "reference_quadrature_points": 352,
        },
        identifiability=_spectrum_identifiability(
            runtime,
            family=family,
            parameters=parameters,
            modes=coarse_modes,
            states=states,
        ),
        details={
            "parameters": parameters,
            "coarse_tunnel_splitting": _float(coarse_split),
            "reference_tunnel_splitting": _float(reference_split),
        },
        assumptions=(
            "The benchmark reports local doublet splitting stability under modal refinement; it does not claim global tunneling-rate validation.",
        ),
    )


def _run_barrier_scattering(runtime: TorchRuntime) -> _CasePayload:
    height = 4.0
    width = 0.4
    center = 0.0
    energy_min = 0.6
    energy_max = 8.0
    coarse_points = 41
    reference_points = 161
    coarse = scattering_spectrum(
        rectangular_barrier(height, width, center=center),
        energy_min=energy_min,
        energy_max=energy_max,
        num_energies=coarse_points,
    )
    reference = scattering_spectrum(
        rectangular_barrier(height, width, center=center),
        energy_min=energy_min,
        energy_max=energy_max,
        num_energies=reference_points,
    )
    interpolated_reference = torch.as_tensor(
        np.interp(
            coarse.energies.detach().cpu().numpy(),
            reference.energies.detach().cpu().numpy(),
            reference.transmission.detach().cpu().numpy(),
        ),
        dtype=coarse.transmission.dtype,
        device=coarse.transmission.device,
    )
    transmission_rms = torch.sqrt(torch.mean((coarse.transmission - interpolated_reference) ** 2))
    unitarity_error = torch.max(torch.abs(coarse.transmission + coarse.reflection - 1.0))

    def observation_fn(values: Tensor) -> Tensor:
        trial_height = max(float(values[0].item()), 1e-8)
        trial_width = max(float(values[1].item()), 1e-8)
        return scattering_spectrum(
            rectangular_barrier(trial_height, trial_width, center=center),
            energy_min=energy_min,
            energy_max=energy_max,
            num_energies=21,
        ).transmission

    return _CasePayload(
        score=max(_float(transmission_rms), _float(unitarity_error)),
        error={
            "transmission_rms_error_to_refined_energy_grid": _float(transmission_rms),
            "max_unitarity_error": _float(unitarity_error),
        },
        mode_budget={
            "barrier_segments": 3,
            "coarse_energy_points": coarse_points,
            "reference_energy_points": reference_points,
            "identifiability_energy_points": 21,
        },
        identifiability=_finite_difference_identifiability(
            parameter_names=("height", "width"),
            values=(height, width),
            observation_fn=observation_fn,
        ),
        details={
            "barrier": {"height": height, "width": width, "center": center},
            "energy_window": [energy_min, energy_max],
            "compute_device": "cpu (the transfer-matrix scattering module currently allocates CPU tensors)",
            "resonance_count": int(coarse.resonance_energies.numel()),
        },
        assumptions=(
            "This is an open-boundary transfer-matrix scattering case, so its mode budget is an energy-grid/segment budget rather than a sine-basis budget.",
        ),
    )


def _run_anharmonic_inverse_fit(runtime: TorchRuntime) -> _CasePayload:
    family = "morse"
    target_parameters = {"D_e": 8.0, "alpha": 3.0, "x_eq": 0.42}
    initial_guess = {"D_e": 6.0, "alpha": 2.2, "x_eq": 0.50}
    states = 4
    target_points = 34
    fit_points = 26
    domain = _domain_for_runtime(runtime)
    family_def = resolve_potential_family(family)
    target = solve_eigenproblem(
        family_def.build_from_mapping(domain, target_parameters),
        domain,
        num_points=target_points,
        num_states=states,
    ).eigenvalues.detach()
    calibration = calibrate_potential_from_spectrum(
        family=family,
        target_eigenvalues=target,
        initial_guess=initial_guess,
        num_points=fit_points,
        optimization_config=GradientOptimizationConfig(steps=12, learning_rate=0.035),
        posterior_config=PosteriorConfig(
            enabled=True,
            compute_coefficient_posterior=False,
            compute_observation_posterior=False,
            compute_hessian_diagnostics=False,
            compute_laplace_evidence=False,
        ),
        device=runtime.device,
    )
    parameter_abs_errors = {
        name: abs(float(calibration.estimated_parameters[name]) - float(target_parameters[name]))
        for name in family_def.parameter_names
    }
    eigenvalue_rms = _rms_relative_error(calibration.predicted_eigenvalues, target)
    return _CasePayload(
        score=max(float(calibration.final_loss), _float(eigenvalue_rms)),
        error={
            "final_loss": float(calibration.final_loss),
            "eigenvalue_rms_relative_error": _float(eigenvalue_rms),
            "max_parameter_absolute_error": max(parameter_abs_errors.values()),
        },
        mode_budget={
            "family": family,
            "states": states,
            "target_points": target_points,
            "fit_points": fit_points,
            "optimization_steps": 12,
        },
        identifiability=_posterior_identifiability(calibration),
        details={
            "target_parameters": target_parameters,
            "initial_guess": initial_guess,
            "estimated_parameters": calibration.estimated_parameters,
            "parameter_absolute_errors": parameter_abs_errors,
        },
        assumptions=(
            "The inverse-fit benchmark uses a short deterministic local optimizer budget; it measures calibration behavior under that budget, not asymptotic optimizer convergence.",
        ),
    )


def _run_noisy_reconstruction(runtime: TorchRuntime) -> _CasePayload:
    num_modes = 12
    grid_points = 48
    simulation = simulate_gaussian_packet(
        center=0.32,
        width=0.075,
        wavenumber=18.0,
        times=(0.0, 0.0015, 0.003),
        num_modes=24,
        quadrature_points=192,
        grid_points=grid_points,
        device=runtime.device,
    )
    clean = normalize_profiles(simulation.densities, simulation.grid)
    phase_grid = torch.linspace(
        0.0,
        2.0 * math.pi,
        grid_points,
        dtype=clean.dtype,
        device=clean.device,
    )
    sample_index = torch.arange(clean.shape[0], dtype=clean.dtype, device=clean.device).reshape(-1, 1)
    pattern = torch.sin((sample_index + 1.0) * phase_grid.reshape(1, -1))
    pattern = pattern / torch.clamp(torch.sqrt(torch.mean(pattern**2)), min=torch.finfo(pattern.dtype).tiny)
    noise_scale = 0.025
    noisy = torch.clamp(clean + noise_scale * torch.mean(clean) * pattern, min=0.0)
    noisy = normalize_profiles(noisy, simulation.grid)
    domain = InfiniteWell1D(left=simulation.grid[0], right=simulation.grid[-1])
    compression = compress_profiles(noisy, simulation.grid, domain=domain, num_modes=num_modes)
    clean_error = torch.mean(relative_l2_error(clean, compression.reconstruction, simulation.grid))
    noisy_error = torch.mean(relative_l2_error(noisy, compression.reconstruction, simulation.grid))
    basis_matrix = compression.basis.evaluate(simulation.grid)
    return _CasePayload(
        score=_float(clean_error),
        error={
            "mean_relative_l2_error_to_clean": _float(clean_error),
            "mean_relative_l2_error_to_noisy": _float(noisy_error),
            "noise_scale": noise_scale,
        },
        mode_budget={
            "simulation_modes": 24,
            "reconstruction_modes": num_modes,
            "quadrature_points": 192,
            "grid_points": grid_points,
            "sample_count": int(clean.shape[0]),
        },
        identifiability=_linear_condition_identifiability(
            basis_matrix,
            parameter_count=num_modes,
            observation_count=grid_points,
            method="retained sine-basis design matrix condition",
        ),
        details={
            "noise_model": "deterministic sinusoidal density perturbation, clipped and renormalized",
            "mean_clean_mass": _float(torch.mean(torch.trapezoid(clean, simulation.grid, dim=-1))),
            "mean_noisy_mass": _float(torch.mean(torch.trapezoid(noisy, simulation.grid, dim=-1))),
        },
        assumptions=(
            "The noise regime is deterministic for reproducibility; it is a reconstruction stress case, not a statistical noise simulator.",
        ),
    )


def _run_reduced_model_tradeoff(runtime: TorchRuntime) -> _CasePayload:
    family_x = "harmonic"
    family_y = "double-well"
    parameters_x = {"omega": 4.0}
    parameters_y = {"a_param": 12.0, "b_param": 4.5}
    combined_states = 6
    coarse_start = time.perf_counter()
    coarse = analyze_separable_tensor_product_spectrum(
        family_x=family_x,
        parameters_x=parameters_x,
        family_y=family_y,
        parameters_y=parameters_y,
        num_points_x=18,
        num_points_y=18,
        num_states_x=4,
        num_states_y=4,
        num_combined_states=combined_states,
        device=runtime.device,
    )
    coarse_elapsed = time.perf_counter() - coarse_start
    reference_start = time.perf_counter()
    reference = analyze_separable_tensor_product_spectrum(
        family_x=family_x,
        parameters_x=parameters_x,
        family_y=family_y,
        parameters_y=parameters_y,
        num_points_x=34,
        num_points_y=34,
        num_states_x=6,
        num_states_y=6,
        num_combined_states=combined_states,
        device=runtime.device,
    )
    reference_elapsed = time.perf_counter() - reference_start
    rms_relative = _rms_relative_error(coarse.combined_eigenvalues, reference.combined_eigenvalues)
    speed_ratio = reference_elapsed / max(coarse_elapsed, torch.finfo(torch.float64).tiny)
    return _CasePayload(
        score=_float(rms_relative),
        error={
            "combined_spectrum_rms_relative_error": _float(rms_relative),
            "reference_to_reduced_elapsed_ratio": float(speed_ratio),
        },
        mode_budget={
            "families": [family_x, family_y],
            "combined_states": combined_states,
            "coarse_tensor_modes": coarse.mode_budget.total_tensor_mode_count,
            "reference_tensor_modes": reference.mode_budget.total_tensor_mode_count,
            "coarse_axis_states": [4, 4],
            "reference_axis_states": [6, 6],
        },
        identifiability={
            "method": "structured separable tensor-product contract",
            "parameter_count": 0,
            "observation_count": combined_states,
            "effective_rank": combined_states,
            "condition_number": 1.0,
            "identifiability_score": 1.0,
            "notes": [
                "This case has no fitted parameters; identifiability reports whether retained combined states are explicitly represented by the structured operator contract."
            ],
        },
        details={
            "coarse_elapsed_seconds": coarse_elapsed,
            "reference_elapsed_seconds": reference_elapsed,
            "coarse_combined_eigenvalues": _tensor_list(coarse.combined_eigenvalues),
            "reference_combined_eigenvalues": _tensor_list(reference.combined_eigenvalues),
        },
        assumptions=(
            "The reduced-model tradeoff compares two separable tensor-product budgets against each other; it does not benchmark a generic dense multidimensional solver.",
        ),
        timing={
            "coarse_elapsed_seconds": coarse_elapsed,
            "reference_elapsed_seconds": reference_elapsed,
        },
    )


_OFFICIAL_CASES: tuple[_BenchmarkCase, ...] = (
    _BenchmarkCase(
        BenchmarkCaseDefinition(
            case_id="harmonic-oscillator",
            label="Harmonic Oscillator",
            category="forward-spectrum",
            objective="Measure bounded-domain harmonic spectrum stability under modal refinement.",
            physical_contract="PotentialFamily(harmonic) + HamiltonianOperator + ObservableSet(eigenvalues)",
            default_mode_budget={
                "solve_modes": 18,
                "reference_modes": 40,
                "states": 4,
            },
            reported_metrics=(
                "eigenvalue_rms_relative_error",
                "eigenvalue_max_absolute_error",
                "local_spectrum_identifiability",
            ),
            honest_limits=(
                "Reference spectrum is a higher-resolution bounded-domain solve, not the unbounded analytic oscillator formula.",
            ),
        ),
        _run_harmonic_oscillator,
    ),
    _BenchmarkCase(
        BenchmarkCaseDefinition(
            case_id="double-well",
            label="Double Well",
            category="forward-spectrum",
            objective="Measure eigenvalue and tunnel-splitting stability for the symmetric quartic double well.",
            physical_contract="PotentialFamily(double-well) + HamiltonianOperator + ObservableSet(tunnel_splitting)",
            default_mode_budget={
                "solve_modes": 20,
                "reference_modes": 44,
                "states": 5,
            },
            reported_metrics=(
                "eigenvalue_rms_relative_error",
                "tunnel_splitting_absolute_error",
                "local_spectrum_identifiability",
            ),
            honest_limits=(
                "Only local low-lying spectral stability is measured; no global tunneling-rate claim is made.",
            ),
        ),
        _run_double_well,
    ),
    _BenchmarkCase(
        BenchmarkCaseDefinition(
            case_id="barrier-scattering",
            label="Barrier Scattering",
            category="open-transport",
            objective="Measure transfer-matrix transmission stability, unitarity, and local barrier-parameter sensitivity.",
            physical_contract="piecewise-constant barrier + transfer-matrix scattering spectrum",
            default_mode_budget={
                "coarse_energy_points": 41,
                "reference_energy_points": 161,
                "barrier_segments": 3,
            },
            reported_metrics=(
                "transmission_rms_error_to_refined_energy_grid",
                "max_unitarity_error",
                "finite_difference_identifiability",
            ),
            honest_limits=(
                "This open-boundary case reports an energy-grid budget rather than a sine-basis modal budget.",
                "The current transfer-matrix scattering implementation allocates CPU tensors, so accelerator backend comparison is runtime context rather than a GPU scattering-kernel benchmark.",
            ),
        ),
        _run_barrier_scattering,
    ),
    _BenchmarkCase(
        BenchmarkCaseDefinition(
            case_id="anharmonic-inverse-fit",
            label="Anharmonic Inverse Fit",
            category="inverse-calibration",
            objective="Measure short-budget Morse-spectrum calibration with local posterior identifiability.",
            physical_contract="PotentialFamily(morse) + differentiable bounded-domain eigensolver + local posterior summary",
            default_mode_budget={
                "target_points": 34,
                "fit_points": 26,
                "states": 4,
                "optimization_steps": 12,
            },
            reported_metrics=(
                "final_loss",
                "eigenvalue_rms_relative_error",
                "local_posterior_identifiability",
            ),
            honest_limits=(
                "The benchmark is a deterministic local optimizer budget, not a proof of global inverse identifiability.",
            ),
        ),
        _run_anharmonic_inverse_fit,
    ),
    _BenchmarkCase(
        BenchmarkCaseDefinition(
            case_id="noisy-reconstruction",
            label="Noisy Reconstruction",
            category="modal-reconstruction",
            objective="Measure modal reconstruction error under a reproducible density perturbation.",
            physical_contract="Gaussian packet simulation + sine-basis profile compression/reconstruction",
            default_mode_budget={
                "simulation_modes": 24,
                "reconstruction_modes": 12,
                "grid_points": 48,
            },
            reported_metrics=(
                "mean_relative_l2_error_to_clean",
                "mean_relative_l2_error_to_noisy",
                "basis_condition_identifiability",
            ),
            honest_limits=(
                "Noise is deterministic and reproducible; it is not a broad stochastic measurement-noise model.",
            ),
        ),
        _run_noisy_reconstruction,
    ),
    _BenchmarkCase(
        BenchmarkCaseDefinition(
            case_id="reduced-model-tradeoff",
            label="Reduced-Model Speed/Accuracy Tradeoff",
            category="reduced-model",
            objective="Measure separable tensor-product accuracy and timing under coarse vs reference budgets.",
            physical_contract="separable tensor-product spectral reduction over bounded 1D axes",
            default_mode_budget={
                "coarse_axis_states": [4, 4],
                "reference_axis_states": [6, 6],
                "combined_states": 6,
            },
            reported_metrics=(
                "combined_spectrum_rms_relative_error",
                "reference_to_reduced_elapsed_ratio",
                "structured_operator_identifiability",
            ),
            honest_limits=(
                "This measures the explicit separable reduced model, not a generic multidimensional solver.",
            ),
        ),
        _run_reduced_model_tradeoff,
    ),
)


def official_benchmark_registry() -> tuple[BenchmarkCaseDefinition, ...]:
    return tuple(case.definition for case in _OFFICIAL_CASES)


def list_official_benchmarks() -> tuple[str, ...]:
    return tuple(case.definition.case_id for case in _OFFICIAL_CASES)


def _resolve_case(case_id: str) -> _BenchmarkCase:
    normalized = str(case_id).strip().lower()
    for case in _OFFICIAL_CASES:
        if case.definition.case_id == normalized:
            return case
    supported = ", ".join(list_official_benchmarks())
    raise ValueError(f"unknown benchmark case '{case_id}'. Supported cases: {supported}")


def _normalize_devices(devices: Sequence[str | torch.device] | str | torch.device | None) -> tuple[str | torch.device, ...]:
    if devices is None:
        return ()
    if isinstance(devices, (str, torch.device)):
        return (devices,)
    return tuple(devices)


def _run_case_once(case: _BenchmarkCase, runtime: TorchRuntime) -> tuple[BenchmarkCaseMetrics, tuple[str, ...]]:
    payload, resources = _measure_payload(lambda: case.runner(runtime), runtime)
    timing = dict(payload.timing or {})
    timing["total_elapsed_seconds"] = resources.elapsed_seconds
    memory = {
        "python_current_bytes": resources.python_current_bytes,
        "python_peak_bytes": resources.python_peak_bytes,
        "accelerator_peak_bytes": resources.accelerator_peak_bytes,
        "notes": list(resources.notes),
    }
    return (
        BenchmarkCaseMetrics(
            score=float(payload.score),
            error=dict(payload.error),
            timing=timing,
            memory=memory,
            mode_budget=dict(payload.mode_budget),
            identifiability=dict(payload.identifiability),
            backend=_runtime_payload(runtime),
            backend_comparison={},
            details=dict(payload.details),
        ),
        payload.assumptions,
    )


def _backend_comparison(
    case: _BenchmarkCase,
    primary_runtime: TorchRuntime,
    primary_metrics: BenchmarkCaseMetrics,
    comparison_devices: Sequence[str | torch.device] | str | torch.device | None,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for requested in _normalize_devices(comparison_devices):
        key = str(requested)
        try:
            comparison_runtime = inspect_torch_runtime(requested)
        except Exception as exc:  # pragma: no cover - depends on host hardware.
            comparison[key] = {"status": "failed", "message": str(exc)}
            continue
        if comparison_runtime.device == primary_runtime.device:
            comparison[key] = {
                "status": "primary_backend",
                "backend": comparison_runtime.backend,
                "device": str(comparison_runtime.device),
                "score": primary_metrics.score,
                "score_delta": 0.0,
                "elapsed_ratio": 1.0,
            }
            continue
        try:
            comparison_metrics, _ = _run_case_once(case, comparison_runtime)
        except Exception as exc:  # pragma: no cover - depends on optional backend support.
            comparison[key] = {
                "status": "failed",
                "backend": comparison_runtime.backend,
                "device": str(comparison_runtime.device),
                "message": str(exc),
            }
            continue
        comparison[key] = {
            "status": "passed",
            "backend": comparison_runtime.backend,
            "device": str(comparison_runtime.device),
            "score": comparison_metrics.score,
            "score_delta": comparison_metrics.score - primary_metrics.score,
            "elapsed_seconds": comparison_metrics.timing["total_elapsed_seconds"],
            "elapsed_ratio": comparison_metrics.timing["total_elapsed_seconds"]
            / max(primary_metrics.timing["total_elapsed_seconds"], torch.finfo(torch.float64).tiny),
            "python_peak_bytes": comparison_metrics.memory["python_peak_bytes"],
            "python_peak_byte_delta": comparison_metrics.memory["python_peak_bytes"]
            - primary_metrics.memory["python_peak_bytes"],
        }
    return comparison


def run_benchmark_case(
    case_id: str,
    *,
    device: str | torch.device = "auto",
    comparison_devices: Sequence[str | torch.device] | str | torch.device | None = ("cpu",),
) -> BenchmarkCaseResult:
    case = _resolve_case(case_id)
    try:
        runtime = inspect_torch_runtime(device)
        metrics, assumptions = _run_case_once(case, runtime)
        metrics = replace(
            metrics,
            backend_comparison=_backend_comparison(case, runtime, metrics, comparison_devices),
        )
    except Exception as exc:
        return BenchmarkCaseResult(
            definition=case.definition,
            status="failed",
            metrics=None,
            assumptions=case.definition.honest_limits,
            message=str(exc),
        )
    return BenchmarkCaseResult(
        definition=case.definition,
        status="passed",
        metrics=metrics,
        assumptions=assumptions,
    )


def run_benchmark_registry(
    *,
    case_ids: Sequence[str] | None = None,
    device: str | torch.device = "auto",
    comparison_devices: Sequence[str | torch.device] | str | torch.device | None = ("cpu",),
) -> BenchmarkRegistryReport:
    runtime = inspect_torch_runtime(device)
    selected_case_ids = tuple(case_ids) if case_ids is not None else list_official_benchmarks()
    results = tuple(
        run_benchmark_case(
            case_id,
            device=runtime.device,
            comparison_devices=comparison_devices,
        )
        for case_id in selected_case_ids
    )
    passed = sum(1 for result in results if result.status == "passed")
    failed = sum(1 for result in results if result.status == "failed")
    scores = [result.metrics.score for result in results if result.metrics is not None]
    total_elapsed = sum(
        result.metrics.timing["total_elapsed_seconds"]
        for result in results
        if result.metrics is not None
    )
    return BenchmarkRegistryReport(
        suite_name="official-spectral-packet-benchmarks",
        suite_version="1",
        runtime=runtime,
        case_results=results,
        summary={
            "case_count": len(results),
            "passed": passed,
            "failed": failed,
            "max_score": max(scores) if scores else None,
            "total_elapsed_seconds": total_elapsed,
            "case_ids": list(selected_case_ids),
        },
        assumptions=(
            "Benchmarks are deterministic engineering checks over explicit physical contracts; they are not statistical certification suites.",
            "Backend comparison runs only for requested comparison devices and records failures rather than silently substituting another backend.",
        ),
    )


__all__ = [
    "BenchmarkCaseDefinition",
    "BenchmarkCaseMetrics",
    "BenchmarkCaseResult",
    "BenchmarkRegistryReport",
    "list_official_benchmarks",
    "official_benchmark_registry",
    "run_benchmark_case",
    "run_benchmark_registry",
]
