from __future__ import annotations

"""Domain-specific workflows built on the shared spectral core."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

from spectral_packet_engine.differentiable_physics import (
    ControlObjective,
    GradientOptimizationConfig,
    PacketControlOptimizationSummary,
    PotentialCalibrationSummary,
    calibrate_potential_from_spectrum,
    optimize_packet_control,
)
from spectral_packet_engine.parametric_potentials import (
    available_potential_families,
    default_parameter_mapping,
)
from spectral_packet_engine.pipelines import TunnelingExperimentReport, analyze_tunneling
from spectral_packet_engine.runtime import inspect_torch_runtime
from spectral_packet_engine.uq import PosteriorConfig

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class PotentialFamilyCandidateSummary:
    family: str
    calibration: PotentialCalibrationSummary
    residual_sum_squares: float
    information_criterion: float
    relative_evidence_weight: float


@dataclass(frozen=True, slots=True)
class PotentialFamilyInferenceSummary:
    target_eigenvalues: Tensor
    candidates: tuple[PotentialFamilyCandidateSummary, ...]
    best_family: str
    family_weights: dict[str, float]
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SpectroscopyWorkflowSummary:
    observed_transition_energies: Tensor
    family_inference: PotentialFamilyInferenceSummary
    best_family_transition_energies: Tensor
    line_assignment_root_mean_square_error: float
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransportResonanceWorkflowSummary:
    barrier_family: str
    barrier_parameters: dict[str, float]
    tunneling: TunnelingExperimentReport
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ControlWorkflowSummary:
    optimization: PacketControlOptimizationSummary
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProfileInferenceWorkflowSummary:
    report: object
    inverse_fit: object
    feature_export: object
    assumptions: tuple[str, ...]


def infer_potential_family_from_spectrum(
    *,
    target_eigenvalues,
    families: Sequence[str] | None = None,
    initial_guesses: Mapping[str, Mapping[str, float]] | None = None,
    domain_length: float = 1.0,
    left: float = 0.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    num_points: int = 128,
    optimization_config: GradientOptimizationConfig = GradientOptimizationConfig(),
    posterior_config: PosteriorConfig | None = PosteriorConfig(),
    device: str | torch.device = "auto",
) -> PotentialFamilyInferenceSummary:
    runtime = inspect_torch_runtime(device)
    target = torch.as_tensor(target_eigenvalues, dtype=runtime.preferred_real_dtype, device=runtime.device).reshape(-1)
    if target.numel() == 0:
        raise ValueError("target_eigenvalues must contain at least one value")

    requested_families = tuple(families) if families is not None else available_potential_families()
    if not requested_families:
        raise ValueError("at least one potential family must be provided")

    from spectral_packet_engine.domain import InfiniteWell1D

    domain = InfiniteWell1D.from_length(
        domain_length,
        left=left,
        mass=mass,
        hbar=hbar,
        dtype=runtime.preferred_real_dtype,
        device=runtime.device,
    )

    calibrated_candidates: list[PotentialFamilyCandidateSummary] = []
    rss_values: list[float] = []
    bic_values: list[float] = []
    n_observations = max(int(target.numel()), 1)
    eps = torch.finfo(target.dtype).eps

    for family in requested_families:
        seed_mapping = (
            dict(initial_guesses[family])
            if initial_guesses is not None and family in initial_guesses
            else default_parameter_mapping(family, domain=domain)
        )
        calibration = calibrate_potential_from_spectrum(
            family=family,
            target_eigenvalues=target,
            initial_guess=seed_mapping,
            domain_length=domain_length,
            left=left,
            mass=mass,
            hbar=hbar,
            num_points=num_points,
            optimization_config=optimization_config,
            posterior_config=posterior_config,
            device=runtime.device,
        )
        residual_sum_squares = float(
            torch.sum((calibration.predicted_eigenvalues - target) ** 2).detach().cpu().item()
        )
        parameter_count = max(len(calibration.parameter_names), 1)
        bic = float(
            n_observations * math.log((residual_sum_squares / n_observations) + float(eps))
            + parameter_count * math.log(float(n_observations))
        )
        calibrated_candidates.append(
            PotentialFamilyCandidateSummary(
                family=family,
                calibration=calibration,
                residual_sum_squares=residual_sum_squares,
                information_criterion=bic,
                relative_evidence_weight=0.0,
            )
        )
        rss_values.append(residual_sum_squares)
        bic_values.append(bic)

    bic_tensor = torch.tensor(bic_values, dtype=target.dtype)
    delta_bic = bic_tensor - torch.min(bic_tensor)
    weights = torch.softmax(-0.5 * delta_bic, dim=0)
    ordered_indices = torch.argsort(bic_tensor)
    ordered_candidates = tuple(
        PotentialFamilyCandidateSummary(
            family=calibrated_candidates[int(index)].family,
            calibration=calibrated_candidates[int(index)].calibration,
            residual_sum_squares=calibrated_candidates[int(index)].residual_sum_squares,
            information_criterion=calibrated_candidates[int(index)].information_criterion,
            relative_evidence_weight=float(weights[int(index)].item()),
        )
        for index in ordered_indices
    )

    return PotentialFamilyInferenceSummary(
        target_eigenvalues=target.detach(),
        candidates=ordered_candidates,
        best_family=ordered_candidates[0].family,
        family_weights={candidate.family: candidate.relative_evidence_weight for candidate in ordered_candidates},
        assumptions=(
            "Family ranking uses the same differentiable bounded-domain eigensolver for each candidate potential family.",
            "Relative evidence weights are derived from a BIC-style approximation and should be interpreted as local model-comparison scores, not exact Bayesian model probabilities.",
            "Default initial guesses are minimal midpoint/scale seeds derived from parameter bounds when an explicit seed is not supplied.",
        ),
    )


def run_spectroscopy_workflow(
    *,
    target_eigenvalues,
    families: Sequence[str] | None = None,
    initial_guesses: Mapping[str, Mapping[str, float]] | None = None,
    domain_length: float = 1.0,
    left: float = 0.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    num_points: int = 128,
    optimization_config: GradientOptimizationConfig = GradientOptimizationConfig(),
    posterior_config: PosteriorConfig | None = PosteriorConfig(),
    device: str | torch.device = "auto",
) -> SpectroscopyWorkflowSummary:
    inference = infer_potential_family_from_spectrum(
        target_eigenvalues=target_eigenvalues,
        families=families,
        initial_guesses=initial_guesses,
        domain_length=domain_length,
        left=left,
        mass=mass,
        hbar=hbar,
        num_points=num_points,
        optimization_config=optimization_config,
        posterior_config=posterior_config,
        device=device,
    )
    observed = inference.target_eigenvalues[1:] - inference.target_eigenvalues[0]
    predicted = inference.candidates[0].calibration.transition_energies
    common = min(int(observed.shape[0]), int(predicted.shape[0]))
    if common > 0:
        rms_error = float(torch.sqrt(torch.mean((predicted[:common] - observed[:common]) ** 2)).item())
    else:
        rms_error = 0.0
    return SpectroscopyWorkflowSummary(
        observed_transition_energies=observed.detach(),
        family_inference=inference,
        best_family_transition_energies=predicted.detach(),
        line_assignment_root_mean_square_error=rms_error,
        assumptions=(
            "This spectroscopy workflow compares low-lying bounded-domain spectra across explicit potential families rather than fitting arbitrary line lists with a black-box regressor.",
            "Transition energies are reported relative to the best-fit family ground state.",
        ),
    )


def run_transport_resonance_workflow(
    *,
    barrier_height: float = 50.0,
    barrier_width_sigma: float = 0.03,
    domain_length: float = 1.0,
    grid_points: int = 512,
    num_modes: int = 128,
    num_energies: int = 500,
    packet_center: float = 0.25,
    packet_width: float = 0.04,
    packet_wavenumber: float = 40.0,
    device: str = "cpu",
) -> TransportResonanceWorkflowSummary:
    report = analyze_tunneling(
        barrier_height=barrier_height,
        barrier_width_sigma=barrier_width_sigma,
        domain_length=domain_length,
        grid_points=grid_points,
        num_modes=num_modes,
        num_energies=num_energies,
        packet_center=packet_center,
        packet_width=packet_width,
        packet_wavenumber=packet_wavenumber,
        device=device,
    )
    return TransportResonanceWorkflowSummary(
        barrier_family="gaussian-barrier",
        barrier_parameters={
            "height": float(barrier_height),
            "width_sigma": float(barrier_width_sigma),
            "center": float(domain_length) / 2.0,
        },
        tunneling=report,
        assumptions=(
            "This transport workflow uses the repository's tunneling experiment pipeline, coupling transfer-matrix scattering, WKB comparison, split-operator propagation, and Wigner diagnostics.",
            "Barrier geometry is restricted to the explicit Gaussian barrier experiment implemented in the shared physics core.",
        ),
    )


def run_control_workflow(
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
) -> ControlWorkflowSummary:
    optimization = optimize_packet_control(
        initial_guess=initial_guess,
        objective=objective,
        target_value=target_value,
        final_time=final_time,
        interval=interval,
        num_modes=num_modes,
        quadrature_points=quadrature_points,
        grid_points=grid_points,
        optimization_config=optimization_config,
        device=device,
    )
    return ControlWorkflowSummary(
        optimization=optimization,
        assumptions=(
            "This control workflow performs differentiable packet steering by optimizing initial state preparation parameters through the shared spectral projection and propagation stack.",
            "It does not claim arbitrary time-dependent pulse control; the control variables are the packet initialization parameters reported in the summary.",
        ),
    )


def run_profile_inference_workflow(
    table,
    *,
    initial_guess: Mapping[str, float],
    analyze_num_modes: int = 16,
    compress_num_modes: int = 8,
    inverse_num_modes: int = 96,
    feature_num_modes: int = 16,
    quadrature_points: int = 2048,
    normalize_each_profile: bool = False,
    device: str | torch.device = "auto",
) -> ProfileInferenceWorkflowSummary:
    from spectral_packet_engine.table_io import load_profile_table
    from spectral_packet_engine.workflows import (
        build_profile_table_report,
        export_feature_table_from_profile_table,
        fit_gaussian_packet_to_profile_table,
    )

    resolved_table = load_profile_table(table) if isinstance(table, (str, Path)) else table
    report = build_profile_table_report(
        resolved_table,
        analyze_num_modes=analyze_num_modes,
        compress_num_modes=compress_num_modes,
        device=device,
        normalize_each_profile=normalize_each_profile,
    )
    inverse_fit = fit_gaussian_packet_to_profile_table(
        resolved_table,
        initial_guess=initial_guess,
        num_modes=inverse_num_modes,
        quadrature_points=quadrature_points,
        device=device,
    )
    feature_export = export_feature_table_from_profile_table(
        resolved_table,
        num_modes=feature_num_modes,
        device=device,
        normalize_each_profile=normalize_each_profile,
    )
    return ProfileInferenceWorkflowSummary(
        report=report,
        inverse_fit=inverse_fit,
        feature_export=feature_export,
        assumptions=(
            "This vertical keeps the profile-table workflow report-first: inspect and compress before interpreting inverse parameters or downstream features.",
            "The inverse stage remains a bounded Gaussian-packet reconstruction with local uncertainty summaries rather than a generic unrestricted latent model.",
        ),
    )


__all__ = [
    "ControlWorkflowSummary",
    "PotentialFamilyCandidateSummary",
    "PotentialFamilyInferenceSummary",
    "ProfileInferenceWorkflowSummary",
    "SpectroscopyWorkflowSummary",
    "TransportResonanceWorkflowSummary",
    "infer_potential_family_from_spectrum",
    "run_control_workflow",
    "run_profile_inference_workflow",
    "run_spectroscopy_workflow",
    "run_transport_resonance_workflow",
]
