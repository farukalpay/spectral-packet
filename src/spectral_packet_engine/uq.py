from __future__ import annotations

"""Shared local uncertainty and identifiability utilities.

This module keeps the reusable local-Gaussian inference logic separate from
packet-specific inverse estimation and potential-family calibration wrappers.

The inference ladder is layered:

Layer 0 (existing) — local Laplace/Gauss-Newton posterior summaries.
Layer 1 — full Hessian diagnostics and Laplace model evidence.
Layer 2 — multi-start identifiability atlas for detecting multimodality.

Layer 3 (MCMC) lives in ``research_uq.py`` behind optional dependencies.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


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


def linearized_covariance(
    output_jacobian: torch.Tensor,
    parameter_covariance: torch.Tensor,
) -> torch.Tensor:
    jacobian = torch.as_tensor(output_jacobian)
    covariance = jacobian @ parameter_covariance @ jacobian.transpose(0, 1)
    return 0.5 * (covariance + covariance.transpose(0, 1))


def _unravel_index(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    coordinates: list[int] = []
    remainder = int(index)
    for size in reversed(shape):
        coordinates.append(remainder % size)
        remainder //= size
    return tuple(reversed(coordinates))


def _coerce_observation_jacobian(
    observation_jacobian: torch.Tensor,
    *,
    num_parameters: int,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    return torch.as_tensor(
        observation_jacobian,
        dtype=dtype,
        device=device,
    ).reshape(-1, num_parameters)


@dataclass(frozen=True, slots=True)
class PosteriorConfig:
    enabled: bool = True
    noise_scale: float | None = None
    noise_floor: float = 1e-6
    fisher_damping: float = 1e-6
    confidence_level: float = 0.95
    compute_coefficient_posterior: bool = True
    compute_sensitivity: bool = True
    compute_observation_posterior: bool = True
    compute_observation_information: bool = True
    # Layer 1: Hessian diagnostics and Laplace evidence
    compute_hessian_diagnostics: bool = False
    compute_laplace_evidence: bool = False
    hessian_adequacy_threshold: float = 2.0

    def __post_init__(self) -> None:
        if self.noise_scale is not None and self.noise_scale <= 0:
            raise ValueError("noise_scale must be positive when provided")
        if self.noise_floor <= 0:
            raise ValueError("noise_floor must be positive")
        if self.fisher_damping < 0:
            raise ValueError("fisher_damping must be non-negative")
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        if self.hessian_adequacy_threshold <= 0:
            raise ValueError("hessian_adequacy_threshold must be positive")


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
    precision_matrix: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class SensitivityMapSummary:
    parameter_names: tuple[str, ...]
    observation_shape: tuple[int, ...]
    gradient: torch.Tensor
    one_sigma_effect: torch.Tensor
    rms_one_sigma_effect: torch.Tensor
    peak_one_sigma_effect: torch.Tensor


@dataclass(frozen=True, slots=True)
class ObservationPosteriorSummary:
    observation_shape: tuple[int, ...]
    mean: torch.Tensor
    standard_deviation: torch.Tensor
    confidence_level: float
    confidence_interval_low: torch.Tensor
    confidence_interval_high: torch.Tensor


@dataclass(frozen=True, slots=True)
class ObservationInformationSummary:
    observation_shape: tuple[int, ...]
    information_density: torch.Tensor
    normalized_information_density: torch.Tensor
    total_information: float
    effective_observation_count: float
    peak_index: tuple[int, ...]
    peak_information_density: float


# ---------------------------------------------------------------------------
# Layer 1: Hessian diagnostics and Laplace evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HessianDiagnostics:
    """Full Hessian vs Gauss-Newton comparison at a calibrated parameter point."""

    gauss_newton_eigenvalues: torch.Tensor
    full_hessian_eigenvalues: torch.Tensor
    eigenvalue_ratio: torch.Tensor
    max_eigenvalue_ratio: float
    gauss_newton_adequate: bool
    negative_curvature_detected: bool
    adequacy_threshold: float


@dataclass(frozen=True, slots=True)
class LaplaceEvidence:
    """Log marginal likelihood under the Laplace approximation."""

    log_likelihood: float
    log_occam_factor: float
    log_marginal_likelihood: float
    num_observations: int
    num_parameters: int
    noise_scale: float
    bic: float


# ---------------------------------------------------------------------------
# Layer 2: Multi-start identifiability atlas
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IdentifiabilityAtlas:
    """Multi-start calibration results clustered into parameter basins."""

    num_starts: int
    parameter_names: tuple[str, ...]
    basin_labels: torch.Tensor
    num_basins: int
    basin_centers: torch.Tensor
    basin_losses: torch.Tensor
    basin_spreads: torch.Tensor
    all_final_parameters: torch.Tensor
    all_final_losses: torch.Tensor
    global_best_index: int
    multimodal: bool


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
        precision_matrix=precision.detach(),
    )


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


def summarize_observation_posterior(
    *,
    observation: torch.Tensor,
    observation_jacobian: torch.Tensor,
    parameter_covariance: torch.Tensor,
    confidence_level: float,
) -> ObservationPosteriorSummary:
    mean = real_view(torch.as_tensor(observation))
    observation_shape = tuple(int(size) for size in mean.shape)
    parameter_covariance_tensor = torch.as_tensor(
        parameter_covariance,
        dtype=mean.dtype,
        device=mean.device,
    )
    covariance = linearized_covariance(
        _coerce_observation_jacobian(
            observation_jacobian,
            num_parameters=int(parameter_covariance_tensor.shape[0]),
            dtype=mean.dtype,
            device=mean.device,
        ),
        parameter_covariance_tensor,
    )
    standard_deviation = torch.sqrt(torch.clamp(torch.diagonal(covariance), min=0.0)).reshape(observation_shape)
    z_score = _central_normal_quantile(
        confidence_level,
        dtype=standard_deviation.dtype,
        device=standard_deviation.device,
    )
    confidence_interval_low = mean - z_score * standard_deviation
    confidence_interval_high = mean + z_score * standard_deviation
    return ObservationPosteriorSummary(
        observation_shape=observation_shape,
        mean=mean.detach(),
        standard_deviation=standard_deviation.detach(),
        confidence_level=confidence_level,
        confidence_interval_low=confidence_interval_low.detach(),
        confidence_interval_high=confidence_interval_high.detach(),
    )


def summarize_observation_information(
    *,
    observation: torch.Tensor,
    observation_jacobian: torch.Tensor,
    noise_scale: float,
) -> ObservationInformationSummary:
    mean = real_view(torch.as_tensor(observation))
    observation_shape = tuple(int(size) for size in mean.shape)
    raw_jacobian = torch.as_tensor(observation_jacobian)
    jacobian = _coerce_observation_jacobian(
        observation_jacobian,
        num_parameters=int(raw_jacobian.shape[-1]),
        dtype=mean.dtype,
        device=mean.device,
    )
    noise = torch.as_tensor(noise_scale, dtype=jacobian.dtype, device=jacobian.device).reshape(())
    if float(noise.item()) <= 0.0:
        raise ValueError("noise_scale must be positive")
    information_density = torch.sum(jacobian**2, dim=-1).reshape(observation_shape) / (noise**2)
    total_information_tensor = torch.sum(information_density)
    total_information = float(total_information_tensor.detach())
    if total_information <= 0.0:
        normalized = torch.zeros_like(information_density)
        effective_observation_count = 0.0
        peak_index = _unravel_index(0, observation_shape)
        peak_information_density = 0.0
    else:
        normalized = information_density / total_information_tensor
        flat_normalized = normalized.reshape(-1)
        positive_mass = flat_normalized[flat_normalized > 0]
        if positive_mass.numel() == 0:
            effective_observation_count = 0.0
        else:
            entropy = -torch.sum(positive_mass * torch.log(positive_mass))
            effective_observation_count = float(torch.exp(entropy).detach())
        peak_flat_index = int(torch.argmax(information_density.reshape(-1)).item())
        peak_index = _unravel_index(peak_flat_index, observation_shape)
        peak_information_density = float(information_density.reshape(-1)[peak_flat_index].detach())
    return ObservationInformationSummary(
        observation_shape=observation_shape,
        information_density=information_density.detach(),
        normalized_information_density=normalized.detach(),
        total_information=total_information,
        effective_observation_count=effective_observation_count,
        peak_index=peak_index,
        peak_information_density=peak_information_density,
    )


# ---------------------------------------------------------------------------
# Layer 1 computation functions
# ---------------------------------------------------------------------------


def compute_hessian_diagnostics(
    *,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    parameter_vector: torch.Tensor,
    gauss_newton_eigenvalues: torch.Tensor,
    adequacy_threshold: float = 2.0,
) -> HessianDiagnostics:
    """Compare full Hessian eigenvalues against the Gauss-Newton approximation.

    The Gauss-Newton approximation (J^T J) drops second-order residual terms.
    When the ratio of full Hessian eigenvalues to GN eigenvalues departs
    significantly from 1, the local Gaussian posterior may be unreliable.
    """
    vector = parameter_vector.detach().requires_grad_(True)
    full_hessian = torch.autograd.functional.hessian(loss_fn, vector)
    full_hessian = full_hessian.detach().reshape(vector.shape[0], vector.shape[0])
    full_hessian = 0.5 * (full_hessian + full_hessian.T)

    full_eigenvalues = torch.linalg.eigvalsh(full_hessian)
    gn_eigenvalues = gauss_newton_eigenvalues.to(
        dtype=full_eigenvalues.dtype, device=full_eigenvalues.device,
    )

    eps = torch.finfo(full_eigenvalues.dtype).eps
    safe_gn = torch.clamp(torch.abs(gn_eigenvalues), min=eps)
    ratio = full_eigenvalues / safe_gn

    max_ratio = float(torch.max(torch.abs(ratio)).item())
    negative_curvature = bool(torch.any(full_eigenvalues < -eps).item())

    return HessianDiagnostics(
        gauss_newton_eigenvalues=gn_eigenvalues.detach(),
        full_hessian_eigenvalues=full_eigenvalues.detach(),
        eigenvalue_ratio=ratio.detach(),
        max_eigenvalue_ratio=max_ratio,
        gauss_newton_adequate=max_ratio < adequacy_threshold,
        negative_curvature_detected=negative_curvature,
        adequacy_threshold=adequacy_threshold,
    )


def compute_laplace_evidence(
    *,
    residual_vector: torch.Tensor,
    noise_scale: float,
    precision_matrix: torch.Tensor,
    num_parameters: int,
) -> LaplaceEvidence:
    """Compute the Laplace approximation to the log marginal likelihood.

    This is a better model-comparison score than BIC: it naturally includes
    an Occam factor that penalises model complexity via the curvature of
    the loss surface, not just parameter count.

    log p(y|M) ≈ log_likelihood + log_occam_factor
    """
    residual = residual_vector.detach().reshape(-1).to(dtype=torch.float64)
    n_obs = int(residual.numel())
    sigma2 = float(noise_scale) ** 2

    rss = float(torch.sum(residual ** 2).item())
    log_likelihood = -0.5 * (n_obs * math.log(2 * math.pi * sigma2) + rss / sigma2)

    precision = precision_matrix.detach().to(dtype=torch.float64)
    precision_eigenvalues = torch.linalg.eigvalsh(precision)
    eps = torch.finfo(precision_eigenvalues.dtype).eps
    positive = precision_eigenvalues[precision_eigenvalues > eps]
    if positive.numel() > 0:
        log_det_precision = float(torch.sum(torch.log(positive)).item())
    else:
        log_det_precision = 0.0
    log_occam_factor = 0.5 * (num_parameters * math.log(2 * math.pi) - log_det_precision)

    log_marginal_likelihood = log_likelihood + log_occam_factor

    # BIC for comparison
    bic = n_obs * math.log(max(rss / n_obs, eps)) + num_parameters * math.log(n_obs)

    return LaplaceEvidence(
        log_likelihood=log_likelihood,
        log_occam_factor=log_occam_factor,
        log_marginal_likelihood=log_marginal_likelihood,
        num_observations=n_obs,
        num_parameters=num_parameters,
        noise_scale=float(noise_scale),
        bic=bic,
    )


# ---------------------------------------------------------------------------
# Layer 2 computation functions
# ---------------------------------------------------------------------------


def _single_linkage_cluster(
    points: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Assign cluster labels via single-linkage agglomerative clustering.

    Pure torch, no scipy/sklearn needed.  Returns integer labels tensor.
    """
    n = points.shape[0]
    labels = torch.arange(n, device=points.device)

    dists = torch.cdist(points.unsqueeze(0), points.unsqueeze(0)).squeeze(0)
    for i in range(n):
        for j in range(i + 1, n):
            if float(dists[i, j].item()) < threshold:
                old_label = int(labels[j].item())
                new_label = int(labels[i].item())
                if old_label != new_label:
                    labels[labels == old_label] = new_label

    # Renumber to 0..k-1
    unique_labels = torch.unique(labels)
    remap = {int(old.item()): new_idx for new_idx, old in enumerate(unique_labels)}
    return torch.tensor([remap[int(l.item())] for l in labels], device=points.device)


def build_identifiability_atlas(
    *,
    parameter_names: tuple[str, ...],
    calibration_fn: Callable[[Mapping[str, float]], tuple[dict[str, float], float]],
    start_points: Sequence[Mapping[str, float]],
    cluster_threshold: float | None = None,
) -> IdentifiabilityAtlas:
    """Run calibration from multiple starting points and cluster the results.

    ``calibration_fn`` takes an initial-guess mapping and returns
    ``(final_parameters_dict, final_loss)``.

    Detects multiple basins of attraction in the loss landscape.  When
    ``multimodal`` is True, the local Laplace posterior is unreliable and
    sampling-based inference (Layer 3) is recommended.
    """
    all_params: list[list[float]] = []
    all_losses: list[float] = []

    for start in start_points:
        final_params, final_loss = calibration_fn(start)
        all_params.append([float(final_params[name]) for name in parameter_names])
        all_losses.append(float(final_loss))

    params_tensor = torch.tensor(all_params, dtype=torch.float64)
    losses_tensor = torch.tensor(all_losses, dtype=torch.float64)
    num_starts = params_tensor.shape[0]

    # Normalize parameters for clustering
    param_std = torch.std(params_tensor, dim=0)
    param_std = torch.clamp(param_std, min=1e-12)
    normalized = params_tensor / param_std

    if cluster_threshold is None:
        dists = torch.cdist(normalized.unsqueeze(0), normalized.unsqueeze(0)).squeeze(0)
        diameter = float(torch.max(dists).item()) if num_starts > 1 else 1.0
        cluster_threshold = 0.1 * max(diameter, 1e-12)

    labels = _single_linkage_cluster(normalized, cluster_threshold)
    num_basins = int(torch.unique(labels).numel())

    basin_centers_list: list[list[float]] = []
    basin_losses_list: list[float] = []
    basin_spreads_list: list[list[float]] = []

    for basin_id in range(num_basins):
        mask = labels == basin_id
        basin_params = params_tensor[mask]
        basin_loss = losses_tensor[mask]
        best_in_basin = int(torch.argmin(basin_loss).item())
        basin_centers_list.append(basin_params[best_in_basin].tolist())
        basin_losses_list.append(float(basin_loss[best_in_basin].item()))
        if basin_params.shape[0] > 1:
            basin_spreads_list.append(torch.std(basin_params, dim=0).tolist())
        else:
            basin_spreads_list.append([0.0] * len(parameter_names))

    basin_centers = torch.tensor(basin_centers_list, dtype=torch.float64)
    basin_losses_t = torch.tensor(basin_losses_list, dtype=torch.float64)
    basin_spreads = torch.tensor(basin_spreads_list, dtype=torch.float64)
    global_best = int(torch.argmin(basin_losses_t).item())

    return IdentifiabilityAtlas(
        num_starts=num_starts,
        parameter_names=parameter_names,
        basin_labels=labels.detach(),
        num_basins=num_basins,
        basin_centers=basin_centers.detach(),
        basin_losses=basin_losses_t.detach(),
        basin_spreads=basin_spreads.detach(),
        all_final_parameters=params_tensor.detach(),
        all_final_losses=losses_tensor.detach(),
        global_best_index=global_best,
        multimodal=num_basins > 1,
    )


__all__ = [
    "HessianDiagnostics",
    "IdentifiabilityAtlas",
    "LaplaceEvidence",
    "ObservationInformationSummary",
    "ObservationPosteriorSummary",
    "ParameterPosteriorSummary",
    "PosteriorConfig",
    "SensitivityMapSummary",
    "build_identifiability_atlas",
    "compute_hessian_diagnostics",
    "compute_laplace_evidence",
    "covariance_to_correlation",
    "flatten_real_view",
    "linearized_covariance",
    "real_view",
    "softplus_inverse",
    "summarize_observation_information",
    "summarize_observation_posterior",
    "summarize_parameter_posterior",
    "summarize_sensitivity_map",
]
