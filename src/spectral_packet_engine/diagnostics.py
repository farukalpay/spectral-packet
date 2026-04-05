from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.domain import coerce_tensor
from spectral_packet_engine.profiles import profile_mass, profile_mean, profile_variance, relative_l2_error

Tensor = torch.Tensor


def spectral_weights(coefficients) -> Tensor:
    coeffs = coerce_tensor(coefficients)
    if coeffs.ndim < 1:
        raise ValueError("spectral_weights expects at least one mode dimension")
    if not torch.isfinite(coeffs).all().item():
        raise ValueError("coefficients must be finite")
    if torch.is_complex(coeffs):
        return torch.abs(coeffs) ** 2
    return coeffs**2


def spectral_tail_mass(coefficients) -> Tensor:
    weights = spectral_weights(coefficients)
    total = torch.sum(weights, dim=-1, keepdim=True)
    safe_total = torch.where(total > 0, total, torch.ones_like(total))
    cumulative = torch.cumsum(weights, dim=-1)
    return 1.0 - cumulative / safe_total


@dataclass(frozen=True, slots=True)
class SpectralTruncationSummary:
    mode_numbers: Tensor
    modal_weights: Tensor
    cumulative_mass: Tensor
    tail_mass: Tensor
    dominant_modes: Tensor
    dominant_weights: Tensor


def summarize_spectral_coefficients(coefficients, *, top_k: int = 8) -> SpectralTruncationSummary:
    weights = spectral_weights(coefficients)
    if weights.ndim != 1:
        raise ValueError("summarize_spectral_coefficients expects a one-dimensional coefficient vector")
    mode_numbers = torch.arange(1, weights.shape[0] + 1, dtype=torch.int64, device=weights.device)
    total = torch.sum(weights)
    safe_total = total if total > 0 else torch.ones_like(total)
    cumulative_mass = torch.cumsum(weights, dim=0) / safe_total
    tail_mass = 1.0 - cumulative_mass

    dominant_order = torch.argsort(weights, descending=True)[: max(1, min(top_k, weights.shape[0]))]
    dominant_modes = mode_numbers[dominant_order]
    dominant_weights = weights[dominant_order]

    return SpectralTruncationSummary(
        mode_numbers=mode_numbers,
        modal_weights=weights,
        cumulative_mass=cumulative_mass,
        tail_mass=tail_mass,
        dominant_modes=dominant_modes,
        dominant_weights=dominant_weights,
    )


@dataclass(frozen=True, slots=True)
class ReconstructionErrorSummary:
    relative_l2_error: Tensor
    mean_relative_l2_error: Tensor
    max_relative_l2_error: Tensor
    root_mean_square_error: Tensor


def summarize_profile_reconstruction(reference, approximation, grid) -> ReconstructionErrorSummary:
    reference_tensor = coerce_tensor(reference)
    approximation_tensor = coerce_tensor(approximation, dtype=reference_tensor.dtype, device=reference_tensor.device)
    error = relative_l2_error(reference_tensor, approximation_tensor, grid)
    residual = reference_tensor - approximation_tensor
    rmse = torch.sqrt(torch.mean(residual**2))
    return ReconstructionErrorSummary(
        relative_l2_error=error,
        mean_relative_l2_error=torch.mean(error),
        max_relative_l2_error=torch.max(error),
        root_mean_square_error=rmse,
    )


def _coerce_thresholds(thresholds, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    values = coerce_tensor(thresholds, device=device, dtype=dtype)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError("capture_thresholds must be one-dimensional")
    if not torch.all((values > 0.0) & (values <= 1.0)).item():
        raise ValueError("capture_thresholds must lie in (0, 1]")
    return values


def mode_counts_for_mass(coefficients, thresholds=(0.9, 0.95, 0.99)) -> Tensor:
    weights = spectral_weights(coefficients)
    if weights.ndim == 1:
        weights = weights.unsqueeze(0)
    if weights.ndim != 2:
        raise ValueError("mode_counts_for_mass expects one- or two-dimensional coefficients")

    thresholds_tensor = _coerce_thresholds(thresholds, device=weights.device, dtype=weights.dtype)
    total = torch.sum(weights, dim=-1, keepdim=True)
    safe_total = torch.where(total > 0, total, torch.ones_like(total))
    cumulative_mass = torch.cumsum(weights, dim=-1) / safe_total
    hits = cumulative_mass[:, None, :] >= thresholds_tensor[None, :, None]
    first_indices = torch.argmax(hits.to(dtype=torch.int64), dim=-1) + 1
    reached = torch.any(hits, dim=-1)
    counts = torch.where(reached, first_indices, torch.zeros_like(first_indices))
    if coefficients.ndim == 1:
        return counts[0]
    return counts


@dataclass(frozen=True, slots=True)
class SpectralBatchSummary:
    mode_numbers: Tensor
    mean_modal_weights: Tensor
    max_modal_weights: Tensor
    mean_cumulative_mass: Tensor
    max_tail_mass: Tensor
    dominant_modes: Tensor
    dominant_mean_weights: Tensor
    capture_thresholds: Tensor
    mean_mode_counts_for_thresholds: Tensor
    max_mode_counts_for_thresholds: Tensor


def summarize_spectral_batch(coefficients, *, capture_thresholds=(0.9, 0.95, 0.99), top_k: int = 8) -> SpectralBatchSummary:
    weights = spectral_weights(coefficients)
    if weights.ndim == 1:
        weights = weights.unsqueeze(0)
    if weights.ndim != 2:
        raise ValueError("summarize_spectral_batch expects one- or two-dimensional coefficients")

    mode_numbers = torch.arange(1, weights.shape[-1] + 1, dtype=torch.int64, device=weights.device)
    total = torch.sum(weights, dim=-1, keepdim=True)
    safe_total = torch.where(total > 0, total, torch.ones_like(total))
    normalized_weights = weights / safe_total
    cumulative_mass = torch.cumsum(normalized_weights, dim=-1)
    tail_mass = 1.0 - cumulative_mass
    threshold_tensor = _coerce_thresholds(capture_thresholds, device=weights.device, dtype=weights.dtype)
    mode_counts = mode_counts_for_mass(coefficients, thresholds=threshold_tensor)
    mean_modal_weights = torch.mean(normalized_weights, dim=0)
    dominant_order = torch.argsort(mean_modal_weights, descending=True)[: max(1, min(top_k, weights.shape[-1]))]

    return SpectralBatchSummary(
        mode_numbers=mode_numbers,
        mean_modal_weights=mean_modal_weights,
        max_modal_weights=torch.max(normalized_weights, dim=0).values,
        mean_cumulative_mass=torch.mean(cumulative_mass, dim=0),
        max_tail_mass=torch.max(tail_mass, dim=0).values,
        dominant_modes=mode_numbers[dominant_order],
        dominant_mean_weights=mean_modal_weights[dominant_order],
        capture_thresholds=threshold_tensor,
        mean_mode_counts_for_thresholds=torch.mean(mode_counts.to(dtype=weights.dtype), dim=0),
        max_mode_counts_for_thresholds=torch.max(mode_counts, dim=0).values.to(dtype=weights.dtype),
    )


@dataclass(frozen=True, slots=True)
class ProfileComparisonSummary:
    relative_l2_error: Tensor
    mean_relative_l2_error: Tensor
    max_relative_l2_error: Tensor
    root_mean_square_error: Tensor
    max_absolute_error: Tensor
    mass_error: Tensor
    mean_position_error: Tensor
    width_error: Tensor
    mean_mass_error: Tensor
    max_mass_error: Tensor
    mean_position_mae: Tensor
    width_mae: Tensor


def summarize_profile_comparison(reference, approximation, grid) -> ProfileComparisonSummary:
    reference_tensor = coerce_tensor(reference)
    approximation_tensor = coerce_tensor(approximation, dtype=reference_tensor.dtype, device=reference_tensor.device)
    reconstruction = summarize_profile_reconstruction(reference_tensor, approximation_tensor, grid)
    residual = approximation_tensor - reference_tensor
    mass_error = profile_mass(approximation_tensor, grid) - profile_mass(reference_tensor, grid)
    mean_position_error = profile_mean(approximation_tensor, grid) - profile_mean(reference_tensor, grid)
    width_error = torch.sqrt(profile_variance(approximation_tensor, grid)) - torch.sqrt(
        profile_variance(reference_tensor, grid)
    )

    return ProfileComparisonSummary(
        relative_l2_error=reconstruction.relative_l2_error,
        mean_relative_l2_error=reconstruction.mean_relative_l2_error,
        max_relative_l2_error=reconstruction.max_relative_l2_error,
        root_mean_square_error=reconstruction.root_mean_square_error,
        max_absolute_error=torch.max(torch.abs(residual)),
        mass_error=mass_error,
        mean_position_error=mean_position_error,
        width_error=width_error,
        mean_mass_error=torch.mean(torch.abs(mass_error)),
        max_mass_error=torch.max(torch.abs(mass_error)),
        mean_position_mae=torch.mean(torch.abs(mean_position_error)),
        width_mae=torch.mean(torch.abs(width_error)),
    )


__all__ = [
    "ProfileComparisonSummary",
    "ReconstructionErrorSummary",
    "SpectralBatchSummary",
    "SpectralTruncationSummary",
    "mode_counts_for_mass",
    "summarize_profile_comparison",
    "spectral_tail_mass",
    "spectral_weights",
    "summarize_spectral_batch",
    "summarize_profile_reconstruction",
    "summarize_spectral_coefficients",
]
