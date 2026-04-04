from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.domain import coerce_tensor

Tensor = torch.Tensor


def _validate_grid(grid) -> Tensor:
    spatial_grid = coerce_tensor(grid)
    if spatial_grid.ndim != 1:
        raise ValueError("grid must be one-dimensional")
    if spatial_grid.shape[0] < 2:
        raise ValueError("grid must contain at least two points")
    if torch.is_complex(spatial_grid):
        raise TypeError("grid must be real-valued")
    if not torch.isfinite(spatial_grid).all().item():
        raise ValueError("grid must be finite")
    if not torch.all(spatial_grid[1:] > spatial_grid[:-1]).item():
        raise ValueError("grid must be strictly increasing")
    return spatial_grid


def _validate_profiles(profiles, grid: Tensor) -> Tensor:
    values = coerce_tensor(profiles, dtype=grid.dtype, device=grid.device)
    if values.shape[-1] != grid.shape[0]:
        raise ValueError("profiles must align with the last grid dimension")
    if torch.is_complex(values):
        raise TypeError("profiles must be real-valued")
    if not torch.isfinite(values).all().item():
        raise ValueError("profiles must be finite")
    return values


def _require_positive_profile_mass(values: Tensor, grid: Tensor, *, context: str) -> Tensor:
    mass = profile_mass(values, grid)
    if torch.any(mass <= 0).item():
        raise ValueError(f"{context} requires profiles with positive mass")
    return mass


def profile_mass(profiles, grid) -> Tensor:
    spatial_grid = _validate_grid(grid)
    values = _validate_profiles(profiles, spatial_grid)
    return torch.trapezoid(values, spatial_grid, dim=-1)


def normalize_profiles(profiles, grid) -> Tensor:
    spatial_grid = _validate_grid(grid)
    values = _validate_profiles(profiles, spatial_grid)
    mass = _require_positive_profile_mass(values, spatial_grid, context="normalize_profiles")
    return values / mass[..., None]


def profile_mean(profiles, grid) -> Tensor:
    spatial_grid = _validate_grid(grid)
    values = _validate_profiles(profiles, spatial_grid)
    mass = _require_positive_profile_mass(values, spatial_grid, context="profile_mean")
    return torch.trapezoid(values * spatial_grid, spatial_grid, dim=-1) / mass


def profile_variance(profiles, grid) -> Tensor:
    spatial_grid = _validate_grid(grid)
    values = _validate_profiles(profiles, spatial_grid)
    mean = profile_mean(values, spatial_grid)
    mass = _require_positive_profile_mass(values, spatial_grid, context="profile_variance")
    centered = spatial_grid - mean[..., None]
    return torch.trapezoid(values * centered**2, spatial_grid, dim=-1) / mass


def project_profiles_onto_basis(
    profiles,
    grid,
    basis: InfiniteWellBasis,
) -> Tensor:
    spatial_grid = _validate_grid(grid).to(dtype=basis.domain.real_dtype, device=basis.domain.device)
    values = _validate_profiles(profiles, spatial_grid)
    basis_values = basis.evaluate(spatial_grid)
    return torch.trapezoid(values[..., :, None] * basis_values, spatial_grid, dim=-2)


def reconstruct_profiles_from_basis(
    coefficients,
    grid,
    basis: InfiniteWellBasis,
) -> Tensor:
    spatial_grid = _validate_grid(grid).to(dtype=basis.domain.real_dtype, device=basis.domain.device)
    coeffs = coerce_tensor(coefficients, dtype=basis.domain.real_dtype, device=basis.domain.device)
    basis_values = basis.evaluate(spatial_grid)
    if coeffs.ndim == 1:
        if coeffs.shape[0] != basis.num_modes:
            raise ValueError("coefficient vector length does not match the basis")
        return coeffs @ basis_values.transpose(0, 1)
    if coeffs.ndim == 2:
        if coeffs.shape[-1] != basis.num_modes:
            raise ValueError("coefficient matrix width does not match the basis")
        return coeffs @ basis_values.transpose(0, 1)
    raise ValueError("coefficients must be one- or two-dimensional")


def relative_l2_error(reference, approximation, grid) -> Tensor:
    spatial_grid = _validate_grid(grid)
    reference_values = _validate_profiles(reference, spatial_grid)
    approximation_values = _validate_profiles(approximation, spatial_grid)
    residual = reference_values - approximation_values
    numerator = torch.sqrt(torch.trapezoid(residual**2, spatial_grid, dim=-1))
    denominator = torch.sqrt(torch.trapezoid(reference_values**2, spatial_grid, dim=-1))
    if torch.any(denominator <= 0).item():
        raise ValueError("reference profiles must have positive L2 norm")
    return numerator / denominator


def modal_energy(coefficients) -> Tensor:
    coeffs = coerce_tensor(coefficients)
    if not torch.isfinite(coeffs).all().item():
        raise ValueError("coefficients must be finite")
    return torch.abs(coeffs) ** 2


def modal_tail(coefficients) -> Tensor:
    energies = modal_energy(coefficients)
    total = torch.sum(energies, dim=-1, keepdim=True)
    safe_total = torch.where(total > 0, total, torch.ones_like(total))
    cumulative = torch.cumsum(energies, dim=-1)
    return 1.0 - cumulative / safe_total


@dataclass(frozen=True, slots=True)
class ProfileCompressionResult:
    basis: InfiniteWellBasis
    coefficients: Tensor
    reconstruction: Tensor
    relative_l2_error: Tensor

    @property
    def mean_relative_l2_error(self) -> Tensor:
        return torch.mean(self.relative_l2_error)

    @property
    def max_relative_l2_error(self) -> Tensor:
        return torch.max(self.relative_l2_error)


@dataclass(frozen=True, slots=True)
class ProfileCompressionSummary:
    mode_counts: Tensor
    mean_relative_l2_error: Tensor
    max_relative_l2_error: Tensor


def compress_profiles(
    profiles,
    grid,
    *,
    basis: InfiniteWellBasis | None = None,
    domain=None,
    num_modes: int | None = None,
) -> ProfileCompressionResult:
    spatial_grid = _validate_grid(grid)
    values = _validate_profiles(profiles, spatial_grid)
    if basis is None:
        if domain is None or num_modes is None:
            raise ValueError("provide either basis or both domain and num_modes")
        basis = InfiniteWellBasis(domain, int(num_modes))
    coefficients = project_profiles_onto_basis(values, spatial_grid, basis)
    reconstruction = reconstruct_profiles_from_basis(coefficients, spatial_grid, basis)
    error = relative_l2_error(values, reconstruction, spatial_grid)
    return ProfileCompressionResult(
        basis=basis,
        coefficients=coefficients,
        reconstruction=reconstruction,
        relative_l2_error=error,
    )


def summarize_profile_compression(
    profiles,
    grid,
    *,
    mode_counts,
    domain,
) -> ProfileCompressionSummary:
    spatial_grid = _validate_grid(grid)
    values = _validate_profiles(profiles, spatial_grid)
    counts = coerce_tensor(mode_counts, dtype=torch.int64, device=spatial_grid.device)
    if counts.ndim != 1:
        raise ValueError("mode_counts must be one-dimensional")
    if counts.numel() == 0:
        raise ValueError("mode_counts must not be empty")
    if not torch.all(counts > 0).item():
        raise ValueError("mode_counts must contain only positive values")

    max_count = int(torch.max(counts).item())
    basis = InfiniteWellBasis(domain, max_count)
    basis_values = basis.evaluate(spatial_grid).to(dtype=values.dtype, device=values.device)
    coefficients = torch.trapezoid(values[..., :, None] * basis_values, spatial_grid, dim=-2)
    mean_errors = []
    max_errors = []
    for count in counts.tolist():
        basis_slice = basis_values[:, : int(count)]
        reconstruction = coefficients[..., : int(count)] @ basis_slice.transpose(0, 1)
        error = relative_l2_error(values, reconstruction, spatial_grid)
        mean_errors.append(torch.mean(error))
        max_errors.append(torch.max(error))

    return ProfileCompressionSummary(
        mode_counts=counts.to(dtype=spatial_grid.dtype),
        mean_relative_l2_error=torch.stack(mean_errors),
        max_relative_l2_error=torch.stack(max_errors),
    )


__all__ = [
    "ProfileCompressionResult",
    "ProfileCompressionSummary",
    "compress_profiles",
    "modal_energy",
    "modal_tail",
    "normalize_profiles",
    "profile_mass",
    "profile_mean",
    "profile_variance",
    "project_profiles_onto_basis",
    "reconstruct_profiles_from_basis",
    "relative_l2_error",
    "summarize_profile_compression",
]
