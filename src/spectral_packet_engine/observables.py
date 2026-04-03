from __future__ import annotations

import torch

from spectral_packet_engine.domain import coerce_scalar_tensor, coerce_tensor

Tensor = torch.Tensor


def _validate_grid(grid) -> Tensor:
    spatial_grid = coerce_tensor(grid)
    if spatial_grid.ndim != 1:
        raise ValueError("spatial grid must be one-dimensional")
    if spatial_grid.shape[0] < 2:
        raise ValueError("spatial grid must contain at least two points")
    if torch.is_complex(spatial_grid):
        raise TypeError("spatial grid must be real-valued")
    if not torch.isfinite(spatial_grid).all().item():
        raise ValueError("spatial grid must be finite")
    if not torch.all(spatial_grid[1:] > spatial_grid[:-1]).item():
        raise ValueError("spatial grid must be strictly increasing")
    return spatial_grid


def _validate_wavefunction(wavefunction, grid: Tensor) -> Tensor:
    values = coerce_tensor(wavefunction, device=grid.device)
    if values.shape[-1] != grid.shape[0]:
        raise ValueError("wavefunction values must align with the last grid dimension")
    return values


def grid_probability_density(wavefunction) -> Tensor:
    values = coerce_tensor(wavefunction)
    return torch.abs(values) ** 2


probability_density = grid_probability_density


def l2_norm(wavefunction, grid) -> Tensor:
    spatial_grid = _validate_grid(grid)
    values = _validate_wavefunction(wavefunction, spatial_grid)
    density = grid_probability_density(values)
    return torch.trapezoid(density, spatial_grid, dim=-1)


def total_probability(wavefunction, grid) -> Tensor:
    return l2_norm(wavefunction, grid)


def expectation_position(wavefunction, grid) -> Tensor:
    spatial_grid = _validate_grid(grid)
    values = _validate_wavefunction(wavefunction, spatial_grid)
    density = grid_probability_density(values)
    norm = torch.trapezoid(density, spatial_grid, dim=-1)
    numerator = torch.trapezoid(density * spatial_grid, spatial_grid, dim=-1)
    return numerator / norm


def variance_position(wavefunction, grid) -> Tensor:
    spatial_grid = _validate_grid(grid)
    values = _validate_wavefunction(wavefunction, spatial_grid)
    density = grid_probability_density(values)
    norm = torch.trapezoid(density, spatial_grid, dim=-1)
    mean = expectation_position(values, spatial_grid)
    centered = spatial_grid - mean[..., None]
    second = torch.trapezoid(density * centered**2, spatial_grid, dim=-1)
    return second / norm


def _interpolate_last_dimension(grid: Tensor, values: Tensor, point: Tensor) -> Tensor:
    if point <= grid[0]:
        return values[..., 0]
    if point >= grid[-1]:
        return values[..., -1]

    index = int(torch.searchsorted(grid, point).item())
    left_x = grid[index - 1]
    right_x = grid[index]
    left_y = values[..., index - 1]
    right_y = values[..., index]
    weight = (point - left_x) / (right_x - left_x)
    return left_y + weight * (right_y - left_y)


def interval_probability(wavefunction, grid, left, right) -> Tensor:
    spatial_grid = _validate_grid(grid)
    values = _validate_wavefunction(wavefunction, spatial_grid)
    density = grid_probability_density(values)

    left_edge = coerce_scalar_tensor(left, dtype=spatial_grid.dtype, device=spatial_grid.device)
    right_edge = coerce_scalar_tensor(right, dtype=spatial_grid.dtype, device=spatial_grid.device)
    if not (right_edge >= left_edge).item():
        raise ValueError("interval requires right >= left")

    support_left = torch.maximum(left_edge, spatial_grid[0])
    support_right = torch.minimum(right_edge, spatial_grid[-1])
    if not (support_right > support_left).item():
        return torch.zeros(density.shape[:-1], dtype=density.dtype, device=density.device)

    interior_mask = (spatial_grid > support_left) & (spatial_grid < support_right)
    interior_grid = spatial_grid[interior_mask]
    interior_density = density[..., interior_mask]

    left_value = _interpolate_last_dimension(spatial_grid, density, support_left)[..., None]
    right_value = _interpolate_last_dimension(spatial_grid, density, support_right)[..., None]
    augmented_grid = torch.cat([support_left.reshape(1), interior_grid, support_right.reshape(1)])
    augmented_density = torch.cat([left_value, interior_density, right_value], dim=-1)
    return torch.trapezoid(augmented_density, augmented_grid, dim=-1)


__all__ = [
    "expectation_position",
    "grid_probability_density",
    "interval_probability",
    "l2_norm",
    "probability_density",
    "total_probability",
    "variance_position",
]
