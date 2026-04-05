from __future__ import annotations

"""Structured reduced models built on the bounded-domain spectral core.

This module is intentionally narrow. It does not claim arbitrary 2D/3D
capabilities. It implements restricted, inspectable reductions that preserve
the repository's spectral center of gravity:

- separable tensor-product spectra from independent 1D axes,
- coupled-channel adiabatic surface analysis for avoided crossings,
- radial effective-coordinate reductions,
- low-rank matrix approximations for structured coefficient objects.
"""

from dataclasses import dataclass
from typing import Mapping

import torch

from spectral_packet_engine.domain import InfiniteWell1D
from spectral_packet_engine.eigensolver import solve_eigenproblem
from spectral_packet_engine.parametric_potentials import resolve_potential_family
from spectral_packet_engine.runtime import inspect_torch_runtime

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class LowRankFactorizationSummary:
    input_shape: tuple[int, int]
    retained_rank: int
    singular_values: Tensor
    energy_capture: float
    reconstruction_error: float
    left_factors: Tensor
    right_factors: Tensor


@dataclass(frozen=True, slots=True)
class SeparableSpectrumSummary:
    family_x: str
    family_y: str
    parameters_x: dict[str, float]
    parameters_y: dict[str, float]
    domain_lengths: tuple[float, float]
    axis_eigenvalues_x: Tensor
    axis_eigenvalues_y: Tensor
    combined_eigenvalues: Tensor
    state_index_pairs: tuple[tuple[int, int], ...]
    transition_energies_from_ground: Tensor
    ground_density_low_rank: LowRankFactorizationSummary
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CoupledChannelSurfaceSummary:
    grid: Tensor
    diabatic_potentials: Tensor
    adiabatic_potentials: Tensor
    coupling_profile: Tensor
    derivative_couplings: Tensor
    minimum_gap: float
    crossing_position: float
    mean_derivative_coupling: float
    max_derivative_coupling: float
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RadialReductionSummary:
    family: str
    parameters: dict[str, float]
    angular_momentum: int
    radial_interval: tuple[float, float]
    eigenvalues: Tensor
    radial_grid: Tensor
    effective_potential: Tensor
    assumptions: tuple[str, ...]


def low_rank_factorize_matrix(
    matrix,
    *,
    rank: int | None = None,
    capture_fraction: float = 0.99,
) -> LowRankFactorizationSummary:
    values = torch.as_tensor(matrix, dtype=torch.float64)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if rank is not None and rank <= 0:
        raise ValueError("rank must be positive when provided")
    if not (0.0 < capture_fraction <= 1.0):
        raise ValueError("capture_fraction must lie in (0, 1]")

    U, S, Vh = torch.linalg.svd(values, full_matrices=False)
    squared = S**2
    total = float(torch.sum(squared).item())
    if total <= 0.0:
        retained_rank = 1 if rank is None else min(rank, int(S.shape[0]))
        retained_rank = max(retained_rank, 1)
    elif rank is None:
        cumulative = torch.cumsum(squared, dim=0) / torch.sum(squared)
        retained_rank = int(torch.searchsorted(cumulative, torch.tensor(capture_fraction, dtype=S.dtype)).item()) + 1
    else:
        retained_rank = min(rank, int(S.shape[0]))

    retained_rank = max(1, retained_rank)
    truncated = (U[:, :retained_rank] * S[:retained_rank]) @ Vh[:retained_rank, :]
    residual = values - truncated
    return LowRankFactorizationSummary(
        input_shape=(int(values.shape[0]), int(values.shape[1])),
        retained_rank=retained_rank,
        singular_values=S.detach(),
        energy_capture=0.0 if total <= 0.0 else float(torch.sum(squared[:retained_rank]).item() / total),
        reconstruction_error=float(torch.linalg.norm(residual).item() / max(float(torch.linalg.norm(values).item()), 1e-12)),
        left_factors=U[:, :retained_rank].detach(),
        right_factors=Vh[:retained_rank, :].detach(),
    )


def analyze_separable_tensor_product_spectrum(
    *,
    family_x: str,
    parameters_x: Mapping[str, float],
    family_y: str,
    parameters_y: Mapping[str, float],
    domain_length_x: float = 1.0,
    domain_length_y: float = 1.0,
    num_points_x: int = 96,
    num_points_y: int = 96,
    num_states_x: int = 6,
    num_states_y: int = 6,
    num_combined_states: int = 12,
    low_rank_rank: int = 1,
    device: str | torch.device = "auto",
) -> SeparableSpectrumSummary:
    runtime = inspect_torch_runtime(device)
    domain_x = InfiniteWell1D.from_length(domain_length_x, dtype=runtime.preferred_real_dtype, device=runtime.device)
    domain_y = InfiniteWell1D.from_length(domain_length_y, dtype=runtime.preferred_real_dtype, device=runtime.device)
    family_def_x = resolve_potential_family(family_x)
    family_def_y = resolve_potential_family(family_y)

    result_x = solve_eigenproblem(
        family_def_x.build_from_mapping(domain_x, parameters_x),
        domain_x,
        num_points=num_points_x,
        num_states=num_states_x,
    )
    result_y = solve_eigenproblem(
        family_def_y.build_from_mapping(domain_y, parameters_y),
        domain_y,
        num_points=num_points_y,
        num_states=num_states_y,
    )

    combined_grid = result_x.eigenvalues[:, None] + result_y.eigenvalues[None, :]
    flat = combined_grid.reshape(-1)
    sorted_values, sorted_indices = torch.sort(flat)
    keep = min(int(num_combined_states), int(sorted_values.shape[0]))
    sorted_values = sorted_values[:keep]
    sorted_indices = sorted_indices[:keep]
    index_pairs = tuple(
        (
            int(index.item() // result_y.eigenvalues.shape[0]),
            int(index.item() % result_y.eigenvalues.shape[0]),
        )
        for index in sorted_indices
    )

    transition_energies = sorted_values[1:] - sorted_values[0] if keep > 1 else torch.empty(0, dtype=sorted_values.dtype)
    ground_density = torch.outer(result_x.eigenstates[0] ** 2, result_y.eigenstates[0] ** 2)
    low_rank = low_rank_factorize_matrix(ground_density, rank=low_rank_rank)

    return SeparableSpectrumSummary(
        family_x=family_def_x.name,
        family_y=family_def_y.name,
        parameters_x={name: float(value) for name, value in parameters_x.items()},
        parameters_y={name: float(value) for name, value in parameters_y.items()},
        domain_lengths=(float(domain_x.length), float(domain_y.length)),
        axis_eigenvalues_x=result_x.eigenvalues.detach(),
        axis_eigenvalues_y=result_y.eigenvalues.detach(),
        combined_eigenvalues=sorted_values.detach(),
        state_index_pairs=index_pairs,
        transition_energies_from_ground=transition_energies.detach(),
        ground_density_low_rank=low_rank,
        assumptions=(
            "The total Hamiltonian is assumed separable into independent x and y 1D components.",
            "The reported combined spectrum is the Kronecker-sum spectrum Ex_i + Ey_j of those axis problems.",
            "The low-rank summary applies to the separable ground-state density on the tensor grid, not to a general non-separable 2D state.",
        ),
    )


def _finite_difference_along_grid(values: Tensor, grid: Tensor) -> Tensor:
    derivative = torch.zeros_like(values)
    derivative[1:-1] = (values[2:] - values[:-2]) / (grid[2:, None, None] - grid[:-2, None, None])
    derivative[0] = (values[1] - values[0]) / (grid[1] - grid[0])
    derivative[-1] = (values[-1] - values[-2]) / (grid[-1] - grid[-2])
    return derivative


def analyze_coupled_channel_surfaces(
    *,
    domain_length: float = 1.0,
    grid_points: int = 256,
    slope: float = 30.0,
    bias: float = 0.0,
    coupling: float = 2.0,
    coupling_width: float = 0.12,
    device: str | torch.device = "cpu",
) -> CoupledChannelSurfaceSummary:
    if grid_points < 4:
        raise ValueError("grid_points must be at least 4")
    if coupling_width <= 0:
        raise ValueError("coupling_width must be positive")

    runtime = inspect_torch_runtime(device)
    grid = torch.linspace(0.0, domain_length, grid_points, dtype=runtime.preferred_real_dtype, device=runtime.device)
    center = torch.tensor(domain_length / 2.0, dtype=grid.dtype, device=grid.device)
    displacement = grid - center
    diabatic_1 = slope * displacement + bias / 2.0
    diabatic_2 = -slope * displacement - bias / 2.0
    coupling_profile = coupling * torch.exp(-(displacement**2) / (2 * coupling_width**2))

    hamiltonian = torch.zeros(grid_points, 2, 2, dtype=grid.dtype, device=grid.device)
    hamiltonian[:, 0, 0] = diabatic_1
    hamiltonian[:, 1, 1] = diabatic_2
    hamiltonian[:, 0, 1] = coupling_profile
    hamiltonian[:, 1, 0] = coupling_profile

    adiabatic_potentials, eigenvectors = torch.linalg.eigh(hamiltonian)
    eigenvector_derivative = _finite_difference_along_grid(eigenvectors, grid)
    derivative_couplings = torch.einsum("xai,xaj->xij", eigenvectors, eigenvector_derivative).permute(1, 2, 0)
    gap = adiabatic_potentials[:, 1] - adiabatic_potentials[:, 0]
    gap_index = int(torch.argmin(gap).item())

    return CoupledChannelSurfaceSummary(
        grid=grid.detach(),
        diabatic_potentials=torch.stack((diabatic_1, diabatic_2)).detach(),
        adiabatic_potentials=adiabatic_potentials.transpose(0, 1).detach(),
        coupling_profile=coupling_profile.detach(),
        derivative_couplings=derivative_couplings.detach(),
        minimum_gap=float(gap[gap_index].item()),
        crossing_position=float(grid[gap_index].item()),
        mean_derivative_coupling=float(torch.mean(torch.abs(derivative_couplings[0, 1])).item()),
        max_derivative_coupling=float(torch.max(torch.abs(derivative_couplings[0, 1])).item()),
        assumptions=(
            "This workflow models a reduced two-channel avoided crossing rather than a full multidimensional electronic-structure problem.",
            "Derivative couplings are estimated by finite differences of local adiabatic eigenvectors on the reported 1D grid.",
        ),
    )


def solve_radial_reduction(
    *,
    family: str,
    parameters: Mapping[str, float],
    angular_momentum: int = 0,
    radial_min: float = 0.05,
    radial_max: float = 3.0,
    num_points: int = 128,
    num_states: int = 6,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> RadialReductionSummary:
    if radial_min <= 0.0:
        raise ValueError("radial_min must be strictly positive for the centrifugal term")
    if radial_max <= radial_min:
        raise ValueError("radial_max must be greater than radial_min")
    if angular_momentum < 0:
        raise ValueError("angular_momentum must be non-negative")

    runtime = inspect_torch_runtime(device)
    domain = InfiniteWell1D(
        left=torch.tensor(radial_min, dtype=runtime.preferred_real_dtype, device=runtime.device),
        right=torch.tensor(radial_max, dtype=runtime.preferred_real_dtype, device=runtime.device),
        mass=torch.tensor(mass, dtype=runtime.preferred_real_dtype, device=runtime.device),
        hbar=torch.tensor(hbar, dtype=runtime.preferred_real_dtype, device=runtime.device),
    )
    family_def = resolve_potential_family(family)
    base_potential = family_def.build_from_mapping(domain, parameters)

    def effective_potential(radius: Tensor) -> Tensor:
        centrifugal = (domain.hbar**2 * angular_momentum * (angular_momentum + 1)) / (2 * domain.mass * radius**2)
        return base_potential(radius) + centrifugal

    result = solve_eigenproblem(
        effective_potential,
        domain,
        num_points=num_points,
        num_states=num_states,
    )

    return RadialReductionSummary(
        family=family_def.name,
        parameters={name: float(value) for name, value in parameters.items()},
        angular_momentum=int(angular_momentum),
        radial_interval=(float(domain.left), float(domain.right)),
        eigenvalues=result.eigenvalues.detach(),
        radial_grid=result.grid.detach(),
        effective_potential=result.potential_on_grid.detach(),
        assumptions=(
            "This is a 1D radial effective-coordinate reduction with Dirichlet boundaries on a finite interval.",
            "The centrifugal term l(l+1)hbar^2/(2mr^2) is included explicitly; no claim is made for full arbitrary 3D geometry.",
        ),
    )


__all__ = [
    "CoupledChannelSurfaceSummary",
    "LowRankFactorizationSummary",
    "RadialReductionSummary",
    "SeparableSpectrumSummary",
    "analyze_coupled_channel_surfaces",
    "analyze_separable_tensor_product_spectrum",
    "low_rank_factorize_matrix",
    "solve_radial_reduction",
]
