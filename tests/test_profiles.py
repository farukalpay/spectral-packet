from __future__ import annotations

import torch

from spectral_packet_engine import (
    InfiniteWell1D,
    InfiniteWellBasis,
    modal_tail,
    normalize_profiles,
    profile_mass,
    profile_mean,
    profile_variance,
    project_profiles_onto_basis,
    reconstruct_profiles_from_basis,
    relative_l2_error,
)


def test_profile_projection_recovers_a_single_basis_mode() -> None:
    domain = InfiniteWell1D.from_length(1.0)
    basis = InfiniteWellBasis(domain, num_modes=4)
    grid = domain.grid(4001)
    profile = basis.evaluate(grid)[:, 2]

    coefficients = project_profiles_onto_basis(profile, grid, basis)
    reconstruction = reconstruct_profiles_from_basis(coefficients, grid, basis)

    torch.testing.assert_close(coefficients, torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=domain.real_dtype), atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(reconstruction, profile, atol=5e-4, rtol=5e-4)


def test_profile_statistics_match_uniform_density() -> None:
    domain = InfiniteWell1D.from_length(1.0)
    grid = domain.grid(4001)
    profile = torch.ones_like(grid)
    normalized = normalize_profiles(profile, grid)

    torch.testing.assert_close(profile_mass(normalized, grid), torch.tensor(1.0, dtype=domain.real_dtype), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(profile_mean(normalized, grid), torch.tensor(0.5, dtype=domain.real_dtype), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(profile_variance(normalized, grid), torch.tensor(1.0 / 12.0, dtype=domain.real_dtype), atol=1e-5, rtol=1e-5)


def test_modal_tail_decreases_monotonically() -> None:
    coefficients = torch.tensor([4.0, 2.0, 1.0, 0.5], dtype=torch.float64)
    tail = modal_tail(coefficients)
    assert torch.all(tail[1:] <= tail[:-1])
    assert tail[-1].item() == 0.0


def test_relative_l2_error_is_small_for_well_resolved_profile() -> None:
    domain = InfiniteWell1D.from_length(1.0)
    basis = InfiniteWellBasis(domain, num_modes=32)
    grid = domain.grid(4001)
    profile = torch.exp(-((grid - 0.35) ** 2) / (2 * 0.08**2))
    profile = normalize_profiles(profile, grid)

    coefficients = project_profiles_onto_basis(profile, grid, basis)
    reconstruction = reconstruct_profiles_from_basis(coefficients, grid, basis)
    error = relative_l2_error(profile, reconstruction, grid)

    assert float(error) < 0.05
