from __future__ import annotations

import pytest
import torch

from spectral_packet_engine import (
    analyze_coupled_channel_surfaces,
    analyze_separable_tensor_product_spectrum,
    solve_radial_reduction,
)


def test_separable_tensor_product_spectrum_reports_rank_one_ground_density() -> None:
    summary = analyze_separable_tensor_product_spectrum(
        family_x="harmonic",
        parameters_x={"omega": 10.0},
        family_y="harmonic",
        parameters_y={"omega": 7.0},
        num_points_x=80,
        num_points_y=80,
        num_states_x=4,
        num_states_y=4,
        num_combined_states=6,
        low_rank_rank=1,
        device="cpu",
    )

    assert summary.family_x == "harmonic"
    assert summary.family_y == "harmonic"
    assert tuple(summary.combined_eigenvalues.shape) == (6,)
    assert summary.ground_density_low_rank.retained_rank == 1
    assert summary.ground_density_low_rank.reconstruction_error < 1e-8
    assert torch.all(summary.transition_energies_from_ground >= 0)


def test_coupled_channel_surface_summary_exposes_gap_and_derivative_couplings() -> None:
    summary = analyze_coupled_channel_surfaces(
        domain_length=1.0,
        grid_points=128,
        slope=20.0,
        coupling=1.5,
        coupling_width=0.10,
        device="cpu",
    )

    assert tuple(summary.diabatic_potentials.shape) == (2, 128)
    assert tuple(summary.adiabatic_potentials.shape) == (2, 128)
    assert tuple(summary.derivative_couplings.shape) == (2, 2, 128)
    assert summary.minimum_gap > 0.0
    assert 0.0 <= summary.crossing_position <= 1.0


def test_radial_reduction_validates_positive_inner_radius() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        solve_radial_reduction(
            family="morse",
            parameters={"D_e": 8.0, "alpha": 2.0, "x_eq": 0.7},
            angular_momentum=0,
            radial_min=0.0,
            radial_max=3.0,
            device="cpu",
        )


def test_radial_reduction_returns_bounded_spectrum() -> None:
    summary = solve_radial_reduction(
        family="morse",
        parameters={"D_e": 8.0, "alpha": 2.0, "x_eq": 0.7},
        angular_momentum=1,
        radial_min=0.1,
        radial_max=3.0,
        num_points=96,
        num_states=4,
        device="cpu",
    )

    assert summary.family == "morse"
    assert tuple(summary.eigenvalues.shape) == (4,)
    assert tuple(summary.radial_grid.shape) == (96,)
    assert tuple(summary.effective_potential.shape) == (96,)
