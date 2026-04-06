from __future__ import annotations

import pytest
import torch

from spectral_packet_engine import (
    analyze_coupled_channel_surfaces,
    analyze_separable_tensor_product_spectrum,
    analyze_structured_coupling,
    build_separable_2d_report,
    solve_radial_reduction,
)


def test_separable_tensor_product_spectrum_reports_structured_basis_budget_and_operator() -> None:
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
    assert summary.basis.tensor_shape == (4, 4)
    assert summary.mode_budget.total_tensor_mode_count == 16
    assert summary.mode_budget.retained_combined_state_count == 6
    assert summary.truncation.combined_state_truncation_applied is True
    assert summary.operator.kind == "kronecker-sum"
    assert summary.operator.coupling_kind == "none"
    assert tuple(summary.combined_eigenvalues.shape) == (6,)
    assert summary.ground_density_low_rank.retained_rank == 1
    assert summary.ground_density_low_rank.reconstruction_error < 1e-8
    assert torch.all(summary.transition_energies_from_ground >= 0)


def test_separable_2d_report_matches_closed_form_box_reference() -> None:
    report = build_separable_2d_report(
        num_modes_x=4,
        num_modes_y=5,
        num_combined_states=7,
        grid_points_x=32,
        grid_points_y=36,
        device="cpu",
    )

    assert report.overview.example_name == "box-plus-box"
    assert report.overview.axis_models == ("infinite-well", "infinite-well")
    assert report.overview.tensor_shape == (4, 5)
    assert report.overview.retained_combined_state_count == 7
    assert report.overview.max_absolute_reference_error < 1e-10
    torch.testing.assert_close(
        report.absolute_error_to_reference,
        torch.zeros_like(report.absolute_error_to_reference),
        atol=1e-10,
        rtol=1e-10,
    )


def test_structured_coupling_reports_low_rank_and_block_structure_without_generic_solver() -> None:
    coupling = torch.zeros(6, 6, dtype=torch.float64)
    coupling[:3, :3] = torch.eye(3)
    coupling[3:, 3:] = 0.5 * torch.eye(3)
    coupling[0, 3] = 0.05
    coupling[3, 0] = 0.05

    summary = analyze_structured_coupling(
        coupling,
        tensor_shape=(2, 3),
        block_partitions=((0, 1, 2), (3, 4, 5)),
        capture_fraction=0.95,
    )

    assert summary.tensor_shape == (2, 3)
    assert summary.matrix_shape == (6, 6)
    assert summary.low_rank_rank > 0
    assert summary.low_rank_energy_capture >= 0.95
    assert summary.block_count == 2
    assert summary.within_block_energy_fraction is not None
    assert summary.within_block_energy_fraction > 0.99
    assert summary.off_block_energy_fraction is not None
    assert summary.additive_diagonal_score is not None


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
