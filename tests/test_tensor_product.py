from __future__ import annotations

import torch

from spectral_packet_engine.tensor_product import (
    KroneckerSumOperator2D,
    TensorProductBasis2D,
    build_tensor_mode_budget_2d,
    make_infinite_well_axis_modes,
    summarize_tensor_product_basis_2d,
    summarize_tensor_truncation_2d,
)


def test_tensor_product_basis_2d_uses_x_major_indexing() -> None:
    axis_x = make_infinite_well_axis_modes("x", num_modes=2, evaluation_grid_points=8)
    axis_y = make_infinite_well_axis_modes("y", num_modes=3, evaluation_grid_points=8)
    basis = TensorProductBasis2D(axis_x=axis_x, axis_y=axis_y)

    assert basis.shape == (2, 3)
    assert basis.total_mode_count == 6
    assert basis.flatten_index(1, 2) == 5
    assert basis.unflatten_index(5) == (1, 2)

    summary = summarize_tensor_product_basis_2d(basis)
    assert summary.flattening_order == "x-major-then-y"
    assert summary.tensor_shape == (2, 3)
    assert summary.total_mode_count == 6


def test_kronecker_sum_operator_matches_explicit_dense_assembly() -> None:
    axis_energies_x = torch.tensor([1.0, 4.0], dtype=torch.float64)
    axis_energies_y = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    coupling = torch.tensor([0.0, 0.5, 0.0, 0.2, 0.0, 0.1], dtype=torch.float64)
    operator = KroneckerSumOperator2D(
        axis_energies_x=axis_energies_x,
        axis_energies_y=axis_energies_y,
        coupling_diagonal=coupling,
    )

    identity_x = torch.eye(2, dtype=torch.float64)
    identity_y = torch.eye(3, dtype=torch.float64)
    expected = (
        torch.kron(torch.diag(axis_energies_x), identity_y)
        + torch.kron(identity_x, torch.diag(axis_energies_y))
        + torch.diag(coupling)
    )

    torch.testing.assert_close(operator.to_matrix(), expected)

    values, pairs = operator.lowest_states(4)
    expected_diagonal, expected_indices = torch.sort(torch.diagonal(expected))
    torch.testing.assert_close(values, expected_diagonal[:4])
    assert pairs == tuple((int(index.item() // 3), int(index.item() % 3)) for index in expected_indices[:4])


def test_tensor_mode_budget_and_truncation_are_explicit() -> None:
    axis_x = make_infinite_well_axis_modes("x", num_modes=3, evaluation_grid_points=8)
    axis_y = make_infinite_well_axis_modes("y", num_modes=4, evaluation_grid_points=8)
    basis = TensorProductBasis2D(axis_x=axis_x, axis_y=axis_y)
    operator = KroneckerSumOperator2D(axis_energies_x=axis_x.eigenvalues, axis_energies_y=axis_y.eigenvalues)
    sorted_diagonal = torch.sort(operator.diagonal_entries())[0]

    budget = build_tensor_mode_budget_2d(
        basis,
        requested_combined_state_count=5,
        retained_combined_state_count=5,
    )
    truncation = summarize_tensor_truncation_2d(
        basis,
        retained_combined_state_count=5,
        sorted_diagonal=sorted_diagonal,
    )

    assert budget.total_tensor_mode_count == 12
    assert budget.requested_combined_state_count == 5
    assert budget.retained_combined_state_count == 5
    assert truncation.combined_state_truncation_applied is True
    assert truncation.discarded_tensor_mode_count == 7
    assert truncation.retained_tensor_fraction == 5 / 12
    assert truncation.first_discarded_eigenvalue is not None
