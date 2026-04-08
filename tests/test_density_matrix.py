from __future__ import annotations

import torch

from spectral_packet_engine import analyze_state_density_matrix


def test_state_density_matrix_diagnostics_detect_pure_normalized_state() -> None:
    coefficients = torch.tensor([1.0, 1.0j], dtype=torch.complex128) / torch.sqrt(torch.tensor(2.0, dtype=torch.float64))

    diagnostics = analyze_state_density_matrix(coefficients)

    assert torch.isclose(diagnostics.trace, torch.tensor(1.0, dtype=torch.float64), atol=1e-12).item()
    assert diagnostics.trace_preserved.item() is True
    assert torch.isclose(diagnostics.normalized_purity, torch.tensor(1.0, dtype=torch.float64), atol=1e-10).item()
    assert torch.isclose(diagnostics.normalized_von_neumann_entropy, torch.tensor(0.0, dtype=torch.float64), atol=1e-10).item()
    assert diagnostics.normalized_rank.item() == 1
    assert diagnostics.normalized_is_pure.item() is True


def test_state_density_matrix_diagnostics_separate_trace_loss_from_mixedness() -> None:
    coefficients = torch.tensor(
        [
            [1.0, 0.0],
            [0.6, 0.8j],
        ],
        dtype=torch.complex128,
    ) / 2.0

    diagnostics = analyze_state_density_matrix(coefficients)

    assert diagnostics.trace.shape == (2,)
    assert torch.all(diagnostics.trace < 1.0).item()
    assert torch.all(~diagnostics.trace_preserved).item()
    assert torch.allclose(diagnostics.normalized_purity, torch.ones_like(diagnostics.normalized_purity), atol=1e-10)
    assert torch.allclose(
        diagnostics.normalized_von_neumann_entropy,
        torch.zeros_like(diagnostics.normalized_von_neumann_entropy),
        atol=1e-10,
    )
    assert torch.equal(diagnostics.normalized_rank, torch.ones_like(diagnostics.normalized_rank))
    assert torch.all(diagnostics.normalized_is_pure).item()
