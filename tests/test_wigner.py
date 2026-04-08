from __future__ import annotations

import torch

from spectral_packet_engine import InfiniteWell1D, InfiniteWellBasis, analyze_state_phase_space


def test_state_phase_space_diagnostics_support_batched_coefficients() -> None:
    domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device="cpu")
    basis = InfiniteWellBasis(domain, num_modes=4)
    coefficients = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0j, 0.0, 0.0],
        ],
        dtype=torch.complex128,
    )
    coefficients[1] = coefficients[1] / torch.sqrt(torch.tensor(2.0, dtype=torch.float64))

    diagnostics = analyze_state_phase_space(
        coefficients,
        basis,
        num_x_points=24,
        num_p_points=24,
    )

    assert diagnostics.W.shape == (2, 24, 24)
    assert diagnostics.x_marginal.shape == (2, 24)
    assert diagnostics.p_marginal.shape == (2, 24)
    assert diagnostics.negativity.shape == (2,)
    assert torch.allclose(
        diagnostics.total_integral,
        torch.ones_like(diagnostics.total_integral),
        atol=5e-2,
        rtol=5e-2,
    )


def test_state_phase_space_total_integral_tracks_state_norm() -> None:
    domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device="cpu")
    basis = InfiniteWellBasis(domain, num_modes=4)
    coefficients = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128) / 2.0

    diagnostics = analyze_state_phase_space(
        coefficients,
        basis,
        num_x_points=24,
        num_p_points=24,
    )

    assert diagnostics.W.shape == (24, 24)
    assert torch.isclose(diagnostics.total_integral, torch.tensor(0.25, dtype=torch.float64), atol=5e-2, rtol=5e-2).item()
