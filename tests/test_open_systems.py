from __future__ import annotations

import torch

from spectral_packet_engine import (
    InstrumentResponse,
    MeasurementNoiseModel,
    apply_instrument_response,
    evolve_lindblad,
    finite_resolution_response_matrix,
    relaxation_lindblad_operator,
)


def test_lindblad_relaxation_preserves_trace_and_decays_excited_population() -> None:
    rho0 = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
    hamiltonian = torch.zeros(2, 2, dtype=torch.complex128)
    times = torch.linspace(0.0, 0.5, 9, dtype=torch.float64)

    summary = evolve_lindblad(
        rho0,
        hamiltonian,
        [relaxation_lindblad_operator(2, source=1, target=0, rate=1.5)],
        times,
    )

    assert tuple(summary.density_matrices.shape) == (9, 2, 2)
    torch.testing.assert_close(summary.trace, torch.ones_like(summary.trace), atol=1e-10, rtol=1e-10)
    assert summary.density_matrices[-1, 1, 1].real < summary.density_matrices[0, 1, 1].real
    assert summary.channels[0]["name"] == "relaxation[1->0]"


def test_instrument_response_preserves_mass_and_reports_uncertainty() -> None:
    coordinates = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    response = InstrumentResponse(
        finite_resolution_response_matrix(coordinates, sigma=0.08),
        observable_axis="position",
    )
    distribution = torch.zeros(9, dtype=torch.float64)
    distribution[4] = 1.0

    summary = apply_instrument_response(
        distribution,
        response,
        noise_model=MeasurementNoiseModel(model="independent-gaussian", scale=0.02),
    )

    assert tuple(summary.measured.shape) == (9,)
    assert torch.isclose(summary.measured.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.max(summary.normalization_error) < 1e-12
    assert torch.all(summary.uncertainty == 0.02)
    assert summary.response["observable_axis"] == "position"
