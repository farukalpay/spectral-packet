from __future__ import annotations

import torch

from spectral_packet_engine import (
    InfiniteWell1D,
    InfiniteWellBasis,
    ProjectionConfig,
    SpectralPropagator,
    SpectralState,
    StateProjector,
    simulate,
    make_truncated_gaussian_packet,
)


def _build_reference_stack(
    *,
    num_modes: int = 200,
    quadrature_points: int = 8192,
):
    domain = InfiniteWell1D.from_length(1.0)
    basis = InfiniteWellBasis(domain, num_modes=num_modes)
    projector = StateProjector(basis, ProjectionConfig(quadrature_points=quadrature_points))
    propagator = SpectralPropagator(basis)
    packet = make_truncated_gaussian_packet(domain, center=0.30, width=0.07, wavenumber=25.0)
    return domain, basis, projector, propagator, packet


def test_projection_recovers_single_mode() -> None:
    domain = InfiniteWell1D.from_length(1.0)
    basis = InfiniteWellBasis(domain, num_modes=3)
    projector = StateProjector(basis, ProjectionConfig(quadrature_points=4001))
    grid = domain.grid(4001)
    reference_state = SpectralState(domain, torch.tensor([0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j], dtype=domain.complex_dtype))
    wavefunction = reference_state.wavefunction(grid)

    projected = projector.project_wavefunction(wavefunction, grid)
    torch.testing.assert_close(
        torch.abs(projected.coefficients),
        torch.tensor([0.0, 0.0, 1.0], dtype=domain.real_dtype),
        atol=5e-4,
        rtol=5e-4,
    )


def test_reference_projection_matches_expected_mode_weights() -> None:
    _, _, projector, _, packet = _build_reference_stack()
    state = projector.project_packet(packet)
    weights = torch.abs(state.coefficients) ** 2

    torch.testing.assert_close(weights[7], torch.tensor(0.17515, dtype=weights.dtype), atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(weights[6], torch.tensor(0.16027, dtype=weights.dtype), atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(weights[8], torch.tensor(0.15800, dtype=weights.dtype), atol=3e-3, rtol=3e-3)


def test_reference_time_evolution_preserves_norm_and_tracks_half_interval_probabilities() -> None:
    domain, _, projector, propagator, packet = _build_reference_stack()
    grid = domain.grid(4000)
    times = torch.tensor([0.0, 1e-3, 3e-3, 5e-3, 1e-2], dtype=domain.real_dtype)

    record = simulate(packet, times, projector=projector, propagator=propagator, grid=grid)
    probabilities = record.total_probability()
    torch.testing.assert_close(probabilities, torch.ones_like(probabilities), atol=2e-4, rtol=2e-4)

    left = record.interval_probability(0.0, 0.5)
    right = record.interval_probability(0.5, 1.0)
    torch.testing.assert_close(left + right, torch.ones_like(left), atol=2e-4, rtol=2e-4)

    torch.testing.assert_close(left, torch.tensor([0.997862, 0.993559, 0.956157, 0.829933, 0.308774], dtype=domain.real_dtype), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(right, torch.tensor([0.002137, 0.006441, 0.043842, 0.170066, 0.691226], dtype=domain.real_dtype), atol=5e-3, rtol=5e-3)
