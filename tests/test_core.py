from __future__ import annotations

import torch

from spectral_packet_engine import (
    GaussianPacketParameters,
    InfiniteWell1D,
    InfiniteWellBasis,
    PacketState,
    PlaneWavePacketParameters,
    SpectralState,
    expectation_position,
    interval_probability,
    make_plane_wave_packet,
    total_probability,
    variance_position,
    make_truncated_gaussian_packet,
)


def test_basis_is_approximately_orthonormal() -> None:
    domain = InfiniteWell1D.from_length(1.0)
    basis = InfiniteWellBasis(domain, num_modes=3)
    grid = domain.grid(20001)
    values = basis.evaluate(grid)

    overlaps = torch.empty((3, 3), dtype=domain.real_dtype)
    for i in range(3):
        for j in range(3):
            overlaps[i, j] = torch.trapezoid(values[:, i] * values[:, j], grid)

    torch.testing.assert_close(overlaps, torch.eye(3, dtype=domain.real_dtype), atol=5e-4, rtol=5e-4)


def test_domain_preserves_tensor_dtype_for_scalar_boundaries() -> None:
    domain = InfiniteWell1D(
        left=torch.tensor(0.0, dtype=torch.float32),
        right=torch.tensor(1.0, dtype=torch.float32),
    )

    assert domain.real_dtype == torch.float32
    assert domain.device == torch.device("cpu")


def test_truncated_gaussian_packet_is_normalized() -> None:
    domain = InfiniteWell1D.from_length(1.0)
    packet = make_truncated_gaussian_packet(
        domain,
        center=0.30,
        width=0.07,
        wavenumber=25.0,
    )
    grid = domain.grid(40001)
    probability = total_probability(packet.wavefunction(grid), grid)
    torch.testing.assert_close(probability, torch.tensor(1.0, dtype=domain.real_dtype), atol=2e-4, rtol=2e-4)


def test_packet_state_supports_multiple_components() -> None:
    domain = InfiniteWell1D.from_length(1.0)
    parameters = GaussianPacketParameters(
        center=torch.tensor([0.30, 0.65], dtype=domain.real_dtype),
        width=torch.tensor([0.07, 0.09], dtype=domain.real_dtype),
        wavenumber=torch.tensor([25.0, -18.0], dtype=domain.real_dtype),
        phase=torch.tensor([0.0, 0.2], dtype=domain.real_dtype),
    )
    state = PacketState(
        domain=domain,
        parameters=parameters,
        weights=torch.tensor([1.0 + 0.0j, 0.4 - 0.1j], dtype=domain.complex_dtype),
    )
    grid = domain.grid(1024)
    wavefunction = state.wavefunction(grid)
    assert wavefunction.shape == grid.shape
    assert torch.is_complex(wavefunction)


def test_plane_wave_packet_is_normalized_on_domain() -> None:
    domain = InfiniteWell1D.from_length(1.0)
    packet = make_plane_wave_packet(domain, wavenumber=12.0)
    grid = domain.grid(40001)

    probability = total_probability(packet.wavefunction(grid), grid)
    torch.testing.assert_close(probability, torch.tensor(1.0, dtype=domain.real_dtype), atol=2e-4, rtol=2e-4)


def test_packet_support_diagnostics_are_explicit() -> None:
    domain = InfiniteWell1D.from_length(1.0)
    gaussian = make_truncated_gaussian_packet(domain, center=0.08, width=0.12, wavenumber=8.0)
    plane = PacketState(
        domain=domain,
        parameters=PlaneWavePacketParameters.single(
            wavenumber=8.0,
            dtype=domain.real_dtype,
            device=domain.device,
        ),
    )

    gaussian_support = gaussian.support_diagnostics()
    plane_support = plane.support_diagnostics()

    assert gaussian_support.outside_probability_mass[0].item() > 0.0
    assert plane_support.outside_probability_mass[0].item() == 0.0
    assert plane_support.boundary_density_mismatch[0].item() > 0.0


def test_ground_state_observables_match_analytic_values() -> None:
    domain = InfiniteWell1D.from_length(1.0)
    state = SpectralState(domain, torch.tensor([1.0 + 0.0j], dtype=domain.complex_dtype))
    grid = domain.grid(20001)
    wavefunction = state.wavefunction(grid)

    torch.testing.assert_close(total_probability(wavefunction, grid), torch.tensor(1.0, dtype=domain.real_dtype), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(expectation_position(wavefunction, grid), torch.tensor(0.5, dtype=domain.real_dtype), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(
        variance_position(wavefunction, grid),
        torch.tensor(1.0 / 12.0 - 1.0 / (2.0 * torch.pi**2), dtype=domain.real_dtype),
        atol=3e-4,
        rtol=3e-4,
    )
    left = interval_probability(wavefunction, grid, 0.0, 0.5)
    right = interval_probability(wavefunction, grid, 0.5, 1.0)
    torch.testing.assert_close(left + right, torch.tensor(1.0, dtype=domain.real_dtype), atol=2e-5, rtol=2e-5)
