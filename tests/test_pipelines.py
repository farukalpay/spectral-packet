from __future__ import annotations

import math

import pytest
import torch

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.domain import InfiniteWell1D
from spectral_packet_engine.pipelines import analyze_quantum_state, analyze_tunneling
from spectral_packet_engine.projector import StateProjector
from spectral_packet_engine.state import make_truncated_gaussian_packet


def test_analyze_quantum_state_reports_position_expectation_separately_from_uncertainty() -> None:
    domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64)
    basis = InfiniteWellBasis(domain, 64)
    projector = StateProjector(basis)
    packet = make_truncated_gaussian_packet(
        domain,
        center=0.3,
        width=0.06,
        wavenumber=18.0,
    )
    state = projector.project_packet(packet)

    report = analyze_quantum_state(state.coefficients, basis)

    assert report.expectation_x == pytest.approx(0.3, abs=0.05)
    assert report.sigma_x > 0.0
    assert abs(report.expectation_x - report.sigma_x) > 1e-3
    assert report.variance_x == pytest.approx(report.sigma_x**2, rel=1e-6)


def test_analyze_tunneling_reports_packet_energy_window_and_transport_probabilities() -> None:
    report = analyze_tunneling(
        barrier_height=20.0,
        barrier_width_sigma=0.03,
        grid_points=128,
        num_modes=48,
        num_energies=64,
        packet_center=0.25,
        packet_width=0.04,
        packet_wavenumber=25.0,
        propagation_steps=64,
        dt=1e-5,
        device="cpu",
    )

    assert report.num_modes == 48
    assert report.energy_range[0] < report.energy_range[1]
    assert report.packet_energy_interval[0] <= report.packet_mean_energy <= report.packet_energy_interval[1]
    assert report.energy_range[0] <= report.comparison_energy <= report.energy_range[1]
    assert math.isfinite(report.transmission_at_packet_energy)
    assert math.isfinite(report.wkb_transmission_at_packet_energy)
    assert 0.0 <= report.transmitted_probability <= 1.0
    assert 0.0 <= report.reflected_probability <= 1.0
    assert report.barrier_left <= report.barrier_right
    assert report.propagation_total_time > 0.0
