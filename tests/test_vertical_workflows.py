from __future__ import annotations

import numpy as np
import torch

from spectral_packet_engine import (
    GradientOptimizationConfig,
    ProfileTable,
    harmonic_potential,
    run_profile_inference_workflow,
    run_spectroscopy_workflow,
    run_transport_resonance_workflow,
    save_profile_table_csv,
    simulate_gaussian_packet,
)
from spectral_packet_engine.domain import InfiniteWell1D
from spectral_packet_engine.eigensolver import solve_eigenproblem


def test_spectroscopy_workflow_prefers_true_harmonic_family() -> None:
    domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device="cpu")
    target = solve_eigenproblem(
        lambda x: harmonic_potential(x, omega=8.5, domain=domain),
        domain,
        num_points=128,
        num_states=3,
    ).eigenvalues

    summary = run_spectroscopy_workflow(
        target_eigenvalues=target,
        families=("harmonic", "double-well"),
        initial_guesses={
            "harmonic": {"omega": 5.0},
            "double-well": {"a_param": 1.5, "b_param": 1.0},
        },
        num_points=128,
        optimization_config=GradientOptimizationConfig(steps=180, learning_rate=0.04),
        device="cpu",
    )

    assert summary.family_inference.best_family == "harmonic"
    assert summary.family_inference.family_weights["harmonic"] > summary.family_inference.family_weights["double-well"]
    assert summary.line_assignment_root_mean_square_error < 5e-2


def test_transport_resonance_workflow_returns_tunneling_report() -> None:
    summary = run_transport_resonance_workflow(
        barrier_height=30.0,
        barrier_width_sigma=0.04,
        grid_points=128,
        num_modes=48,
        num_energies=80,
        device="cpu",
    )

    assert summary.barrier_family == "gaussian-barrier"
    assert summary.tunneling.num_energies == 80
    assert summary.tunneling.num_modes == 48
    assert summary.tunneling.transmission_at_half_barrier >= 0.0


def test_profile_inference_workflow_runs_report_inverse_and_feature_export(tmp_path) -> None:
    forward = simulate_gaussian_packet(
        center=0.30,
        width=0.07,
        wavenumber=25.0,
        times=[0.0, 1e-3, 3e-3],
        num_modes=48,
        quadrature_points=1024,
        grid_points=48,
        device="cpu",
    )
    table_path = tmp_path / "profiles.csv"
    save_profile_table_csv(
        ProfileTable(
            position_grid=forward.grid.detach().cpu().numpy(),
            sample_times=forward.times.detach().cpu().numpy(),
            profiles=forward.densities.detach().cpu().numpy(),
        ),
        table_path,
    )

    summary = run_profile_inference_workflow(
        table_path,
        initial_guess={
            "center": 0.35,
            "width": 0.10,
            "wavenumber": 23.0,
            "phase": 0.0,
        },
        analyze_num_modes=8,
        compress_num_modes=4,
        inverse_num_modes=48,
        feature_num_modes=6,
        quadrature_points=1024,
        device="cpu",
    )

    assert summary.report.overview.analyze_num_modes == 8
    assert summary.inverse_fit.physical_inference is not None
    assert summary.feature_export.num_features == 9
