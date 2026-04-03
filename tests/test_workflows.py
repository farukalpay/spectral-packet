from __future__ import annotations

import numpy as np
import pytest
import torch

from spectral_packet_engine import (
    analyze_profile_table_spectra,
    compare_profile_tables,
    ProfileTable,
    compress_profile_table,
    evaluate_tensorflow_surrogate_on_profile_table,
    fit_gaussian_packet_to_profile_table,
    simulate_gaussian_packet,
    simulate_packet_sweep,
    summarize_profile_table,
    sweep_profile_table_compression,
    TensorFlowRegressorConfig,
    validate_installation,
)


def _synthetic_profile_table() -> ProfileTable:
    grid = np.linspace(0.0, 1.0, 64)
    times = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    profiles = []
    for center, width in zip(np.linspace(0.25, 0.55, len(times)), np.linspace(0.06, 0.09, len(times))):
        profile = np.exp(-((grid - center) ** 2) / (2 * width**2))
        profile = profile / np.trapezoid(profile, grid)
        profiles.append(profile)
    return ProfileTable(position_grid=grid, sample_times=times, profiles=np.asarray(profiles, dtype=np.float64))


def test_compress_profile_table_returns_low_error() -> None:
    summary = compress_profile_table(_synthetic_profile_table(), num_modes=24, device="cpu")

    assert summary.num_modes == 24
    assert float(summary.error_summary.mean_relative_l2_error) < 0.05
    assert float(summary.error_summary.max_relative_l2_error) < 0.08


def test_fit_gaussian_packet_to_profile_table_recovers_simulated_parameters() -> None:
    forward = simulate_gaussian_packet(
        center=0.30,
        width=0.07,
        wavenumber=25.0,
        times=[0.0, 1e-3, 3e-3],
        num_modes=96,
        quadrature_points=2048,
        grid_points=96,
        device="cpu",
    )
    table = ProfileTable(
        position_grid=forward.grid.detach().cpu().numpy(),
        sample_times=forward.times.detach().cpu().numpy(),
        profiles=forward.densities.detach().cpu().numpy(),
    )

    result = fit_gaussian_packet_to_profile_table(
        table,
        initial_guess={
            "center": 0.36,
            "width": 0.11,
            "wavenumber": 22.0,
            "phase": 0.0,
        },
        num_modes=96,
        quadrature_points=1024,
        steps=180,
        learning_rate=0.05,
        device="cpu",
    )

    assert result.final_loss < 1e-3
    assert abs(result.estimated_parameters.center[0].item() - 0.30) < 0.02
    assert abs(result.estimated_parameters.width[0].item() - 0.07) < 0.02
    assert abs(result.estimated_parameters.wavenumber[0].item() - 25.0) < 0.8


def test_profile_table_summary_and_compression_sweep() -> None:
    table = _synthetic_profile_table()

    summary = summarize_profile_table(table, device="cpu")
    sweep = sweep_profile_table_compression(table, mode_counts=[4, 8, 16], device="cpu")

    assert summary.num_samples == table.num_samples
    assert summary.num_positions == table.num_positions
    assert tuple(sweep.mode_counts.detach().cpu().tolist()) == (4.0, 8.0, 16.0)
    assert float(sweep.mean_relative_l2_error[-1]) <= float(sweep.mean_relative_l2_error[0])


def test_spectral_analysis_and_table_comparison() -> None:
    table = _synthetic_profile_table()

    analysis = analyze_profile_table_spectra(table, num_modes=24, device="cpu")
    comparison = compare_profile_tables(table, table, device="cpu")

    assert analysis.coefficients.shape == (table.num_samples, 24)
    assert analysis.spectral_summary.dominant_modes.shape[0] >= 1
    assert torch.all(
        analysis.spectral_summary.max_mode_counts_for_thresholds[1:]
        >= analysis.spectral_summary.max_mode_counts_for_thresholds[:-1]
    )
    assert float(comparison.comparison.max_relative_l2_error) == 0.0
    assert torch.allclose(
        comparison.comparison.mass_error,
        torch.zeros_like(comparison.comparison.mass_error),
    )


def test_validate_installation_and_packet_sweep() -> None:
    validation = validate_installation("cpu")
    sweep = simulate_packet_sweep(
        [
            {"center": 0.25, "width": 0.07, "wavenumber": 22.0},
            {"center": 0.35, "width": 0.08, "wavenumber": 24.0},
        ],
        times=[0.0, 1e-3],
        num_modes=64,
        quadrature_points=1024,
        grid_points=128,
        device="cpu",
    )

    assert validation.core_ready is True
    assert "python" in validation.stable_surfaces
    assert len(sweep.items) == 2
    assert sweep.items[0].final_total_probability > 0.99


def test_tensorflow_evaluation_returns_profile_comparison_when_available() -> None:
    pytest.importorskip("tensorflow")

    table = _synthetic_profile_table()
    evaluation = evaluate_tensorflow_surrogate_on_profile_table(
        table,
        num_modes=8,
        config=TensorFlowRegressorConfig(
            profile_hidden_units=(128, 64),
            time_hidden_units=(8,),
            residual_blocks=1,
            dropout_rate=0.0,
            epochs=8,
            batch_size=4,
            validation_fraction=0.25,
        ),
    )

    assert evaluation.predicted_coefficients.shape == (table.num_samples, 8)
    assert evaluation.predicted_moments.shape == (table.num_samples, 2)
    assert evaluation.reconstructed_profiles.shape == table.profiles.shape
    assert evaluation.comparison.mean_relative_l2_error >= 0.0
