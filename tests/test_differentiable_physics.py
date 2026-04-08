from __future__ import annotations

import torch

from spectral_packet_engine import (
    GradientOptimizationConfig,
    design_potential_for_target_transition,
    harmonic_potential,
    optimize_packet_control,
)
from spectral_packet_engine.differentiable_physics import calibrate_potential_from_spectrum
from spectral_packet_engine.domain import InfiniteWell1D
from spectral_packet_engine.eigensolver import solve_eigenproblem


def test_calibrate_potential_from_spectrum_recovers_harmonic_family_parameter() -> None:
    domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device="cpu")
    target = solve_eigenproblem(
        lambda x: harmonic_potential(x, omega=9.0, domain=domain),
        domain,
        num_points=128,
        num_states=3,
    ).eigenvalues

    summary = calibrate_potential_from_spectrum(
        family="harmonic",
        target_eigenvalues=target,
        initial_guess={"omega": 5.0},
        num_points=128,
        optimization_config=GradientOptimizationConfig(steps=180, learning_rate=0.04),
        device="cpu",
    )

    assert summary.family == "harmonic"
    assert abs(summary.estimated_parameters["omega"] - 9.0) < 0.3
    assert summary.final_loss < 5e-2
    assert summary.parameter_posterior is not None
    assert summary.sensitivity is not None
    assert summary.observation_posterior is not None
    assert summary.observation_posterior.observation_shape == (3,)
    assert summary.observation_information is not None
    assert summary.observation_information.effective_observation_count > 0.0


def test_design_potential_for_target_transition_hits_requested_value() -> None:
    domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device="cpu")
    target_spectrum = solve_eigenproblem(
        lambda x: harmonic_potential(x, omega=8.0, domain=domain),
        domain,
        num_points=128,
        num_states=3,
    ).eigenvalues
    target_transition = float((target_spectrum[1] - target_spectrum[0]).item())

    summary = design_potential_for_target_transition(
        family="harmonic",
        target_transition=target_transition,
        transition_indices=(0, 1),
        initial_guess={"omega": 4.5},
        num_points=128,
        num_states=3,
        optimization_config=GradientOptimizationConfig(steps=180, learning_rate=0.04),
        device="cpu",
    )

    assert summary.family == "harmonic"
    assert abs(summary.achieved_transition - target_transition) < 3e-2
    assert tuple(summary.gradient.shape) == (1,)
    assert summary.final_loss < 1e-3


def test_optimize_packet_control_reports_gradient_and_improves_loss() -> None:
    summary = optimize_packet_control(
        initial_guess={
            "center": 0.25,
            "width": 0.08,
            "wavenumber": 18.0,
            "phase": 0.0,
        },
        objective="position",
        target_value=0.55,
        final_time=0.004,
        num_modes=48,
        quadrature_points=1024,
        grid_points=64,
        optimization_config=GradientOptimizationConfig(steps=60, learning_rate=0.03),
        device="cpu",
    )

    assert summary.history[0] >= summary.history[-1]
    assert tuple(summary.gradient_summary.gradient.shape) == (4,)
    assert tuple(summary.final_density.shape) == (64,)
    assert 0.0 <= summary.final_expectation_position <= 1.0


def test_optimize_packet_control_accepts_interval_probability_objective() -> None:
    summary = optimize_packet_control(
        initial_guess={
            "center": 0.25,
            "width": 0.08,
            "wavenumber": 18.0,
            "phase": 0.0,
        },
        objective="interval_probability",
        target_value=0.35,
        final_time=0.004,
        interval=(0.5, 1.0),
        num_modes=48,
        quadrature_points=1024,
        grid_points=64,
        optimization_config=GradientOptimizationConfig(steps=40, learning_rate=0.03),
        device="cpu",
    )

    assert summary.objective == "interval_probability"
    assert summary.final_interval_probability is not None
    assert 0.0 <= summary.final_interval_probability <= 1.0


def test_optimize_packet_control_preserves_legacy_target_aliases() -> None:
    summary = optimize_packet_control(
        initial_guess={
            "center": 0.25,
            "width": 0.08,
            "wavenumber": 18.0,
            "phase": 0.0,
        },
        objective="target_interval_probability",
        target_value=0.35,
        final_time=0.004,
        interval=(0.5, 1.0),
        num_modes=48,
        quadrature_points=1024,
        grid_points=64,
        optimization_config=GradientOptimizationConfig(steps=10, learning_rate=0.03),
        device="cpu",
    )

    assert summary.objective == "interval_probability"
