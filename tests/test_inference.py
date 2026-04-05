from __future__ import annotations

import torch

from spectral_packet_engine import (
    EstimationConfig,
    GaussianPacketEstimator,
    GaussianPacketParameters,
    InfiniteWell1D,
    ProjectionConfig,
)


def test_density_based_reconstruction_recovers_packet_parameters() -> None:
    domain = InfiniteWell1D.from_length(1.0)
    estimator = GaussianPacketEstimator(
        domain,
        num_modes=96,
        projection_config=ProjectionConfig(quadrature_points=1024),
        estimation_config=EstimationConfig(steps=180, learning_rate=0.05),
    )
    observation_grid = domain.grid(96)
    times = torch.tensor([0.0, 1e-3, 3e-3, 5e-3, 1e-2], dtype=domain.real_dtype)

    truth = GaussianPacketParameters.single(
        center=0.30,
        width=0.07,
        wavenumber=25.0,
        dtype=domain.real_dtype,
        device=domain.device,
    )
    target = estimator.predict(
        truth,
        observation_grid=observation_grid,
        times=times,
        observation_mode="density",
    ).detach()

    initial_guess = GaussianPacketParameters.single(
        center=0.36,
        width=0.11,
        wavenumber=22.0,
        dtype=domain.real_dtype,
        device=domain.device,
    )
    result = estimator.fit(
        observation_grid=observation_grid,
        times=times,
        target=target,
        initial_guess=initial_guess,
        observation_mode="density",
    )

    assert result.final_loss < 1e-3
    assert abs(result.parameters.center[0].item() - 0.30) < 0.02
    assert abs(result.parameters.width[0].item() - 0.07) < 0.02
    assert abs(result.parameters.wavenumber[0].item() - 25.0) < 0.8

    posterior = estimator.infer(
        result.parameters,
        observation_grid=observation_grid,
        times=times,
        target=target,
        observation_mode="density",
    )

    assert posterior.parameter_posterior.parameter_names == ("center", "width", "wavenumber")
    assert posterior.parameter_posterior.mean.shape == (3,)
    assert posterior.parameter_posterior.covariance.shape == (3, 3)
    assert posterior.parameter_posterior.standard_deviation.shape == (3,)
    assert 0.0 <= posterior.parameter_posterior.identifiability_score <= 1.0
    assert posterior.parameter_posterior.noise_scale > 0.0
    assert posterior.coefficient_posterior is not None
    assert posterior.coefficient_posterior.mean.shape == (96,)
    assert posterior.sensitivity is not None
    assert posterior.sensitivity.observation_shape == (5, 96)
    assert posterior.sensitivity.one_sigma_effect.shape == (3, 5, 96)


def test_custom_observation_operator_returns_expected_shape() -> None:
    domain = InfiniteWell1D.from_length(1.0)
    estimator = GaussianPacketEstimator(
        domain,
        num_modes=64,
        projection_config=ProjectionConfig(quadrature_points=1024),
    )
    parameters = GaussianPacketParameters.single(
        center=0.30,
        width=0.07,
        wavenumber=25.0,
        dtype=domain.real_dtype,
        device=domain.device,
    )
    observation_grid = domain.grid(128)
    times = torch.tensor([0.0, 3e-3, 1e-2], dtype=domain.real_dtype)

    summary = estimator.predict(
        parameters,
        observation_grid=observation_grid,
        times=times,
        observation_operator=lambda record: torch.stack(
            [
                record.interval_probability(0.0, 0.5),
                record.interval_probability(0.5, 1.0),
            ],
            dim=-1,
        ),
    )

    assert summary.shape == (3, 2)
