from __future__ import annotations

import pytest
import torch

tensorflow = pytest.importorskip("tensorflow")

from spectral_packet_engine import (
    InfiniteWell1D,
    InfiniteWellBasis,
    TensorFlowModalRegressor,
    TensorFlowRegressorConfig,
    configure_tensorflow_runtime,
    normalize_profiles,
)


def test_tensorflow_modal_regressor_trains_on_synthetic_profiles() -> None:
    runtime = configure_tensorflow_runtime()
    domain = InfiniteWell1D.from_length(1.0)
    basis = InfiniteWellBasis(domain, num_modes=8)
    grid = domain.grid(128)
    sample_times = torch.linspace(0.0, 1.0, 64)

    profiles = []
    for center, sample_time in zip(torch.linspace(0.2, 0.8, 64), sample_times):
        width = 0.05 + 0.02 * sample_time
        profile = torch.exp(-((grid - center) ** 2) / (2 * width**2))
        profiles.append(normalize_profiles(profile, grid))
    profiles = torch.stack(profiles)

    regressor = TensorFlowModalRegressor(
        basis,
        config=TensorFlowRegressorConfig(
            profile_hidden_units=(256, 128),
            time_hidden_units=(16,),
            residual_blocks=1,
            dropout_rate=0.0,
            epochs=20,
            batch_size=16,
            validation_fraction=0.25,
        ),
    )
    result = regressor.fit(profiles, grid, sample_times=sample_times)
    predicted_moments = regressor.predict_moments(profiles[:4], sample_times=sample_times[:4])

    assert runtime.version
    assert result.epochs_ran >= 1
    assert result.best_validation_loss == result.best_validation_loss
    assert result.validation_profile_relative_l2 < 0.2
    assert result.validation_coefficient_mse < 0.01
    assert result.validation_moment_mae < 0.2
    assert result.parameter_count > 0
    assert result.validation_inference_profiles_per_second > 0.0
    assert predicted_moments.shape == (4, 2)
    assert "runtime" in result.to_dict()
