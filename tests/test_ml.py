from __future__ import annotations

import numpy as np
import pytest
import torch

from spectral_packet_engine import (
    InfiniteWell1D,
    InfiniteWellBasis,
    ModalSurrogateConfig,
    PyTorchModalRegressor,
    JAXModalRegressor,
    create_modal_regressor,
    evaluate_modal_surrogate_on_profile_table,
    inspect_ml_backends,
)
from spectral_packet_engine.ml import inspect_tensorflow_backend
from spectral_packet_engine.table_io import ProfileTable


def _synthetic_profile_table() -> ProfileTable:
    grid = np.linspace(0.0, 1.0, 64)
    times = np.linspace(0.0, 0.6, 16, dtype=np.float64)
    profiles = []
    for center, width in zip(np.linspace(0.20, 0.70, len(times)), np.linspace(0.05, 0.10, len(times))):
        profile = np.exp(-((grid - center) ** 2) / (2 * width**2))
        profile = profile / np.trapezoid(profile, grid)
        profiles.append(profile)
    return ProfileTable(position_grid=grid, sample_times=times, profiles=np.asarray(profiles, dtype=np.float64))


def test_inspect_ml_backends_reports_torch_and_jax() -> None:
    report = inspect_ml_backends("cpu")

    assert report.requested_device == "cpu"
    assert report.backends["torch"].available is True
    assert report.backends["jax"].available is True
    assert "torch" in report.runtime_available_backends
    assert set(report.project_supported_backends).issubset(set(report.runtime_available_backends))
    assert report.backends["torch"].python_version
    assert report.backends["jax"].preferred_device == "cpu"
    assert report.preferred_backend in {"torch", "jax", "tensorflow"}


def test_pytorch_modal_regressor_trains_on_synthetic_profiles() -> None:
    table = _synthetic_profile_table()
    domain = InfiniteWell1D.from_length(1.0)
    basis = InfiniteWellBasis(domain, num_modes=8)
    regressor = PyTorchModalRegressor(
        basis,
        config=ModalSurrogateConfig(
            profile_hidden_units=(96, 48),
            time_hidden_units=(8,),
            residual_blocks=0,
            dropout_rate=0.0,
            epochs=40,
            batch_size=8,
            validation_fraction=0.25,
            learning_rate=1e-3,
            device="cpu",
        ),
    )
    result = regressor.fit(
        torch.as_tensor(table.profiles),
        torch.as_tensor(table.position_grid),
        sample_times=torch.as_tensor(table.sample_times),
    )

    assert result.backend == "torch"
    assert result.epochs_ran >= 1
    assert result.parameter_count > 0
    assert result.validation_profile_relative_l2 < 0.4
    assert regressor.predict_moments(table.profiles[:2], sample_times=table.sample_times[:2]).shape == (2, 2)


def test_jax_modal_regressor_trains_on_synthetic_profiles() -> None:
    table = _synthetic_profile_table()
    domain = InfiniteWell1D.from_length(1.0)
    basis = InfiniteWellBasis(domain, num_modes=8)
    regressor = JAXModalRegressor(
        basis,
        config=ModalSurrogateConfig(
            profile_hidden_units=(64, 32),
            time_hidden_units=(8,),
            residual_blocks=0,
            dropout_rate=0.0,
            epochs=60,
            batch_size=8,
            validation_fraction=0.25,
            learning_rate=1e-3,
            device="cpu",
        ),
    )
    result = regressor.fit(
        table.profiles,
        table.position_grid,
        sample_times=table.sample_times,
    )

    assert result.backend == "jax"
    assert result.epochs_ran >= 1
    assert result.parameter_count > 0
    assert result.validation_profile_relative_l2 < 0.5
    assert regressor.predict_coefficients(table.profiles[:2], sample_times=table.sample_times[:2]).shape == (2, 8)


def test_generic_modal_workflow_uses_requested_backend() -> None:
    table = _synthetic_profile_table()
    evaluation = evaluate_modal_surrogate_on_profile_table(
        table,
        backend="torch",
        num_modes=8,
        config=ModalSurrogateConfig(
            profile_hidden_units=(96, 48),
            time_hidden_units=(8,),
            residual_blocks=0,
            epochs=30,
            batch_size=8,
            validation_fraction=0.25,
            learning_rate=1e-3,
            device="cpu",
        ),
    )

    assert evaluation.backend == "torch"
    assert evaluation.predicted_coefficients.shape == (table.num_samples, 8)
    assert evaluation.reconstructed_profiles.shape == table.profiles.shape
    assert evaluation.comparison.mean_relative_l2_error >= 0.0


def test_create_modal_regressor_rejects_unavailable_backend() -> None:
    table = _synthetic_profile_table()
    domain = InfiniteWell1D.from_length(1.0)
    basis = InfiniteWellBasis(domain, num_modes=8)
    if inspect_ml_backends().backends["tensorflow"].available:
        pytest.skip("TensorFlow is available in this environment")
    with pytest.raises(ModuleNotFoundError):
        create_modal_regressor("tensorflow", basis, config=ModalSurrogateConfig())


def test_inspect_tensorflow_backend_reports_runtime_probe_failures(monkeypatch) -> None:
    monkeypatch.setattr("spectral_packet_engine.ml.tensorflow_is_available", lambda: True)

    def _boom(**kwargs):
        raise RuntimeError("broken runtime")

    monkeypatch.setattr("spectral_packet_engine.ml.configure_tensorflow_runtime", _boom)

    report = inspect_tensorflow_backend()

    assert report.available is False
    assert any("broken runtime" in note for note in report.notes)
