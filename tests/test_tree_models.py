from __future__ import annotations

from pathlib import Path

import pytest

from spectral_packet_engine import TabularDataset
from spectral_packet_engine.tree_models import (
    create_tree_estimator,
    inspect_tree_backends,
    prepare_tree_dataset,
    train_tree_model_on_dataset,
    tune_tree_model_on_dataset,
)


def _regression_dataset() -> TabularDataset:
    rows: list[dict[str, float]] = []
    for index in range(24):
        time = float(index) / 23.0
        mode_1 = 0.15 + 0.04 * index
        mode_2 = 1.8 - 0.03 * index
        mean_position = 0.2 + 0.02 * index
        target = 1.5 * mode_1 - 0.8 * mode_2 + 0.4 * mean_position + 0.2 * time
        rows.append(
            {
                "time": time,
                "mode_1": mode_1,
                "mode_2": mode_2,
                "mean_position": mean_position,
                "target": target,
            }
        )
    return TabularDataset.from_rows(rows)


def test_tree_backend_inspection_reports_supported_resolution() -> None:
    report = inspect_tree_backends()

    assert report.libraries["sklearn"].available is True
    assert report.libraries["sklearn"].package_extra == "ml-tree-core"
    assert report.resolution.resolved_library == "sklearn"
    assert "sklearn" in report.available_libraries


def test_prepare_tree_dataset_rejects_non_finite_values() -> None:
    dataset = TabularDataset.from_rows(
        [
            {"mode_1": 0.1, "mode_2": 0.9, "target": 1.0},
            {"mode_1": float("nan"), "mode_2": 0.8, "target": 2.0},
        ]
    )

    with pytest.raises(ValueError, match="missing or non-finite values"):
        prepare_tree_dataset(dataset, target_column="target")


def test_train_tree_model_on_dataset_exports_model_and_metrics(tmp_path) -> None:
    summary = train_tree_model_on_dataset(
        _regression_dataset(),
        target_column="target",
        library="sklearn",
        params={"n_estimators": 24, "max_depth": 5},
        test_fraction=0.25,
        random_state=2,
        export_dir=tmp_path / "trained_model",
    )

    assert summary.library == "sklearn"
    assert summary.model == "RandomForestRegressor"
    assert summary.num_features == 4
    assert summary.split.test_size == summary.predictions.row_count
    assert summary.metrics["rmse"] >= 0.0
    assert summary.train_metrics["r2"] <= 1.0
    assert summary.feature_importances
    assert summary.export_path is not None
    assert Path(summary.export_path).exists()


def test_tune_tree_model_on_dataset_returns_best_training_summary(tmp_path) -> None:
    summary = tune_tree_model_on_dataset(
        _regression_dataset(),
        target_column="target",
        library="sklearn",
        search_space={"n_estimators": [8, 16], "max_depth": [2, 4]},
        search_kind="grid",
        cv=2,
        test_fraction=0.25,
        random_state=3,
        export_dir=tmp_path / "best_model",
    )

    assert summary.library == "sklearn"
    assert summary.candidate_count == 4
    assert summary.results.row_count == 4
    assert summary.best_params["n_estimators"] in {8, 16}
    assert summary.training.export_path is not None
    assert Path(summary.training.export_path).exists()


def test_missing_optional_backend_reports_install_extra_when_unavailable(monkeypatch) -> None:
    from spectral_packet_engine import tree_models as tree_models_module

    original_module_available = tree_models_module._module_available

    def _fake_module_available(module_name: str) -> bool:
        if module_name == "xgboost":
            return False
        return original_module_available(module_name)

    monkeypatch.setattr(tree_models_module, "_module_available", _fake_module_available)
    with pytest.raises(ModuleNotFoundError, match="ml-xgboost"):
        create_tree_estimator("xgboost", task="regression")
