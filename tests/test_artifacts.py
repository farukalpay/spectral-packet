from __future__ import annotations

import json
import socket

import numpy as np
import torch

from spectral_packet_engine import (
    ProfileTable,
    TabularDataset,
    build_profile_table_report,
    export_feature_table_from_profile_table,
    save_profile_table_csv,
    train_tree_model,
    tune_tree_model,
)
from spectral_packet_engine.artifacts import (
    inspect_artifact_directory,
    to_serializable,
    write_feature_table_artifacts,
    write_profile_table_report_artifacts,
    write_tabular_artifacts,
    write_tree_training_artifacts,
    write_tree_tuning_artifacts,
)


def test_to_serializable_sanitizes_non_finite_numbers() -> None:
    payload = {
        "tensor": torch.tensor([1.0, float("nan"), float("inf"), -float("inf")], dtype=torch.float64),
        "array": np.asarray([np.nan, np.inf, -np.inf], dtype=np.float64),
        "scalar": float("nan"),
        "complex": complex(float("inf"), -float("inf")),
    }

    serializable = to_serializable(payload)

    assert serializable["tensor"] == [1.0, "nan", "inf", "-inf"]
    assert serializable["array"] == ["nan", "inf", "-inf"]
    assert serializable["scalar"] == "nan"
    assert serializable["complex"] == {"real": "inf", "imag": "-inf"}


def test_artifact_directory_report_marks_complete_and_cleans_stale_runtime_state(tmp_path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    (output_dir / ".spectral-packet-engine-tmp-stale.json").write_text("stale", encoding="utf-8")
    (output_dir / ".spectral-packet-engine.lock").write_text(
        json.dumps(
            {
                "lock_id": "stale",
                "hostname": socket.gethostname(),
                "pid": 999999,
                "started_at_utc": "2000-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    dataset = TabularDataset.from_rows([{"time": 0.0, "value": 1.0}])
    write_tabular_artifacts(output_dir, dataset, metadata={"workflow": "unit-test"})
    report = inspect_artifact_directory(output_dir)

    assert report.complete is True
    assert report.lock_active is False
    assert report.stale_temporary_files == ()
    assert "artifacts.json" in report.files
    assert report.metadata["workflow"] == "unit-test"


def test_profile_table_report_artifacts_bundle_has_expected_layout(tmp_path) -> None:
    grid = np.linspace(0.0, 1.0, 16)
    times = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    profiles = np.asarray(
        [
            np.exp(-((grid - 0.25) ** 2) / (2 * 0.06**2)),
            np.exp(-((grid - 0.40) ** 2) / (2 * 0.06**2)),
            np.exp(-((grid - 0.55) ** 2) / (2 * 0.06**2)),
        ],
        dtype=np.float64,
    )
    table = ProfileTable(position_grid=grid, sample_times=times, profiles=profiles, source="artifact-test")
    report = build_profile_table_report(table, analyze_num_modes=8, compress_num_modes=4, device="cpu")

    output_dir = tmp_path / "profile_report"
    write_profile_table_report_artifacts(output_dir, report, metadata={"workflow": "profile-report-test"})
    directory = inspect_artifact_directory(output_dir)

    assert directory.complete is True
    assert "profile_table_report.json" in directory.files
    assert "profile_table_summary.json" in directory.files
    assert "analysis/artifacts.json" in directory.files
    assert "compression/artifacts.json" in directory.files
    assert directory.metadata["workflow"] == "profile-report-test"
    assert directory.metadata["analyze_num_modes"] == 8
    assert directory.metadata["compress_num_modes"] == 4


def test_profile_table_report_can_write_its_own_artifacts(tmp_path) -> None:
    grid = np.linspace(0.0, 1.0, 8)
    times = np.asarray([0.0, 0.1], dtype=np.float64)
    profiles = np.asarray(
        [
            [0.1, 0.2, 0.5, 0.9, 0.5, 0.2, 0.1, 0.05],
            [0.05, 0.1, 0.35, 0.8, 0.55, 0.25, 0.1, 0.05],
        ],
        dtype=np.float64,
    )
    report = build_profile_table_report(
        ProfileTable(position_grid=grid, sample_times=times, profiles=profiles),
        analyze_num_modes=4,
        compress_num_modes=3,
        device="cpu",
    )

    directory = report.write_artifacts(
        tmp_path / "report_bundle",
        metadata={"workflow": "report-method-test"},
    )

    assert directory.complete is True
    assert directory.metadata["workflow"] == "report-method-test"


def test_feature_table_artifacts_bundle_has_expected_layout(tmp_path) -> None:
    grid = np.linspace(0.0, 1.0, 16)
    times = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    profiles = np.asarray(
        [
            np.exp(-((grid - 0.25) ** 2) / (2 * 0.06**2)),
            np.exp(-((grid - 0.40) ** 2) / (2 * 0.06**2)),
            np.exp(-((grid - 0.55) ** 2) / (2 * 0.06**2)),
        ],
        dtype=np.float64,
    )
    table_path = tmp_path / "feature_profiles.csv"
    save_profile_table_csv(
        ProfileTable(position_grid=grid, sample_times=times, profiles=profiles, source="feature-artifact-test"),
        table_path,
    )
    summary = export_feature_table_from_profile_table(table_path, num_modes=4, device="cpu")

    output_dir = tmp_path / "feature_bundle"
    write_feature_table_artifacts(output_dir, summary)
    directory = inspect_artifact_directory(output_dir)
    schema_payload = json.loads((output_dir / "features_schema.json").read_text(encoding="utf-8"))

    assert directory.complete is True
    assert "features.csv" in directory.files
    assert "feature_table_export.json" in directory.files
    assert "features_schema.json" in directory.files
    assert directory.metadata["workflow"] == "export-features"
    assert directory.metadata["input_kind"] == "file"
    assert directory.metadata["num_modes"] == 4
    assert directory.metadata["normalize_each_profile"] is False
    assert "numpy" in directory.metadata["library_versions"]
    assert "torch" in directory.metadata["library_versions"]
    assert schema_payload["column_order"] == [
        "time",
        "mode_1",
        "mode_2",
        "mode_3",
        "mode_4",
        "mean_position",
        "width",
        "mass",
    ]
    assert schema_payload["columns"][0]["dtype"] == "float64"
    assert schema_payload["columns"][0]["semantic_meaning"] == "Sample time for the profile row."
    assert schema_payload["columns"][1]["semantic_meaning"] == "Spectral modal coefficient for mode 1."
    assert schema_payload["columns"][-1]["semantic_meaning"] == "Integrated profile mass over the bounded 1D domain."
    assert schema_payload["ordering"]["time"]["policy"] == "preserve-profile-table-sample-order"
    assert "numpy" in schema_payload["library_versions"]


def test_tree_training_and_tuning_artifacts_follow_shared_bundle_contract(tmp_path) -> None:
    dataset = TabularDataset.from_rows(
        [
            {
                "time": float(index) / 11.0,
                "mode_1": 0.2 + 0.03 * index,
                "mode_2": 1.2 - 0.04 * index,
                "mean_position": 0.25 + 0.01 * index,
                "target": 0.8 + 0.06 * index,
            }
            for index in range(12)
        ]
    )
    training = train_tree_model(
        dataset,
        target_column="target",
        library="sklearn",
        params={"n_estimators": 16, "max_depth": 4},
        export_dir=tmp_path / "tree_model_export",
    )
    training_dir = tmp_path / "tree_train_bundle"
    write_tree_training_artifacts(training_dir, training)
    training_report = inspect_artifact_directory(training_dir)

    assert training_report.complete is True
    assert "tree_training.json" in training_report.files
    assert "predictions.csv" in training_report.files
    assert "feature_importance.csv" in training_report.files
    assert any(path.endswith(".pkl") for path in training_report.files)
    assert training_report.metadata["workflow"] == "tree-train"
    assert training_report.metadata["library"] == "sklearn"
    assert training_report.metadata["library_versions"]["sklearn"] is not None

    tuning = tune_tree_model(
        dataset,
        target_column="target",
        library="sklearn",
        search_space={"n_estimators": [8, 16], "max_depth": [2, 4]},
        search_kind="grid",
        cv=2,
        export_dir=tmp_path / "tree_best_model_export",
    )
    tuning_dir = tmp_path / "tree_tune_bundle"
    write_tree_tuning_artifacts(tuning_dir, tuning)
    tuning_report = inspect_artifact_directory(tuning_dir)

    assert tuning_report.complete is True
    assert "tree_tuning.json" in tuning_report.files
    assert "tuning_results.csv" in tuning_report.files
    assert "best_model/artifacts.json" in tuning_report.files
    assert tuning_report.metadata["workflow"] == "tree-tune"
    assert tuning_report.metadata["library"] == "sklearn"
    assert tuning_report.metadata["library_versions"]["sklearn"] is not None
