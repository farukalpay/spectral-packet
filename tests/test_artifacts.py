from __future__ import annotations

import json
import socket

import numpy as np
import torch

from spectral_packet_engine import ProfileTable, build_profile_table_report
from spectral_packet_engine.artifacts import (
    inspect_artifact_directory,
    to_serializable,
    write_profile_table_report_artifacts,
    write_tabular_artifacts,
)
from spectral_packet_engine.tabular import TabularDataset


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
