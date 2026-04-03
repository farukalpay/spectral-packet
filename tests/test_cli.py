from __future__ import annotations

import json
import numpy as np
import pytest

from spectral_packet_engine.cli import build_parser, main
from spectral_packet_engine.table_io import ProfileTable, save_profile_table_csv
from spectral_packet_engine.version import __version__


def _make_csv(tmp_path) -> str:
    grid = np.linspace(0.0, 1.0, 48)
    times = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    profiles = []
    for center in (0.25, 0.4, 0.55):
        profile = np.exp(-((grid - center) ** 2) / (2 * 0.07**2))
        profile = profile / np.trapezoid(profile, grid)
        profiles.append(profile)
    path = tmp_path / "profiles.csv"
    save_profile_table_csv(
        ProfileTable(position_grid=grid, sample_times=times, profiles=np.asarray(profiles, dtype=np.float64)),
        path,
    )
    return str(path)


def _make_shifted_csv(tmp_path) -> str:
    grid = np.linspace(0.0, 1.0, 48)
    times = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    profiles = []
    for center in (0.28, 0.43, 0.58):
        profile = np.exp(-((grid - center) ** 2) / (2 * 0.07**2))
        profile = profile / np.trapezoid(profile, grid)
        profiles.append(profile)
    path = tmp_path / "profiles_shifted.csv"
    save_profile_table_csv(
        ProfileTable(position_grid=grid, sample_times=times, profiles=np.asarray(profiles, dtype=np.float64)),
        path,
    )
    return str(path)


def test_cli_env_emits_json(capsys) -> None:
    exit_code = main(["env", "--device", "cpu"])
    captured = capsys.readouterr().out
    payload = json.loads(captured)

    assert exit_code == 0
    assert payload["torch_runtime"]["backend"] == "cpu"
    assert "tensorflow_available" in payload


def test_cli_help_and_version_surface(capsys) -> None:
    help_text = build_parser().format_help()

    assert "--version" in help_text
    assert "Start here:" in help_text
    assert "ml-backends" in help_text

    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    captured = capsys.readouterr().out

    assert excinfo.value.code == 0
    assert f"spectral-packet-engine {__version__}" in captured


def test_cli_validate_install_and_inspect_table(tmp_path, capsys) -> None:
    csv_path = _make_csv(tmp_path)

    exit_code = main(["validate-install", "--device", "cpu"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["core_ready"] is True

    exit_code = main(["inspect-table", csv_path, "--device", "cpu"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["num_samples"] == 3
    assert payload["num_positions"] == 48

    exit_code = main(["tabular-inspect", csv_path])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["row_count"] == 3
    assert payload["column_count"] == 49

    exit_code = main(["analyze-table", csv_path, "--modes", "12", "--device", "cpu"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["num_modes"] == 12
    assert "spectral_summary" in payload


def test_cli_database_commands(tmp_path, capsys) -> None:
    csv_path = _make_csv(tmp_path)
    database_path = tmp_path / "profiles.sqlite"
    query_dir = tmp_path / "db_query"

    exit_code = main(["db-bootstrap", str(database_path)])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["capability"]["local_bootstrap_supported"] is True

    exit_code = main(
        [
            "db-write-table",
            str(database_path),
            "profiles",
            csv_path,
            "--if-exists",
            "replace",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["table_name"] == "profiles"

    exit_code = main(["db-inspect", str(database_path)])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "profiles" in payload["tables"]

    exit_code = main(["db-describe-table", str(database_path), "profiles"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["row_count"] == 3

    exit_code = main(
        [
            "db-query",
            str(database_path),
            'SELECT time, "0", "0.0212765957447", "0.0425531914894" FROM "profiles" ORDER BY time',
            "--output-dir",
            str(query_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["redacted_url"].endswith("profiles.sqlite")
    assert payload["table"]["row_count"] == 3
    assert (query_dir / "query_result.csv").exists()

    exit_code = main(
        [
            "db-materialize-query",
            str(database_path),
            "profiles_copy",
            'SELECT * FROM "profiles"',
            "--replace",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["table_name"] == "profiles_copy"

    exit_code = main(
        [
            "sql-analyze-table",
            str(database_path),
            'SELECT * FROM "profiles" ORDER BY time',
            "--modes",
            "6",
            "--device",
            "cpu",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["num_modes"] == 6


def test_cli_ml_commands(tmp_path, capsys) -> None:
    csv_path = _make_csv(tmp_path)
    output_dir = tmp_path / "ml_artifacts"

    exit_code = main(["ml-backends", "--device", "cpu"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "torch" in payload["backends"]

    exit_code = main(
        [
            "ml-evaluate-table",
            csv_path,
            "--backend",
            "torch",
            "--modes",
            "8",
            "--epochs",
            "6",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["backend"] == "torch"
    assert (output_dir / "ml_evaluation.json").exists()
    assert (output_dir / "ml_reconstruction.csv").exists()


def test_cli_compress_csv_writes_artifacts(tmp_path, capsys) -> None:
    csv_path = _make_csv(tmp_path)
    output_dir = tmp_path / "artifacts"

    exit_code = main(
        [
            "compress-csv",
            csv_path,
            "--modes",
            "16",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["num_modes"] == 16
    assert (output_dir / "compression_summary.json").exists()
    assert (output_dir / "reconstruction.csv").exists()
    assert (output_dir / "coefficients.csv").exists()
    assert (output_dir / "artifacts.json").exists()
    artifact_index = json.loads((output_dir / "artifacts.json").read_text(encoding="utf-8"))
    assert "artifacts.json" in artifact_index["files"]


def test_cli_compression_sweep_and_packet_sweep(tmp_path, capsys) -> None:
    csv_path = _make_csv(tmp_path)
    sweep_dir = tmp_path / "sweep"
    packet_dir = tmp_path / "packet_sweep"

    exit_code = main(
        [
            "compression-sweep",
            csv_path,
            "--mode-counts",
            "4",
            "8",
            "16",
            "--device",
            "cpu",
            "--output-dir",
            str(sweep_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert (sweep_dir / "compression_sweep.json").exists()
    assert (sweep_dir / "compression_sweep.csv").exists()
    assert (sweep_dir / "artifacts.json").exists()
    assert payload["mode_counts"][-1] == 16.0

    exit_code = main(
        [
            "packet-sweep",
            "--centers",
            "0.25",
            "0.35",
            "--widths",
            "0.07",
            "0.08",
            "--wavenumbers",
            "22.0",
            "24.0",
            "--device",
            "cpu",
            "--output-dir",
            str(packet_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert len(payload["items"]) == 2
    assert (packet_dir / "packet_sweep.json").exists()
    assert (packet_dir / "packet_sweep.csv").exists()
    assert (packet_dir / "artifacts.json").exists()


def test_cli_compare_tables_writes_artifacts(tmp_path, capsys) -> None:
    reference_path = _make_csv(tmp_path)
    candidate_path = _make_shifted_csv(tmp_path)
    output_dir = tmp_path / "comparison"

    exit_code = main(
        [
            "compare-tables",
            reference_path,
            candidate_path,
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["comparison"]["mean_relative_l2_error"] > 0.0
    assert (output_dir / "table_comparison.json").exists()
    assert (output_dir / "residual_profiles.csv").exists()
    assert (output_dir / "sample_metrics.csv").exists()
    assert (output_dir / "artifacts.json").exists()
