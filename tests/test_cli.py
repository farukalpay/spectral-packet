from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
import pytest
import subprocess
import sys

from spectral_packet_engine.cli import build_parser, main
from spectral_packet_engine.table_io import ProfileTable, save_profile_table_csv
from spectral_packet_engine.tabular import TabularDataset, save_tabular_dataset
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


def _make_feature_csv(tmp_path) -> str:
    rows = []
    for index in range(16):
        time = float(index) / 15.0
        mode_1 = 0.2 + 0.03 * index
        mode_2 = 1.0 - 0.02 * index
        mean_position = 0.25 + 0.01 * index
        rows.append(
            {
                "time": time,
                "mode_1": mode_1,
                "mode_2": mode_2,
                "mean_position": mean_position,
                "target": 1.1 * mode_1 - 0.4 * mode_2 + 0.3 * mean_position + 0.2 * time,
            }
        )
    path = tmp_path / "features.csv"
    save_tabular_dataset(TabularDataset.from_rows(rows), path)
    return str(path)


def _make_simulated_packet_csv(tmp_path) -> str:
    from spectral_packet_engine import simulate_gaussian_packet

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
    path = tmp_path / "simulated_profiles.csv"
    save_profile_table_csv(
        ProfileTable(
            position_grid=forward.grid.detach().cpu().numpy(),
            sample_times=forward.times.detach().cpu().numpy(),
            profiles=forward.densities.detach().cpu().numpy(),
        ),
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
    assert payload["mcp_runtime"]["transport"] == "stdio"


def test_cli_product_report_emits_shared_identity_json(capsys) -> None:
    exit_code = main(["inspect-product"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["product_name"] == "Spectral Packet Engine"
    assert payload["hero_workflow"]["workflow_id"] == "profile-table-report"
    assert len(payload["killer_workflows"]) == 3


def test_cli_guide_workflow_emits_opinionated_defaults(capsys) -> None:
    exit_code = main(["guide-workflow", "--input-kind", "profile-table-sql", "--goal", "feature-model"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["surface"] == "cli"
    assert payload["goal"] == "feature-model"
    assert payload["primary_workflow"]["workflow_id"] == "profile-table-report-from-sql"
    assert payload["defaults"]["sort_by_time"] is True
    assert payload["plan_steps"][1]["workflow_id"] == "profile-table-report-from-sql"
    assert any(step["workflow_id"] == "export-feature-table-from-sql" for step in payload["plan_steps"] if "workflow_id" in step)
    assert any(step["workflow_id"] == "tree-model-train" for step in payload["plan_steps"] if "workflow_id" in step)


def test_cli_help_and_version_surface(capsys) -> None:
    help_text = build_parser().format_help()

    assert "--version" in help_text
    assert "Start here:" in help_text
    assert "inspect-product" in help_text
    assert "guide-workflow" in help_text
    assert "inspect-environment" in help_text
    assert "ml-backends" in help_text
    assert "tree-backends" in help_text
    assert "export-features" in help_text
    assert "tree-train" in help_text
    assert "tree-tune" in help_text
    assert "probe-mcp" in help_text
    assert "plan-mcp-tunnel" in help_text
    assert "install-mcp-service" in help_text
    assert "infer-potential-spectrum" in help_text
    assert "analyze-separable-spectrum" in help_text
    assert "design-transition" in help_text
    assert "optimize-packet-control" in help_text
    assert "transport-workflow" in help_text
    assert "profile-inference-workflow" in help_text
    assert "Then run the local release gate:" in help_text
    assert "profile-report" in help_text

    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    captured = capsys.readouterr().out

    assert excinfo.value.code == 0
    assert f"spectral-packet-engine {__version__}" in captured

    with pytest.raises(SystemExit) as mcp_help_excinfo:
        main(["serve-mcp", "--help"])
    mcp_help = capsys.readouterr().out

    assert mcp_help_excinfo.value.code == 0
    assert "--transport" in mcp_help
    assert "--host" in mcp_help
    assert "--port" in mcp_help
    assert "--streamable-http-path" in mcp_help
    assert "--max-concurrent-tasks" in mcp_help
    assert "--slot-timeout-seconds" in mcp_help
    assert "--log-level" in mcp_help
    assert "--log-file" in mcp_help
    assert "--scratch-dir" in mcp_help
    assert "--allowed-host" in mcp_help
    assert "--allowed-origin" in mcp_help
    assert "--allow-unsafe-python" in mcp_help

    with pytest.raises(SystemExit) as probe_help_excinfo:
        main(["probe-mcp", "--help"])
    probe_help = capsys.readouterr().out

    assert probe_help_excinfo.value.code == 0
    assert "--profile" in probe_help

    with pytest.raises(SystemExit) as guide_help_excinfo:
        main(["guide-workflow", "--help"])
    guide_help = capsys.readouterr().out

    assert guide_help_excinfo.value.code == 0
    assert "--goal" in guide_help


def test_cli_generate_mcp_config_and_install_service_dry_run(tmp_path, capsys) -> None:
    exit_code = main(["generate-mcp-config", "--transport", "ssh", "--host", "example-host", "--remote-cwd", "/srv/spe"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["mcpServers"]["spectral-packet-engine"]["command"] == "ssh"
    assert payload["mcpServers"]["spectral-packet-engine"]["args"][0] == "example-host"

    exit_code = main(
        [
            "plan-mcp-tunnel",
            "--host",
            "example-host",
            "--local-port",
            "9876",
            "--remote-port",
            "8765",
            "--streamable-http-path",
            "/mcp",
        ]
    )
    tunnel_payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert tunnel_payload["tunnel_command"][:3] == ["ssh", "-N", "-L"]
    assert tunnel_payload["local_endpoint_url"] == "http://127.0.0.1:9876/mcp"

    exit_code = main(
        [
            "install-mcp-service",
            "--working-directory",
            str(tmp_path),
            "--python-executable",
            "python3",
            "--source-checkout",
            "--allowed-host",
            "lightcap.ai",
            "--allowed-host",
            "lightcap.ai:*",
            "--allowed-origin",
            "https://lightcap.ai",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["dry_run"] is True
    assert payload["plan"]["restart_policy"] == "always"
    assert "--allowed-host" in payload["manifest"]
    assert "--allowed-origin" in payload["manifest"]
    assert "--allow-unsafe-python" not in payload["manifest"]


def test_cli_validate_install_and_inspect_table(tmp_path, capsys) -> None:
    csv_path = _make_csv(tmp_path)

    exit_code = main(["validate-install", "--device", "cpu"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["core_ready"] is True

    exit_code = main(["inspect-profile-table", csv_path, "--device", "cpu"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["num_samples"] == 3
    assert payload["num_positions"] == 48

    exit_code = main(["inspect-tabular-dataset", csv_path])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["row_count"] == 3
    assert payload["column_count"] == 49

    exit_code = main(["analyze-profile-table", csv_path, "--modes", "12", "--device", "cpu"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["num_modes"] == 12
    assert "spectral_summary" in payload


def test_cli_optimize_packet_control_accepts_interval_probability_objective(capsys) -> None:
    exit_code = main(
        [
            "optimize-packet-control",
            "--objective",
            "interval_probability",
            "--target-value",
            "0.35",
            "--final-time",
            "0.004",
            "--interval",
            "0.5",
            "1.0",
            "--modes",
            "48",
            "--quadrature",
            "1024",
            "--grid",
            "64",
            "--steps",
            "20",
            "--learning-rate",
            "0.03",
            "--device",
            "cpu",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["objective"] == "interval_probability"
    assert payload["final_interval_probability"] is not None


def test_cli_inverse_physics_and_reduced_model_commands(tmp_path, capsys) -> None:
    from spectral_packet_engine import InfiniteWell1D, harmonic_potential
    from spectral_packet_engine.eigensolver import solve_eigenproblem

    import torch

    domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device="cpu")
    target = solve_eigenproblem(
        lambda x: harmonic_potential(x, omega=8.0, domain=domain),
        domain,
        num_points=128,
        num_states=3,
    ).eigenvalues.detach().cpu().tolist()

    exit_code = main(
        [
            "infer-potential-spectrum",
            *[str(value) for value in target],
            "--family",
            "harmonic",
            "--family",
            "double-well",
            "--initial-guesses",
            json.dumps(
                {
                    "harmonic": {"omega": 5.0},
                    "double-well": {"a_param": 1.5, "b_param": 1.0},
                }
            ),
            "--steps",
            "120",
            "--learning-rate",
            "0.04",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "spectroscopy"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["family_inference"]["best_family"] == "harmonic"
    assert (tmp_path / "spectroscopy" / "family_inference" / "candidate_ranking.csv").exists()

    exit_code = main(
        [
            "analyze-separable-spectrum",
            "--family-x",
            "harmonic",
            "--params-x",
            json.dumps({"omega": 8.0}),
            "--family-y",
            "harmonic",
            "--params-y",
            json.dumps({"omega": 6.0}),
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "separable"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["family_x"] == "harmonic"
    assert (tmp_path / "separable" / "combined_spectrum.csv").exists()


def test_cli_differentiable_and_vertical_commands(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "design-transition",
            "--family",
            "harmonic",
            "--target-transition",
            "12.0",
            "--initial-guess",
            json.dumps({"omega": 5.0}),
            "--steps",
            "80",
            "--learning-rate",
            "0.03",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "design"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["family"] == "harmonic"
    assert (tmp_path / "design" / "transition_design_spectrum.csv").exists()

    csv_path = _make_simulated_packet_csv(tmp_path)
    exit_code = main(
        [
            "profile-inference-workflow",
            csv_path,
            "--analyze-modes",
            "8",
            "--compress-modes",
            "4",
            "--inverse-modes",
            "48",
            "--feature-modes",
            "6",
            "--quadrature",
            "1024",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "profile_vertical"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["report"]["overview"]["analyze_num_modes"] == 8
    assert (tmp_path / "profile_vertical" / "report" / "artifacts.json").exists()
    assert (tmp_path / "profile_vertical" / "inverse" / "artifacts.json").exists()
    assert (tmp_path / "profile_vertical" / "features" / "artifacts.json").exists()


def test_cli_profile_report_and_artifact_inspection(tmp_path, capsys) -> None:
    csv_path = _make_csv(tmp_path)
    output_dir = tmp_path / "profile_report"

    exit_code = main(
        [
            "profile-report",
            csv_path,
            "--analyze-modes",
            "12",
            "--compress-modes",
            "6",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["overview"]["analyze_num_modes"] == 12
    assert payload["overview"]["compress_num_modes"] == 6
    assert (output_dir / "profile_table_report.json").exists()
    assert (output_dir / "profile_table_summary.json").exists()
    assert (output_dir / "analysis" / "spectral_analysis.json").exists()
    assert (output_dir / "compression" / "compression_summary.json").exists()

    exit_code = main(["inspect-artifacts", str(output_dir)])
    inspection = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert inspection["complete"] is True
    assert inspection["metadata"]["workflow"] == "profile-report"
    assert "analysis/artifacts.json" in inspection["files"]


def test_cli_release_gate_reports_current_environment(capsys) -> None:
    exit_code = main(["release-gate", "--device", "cpu"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["release_ready_current_environment"] is True
    assert "python-core" in payload["validated_surfaces"]
    assert "sql-backend" in payload["validated_surfaces"]


def test_cli_database_commands(tmp_path, capsys) -> None:
    csv_path = _make_csv(tmp_path)
    database_path = tmp_path / "profiles.sqlite"
    query_dir = tmp_path / "db_query"
    compression_dir = tmp_path / "sql_compression"
    feature_dir = tmp_path / "sql_features"

    exit_code = main(["bootstrap-database", str(database_path)])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["capability"]["local_bootstrap_supported"] is True

    exit_code = main(
        [
            "write-database-table",
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

    exit_code = main(["inspect-database", str(database_path)])
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

    execute_statement = main(
        [
            "db-exec",
            str(database_path),
            'CREATE TABLE IF NOT EXISTS "scratch_metrics" (time REAL, value REAL)',
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert execute_statement == 0
    assert payload["mode"] == "statement"
    assert payload["statement_count"] == 1

    execute_script = main(
        [
            "db-script",
            str(database_path),
            """
            INSERT INTO "scratch_metrics" (time, value) VALUES (0.0, 1.0);
            INSERT INTO "scratch_metrics" (time, value) VALUES (0.1, 2.0);
            """,
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert execute_script == 0
    assert payload["mode"] == "script"
    assert payload["statement_count"] == 2

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

    exit_code = main(
        [
            "sql-compress-table",
            str(database_path),
            'SELECT time, "0", "0.0212765957447", "0.0425531914894" FROM "profiles"',
            "--time-column",
            "time",
            "--position-column",
            "0",
            "--position-column",
            "0.0212765957447",
            "--position-column",
            "0.0425531914894",
            "--sort-by-time",
            "--modes",
            "3",
            "--device",
            "cpu",
            "--output-dir",
            str(compression_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["num_modes"] == 3
    artifact_index = json.loads((compression_dir / "artifacts.json").read_text(encoding="utf-8"))
    assert artifact_index["metadata"]["workflow"] == "sql-compress-table"
    assert artifact_index["metadata"]["input"]["query"].startswith("SELECT time")
    assert artifact_index["metadata"]["input"]["profile_table"]["sort_by_time"] is True

    exit_code = main(
        [
            "sql-export-features",
            str(database_path),
            'SELECT time, "0", "0.0212765957447", "0.0425531914894" FROM "profiles"',
            "--time-column",
            "time",
            "--position-column",
            "0",
            "--position-column",
            "0.0212765957447",
            "--position-column",
            "0.0425531914894",
            "--sort-by-time",
            "--modes",
            "3",
            "--device",
            "cpu",
            "--output-dir",
            str(feature_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["source_kind"] == "database-query"
    assert (feature_dir / "features.csv").exists()
    feature_index = json.loads((feature_dir / "artifacts.json").read_text(encoding="utf-8"))
    assert feature_index["metadata"]["workflow"] == "export-features"
    assert feature_index["metadata"]["input"]["profile_table"]["sort_by_time"] is True

    profile_report_dir = tmp_path / "sql_profile_report"
    exit_code = main(
        [
            "sql-profile-report",
            str(database_path),
            'SELECT time, "0", "0.0212765957447", "0.0425531914894" FROM "profiles"',
            "--time-column",
            "time",
            "--position-column",
            "0",
            "--position-column",
            "0.0212765957447",
            "--position-column",
            "0.0425531914894",
            "--sort-by-time",
            "--analyze-modes",
            "4",
            "--compress-modes",
            "3",
            "--device",
            "cpu",
            "--output-dir",
            str(profile_report_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["overview"]["analyze_num_modes"] == 4
    assert payload["overview"]["compress_num_modes"] == 3
    report_index = json.loads((profile_report_dir / "artifacts.json").read_text(encoding="utf-8"))
    assert report_index["metadata"]["workflow"] == "profile-report"
    assert report_index["metadata"]["input"]["query"].startswith("SELECT time")


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


def test_cli_feature_export_and_tree_commands(tmp_path, capsys) -> None:
    profile_csv_path = _make_csv(tmp_path)
    feature_csv_path = _make_feature_csv(tmp_path)
    exported_features_dir = tmp_path / "feature_export"
    train_output_dir = tmp_path / "tree_train"
    tune_output_dir = tmp_path / "tree_tune"

    exit_code = main(["tree-backends", "--library", "sklearn"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["resolution"]["resolved_library"] == "sklearn"

    exit_code = main(
        [
            "export-features",
            profile_csv_path,
            "--modes",
            "6",
            "--device",
            "cpu",
            "--output-dir",
            str(exported_features_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["num_features"] == 9
    assert (exported_features_dir / "features.csv").exists()
    assert (exported_features_dir / "features_schema.json").exists()

    exit_code = main(
        [
            "tree-train",
            feature_csv_path,
            "--target-column",
            "target",
            "--library",
            "sklearn",
            "--params",
            '{"n_estimators": 16, "max_depth": 4}',
            "--output-dir",
            str(train_output_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["library"] == "sklearn"
    assert (train_output_dir / "tree_training.json").exists()
    assert (train_output_dir / "predictions.csv").exists()
    assert (train_output_dir / "artifacts.json").exists()

    exit_code = main(
        [
            "tree-tune",
            feature_csv_path,
            "--target-column",
            "target",
            "--library",
            "sklearn",
            "--search-kind",
            "grid",
            "--search-space",
            '{"n_estimators": [8, 16], "max_depth": [2, 4]}',
            "--cv",
            "2",
            "--output-dir",
            str(tune_output_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["candidate_count"] == 4
    assert (tune_output_dir / "tree_tuning.json").exists()
    assert (tune_output_dir / "best_model" / "artifacts.json").exists()


def test_python_module_cli_entrypoint_executes_commands(tmp_path) -> None:
    output_dir = tmp_path / "module_features"
    repo_root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str((repo_root / "src").resolve())}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spectral_packet_engine.cli",
            "export-features",
            "examples/data/synthetic_profiles.csv",
            "--modes",
            "4",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["num_modes"] == 4
    assert (output_dir / "features.csv").exists()


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
