from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from spectral_packet_engine import (
    ProfileTable,
    api_is_available,
    compress_profile_table,
    create_api_app,
    create_mcp_server,
    inspect_service_status,
    run_release_gate,
    save_profile_table_csv,
    simulate_gaussian_packet,
    write_compression_artifacts,
)
from spectral_packet_engine.cli import main as cli_main


def _synthetic_profile_table(*, samples: int = 6, positions: int = 24) -> ProfileTable:
    grid = np.linspace(0.0, 1.0, positions, dtype=np.float64)
    times = np.linspace(0.0, 0.5, samples, dtype=np.float64)
    profiles = []
    for center, width in zip(np.linspace(0.22, 0.62, samples), np.linspace(0.05, 0.09, samples)):
        profile = np.exp(-((grid - center) ** 2) / (2 * width**2))
        profile = profile / np.trapezoid(profile, grid)
        profiles.append(profile)
    return ProfileTable(
        position_grid=grid,
        sample_times=times,
        profiles=np.asarray(profiles, dtype=np.float64),
        source="synthetic-release-gate",
    )


def test_release_gate_python_golden_path_writes_artifacts(tmp_path) -> None:
    forward = simulate_gaussian_packet(
        center=0.30,
        width=0.07,
        wavenumber=25.0,
        times=[0.0, 1e-3, 3e-3, 5e-3],
        num_modes=64,
        quadrature_points=1024,
        grid_points=64,
        device="cpu",
    )
    table = ProfileTable(
        position_grid=forward.grid.detach().cpu().numpy(),
        sample_times=forward.times.detach().cpu().numpy(),
        profiles=forward.densities.detach().cpu().numpy(),
        source="python-forward",
    )

    summary = compress_profile_table(table, num_modes=12, device="cpu")
    output_dir = tmp_path / "python_artifacts"
    write_compression_artifacts(output_dir, summary)

    assert float(summary.error_summary.mean_relative_l2_error) < 0.25
    assert (output_dir / "compression_summary.json").exists()
    assert (output_dir / "reconstruction.csv").exists()
    assert (output_dir / "coefficients.csv").exists()
    assert (output_dir / "artifacts.json").exists()


def test_release_gate_cli_sql_backend_aware_golden_path(tmp_path, capsys) -> None:
    table = _synthetic_profile_table(samples=8, positions=20)
    csv_path = tmp_path / "profiles.csv"
    save_profile_table_csv(table, csv_path)

    database_path = tmp_path / "profiles.sqlite"
    output_dir = tmp_path / "sql_ml_eval"

    assert cli_main(["db-bootstrap", str(database_path)]) == 0
    capsys.readouterr()
    assert cli_main(["db-write-table", str(database_path), "profiles", str(csv_path), "--if-exists", "replace"]) == 0
    capsys.readouterr()

    exit_code = cli_main(
        [
            "sql-ml-evaluate-table",
            str(database_path),
            'SELECT * FROM "profiles" ORDER BY time',
            "--backend",
            "torch",
            "--modes",
            "6",
            "--epochs",
            "4",
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
    assert (output_dir / "ml_coefficients.csv").exists()
    assert (output_dir / "ml_predicted_moments.csv").exists()
    assert (output_dir / "artifacts.json").exists()


def test_release_gate_api_golden_path_updates_status_and_writes_artifacts(tmp_path) -> None:
    if not api_is_available():
        pytest.skip("FastAPI stack is not available or not compatible in this environment")

    from fastapi.testclient import TestClient

    client = TestClient(create_api_app())
    table = _synthetic_profile_table(samples=5, positions=18)
    output_dir = tmp_path / "api_compression"

    response = client.post(
        "/profiles/compress",
        json={
            "table": table.to_dict(),
            "num_modes": 8,
            "device": "cpu",
            "output_dir": str(output_dir),
        },
    )
    assert response.status_code == 200
    assert response.json()["num_modes"] == 8
    assert (output_dir / "compression_summary.json").exists()

    status = client.get("/status")
    assert status.status_code == 200
    recent_tasks = status.json()["recent_tasks"]
    assert any(task["workflow_id"] == "compress-profile-table" for task in recent_tasks)
    assert any(task["surface_action"] == "POST /profiles/compress" for task in recent_tasks)


def test_release_gate_mcp_golden_path_and_failure_path(tmp_path) -> None:
    pytest.importorskip("mcp.server.fastmcp")

    table = _synthetic_profile_table(samples=5, positions=18)
    csv_path = tmp_path / "profiles.csv"
    save_profile_table_csv(table, csv_path)
    output_dir = tmp_path / "mcp_compression"

    async def _exercise():
        server = create_mcp_server()
        _, success_payload = await server.call_tool(
            "compress_profile_table",
            {
                "table_path": str(csv_path),
                "num_modes": 8,
                "device": "cpu",
                "output_dir": str(output_dir),
            },
        )
        _, status_payload = await server.call_tool("inspect_service_status", {})
        return server, success_payload, status_payload

    _, success_payload, status_payload = asyncio.run(_exercise())

    assert success_payload["num_modes"] == 8
    assert (output_dir / "compression_summary.json").exists()
    assert any(task["workflow_id"] == "compress-profile-table" for task in status_payload["recent_tasks"])
    assert any(task["surface_action"] == "compress_profile_table" for task in status_payload["recent_tasks"])

    async def _missing_file_failure():
        server = create_mcp_server()
        _, error_payload = await server.call_tool(
            "compress_profile_table",
            {
                "table_path": "/definitely/missing.csv",
                "num_modes": 4,
                "device": "cpu",
            },
        )
        return error_payload

    error_payload = asyncio.run(_missing_file_failure())
    assert error_payload.get("error") is True
    assert "missing.csv" in error_payload.get("error_message", "")


def test_release_gate_status_contract_is_consistent_across_python_api_and_mcp() -> None:
    python_status = inspect_service_status().to_dict()

    if api_is_available():
        from fastapi.testclient import TestClient

        api_status = TestClient(create_api_app()).get("/status").json()
        assert set(api_status.keys()) == set(python_status.keys())

    pytest.importorskip("mcp.server.fastmcp")

    async def _status():
        server = create_mcp_server()
        _, payload = await server.call_tool("inspect_service_status", {})
        return payload

    mcp_status = asyncio.run(_status())
    assert "related_tools" in mcp_status
    assert set(mcp_status.keys()) - {"related_tools"} == set(python_status.keys())


def test_release_gate_report_captures_local_validation_state() -> None:
    report = run_release_gate(device="cpu")

    assert report.release_ready_current_environment is True
    assert "python-core" in report.validated_surfaces
    assert "sql-backend" in report.validated_surfaces
    assert "tree-model" in report.validated_surfaces
    assert report.blocked_surfaces == ()
    by_surface = {check.surface: check for check in report.checks}
    assert by_surface["python-core"].status == "passed"
    assert by_surface["sql-backend"].status == "passed"
    assert by_surface["tree-model"].status == "passed"
    assert "release gate validates the current environment in-process" in report.notes[0].lower()
