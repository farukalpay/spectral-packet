from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_mcp_module_entrypoint_help_is_available_from_source_checkout() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "spectral_packet_engine.mcp", "--help"],
        cwd=_repo_root(),
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--max-concurrent-tasks" in completed.stdout
    assert "--allow-unsafe-python" in completed.stdout


def test_mcp_client_config_and_service_plan_are_explicit() -> None:
    from spectral_packet_engine.mcp_deployment import (
        build_local_mcp_client_configuration,
        build_mcp_service_install_plan,
    )

    root = _repo_root()
    config = build_local_mcp_client_configuration(
        working_directory=root,
        python_executable="python3",
        source_checkout=True,
    ).to_dict()
    assert config["mcpServers"]["spectral-packet-engine"]["command"] == "python3"
    assert config["mcpServers"]["spectral-packet-engine"]["args"][:3] == ["-m", "spectral_packet_engine", "serve-mcp"]
    assert config["mcpServers"]["spectral-packet-engine"]["env"]["PYTHONPATH"] == "src"

    linux_plan = build_mcp_service_install_plan(
        working_directory=root,
        python_executable="python3",
        source_checkout=True,
        platform_name="linux",
        scratch_directory=root / "tmp_scratch",
    )
    assert linux_plan.restart_policy == "always"
    assert "Restart=always" in linux_plan.manifest
    assert 'Environment="PYTHONPATH=src"' in linux_plan.manifest
    assert "--transport streamable-http" in linux_plan.manifest
    assert "--scratch-dir" in linux_plan.manifest
    assert linux_plan.endpoint_url.startswith("http://127.0.0.1:")

    macos_plan = build_mcp_service_install_plan(
        working_directory=root,
        python_executable="python3",
        source_checkout=True,
        platform_name="macos",
    )
    assert "<key>KeepAlive</key>" in macos_plan.manifest
    assert "<key>PYTHONPATH</key>" in macos_plan.manifest
    assert macos_plan.enable_commands[0][1] == "bootout"
    assert macos_plan.enable_commands[1][1] == "bootstrap"


def test_execute_python_is_disabled_by_default_when_available() -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from spectral_packet_engine import create_mcp_server

    async def _call():
        server = create_mcp_server()
        _, payload = await server.call_tool("execute_python", {"code": "result = 1"})
        return payload

    payload = __import__("asyncio").run(_call())
    assert payload["error"] is True
    assert payload["error_type"] == "PermissionError"


def test_managed_scratch_tools_reject_path_traversal_when_available() -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from spectral_packet_engine import create_mcp_server

    async def _call():
        server = create_mcp_server()
        _, create_payload = await server.call_tool("create_scratch_database", {"name": "../escape"})
        _, generate_payload = await server.call_tool(
            "generate_synthetic_profiles",
            {"output_name": "../escape_profiles", "num_profiles": 2, "grid_points": 8},
        )
        return create_payload, generate_payload

    create_payload, generate_payload = __import__("asyncio").run(_call())
    assert create_payload["error"] is True
    assert create_payload["error_type"] == "ValueError"
    assert generate_payload["error"] is True
    assert generate_payload["error_type"] == "ValueError"


def test_database_script_execution_supports_sqlite_bootstrap_when_available(tmp_path) -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from spectral_packet_engine import create_mcp_server

    database_path = tmp_path / "commodities.sqlite"

    async def _call():
        server = create_mcp_server()
        _, script_payload = await server.call_tool(
            "execute_database_script",
            {
                "database": str(database_path),
                "script": """
                CREATE TABLE IF NOT EXISTS war_commodities (
                    month_idx INTEGER PRIMARY KEY,
                    month_label TEXT,
                    brent_usd REAL
                );
                INSERT OR REPLACE INTO war_commodities VALUES
                    (1, '2026-01', 65.0),
                    (2, '2026-02', 69.4);
                """,
            },
        )
        _, query_payload = await server.call_tool(
            "query_database",
            {
                "database": str(database_path),
                "query": 'SELECT month_idx, month_label, brent_usd FROM "war_commodities" ORDER BY month_idx',
            },
        )
        return script_payload, query_payload

    script_payload, query_payload = __import__("asyncio").run(_call())
    assert script_payload["mode"] == "script"
    assert script_payload["statement_count"] == 2
    assert query_payload["table"]["row_count"] == 2


def test_execute_python_can_be_enabled_explicitly_when_available() -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from spectral_packet_engine import MCPServerConfig, create_mcp_server

    async def _call():
        server = create_mcp_server(MCPServerConfig(allow_unsafe_python=True))
        _, payload = await server.call_tool("execute_python", {"code": "result = 7"})
        return payload

    payload = __import__("asyncio").run(_call())
    assert payload["result"] == 7


def test_synthetic_profile_generator_round_trips(tmp_path) -> None:
    import numpy as np

    from spectral_packet_engine.synthetic_profiles import generate_synthetic_profile_table
    from spectral_packet_engine.table_io import load_profile_table, save_profile_table_csv

    table = generate_synthetic_profile_table(num_profiles=4, grid_points=16, device="cpu")
    assert np.isfinite(table.profiles).all()

    output_path = tmp_path / "test_synthetic_roundtrip.csv"
    save_profile_table_csv(table, output_path)
    loaded = load_profile_table(output_path)
    assert loaded.profiles.shape == (4, 16)


def test_probe_suite_runs_against_local_stdio_server_when_available(tmp_path) -> None:
    pytest.importorskip("mcp.client.stdio")

    from spectral_packet_engine.mcp_probe import (
        build_local_probe_server_spec,
        run_mcp_probe_suite,
        write_mcp_probe_artifacts,
    )

    report = run_mcp_probe_suite(
        build_local_probe_server_spec(
            working_directory=_repo_root(),
            python_executable="python3",
            source_checkout=True,
            log_file=tmp_path / "server.log",
        ),
        expect_unsafe_python_enabled=False,
    )
    assert report.summary["failed_count"] == 0
    assert report.summary["probe_count"] >= 6
    assert report.profile == "smoke"
    assert any(probe.probe_id == "EXEC-001" and probe.passed for probe in report.probes)
    assert any(probe.probe_id == "DATA-001" and probe.passed for probe in report.probes)
    assert any(probe.probe_id == "SELF-001" and probe.passed for probe in report.probes)

    output_dir = write_mcp_probe_artifacts(report, tmp_path / "probe_artifacts")
    assert (output_dir / "mcp_probe_report.json").exists()
    assert (output_dir / "mcp_tool_calls.jsonl").exists()
    assert (output_dir / "artifacts.json").exists()


def test_server_info_reports_bind_facts_when_available() -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from spectral_packet_engine import create_mcp_server

    async def _call():
        server = create_mcp_server()
        _, payload = await server.call_tool("server_info", {})
        return payload

    payload = __import__("asyncio").run(_call())
    assert payload["transport"] == "stdio"
    assert payload["bind_host"] == "127.0.0.1"
    assert payload["registered_tool_count"] > 0
    assert payload["tool_catalog_fingerprint"] is not None
    assert payload["http_bridge_tool_count"] >= payload["registered_tool_count"]
    assert payload["http_bridge_fingerprint"] is not None
    assert any(route["tool"] == "server_info" and "/Lightcap/server_info" in route["paths"] for route in payload["http_compatibility_routes"])
    assert "network_note" in payload


def test_high_level_physics_pipeline_and_nested_probe_run_when_available() -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from spectral_packet_engine import create_mcp_server

    async def _call():
        server = create_mcp_server()
        _, potential_payload = await server.call_tool(
            "analyze_potential_pipeline",
            {"potential": "double_well", "num_points": 64, "temperature": 5.0, "device": "cpu"},
        )
        _, tunneling_payload = await server.call_tool(
            "tunneling_experiment",
            {
                "barrier_height": 20.0,
                "barrier_width_sigma": 0.03,
                "grid_points": 96,
                "num_modes": 48,
                "num_energies": 48,
                "propagation_steps": 48,
                "dt": 1e-5,
                "device": "cpu",
            },
        )
        _, probe_payload = await server.call_tool("probe_mcp_runtime", {"profile": "smoke"})
        return potential_payload, tunneling_payload, probe_payload

    potential_payload, tunneling_payload, probe_payload = __import__("asyncio").run(_call())
    assert potential_payload["potential_name"] == "double_well"
    assert "related_tools" in potential_payload
    assert tunneling_payload["num_modes"] == 48
    assert "related_tools" in tunneling_payload
    assert "packet_mean_energy" in tunneling_payload["scattering"]
    assert "transmitted_probability" in tunneling_payload["propagation"]
    assert probe_payload["summary"]["failed_count"] == 0


def test_optimize_packet_control_tool_accepts_interval_probability_when_available() -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from spectral_packet_engine import create_mcp_server

    async def _call():
        server = create_mcp_server()
        _, payload = await server.call_tool(
            "optimize_packet_control",
            {
                "center": 0.25,
                "width": 0.08,
                "wavenumber": 18.0,
                "phase": 0.0,
                "objective": "interval_probability",
                "target_value": 0.35,
                "final_time": 0.004,
                "interval": [0.5, 1.0],
                "num_modes": 48,
                "quadrature_points": 1024,
                "grid_points": 64,
                "steps": 20,
                "learning_rate": 0.03,
                "device": "cpu",
            },
        )
        return payload

    payload = __import__("asyncio").run(_call())
    assert payload["objective"] == "interval_probability"
    assert payload["final_interval_probability"] is not None


def test_streamable_http_compatibility_routes_expose_runtime_tools_when_available() -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from starlette.testclient import TestClient

    from spectral_packet_engine import MCPServerConfig, create_mcp_server

    server = create_mcp_server(MCPServerConfig(transport="streamable-http", port=8766))
    client = TestClient(server.streamable_http_app())

    registry_response = client.get("/tool_registry")
    assert registry_response.status_code == 200
    registry = registry_response.json()["tools"]
    assert any(route["tool"] == "inspect_mcp_runtime" and "/inspect_mcp_runtime" in route["paths"] for route in registry)
    assert any(route["tool"] == "probe_mcp_runtime" and "/Lightcap/probe_mcp_runtime" in route["paths"] for route in registry)
    assert any(route["tool"] == "tunneling_experiment" and "/Lightcap/tunneling_experiment" in route["paths"] for route in registry)
    assert any(route["tool"] == "optimize_packet_control" and "/Lightcap/optimize_packet_control" in route["paths"] for route in registry)

    namespaced_registry = client.get("/Lightcap/tool_registry")
    assert namespaced_registry.status_code == 200
    assert "/Lightcap/tool_registry" in namespaced_registry.json()["registry_paths"]

    runtime_response = client.get("/inspect_mcp_runtime")
    assert runtime_response.status_code == 200
    runtime_payload = runtime_response.json()
    assert runtime_payload["transport"] == "streamable-http"
    assert runtime_payload["inspection_scope"] == "running-instance"
    assert runtime_payload["http_bridge_tool_count"] > 0
    assert runtime_payload["http_bridge_fingerprint"] is not None
    assert any(route["tool"] == "validate_installation" and "/Lightcap/validate_installation" in route["paths"] for route in runtime_payload["http_compatibility_routes"])

    validate_response = client.get("/validate_installation")
    assert validate_response.status_code == 200
    validate_payload = validate_response.json()
    assert validate_payload["environment"]["mcp_runtime"]["inspection_scope"] == "package-default"
    assert validate_payload["http_bridge"]["tool"] == "validate_installation"

    inspect_product_response = client.get("/Lightcap/inspect_product")
    assert inspect_product_response.status_code == 200
    inspect_product_payload = inspect_product_response.json()
    assert inspect_product_payload["product_name"] == "Spectral Packet Engine"
    assert inspect_product_payload["http_bridge"]["requested_path"] == "/Lightcap/inspect_product"

    control_response = client.post(
        "/Lightcap/optimize_packet_control",
        json={
            "center": 0.25,
            "width": 0.08,
            "wavenumber": 18.0,
            "phase": 0.0,
            "objective": "interval_probability",
            "target_value": 0.35,
            "final_time": 0.004,
            "interval": [0.5, 1.0],
            "num_modes": 48,
            "quadrature_points": 1024,
            "grid_points": 64,
            "steps": 20,
            "learning_rate": 0.03,
            "device": "cpu",
        },
    )
    assert control_response.status_code == 200
    control_payload = control_response.json()
    assert control_payload["objective"] == "interval_probability"
    assert control_payload["http_bridge"]["requested_path"] == "/Lightcap/optimize_packet_control"

    scattering_response = client.post(
        "/Lightcap/analyze_scattering_pipeline",
        json={
            "barrier_type": "double",
            "barrier_height": 20.0,
            "barrier_width": 0.04,
            "separation": 0.08,
        },
    )
    assert scattering_response.status_code == 200
    scattering_payload = scattering_response.json()
    assert scattering_payload["num_segments"] == 5
    assert scattering_payload["max_transmission"] >= scattering_payload["min_transmission"]
    assert scattering_payload["http_bridge"]["requested_path"] == "/Lightcap/analyze_scattering_pipeline"

    tunneling_response = client.post(
        "/Lightcap/tunneling_experiment",
        json={
            "barrier_height": 20.0,
            "barrier_width_sigma": 0.03,
            "grid_points": 96,
            "num_modes": 48,
            "num_energies": 48,
            "propagation_steps": 48,
            "dt": 1e-5,
            "device": "cpu",
        },
    )
    assert tunneling_response.status_code == 200
    tunneling_payload = tunneling_response.json()
    assert tunneling_payload["num_modes"] == 48
    assert tunneling_payload["http_bridge"]["requested_path"] == "/Lightcap/tunneling_experiment"

    self_test_response = client.get("/self_test")
    assert self_test_response.status_code == 200
    self_test_payload = self_test_response.json()
    assert self_test_payload["overall"] == "all_passed"
    assert self_test_payload["http_bridge"]["tool"] == "self_test"


def test_streamable_http_dynamic_bridge_tracks_live_tool_manager_when_available() -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from starlette.testclient import TestClient

    from spectral_packet_engine import MCPServerConfig, create_mcp_server

    server = create_mcp_server(MCPServerConfig(transport="streamable-http", port=8766))

    @server.tool(name="dynamic_http_probe", description="Probe the dynamic HTTP bridge.", structured_output=True)
    def dynamic_http_probe(answer: int = 7) -> dict[str, int]:
        return {"answer": answer}

    client = TestClient(server.streamable_http_app())

    registry_response = client.get("/Lightcap/tool_registry")
    assert registry_response.status_code == 200
    registry = registry_response.json()["tools"]
    assert any(route["tool"] == "dynamic_http_probe" and "/Lightcap/dynamic_http_probe" in route["paths"] for route in registry)

    probe_response = client.get("/Lightcap/dynamic_http_probe?answer=9")
    assert probe_response.status_code == 200
    probe_payload = probe_response.json()
    assert probe_payload["answer"] == 9
    assert probe_payload["http_bridge"]["requested_path"] == "/Lightcap/dynamic_http_probe"

    missing_response = client.get("/Lightcap/definitely_missing_tool")
    assert missing_response.status_code == 404
    missing_payload = missing_response.json()
    assert missing_payload["error"] is True
    assert missing_payload["error_type"] == "ToolNotFound"
