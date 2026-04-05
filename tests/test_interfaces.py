from __future__ import annotations

import asyncio
from threading import Thread
from time import sleep

import pytest

from spectral_packet_engine import MCPServerConfig, api_is_available, inspect_api_stack, inspect_mcp_runtime
from spectral_packet_engine.artifacts import write_tabular_artifacts
from spectral_packet_engine.mcp import _MCPExecutionController, _tool
from spectral_packet_engine.tabular import TabularDataset


def test_api_app_health_endpoint(tmp_path) -> None:
    if not api_is_available():
        pytest.skip("FastAPI stack is not available or not compatible in this environment")

    from fastapi.testclient import TestClient

    from spectral_packet_engine import create_api_app

    client = TestClient(create_api_app())
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["product_name"] == "Spectral Packet Engine"
    assert response.json()["version"]

    product = client.get("/product")
    assert product.status_code == 200
    assert product.json()["product_name"] == "Spectral Packet Engine"
    assert product.json()["hero_workflow"]["workflow_id"] == "profile-table-report"
    assert len(product.json()["killer_workflows"]) == 3

    workflow_guide = client.get("/workflow/guide", params={"input_kind": "profile-table-sql", "goal": "feature-model"})
    assert workflow_guide.status_code == 200
    assert workflow_guide.json()["primary_workflow"]["workflow_id"] == "profile-table-report-from-sql"
    assert workflow_guide.json()["goal"] == "feature-model"
    assert workflow_guide.json()["defaults"]["sort_by_time"] is True
    assert any(step.get("workflow_id") == "export-feature-table-from-sql" for step in workflow_guide.json()["plan_steps"])

    status = client.get("/status")
    assert status.status_code == 200
    assert status.json()["service_name"] == "Spectral Packet Engine"
    assert "counters" in status.json()

    file_formats = client.get("/file-formats")
    assert file_formats.status_code == 200
    assert "csv" in file_formats.json()

    api_stack = client.get("/api/stack")
    assert api_stack.status_code == 200
    assert api_stack.json()["compatible"] is True

    tabular_formats = client.get("/tabular-formats")
    assert tabular_formats.status_code == 200
    assert "parquet" in tabular_formats.json()

    ml_backends = client.get("/ml/backends")
    assert ml_backends.status_code == 200
    assert "torch" in ml_backends.json()["backends"]

    tree_backends = client.get("/tree/backends")
    assert tree_backends.status_code == 200
    assert tree_backends.json()["resolution"]["resolved_library"] == "sklearn"

    analyze = client.post(
        "/profiles/analyze",
        json={
            "table": {
                "position_grid": [0.0, 0.5, 1.0],
                "sample_times": [0.0, 0.1],
                "profiles": [
                    [0.2, 1.0, 0.2],
                    [0.1, 0.9, 0.1],
                ],
            },
            "num_modes": 3,
            "device": "cpu",
        },
    )
    assert analyze.status_code == 200
    assert analyze.json()["num_modes"] == 3

    profile_report = client.post(
        "/profiles/report",
        json={
            "table": {
                "position_grid": [0.0, 0.5, 1.0],
                "sample_times": [0.0, 0.1],
                "profiles": [
                    [0.2, 1.0, 0.2],
                    [0.1, 0.9, 0.1],
                ],
            },
            "analyze_num_modes": 4,
            "compress_num_modes": 3,
            "device": "cpu",
        },
    )
    assert profile_report.status_code == 200
    assert profile_report.json()["overview"]["analyze_num_modes"] == 4
    assert profile_report.json()["overview"]["compress_num_modes"] == 3

    feature_export = client.post(
        "/features/export",
        json={
            "table": {
                "position_grid": [0.0, 0.5, 1.0],
                "sample_times": [0.0, 0.1],
                "profiles": [
                    [0.2, 1.0, 0.2],
                    [0.1, 0.9, 0.1],
                ],
            },
            "num_modes": 3,
            "device": "cpu",
            "output_dir": str(tmp_path / "api_feature_export"),
        },
    )
    assert feature_export.status_code == 200
    assert feature_export.json()["num_features"] == 6
    assert (tmp_path / "api_feature_export" / "features.csv").exists()

    db_query = client.post(
        "/database/query",
        json={
            "database": ":memory:",
            "query": "SELECT 1 AS value",
        },
    )
    assert db_query.status_code == 200
    assert db_query.json()["table"]["row_count"] == 1
    assert db_query.json()["redacted_url"] == "sqlite:///:memory:"

    database_path = tmp_path / "profiles.sqlite"
    db_execute = client.post(
        "/database/execute",
        json={
            "database": str(database_path),
            "statement": 'CREATE TABLE IF NOT EXISTS "api_metrics" (time REAL, value REAL)',
        },
    )
    assert db_execute.status_code == 200
    assert db_execute.json()["mode"] == "statement"
    assert db_execute.json()["statement_count"] == 1

    db_script = client.post(
        "/database/script",
        json={
            "database": str(database_path),
            "script": """
            INSERT INTO "api_metrics" (time, value) VALUES (0.0, 1.0);
            INSERT INTO "api_metrics" (time, value) VALUES (0.1, 2.0);
            """,
        },
    )
    assert db_script.status_code == 200
    assert db_script.json()["mode"] == "script"
    assert db_script.json()["statement_count"] == 2

    db_write = client.post(
        "/database/write",
        json={
            "database": str(database_path),
            "table_name": "profiles",
            "dataset": {
                "rows": [
                    {"label": "late", "time": 0.2, "x=1.0": 0.15, "x=0.0": 0.2, "x=0.5": 0.85},
                    {"label": "early", "time": 0.0, "x=1.0": 0.1, "x=0.0": 0.1, "x=0.5": 1.0},
                ]
            },
            "if_exists": "replace",
        },
    )
    assert db_write.status_code == 200

    artifact_dir = tmp_path / "artifacts"
    write_tabular_artifacts(
        artifact_dir,
        TabularDataset.from_rows([{"time": 0.0, "value": 1.0}]),
        metadata={"workflow": "api-artifacts-test"},
    )
    artifacts = client.get("/artifacts", params={"output_dir": str(artifact_dir)})
    assert artifacts.status_code == 200
    assert artifacts.json()["complete"] is True
    assert artifacts.json()["metadata"]["workflow"] == "api-artifacts-test"

    compress_from_sql = client.post(
        "/profiles/compress-from-sql",
        json={
            "database": str(database_path),
            "query": 'SELECT label, time, "x=1.0", "x=0.0", "x=0.5" FROM "profiles"',
            "time_column": "time",
            "position_columns": ["x=1.0", "x=0.0", "x=0.5"],
            "sort_by_time": True,
            "num_modes": 3,
            "device": "cpu",
        },
    )
    assert compress_from_sql.status_code == 200
    assert compress_from_sql.json()["num_modes"] == 3

    report_from_sql = client.post(
        "/profiles/report-from-sql",
        json={
            "database": str(database_path),
            "query": 'SELECT label, time, "x=1.0", "x=0.0", "x=0.5" FROM "profiles"',
            "time_column": "time",
            "position_columns": ["x=1.0", "x=0.0", "x=0.5"],
            "sort_by_time": True,
            "analyze_num_modes": 4,
            "compress_num_modes": 3,
            "device": "cpu",
        },
    )
    assert report_from_sql.status_code == 200
    assert report_from_sql.json()["overview"]["analyze_num_modes"] == 4
    assert report_from_sql.json()["overview"]["compress_num_modes"] == 3

    missing_db_query = client.post(
        "/database/query",
        json={
            "database": "/definitely/missing/spectral_packet_engine.sqlite",
            "query": "SELECT 1 AS value",
        },
    )
    assert missing_db_query.status_code == 404
    assert "does not exist" in missing_db_query.json()["error"]

    ml_eval = client.post(
        "/ml/evaluate",
        json={
            "table": {
                "position_grid": [0.0, 0.5, 1.0],
                "sample_times": [0.0, 0.1, 0.2],
                "profiles": [
                    [0.2, 1.0, 0.2],
                    [0.1, 0.9, 0.1],
                    [0.05, 0.8, 0.05],
                ],
            },
            "backend": "torch",
            "num_modes": 3,
            "epochs": 4,
            "batch_size": 2,
            "device": "cpu",
        },
    )
    assert ml_eval.status_code == 200
    assert ml_eval.json()["backend"] == "torch"

    tree_rows = [
        {
            "time": float(index) / 11.0,
            "mode_1": 0.2 + 0.03 * index,
            "mode_2": 1.0 - 0.02 * index,
            "mean_position": 0.25 + 0.01 * index,
            "target": 1.1 + 0.08 * index,
        }
        for index in range(12)
    ]
    tree_train = client.post(
        "/tree/train",
        json={
            "dataset": {"rows": tree_rows},
            "target_column": "target",
            "library": "sklearn",
            "params": {"n_estimators": 16, "max_depth": 4},
            "output_dir": str(tmp_path / "api_tree_train"),
        },
    )
    assert tree_train.status_code == 200
    assert tree_train.json()["library"] == "sklearn"
    assert (tmp_path / "api_tree_train" / "tree_training.json").exists()

    tree_tune = client.post(
        "/tree/tune",
        json={
            "dataset": {"rows": tree_rows},
            "target_column": "target",
            "library": "sklearn",
            "search_space": {"n_estimators": [8, 16], "max_depth": [2, 4]},
            "search_kind": "grid",
            "cv": 2,
            "output_dir": str(tmp_path / "api_tree_tune"),
        },
    )
    assert tree_tune.status_code == 200
    assert tree_tune.json()["candidate_count"] == 4
    assert (tmp_path / "api_tree_tune" / "best_model" / "artifacts.json").exists()


def test_mcp_server_builds_when_available() -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from spectral_packet_engine import create_mcp_server

    server = create_mcp_server()
    assert hasattr(server, "run")


def test_mcp_runtime_report_is_explicit_about_transport_and_supervision() -> None:
    report = inspect_mcp_runtime(MCPServerConfig(max_concurrent_tasks=2, slot_acquire_timeout_seconds=5.0))

    assert report.transport == "stdio"
    assert report.stderr_logging_safe is True
    assert report.forced_cancellation_supported is False
    assert report.config.max_concurrent_tasks == 2
    assert any("Automatic restarts for stdio MCP belong to the client launcher" in note for note in report.notes)

    http_report = inspect_mcp_runtime(
        MCPServerConfig(
            transport="streamable-http",
            host="127.0.0.1",
            port=8765,
            streamable_http_path="/mcp",
        )
    )
    assert http_report.transport == "streamable-http"
    assert http_report.config.endpoint_url == "http://127.0.0.1:8765/mcp"
    assert http_report.recommended_supervision
    public_http_report = inspect_mcp_runtime(
        MCPServerConfig(
            transport="streamable-http",
            host="127.0.0.1",
            port=8765,
            streamable_http_path="/mcp",
            allowed_hosts=("lightcap.ai", "lightcap.ai:*"),
            allowed_origins=("https://lightcap.ai",),
        )
    )
    assert any("lightcap.ai" in note for note in public_http_report.notes)


def test_mcp_execution_controller_fails_cleanly_when_saturated() -> None:
    controller = _MCPExecutionController(
        MCPServerConfig(max_concurrent_tasks=1, slot_acquire_timeout_seconds=0.01)
    )
    controller.acquire()
    try:
        with pytest.raises(RuntimeError, match="saturated"):
            controller.acquire()
    finally:
        controller.release()


def test_bounded_mcp_tool_returns_clear_error_payload_when_queue_times_out() -> None:
    class _FakeServer:
        def tool(self, **kwargs):
            def _register(function):
                return function

            return _register

    runtime = _MCPExecutionController(
        MCPServerConfig(max_concurrent_tasks=1, slot_acquire_timeout_seconds=0.01)
    )

    def _hold_slot() -> None:
        runtime.acquire()
        try:
            sleep(0.05)
        finally:
            runtime.release()

    holder = Thread(target=_hold_slot)
    holder.start()
    sleep(0.005)
    wrapped = _tool(_FakeServer(), runtime, "bounded_test_tool", "Bounded test tool.", bounded=True)(
        lambda: {"ok": True}
    )
    payload = wrapped()
    holder.join()

    assert payload["error"] is True
    assert payload["error_type"] == "RuntimeError"
    assert "saturated" in payload["error_message"]
    assert payload["tool"] == "bounded_test_tool"


def test_api_stack_report_matches_public_boolean() -> None:
    report = inspect_api_stack()

    assert report.compatible is api_is_available()
    if report.installed:
        assert report.fastapi_version is not None
        assert report.uvicorn_version is not None


def test_mcp_server_tool_metadata_is_product_shaped_when_available() -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from spectral_packet_engine import create_mcp_server

    async def _list_tools():
        server = create_mcp_server()
        return await server.list_tools()

    tools = asyncio.run(_list_tools())
    by_name = {tool.name: tool for tool in tools}

    assert "inspect_product" in by_name
    assert "guide_workflow" in by_name
    assert "inspect_environment" in by_name
    assert "inspect_service_status" in by_name
    assert "inspect_environment_tool" not in by_name
    assert "profile_table_report" in by_name
    assert "inspect_tree_backends" in by_name
    assert "export_feature_table" in by_name
    assert "export_feature_table_from_sql" in by_name
    assert "execute_database_statement" in by_name
    assert "execute_database_script" in by_name
    assert "train_tree_model" in by_name
    assert "tune_tree_model" in by_name
    assert "compress_database_profile_query" in by_name
    assert "report_database_profile_query" in by_name
    assert "fit_packet_to_database_profile_query" in by_name
    assert "train_modal_surrogate_from_sql" in by_name
    assert by_name["inspect_environment"].description
    assert by_name["query_database"].description
    assert "read-only" in by_name["query_database"].description


def test_mcp_product_tool_reports_shared_identity_when_available() -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from spectral_packet_engine import create_mcp_server

    async def _inspect():
        server = create_mcp_server()
        _, payload = await server.call_tool("inspect_product", {})
        return payload

    payload = asyncio.run(_inspect())
    assert payload["product_name"] == "Spectral Packet Engine"
    assert payload["hero_workflow"]["workflow_id"] == "profile-table-report"
    assert len(payload["killer_workflows"]) == 3


def test_mcp_workflow_guide_reports_operator_loop_when_available() -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from spectral_packet_engine import create_mcp_server

    async def _guide():
        server = create_mcp_server()
        _, payload = await server.call_tool("guide_workflow", {"input_kind": "profile-table-sql", "goal": "feature-model"})
        return payload

    payload = asyncio.run(_guide())
    assert payload["surface"] == "mcp"
    assert payload["goal"] == "feature-model"
    assert payload["killer_workflow"]["killer_workflow_id"] == "spectral-feature-model-loop"
    assert payload["primary_workflow"]["workflow_id"] == "profile-table-report-from-sql"
    assert any(step.get("workflow_id") == "export-feature-table-from-sql" for step in payload["plan_steps"])


def test_mcp_runtime_and_artifact_tools_report_shared_runtime_state_when_available(tmp_path) -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from spectral_packet_engine import create_mcp_server

    artifact_dir = tmp_path / "artifacts"
    write_tabular_artifacts(
        artifact_dir,
        TabularDataset.from_rows([{"time": 0.0, "value": 1.0}]),
        metadata={"workflow": "mcp-artifacts-test"},
    )

    async def _inspect():
        server = create_mcp_server(
            MCPServerConfig(max_concurrent_tasks=1, slot_acquire_timeout_seconds=1.0, log_level="warning")
        )
        _, runtime_payload = await server.call_tool("inspect_mcp_runtime", {})
        _, artifact_payload = await server.call_tool("list_artifacts", {"output_dir": str(artifact_dir)})
        return runtime_payload, artifact_payload

    runtime_payload, artifact_payload = asyncio.run(_inspect())

    assert runtime_payload["transport"] == "stdio"
    assert runtime_payload["config"]["max_concurrent_tasks"] == 1
    assert runtime_payload["stderr_logging_safe"] is True
    assert artifact_payload["complete"] is True
    assert artifact_payload["metadata"]["workflow"] == "mcp-artifacts-test"


def test_mcp_profile_report_tool_uses_shared_report_workflow_when_available(tmp_path) -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from spectral_packet_engine import create_mcp_server, save_profile_table_csv
    from spectral_packet_engine.table_io import ProfileTable
    import numpy as np

    grid = np.linspace(0.0, 1.0, 8)
    times = np.asarray([0.0, 0.1], dtype=np.float64)
    profiles = np.asarray(
        [
            [0.1, 0.2, 0.5, 0.9, 0.5, 0.2, 0.1, 0.05],
            [0.05, 0.1, 0.35, 0.8, 0.55, 0.25, 0.1, 0.05],
        ],
        dtype=np.float64,
    )
    table_path = tmp_path / "profiles.csv"
    save_profile_table_csv(ProfileTable(position_grid=grid, sample_times=times, profiles=profiles), table_path)

    async def _call():
        server = create_mcp_server()
        _, payload = await server.call_tool(
            "profile_table_report",
            {
                "table_path": str(table_path),
                "analyze_num_modes": 4,
                "compress_num_modes": 3,
                "device": "cpu",
            },
        )
        return payload

    payload = asyncio.run(_call())
    assert payload["overview"]["analyze_num_modes"] == 4
    assert payload["overview"]["compress_num_modes"] == 3


def test_mcp_feature_export_and_tree_tools_use_shared_workflows_when_available(tmp_path) -> None:
    pytest.importorskip("mcp.server.fastmcp")

    import numpy as np

    from spectral_packet_engine import create_mcp_server, save_profile_table_csv
    from spectral_packet_engine.table_io import ProfileTable
    from spectral_packet_engine.tabular import TabularDataset, save_tabular_dataset

    grid = np.linspace(0.0, 1.0, 8)
    times = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    profiles = np.asarray(
        [
            np.exp(-((grid - 0.25) ** 2) / (2 * 0.06**2)),
            np.exp(-((grid - 0.40) ** 2) / (2 * 0.06**2)),
            np.exp(-((grid - 0.55) ** 2) / (2 * 0.06**2)),
        ],
        dtype=np.float64,
    )
    profile_table_path = tmp_path / "profiles.csv"
    save_profile_table_csv(ProfileTable(position_grid=grid, sample_times=times, profiles=profiles), profile_table_path)

    feature_rows = [
        {
            "time": float(index) / 11.0,
            "mode_1": 0.2 + 0.03 * index,
            "mode_2": 1.0 - 0.02 * index,
            "mean_position": 0.25 + 0.01 * index,
            "target": 1.1 + 0.08 * index,
        }
        for index in range(12)
    ]
    features_path = tmp_path / "features.csv"
    save_tabular_dataset(TabularDataset.from_rows(feature_rows), features_path)

    async def _call():
        server = create_mcp_server()
        _, backend_payload = await server.call_tool("inspect_tree_backends", {"library": "sklearn"})
        _, feature_payload = await server.call_tool(
            "export_feature_table",
            {
                "table_path": str(profile_table_path),
                "num_modes": 4,
                "device": "cpu",
                "output_dir": str(tmp_path / "mcp_feature_export"),
            },
        )
        _, training_payload = await server.call_tool(
            "train_tree_model",
            {
                "features_path": str(features_path),
                "target_column": "target",
                "library": "sklearn",
                "params": {"n_estimators": 16, "max_depth": 4},
                "output_dir": str(tmp_path / "mcp_tree_train"),
            },
        )
        _, tuning_payload = await server.call_tool(
            "tune_tree_model",
            {
                "features_path": str(features_path),
                "target_column": "target",
                "library": "sklearn",
                "search_space": {"n_estimators": [8, 16], "max_depth": [2, 4]},
                "search_kind": "grid",
                "cv": 2,
                "output_dir": str(tmp_path / "mcp_tree_tune"),
            },
        )
        return backend_payload, feature_payload, training_payload, tuning_payload

    backend_payload, feature_payload, training_payload, tuning_payload = asyncio.run(_call())

    assert backend_payload["resolution"]["resolved_library"] == "sklearn"
    assert feature_payload["num_features"] == 7
    assert training_payload["library"] == "sklearn"
    assert tuning_payload["candidate_count"] == 4
    assert (tmp_path / "mcp_feature_export" / "features.csv").exists()
    assert (tmp_path / "mcp_tree_train" / "tree_training.json").exists()
    assert (tmp_path / "mcp_tree_tune" / "best_model" / "artifacts.json").exists()
