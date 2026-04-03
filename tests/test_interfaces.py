from __future__ import annotations

import asyncio

import pytest

from spectral_packet_engine import api_is_available, inspect_api_stack


def test_api_app_health_endpoint() -> None:
    if not api_is_available():
        pytest.skip("FastAPI stack is not available or not compatible in this environment")

    from fastapi.testclient import TestClient

    from spectral_packet_engine import create_api_app

    client = TestClient(create_api_app())
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

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


def test_mcp_server_builds_when_available() -> None:
    pytest.importorskip("mcp.server.fastmcp")

    from spectral_packet_engine import create_mcp_server

    server = create_mcp_server()
    assert hasattr(server, "run")


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

    assert "inspect_environment" in by_name
    assert "inspect_environment_tool" not in by_name
    assert "train_modal_surrogate_from_sql" in by_name
    assert by_name["inspect_environment"].description
    assert by_name["query_database"].description
