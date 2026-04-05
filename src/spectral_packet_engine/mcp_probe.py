from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
from time import perf_counter
from typing import Any

from spectral_packet_engine.artifacts import to_serializable, write_mcp_probe_artifacts as _write_mcp_probe_artifacts
from spectral_packet_engine.mcp_deployment import default_mcp_server_command, detect_source_checkout
from spectral_packet_engine.version import __version__


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_tool_payload(result: Any) -> Any:
    structured_content = getattr(result, "structuredContent", None)
    if structured_content is not None:
        return to_serializable(structured_content)
    content = getattr(result, "content", None)
    if not content:
        return None
    first = content[0]
    text = getattr(first, "text", None)
    if text is None:
        return to_serializable(first)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def _probe_summary_line(probe_id: str, passed: bool, message: str) -> str:
    return f"{probe_id}: {'passed' if passed else 'failed'} - {message}"


def _error_payload(tool: str, exc: Exception) -> dict[str, Any]:
    return {
        "error": True,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "tool": tool,
    }


@dataclass(frozen=True, slots=True)
class MCPProbeServerSpec:
    command: tuple[str, ...]
    cwd: str
    env: dict[str, str]
    transport: str = "stdio"
    log_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "cwd": self.cwd,
            "env": dict(self.env),
            "transport": self.transport,
            "log_file": self.log_file,
        }


@dataclass(frozen=True, slots=True)
class MCPRuntimeCheck:
    check_id: str
    command: tuple[str, ...]
    returncode: int
    passed: bool
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "command": list(self.command),
            "returncode": self.returncode,
            "passed": self.passed,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True, slots=True)
class MCPToolCallRecord:
    tool: str
    arguments: dict[str, Any]
    elapsed_ms: float
    is_error: bool
    response: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": to_serializable(self.arguments),
            "elapsed_ms": self.elapsed_ms,
            "is_error": self.is_error,
            "response": to_serializable(self.response),
        }


@dataclass(frozen=True, slots=True)
class MCPProbeResult:
    probe_id: str
    category: str
    description: str
    passed: bool
    severity: str
    tool: str | None
    expected: str
    actual: str
    elapsed_ms: float
    response: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "category": self.category,
            "description": self.description,
            "passed": self.passed,
            "severity": self.severity,
            "tool": self.tool,
            "expected": self.expected,
            "actual": self.actual,
            "elapsed_ms": self.elapsed_ms,
            "response": to_serializable(self.response),
        }


@dataclass(frozen=True, slots=True)
class MCPProbeReport:
    probe_version: str
    profile: str
    started_at_utc: str
    finished_at_utc: str
    server: MCPProbeServerSpec
    runtime_checks: tuple[MCPRuntimeCheck, ...]
    tool_calls: tuple[MCPToolCallRecord, ...]
    probes: tuple[MCPProbeResult, ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_version": self.probe_version,
            "profile": self.profile,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "server": self.server.to_dict(),
            "runtime_checks": [item.to_dict() for item in self.runtime_checks],
            "tool_calls": [item.to_dict() for item in self.tool_calls],
            "probes": [item.to_dict() for item in self.probes],
            "summary": to_serializable(self.summary),
        }


def build_local_probe_server_spec(
    *,
    working_directory: str | Path,
    python_executable: str = "python3",
    max_concurrent_tasks: int = 1,
    slot_timeout_seconds: float = 60.0,
    log_level: str = "warning",
    log_file: str | Path | None = None,
    allow_unsafe_python: bool = False,
    scratch_directory: str | Path | None = None,
    source_checkout: bool | None = None,
) -> MCPProbeServerSpec:
    root = Path(working_directory).resolve()
    use_source_checkout = detect_source_checkout(root) if source_checkout is None else bool(source_checkout)
    env = {"PYTHONPATH": "src"} if use_source_checkout else {}
    command = default_mcp_server_command(
        python_executable=python_executable,
        max_concurrent_tasks=max_concurrent_tasks,
        slot_timeout_seconds=slot_timeout_seconds,
        log_level=log_level,
        log_file=log_file,
        allow_unsafe_python=allow_unsafe_python,
        scratch_directory=scratch_directory,
    )
    return MCPProbeServerSpec(
        command=command,
        cwd=str(root),
        env=env,
        log_file=None if log_file is None else str(log_file),
    )


async def _run_probe_suite_async(
    server: MCPProbeServerSpec,
    expect_unsafe_python_enabled: bool,
    profile: str,
    skip_nested_probe: bool,
) -> MCPProbeReport:
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("The MCP probe suite requires the 'mcp' extra.") from exc

    started_at = _utc_now_iso()
    runtime_checks = []
    if server.command and server.command[0] != "ssh":
        runtime_checks.append(
            _run_module_entrypoint_check(
                cwd=server.cwd,
                env=server.env,
                python_executable=server.command[0],
            )
        )
    tool_calls: list[MCPToolCallRecord] = []
    probes: list[MCPProbeResult] = []

    params = StdioServerParameters(
        command=server.command[0],
        args=list(server.command[1:]),
        env=server.env,
        cwd=server.cwd,
    )

    async with stdio_client(params, errlog=sys.__stderr__) as (read, write):
        async with ClientSession(read, write) as session:
            initialize_result = await session.initialize()
            tools_result = await session.list_tools()
            tool_names = sorted(tool.name for tool in tools_result.tools)

            async def call_tool(name: str, arguments: dict[str, Any]) -> tuple[Any, float]:
                started = perf_counter()
                try:
                    result = await session.call_tool(name, arguments)
                except Exception as exc:
                    elapsed_ms = float((perf_counter() - started) * 1000.0)
                    payload = _error_payload(name, exc)
                    tool_calls.append(
                        MCPToolCallRecord(
                            tool=name,
                            arguments=arguments,
                            elapsed_ms=elapsed_ms,
                            is_error=True,
                            response=payload,
                        )
                    )
                    return payload, elapsed_ms
                elapsed_ms = float((perf_counter() - started) * 1000.0)
                payload = _parse_tool_payload(result)
                tool_calls.append(
                    MCPToolCallRecord(
                        tool=name,
                        arguments=arguments,
                        elapsed_ms=elapsed_ms,
                        is_error=bool(getattr(result, "isError", False)),
                        response=payload,
                    )
                )
                return payload, elapsed_ms

            required_tools = {
                "inspect_product",
                "guide_workflow",
                "inspect_mcp_runtime",
                "self_test",
                "server_info",
                "probe_mcp_runtime",
                "tunneling_experiment",
            }
            probes.append(
                MCPProbeResult(
                    probe_id="RUNTIME-001",
                    category="bootstrap",
                    description="The MCP server initializes over stdio and advertises tools.",
                    passed=bool(tool_names),
                    severity="high",
                    tool=None,
                    expected="initialize succeeds and at least one tool is listed",
                    actual=f"initialized {initialize_result.serverInfo.name} with {len(tool_names)} tools",
                    elapsed_ms=0.0,
                    response={"server": initialize_result.serverInfo.name, "tool_count": len(tool_names)},
                )
            )
            probes.append(
                MCPProbeResult(
                    probe_id="DISCOVERY-001",
                    category="discovery",
                    description="Tool discovery exposes the core product, physics, and self-audit tools an AI client needs.",
                    passed=required_tools.issubset(set(tool_names)),
                    severity="high",
                    tool=None,
                    expected="required MCP tools are present in list_tools()",
                    actual=f"missing={sorted(required_tools.difference(tool_names))}",
                    elapsed_ms=0.0,
                    response={"required_tools": sorted(required_tools), "tool_count": len(tool_names)},
                )
            )

            runtime_payload, runtime_elapsed = await call_tool("inspect_mcp_runtime", {})
            runtime_passed = (
                isinstance(runtime_payload, dict)
                and runtime_payload.get("transport") == "stdio"
                and runtime_payload.get("config", {}).get("allow_unsafe_python") == expect_unsafe_python_enabled
            )
            probes.append(
                MCPProbeResult(
                    probe_id="RUNTIME-002",
                    category="runtime",
                    description="Runtime inspection exposes transport and unsafe-python policy.",
                    passed=runtime_passed,
                    severity="high",
                    tool="inspect_mcp_runtime",
                    expected="transport is stdio and policy matches startup configuration",
                    actual=(
                        f"transport={runtime_payload.get('transport')} "
                        f"allow_unsafe_python={runtime_payload.get('config', {}).get('allow_unsafe_python')}"
                        if isinstance(runtime_payload, dict)
                        else "unexpected response"
                    ),
                    elapsed_ms=runtime_elapsed,
                    response=runtime_payload,
                )
            )

            server_info_payload, server_info_elapsed = await call_tool("server_info", {})
            server_info_passed = (
                isinstance(server_info_payload, dict)
                and server_info_payload.get("transport") == "stdio"
                and "bind_host" in server_info_payload
                and "network_note" in server_info_payload
            )
            probes.append(
                MCPProbeResult(
                    probe_id="RUNTIME-003",
                    category="runtime",
                    description="Server connection metadata reports bind/endpoint facts instead of pretending hostname resolution is authoritative.",
                    passed=server_info_passed,
                    severity="high",
                    tool="server_info",
                    expected="transport, bind_host, and network_note are present",
                    actual=(
                        f"transport={server_info_payload.get('transport')} bind_host={server_info_payload.get('bind_host')}"
                        if isinstance(server_info_payload, dict)
                        else "unexpected response"
                    ),
                    elapsed_ms=server_info_elapsed,
                    response=server_info_payload,
                )
            )

            self_test_payload, self_test_elapsed = await call_tool("self_test", {"device": "cpu"})
            self_test_passed = isinstance(self_test_payload, dict) and self_test_payload.get("overall") == "all_passed"
            probes.append(
                MCPProbeResult(
                    probe_id="STATUS-001",
                    category="health",
                    description="The shared self-test covers core engine, physics, load modeling, scratch, and SQL state.",
                    passed=self_test_passed,
                    severity="high",
                    tool="self_test",
                    expected="overall reports all_passed",
                    actual=str(None if not isinstance(self_test_payload, dict) else self_test_payload.get("overall")),
                    elapsed_ms=self_test_elapsed,
                    response=self_test_payload,
                )
            )

            invalid_payload, invalid_elapsed = await call_tool("simulate_packet", {"num_modes": 0, "device": "cpu"})
            invalid_passed = (
                isinstance(invalid_payload, dict)
                and invalid_payload.get("error") is True
                and invalid_payload.get("error_type") == "ValueError"
            )
            probes.append(
                MCPProbeResult(
                    probe_id="INPUT-001",
                    category="malformed_input",
                    description="Malformed bounded-tool input returns a structured error instead of crashing the server.",
                    passed=invalid_passed,
                    severity="high",
                    tool="simulate_packet",
                    expected="handled ValueError payload",
                    actual=(
                        f"error={invalid_payload.get('error')} error_type={invalid_payload.get('error_type')}"
                        if isinstance(invalid_payload, dict)
                        else "unexpected response"
                    ),
                    elapsed_ms=invalid_elapsed,
                    response=invalid_payload,
                )
            )

            synthetic_payload, synthetic_elapsed = await call_tool(
                "generate_synthetic_profiles",
                {
                    "num_profiles": 4,
                    "grid_points": 16,
                    "output_format": "csv",
                    "output_name": "mcp_probe_profiles",
                    "device": "cpu",
                },
            )
            synthetic_passed = False
            synthetic_actual = "unexpected response"
            if isinstance(synthetic_payload, dict):
                path = synthetic_payload.get("path")
                if isinstance(path, str):
                    try:
                        from spectral_packet_engine.table_io import load_profile_table
                        import numpy as np

                        table = load_profile_table(path)
                        synthetic_passed = bool(np.isfinite(table.profiles).all())
                        synthetic_actual = (
                            f"path={path} finite={synthetic_passed} "
                            f"num_profiles={table.profiles.shape[0]} num_positions={table.profiles.shape[1]}"
                        )
                    except Exception as exc:
                        synthetic_actual = f"load_profile_table failed: {type(exc).__name__}: {exc}"
                else:
                    synthetic_actual = "path missing from response"
            probes.append(
                MCPProbeResult(
                    probe_id="DATA-001",
                    category="data_compatibility",
                    description="Synthetic MCP profile generation writes loadable finite profile-table CSV data.",
                    passed=synthetic_passed,
                    severity="high",
                    tool="generate_synthetic_profiles",
                    expected="CSV can be loaded through load_profile_table and contains finite profile values",
                    actual=synthetic_actual,
                    elapsed_ms=synthetic_elapsed,
                    response=synthetic_payload,
                )
            )

            scratch_db_payload, scratch_db_elapsed = await call_tool("create_scratch_database", {"name": "../escape"})
            scratch_db_passed = (
                isinstance(scratch_db_payload, dict)
                and scratch_db_payload.get("error") is True
                and scratch_db_payload.get("error_type") == "ValueError"
            )
            probes.append(
                MCPProbeResult(
                    probe_id="SCRATCH-001",
                    category="path_containment",
                    description="Scratch database creation rejects path traversal outside the managed scratch directory.",
                    passed=scratch_db_passed,
                    severity="critical",
                    tool="create_scratch_database",
                    expected="handled ValueError payload",
                    actual=(
                        f"error={scratch_db_payload.get('error')} error_type={scratch_db_payload.get('error_type')}"
                        if isinstance(scratch_db_payload, dict)
                        else "unexpected response"
                    ),
                    elapsed_ms=scratch_db_elapsed,
                    response=scratch_db_payload,
                )
            )

            scratch_profiles_payload, scratch_profiles_elapsed = await call_tool(
                "generate_synthetic_profiles",
                {"output_name": "../escape_profiles", "num_profiles": 2, "grid_points": 8},
            )
            scratch_profiles_passed = (
                isinstance(scratch_profiles_payload, dict)
                and scratch_profiles_payload.get("error") is True
                and scratch_profiles_payload.get("error_type") == "ValueError"
            )
            probes.append(
                MCPProbeResult(
                    probe_id="SCRATCH-002",
                    category="path_containment",
                    description="Synthetic profile export rejects path traversal outside the managed scratch directory.",
                    passed=scratch_profiles_passed,
                    severity="critical",
                    tool="generate_synthetic_profiles",
                    expected="handled ValueError payload",
                    actual=(
                        f"error={scratch_profiles_payload.get('error')} error_type={scratch_profiles_payload.get('error_type')}"
                        if isinstance(scratch_profiles_payload, dict)
                        else "unexpected response"
                    ),
                    elapsed_ms=scratch_profiles_elapsed,
                    response=scratch_profiles_payload,
                )
            )

            query_guard_payload, query_guard_elapsed = await call_tool(
                "query_database",
                {"database": ":memory:", "query": "ATTACH DATABASE '/etc/passwd' AS stolen"},
            )
            query_guard_passed = (
                isinstance(query_guard_payload, dict)
                and query_guard_payload.get("error") is True
            )
            probes.append(
                MCPProbeResult(
                    probe_id="SQL-001",
                    category="query_guard",
                    description="SQLite query execution rejects ATTACH side effects when the MCP surface expects a read-only result set.",
                    passed=query_guard_passed,
                    severity="high",
                    tool="query_database",
                    expected="handled error payload",
                    actual=(
                        f"error={query_guard_payload.get('error')} error_type={query_guard_payload.get('error_type')}"
                        if isinstance(query_guard_payload, dict)
                        else "unexpected response"
                    ),
                    elapsed_ms=query_guard_elapsed,
                    response=query_guard_payload,
                )
            )

            tunnel_payload, tunnel_elapsed = await call_tool(
                "tunneling_experiment",
                {
                    "barrier_height": 30.0,
                    "barrier_width_sigma": 0.03,
                    "grid_points": 128,
                    "num_energies": 80,
                    "propagation_steps": 80,
                    "dt": 1e-5,
                    "device": "cpu",
                },
            )
            tunnel_passed = (
                isinstance(tunnel_payload, dict)
                and float(tunnel_payload.get("propagation", {}).get("norm_drift", 1.0)) <= 1e-6
                and float(tunnel_payload.get("propagation", {}).get("energy_drift", 1.0)) <= 1e-6
            )
            probes.append(
                MCPProbeResult(
                    probe_id="PHYSICS-001",
                    category="physics",
                    description="The tunneling pipeline remains numerically stable under MCP execution.",
                    passed=tunnel_passed,
                    severity="medium",
                    tool="tunneling_experiment",
                    expected="propagation norm_drift and energy_drift remain <= 1e-6",
                    actual=(
                        f"norm_drift={tunnel_payload.get('propagation', {}).get('norm_drift')} "
                        f"energy_drift={tunnel_payload.get('propagation', {}).get('energy_drift')}"
                        if isinstance(tunnel_payload, dict)
                        else "unexpected response"
                    ),
                    elapsed_ms=tunnel_elapsed,
                    response=tunnel_payload,
                )
            )

            if not skip_nested_probe:
                nested_probe_payload, nested_probe_elapsed = await call_tool("probe_mcp_runtime", {})
                nested_probe_passed = (
                    isinstance(nested_probe_payload, dict)
                    and isinstance(nested_probe_payload.get("summary"), dict)
                    and nested_probe_payload["summary"].get("failed_count") == 0
                )
                probes.append(
                    MCPProbeResult(
                        probe_id="SELF-001",
                        category="self_reference",
                        description="The MCP surface can call its own child-server probe workflow and receive a clean nested audit report.",
                        passed=nested_probe_passed,
                        severity="high",
                        tool="probe_mcp_runtime",
                        expected="nested summary.failed_count == 0",
                        actual=(
                            f"nested_failed_count={nested_probe_payload.get('summary', {}).get('failed_count')}"
                            if isinstance(nested_probe_payload, dict)
                            else "unexpected response"
                        ),
                        elapsed_ms=nested_probe_elapsed,
                        response=nested_probe_payload,
                    )
                )

            exec_payload, exec_elapsed = await call_tool(
                "execute_python",
                {"code": "import os\nresult={'unsafe_import_available': True, 'env_count': len(os.environ)}"},
            )
            exec_passed = False
            exec_actual = "unexpected response"
            exec_severity = "critical"
            if isinstance(exec_payload, dict):
                if expect_unsafe_python_enabled:
                    exec_passed = exec_payload.get("result", {}).get("unsafe_import_available") is True
                    exec_actual = f"unsafe_import_available={exec_payload.get('result', {}).get('unsafe_import_available')}"
                    exec_severity = "medium"
                else:
                    exec_passed = exec_payload.get("error") is True and exec_payload.get("error_type") == "PermissionError"
                    exec_actual = f"error_type={exec_payload.get('error_type')} message={exec_payload.get('error_message')}"
            probes.append(
                MCPProbeResult(
                    probe_id="EXEC-001",
                    category="capability_gate",
                    description="Arbitrary Python execution is either explicitly disabled or explicitly acknowledged as trusted-only.",
                    passed=exec_passed,
                    severity=exec_severity,
                    tool="execute_python",
                    expected=(
                        "PermissionError payload because unsafe execution is disabled"
                        if not expect_unsafe_python_enabled
                        else "trusted runtime allows explicit arbitrary execution"
                    ),
                    actual=exec_actual,
                    elapsed_ms=exec_elapsed,
                    response=exec_payload,
                )
            )

            if profile in {"stress", "audit"}:
                repeated_results: list[tuple[Any, float]] = []
                for _ in range(2):
                    repeated_results.append(
                        await call_tool(
                            "demo_spectral_pipeline",
                            {
                                "num_profiles": 12,
                                "grid_points": 64,
                                "num_modes": 12,
                                "device": "cpu",
                            },
                        )
                    )
                repeated_passed = all(
                    isinstance(payload, dict) and payload.get("pipeline") == "demo_spectral"
                    for payload, _ in repeated_results
                )
                probes.append(
                    MCPProbeResult(
                        probe_id="LOAD-001",
                        category="repeatability",
                        description="Repeated bounded demo workloads remain stable and return structured success payloads.",
                        passed=repeated_passed,
                        severity="medium",
                        tool="demo_spectral_pipeline",
                        expected="all repeated calls succeed",
                        actual=f"runs={len(repeated_results)} all_success={repeated_passed}",
                        elapsed_ms=sum(elapsed for _, elapsed in repeated_results),
                        response=[payload for payload, _ in repeated_results],
                    )
                )

                burst_requests = [
                    ("demo_spectral_pipeline", {"num_profiles": 10, "grid_points": 48, "num_modes": 8, "device": "cpu"}),
                    (
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
                    ),
                    ("simulate_packet", {"grid_points": 96, "num_modes": 48, "device": "cpu"}),
                ]
                burst_results = await asyncio.gather(
                    *(call_tool(name, arguments) for name, arguments in burst_requests)
                )
                burst_payloads = [payload for payload, _ in burst_results]
                structured_responses = all(isinstance(payload, dict) for payload in burst_payloads)
                saturation_count = sum(
                    1
                    for payload in burst_payloads
                    if isinstance(payload, dict)
                    and payload.get("error_type") == "RuntimeError"
                    and "saturated" in str(payload.get("error_message", "")).lower()
                )
                probes.append(
                    MCPProbeResult(
                        probe_id="LOAD-002",
                        category="burst_load",
                        description="A burst of bounded MCP workloads returns structured responses or structured saturation errors without transport collapse.",
                        passed=structured_responses,
                        severity="high",
                        tool=None,
                        expected="all burst requests produce structured payloads",
                        actual=f"responses={len(burst_payloads)} saturation_count={saturation_count}",
                        elapsed_ms=sum(elapsed for _, elapsed in burst_results),
                        response=burst_payloads,
                    )
                )

                limit_payload, limit_elapsed = await call_tool(
                    "generate_synthetic_profiles",
                    {"num_profiles": 1000000, "grid_points": 8},
                )
                limit_passed = (
                    isinstance(limit_payload, dict)
                    and limit_payload.get("error") is True
                    and limit_payload.get("error_type") == "ValueError"
                )
                probes.append(
                    MCPProbeResult(
                        probe_id="LIMIT-001",
                        category="resource_limits",
                        description="Synthetic-generation limits fail clearly under oversized adversarial requests.",
                        passed=limit_passed,
                        severity="high",
                        tool="generate_synthetic_profiles",
                        expected="handled ValueError payload",
                        actual=(
                            f"error={limit_payload.get('error')} error_type={limit_payload.get('error_type')}"
                            if isinstance(limit_payload, dict)
                            else "unexpected response"
                        ),
                        elapsed_ms=limit_elapsed,
                        response=limit_payload,
                    )
                )

    finished_at = _utc_now_iso()
    passed_count = sum(1 for probe in probes if probe.passed)
    failed = [probe for probe in probes if not probe.passed]
    summary = {
        "profile": profile,
        "probe_count": len(probes),
        "passed_count": passed_count,
        "failed_count": len(failed),
        "bugs_found": len(failed),
        "expect_unsafe_python_enabled": expect_unsafe_python_enabled,
        "high_severity_failures": [probe.probe_id for probe in failed if probe.severity in {"critical", "high"}],
        "summary_lines": [_probe_summary_line(probe.probe_id, probe.passed, probe.actual) for probe in probes],
    }
    return MCPProbeReport(
        probe_version=__version__,
        profile=profile,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        server=server,
        runtime_checks=tuple(runtime_checks),
        tool_calls=tuple(tool_calls),
        probes=tuple(probes),
        summary=summary,
    )


def _run_module_entrypoint_check(
    *,
    cwd: str,
    env: dict[str, str],
    python_executable: str,
) -> MCPRuntimeCheck:
    command = (python_executable, "-m", "spectral_packet_engine.mcp", "--help")
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout or "")[:4000]
    error = (completed.stderr or "")[:4000]
    return MCPRuntimeCheck(
        check_id="module-entrypoint-help",
        command=command,
        returncode=completed.returncode,
        passed=completed.returncode == 0 and "--max-concurrent-tasks" in output,
        stdout=output,
        stderr=error,
    )


def run_mcp_probe_suite(
    server: MCPProbeServerSpec,
    *,
    expect_unsafe_python_enabled: bool = False,
    profile: str = "smoke",
    skip_nested_probe: bool = False,
) -> MCPProbeReport:
    if importlib.util.find_spec("mcp.client.stdio") is None:
        raise ModuleNotFoundError("The MCP probe suite requires the 'mcp' extra.")
    import anyio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return anyio.run(_run_probe_suite_async, server, expect_unsafe_python_enabled, profile, skip_nested_probe)

    result: dict[str, MCPProbeReport] = {}
    error: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["report"] = anyio.run(
                _run_probe_suite_async,
                server,
                expect_unsafe_python_enabled,
                profile,
                skip_nested_probe,
            )
        except BaseException as exc:  # pragma: no cover - thread handoff only
            error["exc"] = exc

    thread = Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "exc" in error:
        raise error["exc"]
    return result["report"]


def write_mcp_probe_artifacts(report: MCPProbeReport, output_dir: str | Path) -> Path:
    return _write_mcp_probe_artifacts(
        output_dir,
        report,
        metadata={
            "probe_version": report.probe_version,
            "profile": report.profile,
            "transport": report.server.transport,
            "log_file": report.server.log_file,
        },
    )


__all__ = [
    "MCPProbeReport",
    "MCPProbeResult",
    "MCPProbeServerSpec",
    "MCPRuntimeCheck",
    "MCPToolCallRecord",
    "build_local_probe_server_spec",
    "run_mcp_probe_suite",
    "write_mcp_probe_artifacts",
]
