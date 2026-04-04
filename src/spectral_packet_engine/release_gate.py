from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import numpy as np

from spectral_packet_engine.api import api_is_available
from spectral_packet_engine.artifacts import (
    to_serializable,
    write_compression_artifacts,
    write_modal_evaluation_artifacts,
)
from spectral_packet_engine.mcp import create_mcp_server, mcp_is_available
from spectral_packet_engine.ml import ModalSurrogateConfig
from spectral_packet_engine.service_status import ServiceStatusReport, inspect_service_status
from spectral_packet_engine.table_io import ProfileTable, save_profile_table_csv
from spectral_packet_engine.workflows import (
    EnvironmentReport,
    InstallationValidation,
    compress_profile_table,
    evaluate_modal_surrogate_from_database_query,
    simulate_gaussian_packet,
    validate_installation,
    write_profile_table_to_database,
)

ReleaseGateStatus = Literal["passed", "skipped", "failed"]


@dataclass(frozen=True, slots=True)
class ReleaseGateCheckResult:
    name: str
    surface: str
    required: bool
    status: ReleaseGateStatus
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "surface": self.surface,
            "required": self.required,
            "status": self.status,
            "detail": self.detail,
            "metrics": dict(self.metrics),
            "artifacts": list(self.artifacts),
        }


@dataclass(frozen=True, slots=True)
class ReleaseGateValidation:
    environment: EnvironmentReport
    installation: InstallationValidation
    service_status: ServiceStatusReport
    checks: tuple[ReleaseGateCheckResult, ...]
    release_ready_current_environment: bool
    validated_surfaces: tuple[str, ...]
    skipped_surfaces: tuple[str, ...]
    blocked_surfaces: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": to_serializable(asdict(self.environment)),
            "installation": to_serializable(asdict(self.installation)),
            "service_status": self.service_status.to_dict(),
            "checks": [item.to_dict() for item in self.checks],
            "release_ready_current_environment": self.release_ready_current_environment,
            "validated_surfaces": list(self.validated_surfaces),
            "skipped_surfaces": list(self.skipped_surfaces),
            "blocked_surfaces": list(self.blocked_surfaces),
            "notes": list(self.notes),
        }


def _artifact_names(directory: Path) -> tuple[str, ...]:
    return tuple(sorted(path.name for path in directory.iterdir() if path.is_file()))


def _synthetic_profile_table(*, samples: int = 8, positions: int = 20) -> ProfileTable:
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
        source="release-gate-synthetic",
    )


def _python_core_check(device: str) -> ReleaseGateCheckResult:
    with TemporaryDirectory(prefix="spectral_packet_engine_release_gate_python_") as directory:
        output_dir = Path(directory) / "python_artifacts"
        forward = simulate_gaussian_packet(
            center=0.30,
            width=0.07,
            wavenumber=25.0,
            times=[0.0, 1e-3, 3e-3, 5e-3],
            num_modes=64,
            quadrature_points=1024,
            grid_points=64,
            device=device,
        )
        table = ProfileTable(
            position_grid=forward.grid.detach().cpu().numpy(),
            sample_times=forward.times.detach().cpu().numpy(),
            profiles=forward.densities.detach().cpu().numpy(),
            source="release-gate-python-core",
        )
        summary = compress_profile_table(table, num_modes=12, device=device)
        mean_error = float(summary.error_summary.mean_relative_l2_error)
        if not np.isfinite(mean_error):
            raise RuntimeError("Python core release-gate path produced a non-finite reconstruction error.")
        if mean_error >= 0.25:
            raise RuntimeError(
                f"Python core release-gate path exceeded the mean relative L2 error budget: {mean_error:.6f} >= 0.250000."
            )
        write_compression_artifacts(output_dir, summary)
        return ReleaseGateCheckResult(
            name="python_core",
            surface="python-core",
            required=True,
            status="passed",
            detail="Simulated a bounded-domain packet and compressed the resulting profile table into modal artifacts.",
            metrics={
                "num_samples": int(table.profiles.shape[0]),
                "num_positions": int(table.profiles.shape[1]),
                "num_modes": int(summary.num_modes),
                "mean_relative_l2_error": mean_error,
            },
            artifacts=_artifact_names(output_dir),
        )


def _sql_backend_check(device: str) -> ReleaseGateCheckResult:
    with TemporaryDirectory(prefix="spectral_packet_engine_release_gate_sql_") as directory:
        database_path = Path(directory) / "profiles.sqlite"
        output_dir = Path(directory) / "sql_backend_artifacts"
        table = _synthetic_profile_table()
        write_profile_table_to_database(database_path, "profiles", table, if_exists="replace")
        summary = evaluate_modal_surrogate_from_database_query(
            database_path,
            'SELECT * FROM "profiles" ORDER BY time',
            backend="auto",
            num_modes=6,
            config=ModalSurrogateConfig(
                epochs=4,
                batch_size=2,
                device=device,
            ),
        )
        mean_error = float(summary.comparison.mean_relative_l2_error)
        if not np.isfinite(mean_error):
            raise RuntimeError("SQL/backend release-gate path produced a non-finite evaluation error.")
        if mean_error >= 0.55:
            raise RuntimeError(
                f"SQL/backend release-gate path exceeded the mean relative L2 error budget: {mean_error:.6f} >= 0.550000."
            )
        write_modal_evaluation_artifacts(output_dir, summary)
        return ReleaseGateCheckResult(
            name="sql_backend_workflow",
            surface="sql-backend",
            required=True,
            status="passed",
            detail="Materialized a profile table into SQLite and ran backend-aware modal evaluation through the shared SQL workflow.",
            metrics={
                "backend": summary.backend,
                "num_modes": int(summary.num_modes),
                "row_count": int(table.profiles.shape[0]),
                "mean_relative_l2_error": mean_error,
            },
            artifacts=_artifact_names(output_dir),
        )


def _api_check(device: str) -> ReleaseGateCheckResult:
    if not api_is_available():
        return ReleaseGateCheckResult(
            name="api_workflow",
            surface="api",
            required=False,
            status="skipped",
            detail="FastAPI is not installed or not compatible in this environment.",
        )

    from fastapi.testclient import TestClient
    from spectral_packet_engine.api import create_api_app

    with TemporaryDirectory(prefix="spectral_packet_engine_release_gate_api_") as directory:
        client = TestClient(create_api_app())
        table = _synthetic_profile_table(samples=5, positions=18)
        output_dir = Path(directory) / "api_artifacts"
        response = client.post(
            "/profiles/compress",
            json={
                "table": table.to_dict(),
                "num_modes": 8,
                "device": device,
                "output_dir": str(output_dir),
            },
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"API release-gate path failed with status {response.status_code}: {response.text}"
            )
        payload = response.json()
        status = client.get("/status")
        if status.status_code != 200:
            raise RuntimeError(
                f"API status check failed with status {status.status_code}: {status.text}"
            )
        recent_tasks = status.json()["recent_tasks"]
        if not any(task.get("workflow_id") == "compress-profile-table" for task in recent_tasks):
            raise RuntimeError(
                "API release-gate path completed without recording the compress-profile-table workflow in service status."
            )
        return ReleaseGateCheckResult(
            name="api_workflow",
            surface="api",
            required=True,
            status="passed",
            detail="Compressed a profile table through the FastAPI surface and verified runtime status reporting.",
            metrics={
                "num_modes": int(payload["num_modes"]),
                "recent_task_count": len(recent_tasks),
            },
            artifacts=_artifact_names(output_dir),
        )


def _mcp_check(device: str) -> ReleaseGateCheckResult:
    if not mcp_is_available():
        return ReleaseGateCheckResult(
            name="mcp_workflow",
            surface="mcp",
            required=False,
            status="skipped",
            detail="The MCP runtime is not installed in this environment.",
        )

    with TemporaryDirectory(prefix="spectral_packet_engine_release_gate_mcp_") as directory:
        table = _synthetic_profile_table(samples=5, positions=18)
        csv_path = Path(directory) / "profiles.csv"
        output_dir = Path(directory) / "mcp_artifacts"
        save_profile_table_csv(table, csv_path)

        async def _exercise() -> tuple[dict[str, Any], dict[str, Any]]:
            server = create_mcp_server()
            _, payload = await server.call_tool(
                "compress_profile_table",
                {
                    "table_path": str(csv_path),
                    "num_modes": 8,
                    "device": device,
                    "output_dir": str(output_dir),
                },
            )
            _, status_payload = await server.call_tool("inspect_service_status", {})
            return payload, status_payload

        payload, status_payload = asyncio.run(_exercise())
        if not any(task.get("workflow_id") == "compress-profile-table" for task in status_payload["recent_tasks"]):
            raise RuntimeError(
                "MCP release-gate path completed without recording the compress-profile-table workflow in service status."
            )
        return ReleaseGateCheckResult(
            name="mcp_workflow",
            surface="mcp",
            required=True,
            status="passed",
            detail="Compressed a profile table through MCP and verified service-status visibility.",
            metrics={
                "num_modes": int(payload["num_modes"]),
                "recent_task_count": len(status_payload["recent_tasks"]),
            },
            artifacts=_artifact_names(output_dir),
        )


def run_release_gate(
    *,
    device: str = "auto",
    include_api: bool = True,
    include_mcp: bool = True,
) -> ReleaseGateValidation:
    installation = validate_installation(device)
    environment = installation.environment
    checks: list[ReleaseGateCheckResult] = []

    def _run_check(fn) -> None:
        try:
            checks.append(fn(device))
        except Exception as exc:
            failed = fn.__name__.removeprefix("_").removesuffix("_check")
            surface = failed.replace("_", "-")
            checks.append(
                ReleaseGateCheckResult(
                    name=failed,
                    surface=surface,
                    required=True,
                    status="failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )

    _run_check(_python_core_check)
    _run_check(_sql_backend_check)

    if include_api:
        try:
            api_check = _api_check(device)
        except Exception as exc:
            api_check = ReleaseGateCheckResult(
                name="api_workflow",
                surface="api",
                required=api_is_available(),
                status="failed",
                detail=f"{type(exc).__name__}: {exc}",
            )
        checks.append(api_check)
    else:
        checks.append(
            ReleaseGateCheckResult(
                name="api_workflow",
                surface="api",
                required=False,
                status="skipped",
                detail="API validation was skipped by request.",
            )
        )

    if include_mcp:
        try:
            mcp_check = _mcp_check(device)
        except Exception as exc:
            mcp_check = ReleaseGateCheckResult(
                name="mcp_workflow",
                surface="mcp",
                required=mcp_is_available(),
                status="failed",
                detail=f"{type(exc).__name__}: {exc}",
            )
        checks.append(mcp_check)
    else:
        checks.append(
            ReleaseGateCheckResult(
                name="mcp_workflow",
                surface="mcp",
                required=False,
                status="skipped",
                detail="MCP validation was skipped by request.",
            )
        )

    validated_surfaces = tuple(check.surface for check in checks if check.status == "passed")
    skipped_surfaces = tuple(check.surface for check in checks if check.status == "skipped")
    blocked_surfaces = tuple(check.surface for check in checks if check.status == "failed")
    release_ready_current_environment = not blocked_surfaces and all(
        check.status == "passed" for check in checks if check.required
    )
    notes = (
        "This release gate validates the current environment in-process; it does not replace isolated install, build, or cross-platform validation.",
        "Use the internal release-gate matrix to decide whether clean Linux, Windows, macOS, and TensorFlow-supported Python environments have also been validated.",
    )

    return ReleaseGateValidation(
        environment=environment,
        installation=installation,
        service_status=inspect_service_status(),
        checks=tuple(checks),
        release_ready_current_environment=release_ready_current_environment,
        validated_surfaces=validated_surfaces,
        skipped_surfaces=skipped_surfaces,
        blocked_surfaces=blocked_surfaces,
        notes=notes,
    )


__all__ = [
    "ReleaseGateCheckResult",
    "ReleaseGateValidation",
    "run_release_gate",
]
