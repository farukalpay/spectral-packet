from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import socket
import sys
from threading import RLock
from time import perf_counter
from typing import Any, Iterator, Literal, Mapping
from uuid import uuid4

from spectral_packet_engine.version import __version__


_LOGGER = logging.getLogger("spectral_packet_engine.service")
if not _LOGGER.handlers:
    _LOGGER.addHandler(logging.NullHandler())
_LOGGER.propagate = False

_MAX_RECENT_TASKS = 32

TaskStatus = Literal["running", "completed", "failed"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _coerce_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return {} if metadata is None else {str(key): value for key, value in metadata.items()}


def _coerce_optional_token(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def configure_service_logging(
    level: str = "WARNING",
    *,
    log_file: str | Path | None = None,
    force: bool = False,
) -> None:
    normalized_level = str(level).strip().upper()
    if normalized_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("service log level must be one of DEBUG, INFO, WARNING, or ERROR")

    if force:
        for handler in list(_LOGGER.handlers):
            _LOGGER.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                _LOGGER.debug("Failed to close logging handler %r during forced reconfigure", handler)
    else:
        managed_handlers = [
            handler
            for handler in _LOGGER.handlers
            if getattr(handler, "_spectral_packet_engine_managed", False)
        ]
        for handler in managed_handlers:
            _LOGGER.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                _LOGGER.debug("Failed to close managed logging handler %r", handler)

    fallback_warning: tuple[Path, OSError] | None = None
    if log_file is None:
        handler = logging.StreamHandler(sys.stderr)
    else:
        log_path = Path(log_file)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_path, encoding="utf-8")
        except OSError as exc:
            handler = logging.StreamHandler(sys.stderr)
            fallback_warning = (log_path, exc)
    handler._spectral_packet_engine_managed = True  # type: ignore[attr-defined]
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(getattr(logging, normalized_level))
    if fallback_warning is not None:
        log_path, error = fallback_warning
        _LOGGER.warning(
            "Falling back to stderr because service log file %s could not be opened: %s",
            log_path,
            error,
        )


def _log_service_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    _LOGGER.log(level, "%s %s", event, json.dumps(fields, sort_keys=True, default=str))


@dataclass(frozen=True, slots=True)
class ServiceTaskCounters:
    active_task_count: int
    total_started: int
    total_completed: int
    total_failed: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ServiceTaskStatusRecord:
    task_id: str
    name: str
    interface: str | None
    workflow_id: str | None
    surface_action: str | None
    status: TaskStatus
    started_at_utc: str
    finished_at_utc: str | None
    duration_seconds: float | None
    error: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ServiceStatusReport:
    service_name: str
    version: str
    process_id: int
    hostname: str
    system: str
    machine: str
    python_version: str
    started_at_utc: str
    uptime_seconds: float
    cpu_count: int | None
    counters: ServiceTaskCounters
    active_tasks: tuple[ServiceTaskStatusRecord, ...]
    recent_tasks: tuple[ServiceTaskStatusRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "version": self.version,
            "process_id": self.process_id,
            "hostname": self.hostname,
            "system": self.system,
            "machine": self.machine,
            "python_version": self.python_version,
            "started_at_utc": self.started_at_utc,
            "uptime_seconds": self.uptime_seconds,
            "cpu_count": self.cpu_count,
            "counters": self.counters.to_dict(),
            "active_tasks": [item.to_dict() for item in self.active_tasks],
            "recent_tasks": [item.to_dict() for item in self.recent_tasks],
        }


class _ServiceTaskRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._started_at = _utc_now()
        self._started_monotonic = perf_counter()
        self._active: dict[str, dict[str, Any]] = {}
        self._recent: deque[ServiceTaskStatusRecord] = deque(maxlen=_MAX_RECENT_TASKS)
        self._total_started = 0
        self._total_completed = 0
        self._total_failed = 0

    def _finalize_task(
        self,
        task_id: str,
        *,
        status: TaskStatus,
        error: str | None = None,
    ) -> ServiceTaskStatusRecord | None:
        if status not in {"completed", "failed"}:
            raise ValueError("status must be 'completed' or 'failed'")
        with self._lock:
            active_record = self._active.pop(task_id, None)
            if active_record is None:
                return None
            started_monotonic = float(active_record.pop("_started_monotonic"))
            summary = ServiceTaskStatusRecord(
                task_id=task_id,
                name=str(active_record["name"]),
                interface=active_record["interface"],
                workflow_id=active_record["workflow_id"],
                surface_action=active_record["surface_action"],
                status=status,
                started_at_utc=str(active_record["started_at_utc"]),
                finished_at_utc=_isoformat_utc(_utc_now()),
                duration_seconds=float(perf_counter() - started_monotonic),
                error=error,
                metadata=dict(active_record["metadata"]),
            )
            self._recent.appendleft(summary)
            if status == "completed":
                self._total_completed += 1
            else:
                self._total_failed += 1
        _log_service_event(
            f"service_task_{status}",
            level=logging.INFO if status == "completed" else logging.ERROR,
            task_id=task_id,
            name=summary.name,
            interface=summary.interface,
            workflow_id=summary.workflow_id,
            surface_action=summary.surface_action,
            duration_seconds=summary.duration_seconds,
            error=summary.error,
        )
        return summary

    def mark_failed(self, task_id: str, exc: Exception) -> ServiceTaskStatusRecord | None:
        return self._finalize_task(
            task_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )

    @contextmanager
    def track(
        self,
        name: str,
        *,
        interface: str | None = None,
        workflow_id: str | None = None,
        surface_action: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[str]:
        task_id = uuid4().hex
        started_at = _utc_now()
        started_monotonic = perf_counter()
        normalized_metadata = _coerce_metadata(metadata)
        normalized_workflow_id = _coerce_optional_token(workflow_id)
        normalized_surface_action = _coerce_optional_token(surface_action)
        active_record = {
            "task_id": task_id,
            "name": str(name),
            "interface": None if interface is None else str(interface),
            "workflow_id": normalized_workflow_id,
            "surface_action": normalized_surface_action,
            "status": "running",
            "started_at_utc": _isoformat_utc(started_at),
            "finished_at_utc": None,
            "duration_seconds": None,
            "error": None,
            "metadata": normalized_metadata,
            "_started_monotonic": started_monotonic,
        }
        with self._lock:
            self._active[task_id] = active_record
            self._total_started += 1
        _log_service_event(
            "service_task_started",
            level=logging.INFO,
            task_id=task_id,
            name=name,
            interface=interface,
            workflow_id=normalized_workflow_id,
            surface_action=normalized_surface_action,
            metadata=normalized_metadata,
        )
        try:
            yield task_id
        except Exception as exc:
            self._finalize_task(
                task_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        else:
            self._finalize_task(task_id, status="completed")

    def snapshot(self, *, recent_limit: int = 10) -> ServiceStatusReport:
        limit = max(0, int(recent_limit))
        with self._lock:
            active_tasks = tuple(
                ServiceTaskStatusRecord(
                    task_id=record["task_id"],
                    name=record["name"],
                    interface=record["interface"],
                    workflow_id=record["workflow_id"],
                    surface_action=record["surface_action"],
                    status=record["status"],
                    started_at_utc=record["started_at_utc"],
                    finished_at_utc=record["finished_at_utc"],
                    duration_seconds=record["duration_seconds"],
                    error=record["error"],
                    metadata=record["metadata"],
                )
                for record in sorted(self._active.values(), key=lambda item: item["started_at_utc"])
            )
            recent_tasks = tuple(list(self._recent)[:limit])
            counters = ServiceTaskCounters(
                active_task_count=len(active_tasks),
                total_started=self._total_started,
                total_completed=self._total_completed,
                total_failed=self._total_failed,
            )
        return ServiceStatusReport(
            service_name="Spectral Packet Engine",
            version=__version__,
            process_id=os.getpid(),
            hostname=socket.gethostname(),
            system=platform.system(),
            machine=platform.machine().lower(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            started_at_utc=_isoformat_utc(self._started_at),
            uptime_seconds=float(perf_counter() - self._started_monotonic),
            cpu_count=os.cpu_count(),
            counters=counters,
            active_tasks=active_tasks,
            recent_tasks=recent_tasks,
        )


_REGISTRY = _ServiceTaskRegistry()


def inspect_service_status(*, recent_limit: int = 10) -> ServiceStatusReport:
    return _REGISTRY.snapshot(recent_limit=recent_limit)


@contextmanager
def track_service_task(
    name: str,
    *,
    interface: str | None = None,
    workflow_id: str | None = None,
    surface_action: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[str]:
    with _REGISTRY.track(
        name,
        interface=interface,
        workflow_id=workflow_id,
        surface_action=surface_action,
        metadata=metadata,
    ) as task_id:
        yield task_id


def mark_service_task_failed(task_id: str, exc: Exception) -> None:
    _REGISTRY.mark_failed(task_id, exc)


__all__ = [
    "ServiceStatusReport",
    "ServiceTaskCounters",
    "ServiceTaskStatusRecord",
    "configure_service_logging",
    "inspect_service_status",
    "mark_service_task_failed",
    "track_service_task",
]
