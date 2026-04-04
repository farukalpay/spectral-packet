from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from typing import Iterator
from uuid import uuid4


_TEMP_PREFIX = ".spectral-packet-engine-tmp-"
_LOCK_FILENAME = ".spectral-packet-engine.lock"
_LOCK_POLL_SECONDS = 0.05
_STALE_UNREADABLE_LOCK_SECONDS = 3600.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_runtime_temporary_path(path: str | Path) -> bool:
    name = Path(path).name
    return name.startswith(_TEMP_PREFIX) or name == _LOCK_FILENAME


def cleanup_runtime_temporary_files(path: str | Path) -> tuple[str, ...]:
    directory = Path(path)
    if not directory.exists():
        return ()
    removed: list[str] = []
    for item in sorted(directory.iterdir()):
        if item.is_file() and item.name.startswith(_TEMP_PREFIX):
            item.unlink(missing_ok=True)
            removed.append(item.name)
    return tuple(removed)


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_payload() -> dict[str, object]:
    return {
        "lock_id": uuid4().hex,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_at_utc": _utc_now_iso(),
    }


def _write_lock_file(path: Path, payload: dict[str, object]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
    finally:
        os.close(descriptor)


def _remove_stale_lock_if_safe(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        age_seconds = time.time() - path.stat().st_mtime
        if age_seconds >= _STALE_UNREADABLE_LOCK_SECONDS:
            path.unlink(missing_ok=True)
            return True
        return False

    hostname = payload.get("hostname")
    pid = payload.get("pid")
    if hostname != socket.gethostname():
        return False
    if not isinstance(pid, int):
        return False
    if _process_is_running(pid):
        return False
    path.unlink(missing_ok=True)
    return True


@contextmanager
def atomic_output_path(path: str | Path) -> Iterator[Path]:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{_TEMP_PREFIX}{uuid4().hex}-{target_path.name}")
    try:
        yield temp_path
        os.replace(temp_path, target_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    else:
        temp_path.unlink(missing_ok=True)


@contextmanager
def directory_lock(
    path: str | Path,
    *,
    timeout_seconds: float = 5.0,
) -> Iterator[Path]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / _LOCK_FILENAME
    payload = _lock_payload()
    deadline = time.monotonic() + timeout_seconds

    while True:
        try:
            _write_lock_file(lock_path, payload)
            break
        except FileExistsError:
            if _remove_stale_lock_if_safe(lock_path):
                continue
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"artifact directory is busy: {directory}. Another task is already writing there."
                )
            time.sleep(_LOCK_POLL_SECONDS)

    try:
        yield lock_path
    finally:
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            existing = None
        if existing is None or existing.get("lock_id") == payload["lock_id"]:
            lock_path.unlink(missing_ok=True)


__all__ = [
    "atomic_output_path",
    "cleanup_runtime_temporary_files",
    "directory_lock",
    "is_runtime_temporary_path",
]
