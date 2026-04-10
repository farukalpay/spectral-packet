from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import sqlite3
from threading import Lock
import time
from typing import Any, Callable, Mapping
import uuid

from spectral_packet_engine.database import DatabaseConfig


_SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


class StorageEconomyError(RuntimeError):
    """Base class for managed storage economy failures."""


class StorageBudgetExceeded(StorageEconomyError):
    """Raised when a mutation exceeds the currently spendable budget."""


@dataclass(frozen=True, slots=True)
class SQLiteFootprint:
    path: str
    exists: bool
    file_bytes: int
    live_bytes: int
    page_size: int
    page_count: int
    freelist_count: int


@dataclass(frozen=True, slots=True)
class StorageMutationReceipt:
    path: str
    changed_bytes: int
    before_live_bytes: int
    after_live_bytes: int
    before_file_bytes: int
    after_file_bytes: int
    protected_bytes: int
    balance_before_bytes: float
    balance_after_bytes: float
    balance_capacity_bytes: int
    refill_rate_bytes_per_second: float
    latest_snapshot_path: str | None


@dataclass(frozen=True, slots=True)
class DatabaseProtectionRecord:
    path: str
    principal_bytes: int
    protected_bytes: int
    last_known_file_bytes: int
    last_known_live_bytes: int
    sealed_snapshot_path: str | None = None
    latest_snapshot_path: str | None = None
    last_restored_at: float | None = None
    last_committed_at: float | None = None


@dataclass(frozen=True, slots=True)
class StorageEconomySnapshot:
    guard_root: str
    protection_window_seconds: float
    minimum_seed_bytes: int
    minimum_mutation_cost_bytes: int
    protected_bytes: int
    spendable_balance_bytes: float
    balance_capacity_bytes: int
    refill_rate_bytes_per_second: float
    registered_databases: tuple[DatabaseProtectionRecord, ...]
    recovered_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "guard_root": self.guard_root,
            "protection_window_seconds": self.protection_window_seconds,
            "minimum_seed_bytes": self.minimum_seed_bytes,
            "minimum_mutation_cost_bytes": self.minimum_mutation_cost_bytes,
            "protected_bytes": self.protected_bytes,
            "spendable_balance_bytes": self.spendable_balance_bytes,
            "balance_capacity_bytes": self.balance_capacity_bytes,
            "refill_rate_bytes_per_second": self.refill_rate_bytes_per_second,
            "registered_databases": [asdict(item) for item in self.registered_databases],
            "recovered_paths": list(self.recovered_paths),
        }


@dataclass(slots=True)
class _DatabaseRecordState:
    path: str
    principal_bytes: int
    protected_bytes: int
    last_known_file_bytes: int
    last_known_live_bytes: int
    sealed_snapshot_path: str | None = None
    latest_snapshot_path: str | None = None
    last_restored_at: float | None = None
    last_committed_at: float | None = None

    def to_public(self) -> DatabaseProtectionRecord:
        return DatabaseProtectionRecord(
            path=self.path,
            principal_bytes=self.principal_bytes,
            protected_bytes=self.protected_bytes,
            last_known_file_bytes=self.last_known_file_bytes,
            last_known_live_bytes=self.last_known_live_bytes,
            sealed_snapshot_path=self.sealed_snapshot_path,
            latest_snapshot_path=self.latest_snapshot_path,
            last_restored_at=self.last_restored_at,
            last_committed_at=self.last_committed_at,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "principal_bytes": self.principal_bytes,
            "protected_bytes": self.protected_bytes,
            "last_known_file_bytes": self.last_known_file_bytes,
            "last_known_live_bytes": self.last_known_live_bytes,
            "sealed_snapshot_path": self.sealed_snapshot_path,
            "latest_snapshot_path": self.latest_snapshot_path,
            "last_restored_at": self.last_restored_at,
            "last_committed_at": self.last_committed_at,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> _DatabaseRecordState:
        return cls(
            path=str(payload["path"]),
            principal_bytes=int(payload.get("principal_bytes", 0)),
            protected_bytes=int(payload.get("protected_bytes", 0)),
            last_known_file_bytes=int(payload.get("last_known_file_bytes", 0)),
            last_known_live_bytes=int(payload.get("last_known_live_bytes", 0)),
            sealed_snapshot_path=payload.get("sealed_snapshot_path"),
            latest_snapshot_path=payload.get("latest_snapshot_path"),
            last_restored_at=payload.get("last_restored_at"),
            last_committed_at=payload.get("last_committed_at"),
        )


@dataclass(slots=True)
class _EconomyState:
    version: int = 1
    balance_bytes: float = 0.0
    last_refill_time: float = field(default_factory=time.time)
    database_records: dict[str, _DatabaseRecordState] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "balance_bytes": self.balance_bytes,
            "last_refill_time": self.last_refill_time,
            "database_records": {
                path: record.to_json()
                for path, record in sorted(self.database_records.items())
            },
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> _EconomyState:
        records_payload = payload.get("database_records", {})
        records = {
            str(path): _DatabaseRecordState.from_json(record)
            for path, record in records_payload.items()
        }
        return cls(
            version=int(payload.get("version", 1)),
            balance_bytes=float(payload.get("balance_bytes", 0.0)),
            last_refill_time=float(payload.get("last_refill_time", time.time())),
            database_records=records,
        )


def _path_token(path: Path) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, str(path)).hex


def _is_sqlite_path(path: Path) -> bool:
    return path.suffix.lower() in _SQLITE_SUFFIXES or path.name == ":memory:"


def _copy_sqlite_database(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if not source.exists():
        raise FileNotFoundError(source)
    try:
        with sqlite3.connect(str(source)) as src_connection:
            with sqlite3.connect(str(destination)) as dst_connection:
                src_connection.backup(dst_connection)
    except sqlite3.Error:
        destination.write_bytes(source.read_bytes())


def _sqlite_footprint(path: Path) -> SQLiteFootprint:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return SQLiteFootprint(
            path=str(resolved),
            exists=False,
            file_bytes=0,
            live_bytes=0,
            page_size=4096,
            page_count=0,
            freelist_count=0,
        )
    file_bytes = resolved.stat().st_size
    try:
        connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return SQLiteFootprint(
            path=str(resolved),
            exists=True,
            file_bytes=file_bytes,
            live_bytes=file_bytes,
            page_size=4096,
            page_count=max(math.ceil(file_bytes / 4096), 1) if file_bytes else 0,
            freelist_count=0,
        )
    with connection:
        page_size = int(connection.execute("PRAGMA page_size").fetchone()[0] or 4096)
        page_count = int(connection.execute("PRAGMA page_count").fetchone()[0] or 0)
        freelist_count = int(connection.execute("PRAGMA freelist_count").fetchone()[0] or 0)
    live_bytes = max((page_count - freelist_count) * page_size, 0)
    return SQLiteFootprint(
        path=str(resolved),
        exists=True,
        file_bytes=file_bytes,
        live_bytes=live_bytes,
        page_size=page_size,
        page_count=page_count,
        freelist_count=freelist_count,
    )


def _page_diff_bytes(before: Path | None, after: Path | None) -> int:
    before_exists = before is not None and before.exists()
    after_exists = after is not None and after.exists()
    if not before_exists and not after_exists:
        return 0
    if not before_exists:
        return after.stat().st_size if after is not None else 0
    if not after_exists:
        return before.stat().st_size if before is not None else 0

    before_footprint = _sqlite_footprint(before)
    after_footprint = _sqlite_footprint(after)
    page_size = max(before_footprint.page_size, after_footprint.page_size, 4096)

    changed_pages = 0
    with before.open("rb") as before_handle, after.open("rb") as after_handle:
        while True:
            before_chunk = before_handle.read(page_size)
            after_chunk = after_handle.read(page_size)
            if not before_chunk and not after_chunk:
                break
            if before_chunk != after_chunk:
                changed_pages += 1
    return changed_pages * page_size


def resolve_sqlite_database_path(reference: str | Path) -> Path | None:
    config = DatabaseConfig.from_reference(reference, create_if_missing=True)
    if not config.is_sqlite:
        return None
    sqlite_path = config.sqlite_path
    if sqlite_path is None:
        return None
    return sqlite_path.expanduser().resolve()


class ManagedSQLiteEconomy:
    """Persisted token-bucket guard for SQLite mutations plus snapshot recovery.

    The economy is global to the configured guard root. The sustainable mutation
    rate is proportional to the currently protected database mass, so reaching a
    database-size mutation volume again takes roughly one protection window.
    """

    def __init__(
        self,
        guard_root: Path,
        *,
        protection_window_seconds: float = 86_400.0,
        minimum_seed_bytes: int = 8 * 1024 * 1024,
        minimum_mutation_cost_bytes: int = 4096,
        snapshot_retention: int = 8,
    ) -> None:
        if protection_window_seconds <= 0:
            raise ValueError("protection_window_seconds must be positive")
        if minimum_seed_bytes < 0:
            raise ValueError("minimum_seed_bytes must be non-negative")
        if minimum_mutation_cost_bytes <= 0:
            raise ValueError("minimum_mutation_cost_bytes must be positive")
        if snapshot_retention <= 0:
            raise ValueError("snapshot_retention must be positive")
        self.guard_root = guard_root.expanduser().resolve()
        self.guard_root.mkdir(parents=True, exist_ok=True)
        self.snapshot_root = self.guard_root / "snapshots"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.guard_root / "state.json"
        self.protection_window_seconds = float(protection_window_seconds)
        self.minimum_seed_bytes = int(minimum_seed_bytes)
        self.minimum_mutation_cost_bytes = int(minimum_mutation_cost_bytes)
        self.snapshot_retention = int(snapshot_retention)
        self._lock = Lock()
        self._state = self._load_state()

    def _load_state(self) -> _EconomyState:
        if not self.state_path.exists():
            return _EconomyState(balance_bytes=float(self.minimum_seed_bytes))
        return _EconomyState.from_json(json.loads(self.state_path.read_text(encoding="utf-8")))

    def _write_state(self) -> None:
        self.state_path.write_text(
            json.dumps(self._state.to_json(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _protected_mass_bytes(self) -> int:
        protected = sum(
            max(record.protected_bytes, record.principal_bytes)
            for record in self._state.database_records.values()
        )
        if protected <= 0:
            return self.minimum_seed_bytes
        return protected

    def _refill_balance(self) -> tuple[float, int, float]:
        now = time.time()
        protected_bytes = self._protected_mass_bytes()
        refill_rate = protected_bytes / self.protection_window_seconds
        elapsed = max(now - self._state.last_refill_time, 0.0)
        capacity = protected_bytes
        balance = min(capacity, self._state.balance_bytes + elapsed * refill_rate)
        self._state.balance_bytes = balance
        self._state.last_refill_time = now
        return balance, capacity, refill_rate

    def _record_for(self, path: Path) -> _DatabaseRecordState | None:
        return self._state.database_records.get(str(path))

    def _record_snapshot_dir(self, path: Path) -> Path:
        target = self.snapshot_root / _path_token(path)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _register_database_unlocked(self, path: Path) -> _DatabaseRecordState:
        resolved = path.expanduser().resolve()
        existing = self._record_for(resolved)
        if existing is not None:
            return existing
        footprint = _sqlite_footprint(resolved)
        snapshot_dir = self._record_snapshot_dir(resolved)
        sealed_snapshot_path = None
        latest_snapshot_path = None
        if footprint.exists:
            sealed_snapshot = snapshot_dir / "sealed.sqlite"
            latest_snapshot = snapshot_dir / "latest.sqlite"
            _copy_sqlite_database(resolved, sealed_snapshot)
            _copy_sqlite_database(resolved, latest_snapshot)
            sealed_snapshot_path = str(sealed_snapshot)
            latest_snapshot_path = str(latest_snapshot)
        record = _DatabaseRecordState(
            path=str(resolved),
            principal_bytes=max(footprint.live_bytes, footprint.file_bytes),
            protected_bytes=max(footprint.live_bytes, footprint.file_bytes),
            last_known_file_bytes=footprint.file_bytes,
            last_known_live_bytes=footprint.live_bytes,
            sealed_snapshot_path=sealed_snapshot_path,
            latest_snapshot_path=latest_snapshot_path,
        )
        if self._state.database_records:
            self._state.balance_bytes = min(self._state.balance_bytes, float(self._protected_mass_bytes()))
        else:
            self._state.balance_bytes = 0.0 if footprint.exists else float(self.minimum_seed_bytes)
        self._state.database_records[str(resolved)] = record
        self._write_state()
        return record

    def register_database(self, reference: str | Path) -> DatabaseProtectionRecord | None:
        path = resolve_sqlite_database_path(reference)
        if path is None:
            return None
        with self._lock:
            record = self._register_database_unlocked(path)
            return record.to_public()

    def register_existing_databases(self, directory: Path) -> tuple[DatabaseProtectionRecord, ...]:
        directory = directory.expanduser().resolve()
        if not directory.exists():
            return ()
        registered: list[DatabaseProtectionRecord] = []
        with self._lock:
            for candidate in sorted(directory.rglob("*")):
                if not candidate.is_file():
                    continue
                try:
                    candidate.resolve().relative_to(self.guard_root)
                    continue
                except ValueError:
                    pass
                if _is_sqlite_path(candidate):
                    registered.append(self._register_database_unlocked(candidate).to_public())
        return tuple(registered)

    def _restore_record_unlocked(self, record: _DatabaseRecordState) -> bool:
        live_path = Path(record.path)
        if live_path.exists():
            return False
        source = None
        if record.latest_snapshot_path is not None and Path(record.latest_snapshot_path).exists():
            source = Path(record.latest_snapshot_path)
        elif record.sealed_snapshot_path is not None and Path(record.sealed_snapshot_path).exists():
            source = Path(record.sealed_snapshot_path)
        if source is None:
            return False
        live_path.parent.mkdir(parents=True, exist_ok=True)
        _copy_sqlite_database(source, live_path)
        footprint = _sqlite_footprint(live_path)
        record.last_known_file_bytes = footprint.file_bytes
        record.last_known_live_bytes = footprint.live_bytes
        record.last_restored_at = time.time()
        record.protected_bytes = max(record.protected_bytes, footprint.file_bytes, footprint.live_bytes)
        return True

    def restore_registered_databases(self) -> tuple[str, ...]:
        restored: list[str] = []
        with self._lock:
            for record in self._state.database_records.values():
                if self._restore_record_unlocked(record):
                    restored.append(record.path)
            if restored:
                self._write_state()
        return tuple(restored)

    def restore_database_if_needed(self, reference: str | Path) -> str | None:
        path = resolve_sqlite_database_path(reference)
        if path is None:
            return None
        with self._lock:
            record = self._record_for(path)
            if record is None:
                if not path.exists():
                    return None
                record = self._register_database_unlocked(path)
            if self._restore_record_unlocked(record):
                self._write_state()
                return record.path
        return None

    def _snapshot_latest_unlocked(self, path: Path, record: _DatabaseRecordState) -> str | None:
        if not path.exists():
            return record.latest_snapshot_path
        snapshot_dir = self._record_snapshot_dir(path)
        latest_snapshot = snapshot_dir / "latest.sqlite"
        _copy_sqlite_database(path, latest_snapshot)
        record.latest_snapshot_path = str(latest_snapshot)
        if record.sealed_snapshot_path is None:
            sealed_snapshot = snapshot_dir / "sealed.sqlite"
            _copy_sqlite_database(path, sealed_snapshot)
            record.sealed_snapshot_path = str(sealed_snapshot)
        history_name = time.strftime("history-%Y%m%dT%H%M%S", time.gmtime()) + ".sqlite"
        history_snapshot = snapshot_dir / history_name
        _copy_sqlite_database(path, history_snapshot)
        histories = sorted(
            p for p in snapshot_dir.glob("history-*.sqlite")
            if p.is_file()
        )
        while len(histories) > self.snapshot_retention:
            histories.pop(0).unlink(missing_ok=True)
        return record.latest_snapshot_path

    def inspect(self) -> StorageEconomySnapshot:
        with self._lock:
            balance, capacity, refill_rate = self._refill_balance()
            self._write_state()
            return StorageEconomySnapshot(
                guard_root=str(self.guard_root),
                protection_window_seconds=self.protection_window_seconds,
                minimum_seed_bytes=self.minimum_seed_bytes,
                minimum_mutation_cost_bytes=self.minimum_mutation_cost_bytes,
                protected_bytes=self._protected_mass_bytes(),
                spendable_balance_bytes=balance,
                balance_capacity_bytes=capacity,
                refill_rate_bytes_per_second=refill_rate,
                registered_databases=tuple(
                    record.to_public()
                    for _, record in sorted(self._state.database_records.items())
                ),
                recovered_paths=(),
            )

    def mutate_sqlite(
        self,
        reference: str | Path,
        mutator: Callable[[str], Any],
        *,
        create_if_missing: bool = True,
    ) -> tuple[Any, StorageMutationReceipt]:
        path = resolve_sqlite_database_path(reference)
        if path is None:
            raise StorageEconomyError("Managed SQLite protection only supports local SQLite database paths.")
        with self._lock:
            record = self._register_database_unlocked(path)
            self._restore_record_unlocked(record)
            if path.exists():
                before_footprint = _sqlite_footprint(path)
            else:
                before_footprint = SQLiteFootprint(
                    path=str(path),
                    exists=False,
                    file_bytes=0,
                    live_bytes=0,
                    page_size=4096,
                    page_count=0,
                    freelist_count=0,
                )
            balance_before, capacity, refill_rate = self._refill_balance()
            if not path.exists() and not create_if_missing:
                raise FileNotFoundError(path)

            candidate = path.with_name(f".{path.name}.candidate-{uuid.uuid4().hex}.sqlite")
            candidate.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                _copy_sqlite_database(path, candidate)
            result: Any = None
            try:
                result = mutator(str(candidate))
                after_footprint = _sqlite_footprint(candidate)
                changed_bytes = _page_diff_bytes(path if path.exists() else None, candidate)
                if changed_bytes > 0:
                    changed_bytes = max(changed_bytes, self.minimum_mutation_cost_bytes)
                if changed_bytes > balance_before:
                    raise StorageBudgetExceeded(
                        "Managed database mutation budget exceeded. "
                        f"Needed {changed_bytes:,} bytes but only {int(balance_before):,} bytes are currently spendable. "
                        f"The sustained refill rate is {refill_rate:.1f} bytes/s over a {self.protection_window_seconds:.0f}s protection window."
                    )
                if changed_bytes > 0:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self._state.balance_bytes = balance_before - changed_bytes
                    if path.exists():
                        self._snapshot_latest_unlocked(path, record)
                    candidate.replace(path)
                    latest_snapshot_path = self._snapshot_latest_unlocked(path, record)
                else:
                    latest_snapshot_path = record.latest_snapshot_path
                    if not path.exists() and candidate.exists():
                        candidate.unlink()
                current_footprint = _sqlite_footprint(path)
                record.protected_bytes = max(
                    record.protected_bytes,
                    record.principal_bytes,
                    current_footprint.file_bytes,
                    current_footprint.live_bytes,
                )
                record.last_known_file_bytes = current_footprint.file_bytes
                record.last_known_live_bytes = current_footprint.live_bytes
                record.last_committed_at = time.time()
                self._write_state()
            finally:
                candidate.unlink(missing_ok=True)
            receipt = StorageMutationReceipt(
                path=str(path),
                changed_bytes=changed_bytes,
                before_live_bytes=before_footprint.live_bytes,
                after_live_bytes=current_footprint.live_bytes,
                before_file_bytes=before_footprint.file_bytes,
                after_file_bytes=current_footprint.file_bytes,
                protected_bytes=self._protected_mass_bytes(),
                balance_before_bytes=balance_before,
                balance_after_bytes=self._state.balance_bytes,
                balance_capacity_bytes=capacity,
                refill_rate_bytes_per_second=refill_rate,
                latest_snapshot_path=latest_snapshot_path,
            )
            return result, receipt


__all__ = [
    "DatabaseProtectionRecord",
    "ManagedSQLiteEconomy",
    "SQLiteFootprint",
    "StorageBudgetExceeded",
    "StorageEconomyError",
    "StorageEconomySnapshot",
    "StorageMutationReceipt",
    "resolve_sqlite_database_path",
]
