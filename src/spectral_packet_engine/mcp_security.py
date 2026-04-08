"""MCP security guards: rate limits, SQL guards, scratch limits, and backups.

Threat model
------------
An attacker (or a misbehaving automated caller) with MCP tool access can:

1. **Spam writes** — flood write_scratch_file / upload_csv_to_database to
   exhaust disk or hit rate limits only after the damage is done.
2. **Overwrite tables** — ``if_exists="replace"`` silently wipes data.
3. **Destructive SQL** — ``DROP TABLE``, ``DELETE FROM``, ``TRUNCATE``
   issued through execute_database_statement / execute_database_script.
4. **File bomb** — create thousands of small scratch files until the
   filesystem is saturated.

Guards implemented here
-----------------------
- **Write rate limiter**: separate, stricter per-minute limit for mutating
  tool calls (write, delete, execute, upload).
- **Scratch file count limit**: caps total files in the scratch directory.
- **Scratch directory total size limit**: caps cumulative bytes.
- **Destructive SQL guard**: blocks DROP TABLE/DATABASE, TRUNCATE, and
  unbounded DELETE (without WHERE) unless explicitly opted in.
- **Pre-mutation table backup**: automatic .bak snapshot before table
  replacement operations.
"""

from __future__ import annotations

import collections
import re
import shutil
import time as _time
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spectral_packet_engine.mcp_runtime import MCPServerConfig


# ---------------------------------------------------------------------------
# Write rate limiter
# ---------------------------------------------------------------------------

class WriteRateLimiter:
    """Sliding-window rate limiter specifically for mutating operations."""

    def __init__(self, max_writes_per_minute: int = 30) -> None:
        self._max = max_writes_per_minute
        self._timestamps: collections.deque[float] = collections.deque()
        self._lock = Lock()

    def check(self, operation: str = "write") -> None:
        if self._max <= 0:
            return
        now = _time.monotonic()
        window_start = now - 60.0
        with self._lock:
            while self._timestamps and self._timestamps[0] < window_start:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._max:
                raise RuntimeError(
                    f"Write rate limit exceeded: {self._max} mutating operations/minute. "
                    f"Operation '{operation}' blocked. Wait before retrying."
                )
            self._timestamps.append(now)

    @property
    def recent_count(self) -> int:
        now = _time.monotonic()
        window_start = now - 60.0
        with self._lock:
            while self._timestamps and self._timestamps[0] < window_start:
                self._timestamps.popleft()
            return len(self._timestamps)


# ---------------------------------------------------------------------------
# Scratch directory guards
# ---------------------------------------------------------------------------

def check_scratch_file_count(
    scratch_dir: Path,
    max_files: int = 200,
) -> None:
    """Reject if too many files already exist in the scratch directory."""
    if not scratch_dir.exists():
        return
    count = sum(1 for f in scratch_dir.iterdir() if f.is_file() and not f.name.startswith("_"))
    if count >= max_files:
        raise RuntimeError(
            f"Scratch directory contains {count} files (limit: {max_files}). "
            f"Delete unused files with delete_scratch_file before creating new ones."
        )


def check_scratch_total_size(
    scratch_dir: Path,
    max_total_mb: float = 512.0,
) -> None:
    """Reject if scratch directory exceeds cumulative size limit."""
    if not scratch_dir.exists():
        return
    total_bytes = sum(f.stat().st_size for f in scratch_dir.rglob("*") if f.is_file())
    total_mb = total_bytes / (1024 * 1024)
    if total_mb >= max_total_mb:
        raise RuntimeError(
            f"Scratch directory total size is {total_mb:.1f} MB (limit: {max_total_mb:.0f} MB). "
            f"Delete unused files or databases to free space."
        )


# ---------------------------------------------------------------------------
# Destructive SQL guard
# ---------------------------------------------------------------------------

# Patterns that indicate destructive/dangerous operations
_DESTRUCTIVE_PATTERNS = [
    (re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE), "DROP TABLE"),
    (re.compile(r"\bDROP\s+DATABASE\b", re.IGNORECASE), "DROP DATABASE"),
    (re.compile(r"\bTRUNCATE\s+TABLE\b", re.IGNORECASE), "TRUNCATE TABLE"),
    (re.compile(r"\bALTER\s+TABLE\s+\w+\s+DROP\b", re.IGNORECASE), "ALTER TABLE ... DROP"),
]

# DELETE without WHERE — too broad, likely accidental
_UNBOUNDED_DELETE = re.compile(
    r"\bDELETE\s+FROM\s+\w+\s*(?:;|\Z)",
    re.IGNORECASE,
)


def check_sql_safety(
    sql: str,
    *,
    allow_destructive: bool = False,
) -> None:
    """Block destructive SQL unless explicitly allowed.

    Raises
    ------
    PermissionError
        If the SQL contains a destructive pattern and ``allow_destructive``
        is False.
    """
    if allow_destructive:
        return

    for pattern, label in _DESTRUCTIVE_PATTERNS:
        if pattern.search(sql):
            raise PermissionError(
                f"Destructive SQL operation detected: {label}. "
                f"This operation is blocked by default. To proceed, "
                f"the server must be configured with allow_destructive_sql=True."
            )

    if _UNBOUNDED_DELETE.search(sql):
        raise PermissionError(
            "Unbounded DELETE (without WHERE clause) detected. "
            "This would delete all rows from the table. Add a WHERE clause "
            "or use allow_destructive_sql=True to override."
        )


def check_sql_script_safety(
    script: str,
    *,
    allow_destructive: bool = False,
) -> None:
    """Check each statement in a multi-statement script."""
    for statement in script.split(";"):
        stripped = statement.strip()
        if stripped:
            check_sql_safety(stripped, allow_destructive=allow_destructive)


# ---------------------------------------------------------------------------
# Pre-mutation table backup
# ---------------------------------------------------------------------------

def backup_table_before_replace(
    db_path: str | Path,
    table_name: str,
    scratch_dir: Path,
) -> str | None:
    """Create a timestamped .bak backup of the database before table replacement.

    Returns the backup path, or None if the database doesn't exist yet.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return None

    ts = _time.strftime("%Y%m%dT%H%M%S", _time.gmtime())
    backup_name = f"_backup_{db_path.stem}_{table_name}_{ts}.db"
    backup_path = scratch_dir / backup_name

    # Only keep last 5 backups per database to prevent accumulation
    existing_backups = sorted(
        scratch_dir.glob(f"_backup_{db_path.stem}_{table_name}_*.db"),
        key=lambda p: p.stat().st_mtime,
    )
    while len(existing_backups) >= 5:
        oldest = existing_backups.pop(0)
        oldest.unlink(missing_ok=True)

    shutil.copy2(db_path, backup_path)
    return str(backup_path)


# ---------------------------------------------------------------------------
# Filename sanitization
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """Ensure a scratch filename stays inside the managed directory.

    This guard prevents path traversal without rejecting ordinary human file
    names such as ``weather data (clean).csv``.
    """

    token = str(name).strip()
    if not token:
        raise ValueError("Filename must not be empty")
    candidate = Path(token)
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name in {".", ".."}:
        raise ValueError(f"Filename contains path traversal characters: {token!r}")
    if any(ord(character) < 32 for character in candidate.name):
        raise ValueError(
            f"Filename contains control characters: {token!r}. "
            f"Use a normal printable filename."
        )
    return candidate.name


__all__ = [
    "WriteRateLimiter",
    "backup_table_before_replace",
    "check_scratch_file_count",
    "check_scratch_total_size",
    "check_sql_safety",
    "check_sql_script_safety",
    "sanitize_filename",
]
