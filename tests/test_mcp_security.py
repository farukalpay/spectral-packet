"""Tests for MCP security guards."""
from __future__ import annotations

import pytest
import tempfile
from pathlib import Path

from spectral_packet_engine.mcp_security import (
    WriteRateLimiter,
    check_scratch_file_count,
    check_scratch_total_size,
    check_sql_safety,
    check_sql_script_safety,
    backup_table_before_replace,
    sanitize_filename,
)


class TestWriteRateLimiter:
    def test_allows_within_limit(self):
        limiter = WriteRateLimiter(max_writes_per_minute=5)
        for _ in range(5):
            limiter.check("test")

    def test_blocks_over_limit(self):
        limiter = WriteRateLimiter(max_writes_per_minute=3)
        for _ in range(3):
            limiter.check("test")
        with pytest.raises(RuntimeError, match="Write rate limit"):
            limiter.check("test")

    def test_recent_count(self):
        limiter = WriteRateLimiter(max_writes_per_minute=10)
        limiter.check("a")
        limiter.check("b")
        assert limiter.recent_count == 2


class TestScratchFileCount:
    def test_allows_under_limit(self, tmp_path):
        for i in range(5):
            (tmp_path / f"file{i}.txt").write_text("x")
        check_scratch_file_count(tmp_path, max_files=10)

    def test_blocks_at_limit(self, tmp_path):
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text("x")
        with pytest.raises(RuntimeError, match="Scratch directory contains 10 files"):
            check_scratch_file_count(tmp_path, max_files=10)

    def test_ignores_internal_files(self, tmp_path):
        """Files starting with _ (like _audit.log) are not counted."""
        for i in range(5):
            (tmp_path / f"_internal{i}.log").write_text("x")
        check_scratch_file_count(tmp_path, max_files=5)


class TestScratchTotalSize:
    def test_allows_under_limit(self, tmp_path):
        (tmp_path / "small.txt").write_text("x" * 100)
        check_scratch_total_size(tmp_path, max_total_mb=1.0)

    def test_blocks_over_limit(self, tmp_path):
        (tmp_path / "big.bin").write_bytes(b"x" * (2 * 1024 * 1024))
        with pytest.raises(RuntimeError, match="total size"):
            check_scratch_total_size(tmp_path, max_total_mb=1.0)


class TestSQLSafety:
    def test_allows_select(self):
        check_sql_safety("SELECT * FROM users WHERE id = 1")

    def test_allows_insert(self):
        check_sql_safety("INSERT INTO users (name) VALUES ('test')")

    def test_allows_update_with_where(self):
        check_sql_safety("UPDATE users SET name = 'x' WHERE id = 1")

    def test_allows_delete_with_where(self):
        check_sql_safety("DELETE FROM users WHERE id = 1")

    def test_blocks_drop_table(self):
        with pytest.raises(PermissionError, match="DROP TABLE"):
            check_sql_safety("DROP TABLE users")

    def test_blocks_drop_table_case_insensitive(self):
        with pytest.raises(PermissionError, match="DROP TABLE"):
            check_sql_safety("drop table users")

    def test_blocks_drop_database(self):
        with pytest.raises(PermissionError, match="DROP DATABASE"):
            check_sql_safety("DROP DATABASE mydb")

    def test_blocks_truncate(self):
        with pytest.raises(PermissionError, match="TRUNCATE TABLE"):
            check_sql_safety("TRUNCATE TABLE users")

    def test_blocks_unbounded_delete(self):
        with pytest.raises(PermissionError, match="Unbounded DELETE"):
            check_sql_safety("DELETE FROM users")

    def test_blocks_unbounded_delete_semicolon(self):
        with pytest.raises(PermissionError, match="Unbounded DELETE"):
            check_sql_safety("DELETE FROM users;")

    def test_allows_destructive_when_opted_in(self):
        check_sql_safety("DROP TABLE users", allow_destructive=True)

    def test_script_safety(self):
        with pytest.raises(PermissionError, match="DROP TABLE"):
            check_sql_script_safety(
                "INSERT INTO t VALUES (1); DROP TABLE users; SELECT 1"
            )


class TestBackup:
    def test_creates_backup(self, tmp_path):
        db = tmp_path / "test.db"
        db.write_text("original data")
        result = backup_table_before_replace(db, "mytable", tmp_path)
        assert result is not None
        backup = Path(result)
        assert backup.exists()
        assert backup.read_text() == "original data"

    def test_returns_none_for_missing_db(self, tmp_path):
        result = backup_table_before_replace(tmp_path / "missing.db", "t", tmp_path)
        assert result is None

    def test_limits_backup_count(self, tmp_path):
        db = tmp_path / "test.db"
        db.write_text("data")
        # Create 6 backups, should keep only 5
        for i in range(6):
            backup_table_before_replace(db, "t", tmp_path)
        backups = list(tmp_path.glob("_backup_test_t_*.db"))
        assert len(backups) <= 5


class TestFilenameSanitization:
    def test_allows_normal_names(self):
        assert sanitize_filename("data.csv") == "data.csv"
        assert sanitize_filename("my-file_2.json") == "my-file_2.json"
        assert sanitize_filename("weather data (clean).csv") == "weather data (clean).csv"

    def test_blocks_path_traversal(self):
        with pytest.raises(ValueError, match="path traversal"):
            sanitize_filename("../etc/passwd")

    def test_blocks_absolute_path(self):
        with pytest.raises(ValueError, match="path traversal"):
            sanitize_filename("/etc/passwd")

    def test_blocks_empty(self):
        with pytest.raises(ValueError, match="empty"):
            sanitize_filename("")

    def test_blocks_control_chars(self):
        with pytest.raises(ValueError, match="control characters"):
            sanitize_filename("bad\nname.csv")
