"""Tests for MCP security guards."""
from __future__ import annotations

import pytest
import sqlite3
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
from spectral_packet_engine.storage_economy import (
    ManagedSQLiteEconomy,
    StorageBudgetExceeded,
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


class TestManagedSQLiteEconomy:
    def test_restores_deleted_registered_database(self, tmp_path):
        database_path = tmp_path / "protected.sqlite"
        with sqlite3.connect(database_path) as connection:
            connection.execute("CREATE TABLE metrics (id INTEGER PRIMARY KEY, value REAL)")
            connection.execute("INSERT INTO metrics (value) VALUES (1.0)")
            connection.commit()

        economy = ManagedSQLiteEconomy(tmp_path / "_guard", minimum_seed_bytes=1024 * 1024)
        record = economy.register_database(database_path)

        assert record is not None
        database_path.unlink()
        assert not database_path.exists()

        restored = economy.restore_database_if_needed(database_path)

        assert restored == str(database_path.resolve())
        with sqlite3.connect(database_path) as connection:
            rows = connection.execute("SELECT value FROM metrics").fetchall()
        assert rows == [(1.0,)]

    def test_existing_database_starts_protected_without_spendable_budget(self, tmp_path):
        database_path = tmp_path / "protected.sqlite"
        with sqlite3.connect(database_path) as connection:
            connection.execute("CREATE TABLE metrics (id INTEGER PRIMARY KEY, value REAL)")
            connection.execute("INSERT INTO metrics (value) VALUES (1.0)")
            connection.commit()

        economy = ManagedSQLiteEconomy(tmp_path / "_guard", minimum_seed_bytes=1024 * 1024)
        economy.register_database(database_path)

        def _insert(candidate_database: str):
            with sqlite3.connect(candidate_database) as connection:
                connection.execute("INSERT INTO metrics (value) VALUES (2.0)")
                connection.commit()

        with pytest.raises(StorageBudgetExceeded):
            economy.mutate_sqlite(database_path, _insert)

        with sqlite3.connect(database_path) as connection:
            values = connection.execute("SELECT value FROM metrics ORDER BY id").fetchall()
        assert values == [(1.0,)]

    def test_new_database_can_bootstrap_from_seed_budget(self, tmp_path):
        database_path = tmp_path / "seeded.sqlite"
        economy = ManagedSQLiteEconomy(tmp_path / "_guard", minimum_seed_bytes=1024 * 1024)

        def _bootstrap(candidate_database: str) -> str:
            with sqlite3.connect(candidate_database) as connection:
                connection.execute("CREATE TABLE metrics (id INTEGER PRIMARY KEY, value REAL)")
                connection.commit()
            return "bootstrapped"

        result, receipt = economy.mutate_sqlite(
            database_path,
            _bootstrap,
        )

        assert result is not None
        assert database_path.exists()
        assert receipt.changed_bytes > 0
        assert receipt.balance_after_bytes < receipt.balance_before_bytes


class TestManagedDatabaseMCPIntegration:
    def test_write_scratch_file_rejects_database_filenames(self, tmp_path):
        pytest.importorskip("mcp.server.fastmcp")
        from spectral_packet_engine import MCPServerConfig, create_mcp_server

        async def _call():
            server = create_mcp_server(MCPServerConfig(scratch_directory=str(tmp_path)))
            _, payload = await server.call_tool(
                "write_scratch_file",
                {"filename": "clobber.db", "content": "not sqlite"},
            )
            return payload

        payload = __import__("asyncio").run(_call())
        assert payload["error"] is True
        assert payload["error_type"] == "PermissionError"

    def test_startup_restore_recovers_deleted_managed_database(self, tmp_path):
        pytest.importorskip("mcp.server.fastmcp")
        from spectral_packet_engine import MCPServerConfig, create_mcp_server

        async def _create_and_insert():
            server = create_mcp_server(
                MCPServerConfig(
                    scratch_directory=str(tmp_path),
                    storage_seed_bytes=1024 * 1024,
                )
            )
            _, create_payload = await server.call_tool(
                "create_scratch_database",
                {"name": "guarded.sqlite"},
            )
            database_path = create_payload["database_path"]
            _, script_payload = await server.call_tool(
                "execute_database_script",
                {
                    "database": database_path,
                    "script": """
                    CREATE TABLE metrics (id INTEGER PRIMARY KEY, value REAL);
                    INSERT INTO metrics (value) VALUES (1.0);
                    """,
                },
            )
            return database_path, script_payload

        database_path, script_payload = __import__("asyncio").run(_create_and_insert())
        assert script_payload.get("error") is not True
        Path(database_path).unlink()
        assert not Path(database_path).exists()

        async def _query_after_restart():
            server = create_mcp_server(
                MCPServerConfig(
                    scratch_directory=str(tmp_path),
                    storage_seed_bytes=1024 * 1024,
                )
            )
            _, payload = await server.call_tool(
                "query_database",
                {
                    "database": database_path,
                    "query": "SELECT value FROM metrics",
                },
            )
            return payload

        payload = __import__("asyncio").run(_query_after_restart())
        assert payload["table"]["row_count"] == 1
        assert payload["table"]["rows"][0]["value"] == 1.0
