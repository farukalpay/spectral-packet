from __future__ import annotations

import pytest

from spectral_packet_engine import (
    DatabaseConfig,
    DatabaseConnection,
    ProfileTableMaterializationConfig,
    TabularDataset,
    TableColumnSpec,
    analyze_profile_table_from_database_query,
    bootstrap_local_database,
    compress_profile_table_from_database_query,
    database_profile_query_artifact_metadata,
    describe_database_table,
    execute_database_query,
    inspect_database,
    materialize_database_query_to_table,
    materialize_database_query,
    materialize_profile_table_from_database_query,
    write_tabular_dataset_to_database,
)


def _profile_dataset() -> TabularDataset:
    return TabularDataset.from_rows(
        [
            {"time": 0.0, "x=0.0": 0.1, "x=0.5": 1.0, "x=1.0": 0.1},
            {"time": 0.1, "x=0.0": 0.2, "x=0.5": 0.9, "x=1.0": 0.2},
            {"time": 0.2, "x=0.0": 0.15, "x=0.5": 0.85, "x=1.0": 0.15},
        ]
    )


def test_database_write_query_and_describe(tmp_path) -> None:
    database_path = tmp_path / "profiles.sqlite"
    dataset = _profile_dataset()
    bootstrap = bootstrap_local_database(database_path)

    write_summary = write_tabular_dataset_to_database(
        database_path,
        "profiles",
        dataset,
        if_exists="replace",
    )
    inspection = inspect_database(database_path)
    schema = describe_database_table(database_path, "profiles")
    query = materialize_database_query(
        database_path,
        'SELECT time, "x=0.0", "x=0.5", "x=1.0" FROM "profiles" ORDER BY time',
    )

    assert bootstrap.capability.local_bootstrap_supported is True
    assert write_summary.row_count == dataset.row_count
    assert "profiles" in inspection.tables
    assert schema.row_count == dataset.row_count
    assert query.dataset.row_count == dataset.row_count
    assert query.dataset.column_names[0] == "time"


def test_materialize_database_query_rejects_attach_side_effects(tmp_path) -> None:
    database_path = tmp_path / "profiles.sqlite"
    bootstrap_local_database(database_path)

    with pytest.raises(Exception, match="not authorized|readonly|read only|not a database"):
        materialize_database_query(
            database_path,
            "ATTACH DATABASE '/etc/passwd' AS stolen",
        )


def test_database_crud_and_sql_backed_spectral_analysis(tmp_path) -> None:
    database_path = tmp_path / "profiles.sqlite"
    dataset = _profile_dataset()
    write_tabular_dataset_to_database(database_path, "profiles", dataset, if_exists="replace")

    with DatabaseConnection(database_path) as connection:
        updated = connection.update_rows(
            "profiles",
            {"x=0.5": 1.1},
            where='"time" = :time',
            parameters={"time": 0.0},
        )
        deleted = connection.delete_rows(
            "profiles",
            where='"time" = :time',
            parameters={"time": 0.2},
        )
        transformed = connection.create_table_from_query(
            "profiles_copy",
            'SELECT time, "x=0.0", "x=0.5", "x=1.0" FROM "profiles"',
        )

    analysis = analyze_profile_table_from_database_query(
        database_path,
        'SELECT time, "x=0.0", "x=0.5", "x=1.0" FROM "profiles" ORDER BY time',
        num_modes=3,
        device="cpu",
    )
    materialized = materialize_database_query_to_table(
        database_path,
        "profiles_snapshot",
        'SELECT * FROM "profiles"',
        replace=True,
    )

    assert updated == 1
    assert deleted == 1
    assert transformed.row_count == 2
    assert materialized.schema.row_count == 2
    assert analysis.coefficients.shape == (2, 3)


def test_database_respects_create_if_missing_false(tmp_path) -> None:
    missing_path = tmp_path / "missing.sqlite"

    with pytest.raises(FileNotFoundError, match=str(missing_path)):
        with DatabaseConnection(DatabaseConfig.sqlite(missing_path, create_if_missing=False)):
            pass

    with pytest.raises(FileNotFoundError, match=str(missing_path)):
        inspect_database(missing_path)

    with pytest.raises(FileNotFoundError, match=str(missing_path)):
        execute_database_query(missing_path, "SELECT 1")


def test_database_profile_query_materialization_and_compression_are_explicit(tmp_path) -> None:
    database_path = tmp_path / "profiles.sqlite"
    dataset = TabularDataset.from_rows(
        [
            {"label": "late", "time": 0.2, "x=1.0": 0.15, "x=0.0": 0.2, "x=0.5": 0.85},
            {"label": "early", "time": 0.0, "x=1.0": 0.1, "x=0.0": 0.1, "x=0.5": 1.0},
        ]
    )
    write_tabular_dataset_to_database(database_path, "profiles", dataset, if_exists="replace")
    config = ProfileTableMaterializationConfig(
        time_column="time",
        position_columns=("x=1.0", "x=0.0", "x=0.5"),
        sort_by_time=True,
    )
    query = 'SELECT label, time, "x=1.0", "x=0.0", "x=0.5" FROM "profiles"'

    materialized = materialize_profile_table_from_database_query(
        database_path,
        query,
        materialization=config,
    )
    compression = compress_profile_table_from_database_query(
        database_path,
        query,
        materialization=config,
        num_modes=3,
        device="cpu",
    )
    metadata = database_profile_query_artifact_metadata(
        database_path,
        query,
        materialization=config,
    )

    assert materialized.layout.position_columns == ("x=0.0", "x=0.5", "x=1.0")
    assert materialized.table.source.endswith("profiles.sqlite")
    assert materialized.table.sample_times.tolist() == [0.0, 0.2]
    assert compression.coefficients.shape == (2, 3)
    assert metadata["input"]["profile_table"]["position_columns"] == ("x=1.0", "x=0.0", "x=0.5")
    assert metadata["input"]["profile_table"]["sort_by_time"] is True


def test_create_table_from_query_avoids_preflight_table_probes(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "profiles.sqlite"
    write_tabular_dataset_to_database(database_path, "profiles", _profile_dataset(), if_exists="replace")

    def _unexpected_list_tables(self):
        raise AssertionError("create_table_from_query should not preflight existence with list_tables")

    monkeypatch.setattr(DatabaseConnection, "list_tables", _unexpected_list_tables)

    with DatabaseConnection(database_path) as connection:
        schema = connection.create_table_from_query(
            "profiles_copy",
            'SELECT time, "x=0.0", "x=0.5", "x=1.0" FROM "profiles"',
        )

    assert schema.row_count == 3


def test_database_config_applies_sqlalchemy_pool_settings(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeEngine:
        def dispose(self) -> None:
            captured["disposed"] = True

    class _FakeSQLAlchemy:
        @staticmethod
        def create_engine(url: str, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return _FakeEngine()

    monkeypatch.setattr("spectral_packet_engine.database._require_sqlalchemy", lambda: _FakeSQLAlchemy)

    connection = DatabaseConnection(
        DatabaseConfig(
            url="postgresql://db.example/spectral",
            pool_pre_ping=False,
            pool_recycle_seconds=120,
            pool_size=7,
            max_overflow=11,
            pool_timeout_seconds=9.5,
        )
    )
    connection.connect()
    connection.close()

    assert captured["url"] == "postgresql://db.example/spectral"
    assert captured["kwargs"] == {
        "future": True,
        "pool_pre_ping": False,
        "pool_recycle": 120,
        "pool_size": 7,
        "max_overflow": 11,
        "pool_timeout": 9.5,
    }
    assert captured["disposed"] is True


def test_remote_describe_table_uses_sqlalchemy_core_select(monkeypatch) -> None:
    sqlalchemy = pytest.importorskip("sqlalchemy")

    monkeypatch.setattr(DatabaseConfig, "is_sqlite", property(lambda self: False))

    def _unexpected_text(*args, **kwargs):
        raise AssertionError("describe_table should use SQLAlchemy Core select(), not raw text()")

    monkeypatch.setattr(sqlalchemy, "text", _unexpected_text)

    with DatabaseConnection(DatabaseConfig(url="sqlite:///:memory:")) as connection:
        connection.create_table(
            "profiles",
            (
                TableColumnSpec("time", "REAL", nullable=False, primary_key=True),
                TableColumnSpec("label", "TEXT"),
            ),
            if_not_exists=False,
        )
        schema = connection.describe_table("profiles")

    assert schema.row_count == 0
    assert schema.columns[0].primary_key is True
    assert schema.columns[1].name == "label"
