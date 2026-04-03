from __future__ import annotations

import pytest

from spectral_packet_engine import (
    DatabaseConfig,
    DatabaseConnection,
    TabularDataset,
    analyze_profile_table_from_database_query,
    bootstrap_local_database,
    describe_database_table,
    execute_database_query,
    inspect_database,
    materialize_database_query_to_table,
    materialize_database_query,
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
