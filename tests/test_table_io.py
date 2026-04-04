from __future__ import annotations

import numpy as np
import pytest

from spectral_packet_engine import (
    ProfileTable,
    ProfileTableMaterializationConfig,
    TabularDataset,
    excel_support_is_available,
    load_profile_table,
    load_profile_table_csv,
    profile_table_layout_from_tabular_dataset,
    profile_table_from_tabular_dataset,
    save_profile_table,
    save_profile_table_csv,
)


def test_profile_table_csv_roundtrip(tmp_path) -> None:
    grid = np.linspace(0.0, 1.0, 8)
    times = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    profiles = np.vstack(
        [
            np.exp(-((grid - 0.2) ** 2) / (2 * 0.08**2)),
            np.exp(-((grid - 0.5) ** 2) / (2 * 0.09**2)),
            np.exp(-((grid - 0.8) ** 2) / (2 * 0.10**2)),
        ]
    )
    table = ProfileTable(position_grid=grid, sample_times=times, profiles=profiles, source="synthetic")
    path = tmp_path / "profiles.csv"

    save_profile_table_csv(table, path)
    loaded = load_profile_table_csv(path)

    assert loaded.source == str(path)
    np.testing.assert_allclose(loaded.position_grid, grid)
    np.testing.assert_allclose(loaded.sample_times, times)
    np.testing.assert_allclose(loaded.profiles, profiles)


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".json"])
def test_profile_table_roundtrip_multiple_formats(tmp_path, suffix: str) -> None:
    grid = np.linspace(0.0, 1.0, 6)
    times = np.asarray([0.0, 0.25], dtype=np.float64)
    profiles = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.25, 0.1, 0.05],
            [0.05, 0.15, 0.25, 0.3, 0.2, 0.05],
        ],
        dtype=np.float64,
    )
    table = ProfileTable(position_grid=grid, sample_times=times, profiles=profiles)
    path = tmp_path / f"profiles{suffix}"

    save_profile_table(table, path)
    loaded = load_profile_table(path)

    np.testing.assert_allclose(loaded.position_grid, grid)
    np.testing.assert_allclose(loaded.sample_times, times)
    np.testing.assert_allclose(loaded.profiles, profiles)


def test_profile_table_xlsx_roundtrip_when_openpyxl_is_available(tmp_path) -> None:
    if not excel_support_is_available():
        pytest.skip("openpyxl is not installed")

    grid = np.linspace(0.0, 1.0, 5)
    times = np.asarray([0.0, 0.3], dtype=np.float64)
    profiles = np.asarray(
        [
            [0.2, 0.3, 0.25, 0.15, 0.1],
            [0.1, 0.2, 0.3, 0.25, 0.15],
        ],
        dtype=np.float64,
    )
    table = ProfileTable(position_grid=grid, sample_times=times, profiles=profiles)
    path = tmp_path / "profiles.xlsx"

    save_profile_table(table, path)
    loaded = load_profile_table(path)

    np.testing.assert_allclose(loaded.position_grid, grid)
    np.testing.assert_allclose(loaded.sample_times, times)
    np.testing.assert_allclose(loaded.profiles, profiles)


def test_profile_table_unsupported_extension_error_mentions_optional_formats(tmp_path) -> None:
    path = tmp_path / "profiles.unsupported"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="supported formats"):
        load_profile_table(path)


def test_profile_table_materialization_config_selects_columns_and_sorts_times() -> None:
    dataset = TabularDataset.from_rows(
        [
            {"Time": 0.2, "x=1.0": 0.1, "x=0.0": 0.2, "x=0.5": 0.9},
            {"Time": 0.0, "x=1.0": 0.15, "x=0.0": 0.1, "x=0.5": 1.0},
        ]
    )
    config = ProfileTableMaterializationConfig(
        time_column="time",
        position_columns=("x=1.0", "x=0.0"),
        sort_by_time=True,
        source="materialized",
    )

    layout = profile_table_layout_from_tabular_dataset(dataset, config=config)
    table = profile_table_from_tabular_dataset(dataset, config=config)

    assert layout.time_column == "Time"
    assert layout.position_columns == ("x=0.0", "x=1.0")
    np.testing.assert_allclose(layout.position_grid, [0.0, 1.0])
    np.testing.assert_allclose(table.sample_times, [0.0, 0.2])
    np.testing.assert_allclose(
        table.profiles,
        [
            [0.1, 0.15],
            [0.2, 0.1],
        ],
    )
    assert table.source == "materialized"
