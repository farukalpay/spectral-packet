from __future__ import annotations

import numpy as np
import pytest

from spectral_packet_engine import (
    JoinSpec,
    TabularDataset,
    load_tabular_dataset,
    parquet_support_is_available,
    profile_table_from_tabular_dataset,
    save_tabular_dataset,
    summarize_tabular_dataset,
)


def _profile_rows() -> list[dict[str, float]]:
    return [
        {"time": 0.0, "x=0.0": 0.1, "x=0.5": 1.0, "x=1.0": 0.1},
        {"time": 0.1, "x=0.0": 0.2, "x=0.5": 0.9, "x=1.0": 0.2},
        {"time": 0.2, "x=0.0": 0.15, "x=0.5": 0.85, "x=1.0": 0.15},
    ]


def test_tabular_dataset_roundtrip_and_summary(tmp_path) -> None:
    dataset = TabularDataset.from_rows(_profile_rows())
    csv_path = tmp_path / "profiles.csv"

    save_tabular_dataset(dataset, csv_path)
    csv_loaded = load_tabular_dataset(csv_path)
    summary = summarize_tabular_dataset(csv_loaded)

    assert csv_loaded.row_count == 3
    assert summary.column_count == 4
    assert "parquet" in summary.supported_formats
    assert summary.validation.missing_value_counts["time"] == 0

    if parquet_support_is_available():
        parquet_path = tmp_path / "profiles.parquet"
        save_tabular_dataset(dataset, parquet_path)
        parquet_loaded = load_tabular_dataset(parquet_path)
        assert parquet_loaded.row_count == 3


def test_tabular_unsupported_extension_error_mentions_optional_formats(tmp_path) -> None:
    path = tmp_path / "dataset.unsupported"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="supported formats"):
        load_tabular_dataset(path)


def test_tabular_join_and_profile_conversion() -> None:
    profiles = TabularDataset.from_rows(_profile_rows())
    metadata = TabularDataset.from_rows(
        [
            {"time": 0.0, "temperature_nk": 20.0},
            {"time": 0.1, "temperature_nk": 21.0},
            {"time": 0.2, "temperature_nk": 22.0},
        ]
    )
    joined = profiles.join(
        metadata,
        spec=JoinSpec(left_keys=("time",), right_keys=("time",), how="inner"),
    )
    table = profile_table_from_tabular_dataset(profiles)

    assert joined.row_count == profiles.row_count
    assert "temperature_nk" in joined.column_names
    np.testing.assert_allclose(table.position_grid, np.asarray([0.0, 0.5, 1.0], dtype=np.float64))
    np.testing.assert_allclose(table.sample_times, np.asarray([0.0, 0.1, 0.2], dtype=np.float64))
