from __future__ import annotations

import json

import numpy as np

from spectral_packet_engine.tabular import TabularDataset
from spectral_packet_engine.weather import fetch_open_meteo_hourly_dataset
from spectral_packet_engine.workflows import ingest_open_meteo_hourly_dataset


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _open_meteo_payload() -> dict[str, object]:
    return {
        "latitude": 41.01,
        "longitude": 28.97,
        "elevation": 39.0,
        "timezone": "GMT",
        "utc_offset_seconds": 0,
        "hourly_units": {
            "time": "unixtime",
            "temperature_2m": "degC",
            "relative_humidity_2m": "%",
        },
        "hourly": {
            "time": [1_775_606_400, 1_775_610_000],
            "temperature_2m": [12.5, 13.0],
            "relative_humidity_2m": [71, 68],
        },
    }


def test_fetch_open_meteo_hourly_dataset_normalizes_hourly_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectral_packet_engine.weather.urlopen",
        lambda request, timeout=30.0: _FakeHTTPResponse(_open_meteo_payload()),
    )

    dataset = fetch_open_meteo_hourly_dataset(
        latitude=41.01,
        longitude=28.97,
        start_date="2026-04-08",
        end_date="2026-04-08",
        hourly_variables=["temperature_2m", "relative_humidity_2m"],
        api_kind="forecast",
    )

    assert dataset.api_kind == "forecast"
    assert dataset.dataset.column_names == (
        "time",
        "time_iso8601",
        "temperature_2m",
        "relative_humidity_2m",
    )
    assert dataset.dataset.source is not None
    assert dataset.dataset.source.kind == "open-meteo"
    assert "api.open-meteo.com" in dataset.request_url
    np.testing.assert_allclose(dataset.dataset.columns["time"], [1_775_606_400.0, 1_775_610_000.0])
    assert dataset.dataset.columns["time_iso8601"].tolist() == [
        "2026-04-08T00:00:00Z",
        "2026-04-08T01:00:00Z",
    ]
    assert dataset.hourly_units == {
        "temperature_2m": "degC",
        "relative_humidity_2m": "%",
        "time": "unixtime",
    }


def test_ingest_open_meteo_hourly_dataset_can_materialize_profile_table(monkeypatch) -> None:
    def _fake_fetch(**_: object):
        return fetch_open_meteo_hourly_dataset(
            latitude=41.01,
            longitude=28.97,
            start_date="2026-04-08",
            end_date="2026-04-08",
            hourly_variables=["temperature_2m", "relative_humidity_2m"],
            api_kind="forecast",
        )

    monkeypatch.setattr(
        "spectral_packet_engine.weather.urlopen",
        lambda request, timeout=30.0: _FakeHTTPResponse(_open_meteo_payload()),
    )
    monkeypatch.setattr(
        "spectral_packet_engine.workflows.fetch_open_meteo_hourly_dataset",
        _fake_fetch,
    )

    result = ingest_open_meteo_hourly_dataset(
        latitude=41.01,
        longitude=28.97,
        start_date="2026-04-08",
        end_date="2026-04-08",
        hourly_variables=["temperature_2m", "relative_humidity_2m"],
        position_columns=["relative_humidity_2m", "temperature_2m"],
        position_values=[1.0, 0.0],
        sort_by_time=True,
    )

    assert isinstance(result.dataset, TabularDataset)
    assert result.dataset_summary.row_count == 2
    assert result.profile_materialization is not None
    assert result.profile_materialization.position_values == (1.0, 0.0)
    assert result.profile_table is not None
    assert result.profile_summary is not None
    np.testing.assert_allclose(result.profile_table.position_grid, [0.0, 1.0])
    np.testing.assert_allclose(result.profile_table.sample_times, [1_775_606_400.0, 1_775_610_000.0])
    np.testing.assert_allclose(
        result.profile_table.profiles,
        [
            [12.5, 71.0],
            [13.0, 68.0],
        ],
    )
