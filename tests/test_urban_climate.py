from __future__ import annotations

import numpy as np

from spectral_packet_engine.tabular import TabularDataset, TabularSource
from spectral_packet_engine.workflows import derive_urban_microclimate_from_tabular_dataset


def test_urban_microclimate_workflow_derives_flux_balance_fields_from_open_meteo_profile() -> None:
    dataset = TabularDataset.from_rows(
        [
            {
                "time": 1_774_995_200.0,
                "temperature_2m": 28.0,
                "relative_humidity_2m": 55.0,
                "wind_speed_10m": 3.2,
                "surface_pressure": 100900.0,
                "shortwave_radiation": 650.0,
                "terrestrial_radiation": 410.0,
            },
            {
                "time": 1_774_998_800.0,
                "temperature_2m": 29.5,
                "relative_humidity_2m": 50.0,
                "wind_speed_10m": 4.1,
                "surface_pressure": 100850.0,
                "shortwave_radiation": 720.0,
                "terrestrial_radiation": 415.0,
            },
        ],
        source=TabularSource(kind="open-meteo", location="https://example.invalid/open-meteo"),
    )

    result = derive_urban_microclimate_from_tabular_dataset(
        dataset,
        mapping_profile="open-meteo",
    )

    assert result.summary.radiative_mode == "flux-balance"
    assert result.summary.pressure_mode == "dataset-pressure"
    assert "operative_temperature_c" in result.dataset.column_names
    assert "thermal_storage_flux_w_m2" in result.dataset.column_names
    assert np.all(result.dataset.columns["air_density_kg_m3"] > 0.0)
    assert np.all(np.isfinite(result.dataset.columns["operative_temperature_c"]))


def test_urban_microclimate_workflow_falls_back_to_reference_pressure_and_air_temperature() -> None:
    dataset = TabularDataset.from_rows(
        [
            {
                "air_temp_c": 12.0,
                "rh": 71.0,
                "wind_m_s": 1.4,
            },
            {
                "air_temp_c": 13.0,
                "rh": 68.0,
                "wind_m_s": 1.1,
            },
        ]
    )

    result = derive_urban_microclimate_from_tabular_dataset(
        dataset,
        air_temperature_column="air_temp_c",
        relative_humidity_column="rh",
        wind_speed_column="wind_m_s",
        pressure_column=None,
    )

    assert result.summary.pressure_mode == "reference-pressure"
    assert result.summary.radiative_mode == "air-temperature-fallback"
    assert result.summary.notes
    np.testing.assert_allclose(
        result.dataset.columns["mean_radiant_temperature_c"],
        dataset.columns["air_temp_c"],
    )
