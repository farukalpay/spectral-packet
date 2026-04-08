from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from spectral_packet_engine.tabular import TabularDataset


_KELVIN_OFFSET = 273.15
_STEFAN_BOLTZMANN = 5.670374419e-8
_VON_KARMAN = 0.4
_DRY_AIR_GAS_CONSTANT = 287.05
_AIR_SPECIFIC_HEAT = 1005.0
_SUTHERLAND_REFERENCE_TEMPERATURE_K = 273.15
_SUTHERLAND_REFERENCE_VISCOSITY_PA_S = 1.716e-5
_SUTHERLAND_CONSTANT_K = 111.0
_EPSILON = 1e-12


def _coerce_positive(value: float, *, field_name: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return number


def _coerce_non_negative(value: float, *, field_name: str) -> float:
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def _numeric_column(dataset: TabularDataset, column_name: str) -> np.ndarray:
    if column_name not in dataset.columns:
        raise ValueError(f"dataset is missing required column {column_name!r}")
    values = np.asarray(dataset.columns[column_name], dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"column {column_name!r} must be one-dimensional")
    return values


def _optional_numeric_column(dataset: TabularDataset, column_name: str | None) -> np.ndarray | None:
    if column_name is None:
        return None
    if column_name not in dataset.columns:
        return None
    return _numeric_column(dataset, column_name)


def _saturation_vapor_pressure_pa(air_temperature_c: np.ndarray) -> np.ndarray:
    # Magnus-Tetens approximation over the liquid-water range.
    return 610.94 * np.exp((17.625 * air_temperature_c) / (air_temperature_c + 243.04))


def _dynamic_viscosity_pa_s(air_temperature_k: np.ndarray) -> np.ndarray:
    numerator = _SUTHERLAND_REFERENCE_VISCOSITY_PA_S * (air_temperature_k / _SUTHERLAND_REFERENCE_TEMPERATURE_K) ** 1.5
    return numerator * (
        (_SUTHERLAND_REFERENCE_TEMPERATURE_K + _SUTHERLAND_CONSTANT_K)
        / (air_temperature_k + _SUTHERLAND_CONSTANT_K)
    )


@dataclass(frozen=True, slots=True)
class UrbanBoundaryLayerConfig:
    measurement_height_m: float = 10.0
    zero_plane_displacement_m: float = 0.0
    roughness_length_m: float = 0.8
    thermal_roughness_length_m: float = 0.1
    min_wind_speed_m_s: float = 0.1
    reference_pressure_pa: float = 101325.0

    def __post_init__(self) -> None:
        _coerce_positive(self.measurement_height_m, field_name="measurement_height_m")
        _coerce_non_negative(self.zero_plane_displacement_m, field_name="zero_plane_displacement_m")
        _coerce_positive(self.roughness_length_m, field_name="roughness_length_m")
        _coerce_positive(self.thermal_roughness_length_m, field_name="thermal_roughness_length_m")
        _coerce_positive(self.min_wind_speed_m_s, field_name="min_wind_speed_m_s")
        _coerce_positive(self.reference_pressure_pa, field_name="reference_pressure_pa")
        if self.zero_plane_displacement_m >= self.measurement_height_m:
            raise ValueError("zero_plane_displacement_m must remain below measurement_height_m")


@dataclass(frozen=True, slots=True)
class UrbanRadiativeTransferConfig:
    shortwave_absorptivity: float = 0.7
    longwave_emissivity: float = 0.97
    shortwave_projected_area_factor: float = 0.7
    sky_view_factor: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "shortwave_absorptivity",
            "longwave_emissivity",
            "shortwave_projected_area_factor",
            "sky_view_factor",
        ):
            value = getattr(self, field_name)
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{field_name} must fall in the interval (0, 1]")


@dataclass(frozen=True, slots=True)
class HumanThermalResponseConfig:
    skin_temperature_c: float = 33.0
    clothing_insulation_clo: float = 0.7
    metabolic_heat_flux_w_m2: float = 80.0

    def __post_init__(self) -> None:
        _coerce_positive(self.clothing_insulation_clo, field_name="clothing_insulation_clo")
        _coerce_positive(self.metabolic_heat_flux_w_m2, field_name="metabolic_heat_flux_w_m2")


@dataclass(frozen=True, slots=True)
class UrbanMicroclimateColumnMapping:
    air_temperature_column: str
    relative_humidity_column: str
    wind_speed_column: str
    pressure_column: str | None = "surface_pressure"
    shortwave_radiation_column: str | None = None
    longwave_radiation_column: str | None = None

    @classmethod
    def open_meteo(
        cls,
        *,
        air_temperature_column: str = "temperature_2m",
        relative_humidity_column: str = "relative_humidity_2m",
        wind_speed_column: str = "wind_speed_10m",
        pressure_column: str = "surface_pressure",
        shortwave_radiation_column: str = "shortwave_radiation",
        longwave_radiation_column: str = "terrestrial_radiation",
    ) -> "UrbanMicroclimateColumnMapping":
        return cls(
            air_temperature_column=air_temperature_column,
            relative_humidity_column=relative_humidity_column,
            wind_speed_column=wind_speed_column,
            pressure_column=pressure_column,
            shortwave_radiation_column=shortwave_radiation_column,
            longwave_radiation_column=longwave_radiation_column,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_urban_microclimate_mapping(
    *,
    mapping_profile: str | None = None,
    air_temperature_column: str | None = None,
    relative_humidity_column: str | None = None,
    wind_speed_column: str | None = None,
    pressure_column: str | None = "surface_pressure",
    shortwave_radiation_column: str | None = None,
    longwave_radiation_column: str | None = None,
) -> UrbanMicroclimateColumnMapping:
    if mapping_profile is not None:
        normalized_profile = str(mapping_profile).strip().lower()
        if normalized_profile != "open-meteo":
            raise ValueError("mapping_profile must be 'open-meteo' when provided")
        mapping = UrbanMicroclimateColumnMapping.open_meteo()
    else:
        if air_temperature_column is None or relative_humidity_column is None or wind_speed_column is None:
            raise ValueError(
                "explicit urban microclimate mapping requires air_temperature_column, relative_humidity_column, and wind_speed_column"
            )
        mapping = UrbanMicroclimateColumnMapping(
            air_temperature_column=air_temperature_column,
            relative_humidity_column=relative_humidity_column,
            wind_speed_column=wind_speed_column,
            pressure_column=pressure_column,
            shortwave_radiation_column=shortwave_radiation_column,
            longwave_radiation_column=longwave_radiation_column,
        )
    overrides = {
        "air_temperature_column": air_temperature_column,
        "relative_humidity_column": relative_humidity_column,
        "wind_speed_column": wind_speed_column,
        "pressure_column": pressure_column,
        "shortwave_radiation_column": shortwave_radiation_column,
        "longwave_radiation_column": longwave_radiation_column,
    }
    payload = mapping.to_dict()
    for key, value in overrides.items():
        if value is not None:
            payload[key] = value
    return UrbanMicroclimateColumnMapping(**payload)


@dataclass(frozen=True, slots=True)
class UrbanMicroclimateSummary:
    row_count: int
    source_kind: str | None
    source_location: str | None
    pressure_mode: str
    radiative_mode: str
    convective_heat_transfer_min_w_m2_k: float
    convective_heat_transfer_max_w_m2_k: float
    operative_temperature_min_c: float
    operative_temperature_max_c: float
    thermal_storage_flux_min_w_m2: float
    thermal_storage_flux_max_w_m2: float
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UrbanMicroclimateResult:
    dataset: TabularDataset
    summary: UrbanMicroclimateSummary
    mapping: UrbanMicroclimateColumnMapping
    boundary_layer: UrbanBoundaryLayerConfig
    radiative_transfer: UrbanRadiativeTransferConfig
    human_thermal_response: HumanThermalResponseConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "summary": self.summary.to_dict(),
            "mapping": self.mapping.to_dict(),
            "boundary_layer": asdict(self.boundary_layer),
            "radiative_transfer": asdict(self.radiative_transfer),
            "human_thermal_response": asdict(self.human_thermal_response),
        }


def derive_urban_microclimate_dataset(
    dataset: TabularDataset,
    *,
    mapping: UrbanMicroclimateColumnMapping,
    boundary_layer: UrbanBoundaryLayerConfig | None = None,
    radiative_transfer: UrbanRadiativeTransferConfig | None = None,
    human_thermal_response: HumanThermalResponseConfig | None = None,
) -> UrbanMicroclimateResult:
    resolved_boundary = UrbanBoundaryLayerConfig() if boundary_layer is None else boundary_layer
    resolved_radiative = UrbanRadiativeTransferConfig() if radiative_transfer is None else radiative_transfer
    resolved_human = HumanThermalResponseConfig() if human_thermal_response is None else human_thermal_response

    air_temperature_c = _numeric_column(dataset, mapping.air_temperature_column)
    relative_humidity_percent = _numeric_column(dataset, mapping.relative_humidity_column)
    wind_speed_m_s = _numeric_column(dataset, mapping.wind_speed_column)

    pressure_column = _optional_numeric_column(dataset, mapping.pressure_column)
    if pressure_column is None:
        pressure_pa = np.full(dataset.row_count, resolved_boundary.reference_pressure_pa, dtype=np.float64)
        pressure_mode = "reference-pressure"
    else:
        pressure_pa = pressure_column
        pressure_mode = "dataset-pressure"

    shortwave_w_m2 = _optional_numeric_column(dataset, mapping.shortwave_radiation_column)
    longwave_w_m2 = _optional_numeric_column(dataset, mapping.longwave_radiation_column)

    air_temperature_k = air_temperature_c + _KELVIN_OFFSET
    saturation_vapor_pressure_pa = _saturation_vapor_pressure_pa(air_temperature_c)
    vapor_pressure_pa = (relative_humidity_percent / 100.0) * saturation_vapor_pressure_pa
    if np.any(vapor_pressure_pa >= pressure_pa):
        raise ValueError("relative humidity / pressure combination produced vapor pressure greater than or equal to air pressure")
    specific_humidity = 0.622 * vapor_pressure_pa / (pressure_pa - 0.378 * vapor_pressure_pa)
    virtual_temperature_k = air_temperature_k * (1.0 + 0.61 * specific_humidity)
    air_density_kg_m3 = pressure_pa / (_DRY_AIR_GAS_CONSTANT * virtual_temperature_k)
    dynamic_viscosity_pa_s = _dynamic_viscosity_pa_s(air_temperature_k)
    kinematic_viscosity_m2_s = dynamic_viscosity_pa_s / air_density_kg_m3

    effective_height_m = resolved_boundary.measurement_height_m - resolved_boundary.zero_plane_displacement_m
    if effective_height_m <= max(resolved_boundary.roughness_length_m, resolved_boundary.thermal_roughness_length_m):
        raise ValueError("measurement height must exceed both roughness lengths after zero-plane displacement")
    stabilized_wind_speed = np.maximum(np.abs(wind_speed_m_s), resolved_boundary.min_wind_speed_m_s)
    momentum_log = np.log(effective_height_m / resolved_boundary.roughness_length_m)
    thermal_log = np.log(effective_height_m / resolved_boundary.thermal_roughness_length_m)
    friction_velocity_m_s = _VON_KARMAN * stabilized_wind_speed / momentum_log
    aerodynamic_resistance_s_m = momentum_log * thermal_log / (_VON_KARMAN**2 * stabilized_wind_speed)
    convective_heat_transfer_w_m2_k = air_density_kg_m3 * _AIR_SPECIFIC_HEAT / aerodynamic_resistance_s_m

    notes: list[str] = []
    if shortwave_w_m2 is not None and longwave_w_m2 is not None:
        absorbed_radiation_w_m2 = (
            resolved_radiative.shortwave_absorptivity
            * resolved_radiative.shortwave_projected_area_factor
            * np.maximum(shortwave_w_m2, 0.0)
            + resolved_radiative.longwave_emissivity
            * resolved_radiative.sky_view_factor
            * np.maximum(longwave_w_m2, 0.0)
        )
        denominator = resolved_radiative.longwave_emissivity * resolved_radiative.sky_view_factor * _STEFAN_BOLTZMANN
        mean_radiant_temperature_k = np.power(np.maximum(absorbed_radiation_w_m2 / max(denominator, _EPSILON), _EPSILON), 0.25)
        radiative_mode = "flux-balance"
    else:
        mean_radiant_temperature_k = air_temperature_k.copy()
        radiative_mode = "air-temperature-fallback"
        notes.append(
            "Mean radiant temperature fell back to air temperature because both shortwave and longwave downwelling flux columns were not available."
        )

    reference_radiative_temperature_k = 0.5 * (air_temperature_k + mean_radiant_temperature_k)
    radiative_heat_transfer_w_m2_k = (
        4.0
        * resolved_radiative.longwave_emissivity
        * _STEFAN_BOLTZMANN
        * reference_radiative_temperature_k**3
    )
    total_surface_exchange_w_m2_k = np.maximum(
        convective_heat_transfer_w_m2_k + radiative_heat_transfer_w_m2_k,
        _EPSILON,
    )
    operative_temperature_k = (
        convective_heat_transfer_w_m2_k * air_temperature_k
        + radiative_heat_transfer_w_m2_k * mean_radiant_temperature_k
    ) / total_surface_exchange_w_m2_k
    clothing_resistance_m2_k_w = 0.155 * resolved_human.clothing_insulation_clo
    skin_temperature_k = resolved_human.skin_temperature_c + _KELVIN_OFFSET
    dry_heat_loss_w_m2 = (
        skin_temperature_k - operative_temperature_k
    ) / (clothing_resistance_m2_k_w + 1.0 / total_surface_exchange_w_m2_k)
    thermal_storage_flux_w_m2 = resolved_human.metabolic_heat_flux_w_m2 - dry_heat_loss_w_m2

    derived_columns = {name: values.copy() for name, values in dataset.columns.items()}
    derived_columns.update(
        {
            "air_temperature_k": air_temperature_k,
            "saturation_vapor_pressure_pa": saturation_vapor_pressure_pa,
            "vapor_pressure_pa": vapor_pressure_pa,
            "specific_humidity_kg_kg": specific_humidity,
            "virtual_temperature_k": virtual_temperature_k,
            "air_density_kg_m3": air_density_kg_m3,
            "dynamic_viscosity_pa_s": dynamic_viscosity_pa_s,
            "kinematic_viscosity_m2_s": kinematic_viscosity_m2_s,
            "friction_velocity_m_s": friction_velocity_m_s,
            "aerodynamic_resistance_s_m": aerodynamic_resistance_s_m,
            "convective_heat_transfer_w_m2_k": convective_heat_transfer_w_m2_k,
            "mean_radiant_temperature_c": mean_radiant_temperature_k - _KELVIN_OFFSET,
            "radiative_heat_transfer_w_m2_k": radiative_heat_transfer_w_m2_k,
            "operative_temperature_c": operative_temperature_k - _KELVIN_OFFSET,
            "dry_heat_loss_w_m2": dry_heat_loss_w_m2,
            "thermal_storage_flux_w_m2": thermal_storage_flux_w_m2,
        }
    )

    return UrbanMicroclimateResult(
        dataset=TabularDataset(columns=derived_columns, source=dataset.source),
        summary=UrbanMicroclimateSummary(
            row_count=dataset.row_count,
            source_kind=None if dataset.source is None else dataset.source.kind,
            source_location=None if dataset.source is None else dataset.source.location,
            pressure_mode=pressure_mode,
            radiative_mode=radiative_mode,
            convective_heat_transfer_min_w_m2_k=float(np.nanmin(convective_heat_transfer_w_m2_k)),
            convective_heat_transfer_max_w_m2_k=float(np.nanmax(convective_heat_transfer_w_m2_k)),
            operative_temperature_min_c=float(np.nanmin(operative_temperature_k - _KELVIN_OFFSET)),
            operative_temperature_max_c=float(np.nanmax(operative_temperature_k - _KELVIN_OFFSET)),
            thermal_storage_flux_min_w_m2=float(np.nanmin(thermal_storage_flux_w_m2)),
            thermal_storage_flux_max_w_m2=float(np.nanmax(thermal_storage_flux_w_m2)),
            notes=tuple(notes),
        ),
        mapping=mapping,
        boundary_layer=resolved_boundary,
        radiative_transfer=resolved_radiative,
        human_thermal_response=resolved_human,
    )


__all__ = [
    "HumanThermalResponseConfig",
    "UrbanBoundaryLayerConfig",
    "UrbanMicroclimateColumnMapping",
    "UrbanMicroclimateResult",
    "UrbanMicroclimateSummary",
    "UrbanRadiativeTransferConfig",
    "derive_urban_microclimate_dataset",
    "resolve_urban_microclimate_mapping",
]
