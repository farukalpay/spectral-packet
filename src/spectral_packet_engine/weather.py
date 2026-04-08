from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from spectral_packet_engine.tabular import TabularDataset, TabularSource


_OPEN_METEO_ENDPOINTS = {
    "forecast": "https://api.open-meteo.com/v1/forecast",
    "historical-forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "historical-weather": "https://archive-api.open-meteo.com/v1/archive",
}


def _coerce_iso_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid ISO date: {value!r}") from exc


def _coerce_open_meteo_api_kind(
    *,
    start_date: str | date,
    end_date: str | date,
    api_kind: str = "auto",
) -> str:
    normalized = str(api_kind).strip().lower()
    if normalized != "auto":
        if normalized not in _OPEN_METEO_ENDPOINTS:
            supported = ", ".join(("auto", *sorted(_OPEN_METEO_ENDPOINTS)))
            raise ValueError(f"api_kind must be one of {supported}")
        return normalized
    start = _coerce_iso_date(start_date)
    end = _coerce_iso_date(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    if end < date.today():
        return "historical-weather"
    return "forecast"


def _coerce_hourly_variables(hourly_variables: Sequence[str]) -> tuple[str, ...]:
    variables = tuple(str(item).strip() for item in hourly_variables)
    if not variables:
        raise ValueError("hourly_variables must contain at least one variable")
    if any(not item for item in variables):
        raise ValueError("hourly_variables must not contain empty names")
    if len(set(variables)) != len(variables):
        raise ValueError("hourly_variables must be unique")
    return variables


def _coerce_string_sequence(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    normalized = tuple(str(item).strip() for item in values)
    if not normalized:
        return None
    if any(not item for item in normalized):
        raise ValueError("sequence values must not contain empty strings")
    return normalized


def _decode_open_meteo_error(error: HTTPError) -> str:
    payload = None
    try:
        payload = json.loads(error.read().decode("utf-8"))
    except Exception:
        payload = None
    if isinstance(payload, Mapping):
        reason = payload.get("reason")
        if reason:
            return str(reason)
        message = payload.get("error")
        if message:
            return str(message)
    return f"{error.code} {error.reason}"


def _build_open_meteo_request_url(
    *,
    latitude: float,
    longitude: float,
    start_date: str | date,
    end_date: str | date,
    hourly_variables: Sequence[str],
    api_kind: str,
    timezone_name: str,
    models: Sequence[str] | None = None,
    temperature_unit: str | None = None,
    wind_speed_unit: str | None = None,
    precipitation_unit: str | None = None,
    cell_selection: str | None = None,
    elevation: float | None = None,
) -> str:
    start = _coerce_iso_date(start_date)
    end = _coerce_iso_date(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    endpoint = _OPEN_METEO_ENDPOINTS[api_kind]
    params: dict[str, Any] = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(_coerce_hourly_variables(hourly_variables)),
        "timezone": str(timezone_name),
        "timeformat": "unixtime",
    }
    resolved_models = _coerce_string_sequence(models)
    if resolved_models is not None:
        params["models"] = ",".join(resolved_models)
    if temperature_unit is not None:
        params["temperature_unit"] = str(temperature_unit)
    if wind_speed_unit is not None:
        params["wind_speed_unit"] = str(wind_speed_unit)
    if precipitation_unit is not None:
        params["precipitation_unit"] = str(precipitation_unit)
    if cell_selection is not None:
        params["cell_selection"] = str(cell_selection)
    if elevation is not None:
        params["elevation"] = float(elevation)
    return f"{endpoint}?{urlencode(params)}"


def _load_open_meteo_payload(url: str, *, timeout_seconds: float) -> Mapping[str, Any]:
    request = Request(url, headers={"User-Agent": "spectral-packet-engine/open-meteo"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Open-Meteo request failed: {_decode_open_meteo_error(exc)}") from exc
    except URLError as exc:
        raise RuntimeError(f"Open-Meteo request failed: {exc.reason}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Open-Meteo response must be a JSON object")
    return payload


def _normalize_open_meteo_hourly_dataset(
    payload: Mapping[str, Any],
    *,
    api_kind: str,
    request_url: str,
    hourly_variables: Sequence[str],
) -> "OpenMeteoHourlyDataset":
    hourly = payload.get("hourly")
    hourly_units = payload.get("hourly_units")
    if not isinstance(hourly, Mapping):
        raise ValueError("Open-Meteo response must include an 'hourly' object")
    if not isinstance(hourly_units, Mapping):
        raise ValueError("Open-Meteo response must include an 'hourly_units' object")

    times = hourly.get("time")
    if not isinstance(times, Sequence) or isinstance(times, (str, bytes)):
        raise ValueError("Open-Meteo hourly response must include a 'time' sequence")
    if not times:
        raise ValueError("Open-Meteo hourly response did not contain any samples")

    resolved_variables = _coerce_hourly_variables(hourly_variables)
    columns: dict[str, Sequence[Any]] = {
        "time": [float(value) for value in times],
        "time_iso8601": [
            datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat().replace("+00:00", "Z")
            for value in times
        ],
    }
    sample_count = len(times)
    units: dict[str, str] = {}
    for variable in resolved_variables:
        values = hourly.get(variable)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"Open-Meteo response did not include hourly variable '{variable}'")
        if len(values) != sample_count:
            raise ValueError(
                f"Open-Meteo hourly variable '{variable}' length {len(values)} does not match time length {sample_count}"
            )
        columns[variable] = list(values)
        if variable in hourly_units:
            units[variable] = str(hourly_units[variable])
    if "time" in hourly_units:
        units["time"] = str(hourly_units["time"])

    dataset = TabularDataset(
        columns={name: values for name, values in columns.items()},
        source=TabularSource(
            kind="open-meteo",
            location=request_url,
            description="Hourly meteorology dataset normalized from Open-Meteo.",
        ),
    )
    return OpenMeteoHourlyDataset(
        api_kind=api_kind,
        request_url=request_url,
        latitude=float(payload["latitude"]),
        longitude=float(payload["longitude"]),
        elevation=None if payload.get("elevation") is None else float(payload["elevation"]),
        timezone=str(payload.get("timezone", "GMT")),
        utc_offset_seconds=int(payload.get("utc_offset_seconds", 0)),
        hourly_variables=resolved_variables,
        hourly_units=units,
        dataset=dataset,
    )


@dataclass(frozen=True, slots=True)
class OpenMeteoHourlyDataset:
    api_kind: str
    request_url: str
    latitude: float
    longitude: float
    elevation: float | None
    timezone: str
    utc_offset_seconds: int
    hourly_variables: tuple[str, ...]
    hourly_units: dict[str, str]
    dataset: TabularDataset


def fetch_open_meteo_hourly_dataset(
    *,
    latitude: float,
    longitude: float,
    start_date: str | date,
    end_date: str | date,
    hourly_variables: Sequence[str],
    api_kind: str = "auto",
    timezone_name: str = "GMT",
    models: Sequence[str] | None = None,
    temperature_unit: str | None = None,
    wind_speed_unit: str | None = None,
    precipitation_unit: str | None = None,
    cell_selection: str | None = None,
    elevation: float | None = None,
    timeout_seconds: float = 30.0,
) -> OpenMeteoHourlyDataset:
    resolved_api_kind = _coerce_open_meteo_api_kind(
        start_date=start_date,
        end_date=end_date,
        api_kind=api_kind,
    )
    request_url = _build_open_meteo_request_url(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        hourly_variables=hourly_variables,
        api_kind=resolved_api_kind,
        timezone_name=timezone_name,
        models=models,
        temperature_unit=temperature_unit,
        wind_speed_unit=wind_speed_unit,
        precipitation_unit=precipitation_unit,
        cell_selection=cell_selection,
        elevation=elevation,
    )
    payload = _load_open_meteo_payload(request_url, timeout_seconds=timeout_seconds)
    return _normalize_open_meteo_hourly_dataset(
        payload,
        api_kind=resolved_api_kind,
        request_url=request_url,
        hourly_variables=hourly_variables,
    )


__all__ = [
    "OpenMeteoHourlyDataset",
    "fetch_open_meteo_hourly_dataset",
]
