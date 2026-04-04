from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch


AggregationMode = Literal["mean", "median"]


@dataclass(frozen=True, slots=True)
class RemoteScanAsset:
    scan_id: str
    filename: str
    url: str
    md5: str
    doi_url: str
    title: str


@dataclass(frozen=True, slots=True)
class DensityPreprocessingConfig:
    aggregate: AggregationMode = "mean"
    nan_fill_value: float = 0.0
    clip_negative: bool = True
    normalize_each_profile: bool = True
    drop_nonpositive_mass: bool = True

    def __post_init__(self) -> None:
        if self.aggregate not in {"mean", "median"}:
            raise ValueError("aggregate must be either 'mean' or 'median'")
        if not np.isfinite(self.nan_fill_value):
            raise ValueError("nan_fill_value must be finite")


@dataclass(frozen=True, slots=True)
class PreparedDensityProfiles:
    scan_id: str
    scan_name: str
    doi_url: str
    title: str
    position_axis_m: np.ndarray
    time_axis_s: np.ndarray
    density_profiles: np.ndarray
    aggregation: AggregationMode
    nan_fill_value: float
    clip_negative: bool
    normalized: bool
    temperature_nK: float
    temperature_std_nK: float
    transverse_trap_frequencies_hz: tuple[float, float]
    shots_per_time: int

    def __post_init__(self) -> None:
        if self.position_axis_m.ndim != 1:
            raise ValueError("position_axis_m must be one-dimensional")
        if self.time_axis_s.ndim != 1:
            raise ValueError("time_axis_s must be one-dimensional")
        if self.density_profiles.ndim != 2:
            raise ValueError("density_profiles must be two-dimensional")
        if self.density_profiles.shape != (self.time_axis_s.shape[0], self.position_axis_m.shape[0]):
            raise ValueError("density_profiles must have shape [time, position]")

    def to_torch(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid = torch.as_tensor(self.position_axis_m, dtype=dtype, device=device)
        times = torch.as_tensor(self.time_axis_s, dtype=dtype, device=device)
        profiles = torch.as_tensor(self.density_profiles, dtype=dtype, device=device)
        return grid, times, profiles

@dataclass(frozen=True, slots=True)
class PreparedShotDensityProfiles:
    scan_id: str
    scan_name: str
    doi_url: str
    title: str
    position_axis_m: np.ndarray
    time_axis_s: np.ndarray
    density_profiles: np.ndarray
    sample_times_s: np.ndarray
    sample_time_indices: np.ndarray
    sample_shot_indices: np.ndarray
    nan_fill_value: float
    clip_negative: bool
    normalized: bool
    dropped_nonpositive_mass: bool

    def __post_init__(self) -> None:
        if self.position_axis_m.ndim != 1:
            raise ValueError("position_axis_m must be one-dimensional")
        if self.time_axis_s.ndim != 1:
            raise ValueError("time_axis_s must be one-dimensional")
        if self.density_profiles.ndim != 2:
            raise ValueError("density_profiles must be two-dimensional")
        if self.density_profiles.shape[1] != self.position_axis_m.shape[0]:
            raise ValueError("density_profiles must have shape [sample, position]")

    def to_torch(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid = torch.as_tensor(self.position_axis_m, dtype=dtype, device=device)
        sample_times = torch.as_tensor(self.sample_times_s, dtype=dtype, device=device)
        profiles = torch.as_tensor(self.density_profiles, dtype=dtype, device=device)
        return grid, sample_times, profiles


@dataclass(frozen=True, slots=True)
class QuantumGasTransportScan:
    asset: RemoteScanAsset
    scan_name: str
    slope_scan_name: str
    position_axis_m: np.ndarray
    time_axis_s: np.ndarray
    density_profiles_by_position_shot_time: np.ndarray
    slope_profiles_by_position_time: np.ndarray
    temperature_nK: float
    temperature_std_nK: float
    transverse_trap_frequencies_hz: tuple[float, float]
    tof_factors: tuple[float, float]

    @property
    def num_positions(self) -> int:
        return int(self.position_axis_m.shape[0])

    @property
    def num_times(self) -> int:
        return int(self.time_axis_s.shape[0])

    @property
    def num_shots(self) -> int:
        return int(self.density_profiles_by_position_shot_time.shape[1])

    def aggregate_density_profiles(self, *, mode: AggregationMode = "mean") -> np.ndarray:
        if mode == "mean":
            return np.nanmean(self.density_profiles_by_position_shot_time, axis=1).T
        if mode == "median":
            return np.nanmedian(self.density_profiles_by_position_shot_time, axis=1).T
        raise ValueError("mode must be either 'mean' or 'median'")

    def prepare_profiles(
        self,
        config: DensityPreprocessingConfig = DensityPreprocessingConfig(),
    ) -> PreparedDensityProfiles:
        profiles = np.asarray(self.aggregate_density_profiles(mode=config.aggregate), dtype=np.float64)
        profiles = np.nan_to_num(profiles, nan=config.nan_fill_value)
        if config.clip_negative:
            profiles = np.clip(profiles, a_min=0.0, a_max=None)
        if config.normalize_each_profile:
            profile_mass = np.trapezoid(profiles, self.position_axis_m, axis=-1)
            safe_mass = np.where(profile_mass > 0.0, profile_mass, 1.0)
            profiles = profiles / safe_mass[:, None]

        return PreparedDensityProfiles(
            scan_id=self.asset.scan_id,
            scan_name=self.scan_name,
            doi_url=self.asset.doi_url,
            title=self.asset.title,
            position_axis_m=np.asarray(self.position_axis_m, dtype=np.float64),
            time_axis_s=np.asarray(self.time_axis_s, dtype=np.float64),
            density_profiles=profiles,
            aggregation=config.aggregate,
            nan_fill_value=config.nan_fill_value,
            clip_negative=config.clip_negative,
            normalized=config.normalize_each_profile,
            temperature_nK=self.temperature_nK,
            temperature_std_nK=self.temperature_std_nK,
            transverse_trap_frequencies_hz=self.transverse_trap_frequencies_hz,
            shots_per_time=self.num_shots,
        )

    def prepare_shot_profiles(
        self,
        config: DensityPreprocessingConfig = DensityPreprocessingConfig(),
    ) -> PreparedShotDensityProfiles:
        profiles = np.asarray(self.density_profiles_by_position_shot_time, dtype=np.float64)
        profiles = np.nan_to_num(profiles, nan=config.nan_fill_value)
        if config.clip_negative:
            profiles = np.clip(profiles, a_min=0.0, a_max=None)

        profiles = profiles.transpose(2, 1, 0)
        time_indices = np.repeat(np.arange(self.num_times), self.num_shots)
        shot_indices = np.tile(np.arange(self.num_shots), self.num_times)
        sample_times = self.time_axis_s[time_indices]
        flattened_profiles = profiles.reshape(self.num_times * self.num_shots, self.num_positions)

        mass = np.trapezoid(flattened_profiles, self.position_axis_m, axis=-1)
        keep_mask = np.ones_like(mass, dtype=bool)
        if config.drop_nonpositive_mass:
            keep_mask = mass > 0.0
            flattened_profiles = flattened_profiles[keep_mask]
            sample_times = sample_times[keep_mask]
            time_indices = time_indices[keep_mask]
            shot_indices = shot_indices[keep_mask]
            mass = mass[keep_mask]

        if config.normalize_each_profile and flattened_profiles.size > 0:
            safe_mass = np.where(mass > 0.0, mass, 1.0)
            flattened_profiles = flattened_profiles / safe_mass[:, None]

        return PreparedShotDensityProfiles(
            scan_id=self.asset.scan_id,
            scan_name=self.scan_name,
            doi_url=self.asset.doi_url,
            title=self.asset.title,
            position_axis_m=np.asarray(self.position_axis_m, dtype=np.float64),
            time_axis_s=np.asarray(self.time_axis_s, dtype=np.float64),
            density_profiles=flattened_profiles,
            sample_times_s=np.asarray(sample_times, dtype=np.float64),
            sample_time_indices=np.asarray(time_indices, dtype=np.int64),
            sample_shot_indices=np.asarray(shot_indices, dtype=np.int64),
            nan_fill_value=config.nan_fill_value,
            clip_negative=config.clip_negative,
            normalized=config.normalize_each_profile,
            dropped_nonpositive_mass=config.drop_nonpositive_mass,
        )


QUANTUM_GAS_TRANSPORT_SCANS: dict[str, RemoteScanAsset] = {
    "scan11879_56": RemoteScanAsset(
        scan_id="scan11879_56",
        filename="scan11879_56_data.mat",
        url="https://zenodo.org/records/16701012/files/scan11879_56_data.mat?download=1",
        md5="f746fcaa847686a8219782c2120eda55",
        doi_url="https://doi.org/10.5281/zenodo.16701012",
        title="Replication Data for: Characterising transport in a quantum gas by measuring Drude weights",
    ),
}


def available_quantum_gas_transport_scans() -> tuple[str, ...]:
    return tuple(sorted(QUANTUM_GAS_TRANSPORT_SCANS))


def fetch_quantum_gas_transport_scan(
    scan_id: str = "scan11879_56",
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    try:
        import pooch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pooch is required for dataset download. Install the 'data' extra."
        ) from exc

    asset = QUANTUM_GAS_TRANSPORT_SCANS.get(scan_id)
    if asset is None:
        raise KeyError(f"unknown scan_id: {scan_id}")

    cache_path = Path(cache_dir) if cache_dir is not None else Path(pooch.os_cache("spectral-packet-engine"))
    cache_path.mkdir(parents=True, exist_ok=True)
    fetched = pooch.retrieve(
        url=asset.url,
        known_hash=f"md5:{asset.md5}",
        path=cache_path,
        fname=asset.filename,
        progressbar=False,
    )
    return Path(fetched)


def load_quantum_gas_transport_scan(
    path: str | Path,
    *,
    scan_id: str = "scan11879_56",
) -> QuantumGasTransportScan:
    try:
        from scipy.io import loadmat
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scipy is required for MATLAB dataset loading. Install the 'data' extra."
        ) from exc

    asset = QUANTUM_GAS_TRANSPORT_SCANS.get(scan_id)
    if asset is None:
        raise KeyError(f"unknown scan_id: {scan_id}")

    loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    clean_data = loaded["clean_data"]

    return QuantumGasTransportScan(
        asset=asset,
        scan_name=str(clean_data.scan_name),
        slope_scan_name=str(clean_data.slope_scan_name),
        position_axis_m=np.asarray(clean_data.z_axis_si, dtype=np.float64),
        time_axis_s=np.asarray(clean_data.t_axis_si, dtype=np.float64),
        density_profiles_by_position_shot_time=np.asarray(
            clean_data.density_profiles_TOF_si_z_shots_t,
            dtype=np.float64,
        ),
        slope_profiles_by_position_time=np.asarray(clean_data.slope_profiles_si_z_shots, dtype=np.float64),
        temperature_nK=float(clean_data.T_nK),
        temperature_std_nK=float(clean_data.T_std_nK),
        transverse_trap_frequencies_hz=(
            float(clean_data.transv_trap_freq_hor_Hz),
            float(clean_data.transv_trap_freq_ver_Hz),
        ),
        tof_factors=(
            float(clean_data.N_TOF_factor),
            float(clean_data.slope_TOF_factor),
        ),
    )
def download_and_prepare_quantum_gas_transport_scan(
    scan_id: str = "scan11879_56",
    *,
    cache_dir: str | Path | None = None,
    preprocessing: DensityPreprocessingConfig = DensityPreprocessingConfig(),
) -> PreparedDensityProfiles:
    path = fetch_quantum_gas_transport_scan(scan_id=scan_id, cache_dir=cache_dir)
    scan = load_quantum_gas_transport_scan(path, scan_id=scan_id)
    return scan.prepare_profiles(preprocessing)
def download_and_prepare_quantum_gas_transport_shots(
    scan_id: str = "scan11879_56",
    *,
    cache_dir: str | Path | None = None,
    preprocessing: DensityPreprocessingConfig = DensityPreprocessingConfig(),
) -> PreparedShotDensityProfiles:
    path = fetch_quantum_gas_transport_scan(scan_id=scan_id, cache_dir=cache_dir)
    scan = load_quantum_gas_transport_scan(path, scan_id=scan_id)
    return scan.prepare_shot_profiles(preprocessing)


__all__ = [
    "DensityPreprocessingConfig",
    "PreparedDensityProfiles",
    "PreparedShotDensityProfiles",
    "QuantumGasTransportScan",
    "QUANTUM_GAS_TRANSPORT_SCANS",
    "available_quantum_gas_transport_scans",
    "download_and_prepare_quantum_gas_transport_shots",
    "download_and_prepare_quantum_gas_transport_scan",
    "fetch_quantum_gas_transport_scan",
    "load_quantum_gas_transport_scan",
]
