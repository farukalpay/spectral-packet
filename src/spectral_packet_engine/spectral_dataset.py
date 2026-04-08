from __future__ import annotations

"""Spectral dataset contract with grid metadata and lineage.

``TabularDataset`` remains the generic clean-table boundary.  This module is
the physics-aware dataset boundary: coordinates, units, uncertainty, split
regimes, and artifact lineage are explicit before data enter inverse,
reduced-model, benchmark, or surrogate workflows.
"""

from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping, Sequence

import torch

from spectral_packet_engine.domain import coerce_tensor
from spectral_packet_engine.table_io import ProfileTable

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class SpectralGridMetadata:
    axis_name: str
    coordinates: Tensor
    units: str = "dimensionless"

    def __post_init__(self) -> None:
        if not self.axis_name.strip():
            raise ValueError("axis_name must be non-empty")
        coordinates = coerce_tensor(self.coordinates, dtype=torch.float64)
        if coordinates.ndim != 1 or coordinates.numel() == 0:
            raise ValueError("coordinates must be a non-empty one-dimensional tensor")
        if not torch.isfinite(coordinates).all().item():
            raise ValueError("coordinates must be finite")
        if coordinates.numel() > 1 and not torch.all(coordinates[1:] > coordinates[:-1]).item():
            raise ValueError("coordinates must be strictly increasing")
        object.__setattr__(self, "coordinates", coordinates.detach())

    @property
    def size(self) -> int:
        return int(self.coordinates.numel())

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_name": self.axis_name,
            "units": self.units,
            "size": self.size,
            "min": float(self.coordinates[0].item()),
            "max": float(self.coordinates[-1].item()),
        }


@dataclass(frozen=True, slots=True)
class SpectralUncertainty:
    model: str
    scale: float | None = None
    values: Tensor | None = None
    units: str = "same-as-values"

    def __post_init__(self) -> None:
        model = self.model.strip().lower()
        object.__setattr__(self, "model", model)
        if model not in {"none", "independent-gaussian", "poisson-approx", "provided-standard-deviation"}:
            raise ValueError(
                "uncertainty model must be one of: none, independent-gaussian, poisson-approx, provided-standard-deviation"
            )
        if self.scale is not None and self.scale < 0.0:
            raise ValueError("uncertainty scale must be non-negative")
        if self.values is not None:
            values = coerce_tensor(self.values, dtype=torch.float64)
            if torch.any(values < 0).item():
                raise ValueError("uncertainty values must be non-negative")
            object.__setattr__(self, "values", values.detach())

    def for_values(self, values: Tensor) -> Tensor:
        observed = coerce_tensor(values, dtype=torch.float64)
        if self.model == "none":
            return torch.zeros_like(observed)
        if self.values is not None:
            if self.values.shape != observed.shape:
                raise ValueError("provided uncertainty values must match dataset values")
            return self.values.to(dtype=observed.dtype, device=observed.device)
        scale = 0.0 if self.scale is None else float(self.scale)
        if self.model == "poisson-approx":
            return scale * torch.sqrt(torch.clamp(observed, min=0.0))
        return torch.full_like(observed, scale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "scale": self.scale,
            "units": self.units,
            "has_values": self.values is not None,
        }


@dataclass(frozen=True, slots=True)
class SpectralDatasetSplit:
    name: str
    indices: tuple[int, ...]
    regime: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("split name must be non-empty")
        indices = tuple(int(index) for index in self.indices)
        if len(indices) != len(set(indices)):
            raise ValueError("split indices must be unique")
        if any(index < 0 for index in indices):
            raise ValueError("split indices must be non-negative")
        object.__setattr__(self, "indices", indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "regime": self.regime,
            "size": len(self.indices),
            "indices": list(self.indices),
        }


@dataclass(frozen=True, slots=True)
class ArtifactLineage:
    source: str | None = None
    transform: str | None = None
    parent_artifact_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_artifact_hashes",
            tuple(str(value) for value in self.parent_artifact_hashes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "transform": self.transform,
            "parent_artifact_hashes": list(self.parent_artifact_hashes),
        }


@dataclass(frozen=True, slots=True)
class SpectralDataset:
    name: str
    values: Tensor
    grids: tuple[SpectralGridMetadata, ...]
    value_semantics: str
    value_units: str = "dimensionless"
    uncertainty: SpectralUncertainty = field(default_factory=lambda: SpectralUncertainty(model="none"))
    splits: tuple[SpectralDatasetSplit, ...] = ()
    lineage: ArtifactLineage = field(default_factory=ArtifactLineage)
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("dataset name must be non-empty")
        values = coerce_tensor(self.values, dtype=torch.float64)
        if values.ndim == 0:
            raise ValueError("dataset values must have at least one dimension")
        if not torch.isfinite(values).all().item():
            raise ValueError("dataset values must be finite")
        if not self.grids:
            raise ValueError("at least one grid metadata object is required")
        expected_shape = tuple(grid.size for grid in self.grids)
        if tuple(values.shape) != expected_shape:
            raise ValueError(f"dataset values shape {tuple(values.shape)} must match grid shape {expected_shape}")
        uncertainty_values = self.uncertainty.for_values(values)
        if uncertainty_values.shape != values.shape:
            raise ValueError("uncertainty must resolve to the same shape as values")
        split_names = [split.name for split in self.splits]
        if len(split_names) != len(set(split_names)):
            raise ValueError("split names must be unique")
        sample_axis_size = values.shape[0]
        for split in self.splits:
            if split.indices and max(split.indices) >= sample_axis_size:
                raise ValueError("split index exceeds the leading dataset axis")
        object.__setattr__(self, "values", values.detach())
        object.__setattr__(self, "metadata", {} if self.metadata is None else dict(self.metadata))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.values.shape)

    @property
    def content_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.name.encode("utf-8"))
        digest.update(self.value_semantics.encode("utf-8"))
        digest.update(self.value_units.encode("utf-8"))
        digest.update(str(self.shape).encode("utf-8"))
        digest.update(self.values.detach().cpu().numpy().tobytes())
        for grid in self.grids:
            digest.update(grid.axis_name.encode("utf-8"))
            digest.update(grid.units.encode("utf-8"))
            digest.update(grid.coordinates.detach().cpu().numpy().tobytes())
        uncertainty_values = self.uncertainty.for_values(self.values).detach().cpu().numpy()
        digest.update(uncertainty_values.tobytes())
        return digest.hexdigest()

    def uncertainty_values(self) -> Tensor:
        return self.uncertainty.for_values(self.values).detach()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "value_semantics": self.value_semantics,
            "value_units": self.value_units,
            "content_hash": self.content_hash,
            "grids": [grid.to_dict() for grid in self.grids],
            "uncertainty": self.uncertainty.to_dict(),
            "splits": [split.to_dict() for split in self.splits],
            "lineage": self.lineage.to_dict(),
            "metadata": dict(self.metadata or {}),
        }


def spectral_dataset_from_profile_table(
    table: ProfileTable,
    *,
    name: str = "profile-table",
    position_units: str = "dimensionless",
    time_units: str = "dimensionless",
    value_units: str = "density",
    uncertainty: SpectralUncertainty | None = None,
    splits: Sequence[SpectralDatasetSplit] = (),
    lineage: ArtifactLineage | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SpectralDataset:
    return SpectralDataset(
        name=name,
        values=torch.as_tensor(table.profiles, dtype=torch.float64),
        grids=(
            SpectralGridMetadata("time", torch.as_tensor(table.sample_times, dtype=torch.float64), units=time_units),
            SpectralGridMetadata("position", torch.as_tensor(table.position_grid, dtype=torch.float64), units=position_units),
        ),
        value_semantics="profile_density",
        value_units=value_units,
        uncertainty=uncertainty or SpectralUncertainty(model="none"),
        splits=tuple(splits),
        lineage=lineage or ArtifactLineage(source=table.source, transform="ProfileTable -> SpectralDataset"),
        metadata=metadata,
    )


__all__ = [
    "ArtifactLineage",
    "SpectralDataset",
    "SpectralDatasetSplit",
    "SpectralGridMetadata",
    "SpectralUncertainty",
    "spectral_dataset_from_profile_table",
]
