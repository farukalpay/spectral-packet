from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from spectral_packet_engine.runtime_files import (
    atomic_output_path,
    cleanup_runtime_temporary_files,
    directory_lock,
    is_runtime_temporary_path,
)
from spectral_packet_engine.table_io import ProfileTable, save_profile_table_csv
from spectral_packet_engine.tabular import TabularDataset, save_tabular_dataset_csv


_ARTIFACT_LOCK_TIMEOUT_SECONDS = 5.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _serialize_float(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_artifact_files(path: str | Path) -> list[str]:
    directory = Path(path)
    if not directory.exists():
        return []
    return [
        str(item.relative_to(directory))
        for item in sorted(directory.rglob("*"))
        if item.is_file() and not is_runtime_temporary_path(item)
    ]


def to_serializable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_serializable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, float):
        return _serialize_float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if detached.ndim == 0:
            return to_serializable(detached.item())
        return to_serializable(detached.tolist())
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return to_serializable(value.item())
        return to_serializable(value.tolist())
    if isinstance(value, np.generic):
        return to_serializable(value.item())
    if isinstance(value, complex):
        return {"real": to_serializable(value.real), "imag": to_serializable(value.imag)}
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    serialized = json.dumps(to_serializable(payload), indent=2, sort_keys=True) + "\n"
    with atomic_output_path(output_path) as temporary_path:
        temporary_path.write_text(serialized, encoding="utf-8")
    return output_path


def write_rows_csv(
    path: str | Path,
    header: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> Path:
    import csv

    output_path = Path(path)
    with atomic_output_path(output_path) as temporary_path:
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(list(header))
            for row in rows:
                writer.writerow([to_serializable(item) for item in row])
    return output_path


@dataclass(frozen=True, slots=True)
class ArtifactDirectoryReport:
    output_dir: str
    exists: bool
    complete: bool
    files: tuple[str, ...]
    metadata: dict[str, Any]
    stale_temporary_files: tuple[str, ...]
    lock_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "exists": self.exists,
            "complete": self.complete,
            "files": list(self.files),
            "metadata": to_serializable(self.metadata),
            "stale_temporary_files": list(self.stale_temporary_files),
            "lock_active": self.lock_active,
        }


def read_artifact_index(path: str | Path) -> dict[str, Any] | None:
    directory = Path(path)
    index_path = directory / "artifacts.json"
    if not index_path.exists():
        return None
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact index must contain a top-level object")
    return payload


def inspect_artifact_directory(path: str | Path) -> ArtifactDirectoryReport:
    directory = Path(path)
    exists = directory.exists()
    files = tuple(list_artifact_files(directory))
    payload = None if not exists else read_artifact_index(directory)
    metadata = {}
    complete = False
    if isinstance(payload, dict):
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
        complete = payload.get("status") == "complete"
    stale_temporary_files = ()
    if exists:
        stale_temporary_files = tuple(
            item.name
            for item in sorted(directory.iterdir())
            if item.is_file() and is_runtime_temporary_path(item)
        )
    return ArtifactDirectoryReport(
        output_dir=str(directory),
        exists=exists,
        complete=complete,
        files=files,
        metadata=metadata,
        stale_temporary_files=stale_temporary_files,
        lock_active=(directory / ".spectral-packet-engine.lock").exists() if exists else False,
    )


@contextmanager
def _artifact_output_session(output_dir: str | Path):
    directory = ensure_directory(output_dir)
    cleanup_runtime_temporary_files(directory)
    with directory_lock(directory, timeout_seconds=_ARTIFACT_LOCK_TIMEOUT_SECONDS):
        cleanup_runtime_temporary_files(directory)
        yield directory
        cleanup_runtime_temporary_files(directory)


def write_artifact_index(path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> Path:
    directory = ensure_directory(path)
    index_path = directory / "artifacts.json"
    files = list_artifact_files(directory)
    if index_path.name not in files:
        files.append(index_path.name)
    files.sort()
    payload = {
        "status": "complete",
        "generated_at_utc": _utc_now_iso(),
        "files": files,
        "metadata": {} if metadata is None else to_serializable(dict(metadata)),
    }
    return write_json(index_path, payload)


def _artifact_metadata(
    base: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(base)
    if extra is not None:
        merged.update(dict(extra))
    return merged


def write_tabular_artifacts(
    output_dir: str | Path,
    dataset: TabularDataset,
    *,
    summary_name: str = "tabular_summary.json",
    table_name: str = "table.csv",
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(
            directory / summary_name,
            {
                "source": None if dataset.source is None else dataset.source,
                "row_count": dataset.row_count,
                "column_names": list(dataset.column_names),
                "schema": dataset.schema,
                "validation": dataset.validation_report(),
                "preview_rows": dataset.to_rows(limit=10),
            },
        )
        save_tabular_dataset_csv(dataset, directory / table_name)
        write_artifact_index(directory, metadata=metadata)
        return directory


def profile_table_from_tensors(grid, times, profiles, *, source: str | None = None) -> ProfileTable:
    return ProfileTable(
        position_grid=torch.as_tensor(grid).detach().cpu().numpy(),
        sample_times=torch.as_tensor(times).detach().cpu().numpy(),
        profiles=torch.as_tensor(profiles).detach().cpu().numpy(),
        source=source,
    )


def write_forward_artifacts(
    output_dir: str | Path,
    summary: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "forward_summary.json", summary)
        save_profile_table_csv(
            profile_table_from_tensors(summary.grid, summary.times, summary.densities, source="forward"),
            directory / "forward_densities.csv",
        )
        write_artifact_index(directory, metadata=_artifact_metadata({"workflow": "forward"}, metadata))
        return directory


def write_compression_artifacts(
    output_dir: str | Path,
    summary: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "compression_summary.json", summary)
        save_profile_table_csv(
            profile_table_from_tensors(
                summary.grid,
                summary.sample_times,
                summary.reconstruction,
                source=summary.source,
            ),
            directory / "reconstruction.csv",
        )
        header = ["time", *[f"mode_{index}" for index in range(1, summary.coefficients.shape[-1] + 1)]]
        rows = [
            [sample_time, *row]
            for sample_time, row in zip(
                summary.sample_times.detach().cpu().tolist(),
                summary.coefficients.detach().cpu().tolist(),
            )
        ]
        write_rows_csv(directory / "coefficients.csv", header, rows)
        write_artifact_index(directory, metadata=_artifact_metadata({"workflow": "compress-table"}, metadata))
        return directory


def write_profile_table_report_artifacts(
    output_dir: str | Path,
    report: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "profile_table_report.json", report.overview)
        write_json(directory / "profile_table_summary.json", report.inspection)
        write_spectral_analysis_artifacts(
            directory / "analysis",
            report.analysis,
            metadata={"parent_workflow": "profile-report"},
        )
        write_compression_artifacts(
            directory / "compression",
            report.compression,
            metadata={"parent_workflow": "profile-report"},
        )
        write_artifact_index(
            directory,
            metadata=_artifact_metadata(
                {
                    "workflow": "profile-report",
                    "source": getattr(report.overview, "source", None),
                    "num_samples": getattr(report.overview, "num_samples", None),
                    "num_positions": getattr(report.overview, "num_positions", None),
                    "analyze_num_modes": getattr(report.overview, "analyze_num_modes", None),
                    "compress_num_modes": getattr(report.overview, "compress_num_modes", None),
                    "normalize_each_profile": getattr(report.overview, "normalize_each_profile", None),
                },
                metadata,
            ),
        )
        return directory


def write_compression_sweep_artifacts(output_dir: str | Path, summary: Any) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "compression_sweep.json", summary)
        rows = [
            [mode, mean_error, max_error]
            for mode, mean_error, max_error in zip(
                summary.mode_counts.detach().cpu().tolist(),
                summary.mean_relative_l2_error.detach().cpu().tolist(),
                summary.max_relative_l2_error.detach().cpu().tolist(),
            )
        ]
        write_rows_csv(
            directory / "compression_sweep.csv",
            ["mode_count", "mean_relative_l2_error", "max_relative_l2_error"],
            rows,
        )
        write_artifact_index(directory, metadata={"workflow": "compression-sweep"})
        return directory


def write_inverse_artifacts(
    output_dir: str | Path,
    summary: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "inverse_fit.json", summary)
        save_profile_table_csv(
            profile_table_from_tensors(
                summary.observation_grid,
                summary.times,
                summary.predicted_density,
                source="inverse_fit_prediction",
            ),
            directory / "predicted_density.csv",
        )
        write_artifact_index(directory, metadata=_artifact_metadata({"workflow": "fit-table"}, metadata))
        return directory


def write_packet_sweep_artifacts(output_dir: str | Path, summary: Any) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "packet_sweep.json", summary)
        rows = [
            [
                item.center,
                item.width,
                item.wavenumber,
                item.phase,
                item.spectral_norm,
                item.final_expectation_position,
                item.final_left_probability,
                item.final_right_probability,
                item.final_total_probability,
            ]
            for item in summary.items
        ]
        write_rows_csv(
            directory / "packet_sweep.csv",
            [
                "center",
                "width",
                "wavenumber",
                "phase",
                "spectral_norm",
                "final_expectation_position",
                "final_left_probability",
                "final_right_probability",
                "final_total_probability",
            ],
            rows,
        )
        write_artifact_index(directory, metadata={"workflow": "packet-sweep"})
        return directory


def write_transport_benchmark_artifacts(output_dir: str | Path, summary: Any) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "transport_benchmark.json", summary)
        write_artifact_index(directory, metadata={"workflow": "transport-benchmark"})
        return directory


def write_tensorflow_training_artifacts(output_dir: str | Path, summary: Any) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "tf_training.json", summary)
        write_artifact_index(directory, metadata={"workflow": "tf-train-table"})
        return directory


def write_modal_training_artifacts(
    output_dir: str | Path,
    summary: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "ml_training.json", summary)
        write_artifact_index(
            directory,
            metadata=_artifact_metadata(
                {"workflow": "ml-train-table", "backend": getattr(summary, "backend", None)},
                metadata,
            ),
        )
        return directory


def _write_coefficient_rows(path: str | Path, sample_times, coefficients) -> Path:
    coefficient_rows = [
        [sample_time, *row]
        for sample_time, row in zip(
            torch.as_tensor(sample_times).detach().cpu().tolist(),
            torch.as_tensor(coefficients).detach().cpu().tolist(),
        )
    ]
    header = ["time", *[f"mode_{index}" for index in range(1, torch.as_tensor(coefficients).shape[-1] + 1)]]
    return write_rows_csv(path, header, coefficient_rows)


def write_spectral_analysis_artifacts(
    output_dir: str | Path,
    summary: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "spectral_analysis.json", summary)
        _write_coefficient_rows(directory / "coefficients.csv", summary.sample_times, summary.coefficients)
        write_rows_csv(
            directory / "sample_metrics.csv",
            ["time", "mass", "mean_position", "width"],
            zip(
                torch.as_tensor(summary.sample_times).detach().cpu().tolist(),
                torch.as_tensor(summary.mass).detach().cpu().tolist(),
                torch.as_tensor(summary.mean_position).detach().cpu().tolist(),
                torch.as_tensor(summary.width).detach().cpu().tolist(),
            ),
        )
        write_rows_csv(
            directory / "mean_modal_weights.csv",
            ["mode", "mean_modal_weight", "max_modal_weight", "mean_cumulative_mass", "max_tail_mass"],
            zip(
                torch.as_tensor(summary.spectral_summary.mode_numbers).detach().cpu().tolist(),
                torch.as_tensor(summary.spectral_summary.mean_modal_weights).detach().cpu().tolist(),
                torch.as_tensor(summary.spectral_summary.max_modal_weights).detach().cpu().tolist(),
                torch.as_tensor(summary.spectral_summary.mean_cumulative_mass).detach().cpu().tolist(),
                torch.as_tensor(summary.spectral_summary.max_tail_mass).detach().cpu().tolist(),
            ),
        )
        write_artifact_index(directory, metadata=_artifact_metadata({"workflow": "analyze-table"}, metadata))
        return directory


def write_profile_comparison_artifacts(output_dir: str | Path, summary: Any) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "table_comparison.json", summary)
        save_profile_table_csv(
            profile_table_from_tensors(
                summary.grid,
                summary.sample_times,
                summary.residual_profiles,
                source="candidate_minus_reference",
            ),
            directory / "residual_profiles.csv",
        )
        write_rows_csv(
            directory / "sample_metrics.csv",
            [
                "time",
                "relative_l2_error",
                "mass_error",
                "mean_position_error",
                "width_error",
            ],
            zip(
                torch.as_tensor(summary.sample_times).detach().cpu().tolist(),
                torch.as_tensor(summary.comparison.relative_l2_error).detach().cpu().tolist(),
                torch.as_tensor(summary.comparison.mass_error).detach().cpu().tolist(),
                torch.as_tensor(summary.comparison.mean_position_error).detach().cpu().tolist(),
                torch.as_tensor(summary.comparison.width_error).detach().cpu().tolist(),
            ),
        )
        write_artifact_index(directory, metadata={"workflow": "compare-tables"})
        return directory


def write_tensorflow_evaluation_artifacts(
    output_dir: str | Path,
    summary: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "tf_evaluation.json", summary)
        save_profile_table_csv(
            profile_table_from_tensors(
                summary.grid,
                summary.sample_times,
                summary.reconstructed_profiles,
                source="tf_reconstruction",
            ),
            directory / "tf_reconstruction.csv",
        )
        _write_coefficient_rows(directory / "tf_coefficients.csv", summary.sample_times, summary.predicted_coefficients)
        write_rows_csv(
            directory / "tf_predicted_moments.csv",
            ["time", "predicted_mean_position", "predicted_width"],
            zip(
                torch.as_tensor(summary.sample_times).detach().cpu().tolist(),
                torch.as_tensor(summary.predicted_moments)[..., 0].detach().cpu().tolist(),
                torch.as_tensor(summary.predicted_moments)[..., 1].detach().cpu().tolist(),
            ),
        )
        write_artifact_index(directory, metadata=_artifact_metadata({"workflow": "tf-evaluate-table"}, metadata))
        return directory


def write_modal_evaluation_artifacts(
    output_dir: str | Path,
    summary: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "ml_evaluation.json", summary)
        save_profile_table_csv(
            profile_table_from_tensors(
                summary.grid,
                summary.sample_times,
                summary.reconstructed_profiles,
                source=f"{summary.backend}_reconstruction",
            ),
            directory / "ml_reconstruction.csv",
        )
        _write_coefficient_rows(directory / "ml_coefficients.csv", summary.sample_times, summary.predicted_coefficients)
        write_rows_csv(
            directory / "ml_predicted_moments.csv",
            ["time", "predicted_mean_position", "predicted_width"],
            zip(
                torch.as_tensor(summary.sample_times).detach().cpu().tolist(),
                torch.as_tensor(summary.predicted_moments)[..., 0].detach().cpu().tolist(),
                torch.as_tensor(summary.predicted_moments)[..., 1].detach().cpu().tolist(),
            ),
        )
        write_artifact_index(
            directory,
            metadata=_artifact_metadata(
                {"workflow": "ml-evaluate-table", "backend": getattr(summary, "backend", None)},
                metadata,
            ),
        )
        return directory


__all__ = [
    "ArtifactDirectoryReport",
    "ensure_directory",
    "inspect_artifact_directory",
    "list_artifact_files",
    "profile_table_from_tensors",
    "read_artifact_index",
    "to_serializable",
    "write_tabular_artifacts",
    "write_compression_artifacts",
    "write_profile_table_report_artifacts",
    "write_compression_sweep_artifacts",
    "write_artifact_index",
    "write_forward_artifacts",
    "write_inverse_artifacts",
    "write_modal_evaluation_artifacts",
    "write_modal_training_artifacts",
    "write_json",
    "write_packet_sweep_artifacts",
    "write_profile_comparison_artifacts",
    "write_rows_csv",
    "write_spectral_analysis_artifacts",
    "write_tensorflow_evaluation_artifacts",
    "write_tensorflow_training_artifacts",
    "write_transport_benchmark_artifacts",
]
