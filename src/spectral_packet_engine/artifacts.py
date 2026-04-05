from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
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
from spectral_packet_engine.tabular import TabularDataset, save_tabular_dataset, save_tabular_dataset_csv


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


def _write_parameter_posterior_csv(path: str | Path, parameter_posterior: Any) -> Path:
    return write_rows_csv(
        path,
        [
            "parameter",
            "mean",
            "standard_deviation",
            "confidence_interval_low",
            "confidence_interval_high",
        ],
        (
            [
                name,
                mean,
                std,
                low,
                high,
            ]
            for name, mean, std, low, high in zip(
                parameter_posterior.parameter_names,
                torch.as_tensor(parameter_posterior.mean).detach().cpu().tolist(),
                torch.as_tensor(parameter_posterior.standard_deviation).detach().cpu().tolist(),
                torch.as_tensor(parameter_posterior.confidence_interval_low).detach().cpu().tolist(),
                torch.as_tensor(parameter_posterior.confidence_interval_high).detach().cpu().tolist(),
            )
        ),
    )


def _write_coefficient_posterior_csv(path: str | Path, coefficient_posterior: Any) -> Path:
    mean = torch.as_tensor(coefficient_posterior.mean)
    return write_rows_csv(
        path,
        [
            "mode",
            "mean_real",
            "mean_imag",
            "real_standard_deviation",
            "imag_standard_deviation",
            "magnitude_standard_deviation",
        ],
        (
            [
                index + 1,
                complex_value.real,
                complex_value.imag,
                real_std,
                imag_std,
                magnitude_std,
            ]
            for index, (complex_value, real_std, imag_std, magnitude_std) in enumerate(
                zip(
                    mean.detach().cpu().reshape(-1).tolist(),
                    torch.as_tensor(coefficient_posterior.real_standard_deviation)
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .tolist(),
                    torch.as_tensor(coefficient_posterior.imag_standard_deviation)
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .tolist(),
                    torch.as_tensor(coefficient_posterior.magnitude_standard_deviation)
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .tolist(),
                )
            )
        ),
    )


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


def _copy_exported_path(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == destination_path.resolve():
        return destination_path
    if source_path.is_dir():
        if destination_path.exists():
            shutil.rmtree(destination_path)
        shutil.copytree(source_path, destination_path)
        return destination_path
    shutil.copy2(source_path, destination_path)
    return destination_path


def _feature_column_semantic_meaning(
    name: str,
    *,
    identifier_columns: Sequence[str],
) -> str:
    if name in identifier_columns:
        if name == "time":
            return "Sample time for the profile row."
        return "Identifier column preserved in the exported feature table."
    if name.startswith("mode_"):
        suffix = name.removeprefix("mode_")
        if suffix.isdigit():
            return f"Spectral modal coefficient for mode {int(suffix)}."
    if name == "mean_position":
        return "Profile mean position on the bounded 1D domain."
    if name == "width":
        return "Profile width on the bounded 1D domain."
    if name == "mass":
        return "Integrated profile mass over the bounded 1D domain."
    return "Feature column exported for downstream spectral-model workflows."


def _feature_table_schema_payload(summary: Any) -> dict[str, Any]:
    identifier_columns = tuple(getattr(summary, "identifier_columns", ()))
    schema_by_name = {column.name: column for column in summary.table.schema.columns}
    column_order = tuple(summary.table.column_names)
    return {
        "column_order": list(column_order),
        "columns": [
            {
                "name": name,
                "order": index,
                "dtype": schema_by_name[name].dtype,
                "nullable": schema_by_name[name].nullable,
                "semantic_role": "identifier" if name in identifier_columns else "feature",
                "semantic_meaning": _feature_column_semantic_meaning(
                    name,
                    identifier_columns=identifier_columns,
                ),
            }
            for index, name in enumerate(column_order)
        ],
        "identifier_columns": list(identifier_columns),
        "feature_names": list(getattr(summary, "feature_names", ())),
        "includes": list(getattr(summary, "includes", ())),
        "ordering": getattr(summary, "ordering", {}),
        "library_versions": getattr(summary, "library_versions", {}),
    }


def write_feature_table_artifacts(
    output_dir: str | Path,
    summary: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    feature_format = getattr(summary, "format", "csv")
    if feature_format not in {"csv", "parquet"}:
        raise ValueError("Feature-table artifacts require format 'csv' or 'parquet'.")
    with _artifact_output_session(output_dir) as directory:
        feature_path = directory / f"features.{feature_format}"
        save_tabular_dataset(summary.table, feature_path)
        write_json(
            directory / "feature_table_export.json",
            {
                "source_kind": getattr(summary, "source_kind", None),
                "source_location": getattr(summary, "source_location", None),
                "identifier_columns": list(getattr(summary, "identifier_columns", ())),
                "feature_names": list(getattr(summary, "feature_names", ())),
                "includes": list(getattr(summary, "includes", ())),
                "num_rows": getattr(summary, "num_rows", None),
                "num_features": getattr(summary, "num_features", None),
                "num_modes": getattr(summary, "num_modes", None),
                "normalize_each_profile": getattr(summary, "normalize_each_profile", None),
                "format": feature_format,
                "output_path": str(feature_path),
                "ordering": getattr(summary, "ordering", {}),
                "library_versions": getattr(summary, "library_versions", {}),
                "metadata": getattr(summary, "metadata", {}),
            },
        )
        write_json(
            directory / "features_schema.json",
            _feature_table_schema_payload(summary),
        )
        summary_metadata = dict(getattr(summary, "metadata", {}))
        input_metadata = summary_metadata.get("input", {})
        if not isinstance(input_metadata, dict):
            input_metadata = {}
        write_artifact_index(
            directory,
            metadata=_artifact_metadata(
                {
                    **summary_metadata,
                    "workflow": "export-features",
                    "format": feature_format,
                    "input_kind": input_metadata.get("kind"),
                    "num_modes": getattr(summary, "num_modes", None),
                    "num_features": getattr(summary, "num_features", None),
                    "normalize_each_profile": getattr(summary, "normalize_each_profile", None),
                    "ordering": getattr(summary, "ordering", {}),
                    "library_versions": getattr(summary, "library_versions", {}),
                },
                metadata,
            ),
        )
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
        physical_inference = getattr(summary, "physical_inference", None)
        if physical_inference is not None:
            write_json(directory / "uncertainty_summary.json", physical_inference)
            parameter_posterior = getattr(physical_inference, "parameter_posterior", None)
            if parameter_posterior is not None:
                _write_parameter_posterior_csv(directory / "parameter_posterior.csv", parameter_posterior)
            coefficient_posterior = getattr(physical_inference, "coefficient_posterior", None)
            if coefficient_posterior is not None:
                _write_coefficient_posterior_csv(directory / "modal_posterior.csv", coefficient_posterior)
            sensitivity = getattr(physical_inference, "sensitivity", None)
            if sensitivity is not None:
                write_json(directory / "sensitivity_map.json", sensitivity)
        artifact_metadata = {"workflow": "fit-table", "has_physical_inference": physical_inference is not None}
        if physical_inference is not None:
            parameter_posterior = getattr(physical_inference, "parameter_posterior", None)
            if parameter_posterior is not None:
                artifact_metadata["identifiability_score"] = getattr(
                    parameter_posterior,
                    "identifiability_score",
                    None,
                )
                artifact_metadata["noise_scale"] = getattr(parameter_posterior, "noise_scale", None)
        write_artifact_index(directory, metadata=_artifact_metadata(artifact_metadata, metadata))
        return directory


def write_potential_inference_artifacts(
    output_dir: str | Path,
    summary: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "potential_family_inference.json", summary)
        write_rows_csv(
            directory / "candidate_ranking.csv",
            [
                "family",
                "final_loss",
                "residual_sum_squares",
                "information_criterion",
                "relative_evidence_weight",
            ],
            [
                [
                    candidate.family,
                    candidate.calibration.final_loss,
                    candidate.residual_sum_squares,
                    candidate.information_criterion,
                    candidate.relative_evidence_weight,
                ]
                for candidate in summary.candidates
            ],
        )
        best = summary.candidates[0]
        write_json(directory / "best_family_calibration.json", best.calibration)
        if best.calibration.parameter_posterior is not None:
            _write_parameter_posterior_csv(
                directory / "best_family_parameter_posterior.csv",
                best.calibration.parameter_posterior,
            )
        if best.calibration.sensitivity is not None:
            write_json(directory / "best_family_sensitivity_map.json", best.calibration.sensitivity)
        write_artifact_index(
            directory,
            metadata=_artifact_metadata(
                {
                    "workflow": "infer-potential-spectrum",
                    "best_family": summary.best_family,
                    "candidate_count": len(summary.candidates),
                },
                metadata,
            ),
        )
        return directory


def write_reduced_model_artifacts(
    output_dir: str | Path,
    summary: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "reduced_model_summary.json", summary)
        model_type = type(summary).__name__
        if hasattr(summary, "combined_eigenvalues"):
            write_rows_csv(
                directory / "combined_spectrum.csv",
                ["state_index", "eigenvalue", "state_index_x", "state_index_y"],
                [
                    [index + 1, value, pair[0], pair[1]]
                    for index, (value, pair) in enumerate(
                        zip(
                            torch.as_tensor(summary.combined_eigenvalues).detach().cpu().tolist(),
                            summary.state_index_pairs,
                        )
                    )
                ],
            )
        if hasattr(summary, "adiabatic_potentials"):
            grid = torch.as_tensor(summary.grid).detach().cpu().tolist()
            adiabatic = torch.as_tensor(summary.adiabatic_potentials).detach().cpu()
            write_rows_csv(
                directory / "adiabatic_surfaces.csv",
                ["position", "surface_1", "surface_2"],
                [
                    [position, adiabatic[0, index].item(), adiabatic[1, index].item()]
                    for index, position in enumerate(grid)
                ],
            )
        if hasattr(summary, "effective_potential"):
            grid = torch.as_tensor(summary.radial_grid).detach().cpu().tolist()
            potential = torch.as_tensor(summary.effective_potential).detach().cpu().tolist()
            write_rows_csv(
                directory / "effective_potential.csv",
                ["radius", "effective_potential"],
                [[radius, value] for radius, value in zip(grid, potential)],
            )
        if hasattr(summary, "singular_values"):
            write_rows_csv(
                directory / "singular_values.csv",
                ["index", "singular_value"],
                [
                    [index + 1, value]
                    for index, value in enumerate(torch.as_tensor(summary.singular_values).detach().cpu().tolist())
                ],
            )
        write_artifact_index(
            directory,
            metadata=_artifact_metadata({"workflow": "reduced-model", "model_type": model_type}, metadata),
        )
        return directory


def write_differentiable_artifacts(
    output_dir: str | Path,
    summary: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "differentiable_summary.json", summary)
        workflow_name = "differentiable-physics"
        if hasattr(summary, "target_transition"):
            workflow_name = "design-transition"
            write_rows_csv(
                directory / "transition_design_spectrum.csv",
                ["state_index", "predicted_eigenvalue"],
                [
                    [index + 1, value]
                    for index, value in enumerate(torch.as_tensor(summary.predicted_eigenvalues).detach().cpu().tolist())
                ],
            )
            write_rows_csv(
                directory / "transition_gradient.csv",
                ["parameter", "gradient"],
                [
                    [name, gradient]
                    for name, gradient in zip(
                        summary.parameter_names,
                        torch.as_tensor(summary.gradient).detach().cpu().tolist(),
                    )
                ],
            )
        elif hasattr(summary, "predicted_eigenvalues") and hasattr(summary, "target_eigenvalues"):
            workflow_name = "calibrate-potential"
            write_rows_csv(
                directory / "predicted_eigenvalues.csv",
                ["state_index", "target_eigenvalue", "predicted_eigenvalue"],
                [
                    [index + 1, target, predicted]
                    for index, (target, predicted) in enumerate(
                        zip(
                            torch.as_tensor(summary.target_eigenvalues).detach().cpu().tolist(),
                            torch.as_tensor(summary.predicted_eigenvalues).detach().cpu().tolist(),
                        )
                    )
                ],
            )
            if getattr(summary, "parameter_posterior", None) is not None:
                _write_parameter_posterior_csv(directory / "parameter_posterior.csv", summary.parameter_posterior)
            if getattr(summary, "sensitivity", None) is not None:
                write_json(directory / "sensitivity_map.json", summary.sensitivity)
        elif hasattr(summary, "final_density"):
            workflow_name = "optimize-packet-control"
            write_rows_csv(
                directory / "optimization_history.csv",
                ["step", "loss"],
                [[index, loss] for index, loss in enumerate(summary.history)],
            )
            write_rows_csv(
                directory / "final_density.csv",
                ["position", "density"],
                [
                    [position, density]
                    for position, density in zip(
                        torch.as_tensor(summary.observation_grid).detach().cpu().tolist(),
                        torch.as_tensor(summary.final_density).detach().cpu().tolist(),
                    )
                ],
            )
            write_rows_csv(
                directory / "objective_gradient.csv",
                ["parameter", "gradient"],
                [
                    [name, gradient]
                    for name, gradient in zip(
                        summary.gradient_summary.parameter_names,
                        torch.as_tensor(summary.gradient_summary.gradient).detach().cpu().tolist(),
                    )
                ],
            )
        write_artifact_index(directory, metadata=_artifact_metadata({"workflow": workflow_name}, metadata))
        return directory


def write_vertical_workflow_artifacts(
    output_dir: str | Path,
    summary: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "vertical_workflow_summary.json", summary)
        workflow_name = "vertical-workflow"
        if hasattr(summary, "family_inference"):
            workflow_name = "spectroscopy-workflow"
            write_potential_inference_artifacts(
                directory / "family_inference",
                summary.family_inference,
                metadata={"parent_workflow": workflow_name},
            )
            write_rows_csv(
                directory / "observed_transitions.csv",
                ["transition_index", "observed_transition", "best_family_transition"],
                [
                    [index + 1, observed, predicted]
                    for index, (observed, predicted) in enumerate(
                        zip(
                            torch.as_tensor(summary.observed_transition_energies).detach().cpu().tolist(),
                            torch.as_tensor(summary.best_family_transition_energies).detach().cpu().tolist(),
                        )
                    )
                ],
            )
        elif hasattr(summary, "tunneling"):
            workflow_name = "transport-workflow"
            write_json(directory / "transport_report.json", summary.tunneling)
            write_rows_csv(
                directory / "resonances.csv",
                ["index", "resonance_energy", "resonance_width"],
                [
                    [index + 1, energy, width]
                    for index, (energy, width) in enumerate(
                        zip(summary.tunneling.resonance_energies, summary.tunneling.resonance_widths)
                    )
                ],
            )
        elif hasattr(summary, "optimization"):
            workflow_name = "control-workflow"
            write_differentiable_artifacts(
                directory / "optimization",
                summary.optimization,
                metadata={"parent_workflow": workflow_name},
            )
        elif hasattr(summary, "inverse_fit") and hasattr(summary, "feature_export"):
            workflow_name = "profile-inference-workflow"
            write_profile_table_report_artifacts(
                directory / "report",
                summary.report,
                metadata={"parent_workflow": workflow_name},
            )
            write_inverse_artifacts(
                directory / "inverse",
                summary.inverse_fit,
                metadata={"parent_workflow": workflow_name},
            )
            write_feature_table_artifacts(
                directory / "features",
                summary.feature_export,
                metadata={"parent_workflow": workflow_name},
            )
        write_artifact_index(directory, metadata=_artifact_metadata({"workflow": workflow_name}, metadata))
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


def write_tree_training_artifacts(
    output_dir: str | Path,
    summary: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    library_name = getattr(summary, "library", None)
    library_version = getattr(getattr(summary, "library_runtime", None), "version", None)
    library_versions = {}
    if library_name is not None and library_version is not None:
        library_versions[str(library_name)] = library_version
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "tree_training.json", summary)
        write_rows_csv(
            directory / "predictions.csv",
            list(summary.predictions.column_names),
            ([row[name] for name in summary.predictions.column_names] for row in summary.predictions.to_rows()),
        )
        write_rows_csv(
            directory / "feature_importance.csv",
            ["feature_name", "importance"],
            (
                [item.feature_name, item.importance]
                for item in getattr(summary, "feature_importances", ())
            ),
        )
        export_path = getattr(summary, "export_path", None)
        if export_path is None:
            raise ValueError("Tree training artifacts require summary.export_path. Pass export_dir when training.")
        exported_model_path = Path(export_path)
        model_target = directory / exported_model_path.name
        _copy_exported_path(exported_model_path, model_target)
        write_artifact_index(
            directory,
            metadata=_artifact_metadata(
                {
                    "workflow": "tree-train",
                    "source_kind": getattr(summary, "source_kind", None),
                    "library": library_name,
                    "library_version": library_version,
                    "library_versions": library_versions,
                    "model": getattr(summary, "model", None),
                    "task": getattr(summary, "task", None),
                    "target_column": getattr(summary, "target_column", None),
                    "num_features": getattr(summary, "num_features", None),
                },
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


def write_tree_tuning_artifacts(
    output_dir: str | Path,
    summary: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    library_name = getattr(summary, "library", None)
    library_version = getattr(getattr(summary, "library_runtime", None), "version", None)
    library_versions = {}
    if library_name is not None and library_version is not None:
        library_versions[str(library_name)] = library_version
    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "tree_tuning.json", summary)
        write_rows_csv(
            directory / "tuning_results.csv",
            list(summary.results.column_names),
            ([row[name] for name in summary.results.column_names] for row in summary.results.to_rows()),
        )
        write_tree_training_artifacts(
            directory / "best_model",
            summary.training,
            metadata={
                "parent_workflow": "tree-tune",
                "library": getattr(summary.training, "library", None),
                "model": getattr(summary.training, "model", None),
            },
        )
        write_artifact_index(
            directory,
            metadata=_artifact_metadata(
                {
                    "workflow": "tree-tune",
                    "source_kind": getattr(summary, "source_kind", None),
                    "library": library_name,
                    "library_version": library_version,
                    "library_versions": library_versions,
                    "model": getattr(summary, "model", None),
                    "task": getattr(summary, "task", None),
                    "target_column": getattr(summary, "target_column", None),
                    "candidate_count": getattr(summary, "candidate_count", None),
                },
                metadata,
            ),
        )
        return directory


def write_mcp_probe_artifacts(
    output_dir: str | Path,
    report: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    if isinstance(report, Mapping):
        probe_results = list(report.get("results", report.get("probes", ())))
        tool_calls = list(report.get("tool_calls", ()))
        started_at = report.get("started_at_utc")
        finished_at = report.get("finished_at_utc")
        total_probes = report.get("total_probes", report.get("summary", {}).get("probe_count"))
        passed_count = report.get("passed", report.get("summary", {}).get("passed_count"))
        failed_count = report.get("failed", report.get("summary", {}).get("failed_count"))
        bugs_found = report.get("bugs_found", report.get("summary", {}).get("bugs_found"))
    else:
        probe_results = list(getattr(report, "results", getattr(report, "probes", ())))
        tool_calls = list(getattr(report, "tool_calls", ()))
        started_at = getattr(report, "started_at_utc", None)
        finished_at = getattr(report, "finished_at_utc", None)
        total_probes = getattr(report, "total_probes", getattr(getattr(report, "summary", {}), "get", lambda *_: None)("probe_count"))
        passed_count = getattr(report, "passed", getattr(getattr(report, "summary", {}), "get", lambda *_: None)("passed_count"))
        failed_count = getattr(report, "failed", getattr(getattr(report, "summary", {}), "get", lambda *_: None)("failed_count"))
        bugs_found = getattr(report, "bugs_found", getattr(getattr(report, "summary", {}), "get", lambda *_: None)("bugs_found"))
    summary_lines = [
        "# MCP Probe Report",
        "",
        f"- Started: {started_at}",
        f"- Finished: {finished_at}",
        f"- Total probes: {total_probes}",
        f"- Passed: {passed_count}",
        f"- Failed: {failed_count}",
        f"- Bugs found: {bugs_found}",
        "",
        "## Findings",
        "",
    ]
    if not probe_results:
        summary_lines.append("No probe results were recorded.")
    else:
        for item in probe_results:
            if isinstance(item, Mapping):
                passed = bool(item.get("passed", False))
                bug_found = bool(item.get("bug_found", not passed))
                probe_id = item.get("probe_id", "")
                summary = item.get("summary", item.get("actual", ""))
            else:
                passed = bool(getattr(item, "passed", False))
                bug_found = bool(getattr(item, "bug_found", not passed))
                probe_id = getattr(item, "probe_id", "")
                summary = getattr(item, "summary", getattr(item, "actual", ""))
            status = "PASS" if passed else "BUG" if bug_found else "FAIL"
            summary_lines.append(
                f"- `{probe_id}` {status}: {summary}"
            )

    with _artifact_output_session(output_dir) as directory:
        write_json(directory / "mcp_probe_report.json", report)
        with atomic_output_path(directory / "mcp_probe_results.jsonl") as temporary_path:
            with temporary_path.open("w", encoding="utf-8") as handle:
                for item in probe_results:
                    handle.write(json.dumps(to_serializable(item), sort_keys=True) + "\n")
        with atomic_output_path(directory / "mcp_tool_calls.jsonl") as temporary_path:
            with temporary_path.open("w", encoding="utf-8") as handle:
                for item in tool_calls:
                    handle.write(json.dumps(to_serializable(item), sort_keys=True) + "\n")
        with atomic_output_path(directory / "mcp_probe_summary.md") as temporary_path:
            temporary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        write_artifact_index(
            directory,
            metadata=_artifact_metadata(
                {
                    "workflow": "mcp-probe",
                    "total_probes": total_probes,
                    "passed": passed_count,
                    "failed": failed_count,
                    "bugs_found": bugs_found,
                },
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
    "write_feature_table_artifacts",
    "write_profile_table_report_artifacts",
    "write_compression_sweep_artifacts",
    "write_artifact_index",
    "write_forward_artifacts",
    "write_inverse_artifacts",
    "write_modal_evaluation_artifacts",
    "write_modal_training_artifacts",
    "write_mcp_probe_artifacts",
    "write_json",
    "write_packet_sweep_artifacts",
    "write_profile_comparison_artifacts",
    "write_rows_csv",
    "write_spectral_analysis_artifacts",
    "write_tree_training_artifacts",
    "write_tree_tuning_artifacts",
    "write_tensorflow_evaluation_artifacts",
    "write_tensorflow_training_artifacts",
    "write_transport_benchmark_artifacts",
    "write_differentiable_artifacts",
    "write_potential_inference_artifacts",
    "write_reduced_model_artifacts",
    "write_vertical_workflow_artifacts",
]
