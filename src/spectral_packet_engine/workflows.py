from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
import importlib.util
from pathlib import Path
import platform
import sys
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, TypeVar

import numpy as np
import torch

if TYPE_CHECKING:
    from spectral_packet_engine.artifacts import ArtifactDirectoryReport

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.database import (
    DatabaseCapabilityReport,
    DatabaseConfig,
    DatabaseConnection,
    QueryResult,
    TableSchemaSummary,
    inspect_database_capabilities,
    sqlalchemy_is_available,
)
from spectral_packet_engine.datasets import (
    DensityPreprocessingConfig,
    available_quantum_gas_transport_scans,
    download_and_prepare_quantum_gas_transport_scan,
)
from spectral_packet_engine.diagnostics import (
    ProfileComparisonSummary,
    ReconstructionErrorSummary,
    SpectralBatchSummary,
    SpectralTruncationSummary,
    summarize_profile_comparison,
    summarize_profile_reconstruction,
    summarize_spectral_batch,
    summarize_spectral_coefficients,
)
from spectral_packet_engine.domain import InfiniteWell1D, coerce_tensor
from spectral_packet_engine.dynamics import SpectralPropagator
from spectral_packet_engine.inference import (
    EstimationConfig,
    GaussianPacketEstimator,
)
from spectral_packet_engine.ml import (
    MLBackendReport,
    ModalRegressionResult,
    ModalSurrogateConfig,
    create_modal_regressor,
    inspect_ml_backends,
)
from spectral_packet_engine.mcp_runtime import MCPRuntimeReport, inspect_mcp_runtime
from spectral_packet_engine.observables import total_probability
from spectral_packet_engine.product import (
    DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES,
    DEFAULT_PROFILE_REPORT_CAPTURE_THRESHOLDS,
    DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES,
)
from spectral_packet_engine.profiles import (
    compress_profiles,
    normalize_profiles,
    profile_mass,
    profile_mean,
    profile_variance,
    project_profiles_onto_basis,
    summarize_profile_compression,
)
from spectral_packet_engine.projector import ProjectionConfig, StateProjector
from spectral_packet_engine.runtime import TorchRuntime, inspect_torch_runtime
from spectral_packet_engine.service_runtime import APIStackRuntime, inspect_api_stack
from spectral_packet_engine.simulation import simulate
from spectral_packet_engine.state import (
    GaussianPacketParameters,
    PacketState,
    make_truncated_gaussian_packet,
)
from spectral_packet_engine.table_io import (
    ProfileTable,
    ProfileTableLayout,
    ProfileTableMaterializationConfig,
    load_profile_table,
    profile_table_from_tabular_dataset,
    resolve_profile_table_layout_from_tabular_dataset,
    tabular_dataset_from_profile_table,
)
from spectral_packet_engine.table_io import supported_profile_table_formats
from spectral_packet_engine.tabular import (
    TabularDataset,
    TabularSchema,
    TabularSource,
    TabularValidationReport,
    load_tabular_dataset,
    parquet_support_is_available,
    supported_tabular_formats,
)
from spectral_packet_engine.tree_models import (
    TreeBackendReport,
    TreeLibrary,
    TreeTask,
    TreeTrainingSummary,
    TreeTuningSummary,
    inspect_tree_backends,
    train_tree_model_on_dataset,
    tune_tree_model_on_dataset,
)
from spectral_packet_engine.tf_surrogate import (
    TensorFlowHostPlatform,
    TensorFlowModalRegressor,
    TensorFlowModalRegressionResult,
    TensorFlowRegressorConfig,
    inspect_tensorflow_host,
    tensorflow_is_available,
)

Tensor = torch.Tensor
_WorkflowResult = TypeVar("_WorkflowResult")


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _safe_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    system: str
    machine: str
    python_version: str
    torch_runtime: TorchRuntime
    ml_backends: MLBackendReport
    tree_backends: TreeBackendReport
    api_stack: APIStackRuntime
    mcp_runtime: MCPRuntimeReport
    tensorflow_host: TensorFlowHostPlatform
    tensorflow_available: bool
    optional_features: dict[str, bool]
    available_transport_scans: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InstallationValidation:
    environment: EnvironmentReport
    core_ready: bool
    stable_surfaces: tuple[str, ...]
    beta_surfaces: tuple[str, ...]
    experimental_surfaces: tuple[str, ...]
    supported_profile_table_formats: dict[str, bool]
    supported_tabular_formats: dict[str, bool]
    notes: tuple[str, ...]


def inspect_environment(preferred_device: str | torch.device | None = "auto") -> EnvironmentReport:
    ml_report = inspect_ml_backends(preferred_device)
    tree_report = inspect_tree_backends()
    api_stack = inspect_api_stack()
    mcp_runtime = inspect_mcp_runtime()
    profile_formats = supported_profile_table_formats()
    tabular_formats = supported_tabular_formats()
    optional_features = {
        "transport-data": _module_available("pooch") and _module_available("scipy"),
        "tensorflow": ml_report.backends["tensorflow"].available,
        "pytorch": ml_report.backends["torch"].available,
        "jax": ml_report.backends["jax"].available,
        "sqlite": True,
        "sqlalchemy": sqlalchemy_is_available(),
        "parquet": tabular_formats["parquet"],
        "xlsx": profile_formats["xlsx"],
        "mcp": _module_available("mcp.server.fastmcp"),
        "api": api_stack.compatible,
        "tree-sklearn": tree_report.libraries["sklearn"].available,
        "tree-xgboost": tree_report.libraries["xgboost"].available,
        "tree-lightgbm": tree_report.libraries["lightgbm"].available,
        "tree-catboost": tree_report.libraries["catboost"].available,
    }
    return EnvironmentReport(
        system=platform.system(),
        machine=platform.machine().lower(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        torch_runtime=inspect_torch_runtime(preferred_device),
        ml_backends=ml_report,
        tree_backends=tree_report,
        api_stack=api_stack,
        mcp_runtime=mcp_runtime,
        tensorflow_host=inspect_tensorflow_host(),
        tensorflow_available=ml_report.backends["tensorflow"].available,
        optional_features=optional_features,
        available_transport_scans=available_quantum_gas_transport_scans(),
    )


def validate_installation(preferred_device: str | torch.device | None = "auto") -> InstallationValidation:
    environment = inspect_environment(preferred_device)
    profile_formats = supported_profile_table_formats()
    tabular_formats = supported_tabular_formats()
    notes: list[str] = []
    if environment.ml_backends.preferred_backend is not None:
        notes.append(
            f"Backend-aware modal surrogate workflows are available; requested device is {environment.ml_backends.requested_device} and the routed backend is {environment.ml_backends.preferred_backend}."
        )
    else:
        notes.append("No ML surrogate backend is available; core spectral workflows remain supported.")
    notes.append(environment.ml_backends.resolution.reason)
    notes.extend(environment.ml_backends.resolution.warnings)
    if environment.tensorflow_available:
        notes.append("TensorFlow-backed workflows are available in this environment.")
    else:
        notes.append("TensorFlow-backed workflows are unavailable in this environment.")
    if environment.optional_features["jax"]:
        notes.append("JAX-backed surrogate workflows are available.")
    else:
        notes.append("JAX-backed surrogate workflows are unavailable in this environment.")
    if environment.api_stack.compatible:
        notes.append("The FastAPI stack is available and compatible.")
    elif environment.api_stack.installed and environment.api_stack.error is not None:
        notes.append(f"The FastAPI stack is installed but incompatible: {environment.api_stack.error}")
    else:
        notes.append("The HTTP API surface is not installed in this environment.")
    if environment.optional_features["mcp"]:
        notes.append("The MCP runtime is available.")
    else:
        notes.append("The MCP runtime is not installed.")
    notes.extend(environment.mcp_runtime.notes[:2])
    if environment.optional_features["sqlalchemy"]:
        notes.append("Remote SQL backends are available through SQLAlchemy.")
    else:
        notes.append("SQLite workflows are available; remote SQL backends require the 'sql' extra.")
    if environment.optional_features["xlsx"]:
        notes.append("XLSX-backed file ingestion is available.")
    else:
        notes.append("XLSX-backed file ingestion requires the 'files' extra.")
    if environment.optional_features["parquet"]:
        notes.append("Parquet-backed tabular ingestion is available.")
    else:
        notes.append("Parquet-backed tabular ingestion is unavailable in this environment.")
    if environment.tree_backends.preferred_library is not None:
        notes.append(
            "Tree-model workflows are available; the preferred runtime backend is "
            f"{environment.tree_backends.preferred_library}."
        )
    else:
        notes.append("Tree-model workflows are unavailable in this environment.")

    return InstallationValidation(
        environment=environment,
        core_ready=True,
        stable_surfaces=(
            "python",
            "cli",
            "core-library",
            "csv-tsv-json-file-io",
            "sqlite-workflows",
            "tabular-datasets",
            "torch-modal-surrogate",
        ),
        beta_surfaces=tuple(
            surface
            for surface, available in (
                ("remote-sql", environment.optional_features["sqlalchemy"]),
                ("jax-modal-surrogate", environment.optional_features["jax"]),
                ("tree-model-workflows", environment.tree_backends.preferred_library is not None),
                ("mcp", environment.optional_features["mcp"]),
                ("api", environment.optional_features["api"]),
            )
            if available
        ),
        experimental_surfaces=tuple(
            surface
            for surface, available in (
                ("tensorflow", environment.tensorflow_available),
                ("transport-dataset", environment.optional_features["transport-data"]),
            )
            if available
        ),
        supported_profile_table_formats=profile_formats,
        supported_tabular_formats=tabular_formats,
        notes=tuple(notes),
    )


@dataclass(frozen=True, slots=True)
class EngineContext:
    runtime: TorchRuntime
    domain: InfiniteWell1D
    basis: InfiniteWellBasis
    projector: StateProjector
    propagator: SpectralPropagator
    quadrature_points: int


def build_engine(
    *,
    num_modes: int = 128,
    quadrature_points: int = 4096,
    domain_length: float = 1.0,
    left: float = 0.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device | None = "auto",
) -> EngineContext:
    runtime = inspect_torch_runtime(device)
    domain = InfiniteWell1D.from_length(
        domain_length,
        left=left,
        mass=mass,
        hbar=hbar,
        dtype=runtime.preferred_real_dtype,
        device=runtime.device,
    )
    basis = InfiniteWellBasis(domain, num_modes=num_modes)
    projector = StateProjector(basis, ProjectionConfig(quadrature_points=quadrature_points))
    propagator = SpectralPropagator(basis)
    return EngineContext(
        runtime=runtime,
        domain=domain,
        basis=basis,
        projector=projector,
        propagator=propagator,
        quadrature_points=quadrature_points,
    )


def _coerce_packet_parameters(
    domain: InfiniteWell1D,
    parameters: GaussianPacketParameters | Mapping[str, Any],
) -> GaussianPacketParameters:
    if isinstance(parameters, GaussianPacketParameters):
        return parameters.to(dtype=domain.real_dtype, device=domain.device)
    if isinstance(parameters, Mapping):
        return GaussianPacketParameters.single(
            center=parameters["center"],
            width=parameters["width"],
            wavenumber=parameters["wavenumber"],
            phase=parameters.get("phase", 0.0),
            dtype=domain.real_dtype,
            device=domain.device,
        )
    raise TypeError("parameters must be GaussianPacketParameters or a mapping")


def make_packet_state(
    domain: InfiniteWell1D,
    *,
    center: float,
    width: float,
    wavenumber: float,
    phase: float = 0.0,
    weight: complex = 1.0 + 0.0j,
) -> PacketState:
    return make_truncated_gaussian_packet(
        domain,
        center=center,
        width=width,
        wavenumber=wavenumber,
        phase=phase,
        weight=weight,
    )


@dataclass(frozen=True, slots=True)
class ForwardSimulationSummary:
    runtime: TorchRuntime
    times: Tensor
    grid: Tensor
    densities: Tensor
    expectation_position: Tensor
    left_probability: Tensor
    right_probability: Tensor
    total_probability: Tensor
    spectral_norm: Tensor
    spatial_norm: Tensor
    truncation: SpectralTruncationSummary


def _simulate_packet_with_context(
    context: EngineContext,
    *,
    center: float,
    width: float,
    wavenumber: float,
    phase: float = 0.0,
    weight: complex = 1.0 + 0.0j,
    evaluation_times: Tensor,
    grid: Tensor,
) -> ForwardSimulationSummary:
    packet = make_packet_state(
        context.domain,
        center=center,
        width=width,
        wavenumber=wavenumber,
        phase=phase,
        weight=weight,
    )
    initial_state = context.projector.project_packet(packet)
    record = simulate(
        packet,
        evaluation_times,
        projector=context.projector,
        propagator=context.propagator,
        grid=grid,
    )
    midpoint = context.domain.midpoint
    initial_wavefunction = packet.wavefunction(grid)
    return ForwardSimulationSummary(
        runtime=context.runtime,
        times=record.times,
        grid=grid,
        densities=record.densities if record.densities is not None else torch.empty(0, device=grid.device),
        expectation_position=record.expectation_position(),
        left_probability=record.interval_probability(context.domain.left, midpoint),
        right_probability=record.interval_probability(midpoint, context.domain.right),
        total_probability=record.total_probability(),
        spectral_norm=initial_state.norm_squared,
        spatial_norm=total_probability(initial_wavefunction, grid),
        truncation=summarize_spectral_coefficients(initial_state.coefficients),
    )


def simulate_gaussian_packet(
    *,
    center: float = 0.30,
    width: float = 0.07,
    wavenumber: float = 25.0,
    phase: float = 0.0,
    weight: complex = 1.0 + 0.0j,
    times: Sequence[float] = (0.0, 1e-3, 5e-3),
    num_modes: int = 128,
    quadrature_points: int = 4096,
    grid_points: int = 512,
    device: str | torch.device | None = "auto",
) -> ForwardSimulationSummary:
    context = build_engine(
        num_modes=num_modes,
        quadrature_points=quadrature_points,
        device=device,
    )
    grid = context.domain.grid(grid_points)
    evaluation_times = torch.as_tensor(times, dtype=context.domain.real_dtype, device=context.domain.device)
    return _simulate_packet_with_context(
        context,
        center=center,
        width=width,
        wavenumber=wavenumber,
        phase=phase,
        weight=weight,
        evaluation_times=evaluation_times,
        grid=grid,
    )


@dataclass(frozen=True, slots=True)
class PacketProjectionSummary:
    runtime: TorchRuntime
    coefficients: Tensor
    reconstruction_error: Tensor
    spectral_norm: Tensor
    truncation: SpectralTruncationSummary


def project_gaussian_packet(
    *,
    center: float = 0.30,
    width: float = 0.07,
    wavenumber: float = 25.0,
    phase: float = 0.0,
    weight: complex = 1.0 + 0.0j,
    num_modes: int = 128,
    quadrature_points: int = 4096,
    grid_points: int = 2048,
    device: str | torch.device | None = "auto",
) -> PacketProjectionSummary:
    context = build_engine(
        num_modes=num_modes,
        quadrature_points=quadrature_points,
        device=device,
    )
    packet = make_packet_state(
        context.domain,
        center=center,
        width=width,
        wavenumber=wavenumber,
        phase=phase,
        weight=weight,
    )
    spectral_state = context.projector.project_packet(packet)
    grid = context.domain.grid(grid_points)
    reference = packet.wavefunction(grid)
    reconstruction = context.projector.reconstruct(spectral_state, grid)
    reconstruction_error = torch.sqrt(total_probability(reconstruction - reference, grid))
    return PacketProjectionSummary(
        runtime=context.runtime,
        coefficients=spectral_state.coefficients,
        reconstruction_error=reconstruction_error,
        spectral_norm=spectral_state.norm_squared,
        truncation=summarize_spectral_coefficients(spectral_state.coefficients),
    )


@dataclass(frozen=True, slots=True)
class InverseFitSummary:
    runtime: TorchRuntime
    observation_grid: Tensor
    times: Tensor
    estimated_parameters: GaussianPacketParameters
    final_loss: float
    history: tuple[float, ...]
    predicted_density: Tensor


@dataclass(frozen=True, slots=True)
class ProfileTableSummary:
    source: str | None
    num_samples: int
    num_positions: int
    sample_time_min: float
    sample_time_max: float
    position_min: float
    position_max: float
    mean_position_min: float
    mean_position_max: float
    width_min: float
    width_max: float
    profile_mass_min: float
    profile_mass_max: float
    supported_formats: dict[str, bool]


@dataclass(frozen=True, slots=True)
class TabularDatasetSummary:
    source_kind: str | None
    source_location: str | None
    row_count: int
    column_count: int
    schema: TabularSchema
    validation: TabularValidationReport
    supported_formats: dict[str, bool]
    preview_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class DatabaseInspectionSummary:
    capability: DatabaseCapabilityReport
    tables: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatabaseQuerySummary:
    redacted_url: str
    query: str
    parameters: dict[str, Any]
    table: TabularDatasetSummary


@dataclass(frozen=True, slots=True)
class DatabaseExecutionSummary:
    redacted_url: str
    statement: str
    parameters: dict[str, Any]
    row_count: int | None
    statement_count: int
    mode: str


@dataclass(frozen=True, slots=True)
class DatabaseWriteSummary:
    redacted_url: str
    table_name: str
    row_count: int
    schema: TableSchemaSummary
    if_exists: str


@dataclass(frozen=True, slots=True)
class DatabaseMaterializationSummary:
    redacted_url: str
    table_name: str
    source_query: str
    schema: TableSchemaSummary
    replace: bool


@dataclass(frozen=True, slots=True)
class DatabaseProfileTableMaterialization:
    redacted_url: str
    query: str
    parameters: dict[str, Any]
    materialization: ProfileTableMaterializationConfig
    layout: ProfileTableLayout
    ordering: dict[str, Any]
    table: ProfileTable


def fit_gaussian_packet_to_density(
    *,
    target_density,
    observation_grid,
    times,
    initial_guess: GaussianPacketParameters | Mapping[str, Any],
    num_modes: int = 128,
    quadrature_points: int = 2048,
    device: str | torch.device | None = "auto",
    estimation_config: EstimationConfig | None = None,
    steps: int | None = None,
    learning_rate: float | None = None,
) -> InverseFitSummary:
    context = build_engine(
        num_modes=num_modes,
        quadrature_points=quadrature_points,
        device=device,
    )
    grid = coerce_tensor(observation_grid, dtype=context.domain.real_dtype, device=context.domain.device)
    evaluation_times = coerce_tensor(times, dtype=context.domain.real_dtype, device=context.domain.device)
    estimator = GaussianPacketEstimator(
        context.domain,
        basis=context.basis,
        projection_config=ProjectionConfig(quadrature_points=quadrature_points),
        estimation_config=estimation_config or EstimationConfig(),
    )
    guess = _coerce_packet_parameters(context.domain, initial_guess)
    result = estimator.fit(
        observation_grid=grid,
        times=evaluation_times,
        target=target_density,
        initial_guess=guess,
        observation_mode="density",
        steps=steps,
        learning_rate=learning_rate,
    )
    prediction = estimator.predict(
        result.parameters,
        observation_grid=grid,
        times=evaluation_times,
        observation_mode="density",
    )
    return InverseFitSummary(
        runtime=context.runtime,
        observation_grid=grid,
        times=evaluation_times,
        estimated_parameters=result.parameters,
        final_loss=result.final_loss,
        history=result.history,
        predicted_density=prediction,
    )


def fit_gaussian_packet_to_profile_table(
    table: ProfileTable,
    *,
    initial_guess: GaussianPacketParameters | Mapping[str, Any],
    num_modes: int = 128,
    quadrature_points: int = 2048,
    device: str | torch.device | None = "auto",
    estimation_config: EstimationConfig | None = None,
    steps: int | None = None,
    learning_rate: float | None = None,
) -> InverseFitSummary:
    runtime = inspect_torch_runtime(device)
    grid, times, profiles = table.to_torch(dtype=runtime.preferred_real_dtype, device=runtime.device)
    return fit_gaussian_packet_to_density(
        target_density=profiles,
        observation_grid=grid,
        times=times,
        initial_guess=initial_guess,
        num_modes=num_modes,
        quadrature_points=quadrature_points,
        device=runtime.device,
        estimation_config=estimation_config,
        steps=steps,
        learning_rate=learning_rate,
    )


def fit_gaussian_packet_to_profile_table_from_database_query(
    database: DatabaseConfig | str | Any,
    query: str,
    *,
    initial_guess: GaussianPacketParameters | Mapping[str, Any],
    parameters: Mapping[str, Any] | None = None,
    materialization: ProfileTableMaterializationConfig | None = None,
    time_column: str = "time",
    position_columns: Sequence[str] | None = None,
    sort_by_time: bool = False,
    num_modes: int = 128,
    quadrature_points: int = 2048,
    device: str | torch.device | None = "auto",
    estimation_config: EstimationConfig | None = None,
    steps: int | None = None,
    learning_rate: float | None = None,
    create_if_missing: bool = False,
) -> InverseFitSummary:
    return _run_profile_table_workflow_from_database_query(
        database,
        query,
        workflow=lambda materialized: fit_gaussian_packet_to_profile_table(
            materialized.table,
            initial_guess=initial_guess,
            num_modes=num_modes,
            quadrature_points=quadrature_points,
            device=device,
            estimation_config=estimation_config,
            steps=steps,
            learning_rate=learning_rate,
        ),
        parameters=parameters,
        materialization=materialization,
        time_column=time_column,
        position_columns=position_columns,
        sort_by_time=sort_by_time,
        create_if_missing=create_if_missing,
    )


@dataclass(frozen=True, slots=True)
class ProfileCompressionWorkflowResult:
    runtime: TorchRuntime
    num_modes: int
    grid: Tensor
    sample_times: Tensor
    source: str | None
    coefficients: Tensor
    reconstruction: Tensor
    mean_position: Tensor
    width: Tensor
    mass: Tensor
    error_summary: ReconstructionErrorSummary
    spectral_summary: SpectralBatchSummary


@dataclass(frozen=True, slots=True)
class FeatureTableExportSummary:
    source_kind: str | None
    source_location: str | None
    identifier_columns: tuple[str, ...]
    feature_names: tuple[str, ...]
    includes: tuple[str, ...]
    num_rows: int
    num_features: int
    num_modes: int
    normalize_each_profile: bool
    format: str
    output_path: str | None
    table: TabularDataset
    ordering: dict[str, Any]
    library_versions: dict[str, str]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CaptureModeBudget:
    threshold: float
    mean_mode_count: float
    max_mode_count: float


@dataclass(frozen=True, slots=True)
class ProfileTableReportOverview:
    source: str | None
    num_samples: int
    num_positions: int
    analyze_num_modes: int
    compress_num_modes: int
    normalize_each_profile: bool
    dominant_modes: tuple[int, ...]
    capture_mode_budgets: tuple[CaptureModeBudget, ...]
    mean_relative_l2_error: float
    max_relative_l2_error: float
    root_mean_square_error: float
    mean_position_min: float
    mean_position_max: float
    width_min: float
    width_max: float
    profile_mass_min: float
    profile_mass_max: float


@dataclass(frozen=True, slots=True)
class ProfileTableReport:
    overview: ProfileTableReportOverview
    inspection: ProfileTableSummary
    analysis: ProfileTableSpectralSummary
    compression: ProfileCompressionWorkflowResult

    def write_artifacts(
        self,
        output_dir: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactDirectoryReport:
        from spectral_packet_engine.artifacts import (
            inspect_artifact_directory,
            write_profile_table_report_artifacts,
        )

        write_profile_table_report_artifacts(output_dir, self, metadata=metadata)
        return inspect_artifact_directory(output_dir)


@dataclass(frozen=True, slots=True)
class ProfileTableSpectralSummary:
    runtime: TorchRuntime
    source: str | None
    num_modes: int
    grid: Tensor
    sample_times: Tensor
    coefficients: Tensor
    mean_position: Tensor
    width: Tensor
    mass: Tensor
    spectral_summary: SpectralBatchSummary


@dataclass(frozen=True, slots=True)
class ProfileCompressionSweepSummary:
    runtime: TorchRuntime
    source: str | None
    mode_counts: Tensor
    mean_relative_l2_error: Tensor
    max_relative_l2_error: Tensor


@dataclass(frozen=True, slots=True)
class PacketSweepItemSummary:
    center: float
    width: float
    wavenumber: float
    phase: float
    spectral_norm: float
    final_expectation_position: float
    final_left_probability: float
    final_right_probability: float
    final_total_probability: float


@dataclass(frozen=True, slots=True)
class PacketSweepSummary:
    runtime: TorchRuntime
    times: Tensor
    items: tuple[PacketSweepItemSummary, ...]


@dataclass(frozen=True, slots=True)
class ProfileTableComparisonWorkflowResult:
    runtime: TorchRuntime
    reference_source: str | None
    candidate_source: str | None
    grid: Tensor
    sample_times: Tensor
    reference_profiles: Tensor
    candidate_profiles: Tensor
    residual_profiles: Tensor
    comparison: ProfileComparisonSummary


def _coerce_database_config(
    database: DatabaseConfig | str | Any,
    *,
    create_if_missing: bool = True,
    read_only: bool = False,
) -> DatabaseConfig:
    if isinstance(database, DatabaseConfig):
        return database
    return DatabaseConfig.from_reference(
        database,
        create_if_missing=create_if_missing,
        read_only=read_only,
    )


def _coerce_profile_table_materialization_config(
    *,
    materialization: ProfileTableMaterializationConfig | None = None,
    time_column: str = "time",
    position_columns: Sequence[str] | None = None,
    sort_by_time: bool = False,
    source: str | None = None,
) -> ProfileTableMaterializationConfig:
    if materialization is not None:
        if position_columns is not None or sort_by_time or source is not None or time_column != "time":
            raise ValueError("pass either materialization or explicit profile-table arguments, not both")
        return materialization
    return ProfileTableMaterializationConfig(
        time_column=time_column,
        position_columns=None if position_columns is None else tuple(position_columns),
        sort_by_time=sort_by_time,
        source=source,
    )


def database_query_artifact_metadata(
    database: DatabaseConfig | str | Any,
    query: str,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = _coerce_database_config(database, create_if_missing=False)
    return {
        "input": {
            "kind": "database-query",
            "database": config.redacted_url,
            "query": query,
            "parameters": {} if parameters is None else dict(parameters),
        }
    }


def database_profile_query_artifact_metadata(
    database: DatabaseConfig | str | Any,
    query: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    materialization: ProfileTableMaterializationConfig | None = None,
    time_column: str = "time",
    position_columns: Sequence[str] | None = None,
    sort_by_time: bool = False,
) -> dict[str, Any]:
    metadata = database_query_artifact_metadata(
        database,
        query,
        parameters=parameters,
    )
    resolved_materialization = _coerce_profile_table_materialization_config(
        materialization=materialization,
        time_column=time_column,
        position_columns=position_columns,
        sort_by_time=sort_by_time,
    )
    metadata["input"]["profile_table"] = {
        "time_column": resolved_materialization.time_column,
        "position_columns": resolved_materialization.position_columns,
        "sort_by_time": resolved_materialization.sort_by_time,
    }
    return metadata


def database_query_workflow_artifact_metadata(
    workflow: str,
    database: DatabaseConfig | str | Any,
    query: str,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "workflow": workflow,
        **database_query_artifact_metadata(
            database,
            query,
            parameters=parameters,
        ),
    }


def database_profile_query_workflow_artifact_metadata(
    workflow: str,
    database: DatabaseConfig | str | Any,
    query: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    materialization: ProfileTableMaterializationConfig | None = None,
    time_column: str = "time",
    position_columns: Sequence[str] | None = None,
    sort_by_time: bool = False,
) -> dict[str, Any]:
    return {
        "workflow": workflow,
        **database_profile_query_artifact_metadata(
            database,
            query,
            parameters=parameters,
            materialization=materialization,
            time_column=time_column,
            position_columns=position_columns,
            sort_by_time=sort_by_time,
        ),
    }


def summarize_tabular_dataset(dataset: TabularDataset) -> TabularDatasetSummary:
    validation = dataset.validation_report()
    source_kind = None if dataset.source is None else dataset.source.kind
    source_location = None if dataset.source is None else dataset.source.location
    return TabularDatasetSummary(
        source_kind=source_kind,
        source_location=source_location,
        row_count=dataset.row_count,
        column_count=len(dataset.column_names),
        schema=dataset.schema,
        validation=validation,
        supported_formats=supported_tabular_formats(),
        preview_rows=tuple(dataset.to_rows(limit=10)),
    )


def _coerce_profile_table_input(table: ProfileTable | str | Path) -> ProfileTable:
    if isinstance(table, ProfileTable):
        return table
    return load_profile_table(table)


def _coerce_tabular_dataset_input(features: TabularDataset | str | Path) -> TabularDataset:
    if isinstance(features, TabularDataset):
        return features
    return load_tabular_dataset(features)


def inspect_tree_backend_support(*, requested_library: TreeLibrary = "auto") -> TreeBackendReport:
    return inspect_tree_backends(requested_library=requested_library)


def _feature_export_library_versions(*, output_format: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in ("numpy", "torch"):
        package_version = _safe_package_version(package_name)
        if package_version is not None:
            versions[package_name] = package_version
    if output_format == "parquet":
        pyarrow_version = _safe_package_version("pyarrow")
        if pyarrow_version is not None:
            versions["pyarrow"] = pyarrow_version
    return versions


def _profile_table_feature_export_ordering(table: ProfileTable) -> dict[str, Any]:
    return {
        "time": {
            "policy": "preserve-profile-table-sample-order",
            "requested_sort": False,
            "was_reordered": False,
            "input_sample_count": int(table.num_samples),
        },
        "positions": {
            "policy": "preserve-profile-table-grid-order",
            "was_reordered": False,
            "input_position_count": int(table.num_positions),
        },
    }


def _database_feature_export_ordering(
    *,
    source_sample_times: np.ndarray,
    requested_sort_by_time: bool,
    source_position_columns: Sequence[str],
    materialized_position_columns: Sequence[str],
) -> dict[str, Any]:
    order = np.argsort(source_sample_times, kind="stable")
    time_was_reordered = bool(
        requested_sort_by_time
        and np.any(order != np.arange(source_sample_times.shape[0], dtype=order.dtype))
    )
    source_positions = tuple(str(name) for name in source_position_columns)
    materialized_positions = tuple(str(name) for name in materialized_position_columns)
    return {
        "time": {
            "policy": "stable-ascending" if requested_sort_by_time else "preserve-input-order",
            "requested_sort": bool(requested_sort_by_time),
            "was_reordered": time_was_reordered,
            "input_sample_count": int(source_sample_times.shape[0]),
        },
        "positions": {
            "policy": "numeric-position-ascending",
            "was_reordered": source_positions != materialized_positions,
            "source_columns": list(source_positions),
            "materialized_columns": list(materialized_positions),
        },
    }


def _build_feature_export_metadata(
    *,
    input_metadata: Mapping[str, Any],
    includes: Sequence[str],
    num_modes: int,
    normalize_each_profile: bool,
    ordering: Mapping[str, Any],
    output_format: str,
) -> dict[str, Any]:
    input_payload = dict(input_metadata)
    return {
        "workflow": "export-features",
        "input": input_payload,
        "feature_generation": {
            "includes": [str(item) for item in includes],
            "num_modes": int(num_modes),
            "normalize_each_profile": bool(normalize_each_profile),
            "format": output_format,
        },
        "ordering": dict(ordering),
        "library_versions": _feature_export_library_versions(output_format=output_format),
    }


def _feature_table_from_spectral_summary(
    summary: ProfileTableSpectralSummary,
    *,
    normalize_each_profile: bool,
    include_coefficients: bool = True,
    include_moments: bool = True,
    source_kind: str | None = "profile-table",
    source_location: str | None = None,
    ordering: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    output_format: str = "csv",
) -> FeatureTableExportSummary:
    if not include_coefficients and not include_moments:
        raise ValueError("At least one feature group must be selected.")
    if output_format not in {"csv", "parquet"}:
        raise ValueError("format must be 'csv' or 'parquet'")
    if output_format == "parquet" and not parquet_support_is_available():
        raise ModuleNotFoundError("Parquet feature export requires pyarrow. Install the 'files' extra.")

    times = torch.as_tensor(summary.sample_times).detach().cpu().tolist()
    coefficient_rows = torch.as_tensor(summary.coefficients).detach().cpu().tolist()
    mean_position = torch.as_tensor(summary.mean_position).detach().cpu().tolist()
    width = torch.as_tensor(summary.width).detach().cpu().tolist()
    mass = torch.as_tensor(summary.mass).detach().cpu().tolist()

    feature_names: list[str] = []
    includes: list[str] = []
    if include_coefficients:
        feature_names.extend(f"mode_{index}" for index in range(1, summary.num_modes + 1))
        includes.append("coefficients")
    if include_moments:
        feature_names.extend(("mean_position", "width", "mass"))
        includes.append("moments")

    rows: list[dict[str, Any]] = []
    for row_index, sample_time in enumerate(times):
        row: dict[str, Any] = {"time": sample_time}
        if include_coefficients:
            for mode_index, coefficient in enumerate(coefficient_rows[row_index], start=1):
                row[f"mode_{mode_index}"] = coefficient
        if include_moments:
            row["mean_position"] = mean_position[row_index]
            row["width"] = width[row_index]
            row["mass"] = mass[row_index]
        rows.append(row)

    dataset_metadata = {
        "workflow": "export-features",
    }
    if metadata is not None:
        dataset_metadata.update(dict(metadata))
    resolved_ordering = {} if ordering is None else dict(ordering)
    library_versions = _feature_export_library_versions(output_format=output_format)
    dataset = TabularDataset.from_rows(
        rows,
        source=TabularSource(
            kind=source_kind or "profile-table",
            location=source_location,
            description="Spectral feature table exported from profile-table workflows.",
        ),
    )
    return FeatureTableExportSummary(
        source_kind=dataset.source.kind if dataset.source is not None else None,
        source_location=dataset.source.location if dataset.source is not None else None,
        identifier_columns=("time",),
        feature_names=tuple(feature_names),
        includes=tuple(includes),
        num_rows=dataset.row_count,
        num_features=len(feature_names),
        num_modes=summary.num_modes,
        normalize_each_profile=normalize_each_profile,
        format=output_format,
        output_path=None,
        table=dataset,
        ordering=resolved_ordering,
        library_versions=library_versions,
        metadata=dataset_metadata,
    )


def load_tabular_dataset_from_path(path: str | Any) -> TabularDatasetSummary:
    return summarize_tabular_dataset(load_tabular_dataset(path))


def inspect_ml_backend_support(preferred_device: str | torch.device | None = "auto") -> MLBackendReport:
    return inspect_ml_backends(preferred_device)


def inspect_database(
    database: DatabaseConfig | str | Any,
    *,
    create_if_missing: bool = False,
) -> DatabaseInspectionSummary:
    config = _coerce_database_config(database, create_if_missing=create_if_missing)
    with DatabaseConnection(config) as connection:
        return DatabaseInspectionSummary(
            capability=inspect_database_capabilities(config),
            tables=connection.list_tables(),
        )


def bootstrap_local_database(path: str | Path | Any) -> DatabaseInspectionSummary:
    config = DatabaseConfig.sqlite(path, create_if_missing=True)
    return inspect_database(config, create_if_missing=True)


def list_database_tables(
    database: DatabaseConfig | str | Any,
    *,
    create_if_missing: bool = False,
) -> tuple[str, ...]:
    config = _coerce_database_config(database, create_if_missing=create_if_missing)
    with DatabaseConnection(config) as connection:
        return connection.list_tables()


def describe_database_table(
    database: DatabaseConfig | str | Any,
    table_name: str,
    *,
    create_if_missing: bool = False,
) -> TableSchemaSummary:
    config = _coerce_database_config(database, create_if_missing=create_if_missing)
    with DatabaseConnection(config) as connection:
        return connection.describe_table(table_name)


def summarize_database_query_result(
    database: DatabaseConfig | str | Any,
    query: str,
    result: QueryResult,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> DatabaseQuerySummary:
    config = _coerce_database_config(database, create_if_missing=False)
    return DatabaseQuerySummary(
        redacted_url=config.redacted_url,
        query=query,
        parameters={} if parameters is None else dict(parameters),
        table=summarize_tabular_dataset(result.dataset),
    )


def execute_database_query(
    database: DatabaseConfig | str | Any,
    query: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    create_if_missing: bool = False,
) -> DatabaseQuerySummary:
    config = _coerce_database_config(database, create_if_missing=create_if_missing)
    with DatabaseConnection(config) as connection:
        result = connection.query(query, parameters=parameters)
        return summarize_database_query_result(
            config,
            query,
            result,
            parameters=parameters,
        )


def _summarize_database_execution(
    config: DatabaseConfig,
    statement: str,
    *,
    parameters: Mapping[str, Any] | None,
    row_count: int | None,
    statement_count: int,
    mode: str,
) -> DatabaseExecutionSummary:
    return DatabaseExecutionSummary(
        redacted_url=config.redacted_url,
        statement=statement,
        parameters={} if parameters is None else dict(parameters),
        row_count=row_count,
        statement_count=statement_count,
        mode=mode,
    )


def execute_database_statement(
    database: DatabaseConfig | str | Any,
    statement: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    create_if_missing: bool = False,
) -> DatabaseExecutionSummary:
    config = _coerce_database_config(database, create_if_missing=create_if_missing)
    with DatabaseConnection(config) as connection:
        result = connection.execute(statement, parameters=parameters)
    return _summarize_database_execution(
        config,
        result.statement,
        parameters=result.parameters,
        row_count=result.row_count,
        statement_count=1,
        mode="statement",
    )


def execute_database_script(
    database: DatabaseConfig | str | Any,
    script: str,
    *,
    create_if_missing: bool = False,
) -> DatabaseExecutionSummary:
    config = _coerce_database_config(database, create_if_missing=create_if_missing)
    with DatabaseConnection(config) as connection:
        result = connection.execute_script(script)
    return _summarize_database_execution(
        config,
        result.script,
        parameters=None,
        row_count=None,
        statement_count=result.statement_count,
        mode="script",
    )


def materialize_database_query(
    database: DatabaseConfig | str | Any,
    query: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    create_if_missing: bool = False,
) -> QueryResult:
    config = _coerce_database_config(database, create_if_missing=create_if_missing)
    with DatabaseConnection(config) as connection:
        return connection.query(query, parameters=parameters)


def materialize_profile_table_from_database_query(
    database: DatabaseConfig | str | Any,
    query: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    materialization: ProfileTableMaterializationConfig | None = None,
    time_column: str = "time",
    position_columns: Sequence[str] | None = None,
    sort_by_time: bool = False,
    create_if_missing: bool = False,
) -> DatabaseProfileTableMaterialization:
    config = _coerce_database_config(database, create_if_missing=create_if_missing)
    result = materialize_database_query(
        config,
        query,
        parameters=parameters,
        create_if_missing=create_if_missing,
    )
    resolved_materialization = _coerce_profile_table_materialization_config(
        materialization=materialization,
        time_column=time_column,
        position_columns=position_columns,
        sort_by_time=sort_by_time,
    )
    if resolved_materialization.source is None:
        resolved_materialization = replace(
            resolved_materialization,
            source=config.redacted_url,
        )
    layout_resolution = resolve_profile_table_layout_from_tabular_dataset(
        result.dataset,
        config=resolved_materialization,
    )
    layout = layout_resolution.layout
    table = profile_table_from_tabular_dataset(
        result.dataset,
        config=resolved_materialization,
    )
    return DatabaseProfileTableMaterialization(
        redacted_url=config.redacted_url,
        query=query,
        parameters={} if parameters is None else dict(parameters),
        materialization=resolved_materialization,
        layout=layout,
        ordering=_database_feature_export_ordering(
            source_sample_times=np.asarray(result.dataset.columns[layout_resolution.time_column], dtype=np.float64),
            requested_sort_by_time=resolved_materialization.sort_by_time,
            source_position_columns=layout_resolution.source_position_columns,
            materialized_position_columns=layout.position_columns,
        ),
        table=table,
    )


def _run_profile_table_workflow_from_database_query(
    database: DatabaseConfig | str | Any,
    query: str,
    *,
    workflow: Callable[[DatabaseProfileTableMaterialization], _WorkflowResult],
    parameters: Mapping[str, Any] | None = None,
    materialization: ProfileTableMaterializationConfig | None = None,
    time_column: str = "time",
    position_columns: Sequence[str] | None = None,
    sort_by_time: bool = False,
    create_if_missing: bool = False,
) -> _WorkflowResult:
    materialized = materialize_profile_table_from_database_query(
        database,
        query,
        parameters=parameters,
        materialization=materialization,
        time_column=time_column,
        position_columns=position_columns,
        sort_by_time=sort_by_time,
        create_if_missing=create_if_missing,
    )
    return workflow(materialized)


def write_tabular_dataset_to_database(
    database: DatabaseConfig | str | Any,
    table_name: str,
    dataset: TabularDataset,
    *,
    if_exists: str = "fail",
    create_if_missing: bool = True,
) -> DatabaseWriteSummary:
    config = _coerce_database_config(database, create_if_missing=create_if_missing)
    with DatabaseConnection(config) as connection:
        schema = connection.write_dataset(table_name, dataset, if_exists=if_exists)
        return DatabaseWriteSummary(
            redacted_url=config.redacted_url,
            table_name=table_name,
            row_count=dataset.row_count,
            schema=schema,
            if_exists=if_exists,
        )


def write_profile_table_to_database(
    database: DatabaseConfig | str | Any,
    table_name: str,
    table: ProfileTable,
    *,
    if_exists: str = "fail",
    create_if_missing: bool = True,
) -> DatabaseWriteSummary:
    return write_tabular_dataset_to_database(
        database,
        table_name,
        tabular_dataset_from_profile_table(table),
        if_exists=if_exists,
        create_if_missing=create_if_missing,
    )


def materialize_database_query_to_table(
    database: DatabaseConfig | str | Any,
    table_name: str,
    query: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    replace: bool = False,
    create_if_missing: bool = True,
) -> DatabaseMaterializationSummary:
    config = _coerce_database_config(database, create_if_missing=create_if_missing)
    with DatabaseConnection(config) as connection:
        schema = connection.create_table_from_query(
            table_name,
            query,
            parameters=parameters,
            replace=replace,
        )
    return DatabaseMaterializationSummary(
        redacted_url=config.redacted_url,
        table_name=table_name,
        source_query=query,
        schema=schema,
        replace=replace,
    )


def coerce_database_table_types(
    database: DatabaseConfig | str | Any,
    table_name: str,
    *,
    create_if_missing: bool = False,
) -> DatabaseMaterializationSummary:
    """Detect and fix column type affinities for an existing table."""
    config = _coerce_database_config(database, create_if_missing=create_if_missing)
    with DatabaseConnection(config) as connection:
        schema = connection.coerce_column_types(table_name)
    return DatabaseMaterializationSummary(
        redacted_url=config.redacted_url,
        table_name=table_name,
        source_query="(type coercion)",
        schema=schema,
        replace=False,
    )


def pivot_database_table(
    database: DatabaseConfig | str | Any,
    table_name: str,
    target_table: str,
    index_column: str,
    pivot_column: str,
    value_column: str,
    *,
    aggregate: str = "MAX",
    replace: bool = False,
    create_if_missing: bool = True,
) -> DatabaseMaterializationSummary:
    """Pivot a long-format table into wide format and persist the result."""
    config = _coerce_database_config(database, create_if_missing=create_if_missing)
    with DatabaseConnection(config) as connection:
        pivot_sql = connection.pivot_query(
            table_name, index_column, pivot_column, value_column, aggregate=aggregate,
        )
        schema = connection.create_table_from_query(
            target_table, pivot_sql, replace=replace,
        )
    return DatabaseMaterializationSummary(
        redacted_url=config.redacted_url,
        table_name=target_table,
        source_query=pivot_sql,
        schema=schema,
        replace=replace,
    )


def unpivot_database_table(
    database: DatabaseConfig | str | Any,
    table_name: str,
    target_table: str,
    id_columns: Sequence[str],
    value_columns: Sequence[str] | None = None,
    *,
    replace: bool = False,
    create_if_missing: bool = True,
) -> DatabaseMaterializationSummary:
    """Unpivot (melt) a wide-format table into long format and persist."""
    config = _coerce_database_config(database, create_if_missing=create_if_missing)
    with DatabaseConnection(config) as connection:
        unpivot_sql = connection.unpivot_query(
            table_name, id_columns, value_columns,
        )
        schema = connection.create_table_from_query(
            target_table, unpivot_sql, replace=replace,
        )
    return DatabaseMaterializationSummary(
        redacted_url=config.redacted_url,
        table_name=target_table,
        source_query=unpivot_sql,
        schema=schema,
        replace=replace,
    )


def interpolate_database_time_series(
    database: DatabaseConfig | str | Any,
    table_name: str,
    target_table: str,
    time_column: str,
    value_columns: Sequence[str],
    *,
    step: float = 1.0,
    replace: bool = False,
    create_if_missing: bool = True,
) -> DatabaseMaterializationSummary:
    """Interpolate missing time steps in a time series and persist."""
    config = _coerce_database_config(database, create_if_missing=create_if_missing)
    with DatabaseConnection(config) as connection:
        interp_sql = connection.interpolate_time_series(
            table_name, time_column, value_columns, step=step,
        )
        schema = connection.create_table_from_query(
            target_table, interp_sql, replace=replace,
        )
    return DatabaseMaterializationSummary(
        redacted_url=config.redacted_url,
        table_name=target_table,
        source_query=interp_sql,
        schema=schema,
        replace=replace,
    )


def window_aggregate_database_query(
    database: DatabaseConfig | str | Any,
    table_name: str,
    value_column: str,
    order_by: str,
    *,
    window_size: int = 3,
    functions: Sequence[str] = ("AVG", "SUM", "COUNT"),
    create_if_missing: bool = False,
) -> DatabaseQuerySummary:
    """Run a sliding window aggregate query and return results."""
    config = _coerce_database_config(database, create_if_missing=create_if_missing)
    with DatabaseConnection(config) as connection:
        query = connection.window_aggregate_query(
            table_name, value_column, order_by=order_by,
            window_size=window_size, functions=functions,
        )
        result = connection.query(query)
    return summarize_database_query_result(database, query, result)


def analyze_profile_table_from_database_query(
    database: DatabaseConfig | str | Any,
    query: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    materialization: ProfileTableMaterializationConfig | None = None,
    time_column: str = "time",
    position_columns: Sequence[str] | None = None,
    sort_by_time: bool = False,
    num_modes: int = 32,
    device: str | torch.device | None = "auto",
    normalize_each_profile: bool = False,
    create_if_missing: bool = False,
) -> ProfileTableSpectralSummary:
    return _run_profile_table_workflow_from_database_query(
        database,
        query,
        workflow=lambda materialized: analyze_profile_table_spectra(
            materialized.table,
            num_modes=num_modes,
            device=device,
            normalize_each_profile=normalize_each_profile,
        ),
        parameters=parameters,
        materialization=materialization,
        time_column=time_column,
        position_columns=position_columns,
        sort_by_time=sort_by_time,
        create_if_missing=create_if_missing,
    )


def compress_profile_table_from_database_query(
    database: DatabaseConfig | str | Any,
    query: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    materialization: ProfileTableMaterializationConfig | None = None,
    time_column: str = "time",
    position_columns: Sequence[str] | None = None,
    sort_by_time: bool = False,
    num_modes: int = 32,
    device: str | torch.device | None = "auto",
    normalize_each_profile: bool = False,
    create_if_missing: bool = False,
) -> ProfileCompressionWorkflowResult:
    return _run_profile_table_workflow_from_database_query(
        database,
        query,
        workflow=lambda materialized: compress_profile_table(
            materialized.table,
            num_modes=num_modes,
            device=device,
            normalize_each_profile=normalize_each_profile,
        ),
        parameters=parameters,
        materialization=materialization,
        time_column=time_column,
        position_columns=position_columns,
        sort_by_time=sort_by_time,
        create_if_missing=create_if_missing,
    )


def summarize_profile_table(
    table: ProfileTable,
    *,
    device: str | torch.device | None = "auto",
) -> ProfileTableSummary:
    runtime = inspect_torch_runtime(device)
    with torch.inference_mode():
        grid, _, profiles = table.to_torch(dtype=runtime.preferred_real_dtype, device=runtime.device)
        mean_position = profile_mean(profiles, grid)
        width = torch.sqrt(profile_variance(profiles, grid))
        mass = torch.trapezoid(profiles, grid, dim=-1)
    return ProfileTableSummary(
        source=table.source,
        num_samples=table.num_samples,
        num_positions=table.num_positions,
        sample_time_min=float(np.min(table.sample_times)),
        sample_time_max=float(np.max(table.sample_times)),
        position_min=float(np.min(table.position_grid)),
        position_max=float(np.max(table.position_grid)),
        mean_position_min=float(torch.min(mean_position)),
        mean_position_max=float(torch.max(mean_position)),
        width_min=float(torch.min(width)),
        width_max=float(torch.max(width)),
        profile_mass_min=float(torch.min(mass)),
        profile_mass_max=float(torch.max(mass)),
        supported_formats=supported_profile_table_formats(),
    )


def analyze_profile_table_spectra(
    table: ProfileTable,
    *,
    num_modes: int,
    device: str | torch.device | None = "auto",
    normalize_each_profile: bool = False,
    capture_thresholds: Sequence[float] = (0.9, 0.95, 0.99),
) -> ProfileTableSpectralSummary:
    runtime = inspect_torch_runtime(device)
    with torch.inference_mode():
        grid, times, profiles = table.to_torch(dtype=runtime.preferred_real_dtype, device=runtime.device)
        if normalize_each_profile:
            profiles = normalize_profiles(profiles, grid)
        domain = InfiniteWell1D(left=grid[0], right=grid[-1])
        basis = InfiniteWellBasis(domain, num_modes=num_modes)
        coefficients = project_profiles_onto_basis(profiles, grid, basis)
    return ProfileTableSpectralSummary(
        runtime=runtime,
        source=table.source,
        num_modes=num_modes,
        grid=grid,
        sample_times=times,
        coefficients=coefficients,
        mean_position=profile_mean(profiles, grid),
        width=torch.sqrt(profile_variance(profiles, grid)),
        mass=profile_mass(profiles, grid),
        spectral_summary=summarize_spectral_batch(coefficients, capture_thresholds=capture_thresholds),
    )


def compress_profile_table(
    table: ProfileTable,
    *,
    num_modes: int,
    device: str | torch.device | None = "auto",
    normalize_each_profile: bool = False,
    ) -> ProfileCompressionWorkflowResult:
    runtime = inspect_torch_runtime(device)
    with torch.inference_mode():
        grid, times, profiles = table.to_torch(dtype=runtime.preferred_real_dtype, device=runtime.device)
        if normalize_each_profile:
            profiles = normalize_profiles(profiles, grid)
        domain = InfiniteWell1D(left=grid[0], right=grid[-1])
        compression = compress_profiles(profiles, grid, domain=domain, num_modes=num_modes)
        mean_position = profile_mean(profiles, grid)
        width = torch.sqrt(profile_variance(profiles, grid))
    return ProfileCompressionWorkflowResult(
        runtime=runtime,
        num_modes=num_modes,
        grid=grid,
        sample_times=times,
        source=table.source,
        coefficients=compression.coefficients,
        reconstruction=compression.reconstruction,
        mean_position=mean_position,
        width=width,
        mass=profile_mass(profiles, grid),
        error_summary=summarize_profile_reconstruction(profiles, compression.reconstruction, grid),
        spectral_summary=summarize_spectral_batch(compression.coefficients),
    )


def _capture_mode_budgets(summary: SpectralBatchSummary) -> tuple[CaptureModeBudget, ...]:
    thresholds = torch.as_tensor(summary.capture_thresholds).detach().cpu().tolist()
    mean_counts = torch.as_tensor(summary.mean_mode_counts_for_thresholds).detach().cpu().tolist()
    max_counts = torch.as_tensor(summary.max_mode_counts_for_thresholds).detach().cpu().tolist()
    return tuple(
        CaptureModeBudget(
            threshold=float(threshold),
            mean_mode_count=float(mean_count),
            max_mode_count=float(max_count),
        )
        for threshold, mean_count, max_count in zip(thresholds, mean_counts, max_counts)
    )


def build_profile_table_report(
    table: ProfileTable,
    *,
    analyze_num_modes: int = DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES,
    compress_num_modes: int = DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES,
    device: str | torch.device | None = "auto",
    normalize_each_profile: bool = False,
    capture_thresholds: Sequence[float] = DEFAULT_PROFILE_REPORT_CAPTURE_THRESHOLDS,
) -> ProfileTableReport:
    """Inspect, analyze, and compress one profile table through the shared spectral engine."""

    inspection = summarize_profile_table(table, device=device)
    analysis = analyze_profile_table_spectra(
        table,
        num_modes=analyze_num_modes,
        device=device,
        normalize_each_profile=normalize_each_profile,
        capture_thresholds=capture_thresholds,
    )
    compression = compress_profile_table(
        table,
        num_modes=compress_num_modes,
        device=device,
        normalize_each_profile=normalize_each_profile,
    )
    dominant_modes = tuple(
        int(mode)
        for mode in torch.as_tensor(analysis.spectral_summary.dominant_modes).detach().cpu().tolist()
    )
    return ProfileTableReport(
        overview=ProfileTableReportOverview(
            source=inspection.source,
            num_samples=inspection.num_samples,
            num_positions=inspection.num_positions,
            analyze_num_modes=analyze_num_modes,
            compress_num_modes=compress_num_modes,
            normalize_each_profile=normalize_each_profile,
            dominant_modes=dominant_modes,
            capture_mode_budgets=_capture_mode_budgets(analysis.spectral_summary),
            mean_relative_l2_error=float(compression.error_summary.mean_relative_l2_error),
            max_relative_l2_error=float(compression.error_summary.max_relative_l2_error),
            root_mean_square_error=float(compression.error_summary.root_mean_square_error),
            mean_position_min=inspection.mean_position_min,
            mean_position_max=inspection.mean_position_max,
            width_min=inspection.width_min,
            width_max=inspection.width_max,
            profile_mass_min=inspection.profile_mass_min,
            profile_mass_max=inspection.profile_mass_max,
        ),
        inspection=inspection,
        analysis=analysis,
        compression=compression,
    )


def export_feature_table_from_profile_table(
    table: ProfileTable | str | Path,
    *,
    num_modes: int = 32,
    device: str | torch.device | None = "auto",
    normalize_each_profile: bool = False,
    include_coefficients: bool = True,
    include_moments: bool = True,
    format: str = "csv",
) -> FeatureTableExportSummary:
    input_kind = "file" if isinstance(table, (str, Path)) else "profile-table"
    resolved_table = _coerce_profile_table_input(table)
    includes = tuple(
        item
        for item, enabled in (
            ("coefficients", include_coefficients),
            ("moments", include_moments),
        )
        if enabled
    )
    analysis = analyze_profile_table_spectra(
        resolved_table,
        num_modes=num_modes,
        device=device,
        normalize_each_profile=normalize_each_profile,
    )
    return _feature_table_from_spectral_summary(
        analysis,
        normalize_each_profile=normalize_each_profile,
        include_coefficients=include_coefficients,
        include_moments=include_moments,
        source_kind="profile-table",
        source_location=resolved_table.source,
        ordering=_profile_table_feature_export_ordering(resolved_table),
        metadata=_build_feature_export_metadata(
            input_metadata={
                "kind": input_kind,
                "source": resolved_table.source,
            },
            includes=includes,
            num_modes=num_modes,
            normalize_each_profile=normalize_each_profile,
            ordering=_profile_table_feature_export_ordering(resolved_table),
            output_format=format,
        ),
        output_format=format,
    )


def export_feature_table_from_database_query(
    database: DatabaseConfig | str | Any,
    query: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    materialization: ProfileTableMaterializationConfig | None = None,
    time_column: str = "time",
    position_columns: Sequence[str] | None = None,
    sort_by_time: bool = False,
    num_modes: int = 32,
    device: str | torch.device | None = "auto",
    normalize_each_profile: bool = False,
    include_coefficients: bool = True,
    include_moments: bool = True,
    format: str = "csv",
    create_if_missing: bool = False,
) -> FeatureTableExportSummary:
    def _export(materialized: DatabaseProfileTableMaterialization) -> FeatureTableExportSummary:
        includes = tuple(
            item
            for item, enabled in (
                ("coefficients", include_coefficients),
                ("moments", include_moments),
            )
            if enabled
        )
        analysis = analyze_profile_table_spectra(
            materialized.table,
            num_modes=num_modes,
            device=device,
            normalize_each_profile=normalize_each_profile,
        )
        return _feature_table_from_spectral_summary(
            analysis,
            normalize_each_profile=normalize_each_profile,
            include_coefficients=include_coefficients,
            include_moments=include_moments,
            source_kind="database-query",
            source_location=materialized.redacted_url,
            ordering=materialized.ordering,
            metadata=_build_feature_export_metadata(
                input_metadata=database_profile_query_artifact_metadata(
                    database,
                    query,
                    parameters=parameters,
                    materialization=materialization,
                    time_column=time_column,
                    position_columns=position_columns,
                    sort_by_time=sort_by_time,
                )["input"],
                includes=includes,
                num_modes=num_modes,
                normalize_each_profile=normalize_each_profile,
                ordering=materialized.ordering,
                output_format=format,
            ),
            output_format=format,
        )

    return _run_profile_table_workflow_from_database_query(
        database,
        query,
        workflow=_export,
        parameters=parameters,
        materialization=materialization,
        time_column=time_column,
        position_columns=position_columns,
        sort_by_time=sort_by_time,
        create_if_missing=create_if_missing,
    )


def train_tree_model(
    features: TabularDataset | str | Path,
    *,
    target_column: str,
    task: TreeTask = "regression",
    library: TreeLibrary = "auto",
    model: str | None = None,
    params: Mapping[str, Any] | None = None,
    feature_columns: Sequence[str] | None = None,
    test_fraction: float = 0.2,
    random_state: int = 0,
    export_dir: str | Path | None = None,
) -> TreeTrainingSummary:
    return train_tree_model_on_dataset(
        _coerce_tabular_dataset_input(features),
        target_column=target_column,
        task=task,
        library=library,
        model=model,
        params=params,
        feature_columns=feature_columns,
        test_fraction=test_fraction,
        random_state=random_state,
        export_dir=export_dir,
    )


def tune_tree_model(
    features: TabularDataset | str | Path,
    *,
    target_column: str,
    task: TreeTask = "regression",
    library: TreeLibrary = "auto",
    model: str | None = None,
    feature_columns: Sequence[str] | None = None,
    search_space: Mapping[str, Sequence[Any]] | None = None,
    search_kind: str = "random",
    n_iter: int = 30,
    cv: int = 5,
    scoring: str | None = None,
    test_fraction: float = 0.2,
    random_state: int = 0,
    export_dir: str | Path | None = None,
) -> TreeTuningSummary:
    return tune_tree_model_on_dataset(
        _coerce_tabular_dataset_input(features),
        target_column=target_column,
        task=task,
        library=library,
        model=model,
        feature_columns=feature_columns,
        search_space=search_space,
        search_kind=search_kind,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        test_fraction=test_fraction,
        random_state=random_state,
        export_dir=export_dir,
    )


def compare_profile_tables(
    reference: ProfileTable,
    candidate: ProfileTable,
    *,
    device: str | torch.device | None = "auto",
) -> ProfileTableComparisonWorkflowResult:
    runtime = inspect_torch_runtime(device)
    with torch.inference_mode():
        reference_grid, reference_times, reference_profiles = reference.to_torch(
            dtype=runtime.preferred_real_dtype,
            device=runtime.device,
        )
        candidate_grid, candidate_times, candidate_profiles = candidate.to_torch(
            dtype=runtime.preferred_real_dtype,
            device=runtime.device,
        )

        if reference_profiles.shape != candidate_profiles.shape:
            raise ValueError("reference and candidate tables must have matching profile shapes")
        if not torch.allclose(reference_grid, candidate_grid, atol=1e-9, rtol=1e-7):
            raise ValueError("reference and candidate tables must share the same position grid")
        if not torch.allclose(reference_times, candidate_times, atol=1e-9, rtol=1e-7):
            raise ValueError("reference and candidate tables must share the same sample times")

        residual_profiles = candidate_profiles - reference_profiles
    return ProfileTableComparisonWorkflowResult(
        runtime=runtime,
        reference_source=reference.source,
        candidate_source=candidate.source,
        grid=reference_grid,
        sample_times=reference_times,
        reference_profiles=reference_profiles,
        candidate_profiles=candidate_profiles,
        residual_profiles=residual_profiles,
        comparison=summarize_profile_comparison(reference_profiles, candidate_profiles, reference_grid),
    )


def sweep_profile_table_compression(
    table: ProfileTable,
    *,
    mode_counts: Sequence[int],
    device: str | torch.device | None = "auto",
    normalize_each_profile: bool = False,
) -> ProfileCompressionSweepSummary:
    runtime = inspect_torch_runtime(device)
    with torch.inference_mode():
        grid, _, profiles = table.to_torch(dtype=runtime.preferred_real_dtype, device=runtime.device)
        if normalize_each_profile:
            profiles = normalize_profiles(profiles, grid)
        domain = InfiniteWell1D(left=grid[0], right=grid[-1])
        counts = torch.as_tensor(mode_counts, dtype=torch.int64, device=runtime.device)
        summary = summarize_profile_compression(
            profiles,
            grid,
            mode_counts=counts,
            domain=domain,
        )
    return ProfileCompressionSweepSummary(
        runtime=runtime,
        source=table.source,
        mode_counts=summary.mode_counts,
        mean_relative_l2_error=summary.mean_relative_l2_error,
        max_relative_l2_error=summary.max_relative_l2_error,
    )


@dataclass(frozen=True, slots=True)
class TransportBenchmarkSummary:
    runtime: TorchRuntime
    scan_id: str
    title: str
    doi_url: str
    position_count: int
    time_count: int
    shots_per_time: int
    temperature_nK: float
    temperature_std_nK: float
    compression_summary: Any


def benchmark_transport_scan(
    *,
    scan_id: str = "scan11879_56",
    mode_counts: Sequence[int] = (8, 16, 32, 64),
    device: str | torch.device | None = "auto",
    cache_dir: str | None = None,
    preprocessing: DensityPreprocessingConfig | None = None,
) -> TransportBenchmarkSummary:
    runtime = inspect_torch_runtime(device)
    prepared = download_and_prepare_quantum_gas_transport_scan(
        scan_id=scan_id,
        cache_dir=cache_dir,
        preprocessing=preprocessing
        or DensityPreprocessingConfig(
            aggregate="mean",
            nan_fill_value=0.0,
            clip_negative=True,
            normalize_each_profile=True,
            drop_nonpositive_mass=True,
        ),
    )
    grid, _, profiles = prepared.to_torch(dtype=runtime.preferred_real_dtype, device=runtime.device)
    domain = InfiniteWell1D(left=grid[0], right=grid[-1])
    compression_summary = summarize_profile_compression(
        profiles,
        grid,
        mode_counts=torch.as_tensor(mode_counts, dtype=torch.int64, device=runtime.device),
        domain=domain,
    )
    return TransportBenchmarkSummary(
        runtime=runtime,
        scan_id=prepared.scan_id,
        title=prepared.title,
        doi_url=prepared.doi_url,
        position_count=int(prepared.position_axis_m.shape[0]),
        time_count=int(prepared.time_axis_s.shape[0]),
        shots_per_time=int(prepared.shots_per_time),
        temperature_nK=float(prepared.temperature_nK),
        temperature_std_nK=float(prepared.temperature_std_nK),
        compression_summary=compression_summary,
    )


def _fit_modal_surrogate_on_profile_table(
    table: ProfileTable,
    *,
    backend: str | None = "auto",
    num_modes: int,
    normalize_each_profile: bool = False,
    config: ModalSurrogateConfig | None = None,
    export_dir: str | None = None,
) -> tuple[Any, ModalTrainingSummary, Tensor, Tensor, Tensor]:
    chosen_config = config or ModalSurrogateConfig()
    grid, times, profiles = table.to_torch(dtype=torch.float64)
    if normalize_each_profile:
        profiles = normalize_profiles(profiles, grid)
    domain = InfiniteWell1D(left=grid[0], right=grid[-1])
    basis = InfiniteWellBasis(domain, num_modes=num_modes)
    regressor = create_modal_regressor("auto" if backend is None else backend, basis, config=chosen_config)
    result = regressor.fit(profiles, grid, sample_times=times)
    exported = None
    if export_dir is not None:
        suffix = {
            "torch": ".pt",
            "jax": ".pkl",
            "tensorflow": "",
        }.get(result.backend, "")
        export_path = Path(export_dir)
        if suffix and export_path.suffix != suffix:
            export_path = export_path / f"{result.backend}_modal_surrogate{suffix}" if export_path.suffix == "" else export_path.with_suffix(suffix)
        exported = str(regressor.export(export_path))
    training = ModalTrainingSummary(
        source=table.source,
        backend=result.backend,
        num_modes=num_modes,
        result=result,
        export_path=exported,
    )
    return regressor, training, grid, times, profiles


@dataclass(frozen=True, slots=True)
class TensorFlowTrainingSummary:
    source: str | None
    num_modes: int
    result: TensorFlowModalRegressionResult
    export_path: str | None


@dataclass(frozen=True, slots=True)
class ModalTrainingSummary:
    source: str | None
    backend: str
    num_modes: int
    result: ModalRegressionResult
    export_path: str | None


@dataclass(frozen=True, slots=True)
class TensorFlowEvaluationSummary:
    source: str | None
    num_modes: int
    training: TensorFlowTrainingSummary
    sample_times: Tensor
    grid: Tensor
    predicted_coefficients: Tensor
    predicted_moments: Tensor
    reconstructed_profiles: Tensor
    comparison: ProfileComparisonSummary


@dataclass(frozen=True, slots=True)
class ModalEvaluationSummary:
    source: str | None
    backend: str
    num_modes: int
    training: ModalTrainingSummary
    sample_times: Tensor
    grid: Tensor
    predicted_coefficients: Tensor
    predicted_moments: Tensor
    reconstructed_profiles: Tensor
    comparison: ProfileComparisonSummary


def _fit_tensorflow_surrogate_on_profile_table(
    table: ProfileTable,
    *,
    num_modes: int,
    normalize_each_profile: bool = False,
    config: TensorFlowRegressorConfig | None = None,
    export_dir: str | None = None,
) -> tuple[TensorFlowModalRegressor, TensorFlowTrainingSummary, Tensor, Tensor, Tensor]:
    grid, times, profiles = table.to_torch(dtype=torch.float64)
    if normalize_each_profile:
        profiles = normalize_profiles(profiles, grid)
    domain = InfiniteWell1D(left=grid[0], right=grid[-1])
    basis = InfiniteWellBasis(domain, num_modes=num_modes)
    regressor = TensorFlowModalRegressor(basis, config=config or TensorFlowRegressorConfig())
    result = regressor.fit(profiles, grid, sample_times=times)
    exported = None
    if export_dir is not None:
        exported = str(regressor.export(export_dir))
    training = TensorFlowTrainingSummary(
        source=table.source,
        num_modes=num_modes,
        result=result,
        export_path=exported,
    )
    return regressor, training, grid, times, profiles


def train_tensorflow_surrogate_on_profile_table(
    table: ProfileTable,
    *,
    num_modes: int,
    normalize_each_profile: bool = False,
    config: TensorFlowRegressorConfig | None = None,
    export_dir: str | None = None,
) -> TensorFlowTrainingSummary:
    _, training, _, _, _ = _fit_tensorflow_surrogate_on_profile_table(
        table,
        num_modes=num_modes,
        normalize_each_profile=normalize_each_profile,
        config=config,
        export_dir=export_dir,
    )
    return training


def evaluate_tensorflow_surrogate_on_profile_table(
    table: ProfileTable,
    *,
    num_modes: int,
    normalize_each_profile: bool = False,
    config: TensorFlowRegressorConfig | None = None,
    export_dir: str | None = None,
) -> TensorFlowEvaluationSummary:
    regressor, training, grid, times, profiles = _fit_tensorflow_surrogate_on_profile_table(
        table,
        num_modes=num_modes,
        normalize_each_profile=normalize_each_profile,
        config=config,
        export_dir=export_dir,
    )
    predicted_coefficients = torch.as_tensor(
        regressor.predict_coefficients(profiles.detach().cpu().numpy(), sample_times=times.detach().cpu().numpy()),
        dtype=grid.dtype,
    )
    predicted_moments = torch.as_tensor(
        regressor.predict_moments(profiles.detach().cpu().numpy(), sample_times=times.detach().cpu().numpy()),
        dtype=grid.dtype,
    )
    reconstructed_profiles = regressor.reconstruct_profiles(
        profiles.detach().cpu().numpy(),
        grid.detach().cpu().numpy(),
        sample_times=times.detach().cpu().numpy(),
    ).to(dtype=grid.dtype)
    return TensorFlowEvaluationSummary(
        source=table.source,
        num_modes=num_modes,
        training=training,
        sample_times=times,
        grid=grid,
        predicted_coefficients=predicted_coefficients,
        predicted_moments=predicted_moments,
        reconstructed_profiles=reconstructed_profiles,
        comparison=summarize_profile_comparison(profiles, reconstructed_profiles, grid),
    )


def train_modal_surrogate_on_profile_table(
    table: ProfileTable,
    *,
    backend: str | None = "auto",
    num_modes: int,
    normalize_each_profile: bool = False,
    config: ModalSurrogateConfig | None = None,
    export_dir: str | None = None,
) -> ModalTrainingSummary:
    _, training, _, _, _ = _fit_modal_surrogate_on_profile_table(
        table,
        backend=backend,
        num_modes=num_modes,
        normalize_each_profile=normalize_each_profile,
        config=config,
        export_dir=export_dir,
    )
    return training


def evaluate_modal_surrogate_on_profile_table(
    table: ProfileTable,
    *,
    backend: str | None = "auto",
    num_modes: int,
    normalize_each_profile: bool = False,
    config: ModalSurrogateConfig | None = None,
    export_dir: str | None = None,
) -> ModalEvaluationSummary:
    regressor, training, grid, times, profiles = _fit_modal_surrogate_on_profile_table(
        table,
        backend=backend,
        num_modes=num_modes,
        normalize_each_profile=normalize_each_profile,
        config=config,
        export_dir=export_dir,
    )
    predicted_coefficients = torch.as_tensor(
        regressor.predict_coefficients(
            profiles.detach().cpu().numpy(),
            sample_times=times.detach().cpu().numpy(),
        ),
        dtype=grid.dtype,
    )
    predicted_moments = torch.as_tensor(
        regressor.predict_moments(
            profiles.detach().cpu().numpy(),
            sample_times=times.detach().cpu().numpy(),
        ),
        dtype=grid.dtype,
    )
    reconstructed_profiles = regressor.reconstruct_profiles(
        profiles.detach().cpu().numpy(),
        grid.detach().cpu().numpy(),
        sample_times=times.detach().cpu().numpy(),
    ).to(dtype=grid.dtype)
    return ModalEvaluationSummary(
        source=table.source,
        backend=training.backend,
        num_modes=num_modes,
        training=training,
        sample_times=times,
        grid=grid,
        predicted_coefficients=predicted_coefficients,
        predicted_moments=predicted_moments,
        reconstructed_profiles=reconstructed_profiles,
        comparison=summarize_profile_comparison(profiles, reconstructed_profiles, grid),
    )


def train_modal_surrogate_from_database_query(
    database: DatabaseConfig | str | Any,
    query: str,
    *,
    backend: str | None = "auto",
    parameters: Mapping[str, Any] | None = None,
    materialization: ProfileTableMaterializationConfig | None = None,
    time_column: str = "time",
    position_columns: Sequence[str] | None = None,
    sort_by_time: bool = False,
    num_modes: int,
    normalize_each_profile: bool = False,
    config: ModalSurrogateConfig | None = None,
    export_dir: str | None = None,
    create_if_missing: bool = False,
) -> ModalTrainingSummary:
    return _run_profile_table_workflow_from_database_query(
        database,
        query,
        workflow=lambda materialized: train_modal_surrogate_on_profile_table(
            materialized.table,
            backend=backend,
            num_modes=num_modes,
            normalize_each_profile=normalize_each_profile,
            config=config,
            export_dir=export_dir,
        ),
        parameters=parameters,
        materialization=materialization,
        time_column=time_column,
        position_columns=position_columns,
        sort_by_time=sort_by_time,
        create_if_missing=create_if_missing,
    )


def evaluate_modal_surrogate_from_database_query(
    database: DatabaseConfig | str | Any,
    query: str,
    *,
    backend: str | None = "auto",
    parameters: Mapping[str, Any] | None = None,
    materialization: ProfileTableMaterializationConfig | None = None,
    time_column: str = "time",
    position_columns: Sequence[str] | None = None,
    sort_by_time: bool = False,
    num_modes: int,
    normalize_each_profile: bool = False,
    config: ModalSurrogateConfig | None = None,
    export_dir: str | None = None,
    create_if_missing: bool = False,
) -> ModalEvaluationSummary:
    return _run_profile_table_workflow_from_database_query(
        database,
        query,
        workflow=lambda materialized: evaluate_modal_surrogate_on_profile_table(
            materialized.table,
            backend=backend,
            num_modes=num_modes,
            normalize_each_profile=normalize_each_profile,
            config=config,
            export_dir=export_dir,
        ),
        parameters=parameters,
        materialization=materialization,
        time_column=time_column,
        position_columns=position_columns,
        sort_by_time=sort_by_time,
        create_if_missing=create_if_missing,
    )


def load_and_compress_profile_table(
    path: str,
    *,
    num_modes: int,
    device: str | torch.device | None = "auto",
    normalize_each_profile: bool = False,
) -> ProfileCompressionWorkflowResult:
    return compress_profile_table(
        load_profile_table(path),
        num_modes=num_modes,
        device=device,
        normalize_each_profile=normalize_each_profile,
    )


def load_profile_table_report(
    path: str | Path,
    *,
    analyze_num_modes: int = DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES,
    compress_num_modes: int = DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES,
    device: str | torch.device | None = "auto",
    normalize_each_profile: bool = False,
    capture_thresholds: Sequence[float] = DEFAULT_PROFILE_REPORT_CAPTURE_THRESHOLDS,
) -> ProfileTableReport:
    """Load a file-backed profile table and build the shared inspect-analyze-compress report."""

    return build_profile_table_report(
        load_profile_table(path),
        analyze_num_modes=analyze_num_modes,
        compress_num_modes=compress_num_modes,
        device=device,
        normalize_each_profile=normalize_each_profile,
        capture_thresholds=capture_thresholds,
    )


def build_profile_table_report_from_database_query(
    database: DatabaseConfig | str | Any,
    query: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    materialization: ProfileTableMaterializationConfig | None = None,
    time_column: str = "time",
    position_columns: Sequence[str] | None = None,
    sort_by_time: bool = False,
    analyze_num_modes: int = DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES,
    compress_num_modes: int = DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES,
    device: str | torch.device | None = "auto",
    normalize_each_profile: bool = False,
    capture_thresholds: Sequence[float] = DEFAULT_PROFILE_REPORT_CAPTURE_THRESHOLDS,
    create_if_missing: bool = False,
) -> ProfileTableReport:
    """Materialize a SQL query through the shared profile-table contract and build the same report."""

    return _run_profile_table_workflow_from_database_query(
        database,
        query,
        workflow=lambda materialized: build_profile_table_report(
            materialized.table,
            analyze_num_modes=analyze_num_modes,
            compress_num_modes=compress_num_modes,
            device=device,
            normalize_each_profile=normalize_each_profile,
            capture_thresholds=capture_thresholds,
        ),
        parameters=parameters,
        materialization=materialization,
        time_column=time_column,
        position_columns=position_columns,
        sort_by_time=sort_by_time,
        create_if_missing=create_if_missing,
    )


def simulate_packet_sweep(
    packet_specs: Sequence[Mapping[str, float]],
    *,
    times: Sequence[float] = (0.0, 1e-3, 5e-3),
    num_modes: int = 128,
    quadrature_points: int = 4096,
    grid_points: int = 512,
    device: str | torch.device | None = "auto",
) -> PacketSweepSummary:
    if not packet_specs:
        raise ValueError("packet_specs must contain at least one packet specification")
    context = build_engine(
        num_modes=num_modes,
        quadrature_points=quadrature_points,
        device=device,
    )
    grid = context.domain.grid(grid_points)
    evaluation_times = torch.as_tensor(times, dtype=context.domain.real_dtype, device=context.domain.device)
    items: list[PacketSweepItemSummary] = []
    shared_times = evaluation_times.detach().cpu()
    for spec in packet_specs:
        summary = _simulate_packet_with_context(
            context,
            center=float(spec["center"]),
            width=float(spec["width"]),
            wavenumber=float(spec["wavenumber"]),
            phase=float(spec.get("phase", 0.0)),
            evaluation_times=evaluation_times,
            grid=grid,
        )
        items.append(
            PacketSweepItemSummary(
                center=float(spec["center"]),
                width=float(spec["width"]),
                wavenumber=float(spec["wavenumber"]),
                phase=float(spec.get("phase", 0.0)),
                spectral_norm=float(summary.spectral_norm),
                final_expectation_position=float(summary.expectation_position[-1]),
                final_left_probability=float(summary.left_probability[-1]),
                final_right_probability=float(summary.right_probability[-1]),
                final_total_probability=float(summary.total_probability[-1]),
            )
        )
        shared_times = summary.times.detach().cpu()
    return PacketSweepSummary(
        runtime=context.runtime,
        times=shared_times,
        items=tuple(items),
    )


__all__ = [
    "CaptureModeBudget",
    "DatabaseExecutionSummary",
    "DatabaseInspectionSummary",
    "DatabaseProfileTableMaterialization",
    "DatabaseQuerySummary",
    "DatabaseWriteSummary",
    "EngineContext",
    "EnvironmentReport",
    "FeatureTableExportSummary",
    "ForwardSimulationSummary",
    "InverseFitSummary",
    "InstallationValidation",
    "ModalEvaluationSummary",
    "ModalTrainingSummary",
    "PacketProjectionSummary",
    "PacketSweepItemSummary",
    "PacketSweepSummary",
    "ProfileTableComparisonWorkflowResult",
    "ProfileTableReport",
    "ProfileTableReportOverview",
    "ProfileCompressionWorkflowResult",
    "ProfileCompressionSweepSummary",
    "ProfileTableSpectralSummary",
    "ProfileTableSummary",
    "TabularDatasetSummary",
    "TensorFlowEvaluationSummary",
    "TensorFlowTrainingSummary",
    "TransportBenchmarkSummary",
    "analyze_profile_table_spectra",
    "analyze_profile_table_from_database_query",
    "benchmark_transport_scan",
    "build_profile_table_report",
    "build_profile_table_report_from_database_query",
    "bootstrap_local_database",
    "coerce_database_table_types",
    "build_engine",
    "compare_profile_tables",
    "compress_profile_table",
    "compress_profile_table_from_database_query",
    "database_profile_query_artifact_metadata",
    "database_query_artifact_metadata",
    "describe_database_table",
    "execute_database_script",
    "execute_database_statement",
    "execute_database_query",
    "evaluate_modal_surrogate_from_database_query",
    "evaluate_modal_surrogate_on_profile_table",
    "evaluate_tensorflow_surrogate_on_profile_table",
    "export_feature_table_from_database_query",
    "export_feature_table_from_profile_table",
    "fit_gaussian_packet_to_density",
    "fit_gaussian_packet_to_profile_table",
    "fit_gaussian_packet_to_profile_table_from_database_query",
    "inspect_database",
    "inspect_environment",
    "interpolate_database_time_series",
    "inspect_ml_backend_support",
    "inspect_tree_backend_support",
    "list_database_tables",
    "load_and_compress_profile_table",
    "load_profile_table_report",
    "load_tabular_dataset_from_path",
    "make_packet_state",
    "materialize_database_query",
    "materialize_profile_table_from_database_query",
    "materialize_database_query_to_table",
    "pivot_database_table",
    "project_gaussian_packet",
    "simulate_gaussian_packet",
    "simulate_packet_sweep",
    "summarize_database_query_result",
    "summarize_profile_table",
    "summarize_tabular_dataset",
    "sweep_profile_table_compression",
    "train_tree_model",
    "train_modal_surrogate_from_database_query",
    "train_modal_surrogate_on_profile_table",
    "train_tensorflow_surrogate_on_profile_table",
    "tune_tree_model",
    "unpivot_database_table",
    "validate_installation",
    "window_aggregate_database_query",
    "write_profile_table_to_database",
    "write_tabular_dataset_to_database",
]
