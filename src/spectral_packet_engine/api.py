from dataclasses import replace
from pathlib import Path
from typing import Literal
import warnings

from spectral_packet_engine.artifacts import (
    inspect_artifact_directory,
    to_serializable,
    write_compression_artifacts,
    write_compression_sweep_artifacts,
    write_feature_table_artifacts,
    write_forward_artifacts,
    write_inverse_artifacts,
    write_modal_evaluation_artifacts,
    write_modal_training_artifacts,
    write_packet_sweep_artifacts,
    write_profile_comparison_artifacts,
    write_spectral_analysis_artifacts,
    write_tree_training_artifacts,
    write_tree_tuning_artifacts,
    write_tabular_artifacts,
    write_tensorflow_evaluation_artifacts,
    write_tensorflow_training_artifacts,
    write_transport_benchmark_artifacts,
)
from spectral_packet_engine.ml import ModalSurrogateConfig
from spectral_packet_engine.product import (
    DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES,
    DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES,
    PRODUCT_NAME,
    PRODUCT_SPINE_STATEMENT,
    RUNTIME_SPINE_STATEMENT,
    guide_workflow,
    inspect_product_identity,
    resolve_workflow_identity,
)
from spectral_packet_engine.service_runtime import inspect_api_stack
from spectral_packet_engine.service_status import inspect_service_status, track_service_task
from spectral_packet_engine.tabular import TabularDataset, supported_tabular_formats
from spectral_packet_engine.table_io import ProfileTable, supported_profile_table_formats
from spectral_packet_engine.tf_surrogate import TensorFlowRegressorConfig
from spectral_packet_engine.workflows import (
    analyze_profile_table_spectra,
    analyze_profile_table_from_database_query,
    benchmark_transport_scan,
    build_profile_table_report,
    build_profile_table_report_from_database_query,
    bootstrap_local_database,
    compare_profile_tables,
    compress_profile_table,
    compress_profile_table_from_database_query,
    database_profile_query_workflow_artifact_metadata,
    database_query_workflow_artifact_metadata,
    describe_database_table,
    execute_database_script,
    execute_database_statement,
    evaluate_modal_surrogate_from_database_query,
    evaluate_modal_surrogate_on_profile_table,
    evaluate_tensorflow_surrogate_on_profile_table,
    export_feature_table_from_database_query,
    export_feature_table_from_profile_table,
    inspect_database,
    fit_gaussian_packet_to_density,
    fit_gaussian_packet_to_profile_table_from_database_query,
    inspect_environment,
    inspect_ml_backend_support,
    inspect_tree_backend_support,
    materialize_database_query,
    materialize_database_query_to_table,
    simulate_packet_sweep,
    project_gaussian_packet,
    simulate_gaussian_packet,
    summarize_database_query_result,
    summarize_profile_table,
    summarize_tabular_dataset,
    sweep_profile_table_compression,
    train_tree_model,
    train_modal_surrogate_from_database_query,
    train_modal_surrogate_on_profile_table,
    train_tensorflow_surrogate_on_profile_table,
    tune_tree_model,
    validate_installation,
    write_tabular_dataset_to_database,
)


def api_is_available() -> bool:
    return inspect_api_stack().compatible


def create_api_app():
    api_stack = inspect_api_stack()
    if not api_stack.installed:
        raise ModuleNotFoundError("The API service requires the 'api' extra.")
    if not api_stack.compatible:
        detail = "" if api_stack.error is None else f" ({api_stack.error})"
        raise RuntimeError(
            "FastAPI is installed, but the FastAPI/Starlette stack is incompatible in this environment. "
            f"Reinstall the 'api' extra in a clean environment{detail}."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            from fastapi import FastAPI, Request
            from fastapi.responses import JSONResponse
            from pydantic import BaseModel, Field
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("The API service requires the 'api' extra.") from exc

    class PacketSpec(BaseModel):
        center: float = 0.30
        width: float = 0.07
        wavenumber: float = 25.0
        phase: float = 0.0

    class PacketSweepRequest(BaseModel):
        packet_specs: list[PacketSpec]
        times: list[float] = Field(default_factory=lambda: [0.0, 1e-3, 5e-3])
        num_modes: int = 128
        quadrature_points: int = 4096
        grid_points: int = 512
        device: str = "auto"
        output_dir: str | None = None

    class ForwardRequest(BaseModel):
        packet: PacketSpec = Field(default_factory=PacketSpec)
        times: list[float] = Field(default_factory=lambda: [0.0, 1e-3, 5e-3])
        num_modes: int = 128
        quadrature_points: int = 4096
        grid_points: int = 512
        device: str = "auto"
        output_dir: str | None = None

    class ProfileTablePayload(BaseModel):
        position_grid: list[float]
        sample_times: list[float]
        profiles: list[list[float]]
        source: str | None = None

        def to_table(self) -> ProfileTable:
            return ProfileTable(
                position_grid=self.position_grid,
                sample_times=self.sample_times,
                profiles=self.profiles,
                source=self.source,
            )

    class TabularDatasetPayload(BaseModel):
        rows: list[dict[str, object]]

        def to_dataset(self) -> TabularDataset:
            if not self.rows:
                raise ValueError("rows must not be empty")
            from spectral_packet_engine.tabular import TabularDataset as RuntimeTabularDataset

            return RuntimeTabularDataset.from_rows(self.rows)

    class DatabaseQueryRequest(BaseModel):
        database: str
        query: str
        parameters: dict[str, object] = Field(default_factory=dict)
        output_dir: str | None = None

    class DatabaseWriteRequest(BaseModel):
        database: str
        table_name: str
        dataset: TabularDatasetPayload
        if_exists: Literal["fail", "replace", "append"] = "fail"

    class DatabaseExecuteRequest(BaseModel):
        database: str
        statement: str
        parameters: dict[str, object] = Field(default_factory=dict)

    class DatabaseScriptRequest(BaseModel):
        database: str
        script: str

    class DatabaseMaterializeRequest(BaseModel):
        database: str
        table_name: str
        query: str
        parameters: dict[str, object] = Field(default_factory=dict)
        replace: bool = False

    class DatabaseAnalyzeRequest(BaseModel):
        database: str
        query: str
        parameters: dict[str, object] = Field(default_factory=dict)
        time_column: str = "time"
        position_columns: list[str] | None = None
        sort_by_time: bool = False
        num_modes: int = 32
        device: str = "auto"
        normalize_each_profile: bool = False
        output_dir: str | None = None

    class DatabaseCompressionRequest(BaseModel):
        database: str
        query: str
        parameters: dict[str, object] = Field(default_factory=dict)
        time_column: str = "time"
        position_columns: list[str] | None = None
        sort_by_time: bool = False
        num_modes: int = 32
        device: str = "auto"
        normalize_each_profile: bool = False
        output_dir: str | None = None

    class ProfileReportRequest(BaseModel):
        table: ProfileTablePayload
        analyze_num_modes: int = DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES
        compress_num_modes: int = DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES
        device: str = "auto"
        normalize_each_profile: bool = False
        output_dir: str | None = None

    class DatabaseProfileReportRequest(BaseModel):
        database: str
        query: str
        parameters: dict[str, object] = Field(default_factory=dict)
        time_column: str = "time"
        position_columns: list[str] | None = None
        sort_by_time: bool = False
        analyze_num_modes: int = DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES
        compress_num_modes: int = DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES
        device: str = "auto"
        normalize_each_profile: bool = False
        output_dir: str | None = None

    class FeatureExportRequest(BaseModel):
        table: ProfileTablePayload
        num_modes: int = 32
        device: str = "auto"
        normalize_each_profile: bool = False
        include: list[Literal["coefficients", "moments"]] = Field(
            default_factory=lambda: ["coefficients", "moments"]
        )
        format: Literal["csv", "parquet"] = "csv"
        output_dir: str | None = None

    class DatabaseFeatureExportRequest(BaseModel):
        database: str
        query: str
        parameters: dict[str, object] = Field(default_factory=dict)
        time_column: str = "time"
        position_columns: list[str] | None = None
        sort_by_time: bool = False
        num_modes: int = 32
        device: str = "auto"
        normalize_each_profile: bool = False
        include: list[Literal["coefficients", "moments"]] = Field(
            default_factory=lambda: ["coefficients", "moments"]
        )
        format: Literal["csv", "parquet"] = "csv"
        output_dir: str | None = None

    class CompressionRequest(BaseModel):
        table: ProfileTablePayload
        num_modes: int = 32
        device: str = "auto"
        normalize_each_profile: bool = False
        output_dir: str | None = None

    class AnalysisRequest(BaseModel):
        table: ProfileTablePayload
        num_modes: int = 32
        device: str = "auto"
        normalize_each_profile: bool = False
        output_dir: str | None = None

    class CompressionSweepRequest(BaseModel):
        table: ProfileTablePayload
        mode_counts: list[int] = Field(default_factory=lambda: [4, 8, 16, 32])
        device: str = "auto"
        normalize_each_profile: bool = False
        output_dir: str | None = None

    class TableComparisonRequest(BaseModel):
        reference: ProfileTablePayload
        candidate: ProfileTablePayload
        device: str = "auto"
        output_dir: str | None = None

    class InverseRequest(BaseModel):
        table: ProfileTablePayload
        initial_center: float = 0.36
        initial_width: float = 0.11
        initial_wavenumber: float = 22.0
        initial_phase: float = 0.0
        num_modes: int = 128
        quadrature_points: int = 2048
        steps: int = 200
        learning_rate: float = 0.05
        device: str = "auto"
        output_dir: str | None = None

    class DatabaseInverseRequest(BaseModel):
        database: str
        query: str
        parameters: dict[str, object] = Field(default_factory=dict)
        time_column: str = "time"
        position_columns: list[str] | None = None
        sort_by_time: bool = False
        initial_center: float = 0.36
        initial_width: float = 0.11
        initial_wavenumber: float = 22.0
        initial_phase: float = 0.0
        num_modes: int = 128
        quadrature_points: int = 2048
        steps: int = 200
        learning_rate: float = 0.05
        device: str = "auto"
        output_dir: str | None = None

    class TransportRequest(BaseModel):
        scan_id: str = "scan11879_56"
        mode_counts: list[int] = Field(default_factory=lambda: [8, 16, 32, 64])
        device: str = "auto"
        output_dir: str | None = None

    class TensorFlowRequest(BaseModel):
        table: ProfileTablePayload
        num_modes: int = 16
        epochs: int = 20
        batch_size: int = 64
        normalize_each_profile: bool = False
        export_dir: str | None = None
        output_dir: str | None = None

    class TensorFlowEvaluationRequest(BaseModel):
        table: ProfileTablePayload
        num_modes: int = 16
        epochs: int = 20
        batch_size: int = 64
        normalize_each_profile: bool = False
        export_dir: str | None = None
        output_dir: str | None = None

    class ModalTrainingRequest(BaseModel):
        table: ProfileTablePayload
        backend: Literal["auto", "torch", "jax", "tensorflow"] = "auto"
        num_modes: int = 16
        epochs: int = 20
        batch_size: int = 64
        normalize_each_profile: bool = False
        device: str = "auto"
        export_dir: str | None = None
        output_dir: str | None = None

    class ModalEvaluationRequest(BaseModel):
        table: ProfileTablePayload
        backend: Literal["auto", "torch", "jax", "tensorflow"] = "auto"
        num_modes: int = 16
        epochs: int = 20
        batch_size: int = 64
        normalize_each_profile: bool = False
        device: str = "auto"
        export_dir: str | None = None
        output_dir: str | None = None

    class DatabaseModalTrainingRequest(BaseModel):
        database: str
        query: str
        parameters: dict[str, object] = Field(default_factory=dict)
        backend: Literal["auto", "torch", "jax", "tensorflow"] = "auto"
        time_column: str = "time"
        position_columns: list[str] | None = None
        sort_by_time: bool = False
        num_modes: int = 16
        epochs: int = 20
        batch_size: int = 64
        normalize_each_profile: bool = False
        device: str = "auto"
        export_dir: str | None = None
        output_dir: str | None = None

    class DatabaseModalEvaluationRequest(BaseModel):
        database: str
        query: str
        parameters: dict[str, object] = Field(default_factory=dict)
        backend: Literal["auto", "torch", "jax", "tensorflow"] = "auto"
        time_column: str = "time"
        position_columns: list[str] | None = None
        sort_by_time: bool = False
        num_modes: int = 16
        epochs: int = 20
        batch_size: int = 64
        normalize_each_profile: bool = False
        device: str = "auto"
        export_dir: str | None = None
        output_dir: str | None = None

    class TreeTrainingRequest(BaseModel):
        dataset: TabularDatasetPayload
        target_column: str
        feature_columns: list[str] | None = None
        task: Literal["regression", "classification"] = "regression"
        library: Literal["auto", "sklearn", "xgboost", "lightgbm", "catboost"] = "auto"
        model: str | None = None
        params: dict[str, object] = Field(default_factory=dict)
        test_fraction: float = 0.2
        random_state: int = 0
        export_dir: str | None = None
        output_dir: str | None = None

    class TreeTuningRequest(BaseModel):
        dataset: TabularDatasetPayload
        target_column: str
        feature_columns: list[str] | None = None
        task: Literal["regression", "classification"] = "regression"
        library: Literal["auto", "sklearn", "xgboost", "lightgbm", "catboost"] = "auto"
        model: str | None = None
        search_space: dict[str, list[object]]
        search_kind: Literal["grid", "random"] = "random"
        n_iter: int = 30
        cv: int = 5
        scoring: str | None = None
        test_fraction: float = 0.2
        random_state: int = 0
        export_dir: str | None = None
        output_dir: str | None = None

    app = FastAPI(
        title=f"{PRODUCT_NAME} API",
        version="0.2.0",
        description=f"{PRODUCT_SPINE_STATEMENT} Runtime spine: {RUNTIME_SPINE_STATEMENT}",
    )

    def _workflow_for_route(method: str, path: str):
        return resolve_workflow_identity("api", f"{method.upper()} {path}")

    @app.middleware("http")
    async def track_request_runtime(request, call_next):
        route = request.scope.get("route")
        route_path = request.url.path if route is None else getattr(route, "path", request.url.path)
        workflow = _workflow_for_route(request.method, route_path)
        with track_service_task(
            workflow.workflow_id if workflow is not None else f"{request.method} {route_path}",
            interface="api",
            workflow_id=None if workflow is None else workflow.workflow_id,
            surface_action=f"{request.method} {route_path}",
            metadata={"method": request.method, "path": route_path},
        ):
            return await call_next(request)

    @app.exception_handler(FileNotFoundError)
    async def handle_file_not_found(_: Request, exc: FileNotFoundError):
        return JSONResponse(status_code=404, content={"error": str(exc)})

    @app.exception_handler(ModuleNotFoundError)
    async def handle_missing_module(_: Request, exc: ModuleNotFoundError):
        return JSONResponse(status_code=503, content={"error": str(exc)})

    @app.exception_handler(ValueError)
    @app.exception_handler(TypeError)
    @app.exception_handler(RuntimeError)
    async def handle_request_error(_: Request, exc: Exception):
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "product_name": PRODUCT_NAME, "version": "0.2.0"}

    @app.get("/product")
    def product():
        return to_serializable(inspect_product_identity())

    @app.get("/workflow/guide")
    def workflow_guide(
        input_kind: Literal["profile-table-file", "profile-table-sql"] = "profile-table-file",
        goal: Literal["report", "inverse-fit", "feature-model"] = "report",
    ):
        return to_serializable(guide_workflow(surface="api", input_kind=input_kind, goal=goal))

    @app.get("/status")
    def status():
        return to_serializable(inspect_service_status())

    @app.get("/capabilities")
    def capabilities(device: str = "auto"):
        return to_serializable(inspect_environment(device))

    @app.get("/validate-install")
    def validate_install(device: str = "auto"):
        return to_serializable(validate_installation(device))

    @app.get("/api/stack")
    def api_stack_report():
        return to_serializable(inspect_api_stack())

    @app.get("/ml/backends")
    def ml_backends(device: str = "auto"):
        return to_serializable(inspect_ml_backend_support(device))

    @app.get("/tree/backends")
    def tree_backends(library: Literal["auto", "sklearn", "xgboost", "lightgbm", "catboost"] = "auto"):
        return to_serializable(inspect_tree_backend_support(requested_library=library))

    @app.get("/file-formats")
    def file_formats():
        return supported_profile_table_formats()

    @app.get("/tabular-formats")
    def tabular_formats():
        return supported_tabular_formats()

    @app.get("/artifacts")
    def artifacts(output_dir: str):
        return to_serializable(inspect_artifact_directory(output_dir))

    @app.get("/database/inspect")
    def database_inspect(database: str):
        return to_serializable(inspect_database(database))

    @app.get("/database/bootstrap")
    def database_bootstrap(database: str):
        return to_serializable(bootstrap_local_database(database))

    @app.get("/database/tables/{table_name}")
    def database_describe(database: str, table_name: str):
        return to_serializable(describe_database_table(database, table_name))

    @app.post("/database/query")
    def database_query(request: DatabaseQueryRequest):
        result = materialize_database_query(
            request.database,
            request.query,
            parameters=request.parameters,
        )
        if request.output_dir is not None:
            write_tabular_artifacts(
                request.output_dir,
                result.dataset,
                summary_name="db_query_summary.json",
                table_name="query_result.csv",
                metadata=database_query_workflow_artifact_metadata(
                    "db-query",
                    request.database,
                    request.query,
                    parameters=request.parameters,
                ),
            )
        return to_serializable(
            summarize_database_query_result(
                request.database,
                request.query,
                result,
                parameters=request.parameters,
            )
        )

    @app.post("/database/execute")
    def database_execute(request: DatabaseExecuteRequest):
        return to_serializable(
            execute_database_statement(
                request.database,
                request.statement,
                parameters=request.parameters,
                create_if_missing=True,
            )
        )

    @app.post("/database/script")
    def database_script(request: DatabaseScriptRequest):
        return to_serializable(
            execute_database_script(
                request.database,
                request.script,
                create_if_missing=True,
            )
        )

    @app.post("/database/write")
    def database_write(request: DatabaseWriteRequest):
        return to_serializable(
            write_tabular_dataset_to_database(
                request.database,
                request.table_name,
                request.dataset.to_dataset(),
                if_exists=request.if_exists,
            )
        )

    @app.post("/database/materialize")
    def database_materialize(request: DatabaseMaterializeRequest):
        return to_serializable(
            materialize_database_query_to_table(
                request.database,
                request.table_name,
                request.query,
                parameters=request.parameters,
                replace=request.replace,
            )
        )

    @app.post("/forward")
    def forward(request: ForwardRequest):
        summary = simulate_gaussian_packet(
            center=request.packet.center,
            width=request.packet.width,
            wavenumber=request.packet.wavenumber,
            phase=request.packet.phase,
            times=request.times,
            num_modes=request.num_modes,
            quadrature_points=request.quadrature_points,
            grid_points=request.grid_points,
            device=request.device,
        )
        if request.output_dir is not None:
            write_forward_artifacts(request.output_dir, summary)
        return to_serializable(summary)

    @app.post("/project")
    def project(request: ForwardRequest):
        return to_serializable(
            project_gaussian_packet(
                center=request.packet.center,
                width=request.packet.width,
                wavenumber=request.packet.wavenumber,
                phase=request.packet.phase,
                num_modes=request.num_modes,
                quadrature_points=request.quadrature_points,
                grid_points=request.grid_points,
                device=request.device,
            )
        )

    @app.post("/packet-sweep")
    def packet_sweep(request: PacketSweepRequest):
        summary = simulate_packet_sweep(
            [
                {
                    "center": item.center,
                    "width": item.width,
                    "wavenumber": item.wavenumber,
                    "phase": item.phase,
                }
                for item in request.packet_specs
            ],
            times=request.times,
            num_modes=request.num_modes,
            quadrature_points=request.quadrature_points,
            grid_points=request.grid_points,
            device=request.device,
        )
        if request.output_dir is not None:
            write_packet_sweep_artifacts(request.output_dir, summary)
        return to_serializable(summary)

    @app.post("/profiles/inspect")
    def inspect_table(request: ProfileTablePayload, device: str = "auto"):
        return to_serializable(summarize_profile_table(request.to_table(), device=device))

    @app.post("/features/export")
    def export_features(request: FeatureExportRequest):
        requested_includes = set(request.include)
        summary = export_feature_table_from_profile_table(
            request.table.to_table(),
            num_modes=request.num_modes,
            device=request.device,
            normalize_each_profile=request.normalize_each_profile,
            include_coefficients="coefficients" in requested_includes,
            include_moments="moments" in requested_includes,
            format=request.format,
        )
        if request.output_dir is not None:
            write_feature_table_artifacts(request.output_dir, summary)
            summary = replace(summary, output_path=str(Path(request.output_dir) / f"features.{request.format}"))
        return to_serializable(summary)

    @app.post("/features/export-from-sql")
    def export_features_from_sql(request: DatabaseFeatureExportRequest):
        requested_includes = set(request.include)
        summary = export_feature_table_from_database_query(
            request.database,
            request.query,
            parameters=request.parameters,
            time_column=request.time_column,
            position_columns=request.position_columns,
            sort_by_time=request.sort_by_time,
            num_modes=request.num_modes,
            device=request.device,
            normalize_each_profile=request.normalize_each_profile,
            include_coefficients="coefficients" in requested_includes,
            include_moments="moments" in requested_includes,
            format=request.format,
        )
        if request.output_dir is not None:
            write_feature_table_artifacts(
                request.output_dir,
                summary,
                metadata=database_profile_query_workflow_artifact_metadata(
                    "export-features",
                    request.database,
                    request.query,
                    parameters=request.parameters,
                    time_column=request.time_column,
                    position_columns=request.position_columns,
                    sort_by_time=request.sort_by_time,
                ),
            )
            summary = replace(summary, output_path=str(Path(request.output_dir) / f"features.{request.format}"))
        return to_serializable(summary)

    @app.post("/profiles/report")
    def profile_report(request: ProfileReportRequest):
        report = build_profile_table_report(
            request.table.to_table(),
            analyze_num_modes=request.analyze_num_modes,
            compress_num_modes=request.compress_num_modes,
            device=request.device,
            normalize_each_profile=request.normalize_each_profile,
        )
        if request.output_dir is not None:
            report.write_artifacts(request.output_dir)
        return to_serializable(report)

    @app.post("/profiles/analyze")
    def analyze(request: AnalysisRequest):
        summary = analyze_profile_table_spectra(
            request.table.to_table(),
            num_modes=request.num_modes,
            device=request.device,
            normalize_each_profile=request.normalize_each_profile,
        )
        if request.output_dir is not None:
            write_spectral_analysis_artifacts(request.output_dir, summary)
        return to_serializable(summary)

    @app.post("/profiles/analyze-from-sql")
    def analyze_from_sql(request: DatabaseAnalyzeRequest):
        summary = analyze_profile_table_from_database_query(
            request.database,
            request.query,
            parameters=request.parameters,
            time_column=request.time_column,
            position_columns=request.position_columns,
            sort_by_time=request.sort_by_time,
            num_modes=request.num_modes,
            device=request.device,
            normalize_each_profile=request.normalize_each_profile,
        )
        if request.output_dir is not None:
            write_spectral_analysis_artifacts(
                request.output_dir,
                summary,
                metadata=database_profile_query_workflow_artifact_metadata(
                    "sql-analyze-table",
                    request.database,
                    request.query,
                    parameters=request.parameters,
                    time_column=request.time_column,
                    position_columns=request.position_columns,
                    sort_by_time=request.sort_by_time,
                ),
            )
        return to_serializable(summary)

    @app.post("/profiles/compress-from-sql")
    def compress_from_sql(request: DatabaseCompressionRequest):
        summary = compress_profile_table_from_database_query(
            request.database,
            request.query,
            parameters=request.parameters,
            time_column=request.time_column,
            position_columns=request.position_columns,
            sort_by_time=request.sort_by_time,
            num_modes=request.num_modes,
            device=request.device,
            normalize_each_profile=request.normalize_each_profile,
        )
        if request.output_dir is not None:
            write_compression_artifacts(
                request.output_dir,
                summary,
                metadata=database_profile_query_workflow_artifact_metadata(
                    "sql-compress-table",
                    request.database,
                    request.query,
                    parameters=request.parameters,
                    time_column=request.time_column,
                    position_columns=request.position_columns,
                    sort_by_time=request.sort_by_time,
                ),
            )
        return to_serializable(summary)

    @app.post("/profiles/report-from-sql")
    def profile_report_from_sql(request: DatabaseProfileReportRequest):
        report = build_profile_table_report_from_database_query(
            request.database,
            request.query,
            parameters=request.parameters,
            time_column=request.time_column,
            position_columns=request.position_columns,
            sort_by_time=request.sort_by_time,
            analyze_num_modes=request.analyze_num_modes,
            compress_num_modes=request.compress_num_modes,
            device=request.device,
            normalize_each_profile=request.normalize_each_profile,
        )
        if request.output_dir is not None:
            report.write_artifacts(
                request.output_dir,
                metadata=database_profile_query_workflow_artifact_metadata(
                    "profile-report",
                    request.database,
                    request.query,
                    parameters=request.parameters,
                    time_column=request.time_column,
                    position_columns=request.position_columns,
                    sort_by_time=request.sort_by_time,
                ),
            )
        return to_serializable(report)

    @app.post("/profiles/compress")
    def compress(request: CompressionRequest):
        summary = compress_profile_table(
            request.table.to_table(),
            num_modes=request.num_modes,
            device=request.device,
            normalize_each_profile=request.normalize_each_profile,
        )
        if request.output_dir is not None:
            write_compression_artifacts(request.output_dir, summary)
        return to_serializable(summary)

    @app.post("/profiles/compression-sweep")
    def compression_sweep(request: CompressionSweepRequest):
        summary = sweep_profile_table_compression(
            request.table.to_table(),
            mode_counts=request.mode_counts,
            device=request.device,
            normalize_each_profile=request.normalize_each_profile,
        )
        if request.output_dir is not None:
            write_compression_sweep_artifacts(request.output_dir, summary)
        return to_serializable(summary)

    @app.post("/profiles/compare")
    def compare(request: TableComparisonRequest):
        summary = compare_profile_tables(
            request.reference.to_table(),
            request.candidate.to_table(),
            device=request.device,
        )
        if request.output_dir is not None:
            write_profile_comparison_artifacts(request.output_dir, summary)
        return to_serializable(summary)

    @app.post("/inverse/fit")
    def inverse_fit(request: InverseRequest):
        table = request.table.to_table()
        grid, times, profiles = table.to_torch()
        summary = fit_gaussian_packet_to_density(
            target_density=profiles,
            observation_grid=grid,
            times=times,
            initial_guess={
                "center": request.initial_center,
                "width": request.initial_width,
                "wavenumber": request.initial_wavenumber,
                "phase": request.initial_phase,
            },
            num_modes=request.num_modes,
            quadrature_points=request.quadrature_points,
            device=request.device,
            steps=request.steps,
            learning_rate=request.learning_rate,
        )
        if request.output_dir is not None:
            write_inverse_artifacts(request.output_dir, summary)
        return to_serializable(summary)

    @app.post("/inverse/fit-from-sql")
    def inverse_fit_from_sql(request: DatabaseInverseRequest):
        summary = fit_gaussian_packet_to_profile_table_from_database_query(
            request.database,
            request.query,
            parameters=request.parameters,
            initial_guess={
                "center": request.initial_center,
                "width": request.initial_width,
                "wavenumber": request.initial_wavenumber,
                "phase": request.initial_phase,
            },
            time_column=request.time_column,
            position_columns=request.position_columns,
            sort_by_time=request.sort_by_time,
            num_modes=request.num_modes,
            quadrature_points=request.quadrature_points,
            device=request.device,
            steps=request.steps,
            learning_rate=request.learning_rate,
        )
        if request.output_dir is not None:
            write_inverse_artifacts(
                request.output_dir,
                summary,
                metadata=database_profile_query_workflow_artifact_metadata(
                    "sql-fit-table",
                    request.database,
                    request.query,
                    parameters=request.parameters,
                    time_column=request.time_column,
                    position_columns=request.position_columns,
                    sort_by_time=request.sort_by_time,
                ),
            )
        return to_serializable(summary)

    @app.post("/transport/benchmark")
    def transport(request: TransportRequest):
        summary = benchmark_transport_scan(
            scan_id=request.scan_id,
            mode_counts=request.mode_counts,
            device=request.device,
        )
        if request.output_dir is not None:
            write_transport_benchmark_artifacts(request.output_dir, summary)
        return to_serializable(summary)

    @app.post("/tensorflow/train")
    def tensorflow_train(request: TensorFlowRequest):
        summary = train_tensorflow_surrogate_on_profile_table(
            request.table.to_table(),
            num_modes=request.num_modes,
            normalize_each_profile=request.normalize_each_profile,
            config=TensorFlowRegressorConfig(
                epochs=request.epochs,
                batch_size=request.batch_size,
            ),
            export_dir=request.export_dir,
        )
        if request.output_dir is not None:
            write_tensorflow_training_artifacts(request.output_dir, summary)
        return to_serializable(summary)

    @app.post("/tensorflow/evaluate")
    def tensorflow_evaluate(request: TensorFlowEvaluationRequest):
        summary = evaluate_tensorflow_surrogate_on_profile_table(
            request.table.to_table(),
            num_modes=request.num_modes,
            normalize_each_profile=request.normalize_each_profile,
            config=TensorFlowRegressorConfig(
                epochs=request.epochs,
                batch_size=request.batch_size,
            ),
            export_dir=request.export_dir,
        )
        if request.output_dir is not None:
            write_tensorflow_evaluation_artifacts(request.output_dir, summary)
        return to_serializable(summary)

    @app.post("/ml/train")
    def ml_train(request: ModalTrainingRequest):
        summary = train_modal_surrogate_on_profile_table(
            request.table.to_table(),
            backend=request.backend,
            num_modes=request.num_modes,
            normalize_each_profile=request.normalize_each_profile,
            config=ModalSurrogateConfig(
                epochs=request.epochs,
                batch_size=request.batch_size,
                device=request.device,
            ),
            export_dir=request.export_dir,
        )
        if request.output_dir is not None:
            write_modal_training_artifacts(request.output_dir, summary)
        return to_serializable(summary)

    @app.post("/ml/evaluate")
    def ml_evaluate(request: ModalEvaluationRequest):
        summary = evaluate_modal_surrogate_on_profile_table(
            request.table.to_table(),
            backend=request.backend,
            num_modes=request.num_modes,
            normalize_each_profile=request.normalize_each_profile,
            config=ModalSurrogateConfig(
                epochs=request.epochs,
                batch_size=request.batch_size,
                device=request.device,
            ),
            export_dir=request.export_dir,
        )
        if request.output_dir is not None:
            write_modal_evaluation_artifacts(request.output_dir, summary)
        return to_serializable(summary)

    @app.post("/ml/train-from-sql")
    def ml_train_from_sql(request: DatabaseModalTrainingRequest):
        summary = train_modal_surrogate_from_database_query(
            request.database,
            request.query,
            backend=request.backend,
            parameters=request.parameters,
            time_column=request.time_column,
            position_columns=request.position_columns,
            sort_by_time=request.sort_by_time,
            num_modes=request.num_modes,
            normalize_each_profile=request.normalize_each_profile,
            config=ModalSurrogateConfig(
                epochs=request.epochs,
                batch_size=request.batch_size,
                device=request.device,
            ),
            export_dir=request.export_dir,
        )
        if request.output_dir is not None:
            write_modal_training_artifacts(
                request.output_dir,
                summary,
                metadata=database_profile_query_workflow_artifact_metadata(
                    "sql-ml-train-table",
                    request.database,
                    request.query,
                    parameters=request.parameters,
                    time_column=request.time_column,
                    position_columns=request.position_columns,
                    sort_by_time=request.sort_by_time,
                ),
            )
        return to_serializable(summary)

    @app.post("/ml/evaluate-from-sql")
    def ml_evaluate_from_sql(request: DatabaseModalEvaluationRequest):
        summary = evaluate_modal_surrogate_from_database_query(
            request.database,
            request.query,
            backend=request.backend,
            parameters=request.parameters,
            time_column=request.time_column,
            position_columns=request.position_columns,
            sort_by_time=request.sort_by_time,
            num_modes=request.num_modes,
            normalize_each_profile=request.normalize_each_profile,
            config=ModalSurrogateConfig(
                epochs=request.epochs,
                batch_size=request.batch_size,
                device=request.device,
            ),
            export_dir=request.export_dir,
        )
        if request.output_dir is not None:
            write_modal_evaluation_artifacts(
                request.output_dir,
                summary,
                metadata=database_profile_query_workflow_artifact_metadata(
                    "sql-ml-evaluate-table",
                    request.database,
                    request.query,
                    parameters=request.parameters,
                    time_column=request.time_column,
                    position_columns=request.position_columns,
                    sort_by_time=request.sort_by_time,
                ),
            )
        return to_serializable(summary)

    @app.post("/tree/train")
    def tree_train(request: TreeTrainingRequest):
        export_dir = request.export_dir
        if export_dir is None and request.output_dir is not None:
            export_dir = request.output_dir
        summary = train_tree_model(
            request.dataset.to_dataset(),
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            task=request.task,
            library=request.library,
            model=request.model,
            params=request.params,
            test_fraction=request.test_fraction,
            random_state=request.random_state,
            export_dir=export_dir,
        )
        if request.output_dir is not None:
            write_tree_training_artifacts(request.output_dir, summary)
        return to_serializable(summary)

    @app.post("/tree/tune")
    def tree_tune(request: TreeTuningRequest):
        export_dir = request.export_dir
        if export_dir is None and request.output_dir is not None:
            export_dir = str(Path(request.output_dir) / "best_model")
        summary = tune_tree_model(
            request.dataset.to_dataset(),
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            task=request.task,
            library=request.library,
            model=request.model,
            search_space=request.search_space,
            search_kind=request.search_kind,
            n_iter=request.n_iter,
            cv=request.cv,
            scoring=request.scoring,
            test_fraction=request.test_fraction,
            random_state=request.random_state,
            export_dir=export_dir,
        )
        if request.output_dir is not None:
            write_tree_tuning_artifacts(request.output_dir, summary)
        return to_serializable(summary)

    return app


__all__ = [
    "api_is_available",
    "create_api_app",
]
