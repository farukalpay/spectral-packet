from __future__ import annotations

import importlib.util
from functools import wraps
from threading import BoundedSemaphore
from time import perf_counter
from typing import Any

from spectral_packet_engine.artifacts import (
    inspect_artifact_directory,
    to_serializable,
    write_compression_artifacts,
    write_compression_sweep_artifacts,
    write_forward_artifacts,
    write_inverse_artifacts,
    write_modal_evaluation_artifacts,
    write_modal_training_artifacts,
    write_packet_sweep_artifacts,
    write_profile_comparison_artifacts,
    write_spectral_analysis_artifacts,
    write_tabular_artifacts,
    write_tensorflow_evaluation_artifacts,
    write_tensorflow_training_artifacts,
    write_transport_benchmark_artifacts,
)
from spectral_packet_engine.mcp_runtime import MCPServerConfig, inspect_mcp_runtime
from spectral_packet_engine.ml import ModalSurrogateConfig
from spectral_packet_engine.product import (
    DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES,
    DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES,
    PRODUCT_SPINE_STATEMENT,
    RUNTIME_SPINE_STATEMENT,
    guide_workflow,
    inspect_product_identity,
    resolve_workflow_identity,
)
from spectral_packet_engine.service_status import (
    configure_service_logging,
    inspect_service_status,
    track_service_task,
)
from spectral_packet_engine.tabular import load_tabular_dataset, supported_tabular_formats
from spectral_packet_engine.table_io import load_profile_table, supported_profile_table_formats
from spectral_packet_engine.tf_surrogate import TensorFlowRegressorConfig
from spectral_packet_engine.workflows import (
    analyze_profile_table_spectra,
    analyze_profile_table_from_database_query,
    benchmark_transport_scan,
    build_profile_table_report_from_database_query,
    bootstrap_local_database,
    compare_profile_tables,
    compress_profile_table,
    compress_profile_table_from_database_query,
    database_profile_query_artifact_metadata,
    database_query_artifact_metadata,
    describe_database_table,
    evaluate_modal_surrogate_from_database_query,
    evaluate_modal_surrogate_on_profile_table,
    evaluate_tensorflow_surrogate_on_profile_table,
    fit_gaussian_packet_to_profile_table,
    fit_gaussian_packet_to_profile_table_from_database_query,
    inspect_database,
    inspect_environment,
    inspect_ml_backend_support,
    load_profile_table_report,
    materialize_database_query,
    materialize_database_query_to_table,
    simulate_packet_sweep,
    project_gaussian_packet,
    simulate_gaussian_packet,
    summarize_database_query_result,
    summarize_profile_table,
    summarize_tabular_dataset,
    sweep_profile_table_compression,
    train_modal_surrogate_from_database_query,
    train_modal_surrogate_on_profile_table,
    train_tensorflow_surrogate_on_profile_table,
    validate_installation,
    write_tabular_dataset_to_database,
)


def mcp_is_available() -> bool:
    return importlib.util.find_spec("mcp.server.fastmcp") is not None


def _coerce_parameters(parameters: dict[str, Any] | None) -> dict[str, Any]:
    return {} if parameters is None else {str(key): value for key, value in parameters.items()}


class _MCPExecutionController:
    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._semaphore = BoundedSemaphore(config.max_concurrent_tasks)

    def acquire(self) -> dict[str, float | int]:
        started = perf_counter()
        acquired = self._semaphore.acquire(timeout=self.config.slot_acquire_timeout_seconds)
        waited = float(perf_counter() - started)
        if not acquired:
            raise RuntimeError(
                "MCP runtime is saturated; no execution slot became available before the configured timeout. "
                "Retry later or increase the max_concurrent_tasks setting."
            )
        return {
            "queued_seconds": waited,
            "max_concurrent_tasks": self.config.max_concurrent_tasks,
            "slot_acquire_timeout_seconds": self.config.slot_acquire_timeout_seconds,
        }

    def release(self) -> None:
        self._semaphore.release()


def _tool(server, runtime: _MCPExecutionController, name: str, description: str, *, bounded: bool = False):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            workflow = resolve_workflow_identity("mcp", name)
            metadata = {
                "tool": name,
                "bounded_execution": bounded,
                "max_concurrent_tasks": runtime.config.max_concurrent_tasks,
            }
            with track_service_task(
                name if workflow is None else workflow.workflow_id,
                interface="mcp",
                workflow_id=None if workflow is None else workflow.workflow_id,
                surface_action=name,
                metadata=metadata,
            ):
                if not bounded:
                    return function(*args, **kwargs)
                runtime.acquire()
                try:
                    return function(*args, **kwargs)
                finally:
                    runtime.release()

        return server.tool(
            name=name,
            description=description,
            structured_output=True,
        )(wrapped)

    return decorator


def create_mcp_server(config: MCPServerConfig | None = None):
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("The MCP server requires the 'mcp' extra.") from exc

    runtime_config = config or MCPServerConfig()
    configure_service_logging(
        runtime_config.log_level,
        log_file=runtime_config.log_file,
    )
    runtime = _MCPExecutionController(runtime_config)

    server = FastMCP(
        "Spectral Packet Engine",
        instructions=(
            f"{PRODUCT_SPINE_STATEMENT} "
            f"Operationally: {RUNTIME_SPINE_STATEMENT}"
        ),
    )

    @_tool(server, runtime, "inspect_product", "Inspect the shared product spine, runtime model, and canonical workflow map.")
    def inspect_product_tool() -> dict[str, Any]:
        return to_serializable(inspect_product_identity())

    @_tool(server, runtime, "guide_workflow", "Recommend the default high-value workflow and defaults for a file-backed or SQL-backed session.")
    def guide_workflow_tool(input_kind: str = "profile-table-file") -> dict[str, Any]:
        return to_serializable(guide_workflow(surface="mcp", input_kind=input_kind))

    @_tool(server, runtime, "inspect_environment", "Inspect machine capabilities, optional surfaces, and runtime backends.")
    def inspect_environment_tool(device: str = "auto") -> dict[str, Any]:
        return to_serializable(inspect_environment(device))

    @_tool(server, runtime, "inspect_mcp_runtime", "Inspect MCP transport, bounded-execution policy, platform notes, and logging configuration.")
    def inspect_mcp_runtime_tool() -> dict[str, Any]:
        return to_serializable(inspect_mcp_runtime(runtime_config))

    @_tool(server, runtime, "inspect_service_status", "Inspect service uptime, task counters, and recent execution history.")
    def inspect_service_status_tool() -> dict[str, Any]:
        return to_serializable(inspect_service_status())

    @_tool(server, runtime, "validate_installation", "Validate the current installation and report stable, beta, and experimental surfaces.")
    def validate_installation_tool(device: str = "auto") -> dict[str, Any]:
        return to_serializable(validate_installation(device))

    @_tool(server, runtime, "inspect_ml_backends", "Inspect available modal-surrogate backends and their runtime capabilities.")
    def inspect_ml_backends_tool(device: str = "auto") -> dict[str, Any]:
        return to_serializable(inspect_ml_backend_support(device))

    @_tool(server, runtime, "supported_profile_formats", "List supported file formats for profile-table workflows.")
    def supported_profile_formats_tool() -> dict[str, bool]:
        return supported_profile_table_formats()

    @_tool(server, runtime, "supported_tabular_formats", "List supported file formats for generic tabular datasets.")
    def supported_tabular_formats_tool() -> dict[str, bool]:
        return supported_tabular_formats()

    @_tool(server, runtime, "simulate_packet", "Run a bounded-domain Gaussian packet simulation and optionally write artifacts.", bounded=True)
    def simulate_packet_tool(
        center: float = 0.30,
        width: float = 0.07,
        wavenumber: float = 25.0,
        phase: float = 0.0,
        times: list[float] | None = None,
        num_modes: int = 128,
        quadrature_points: int = 4096,
        grid_points: int = 512,
        device: str = "auto",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = simulate_gaussian_packet(
            center=center,
            width=width,
            wavenumber=wavenumber,
            phase=phase,
            times=times or [0.0, 1e-3, 5e-3],
            num_modes=num_modes,
            quadrature_points=quadrature_points,
            grid_points=grid_points,
            device=device,
        )
        if output_dir is not None:
            write_forward_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "project_packet", "Project a Gaussian packet into the bounded-domain modal basis.", bounded=True)
    def project_packet_tool(
        center: float = 0.30,
        width: float = 0.07,
        wavenumber: float = 25.0,
        phase: float = 0.0,
        num_modes: int = 128,
        quadrature_points: int = 4096,
        grid_points: int = 2048,
        device: str = "auto",
    ) -> dict[str, Any]:
        return to_serializable(
            project_gaussian_packet(
                center=center,
                width=width,
                wavenumber=wavenumber,
                phase=phase,
                num_modes=num_modes,
                quadrature_points=quadrature_points,
                grid_points=grid_points,
                device=device,
            )
        )

    @_tool(server, runtime, "inspect_profile_table", "Inspect a profile table before spectral analysis or reconstruction.")
    def inspect_profile_table_tool(table_path: str, device: str = "auto") -> dict[str, Any]:
        return to_serializable(summarize_profile_table(load_profile_table(table_path), device=device))

    @_tool(server, runtime, "profile_table_report", "Inspect, analyze, and compress a profile table into one reference-grade report.", bounded=True)
    def profile_table_report_tool(
        table_path: str,
        analyze_num_modes: int = DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES,
        compress_num_modes: int = DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES,
        device: str = "auto",
        normalize_each_profile: bool = False,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        report = load_profile_table_report(
            table_path,
            analyze_num_modes=analyze_num_modes,
            compress_num_modes=compress_num_modes,
            device=device,
            normalize_each_profile=normalize_each_profile,
        )
        if output_dir is not None:
            report.write_artifacts(
                output_dir,
                metadata={"input": {"table_path": table_path}},
            )
        return to_serializable(report)

    @_tool(server, runtime, "inspect_tabular_dataset", "Inspect a generic tabular dataset and report schema, validation, and preview rows.")
    def inspect_tabular_dataset_tool(dataset_path: str) -> dict[str, Any]:
        return to_serializable(summarize_tabular_dataset(load_tabular_dataset(dataset_path)))

    @_tool(server, runtime, "analyze_profile_table", "Analyze the modal structure of a profile table and optionally write spectral artifacts.", bounded=True)
    def analyze_profile_table_tool(
        table_path: str,
        num_modes: int = 32,
        device: str = "auto",
        normalize_each_profile: bool = False,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = analyze_profile_table_spectra(
            load_profile_table(table_path),
            num_modes=num_modes,
            device=device,
            normalize_each_profile=normalize_each_profile,
        )
        if output_dir is not None:
            write_spectral_analysis_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "compress_profile_table", "Compress a profile table into modal coefficients and reconstruction artifacts.", bounded=True)
    def compress_profile_table_tool(
        table_path: str,
        num_modes: int = 32,
        device: str = "auto",
        normalize_each_profile: bool = False,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = compress_profile_table(
            load_profile_table(table_path),
            num_modes=num_modes,
            device=device,
            normalize_each_profile=normalize_each_profile,
        )
        if output_dir is not None:
            write_compression_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "compression_sweep", "Evaluate compression quality over multiple modal truncation levels.", bounded=True)
    def compression_sweep_tool(
        table_path: str,
        mode_counts: list[int] | None = None,
        device: str = "auto",
        normalize_each_profile: bool = False,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = sweep_profile_table_compression(
            load_profile_table(table_path),
            mode_counts=mode_counts or [4, 8, 16, 32],
            device=device,
            normalize_each_profile=normalize_each_profile,
        )
        if output_dir is not None:
            write_compression_sweep_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "fit_packet_to_profile_table", "Fit Gaussian packet parameters to an observed profile table.", bounded=True)
    def fit_packet_to_profile_table_tool(
        table_path: str,
        center: float = 0.36,
        width: float = 0.11,
        wavenumber: float = 22.0,
        phase: float = 0.0,
        num_modes: int = 128,
        quadrature_points: int = 2048,
        steps: int = 200,
        learning_rate: float = 0.05,
        device: str = "auto",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = fit_gaussian_packet_to_profile_table(
            load_profile_table(table_path),
            initial_guess={
                "center": center,
                "width": width,
                "wavenumber": wavenumber,
                "phase": phase,
            },
            num_modes=num_modes,
            quadrature_points=quadrature_points,
            steps=steps,
            learning_rate=learning_rate,
            device=device,
        )
        if output_dir is not None:
            write_inverse_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "compare_profile_tables", "Compare candidate and reference profile tables with domain-aware error metrics.", bounded=True)
    def compare_profile_tables_tool(
        reference_table_path: str,
        candidate_table_path: str,
        device: str = "auto",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = compare_profile_tables(
            load_profile_table(reference_table_path),
            load_profile_table(candidate_table_path),
            device=device,
        )
        if output_dir is not None:
            write_profile_comparison_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "inspect_database", "Inspect a database reference and list available tables and capabilities.")
    def inspect_database_tool(database: str) -> dict[str, Any]:
        return to_serializable(inspect_database(database))

    @_tool(server, runtime, "bootstrap_database", "Create or open a local SQLite database path and report its capabilities.")
    def bootstrap_database_tool(database: str) -> dict[str, Any]:
        return to_serializable(bootstrap_local_database(database))

    @_tool(server, runtime, "describe_database_table", "Describe a database table schema and row count.")
    def describe_database_table_tool(database: str, table_name: str) -> dict[str, Any]:
        return to_serializable(describe_database_table(database, table_name))

    @_tool(server, runtime, "query_database", "Run a parameterized SQL query and return a tabular summary of the result.", bounded=True)
    def query_database_tool(
        database: str,
        query: str,
        parameters: dict[str, Any] | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        result = materialize_database_query(database, query, parameters=_coerce_parameters(parameters))
        if output_dir is not None:
            write_tabular_artifacts(
                output_dir,
                result.dataset,
                summary_name="db_query_summary.json",
                table_name="query_result.csv",
                metadata={
                    "workflow": "db-query",
                    **database_query_artifact_metadata(
                        database,
                        query,
                        parameters=_coerce_parameters(parameters),
                    ),
                },
            )
        return to_serializable(
            summarize_database_query_result(
                database,
                query,
                result,
                parameters=_coerce_parameters(parameters),
            )
        )

    @_tool(server, runtime, "write_database_table", "Load a tabular dataset from disk and write it into a database table.", bounded=True)
    def write_database_table_tool(
        database: str,
        table_name: str,
        dataset_path: str,
        if_exists: str = "fail",
    ) -> dict[str, Any]:
        return to_serializable(
            write_tabular_dataset_to_database(
                database,
                table_name,
                load_tabular_dataset(dataset_path),
                if_exists=if_exists,
            )
        )

    @_tool(server, runtime, "materialize_query_table", "Run a query and persist its result as a managed database table.", bounded=True)
    def materialize_query_table_tool(
        database: str,
        table_name: str,
        query: str,
        parameters: dict[str, Any] | None = None,
        replace: bool = False,
    ) -> dict[str, Any]:
        return to_serializable(
            materialize_database_query_to_table(
                database,
                table_name,
                query,
                parameters=_coerce_parameters(parameters),
                replace=replace,
            )
        )

    @_tool(server, runtime, "analyze_database_profile_query", "Run a profile-table-shaped SQL query and analyze its modal structure.", bounded=True)
    def analyze_database_profile_query_tool(
        database: str,
        query: str,
        num_modes: int = 32,
        device: str = "auto",
        time_column: str = "time",
        position_columns: list[str] | None = None,
        sort_by_time: bool = False,
        normalize_each_profile: bool = False,
        parameters: dict[str, Any] | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = analyze_profile_table_from_database_query(
            database,
            query,
            parameters=_coerce_parameters(parameters),
            time_column=time_column,
            position_columns=position_columns,
            sort_by_time=sort_by_time,
            num_modes=num_modes,
            device=device,
            normalize_each_profile=normalize_each_profile,
        )
        if output_dir is not None:
            write_spectral_analysis_artifacts(
                output_dir,
                summary,
                metadata={
                    "workflow": "sql-analyze-table",
                    **database_profile_query_artifact_metadata(
                        database,
                        query,
                        parameters=_coerce_parameters(parameters),
                        time_column=time_column,
                        position_columns=position_columns,
                        sort_by_time=sort_by_time,
                    ),
                },
            )
        return to_serializable(summary)

    @_tool(server, runtime, "compress_database_profile_query", "Run a profile-table-shaped SQL query and compress it into modal coefficients and reconstructions.", bounded=True)
    def compress_database_profile_query_tool(
        database: str,
        query: str,
        num_modes: int = 32,
        device: str = "auto",
        time_column: str = "time",
        position_columns: list[str] | None = None,
        sort_by_time: bool = False,
        normalize_each_profile: bool = False,
        parameters: dict[str, Any] | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = compress_profile_table_from_database_query(
            database,
            query,
            parameters=_coerce_parameters(parameters),
            time_column=time_column,
            position_columns=position_columns,
            sort_by_time=sort_by_time,
            num_modes=num_modes,
            device=device,
            normalize_each_profile=normalize_each_profile,
        )
        if output_dir is not None:
            write_compression_artifacts(
                output_dir,
                summary,
                metadata={
                    "workflow": "sql-compress-table",
                    **database_profile_query_artifact_metadata(
                        database,
                        query,
                        parameters=_coerce_parameters(parameters),
                        time_column=time_column,
                        position_columns=position_columns,
                        sort_by_time=sort_by_time,
                    ),
                },
        )
        return to_serializable(summary)

    @_tool(server, runtime, "report_database_profile_query", "Run a profile-table-shaped SQL query through the shared inspect-analyze-compress report workflow.", bounded=True)
    def report_database_profile_query_tool(
        database: str,
        query: str,
        analyze_num_modes: int = DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES,
        compress_num_modes: int = DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES,
        device: str = "auto",
        time_column: str = "time",
        position_columns: list[str] | None = None,
        sort_by_time: bool = False,
        normalize_each_profile: bool = False,
        parameters: dict[str, Any] | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        report = build_profile_table_report_from_database_query(
            database,
            query,
            parameters=_coerce_parameters(parameters),
            time_column=time_column,
            position_columns=position_columns,
            sort_by_time=sort_by_time,
            analyze_num_modes=analyze_num_modes,
            compress_num_modes=compress_num_modes,
            device=device,
            normalize_each_profile=normalize_each_profile,
        )
        if output_dir is not None:
            report.write_artifacts(
                output_dir,
                metadata=database_profile_query_artifact_metadata(
                    database,
                    query,
                    parameters=_coerce_parameters(parameters),
                    time_column=time_column,
                    position_columns=position_columns,
                    sort_by_time=sort_by_time,
                ),
            )
        return to_serializable(report)

    @_tool(server, runtime, "fit_packet_to_database_profile_query", "Run a profile-table-shaped SQL query and fit Gaussian packet parameters to it.", bounded=True)
    def fit_packet_to_database_profile_query_tool(
        database: str,
        query: str,
        center: float = 0.36,
        width: float = 0.11,
        wavenumber: float = 22.0,
        phase: float = 0.0,
        num_modes: int = 128,
        quadrature_points: int = 2048,
        steps: int = 200,
        learning_rate: float = 0.05,
        device: str = "auto",
        time_column: str = "time",
        position_columns: list[str] | None = None,
        sort_by_time: bool = False,
        parameters: dict[str, Any] | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = fit_gaussian_packet_to_profile_table_from_database_query(
            database,
            query,
            parameters=_coerce_parameters(parameters),
            initial_guess={
                "center": center,
                "width": width,
                "wavenumber": wavenumber,
                "phase": phase,
            },
            time_column=time_column,
            position_columns=position_columns,
            sort_by_time=sort_by_time,
            num_modes=num_modes,
            quadrature_points=quadrature_points,
            steps=steps,
            learning_rate=learning_rate,
            device=device,
        )
        if output_dir is not None:
            write_inverse_artifacts(
                output_dir,
                summary,
                metadata={
                    "workflow": "sql-fit-table",
                    **database_profile_query_artifact_metadata(
                        database,
                        query,
                        parameters=_coerce_parameters(parameters),
                        time_column=time_column,
                        position_columns=position_columns,
                        sort_by_time=sort_by_time,
                    ),
                },
            )
        return to_serializable(summary)

    @_tool(server, runtime, "packet_sweep", "Run a batch of Gaussian packet simulations with shared settings.", bounded=True)
    def packet_sweep_tool(
        packet_specs: list[dict[str, float]],
        times: list[float] | None = None,
        num_modes: int = 128,
        quadrature_points: int = 4096,
        grid_points: int = 512,
        device: str = "auto",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = simulate_packet_sweep(
            packet_specs,
            times=times or [0.0, 1e-3, 5e-3],
            num_modes=num_modes,
            quadrature_points=quadrature_points,
            grid_points=grid_points,
            device=device,
        )
        if output_dir is not None:
            write_packet_sweep_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "benchmark_transport", "Benchmark the published transport dataset against modal compression settings.", bounded=True)
    def benchmark_transport_tool(
        scan_id: str = "scan11879_56",
        mode_counts: list[int] | None = None,
        device: str = "auto",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = benchmark_transport_scan(
            scan_id=scan_id,
            mode_counts=mode_counts or [8, 16, 32, 64],
            device=device,
        )
        if output_dir is not None:
            write_transport_benchmark_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "train_tensorflow_surrogate", "Train the TensorFlow compatibility surrogate on a profile table.", bounded=True)
    def train_tensorflow_surrogate_tool(
        table_path: str,
        num_modes: int = 16,
        epochs: int = 20,
        batch_size: int = 64,
        normalize_each_profile: bool = False,
        export_dir: str | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = train_tensorflow_surrogate_on_profile_table(
            load_profile_table(table_path),
            num_modes=num_modes,
            normalize_each_profile=normalize_each_profile,
            config=TensorFlowRegressorConfig(
                epochs=epochs,
                batch_size=batch_size,
            ),
            export_dir=export_dir,
        )
        if output_dir is not None:
            write_tensorflow_training_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "evaluate_tensorflow_surrogate", "Train and evaluate the TensorFlow compatibility surrogate on a profile table.", bounded=True)
    def evaluate_tensorflow_surrogate_tool(
        table_path: str,
        num_modes: int = 16,
        epochs: int = 20,
        batch_size: int = 64,
        normalize_each_profile: bool = False,
        export_dir: str | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = evaluate_tensorflow_surrogate_on_profile_table(
            load_profile_table(table_path),
            num_modes=num_modes,
            normalize_each_profile=normalize_each_profile,
            config=TensorFlowRegressorConfig(
                epochs=epochs,
                batch_size=batch_size,
            ),
            export_dir=export_dir,
        )
        if output_dir is not None:
            write_tensorflow_evaluation_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "train_modal_surrogate", "Train a backend-aware modal surrogate on a profile table.", bounded=True)
    def train_modal_surrogate_tool(
        table_path: str,
        backend: str = "auto",
        num_modes: int = 16,
        epochs: int = 20,
        batch_size: int = 64,
        normalize_each_profile: bool = False,
        export_dir: str | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = train_modal_surrogate_on_profile_table(
            load_profile_table(table_path),
            backend=backend,
            num_modes=num_modes,
            normalize_each_profile=normalize_each_profile,
            config=ModalSurrogateConfig(
                epochs=epochs,
                batch_size=batch_size,
            ),
            export_dir=export_dir,
        )
        if output_dir is not None:
            write_modal_training_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "evaluate_modal_surrogate", "Train and evaluate a backend-aware modal surrogate on a profile table.", bounded=True)
    def evaluate_modal_surrogate_tool(
        table_path: str,
        backend: str = "auto",
        num_modes: int = 16,
        epochs: int = 20,
        batch_size: int = 64,
        normalize_each_profile: bool = False,
        export_dir: str | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = evaluate_modal_surrogate_on_profile_table(
            load_profile_table(table_path),
            backend=backend,
            num_modes=num_modes,
            normalize_each_profile=normalize_each_profile,
            config=ModalSurrogateConfig(
                epochs=epochs,
                batch_size=batch_size,
            ),
            export_dir=export_dir,
        )
        if output_dir is not None:
            write_modal_evaluation_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "train_modal_surrogate_from_sql", "Materialize a profile-table-shaped SQL query and train a backend-aware modal surrogate on it.", bounded=True)
    def train_modal_surrogate_from_sql_tool(
        database: str,
        query: str,
        backend: str = "auto",
        num_modes: int = 16,
        epochs: int = 20,
        batch_size: int = 64,
        time_column: str = "time",
        position_columns: list[str] | None = None,
        sort_by_time: bool = False,
        normalize_each_profile: bool = False,
        parameters: dict[str, Any] | None = None,
        export_dir: str | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = train_modal_surrogate_from_database_query(
            database,
            query,
            backend=backend,
            parameters=_coerce_parameters(parameters),
            time_column=time_column,
            position_columns=position_columns,
            sort_by_time=sort_by_time,
            num_modes=num_modes,
            normalize_each_profile=normalize_each_profile,
            config=ModalSurrogateConfig(
                epochs=epochs,
                batch_size=batch_size,
            ),
            export_dir=export_dir,
        )
        if output_dir is not None:
            write_modal_training_artifacts(
                output_dir,
                summary,
                metadata={
                    "workflow": "sql-ml-train-table",
                    **database_profile_query_artifact_metadata(
                        database,
                        query,
                        parameters=_coerce_parameters(parameters),
                        time_column=time_column,
                        position_columns=position_columns,
                        sort_by_time=sort_by_time,
                    ),
                },
            )
        return to_serializable(summary)

    @_tool(server, runtime, "evaluate_modal_surrogate_from_sql", "Materialize a profile-table-shaped SQL query and evaluate a backend-aware modal surrogate on it.", bounded=True)
    def evaluate_modal_surrogate_from_sql_tool(
        database: str,
        query: str,
        backend: str = "auto",
        num_modes: int = 16,
        epochs: int = 20,
        batch_size: int = 64,
        time_column: str = "time",
        position_columns: list[str] | None = None,
        sort_by_time: bool = False,
        normalize_each_profile: bool = False,
        parameters: dict[str, Any] | None = None,
        export_dir: str | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = evaluate_modal_surrogate_from_database_query(
            database,
            query,
            backend=backend,
            parameters=_coerce_parameters(parameters),
            time_column=time_column,
            position_columns=position_columns,
            sort_by_time=sort_by_time,
            num_modes=num_modes,
            normalize_each_profile=normalize_each_profile,
            config=ModalSurrogateConfig(
                epochs=epochs,
                batch_size=batch_size,
            ),
            export_dir=export_dir,
        )
        if output_dir is not None:
            write_modal_evaluation_artifacts(
                output_dir,
                summary,
                metadata={
                    "workflow": "sql-ml-evaluate-table",
                    **database_profile_query_artifact_metadata(
                        database,
                        query,
                        parameters=_coerce_parameters(parameters),
                        time_column=time_column,
                        position_columns=position_columns,
                        sort_by_time=sort_by_time,
                    ),
                },
            )
        return to_serializable(summary)

    @_tool(server, runtime, "list_artifacts", "Inspect an artifact directory and report completion state, metadata, and files.")
    def list_artifacts_tool(output_dir: str) -> dict[str, Any]:
        return to_serializable(inspect_artifact_directory(output_dir))

    return server


def main(config: MCPServerConfig | None = None) -> None:
    runtime_config = config or MCPServerConfig()
    configure_service_logging(
        runtime_config.log_level,
        log_file=runtime_config.log_file,
        force=True,
    )
    create_mcp_server(runtime_config).run()


__all__ = [
    "MCPServerConfig",
    "create_mcp_server",
    "main",
    "mcp_is_available",
]
