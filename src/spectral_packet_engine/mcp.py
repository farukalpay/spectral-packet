from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from spectral_packet_engine.artifacts import (
    list_artifact_files,
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
from spectral_packet_engine.ml import ModalSurrogateConfig
from spectral_packet_engine.tabular import load_tabular_dataset, supported_tabular_formats
from spectral_packet_engine.table_io import load_profile_table, supported_profile_table_formats
from spectral_packet_engine.tf_surrogate import TensorFlowRegressorConfig
from spectral_packet_engine.workflows import (
    analyze_profile_table_spectra,
    analyze_profile_table_from_database_query,
    benchmark_transport_scan,
    bootstrap_local_database,
    compare_profile_tables,
    compress_profile_table,
    describe_database_table,
    evaluate_modal_surrogate_from_database_query,
    evaluate_modal_surrogate_on_profile_table,
    evaluate_tensorflow_surrogate_on_profile_table,
    fit_gaussian_packet_to_profile_table,
    inspect_database,
    inspect_environment,
    inspect_ml_backend_support,
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


def _tool(server, name: str, description: str):
    return server.tool(
        name=name,
        description=description,
        structured_output=True,
    )


def create_mcp_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("The MCP server requires the 'mcp' extra.") from exc

    server = FastMCP(
        "Spectral Packet Engine",
        instructions=(
            "Use this server for bounded-domain modal simulation, packet projection, "
            "spectral table analysis, profile compression, table comparison, inverse reconstruction, "
            "database-backed tabular workflows, transport benchmarks, and backend-aware modal surrogate workflows."
        ),
    )

    @_tool(server, "inspect_environment", "Inspect machine capabilities, optional surfaces, and runtime backends.")
    def inspect_environment_tool(device: str = "auto") -> dict[str, Any]:
        return to_serializable(inspect_environment(device))

    @_tool(server, "validate_installation", "Validate the current installation and report stable, beta, and experimental surfaces.")
    def validate_installation_tool(device: str = "auto") -> dict[str, Any]:
        return to_serializable(validate_installation(device))

    @_tool(server, "inspect_ml_backends", "Inspect available modal-surrogate backends and their runtime capabilities.")
    def inspect_ml_backends_tool(device: str = "auto") -> dict[str, Any]:
        return to_serializable(inspect_ml_backend_support(device))

    @_tool(server, "supported_profile_formats", "List supported file formats for profile-table workflows.")
    def supported_profile_formats_tool() -> dict[str, bool]:
        return supported_profile_table_formats()

    @_tool(server, "supported_tabular_formats", "List supported file formats for generic tabular datasets.")
    def supported_tabular_formats_tool() -> dict[str, bool]:
        return supported_tabular_formats()

    @_tool(server, "simulate_packet", "Run a bounded-domain Gaussian packet simulation and optionally write artifacts.")
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

    @_tool(server, "project_packet", "Project a Gaussian packet into the bounded-domain modal basis.")
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

    @_tool(server, "inspect_profile_table", "Inspect a profile table before spectral analysis or reconstruction.")
    def inspect_profile_table_tool(table_path: str, device: str = "auto") -> dict[str, Any]:
        return to_serializable(summarize_profile_table(load_profile_table(table_path), device=device))

    @_tool(server, "inspect_tabular_dataset", "Inspect a generic tabular dataset and report schema, validation, and preview rows.")
    def inspect_tabular_dataset_tool(dataset_path: str) -> dict[str, Any]:
        return to_serializable(summarize_tabular_dataset(load_tabular_dataset(dataset_path)))

    @_tool(server, "analyze_profile_table", "Analyze the modal structure of a profile table and optionally write spectral artifacts.")
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

    @_tool(server, "compress_profile_table", "Compress a profile table into modal coefficients and reconstruction artifacts.")
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

    @_tool(server, "compression_sweep", "Evaluate compression quality over multiple modal truncation levels.")
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

    @_tool(server, "fit_packet_to_profile_table", "Fit Gaussian packet parameters to an observed profile table.")
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

    @_tool(server, "compare_profile_tables", "Compare candidate and reference profile tables with domain-aware error metrics.")
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

    @_tool(server, "inspect_database", "Inspect a database reference and list available tables and capabilities.")
    def inspect_database_tool(database: str) -> dict[str, Any]:
        return to_serializable(inspect_database(database))

    @_tool(server, "bootstrap_database", "Create or open a local SQLite database path and report its capabilities.")
    def bootstrap_database_tool(database: str) -> dict[str, Any]:
        return to_serializable(bootstrap_local_database(database))

    @_tool(server, "describe_database_table", "Describe a database table schema and row count.")
    def describe_database_table_tool(database: str, table_name: str) -> dict[str, Any]:
        return to_serializable(describe_database_table(database, table_name))

    @_tool(server, "query_database", "Run a parameterized SQL query and return a tabular summary of the result.")
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
                metadata={"workflow": "db-query"},
            )
        return to_serializable(
            summarize_database_query_result(
                database,
                query,
                result,
                parameters=_coerce_parameters(parameters),
            )
        )

    @_tool(server, "write_database_table", "Load a tabular dataset from disk and write it into a database table.")
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

    @_tool(server, "materialize_query_table", "Run a query and persist its result as a managed database table.")
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

    @_tool(server, "analyze_database_profile_query", "Run a profile-table-shaped SQL query and analyze its modal structure.")
    def analyze_database_profile_query_tool(
        database: str,
        query: str,
        num_modes: int = 32,
        device: str = "auto",
        time_column: str = "time",
        normalize_each_profile: bool = False,
        parameters: dict[str, Any] | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = analyze_profile_table_from_database_query(
            database,
            query,
            parameters=_coerce_parameters(parameters),
            time_column=time_column,
            num_modes=num_modes,
            device=device,
            normalize_each_profile=normalize_each_profile,
        )
        if output_dir is not None:
            write_spectral_analysis_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, "packet_sweep", "Run a batch of Gaussian packet simulations with shared settings.")
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

    @_tool(server, "benchmark_transport", "Benchmark the published transport dataset against modal compression settings.")
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

    @_tool(server, "train_tensorflow_surrogate", "Train the TensorFlow compatibility surrogate on a profile table.")
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

    @_tool(server, "evaluate_tensorflow_surrogate", "Train and evaluate the TensorFlow compatibility surrogate on a profile table.")
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

    @_tool(server, "train_modal_surrogate", "Train a backend-aware modal surrogate on a profile table.")
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

    @_tool(server, "evaluate_modal_surrogate", "Train and evaluate a backend-aware modal surrogate on a profile table.")
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

    @_tool(server, "train_modal_surrogate_from_sql", "Materialize a profile-table-shaped SQL query and train a backend-aware modal surrogate on it.")
    def train_modal_surrogate_from_sql_tool(
        database: str,
        query: str,
        backend: str = "auto",
        num_modes: int = 16,
        epochs: int = 20,
        batch_size: int = 64,
        time_column: str = "time",
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

    @_tool(server, "evaluate_modal_surrogate_from_sql", "Materialize a profile-table-shaped SQL query and evaluate a backend-aware modal surrogate on it.")
    def evaluate_modal_surrogate_from_sql_tool(
        database: str,
        query: str,
        backend: str = "auto",
        num_modes: int = 16,
        epochs: int = 20,
        batch_size: int = 64,
        time_column: str = "time",
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

    @_tool(server, "list_artifacts", "List files from a workflow artifact directory.")
    def list_artifacts_tool(output_dir: str) -> dict[str, Any]:
        directory = Path(output_dir)
        return {
            "output_dir": str(directory),
            "exists": directory.exists(),
            "files": list_artifact_files(directory),
        }

    return server


def main() -> None:
    create_mcp_server().run()


__all__ = [
    "create_mcp_server",
    "main",
    "mcp_is_available",
]
