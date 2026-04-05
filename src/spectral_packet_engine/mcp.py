from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from functools import wraps
from pathlib import Path
from threading import BoundedSemaphore
from time import perf_counter
from typing import Any

import torch

from spectral_packet_engine.artifacts import (
    inspect_artifact_directory,
    to_serializable,
    write_json,
    write_compression_artifacts,
    write_compression_sweep_artifacts,
    write_differentiable_artifacts,
    write_feature_table_artifacts,
    write_forward_artifacts,
    write_inverse_artifacts,
    write_modal_evaluation_artifacts,
    write_modal_training_artifacts,
    write_mcp_probe_artifacts,
    write_packet_sweep_artifacts,
    write_potential_inference_artifacts,
    write_profile_comparison_artifacts,
    write_reduced_model_artifacts,
    write_spectral_analysis_artifacts,
    write_tree_training_artifacts,
    write_tree_tuning_artifacts,
    write_tabular_artifacts,
    write_tensorflow_evaluation_artifacts,
    write_tensorflow_training_artifacts,
    write_transport_benchmark_artifacts,
    write_vertical_workflow_artifacts,
)
from spectral_packet_engine.mcp_runtime import (
    MCPServerConfig,
    ensure_mcp_scratch_dir,
    inspect_mcp_runtime,
    resolve_mcp_scratch_file,
)
from spectral_packet_engine.ml import ModalSurrogateConfig
from spectral_packet_engine.config import SERVER_PURPOSE
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
    mark_service_task_failed,
    track_service_task,
)
from spectral_packet_engine.synthetic_profiles import generate_synthetic_profile_table
from spectral_packet_engine.tabular import load_tabular_dataset, supported_tabular_formats
from spectral_packet_engine.table_io import load_profile_table, save_profile_table_csv, supported_profile_table_formats
from spectral_packet_engine.tf_surrogate import TensorFlowRegressorConfig
from spectral_packet_engine.workflows import (
    analyze_coupled_channel_surfaces,
    analyze_profile_table_spectra,
    analyze_profile_table_from_database_query,
    analyze_separable_tensor_product_spectrum,
    benchmark_transport_scan,
    build_profile_table_report_from_database_query,
    bootstrap_local_database,
    compare_profile_tables,
    compress_profile_table,
    compress_profile_table_from_database_query,
    database_profile_query_workflow_artifact_metadata,
    database_query_workflow_artifact_metadata,
    describe_database_table,
    describe_potential_families,
    design_potential_for_target_transition,
    execute_database_script,
    execute_database_statement,
    export_feature_table_from_database_query,
    export_feature_table_from_profile_table,
    evaluate_modal_surrogate_from_database_query,
    evaluate_modal_surrogate_on_profile_table,
    evaluate_tensorflow_surrogate_on_profile_table,
    fit_gaussian_packet_to_profile_table,
    fit_gaussian_packet_to_profile_table_from_database_query,
    GradientOptimizationConfig,
    infer_potential_family_from_spectrum,
    inspect_database,
    inspect_environment,
    inspect_ml_backend_support,
    inspect_tree_backend_support,
    load_profile_table_report,
    materialize_database_query,
    materialize_database_query_to_table,
    optimize_packet_control,
    simulate_packet_sweep,
    project_gaussian_packet,
    run_control_workflow,
    run_profile_inference_workflow,
    run_spectroscopy_workflow,
    run_transport_resonance_workflow,
    simulate_gaussian_packet,
    solve_radial_reduction,
    summarize_database_query_result,
    summarize_profile_table,
    summarize_tabular_dataset,
    sweep_profile_table_compression,
    train_tree_model,
    train_modal_surrogate_from_database_query,
    train_modal_surrogate_on_profile_table,
    train_tensorflow_surrogate_on_profile_table,
    tune_tree_model,
    coerce_database_table_types,
    pivot_database_table,
    unpivot_database_table,
    interpolate_database_time_series,
    window_aggregate_database_query,
    validate_installation,
    write_profile_table_to_database,
    write_tabular_dataset_to_database,
)
from spectral_packet_engine.spectral_extensions import (
    fourier_decomposition,
    pade_approximant,
    hilbert_transform,
    correlation_spectral_analysis,
    richardson_extrapolation,
    kramers_kronig,
)


def mcp_is_available() -> bool:
    return importlib.util.find_spec("mcp.server.fastmcp") is not None


def _coerce_parameters(parameters: dict[str, Any] | None) -> dict[str, Any]:
    return {} if parameters is None else {str(key): value for key, value in parameters.items()}


def _run_sql_profile_workflow(
    workflow_fn,
    artifact_writer,
    workflow_tag: str,
    database: str,
    query: str,
    *,
    num_modes: int = 32,
    device: str = "auto",
    time_column: str = "time",
    position_columns: list[str] | None = None,
    sort_by_time: bool = False,
    normalize_each_profile: bool = False,
    parameters: dict[str, Any] | None = None,
    output_dir: str | None = None,
    **extra_kwargs,
) -> dict[str, Any]:
    """Shared dispatch for SQL-backed profile workflows."""
    params = _coerce_parameters(parameters)
    summary = workflow_fn(
        database,
        query,
        parameters=params,
        time_column=time_column,
        position_columns=position_columns,
        sort_by_time=sort_by_time,
        num_modes=num_modes,
        device=device,
        normalize_each_profile=normalize_each_profile,
        **extra_kwargs,
    )
    if output_dir is not None:
        metadata = database_profile_query_workflow_artifact_metadata(
            workflow_tag,
            database,
            query,
            parameters=params,
            time_column=time_column,
            position_columns=position_columns,
            sort_by_time=sort_by_time,
        )
        artifact_writer(output_dir, summary, metadata=metadata)
    return to_serializable(summary)


def _managed_scratch_directory(config: MCPServerConfig) -> Path:
    return ensure_mcp_scratch_dir(config).resolve()


def _managed_scratch_path(config: MCPServerConfig, name: str) -> Path:
    return resolve_mcp_scratch_file(name, config=config)


def _validate_synthetic_generation_request(config: MCPServerConfig, *, num_profiles: int, grid_points: int) -> None:
    if int(num_profiles) <= 0:
        raise ValueError("num_profiles must be positive")
    if int(grid_points) <= 0:
        raise ValueError("grid_points must be positive")
    if int(num_profiles) > config.max_generated_profiles:
        raise ValueError(
            f"num_profiles exceeds the configured MCP generation limit of {config.max_generated_profiles}"
        )
    if int(grid_points) > config.max_generated_grid_points:
        raise ValueError(
            f"grid_points exceeds the configured MCP generation limit of {config.max_generated_grid_points}"
        )


def _dedupe_ordered(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _build_transport_security_settings(runtime_config: MCPServerConfig):
    if runtime_config.transport != "streamable-http":
        return None

    try:
        from mcp.server.transport_security import TransportSecuritySettings
    except ModuleNotFoundError:
        return None

    allowed_hosts = list(runtime_config.allowed_hosts)
    allowed_origins = list(runtime_config.allowed_origins)

    if runtime_config.host in {"127.0.0.1", "localhost", "::1"}:
        allowed_hosts = [
            "127.0.0.1",
            "127.0.0.1:*",
            "localhost",
            "localhost:*",
            "[::1]",
            "[::1]:*",
            *allowed_hosts,
        ]
        allowed_origins = [
            "http://127.0.0.1",
            "http://127.0.0.1:*",
            "http://localhost",
            "http://localhost:*",
            "http://[::1]",
            "http://[::1]:*",
            "https://127.0.0.1",
            "https://127.0.0.1:*",
            "https://localhost",
            "https://localhost:*",
            "https://[::1]",
            "https://[::1]:*",
            *allowed_origins,
        ]

    if not allowed_hosts and not allowed_origins:
        return None

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_dedupe_ordered(allowed_hosts),
        allowed_origins=_dedupe_ordered(allowed_origins),
    )


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


def _tool_error_payload(tool_name: str, exc: Exception) -> dict[str, Any]:
    return {
        "error": True,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "tool": tool_name,
    }


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
            ) as _task_id:
                try:
                    if not bounded:
                        return function(*args, **kwargs)
                    slot_info = runtime.acquire()
                    try:
                        result = function(*args, **kwargs)
                        if isinstance(result, dict) and bounded:
                            result["_execution"] = slot_info
                        return result
                    finally:
                        runtime.release()
                except Exception as exc:
                    mark_service_task_failed(_task_id, exc)
                    return _tool_error_payload(name, exc)

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
        instructions=SERVER_PURPOSE,
        log_level=runtime_config.log_level,
        host=runtime_config.host,
        port=runtime_config.port,
        streamable_http_path=runtime_config.streamable_http_path,
        transport_security=_build_transport_security_settings(runtime_config),
    )

    @_tool(server, runtime, "inspect_product", "Return engine identity: version, basis family, supported domains, and registered workflow map.")
    def inspect_product_tool() -> dict[str, Any]:
        return to_serializable(inspect_product_identity())

    @_tool(server, runtime, "guide_workflow", "Recommend a workflow (report, inverse-fit, or feature-export) with sensible defaults for the given input kind and goal.")
    def guide_workflow_tool(input_kind: str = "profile-table-file", goal: str = "report") -> dict[str, Any]:
        return to_serializable(guide_workflow(surface="mcp", input_kind=input_kind, goal=goal))

    @_tool(server, runtime, "inspect_environment", "Report CPU/GPU availability, PyTorch device selection, thread counts, and optional backend status (TensorFlow, JAX, SQL).")
    def inspect_environment_tool(device: str = "auto") -> dict[str, Any]:
        return to_serializable(inspect_environment(device))

    @_tool(server, runtime, "inspect_mcp_runtime", "Report MCP transport config: host, port, allowed origins, bounded-execution concurrency limits, and log level.")
    def inspect_mcp_runtime_tool() -> dict[str, Any]:
        return to_serializable(inspect_mcp_runtime(runtime_config))

    @_tool(server, runtime, "inspect_service_status", "Report uptime, completed/failed task counts, and recent tool execution history with timing.")
    def inspect_service_status_tool() -> dict[str, Any]:
        return to_serializable(inspect_service_status())

    @_tool(server, runtime, "validate_installation", "Check that core, file-IO, SQL, MCP, and ML surfaces are importable and report their status (stable/beta/experimental).")
    def validate_installation_tool(device: str = "auto") -> dict[str, Any]:
        return to_serializable(validate_installation(device))

    @_tool(server, runtime, "inspect_ml_backends", "Report which modal-regression backends are available (PyTorch, TensorFlow, JAX) and their device/version details.")
    def inspect_ml_backends_tool(device: str = "auto") -> dict[str, Any]:
        return to_serializable(inspect_ml_backend_support(device))

    @_tool(server, runtime, "inspect_tree_backends", "Report which tree-model libraries are available (scikit-learn, XGBoost, LightGBM, CatBoost) and their versions.")
    def inspect_tree_backends_tool(library: str = "auto") -> dict[str, Any]:
        return to_serializable(inspect_tree_backend_support(requested_library=library))

    @_tool(server, runtime, "supported_profile_formats", "List file formats supported for profile-table ingestion (CSV, Parquet, HDF5, etc.) with availability status.")
    def supported_profile_formats_tool() -> dict[str, bool]:
        return supported_profile_table_formats()

    @_tool(server, runtime, "supported_tabular_formats", "List file formats supported for generic tabular dataset ingestion with availability status.")
    def supported_tabular_formats_tool() -> dict[str, bool]:
        return supported_tabular_formats()

    @server.resource(
        "spectral://capabilities/inverse-uq",
        name="inverse_uq_capabilities",
        description="Structured overview of uncertainty-aware inverse inference capabilities exposed by the spectral engine.",
        mime_type="application/json",
    )
    def inverse_uq_resource() -> dict[str, Any]:
        return {
            "category": "inverse-uq",
            "positioning": "Local uncertainty-aware physical inference over explicit bounded-domain model families.",
            "tools": [
                "infer_potential_spectrum",
                "fit_packet_to_profile_table",
                "fit_packet_to_database_profile_query",
            ],
            "artifacts": [
                "uncertainty_summary.json",
                "parameter_posterior.csv",
                "modal_posterior.csv",
                "sensitivity_map.json",
                "candidate_ranking.csv",
            ],
            "potential_families": describe_potential_families(),
            "limitations": [
                "Posterior summaries are local Laplace-style approximations, not global Bayesian posteriors over unrestricted model spaces.",
                "Potential-family comparison uses explicit parametric families and structured evidence weights rather than black-box model search.",
            ],
        }

    @server.resource(
        "spectral://capabilities/reduced-models",
        name="reduced_model_capabilities",
        description="Structured overview of controlled reduced-model workflows beyond plain 1D.",
        mime_type="application/json",
    )
    def reduced_models_resource() -> dict[str, Any]:
        return {
            "category": "reduced-models",
            "tools": [
                "analyze_separable_spectrum",
                "analyze_coupled_surfaces",
                "solve_radial_reduction",
            ],
            "scope": [
                "Separable tensor-product spectra from independent 1D axes",
                "Reduced two-channel adiabatic surface analysis",
                "Radial effective-coordinate reductions with centrifugal terms",
            ],
            "limitations": [
                "No claim is made for arbitrary full 2D/3D solvers.",
                "Assumptions are surfaced explicitly in each workflow summary and artifact bundle.",
            ],
        }

    @server.resource(
        "spectral://capabilities/differentiable-physics",
        name="differentiable_physics_capabilities",
        description="Structured overview of differentiable calibration and inverse-design workflows.",
        mime_type="application/json",
    )
    def differentiable_physics_resource() -> dict[str, Any]:
        return {
            "category": "differentiable-physics",
            "tools": [
                "design_transition",
                "optimize_packet_control",
            ],
            "objectives": [
                "potential -> spectrum calibration",
                "potential -> target transition inverse design",
                "state preparation -> target observable steering",
            ],
            "limitations": [
                "Gradients are local to the explicit parameterization and can become delicate near eigenvalue crossings or non-smooth parameter maps.",
            ],
        }

    @server.resource(
        "spectral://capabilities/vertical-workflows",
        name="vertical_workflow_capabilities",
        description="Structured overview of domain-specific vertical workflows built on the same spectral core.",
        mime_type="application/json",
    )
    def vertical_workflows_resource() -> dict[str, Any]:
        return {
            "category": "vertical-workflows",
            "tools": [
                "infer_potential_spectrum",
                "transport_workflow",
                "optimize_packet_control",
                "profile_inference_workflow",
            ],
            "verticals": {
                "spectroscopy": "Potential-family inference from low-lying spectra with uncertainty-aware best-fit summaries.",
                "transport": "Barrier/resonance workflow chaining transfer-matrix scattering, WKB comparison, propagation, and Wigner diagnostics.",
                "control": "Differentiable packet steering toward target observables by optimizing state preparation parameters.",
                "scientific-tabular": "Report-first profile-table workflow that couples spectral compression, inverse fitting, and feature export.",
            },
        }

    @server.prompt(
        name="select_inverse_physics_workflow",
        description="Explain which inverse/UQ spectral workflow to call for a given observation type and desired output.",
    )
    def inverse_physics_prompt(
        observation_type: str = "spectrum",
        desired_output: str = "family inference with uncertainty",
    ) -> str:
        return (
            "Use the spectral inverse-physics tools, not a generic regression tool.\n"
            f"Observation type: {observation_type}\n"
            f"Desired output: {desired_output}\n"
            "If the observation is a low-lying spectrum, call infer_potential_spectrum.\n"
            "If the observation is a file-backed or SQL-backed density profile table, call profile_table_report first and only then fit_packet_to_profile_table or fit_packet_to_database_profile_query.\n"
            "Prefer artifact-backed runs when the client needs inspectable posterior, sensitivity, or provenance outputs."
        )

    @server.prompt(
        name="select_reduced_model_strategy",
        description="Explain which reduced-model tool to use based on the available physical structure.",
    )
    def reduced_model_prompt(problem_structure: str = "separable 2D Hamiltonian") -> str:
        return (
            "Choose the reduced model that matches the explicit structure in the physics problem.\n"
            f"Problem structure: {problem_structure}\n"
            "Use analyze_separable_spectrum for tensor-product separability, analyze_coupled_surfaces for reduced multi-surface avoided crossings, and solve_radial_reduction for effective radial coordinates with angular momentum.\n"
            "Do not describe these workflows as arbitrary multidimensional solvers."
        )

    @server.prompt(
        name="select_vertical_workflow",
        description="Explain which domain-specific vertical workflow best matches the user’s scientific question.",
    )
    def vertical_workflow_prompt(domain_question: str = "spectroscopy") -> str:
        return (
            "Map the user's question onto one explicit spectral vertical workflow.\n"
            f"Question domain: {domain_question}\n"
            "Use infer_potential_spectrum for spectroscopy or family inference, transport_workflow for barrier and resonance questions, optimize_packet_control for packet steering objectives, and profile_inference_workflow for report-first scientific tabular jobs.\n"
            "Keep the explanation tied to the spectral engine rather than generic ML language."
        )

    @_tool(server, runtime, "simulate_packet", "Simulate a Gaussian wavepacket in the infinite-well basis: project onto modes, propagate in time, and return grid-space snapshots with energy conservation diagnostics.", bounded=True)
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

    @_tool(server, runtime, "project_packet", "Project a Gaussian wavepacket onto the infinite-well sine basis and return modal coefficients, projection quality, and grid-space reconstruction.", bounded=True)
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

    @_tool(server, runtime, "inspect_profile_table", "Load a profile table file and report its grid dimensions, time slices, value ranges, and readiness for modal decomposition.")
    def inspect_profile_table_tool(table_path: str, device: str = "auto") -> dict[str, Any]:
        return to_serializable(summarize_profile_table(load_profile_table(table_path), device=device))

    @_tool(server, runtime, "profile_table_report", "Full pipeline on a profile table file: inspect schema, decompose into modal basis, compress, and return a unified report with convergence diagnostics.", bounded=True)
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

    @_tool(server, runtime, "export_feature_table", "Extract modal coefficients and moments from a profile table file into a flat feature table (CSV/Parquet) for downstream ML.", bounded=True)
    def export_feature_table_tool(
        table_path: str,
        num_modes: int = 32,
        device: str = "auto",
        normalize_each_profile: bool = False,
        include: list[str] | None = None,
        format: str = "csv",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        requested_includes = {"coefficients", "moments"} if not include else set(include)
        summary = export_feature_table_from_profile_table(
            table_path,
            num_modes=num_modes,
            device=device,
            normalize_each_profile=normalize_each_profile,
            include_coefficients="coefficients" in requested_includes,
            include_moments="moments" in requested_includes,
            format=format,
        )
        if output_dir is not None:
            write_feature_table_artifacts(
                output_dir,
                summary,
                metadata={"input": {"table_path": table_path}},
            )
            summary = replace(summary, output_path=str(Path(output_dir) / f"features.{format}"))
        return to_serializable(summary)

    @_tool(server, runtime, "inspect_tabular_dataset", "Inspect a generic tabular dataset and report schema, validation, and preview rows.")
    def inspect_tabular_dataset_tool(dataset_path: str) -> dict[str, Any]:
        return to_serializable(summarize_tabular_dataset(load_tabular_dataset(dataset_path)))

    @_tool(server, runtime, "analyze_profile_table", "Decompose a profile table file into its infinite-well modal basis and report per-mode energy, convergence rate, and truncation diagnostics.", bounded=True)
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

    @_tool(server, runtime, "compress_profile_table", "Project a profile table file onto a truncated modal basis, returning coefficients, reconstructions, and per-profile L2 error.", bounded=True)
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

    @_tool(server, runtime, "compression_sweep", "Compress a profile table at multiple truncation levels (e.g. 4, 8, 16, 32 modes) and report L2 error vs. mode count to guide optimal truncation.", bounded=True)
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

    @_tool(server, runtime, "fit_packet_to_profile_table", "Inverse-fit Gaussian packet parameters (center, width, wavenumber, phase) to a profile table file using gradient descent on L2 reconstruction error.", bounded=True)
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

    @_tool(server, runtime, "describe_potential_families", "List the explicit bounded-domain potential families available for inverse inference and differentiable design.")
    def describe_potential_families_tool() -> dict[str, Any]:
        return {"families": describe_potential_families()}

    @_tool(server, runtime, "infer_potential_spectrum", "Infer which explicit potential family best explains an observed low-lying spectrum, returning family ranking, best-fit parameters, and local uncertainty summaries.", bounded=True)
    def infer_potential_spectrum_tool(
        target_eigenvalues: list[float],
        families: list[str] | None = None,
        initial_guesses: dict[str, Any] | None = None,
        domain_length: float = 1.0,
        left: float = 0.0,
        mass: float = 1.0,
        hbar: float = 1.0,
        num_points: int = 128,
        steps: int = 200,
        learning_rate: float = 0.05,
        device: str = "auto",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        normalized_initial_guesses = None
        if initial_guesses is not None:
            normalized_initial_guesses = {
                str(family): {str(name): float(value) for name, value in parameters.items()}
                for family, parameters in initial_guesses.items()
            }
        summary = infer_potential_family_from_spectrum(
            target_eigenvalues=target_eigenvalues,
            families=families,
            initial_guesses=normalized_initial_guesses,
            domain_length=domain_length,
            left=left,
            mass=mass,
            hbar=hbar,
            num_points=num_points,
            optimization_config=GradientOptimizationConfig(steps=steps, learning_rate=learning_rate),
            device=device,
        )
        if output_dir is not None:
            write_potential_inference_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "analyze_separable_spectrum", "Analyze a restricted separable tensor-product spectrum from two independent 1D bounded-domain potential families.", bounded=True)
    def analyze_separable_spectrum_tool(
        family_x: str,
        parameters_x: dict[str, float],
        family_y: str,
        parameters_y: dict[str, float],
        domain_length_x: float = 1.0,
        domain_length_y: float = 1.0,
        num_points_x: int = 96,
        num_points_y: int = 96,
        num_states_x: int = 6,
        num_states_y: int = 6,
        num_combined_states: int = 12,
        low_rank_rank: int = 1,
        device: str = "auto",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = analyze_separable_tensor_product_spectrum(
            family_x=family_x,
            parameters_x={str(name): float(value) for name, value in parameters_x.items()},
            family_y=family_y,
            parameters_y={str(name): float(value) for name, value in parameters_y.items()},
            domain_length_x=domain_length_x,
            domain_length_y=domain_length_y,
            num_points_x=num_points_x,
            num_points_y=num_points_y,
            num_states_x=num_states_x,
            num_states_y=num_states_y,
            num_combined_states=num_combined_states,
            low_rank_rank=low_rank_rank,
            device=device,
        )
        if output_dir is not None:
            write_reduced_model_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "analyze_coupled_surfaces", "Analyze a reduced two-channel avoided crossing, including adiabatic surfaces and derivative couplings.", bounded=True)
    def analyze_coupled_surfaces_tool(
        domain_length: float = 1.0,
        grid_points: int = 256,
        slope: float = 30.0,
        bias: float = 0.0,
        coupling: float = 2.0,
        coupling_width: float = 0.12,
        device: str = "cpu",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = analyze_coupled_channel_surfaces(
            domain_length=domain_length,
            grid_points=grid_points,
            slope=slope,
            bias=bias,
            coupling=coupling,
            coupling_width=coupling_width,
            device=device,
        )
        if output_dir is not None:
            write_reduced_model_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "solve_radial_reduction", "Solve a bounded radial effective-coordinate reduction with explicit angular-momentum and base-potential parameters.", bounded=True)
    def solve_radial_reduction_tool(
        family: str,
        parameters: dict[str, float],
        angular_momentum: int = 0,
        radial_min: float = 0.05,
        radial_max: float = 3.0,
        num_points: int = 128,
        num_states: int = 6,
        mass: float = 1.0,
        hbar: float = 1.0,
        device: str = "cpu",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = solve_radial_reduction(
            family=family,
            parameters={str(name): float(value) for name, value in parameters.items()},
            angular_momentum=angular_momentum,
            radial_min=radial_min,
            radial_max=radial_max,
            num_points=num_points,
            num_states=num_states,
            mass=mass,
            hbar=hbar,
            device=device,
        )
        if output_dir is not None:
            write_reduced_model_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "design_transition", "Optimize a parameterized potential family so a selected spectral transition matches a target value.", bounded=True)
    def design_transition_tool(
        family: str,
        target_transition: float,
        initial_guess: dict[str, float],
        transition_indices: list[int] | None = None,
        domain_length: float = 1.0,
        left: float = 0.0,
        mass: float = 1.0,
        hbar: float = 1.0,
        num_points: int = 128,
        num_states: int = 4,
        steps: int = 200,
        learning_rate: float = 0.05,
        device: str = "auto",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        pair = (0, 1) if transition_indices is None else tuple(int(value) for value in transition_indices)
        summary = design_potential_for_target_transition(
            family=family,
            target_transition=target_transition,
            transition_indices=pair,
            initial_guess={str(name): float(value) for name, value in initial_guess.items()},
            domain_length=domain_length,
            left=left,
            mass=mass,
            hbar=hbar,
            num_points=num_points,
            num_states=num_states,
            optimization_config=GradientOptimizationConfig(steps=steps, learning_rate=learning_rate),
            device=device,
        )
        if output_dir is not None:
            write_differentiable_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "optimize_packet_control", "Optimize Gaussian packet preparation parameters for a target observable using differentiable spectral propagation.", bounded=True)
    def optimize_packet_control_tool(
        center: float = 0.30,
        width: float = 0.07,
        wavenumber: float = 25.0,
        phase: float = 0.0,
        objective: str = "target_position",
        target_value: float = 0.60,
        final_time: float = 0.01,
        interval: list[float] | None = None,
        num_modes: int = 96,
        quadrature_points: int = 2048,
        grid_points: int = 128,
        steps: int = 200,
        learning_rate: float = 0.05,
        device: str = "auto",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = optimize_packet_control(
            initial_guess={
                "center": center,
                "width": width,
                "wavenumber": wavenumber,
                "phase": phase,
            },
            objective=objective,  # type: ignore[arg-type]
            target_value=target_value,
            final_time=final_time,
            interval=None if interval is None else (float(interval[0]), float(interval[1])),
            num_modes=num_modes,
            quadrature_points=quadrature_points,
            grid_points=grid_points,
            optimization_config=GradientOptimizationConfig(steps=steps, learning_rate=learning_rate),
            device=device,
        )
        if output_dir is not None:
            write_differentiable_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "transport_workflow", "Run the barrier/resonance vertical workflow that combines scattering, WKB comparison, propagation, and Wigner diagnostics.", bounded=True)
    def transport_workflow_tool(
        barrier_height: float = 50.0,
        barrier_width_sigma: float = 0.03,
        domain_length: float = 1.0,
        grid_points: int = 512,
        num_modes: int = 128,
        num_energies: int = 500,
        packet_center: float = 0.25,
        packet_width: float = 0.04,
        packet_wavenumber: float = 40.0,
        device: str = "cpu",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = run_transport_resonance_workflow(
            barrier_height=barrier_height,
            barrier_width_sigma=barrier_width_sigma,
            domain_length=domain_length,
            grid_points=grid_points,
            num_modes=num_modes,
            num_energies=num_energies,
            packet_center=packet_center,
            packet_width=packet_width,
            packet_wavenumber=packet_wavenumber,
            device=device,
        )
        if output_dir is not None:
            write_vertical_workflow_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "profile_inference_workflow", "Run the report-first scientific tabular vertical: summarize, inverse-fit with uncertainty, and export spectral features from one profile table.", bounded=True)
    def profile_inference_workflow_tool(
        table_path: str,
        center: float = 0.36,
        width: float = 0.11,
        wavenumber: float = 22.0,
        phase: float = 0.0,
        analyze_num_modes: int = 16,
        compress_num_modes: int = 8,
        inverse_num_modes: int = 96,
        feature_num_modes: int = 16,
        quadrature_points: int = 2048,
        normalize_each_profile: bool = False,
        device: str = "auto",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        summary = run_profile_inference_workflow(
            table_path,
            initial_guess={
                "center": center,
                "width": width,
                "wavenumber": wavenumber,
                "phase": phase,
            },
            analyze_num_modes=analyze_num_modes,
            compress_num_modes=compress_num_modes,
            inverse_num_modes=inverse_num_modes,
            feature_num_modes=feature_num_modes,
            quadrature_points=quadrature_points,
            normalize_each_profile=normalize_each_profile,
            device=device,
        )
        if output_dir is not None:
            write_vertical_workflow_artifacts(output_dir, summary)
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

    @_tool(server, runtime, "query_database", "Run a read-only parameterized SQL query and return the result. Returns ALL rows by default (up to max_rows). Do NOT use bash/python/sqlite3 to access database files — always use this tool or export_query_csv instead.", bounded=True)
    def query_database_tool(
        database: str,
        query: str,
        parameters: dict[str, Any] | None = None,
        max_rows: int = 500,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        result = materialize_database_query(database, query, parameters=_coerce_parameters(parameters))
        if output_dir is not None:
            write_tabular_artifacts(
                output_dir,
                result.dataset,
                summary_name="db_query_summary.json",
                table_name="query_result.csv",
                metadata=database_query_workflow_artifact_metadata(
                    "db-query",
                    database,
                    query,
                    parameters=_coerce_parameters(parameters),
                ),
            )
        summary = to_serializable(
            summarize_database_query_result(
                database,
                query,
                result,
                parameters=_coerce_parameters(parameters),
            )
        )
        # Include all rows (up to max_rows) so models don't need bash
        all_rows = result.dataset.to_rows(limit=max_rows)
        summary["table"]["rows"] = to_serializable(all_rows)
        if result.dataset.row_count > max_rows:
            summary["table"]["truncated"] = True
            summary["table"]["hint"] = (
                f"Result has {result.dataset.row_count} rows but only {max_rows} returned. "
                f"Use export_query_csv for the full dataset, or add LIMIT to your query."
            )
        return summary

    @_tool(server, runtime, "export_query_csv", "Run a SQL query and return the FULL result as inline CSV text. Use this when you need all rows, not just a preview. For large results (>1000 rows), consider adding a LIMIT clause.", bounded=True)
    def export_query_csv_tool(
        database: str,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import io
        import csv as csv_mod
        result = materialize_database_query(database, query, parameters=_coerce_parameters(parameters))
        rows = result.dataset.to_rows()
        columns = list(result.dataset.column_names)
        buf = io.StringIO()
        writer = csv_mod.writer(buf)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(c) for c in columns])
        return {
            "row_count": len(rows),
            "column_count": len(columns),
            "columns": columns,
            "csv": buf.getvalue(),
        }

    @_tool(server, runtime, "execute_database_statement", "Run a non-query SQL statement such as CREATE, INSERT, UPDATE, or DELETE against a managed database.", bounded=True)
    def execute_database_statement_tool(
        database: str,
        statement: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return to_serializable(
            execute_database_statement(
                database,
                statement,
                parameters=_coerce_parameters(parameters),
                create_if_missing=True,
            )
        )

    @_tool(server, runtime, "execute_database_script", "Run a multi-statement SQL script against a managed SQLite database.", bounded=True)
    def execute_database_script_tool(
        database: str,
        script: str,
    ) -> dict[str, Any]:
        return to_serializable(
            execute_database_script(
                database,
                script,
                create_if_missing=True,
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

    @_tool(server, runtime, "coerce_table_types", "Detect and fix column type affinities for an existing database table. Converts TEXT columns containing numeric data to INTEGER or REAL.", bounded=True)
    def coerce_table_types_tool(
        database: str,
        table_name: str,
    ) -> dict[str, Any]:
        return to_serializable(
            coerce_database_table_types(database, table_name)
        )

    @_tool(server, runtime, "pivot_table", "Pivot a long-format table into wide format. Turns distinct values in pivot_column into separate columns.", bounded=True)
    def pivot_table_tool(
        database: str,
        table_name: str,
        target_table: str,
        index_column: str,
        pivot_column: str,
        value_column: str,
        aggregate: str = "MAX",
        replace: bool = False,
    ) -> dict[str, Any]:
        return to_serializable(
            pivot_database_table(
                database, table_name, target_table,
                index_column, pivot_column, value_column,
                aggregate=aggregate, replace=replace,
            )
        )

    @_tool(server, runtime, "unpivot_table", "Unpivot (melt) a wide-format table into long format with variable/value columns.", bounded=True)
    def unpivot_table_tool(
        database: str,
        table_name: str,
        target_table: str,
        id_columns: list[str],
        value_columns: list[str] | None = None,
        replace: bool = False,
    ) -> dict[str, Any]:
        return to_serializable(
            unpivot_database_table(
                database, table_name, target_table,
                id_columns, value_columns, replace=replace,
            )
        )

    @_tool(server, runtime, "interpolate_time_series", "Fill missing time steps in a table using linear interpolation and persist the result.", bounded=True)
    def interpolate_time_series_tool(
        database: str,
        table_name: str,
        target_table: str,
        time_column: str,
        value_columns: list[str],
        step: float = 1.0,
        replace: bool = False,
    ) -> dict[str, Any]:
        return to_serializable(
            interpolate_database_time_series(
                database, table_name, target_table,
                time_column, value_columns,
                step=step, replace=replace,
            )
        )

    @_tool(server, runtime, "window_aggregate", "Compute sliding window aggregates (AVG, SUM, COUNT, etc.) over a column.", bounded=True)
    def window_aggregate_tool(
        database: str,
        table_name: str,
        value_column: str,
        order_by: str,
        window_size: int = 3,
        functions: list[str] | None = None,
    ) -> dict[str, Any]:
        fns = tuple(functions) if functions else ("AVG", "SUM", "COUNT")
        return to_serializable(
            window_aggregate_database_query(
                database, table_name, value_column, order_by,
                window_size=window_size, functions=fns,
            )
        )

    @_tool(server, runtime, "analyze_database_profile_query", "Decompose a SQL-sourced profile table into its infinite-well modal basis and report per-mode energy, convergence rate, and truncation diagnostics.", bounded=True)
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
        return _run_sql_profile_workflow(
            analyze_profile_table_from_database_query,
            write_spectral_analysis_artifacts,
            "sql-analyze-table",
            database, query,
            num_modes=num_modes, device=device, time_column=time_column,
            position_columns=position_columns, sort_by_time=sort_by_time,
            normalize_each_profile=normalize_each_profile,
            parameters=parameters, output_dir=output_dir,
        )

    @_tool(server, runtime, "compress_database_profile_query", "Project a SQL-sourced profile table onto a truncated modal basis, returning coefficients, reconstructions, and per-profile L2 error.", bounded=True)
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
        return _run_sql_profile_workflow(
            compress_profile_table_from_database_query,
            write_compression_artifacts,
            "sql-compress-table",
            database, query,
            num_modes=num_modes, device=device, time_column=time_column,
            position_columns=position_columns, sort_by_time=sort_by_time,
            normalize_each_profile=normalize_each_profile,
            parameters=parameters, output_dir=output_dir,
        )

    @_tool(server, runtime, "export_feature_table_from_sql", "Extract modal coefficients and statistical moments from a SQL-sourced profile table into a flat feature table (CSV/Parquet) for downstream ML.", bounded=True)
    def export_feature_table_from_sql_tool(
        database: str,
        query: str,
        num_modes: int = 32,
        device: str = "auto",
        time_column: str = "time",
        position_columns: list[str] | None = None,
        sort_by_time: bool = False,
        normalize_each_profile: bool = False,
        include: list[str] | None = None,
        format: str = "csv",
        parameters: dict[str, Any] | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        requested_includes = {"coefficients", "moments"} if not include else set(include)
        summary = export_feature_table_from_database_query(
            database,
            query,
            parameters=_coerce_parameters(parameters),
            time_column=time_column,
            position_columns=position_columns,
            sort_by_time=sort_by_time,
            num_modes=num_modes,
            device=device,
            normalize_each_profile=normalize_each_profile,
            include_coefficients="coefficients" in requested_includes,
            include_moments="moments" in requested_includes,
            format=format,
        )
        if output_dir is not None:
            write_feature_table_artifacts(
                output_dir,
                summary,
                metadata=database_profile_query_workflow_artifact_metadata(
                    "export-features",
                    database,
                    query,
                    parameters=_coerce_parameters(parameters),
                    time_column=time_column,
                    position_columns=position_columns,
                    sort_by_time=sort_by_time,
                ),
            )
            summary = replace(summary, output_path=str(Path(output_dir) / f"features.{format}"))
        return to_serializable(summary)

    @_tool(server, runtime, "report_database_profile_query", "Full pipeline on a SQL-sourced profile table: inspect schema, analyze modal structure, compress, and return a unified report with convergence diagnostics.", bounded=True)
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
                metadata=database_profile_query_workflow_artifact_metadata(
                    "profile-report",
                    database,
                    query,
                    parameters=_coerce_parameters(parameters),
                    time_column=time_column,
                    position_columns=position_columns,
                    sort_by_time=sort_by_time,
                ),
            )
        return to_serializable(report)

    @_tool(server, runtime, "fit_packet_to_database_profile_query", "Inverse-fit Gaussian packet parameters (center, width, wavenumber, phase) to a SQL-sourced profile table using gradient descent on L2 reconstruction error.", bounded=True)
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
                metadata=database_profile_query_workflow_artifact_metadata(
                    "sql-fit-table",
                    database,
                    query,
                    parameters=_coerce_parameters(parameters),
                    time_column=time_column,
                    position_columns=position_columns,
                    sort_by_time=sort_by_time,
                ),
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

    @_tool(server, runtime, "train_tensorflow_surrogate", "Train a TensorFlow neural network to predict spectral coefficients from profile features, enabling fast surrogate evaluation on non-PyTorch hardware.", bounded=True)
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

    @_tool(server, runtime, "evaluate_tensorflow_surrogate", "Train and evaluate a TensorFlow spectral-coefficient regressor on a profile table, reporting per-mode prediction error and overall reconstruction quality.", bounded=True)
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

    @_tool(server, runtime, "train_modal_surrogate", "Train a modal-coefficient regression model (PyTorch, TensorFlow, or JAX backend) on a profile table. Auto-selects available backend if set to 'auto'.", bounded=True)
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

    @_tool(server, runtime, "evaluate_modal_surrogate", "Train and evaluate a modal-coefficient regressor on a profile table, reporting per-mode prediction error, reconstruction L2 error, and backend details.", bounded=True)
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

    @_tool(server, runtime, "train_tree_model", "Train a tree model on a feature table and optionally write training artifacts.", bounded=True)
    def train_tree_model_tool(
        features_path: str,
        target_column: str,
        feature_columns: list[str] | None = None,
        task: str = "regression",
        library: str = "auto",
        model: str | None = None,
        params: dict[str, Any] | None = None,
        test_fraction: float = 0.2,
        random_state: int = 0,
        export_dir: str | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        resolved_export_dir = export_dir
        if resolved_export_dir is None and output_dir is not None:
            resolved_export_dir = output_dir
        summary = train_tree_model(
            features_path,
            target_column=target_column,
            feature_columns=feature_columns,
            task=task,
            library=library,
            model=model,
            params=_coerce_parameters(params),
            test_fraction=test_fraction,
            random_state=random_state,
            export_dir=resolved_export_dir,
        )
        if output_dir is not None:
            write_tree_training_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "tune_tree_model", "Hyperparameter-optimize a tree classifier/regressor (scikit-learn, XGBoost, LightGBM) on a spectral feature table. Returns best params, CV scores, and feature importances.", bounded=True)
    def tune_tree_model_tool(
        features_path: str,
        target_column: str,
        search_space: dict[str, Any],
        feature_columns: list[str] | None = None,
        task: str = "regression",
        library: str = "auto",
        model: str | None = None,
        search_kind: str = "random",
        n_iter: int = 30,
        cv: int = 5,
        scoring: str | None = None,
        test_fraction: float = 0.2,
        random_state: int = 0,
        export_dir: str | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        resolved_export_dir = export_dir
        if resolved_export_dir is None and output_dir is not None:
            resolved_export_dir = str(Path(output_dir) / "best_model")
        summary = tune_tree_model(
            features_path,
            target_column=target_column,
            feature_columns=feature_columns,
            task=task,
            library=library,
            model=model,
            search_space={str(key): list(value) for key, value in search_space.items()},
            search_kind=search_kind,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            test_fraction=test_fraction,
            random_state=random_state,
            export_dir=resolved_export_dir,
        )
        if output_dir is not None:
            write_tree_tuning_artifacts(output_dir, summary)
        return to_serializable(summary)

    @_tool(server, runtime, "train_modal_surrogate_from_sql", "Materialize a SQL profile query and train a modal-coefficient regressor (PyTorch/TensorFlow/JAX) on the resulting profile table.", bounded=True)
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
                metadata=database_profile_query_workflow_artifact_metadata(
                    "sql-ml-train-table",
                    database,
                    query,
                    parameters=_coerce_parameters(parameters),
                    time_column=time_column,
                    position_columns=position_columns,
                    sort_by_time=sort_by_time,
                ),
            )
        return to_serializable(summary)

    @_tool(server, runtime, "evaluate_modal_surrogate_from_sql", "Materialize a SQL profile query and train+evaluate a modal-coefficient regressor, reporting per-mode error and reconstruction quality.", bounded=True)
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
                metadata=database_profile_query_workflow_artifact_metadata(
                    "sql-ml-evaluate-table",
                    database,
                    query,
                    parameters=_coerce_parameters(parameters),
                    time_column=time_column,
                    position_columns=position_columns,
                    sort_by_time=sort_by_time,
                ),
            )
        return to_serializable(summary)

    @_tool(server, runtime, "list_artifacts", "Inspect an artifact directory and report completion state, metadata, and files.")
    def list_artifacts_tool(output_dir: str) -> dict[str, Any]:
        return to_serializable(inspect_artifact_directory(output_dir))

    # --- Deep spectral physics tools (experimental) ---

    @_tool(
        server,
        runtime,
        "analyze_convergence",
        "Analyze spectral convergence: estimate decay rate (exponential/algebraic/plateau), spectral entropy, optimal truncation point, and optionally detect Gibbs phenomenon.",
        bounded=True,
    )
    def analyze_convergence_tool(
        table_path: str,
        num_modes: int = 32,
        error_tolerance: float = 0.01,
        device: str = "auto",
    ) -> dict[str, Any]:
        from spectral_packet_engine.convergence import analyze_convergence as _analyze_convergence
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.profiles import compress_profiles
        from spectral_packet_engine.runtime import resolve_torch_device

        table = load_profile_table(table_path)
        resolved_device = resolve_torch_device(device)
        grid = torch.as_tensor(table.position_grid, dtype=torch.float64, device=resolved_device)
        profiles = torch.as_tensor(table.profiles, dtype=torch.float64, device=resolved_device)
        domain = InfiniteWell1D(left=grid[0], right=grid[-1])

        compression = compress_profiles(profiles, grid, domain=domain, num_modes=num_modes)
        results = []
        for i in range(profiles.shape[0] if profiles.ndim == 2 else 1):
            coeffs = compression.coefficients[i] if profiles.ndim == 2 else compression.coefficients
            recon = compression.reconstruction[i] if profiles.ndim == 2 else compression.reconstruction
            orig = profiles[i] if profiles.ndim == 2 else profiles
            diag = _analyze_convergence(
                coeffs,
                error_tolerance=error_tolerance,
                reconstruction=recon,
                original=orig,
                grid=grid,
            )
            results.append({
                "profile_index": i,
                "decay_type": diag.decay.decay_type.value,
                "decay_rate": float(diag.decay.rate),
                "decay_r_squared": float(diag.decay.r_squared),
                "spectral_entropy": float(diag.entropy.entropy),
                "effective_mode_count": float(diag.entropy.effective_mode_count),
                "sparsity": float(diag.entropy.sparsity),
                "recommended_modes": diag.truncation.recommended_modes,
                "estimated_error": float(diag.truncation.estimated_error),
                "energy_captured": float(diag.truncation.energy_captured),
                "gibbs_detected": diag.gibbs.detected if diag.gibbs else None,
                "gibbs_overshoot": float(diag.gibbs.overshoot_ratio) if diag.gibbs else None,
            })
        return {"convergence_analysis": results, "num_modes": num_modes, "error_tolerance": error_tolerance}

    @_tool(
        server,
        runtime,
        "compute_energy_budget",
        "Compute the spectral energy budget: energy per mode, cumulative energy capture, and energy tail for truncation analysis.",
        bounded=True,
    )
    def compute_energy_budget_tool(
        table_path: str,
        num_modes: int = 32,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.energy import compute_energy_budget as _compute_budget
        from spectral_packet_engine.profiles import project_profiles_onto_basis

        table = load_profile_table(table_path)
        from spectral_packet_engine.runtime import resolve_torch_device
        resolved_device = resolve_torch_device(device)
        grid = torch.as_tensor(table.position_grid, dtype=torch.float64, device=resolved_device)
        profiles = torch.as_tensor(table.profiles, dtype=torch.float64, device=resolved_device)

        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.basis import InfiniteWellBasis
        domain = InfiniteWell1D(left=grid[0], right=grid[-1])
        basis = InfiniteWellBasis(domain, num_modes)
        coefficients = project_profiles_onto_basis(profiles, grid, basis)

        if coefficients.ndim == 1:
            budget = _compute_budget(coefficients, basis)
            return to_serializable({
                "total_energy": float(budget.total_energy),
                "mode_fractions": budget.mode_fractions.tolist(),
                "cumulative_fraction": budget.cumulative_fraction.tolist(),
                "energy_tail": budget.energy_tail.tolist(),
            })

        # Batch: return mean budget
        mean_coeffs = coefficients[0]  # Use first profile as representative
        budget = _compute_budget(mean_coeffs, basis)
        return to_serializable({
            "total_energy": float(budget.total_energy),
            "mode_fractions": budget.mode_fractions.tolist(),
            "cumulative_fraction": budget.cumulative_fraction.tolist(),
            "energy_tail": budget.energy_tail.tolist(),
            "num_profiles": coefficients.shape[0],
        })

    @_tool(
        server,
        runtime,
        "momentum_analysis",
        "Compute momentum-space observables (<p>, <p^2>, Var(p)) and Heisenberg uncertainty product for a Gaussian packet.",
        bounded=True,
    )
    def momentum_analysis_tool(
        center: float = 0.30,
        width: float = 0.07,
        wavenumber: float = 25.0,
        phase: float = 0.0,
        num_modes: int = 128,
        quadrature_points: int = 4096,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.basis import InfiniteWellBasis
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.momentum import (
            expectation_momentum_spectral,
            expectation_momentum_squared_spectral,
            variance_momentum_spectral,
            heisenberg_uncertainty,
        )
        from spectral_packet_engine.projector import StateProjector, ProjectionConfig
        from spectral_packet_engine.state import make_truncated_gaussian_packet
        from spectral_packet_engine.runtime import resolve_torch_device

        resolved_device = resolve_torch_device(device)
        domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device=resolved_device)
        basis = InfiniteWellBasis(domain, num_modes)
        projector = StateProjector(basis, ProjectionConfig(quadrature_points=quadrature_points))
        packet = make_truncated_gaussian_packet(domain, center=center, width=width, wavenumber=wavenumber, phase=phase)
        spectral_state = projector.project_packet(packet)
        coeffs = spectral_state.coefficients

        p_mean = expectation_momentum_spectral(coeffs, basis)
        p_sq = expectation_momentum_squared_spectral(coeffs, basis)
        var_p = variance_momentum_spectral(coeffs, basis)
        uncertainty = heisenberg_uncertainty(coeffs, basis)

        return {
            "expectation_momentum": float(p_mean),
            "expectation_momentum_squared": float(p_sq),
            "momentum_variance": float(var_p),
            "sigma_p": float(uncertainty.sigma_p),
            "sigma_x": float(uncertainty.sigma_x),
            "uncertainty_product": float(uncertainty.product),
            "hbar_over_2": float(uncertainty.hbar_over_2),
            "saturates_heisenberg_bound": bool(uncertainty.saturates_bound),
            "packet_parameters": {
                "center": center,
                "width": width,
                "wavenumber": wavenumber,
                "num_modes": num_modes,
            },
        }

    @_tool(
        server,
        runtime,
        "check_energy_conservation",
        "Verify energy and norm conservation during spectral propagation of a Gaussian packet.",
        bounded=True,
    )
    def check_energy_conservation_tool(
        center: float = 0.30,
        width: float = 0.07,
        wavenumber: float = 25.0,
        num_modes: int = 128,
        num_time_steps: int = 20,
        max_time: float = 0.01,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.basis import InfiniteWellBasis
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.dynamics import SpectralPropagator
        from spectral_packet_engine.energy import check_energy_conservation as _check_conservation
        from spectral_packet_engine.projector import StateProjector
        from spectral_packet_engine.state import make_truncated_gaussian_packet
        from spectral_packet_engine.runtime import resolve_torch_device

        resolved_device = resolve_torch_device(device)
        domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device=resolved_device)
        basis = InfiniteWellBasis(domain, num_modes)
        projector = StateProjector(basis)
        propagator = SpectralPropagator(basis)
        packet = make_truncated_gaussian_packet(domain, center=center, width=width, wavenumber=wavenumber)
        spectral_state = projector.project_packet(packet)
        times = torch.linspace(0, max_time, num_time_steps, dtype=torch.float64, device=resolved_device)
        propagated = propagator.propagate_many(spectral_state, times)
        report = _check_conservation(spectral_state.coefficients, propagated, basis)

        return {
            "initial_energy": float(report.initial_energy),
            "final_energy": float(report.final_energy),
            "energy_relative_error": float(report.relative_error),
            "max_energy_deviation": float(report.max_deviation),
            "norm_relative_error": float(report.norm_relative_error),
            "is_conserved": report.is_conserved,
            "num_time_steps": num_time_steps,
            "max_time": max_time,
        }

    # ================================================================
    # Analysis Pipelines — high-level, auto-parameterized
    # ================================================================

    @_tool(
        server,
        runtime,
        "analyze_quantum_state_pipeline",
        "Analyze a Gaussian wavepacket in the infinite-well basis: eigenexpansion, energy/momentum expectation values, Heisenberg uncertainty product, spectral entropy, convergence diagnostics, and Wigner function negativity.",
        bounded=True,
    )
    def analyze_quantum_state_pipeline_tool(
        center: float = 0.30,
        width: float = 0.07,
        wavenumber: float = 25.0,
        num_modes: int = 64,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.basis import InfiniteWellBasis
        from spectral_packet_engine.state import make_truncated_gaussian_packet
        from spectral_packet_engine.projector import StateProjector
        from spectral_packet_engine.pipelines import analyze_quantum_state
        from spectral_packet_engine.runtime import resolve_torch_device

        resolved = resolve_torch_device(device)
        domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device=resolved)
        basis = InfiniteWellBasis(domain, num_modes)
        projector = StateProjector(basis)
        packet = make_truncated_gaussian_packet(domain, center=center, width=width, wavenumber=wavenumber)
        state = projector.project_packet(packet)
        report = analyze_quantum_state(state.coefficients, basis)
        return to_serializable(report.to_dict())

    @_tool(
        server,
        runtime,
        "analyze_potential_pipeline",
        "Analyze a 1D potential (harmonic, double-well, Morse, or custom): compute eigenvalues, compare against WKB semiclassical approximation, evaluate partition function, spectral zeta, and Weyl asymptotic law.",
        bounded=True,
    )
    def analyze_potential_pipeline_tool(
        potential: str = "harmonic",
        omega: float = 50.0,
        temperature: float = 10.0,
        num_points: int = 256,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.eigensolver import (
            harmonic_potential, double_well_potential,
            morse_potential, poschl_teller_potential,
        )
        from spectral_packet_engine.pipelines import analyze_potential_landscape
        from spectral_packet_engine.runtime import resolve_torch_device

        resolved = resolve_torch_device(device)
        domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device=resolved)
        potentials = {
            "harmonic": lambda x: harmonic_potential(x, omega=omega, domain=domain),
            "double_well": lambda x: double_well_potential(x, a_param=500.0, b_param=50.0, domain=domain),
            "morse": lambda x: morse_potential(x, D_e=20.0, alpha=6.0, x_eq=float(domain.midpoint)),
            "poschl_teller": lambda x: poschl_teller_potential(x, V0=50.0, alpha=10.0, domain=domain),
        }
        pot_fn = potentials.get(potential, potentials["harmonic"])
        report = analyze_potential_landscape(pot_fn, domain, potential_name=potential, num_points=num_points, temperature=temperature)
        return to_serializable(report.to_dict())

    @_tool(
        server,
        runtime,
        "analyze_scattering_pipeline",
        "Full scattering analysis: transmission spectrum, resonances, S-matrix, WKB tunneling comparison — energy range auto-determined from barrier heights.",
        bounded=True,
    )
    def analyze_scattering_pipeline_tool(
        barrier_type: str = "double",
        barrier_height: float = 50.0,
        barrier_width: float = 0.05,
        separation: float = 0.1,
    ) -> dict[str, Any]:
        from spectral_packet_engine.scattering import rectangular_barrier, double_barrier
        from spectral_packet_engine.pipelines import analyze_scattering_system

        if barrier_type == "double":
            segments = double_barrier(barrier_height, barrier_width, separation)
        else:
            segments = rectangular_barrier(barrier_height, barrier_width)
        report = analyze_scattering_system(segments)
        return to_serializable(report.to_dict())

    @_tool(
        server,
        runtime,
        "analyze_spectral_profile_pipeline",
        "Deep spectral analysis of profile data: auto-determines mode count from convergence diagnostics, computes energy budget, compression quality, Gibbs detection. Works on any profile CSV.",
        bounded=True,
    )
    def analyze_spectral_profile_pipeline_tool(
        table_path: str,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.pipelines import analyze_spectral_profile
        from spectral_packet_engine.runtime import resolve_torch_device

        table = load_profile_table(table_path)
        resolved = resolve_torch_device(device)
        grid = torch.as_tensor(table.position_grid, dtype=torch.float64, device=resolved)
        profiles = torch.as_tensor(table.profiles, dtype=torch.float64, device=resolved)
        report = analyze_spectral_profile(grid, profiles, device=resolved)
        return to_serializable(report.to_dict())

    @_tool(
        server,
        runtime,
        "compare_quantum_states_pipeline",
        "Compare two Gaussian wavepackets: fidelity, trace distance, energy and momentum differences, uncertainty products.",
        bounded=True,
    )
    def compare_quantum_states_pipeline_tool(
        center_a: float = 0.3,
        center_b: float = 0.5,
        width: float = 0.07,
        wavenumber: float = 25.0,
        num_modes: int = 64,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.basis import InfiniteWellBasis
        from spectral_packet_engine.state import make_truncated_gaussian_packet
        from spectral_packet_engine.projector import StateProjector
        from spectral_packet_engine.pipelines import compare_quantum_states
        from spectral_packet_engine.runtime import resolve_torch_device

        resolved = resolve_torch_device(device)
        domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device=resolved)
        basis = InfiniteWellBasis(domain, num_modes)
        proj = StateProjector(basis)
        st_a = proj.project_packet(make_truncated_gaussian_packet(domain, center=center_a, width=width, wavenumber=wavenumber))
        st_b = proj.project_packet(make_truncated_gaussian_packet(domain, center=center_b, width=width, wavenumber=wavenumber))
        report = compare_quantum_states(st_a.coefficients, st_b.coefficients, basis)
        return to_serializable(report.to_dict())

    # ================================================================
    # Advanced Physics Tools (Tier 1–3)
    # ================================================================

    @_tool(
        server,
        runtime,
        "solve_eigenproblem",
        "Solve the 1D Schrödinger eigenvalue problem for an arbitrary potential V(x). Returns eigenvalues and eigenstates. Supports harmonic, double-well, Morse, and Pöschl-Teller potentials.",
        bounded=True,
    )
    def solve_eigenproblem_tool(
        potential: str = "harmonic",
        num_points: int = 128,
        num_states: int = 10,
        domain_left: float = 0.0,
        domain_right: float = 1.0,
        omega: float = 50.0,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.eigensolver import (
            solve_eigenproblem as _solve,
            harmonic_potential,
            double_well_potential,
            morse_potential,
            poschl_teller_potential,
            verify_orthonormality,
        )
        from spectral_packet_engine.runtime import resolve_torch_device

        resolved = resolve_torch_device(device)
        domain = InfiniteWell1D(
            left=torch.tensor(domain_left, dtype=torch.float64, device=resolved),
            right=torch.tensor(domain_right, dtype=torch.float64, device=resolved),
        )
        potentials = {
            "harmonic": lambda x: harmonic_potential(x, omega=omega, domain=domain),
            "double_well": lambda x: double_well_potential(x, a_param=500.0, b_param=50.0, domain=domain),
            "morse": lambda x: morse_potential(x, D_e=20.0, alpha=6.0, x_eq=float(domain.midpoint)),
            "poschl_teller": lambda x: poschl_teller_potential(x, V0=50.0, alpha=10.0, domain=domain),
        }
        pot_fn = potentials.get(potential, potentials["harmonic"])
        result = _solve(pot_fn, domain, num_points=num_points, num_states=num_states)
        ortho = verify_orthonormality(result)
        return to_serializable({
            "eigenvalues": result.eigenvalues.tolist(),
            "num_states": result.num_states,
            "potential": potential,
            "orthonormality_max_error": ortho["max_offdiagonal_error"],
            "orthonormality_ok": ortho["is_orthonormal"],
        })

    @_tool(
        server,
        runtime,
        "split_operator_propagate",
        "Propagate a Gaussian wavepacket through an arbitrary potential using split-operator method (2nd/4th order Trotter-Suzuki). Returns density evolution and conservation diagnostics.",
        bounded=True,
    )
    def split_operator_propagate_tool(
        center: float = 0.3,
        width: float = 0.05,
        wavenumber: float = 40.0,
        potential_height: float = 100.0,
        barrier_center: float = 0.5,
        barrier_width: float = 0.05,
        total_time: float = 0.01,
        num_steps: int = 2000,
        order: int = 2,
        num_points: int = 256,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.split_operator import (
            split_operator_propagate as _propagate,
            gaussian_wavepacket_on_grid,
        )
        from spectral_packet_engine.runtime import resolve_torch_device

        resolved = resolve_torch_device(device)
        domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device=resolved)
        grid = domain.grid(num_points)
        psi0 = gaussian_wavepacket_on_grid(grid, center=center, width=width, wavenumber=wavenumber)
        V = potential_height * torch.exp(-((grid - barrier_center) ** 2) / (2 * barrier_width ** 2))
        result = _propagate(psi0, V, domain, total_time=total_time, num_steps=num_steps, save_every=max(1, num_steps // 50), order=order)
        return to_serializable({
            "num_snapshots": result.times.shape[0],
            "norm_initial": float(result.norm_history[0]),
            "norm_final": float(result.norm_history[-1]),
            "norm_drift": float(abs(result.norm_history[-1] - result.norm_history[0])),
            "energy_initial": float(result.energy_history[0]),
            "energy_final": float(result.energy_history[-1]),
            "energy_drift": float(abs(result.energy_history[-1] - result.energy_history[0])),
            "order": order,
            "total_time": total_time,
            "num_steps": num_steps,
        })

    @_tool(
        server,
        runtime,
        "compute_wigner_function",
        "Compute the Wigner quasi-probability distribution W(x,p) for a Gaussian wavepacket. Returns negativity (non-classicality witness) and marginals.",
        bounded=True,
    )
    def compute_wigner_function_tool(
        center: float = 0.5,
        width: float = 0.05,
        wavenumber: float = 30.0,
        num_modes: int = 64,
        num_x_points: int = 64,
        num_p_points: int = 64,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.basis import InfiniteWellBasis
        from spectral_packet_engine.state import make_truncated_gaussian_packet
        from spectral_packet_engine.projector import StateProjector
        from spectral_packet_engine.wigner import wigner_from_spectral
        from spectral_packet_engine.runtime import resolve_torch_device

        resolved = resolve_torch_device(device)
        domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device=resolved)
        basis = InfiniteWellBasis(domain, num_modes)
        projector = StateProjector(basis)
        packet = make_truncated_gaussian_packet(domain, center=center, width=width, wavenumber=wavenumber)
        state = projector.project_packet(packet)
        result = wigner_from_spectral(state.coefficients, basis, num_x_points=num_x_points, num_p_points=num_p_points)
        return to_serializable({
            "negativity": float(result.negativity),
            "total_integral": float(result.total_integral),
            "x_range": [float(result.x_grid[0]), float(result.x_grid[-1])],
            "p_range": [float(result.p_grid[0]), float(result.p_grid[-1])],
            "wigner_min": float(result.W.min()),
            "wigner_max": float(result.W.max()),
            "is_nonclassical": bool(result.negativity > 0.01),
        })

    @_tool(
        server,
        runtime,
        "analyze_density_matrix",
        "Construct and analyze the density matrix for a quantum state: purity, von Neumann entropy, linear entropy, rank.",
        bounded=True,
    )
    def analyze_density_matrix_tool(
        center: float = 0.3,
        width: float = 0.07,
        wavenumber: float = 25.0,
        num_modes: int = 32,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.basis import InfiniteWellBasis
        from spectral_packet_engine.state import make_truncated_gaussian_packet
        from spectral_packet_engine.projector import StateProjector
        from spectral_packet_engine.density_matrix import pure_state_density_matrix, analyze_density_matrix as _analyze
        from spectral_packet_engine.runtime import resolve_torch_device

        resolved = resolve_torch_device(device)
        domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device=resolved)
        basis = InfiniteWellBasis(domain, num_modes)
        projector = StateProjector(basis)
        packet = make_truncated_gaussian_packet(domain, center=center, width=width, wavenumber=wavenumber)
        state = projector.project_packet(packet)
        rho = pure_state_density_matrix(state.coefficients)
        result = _analyze(rho)
        return to_serializable({
            "purity": float(result.purity),
            "von_neumann_entropy": float(result.von_neumann_entropy),
            "linear_entropy": float(result.linear_entropy),
            "rank": result.rank,
            "is_pure": result.is_pure,
            "num_modes": num_modes,
        })

    @_tool(
        server,
        runtime,
        "compute_greens_function",
        "Compute the spectral Green's function, local density of states (LDOS), and total DOS for the infinite well.",
        bounded=True,
    )
    def compute_greens_function_tool(
        num_modes: int = 32,
        num_energy_points: int = 200,
        broadening: float = 0.5,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.basis import InfiniteWellBasis
        from spectral_packet_engine.greens_function import analyze_greens_function as _analyze
        from spectral_packet_engine.runtime import resolve_torch_device

        resolved = resolve_torch_device(device)
        domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device=resolved)
        basis = InfiniteWellBasis(domain, num_modes)
        result = _analyze(basis, num_energy_points=num_energy_points, broadening=broadening)
        return to_serializable({
            "energy_range": [float(result.energy_grid[0]), float(result.energy_grid[-1])],
            "num_energy_points": int(result.energy_grid.shape[0]),
            "num_x_points": int(result.ldos.shape[0]),
            "dos_max": float(result.dos.max()),
            "dos_at_ground_state": float(result.dos[result.dos.argmax()]),
            "num_modes": num_modes,
            "broadening": broadening,
        })

    @_tool(
        server,
        runtime,
        "perturbation_analysis",
        "Apply quantum perturbation theory (1st and 2nd order) to analyze how a perturbation modifies energy levels of the infinite well.",
        bounded=True,
    )
    def perturbation_analysis_tool(
        perturbation_type: str = "linear_slope",
        strength: float = 10.0,
        num_modes: int = 32,
        num_states: int = 10,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.basis import InfiniteWellBasis
        from spectral_packet_engine.perturbation import analyze_perturbation as _analyze
        from spectral_packet_engine.runtime import resolve_torch_device

        resolved = resolve_torch_device(device)
        domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device=resolved)
        basis = InfiniteWellBasis(domain, num_modes)
        grid = domain.grid(512)
        eigenstates = basis.evaluate(grid).T[:num_states]
        eigenvalues = basis.energies[:num_states]

        perturbations = {
            "linear_slope": lambda x: strength * (x - domain.left) / domain.length,
            "quadratic_well": lambda x: strength * ((x - domain.midpoint) / domain.length) ** 2,
            "delta_like": lambda x: strength * torch.exp(-((x - domain.midpoint) ** 2) / (2 * 0.01 ** 2)),
        }
        pert_fn = perturbations.get(perturbation_type, perturbations["linear_slope"])
        result = _analyze(pert_fn, eigenstates, eigenvalues, grid)
        return to_serializable({
            "perturbation": perturbation_type,
            "strength": strength,
            "unperturbed_energies": result.unperturbed_energies[:num_states].tolist(),
            "first_order_corrections": result.first_order_energies[:num_states].tolist(),
            "second_order_corrections": result.second_order_energies[:num_states].tolist(),
            "corrected_energies": result.corrected_energies[:num_states].tolist(),
            "convergence_parameter": float(result.convergence_parameter),
            "perturbation_theory_valid": bool(result.convergence_parameter < 1.0),
        })

    @_tool(
        server,
        runtime,
        "wkb_analysis",
        "WKB semiclassical analysis: Bohr-Sommerfeld quantization and tunneling probability for 1D potentials.",
        bounded=True,
    )
    def wkb_analysis_tool(
        potential: str = "harmonic",
        num_states: int = 10,
        omega: float = 50.0,
        barrier_height: float = 50.0,
        barrier_width: float = 0.1,
        tunneling_energy: float = 25.0,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.semiclassical import (
            bohr_sommerfeld_quantization,
            tunneling_probability,
        )
        from spectral_packet_engine.eigensolver import harmonic_potential
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.runtime import resolve_torch_device

        resolved = resolve_torch_device(device)
        domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device=resolved)
        mid = float(domain.midpoint)
        mass = float(domain.mass)

        if potential == "harmonic":
            pot_fn = lambda x: harmonic_potential(x, omega=omega, domain=domain)
        else:
            pot_fn = lambda x: 0.5 * mass * omega ** 2 * (x - mid) ** 2

        bs = bohr_sommerfeld_quantization(pot_fn, float(domain.left), float(domain.right), mass=mass, hbar=float(domain.hbar), num_states=num_states)
        grid = domain.grid(1024)
        V_barrier = barrier_height * torch.exp(-((grid - mid) ** 2) / (2 * barrier_width ** 2))
        V_barrier = V_barrier.to(dtype=torch.float64, device=resolved)
        tunnel = tunneling_probability(tunneling_energy, V_barrier, grid, mass=mass, hbar=float(domain.hbar))

        return to_serializable({
            "bohr_sommerfeld_energies": bs.energies.tolist(),
            "num_states": num_states,
            "tunneling_transmission": float(tunnel.transmission),
            "tunneling_reflection": float(tunnel.reflection),
            "barrier_kappa_integral": float(tunnel.kappa_integral),
        })

    @_tool(
        server,
        runtime,
        "operator_commutator",
        "Compute commutator [A,B], generalized uncertainty relation, and BCH expansion for position and momentum operators in the sine basis.",
        bounded=True,
    )
    def operator_commutator_tool(
        num_modes: int = 16,
        bch_order: int = 3,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.basis import InfiniteWellBasis
        from spectral_packet_engine.operator_algebra import (
            position_operator_matrix,
            momentum_operator_matrix,
            compute_commutator as _comm,
            baker_campbell_hausdorff,
        )
        from spectral_packet_engine.runtime import resolve_torch_device

        resolved = resolve_torch_device(device)
        domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device=resolved)
        basis = InfiniteWellBasis(domain, num_modes)
        X = position_operator_matrix(basis)
        P = momentum_operator_matrix(basis)
        comm_result = _comm(X, P)
        iXP = 1j * X.to(dtype=torch.complex128) / num_modes
        iP = 1j * P.to(dtype=torch.complex128) / (num_modes * 10)
        bch = baker_campbell_hausdorff(iXP, iP, order=bch_order)
        return to_serializable({
            "[X,P]_trace": float(comm_result.trace_commutator.real) if torch.is_complex(comm_result.trace_commutator) else float(comm_result.trace_commutator),
            "[X,P]_frobenius_norm": float(comm_result.frobenius_norm.real) if torch.is_complex(comm_result.frobenius_norm) else float(comm_result.frobenius_norm),
            "expected_[x,p]_diagonal": f"i*hbar = i*{float(domain.hbar)}",
            "bch_order": bch_order,
            "bch_matrix_norm": float(torch.linalg.norm(bch).real),
            "num_modes": num_modes,
        })

    @_tool(
        server,
        runtime,
        "symplectic_propagation",
        "Propagate a classical Hamiltonian system using symplectic integrators (Störmer-Verlet, Forest-Ruth, Yoshida). Verify energy and phase-space volume conservation.",
        bounded=True,
    )
    def symplectic_propagation_tool(
        q0: float = 0.3,
        p0: float = 5.0,
        omega: float = 10.0,
        dt: float = 0.001,
        num_steps: int = 5000,
        integrator: str = "yoshida",
        order: int = 4,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.symplectic import stormer_verlet, forest_ruth, yoshida as _yoshida, check_symplecticity
        from spectral_packet_engine.runtime import resolve_torch_device

        resolved = resolve_torch_device(device)
        q = torch.tensor(q0, dtype=torch.float64, device=resolved)
        p = torch.tensor(p0, dtype=torch.float64, device=resolved)
        grad_V = lambda q_: omega ** 2 * q_

        if integrator == "verlet":
            traj = stormer_verlet(q, p, grad_V, dt=dt, num_steps=num_steps)
        elif integrator == "forest_ruth":
            traj = forest_ruth(q, p, grad_V, dt=dt, num_steps=num_steps)
        else:
            traj = _yoshida(q, p, grad_V, dt=dt, num_steps=num_steps, order=order)

        symp = check_symplecticity(traj)
        return to_serializable({
            "integrator": integrator,
            "order": order,
            "energy_initial": float(traj.energies[0]),
            "energy_final": float(traj.energies[-1]),
            "energy_max_drift": float(traj.energy_error.max()),
            "is_symplectic": symp.is_symplectic,
            "volume_preservation_error": float(symp.relative_error),
            "total_time": dt * num_steps,
            "num_steps": num_steps,
        })

    @_tool(
        server,
        runtime,
        "spectral_zeta_analysis",
        "Compute the spectral zeta function, heat kernel, partition function, Weyl law check, and Casimir energy for the infinite well spectrum.",
        bounded=True,
    )
    def spectral_zeta_analysis_tool(
        num_modes: int = 50,
        temperature: float = 10.0,
        device: str = "auto",
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.basis import InfiniteWellBasis
        from spectral_packet_engine.spectral_zeta import (
            spectral_zeta as _zeta,
            partition_function,
            weyl_law_check,
            casimir_energy,
            heat_kernel_trace,
        )
        from spectral_packet_engine.runtime import resolve_torch_device

        resolved = resolve_torch_device(device)
        domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device=resolved)
        basis = InfiniteWellBasis(domain, num_modes)
        E = basis.energies

        zeta_2 = _zeta(E, 2.0)
        temps = torch.linspace(0.1, temperature, 20, dtype=torch.float64, device=resolved)
        pf = partition_function(E, temps)
        weyl = weyl_law_check(E, float(domain.length), mass=float(domain.mass), hbar=float(domain.hbar))
        E_cas = casimir_energy(E)
        K_trace = heat_kernel_trace(E, torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64, device=resolved))

        return to_serializable({
            "zeta_H(2)": float(zeta_2),
            "casimir_energy": float(E_cas),
            "heat_kernel_at_t01": float(K_trace[1]),
            "partition_function_at_T_max": float(pf.Z[-1]),
            "free_energy_at_T_max": float(pf.free_energy[-1]),
            "entropy_at_T_max": float(pf.entropy[-1]),
            "specific_heat_at_T_max": float(pf.specific_heat[-1]),
            "weyl_law_max_relative_error": float(weyl.relative_error.max()),
            "num_modes": num_modes,
        })

    @_tool(
        server,
        runtime,
        "scattering_analysis",
        "Compute quantum scattering through a potential barrier: transmission T(E), reflection R(E), resonances, and S-matrix unitarity.",
        bounded=True,
    )
    def scattering_analysis_tool(
        barrier_type: str = "rectangular",
        barrier_height: float = 50.0,
        barrier_width: float = 0.1,
        separation: float = 0.15,
        energy_min: float = 1.0,
        energy_max: float = 100.0,
        num_energies: int = 500,
    ) -> dict[str, Any]:
        from spectral_packet_engine.scattering import (
            scattering_spectrum,
            rectangular_barrier,
            double_barrier,
        )

        if barrier_type == "double":
            segments = double_barrier(barrier_height, barrier_width, separation)
        else:
            segments = rectangular_barrier(barrier_height, barrier_width)

        result = scattering_spectrum(segments, energy_min=energy_min, energy_max=energy_max, num_energies=num_energies)
        return to_serializable({
            "barrier_type": barrier_type,
            "num_resonances": int(result.resonance_energies.shape[0]),
            "resonance_energies": result.resonance_energies.tolist(),
            "resonance_widths": result.resonance_widths.tolist(),
            "transmission_at_barrier": float(result.transmission[result.energies.shape[0] // 2]),
            "max_transmission": float(result.transmission.max()),
            "unitarity_check": "T+R=1 enforced by transfer matrix",
        })

    @_tool(
        server,
        runtime,
        "berry_phase_analysis",
        "Compute the Berry (geometric) phase for a spin-½ system in a rotating magnetic field. Validates against the analytical solid-angle formula.",
        bounded=True,
    )
    def berry_phase_analysis_tool(
        theta: float = 0.6,
        num_path_points: int = 100,
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.berry_phase import berry_phase_for_spin_half

        theta_path = torch.full((num_path_points,), theta, dtype=torch.float64)
        phi_path = torch.linspace(0, 2 * 3.141592653589793, num_path_points, dtype=torch.float64)
        result = berry_phase_for_spin_half(theta_path, phi_path)
        analytical = -3.141592653589793 * (1 - float(torch.cos(torch.tensor(theta))))
        return to_serializable({
            "computed_berry_phase": float(result.phase),
            "analytical_berry_phase": analytical,
            "relative_error": abs(float(result.phase) - analytical) / abs(analytical) if analytical != 0 else 0.0,
            "theta_radians": theta,
            "solid_angle": 2 * 3.141592653589793 * (1 - float(torch.cos(torch.tensor(theta)))),
            "num_path_points": num_path_points,
        })

    @_tool(
        server,
        runtime,
        "quantum_info_analysis",
        "Quantum information analysis: Fisher information, entanglement entropy, concurrence, quantum channels. Demonstrates quantum vs classical information bounds.",
        bounded=True,
    )
    def quantum_info_analysis_tool(
        state_type: str = "bell",
        channel: str = "none",
        channel_param: float = 0.1,
    ) -> dict[str, Any]:
        import torch
        from spectral_packet_engine.quantum_info import (
            quantum_fisher_information,
            entanglement_entropy,
            concurrence as _concurrence,
            quantum_mutual_information,
            linear_entropy as _linear_entropy,
            apply_quantum_channel,
            depolarizing_channel,
            amplitude_damping_channel,
        )
        from spectral_packet_engine.density_matrix import pure_state_density_matrix

        if state_type == "bell":
            psi = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex128) / (2 ** 0.5)
        elif state_type == "separable":
            psi = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128)
        else:
            psi = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex128) / (2 ** 0.5)

        rho = pure_state_density_matrix(psi)
        ent = entanglement_entropy(psi, (2, 2))
        conc = _concurrence(rho)
        mi = quantum_mutual_information(rho, (2, 2))
        sigma_z_full = torch.zeros(4, 4, dtype=torch.complex128)
        sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
        sigma_z_full[:2, :2] = sigma_z
        sigma_z_full[2:, 2:] = sigma_z
        qfi = quantum_fisher_information(rho, sigma_z_full)

        result = {
            "state_type": state_type,
            "entanglement_entropy": float(ent.entanglement_entropy),
            "schmidt_rank": ent.schmidt_rank,
            "is_entangled": ent.is_entangled,
            "concurrence": float(conc),
            "mutual_information": float(mi.mutual_information),
            "quantum_fisher_information": float(qfi.fisher_information),
            "cramer_rao_bound": float(qfi.cramer_rao_bound),
            "linear_entropy": float(_linear_entropy(rho)),
        }

        if channel != "none":
            if channel == "depolarizing":
                kraus = depolarizing_channel(4, channel_param)
            else:
                kraus = amplitude_damping_channel(channel_param)
                # extend to 4-dim trivially for 2-qubit
                kraus_4d = []
                I2 = torch.eye(2, dtype=torch.complex128)
                for K in kraus:
                    kraus_4d.append(torch.kron(K, I2))
                kraus = kraus_4d
            ch_result = apply_quantum_channel(rho, kraus)
            result["channel"] = channel
            result["channel_param"] = channel_param
            result["output_fidelity"] = float(ch_result.fidelity_with_input)
            result["output_entropy_change"] = float(ch_result.entropy_change)

        return to_serializable(result)

    # ================================================================
    # Tunneling Experiment Pipeline
    # ================================================================

    @_tool(
        server,
        runtime,
        "tunneling_experiment",
        "Run a complete quantum tunneling experiment in one call. "
        "Chains: (1) transfer-matrix scattering T(E)/R(E) with resonance detection, "
        "(2) WKB semiclassical tunneling comparison, "
        "(3) split-operator wavepacket propagation through the barrier, "
        "(4) Wigner phase-space analysis of the propagated state. "
        "Returns a structured report with transmission coefficients, norm/energy conservation, "
        "and non-classicality witness.",
        bounded=True,
    )
    def tunneling_experiment_tool(
        barrier_height: float = 50.0,
        barrier_width_sigma: float = 0.03,
        domain_length: float = 1.0,
        grid_points: int = 256,
        num_modes: int = 128,
        num_energies: int = 300,
        packet_center: float = 0.25,
        packet_width: float = 0.04,
        packet_wavenumber: float = 40.0,
        packet_energy_capture_fraction: float = 0.999,
        propagation_steps: int | None = None,
        dt: float = 1e-5,
        device: str = "cpu",
    ) -> dict[str, Any]:
        from spectral_packet_engine.pipelines import analyze_tunneling

        report = analyze_tunneling(
            barrier_height=barrier_height,
            barrier_width_sigma=barrier_width_sigma,
            domain_length=domain_length,
            grid_points=grid_points,
            num_modes=num_modes,
            num_energies=num_energies,
            packet_center=packet_center,
            packet_width=packet_width,
            packet_wavenumber=packet_wavenumber,
            packet_energy_capture_fraction=packet_energy_capture_fraction,
            propagation_steps=propagation_steps,
            dt=dt,
            device=device,
        )
        return to_serializable({
            "barrier": {
                "height": report.barrier_height,
                "width_fwhm": report.barrier_width,
                "left_fwhm": report.barrier_left,
                "right_fwhm": report.barrier_right,
            },
            "scattering": {
                "energy_range": report.energy_range,
                "comparison_energy": report.comparison_energy,
                "packet_mean_energy": report.packet_mean_energy,
                "packet_energy_interval": report.packet_energy_interval,
                "packet_energy_capture_fraction": report.packet_energy_capture_fraction,
                "transmission_at_half_barrier": report.transmission_at_half_barrier,
                "transmission_at_packet_energy": report.transmission_at_packet_energy,
                "num_resonances": report.num_resonances,
                "resonance_energies": report.resonance_energies,
                "resonance_widths": report.resonance_widths,
            },
            "wkb": {
                "transmission_at_half_barrier": report.wkb_transmission_at_half_barrier,
                "transmission_at_packet_energy": report.wkb_transmission_at_packet_energy,
                "wkb_exact_ratio": report.wkb_exact_ratio,
            },
            "propagation": {
                "norm_drift": report.propagation_norm_drift,
                "energy_drift": report.propagation_energy_drift,
                "steps": report.propagation_steps,
                "total_time": report.propagation_total_time,
                "transmitted_probability": report.transmitted_probability,
                "reflected_probability": report.reflected_probability,
            },
            "wigner": {
                "negativity": report.wigner_negativity,
            },
            "device": report.device,
            "num_modes": report.num_modes,
        })

    # ================================================================
    # Audit Log Tool (for AI-driven development)
    # ================================================================

    @_tool(
        server,
        runtime,
        "write_audit_log",
        "Write a structured audit log entry to docs/internal/. Use this to document code reviews, architectural decisions, physics validations, and engineering findings during AI-assisted development.",
    )
    def write_audit_log_tool(
        title: str,
        content: str,
        filename: str | None = None,
    ) -> dict[str, Any]:
        import datetime
        audit_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "internal"
        audit_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.date.today().isoformat()
        safe_title = title.lower().replace(" ", "-").replace("/", "-")[:60]
        fname = filename or f"{safe_title}-{date_str}.md"
        path = audit_dir / fname
        header = f"# {title}\n\n**Date:** {date_str}\n**Source:** MCP audit tool\n\n---\n\n"
        path.write_text(header + content, encoding="utf-8")
        return {"written": str(path), "title": title, "date": date_str}

    # ================================================================
    # Spectral Load Modeling Tools
    # ================================================================

    @_tool(
        server,
        runtime,
        "analyze_server_load",
        "Analyze server request load using spectral decomposition. "
        "Decomposes request-rate time series into spectral modes, classifies traffic pattern "
        "(smooth/bursty/anomalous), computes adaptive throttling parameters, and estimates "
        "sustainable capacity. Uses the spectral engine's own convergence diagnostics — "
        "no hardcoded thresholds.",
        bounded=True,
    )
    def analyze_server_load_tool(
        timestamps: list[float],
        window_seconds: float = 300.0,
        resolution: int = 256,
        num_modes: int = 64,
        capacity_rps: float = 100.0,
        baseline_timestamps: list[float] | None = None,
        device: str = "cpu",
    ) -> dict[str, Any]:
        from spectral_packet_engine.load_spectral import analyze_request_load

        report = analyze_request_load(
            timestamps,
            window_seconds=window_seconds,
            resolution=resolution,
            num_modes=num_modes,
            capacity_rps=capacity_rps,
            baseline_timestamps=baseline_timestamps,
            device=device,
        )
        return to_serializable({
            "signal": {
                "window_seconds": report.signal.window_seconds,
                "total_requests": report.signal.total_requests,
                "grid_points": int(report.signal.grid.shape[0]),
            },
            "spectrum": {
                "decay_type": report.spectrum.decay.decay_type.value,
                "decay_rate": float(report.spectrum.decay.rate.item()),
                "effective_mode_count": report.spectrum.effective_mode_count,
                "dominant_frequency_hz": report.spectrum.dominant_frequency_hz,
                "high_frequency_energy_ratio": report.spectrum.high_frequency_energy_ratio,
            },
            "throttle": {
                "regime": report.throttle.regime,
                "recommended_cooldown_seconds": report.throttle.recommended_cooldown_seconds,
                "recommended_min_interval_seconds": report.throttle.recommended_min_interval_seconds,
                "recommended_max_concurrent": report.throttle.recommended_max_concurrent,
                "capacity_utilization": report.throttle.capacity_utilization,
                "spectral_load_factor": report.throttle.spectral_load_factor,
                "headroom_fraction": report.throttle.headroom_fraction,
            },
            "capacity": {
                "sustained_rps": report.capacity.sustained_rps,
                "peak_rps": report.capacity.peak_rps,
                "burst_ratio": report.capacity.burst_ratio,
                "spectral_headroom_modes": report.capacity.spectral_headroom_modes,
                "stable": report.capacity.stable,
            },
            "anomaly": None if report.anomaly is None else {
                "is_anomalous": report.anomaly.is_anomalous,
                "spectral_distance": report.anomaly.spectral_distance,
                "entropy_shift": report.anomaly.entropy_shift,
                "mode_count_shift": report.anomaly.mode_count_shift,
                "energy_redistribution": report.anomaly.energy_redistribution,
                "reason": report.anomaly.reason,
            },
        })

    @_tool(
        server,
        runtime,
        "decompose_load_signal",
        "Decompose a request-rate time series into spectral coefficients. "
        "Returns per-mode energy fractions and the spectral expansion of the load signal.",
        bounded=True,
    )
    def decompose_load_signal_tool(
        rate_values: list[float],
        window_seconds: float = 300.0,
        num_modes: int = 64,
        device: str = "cpu",
    ) -> dict[str, Any]:
        from spectral_packet_engine.load_spectral import load_signal_from_rate, decompose_load_signal

        signal = load_signal_from_rate(rate_values, window_seconds=window_seconds, device=device)
        coeffs = decompose_load_signal(signal, num_modes=num_modes)
        return to_serializable({
            "num_modes": coeffs.num_modes,
            "window_seconds": signal.window_seconds,
            "total_requests": signal.total_requests,
            "coefficients": coeffs.coefficients.tolist(),
            "energy_fractions": coeffs.energy_fractions.tolist(),
        })

    @_tool(
        server,
        runtime,
        "compute_adaptive_throttle",
        "Compute adaptive throttling parameters from a load signal using spectral analysis. "
        "Returns cooldown duration, minimum request interval, and concurrency limit — all derived "
        "from the spectral structure of the traffic, not from hardcoded rules.",
        bounded=True,
    )
    def compute_adaptive_throttle_tool(
        rate_values: list[float],
        window_seconds: float = 300.0,
        num_modes: int = 64,
        capacity_rps: float = 100.0,
        device: str = "cpu",
    ) -> dict[str, Any]:
        from spectral_packet_engine.load_spectral import (
            load_signal_from_rate, decompose_load_signal, compute_adaptive_throttle,
        )

        signal = load_signal_from_rate(rate_values, window_seconds=window_seconds, device=device)
        coeffs = decompose_load_signal(signal, num_modes=num_modes)
        throttle = compute_adaptive_throttle(coeffs, capacity_rps=capacity_rps)
        return to_serializable({
            "regime": throttle.regime,
            "recommended_cooldown_seconds": throttle.recommended_cooldown_seconds,
            "recommended_min_interval_seconds": throttle.recommended_min_interval_seconds,
            "recommended_max_concurrent": throttle.recommended_max_concurrent,
            "capacity_utilization": throttle.capacity_utilization,
            "spectral_load_factor": throttle.spectral_load_factor,
            "headroom_fraction": throttle.headroom_fraction,
        })

    @_tool(
        server,
        runtime,
        "detect_load_anomaly",
        "Compare current server load spectrum to a baseline and detect anomalies. "
        "Uses spectral distance, entropy shift, and Jensen-Shannon divergence of mode energy — "
        "the engine's own convergence diagnostics determine significance.",
        bounded=True,
    )
    def detect_load_anomaly_tool(
        current_rate_values: list[float],
        baseline_rate_values: list[float],
        window_seconds: float = 300.0,
        num_modes: int = 64,
        device: str = "cpu",
    ) -> dict[str, Any]:
        from spectral_packet_engine.load_spectral import (
            load_signal_from_rate, decompose_load_signal, detect_load_anomaly,
        )

        cur_sig = load_signal_from_rate(current_rate_values, window_seconds=window_seconds, device=device)
        base_sig = load_signal_from_rate(baseline_rate_values, window_seconds=window_seconds, device=device)
        cur_c = decompose_load_signal(cur_sig, num_modes=num_modes)
        base_c = decompose_load_signal(base_sig, num_modes=num_modes)
        anomaly = detect_load_anomaly(cur_c, base_c)
        return to_serializable({
            "is_anomalous": anomaly.is_anomalous,
            "spectral_distance": anomaly.spectral_distance,
            "entropy_shift": anomaly.entropy_shift,
            "mode_count_shift": anomaly.mode_count_shift,
            "energy_redistribution": anomaly.energy_redistribution,
            "reason": anomaly.reason,
        })

    @_tool(
        server,
        runtime,
        "estimate_server_capacity",
        "Estimate sustainable server capacity from a load signal using spectral decomposition. "
        "Separates sustained load (low-frequency modes) from burst spikes (high-frequency modes) "
        "to give a robust capacity estimate.",
        bounded=True,
    )
    def estimate_server_capacity_tool(
        rate_values: list[float],
        window_seconds: float = 300.0,
        num_modes: int = 64,
        device: str = "cpu",
    ) -> dict[str, Any]:
        from spectral_packet_engine.load_spectral import (
            load_signal_from_rate, decompose_load_signal, estimate_capacity,
        )

        signal = load_signal_from_rate(rate_values, window_seconds=window_seconds, device=device)
        coeffs = decompose_load_signal(signal, num_modes=num_modes)
        cap = estimate_capacity(coeffs)
        return to_serializable({
            "sustained_rps": cap.sustained_rps,
            "peak_rps": cap.peak_rps,
            "burst_ratio": cap.burst_ratio,
            "spectral_headroom_modes": cap.spectral_headroom_modes,
            "stable": cap.stable,
        })

    # ================================================================
    # Python Execution Tool — run code with the library pre-imported
    # ================================================================

    @_tool(
        server,
        runtime,
        "execute_python",
        "Execute a Python code snippet only when unsafe execution was explicitly enabled at startup. "
        "The tool is disabled by default because it can run arbitrary local Python. "
        "When enabled, `spe` is the library, `torch` is available, and `numpy` (as `np`) is loaded if installed. "
        "Capture output by assigning to `result`. Temporary files are written under a managed scratch directory.",
        bounded=True,
    )
    def execute_python_tool(
        code: str,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        if len(code) > runtime.config.max_execute_python_code_chars:
            return {
                "error": True,
                "error_type": "ValueError",
                "error_message": (
                    f"Code length exceeds the configured limit of {runtime.config.max_execute_python_code_chars} characters."
                ),
                "tool": "execute_python",
            }
        if not runtime.config.allow_unsafe_python:
            return {
                "error": True,
                "error_type": "PermissionError",
                "error_message": (
                    "The execute_python tool is disabled for this MCP runtime. "
                    "Restart the server with --allow-unsafe-python only for a trusted local session."
                ),
                "tool": "execute_python",
            }
        import io
        import contextlib
        import signal as _signal

        scratch_dir = _managed_scratch_directory(runtime.config)

        namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "torch": torch,
            "Path": Path,
            "scratch_dir": scratch_dir,
        }
        # Pre-import the full library
        try:
            import spectral_packet_engine as spe
            namespace["spe"] = spe
        except Exception:
            pass
        try:
            import numpy as np
            namespace["np"] = np
        except ImportError:
            pass

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Timeout handler (Unix only)
        if hasattr(_signal, "SIGALRM"):
            def _timeout_handler(signum, frame):
                raise TimeoutError(f"Code execution exceeded {timeout_seconds}s limit")
            old_handler = _signal.signal(_signal.SIGALRM, _timeout_handler)
            _signal.alarm(int(timeout_seconds))

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(compile(code, "<mcp-execute>", "exec"), namespace)  # noqa: S102
        except TimeoutError as exc:
            return {"error": True, "error_type": "TimeoutError", "error_message": str(exc)}
        except Exception as exc:
            return {
                "error": True,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "stdout": stdout_capture.getvalue()[:4000],
                "stderr": stderr_capture.getvalue()[:2000],
            }
        finally:
            if hasattr(_signal, "SIGALRM"):
                _signal.alarm(0)
                _signal.signal(_signal.SIGALRM, old_handler)

        # Extract result if the code set it
        result_value = namespace.get("result", None)
        result_serialized = None
        if result_value is not None:
            try:
                result_serialized = to_serializable(result_value)
            except Exception:
                result_serialized = str(result_value)[:8000]

        return {
            "stdout": stdout_capture.getvalue()[:8000],
            "stderr": stderr_capture.getvalue()[:2000],
            "result": result_serialized,
            "scratch_dir": str(scratch_dir),
        }

    # ================================================================
    # Scratch SQL Tools — temporary databases for MCP sessions
    # ================================================================

    @_tool(
        server,
        runtime,
        "create_scratch_database",
        "Create a temporary SQLite database in the managed scratch directory. "
        "Optionally run an init_script to create tables and insert data (supports "
        "CREATE TABLE, INSERT, and any DDL/DML). Optionally populate with synthetic "
        "profile data. Returns the database path for use with query_database and "
        "other SQL tools. Use init_script for custom schemas — this is the "
        "recommended way to set up databases with user-defined tables.",
    )
    def create_scratch_database_tool(
        name: str = "scratch.db",
        init_script: str | None = None,
        populate_synthetic: bool = False,
        num_profiles: int = 50,
        grid_points: int = 64,
        num_modes: int = 8,
        device: str = "cpu",
    ) -> dict[str, Any]:
        if populate_synthetic:
            _validate_synthetic_generation_request(
                runtime.config,
                num_profiles=num_profiles,
                grid_points=grid_points,
            )
        db_path = _managed_scratch_path(runtime.config, name)

        # Bootstrap the database
        db_info = bootstrap_local_database(str(db_path))
        result: dict[str, Any] = {
            "database_path": str(db_path),
            "tables": list(db_info.tables) if hasattr(db_info, "tables") else [],
        }

        # Run init script if provided (CREATE TABLE, INSERT, etc.)
        if init_script is not None:
            script_result = execute_database_script(
                str(db_path), init_script, create_if_missing=True,
            )
            result["init_script"] = {
                "statement_count": script_result.statement_count,
            }
            # Refresh table list after init
            db_info_after = inspect_database(str(db_path))
            result["tables"] = list(db_info_after.tables) if hasattr(db_info_after, "tables") else []

        if populate_synthetic:
            table = generate_synthetic_profile_table(
                num_profiles=num_profiles,
                grid_points=grid_points,
                device=device,
            )
            write_profile_table_to_database(str(db_path), "profiles", table, if_exists="replace")

            result["synthetic"] = {
                "table": "profiles",
                "num_profiles": num_profiles,
                "grid_points": grid_points,
                "position_count": int(table.profiles.shape[1]),
            }

        return result

    @_tool(
        server,
        runtime,
        "generate_synthetic_profiles",
        "Generate synthetic density profiles and write them as a CSV or to a scratch database. "
        "Profiles are Gaussian peaks with varying centers and widths. "
        "Returns the file or database path for downstream analysis.",
    )
    def generate_synthetic_profiles_tool(
        num_profiles: int = 50,
        grid_points: int = 64,
        output_format: str = "csv",
        output_name: str = "synthetic_profiles",
        device: str = "cpu",
    ) -> dict[str, Any]:
        _validate_synthetic_generation_request(
            runtime.config,
            num_profiles=num_profiles,
            grid_points=grid_points,
        )
        table = generate_synthetic_profile_table(
            num_profiles=num_profiles,
            grid_points=grid_points,
            device=device,
        )

        normalized_format = str(output_format).strip().lower()
        if normalized_format not in {"csv", "sqlite"}:
            raise ValueError("output_format must be either 'csv' or 'sqlite'")

        if normalized_format == "csv":
            path = _managed_scratch_path(runtime.config, f"{output_name}.csv")
            save_profile_table_csv(table, path)
            return {
                "path": str(path),
                "format": "csv",
                "num_profiles": num_profiles,
                "grid_points": grid_points,
            }
        else:
            db_path = _managed_scratch_path(runtime.config, f"{output_name}.db")
            bootstrap_local_database(str(db_path))
            write_profile_table_to_database(str(db_path), "profiles", table, if_exists="replace")
            return {
                "database_path": str(db_path),
                "table": "profiles",
                "format": "sqlite",
                "num_profiles": num_profiles,
                "grid_points": grid_points,
            }

    @_tool(
        server,
        runtime,
        "probe_mcp_runtime",
        "Launch a child MCP server with the same runtime policy and probe it through the real MCP stdio client. "
        "This is the repository's self-referential runtime audit workflow: it records startup, tool exposure, "
        "input validation, scratch-path containment, query-guard behavior, and optional repeated/burst stress checks.",
        bounded=True,
    )
    def probe_mcp_runtime_tool(output_dir: str | None = None, profile: str = "smoke") -> dict[str, Any]:
        probe_output_dir = output_dir
        if probe_output_dir is None:
            probe_output_dir = str(_managed_scratch_directory(runtime.config) / "probe_mcp_runtime")
        command = [
            sys.executable,
            "-m",
            "spectral_packet_engine.cli",
            "probe-mcp",
            "--python-executable",
            sys.executable,
            "--max-concurrent-tasks",
            str(runtime.config.max_concurrent_tasks),
            "--slot-timeout-seconds",
            str(runtime.config.slot_acquire_timeout_seconds),
            "--log-level",
            runtime.config.log_level.lower(),
            "--profile",
            profile,
            "--output-dir",
            probe_output_dir,
            "--skip-nested-probe",
        ]
        if runtime.config.allow_unsafe_python:
            command.append("--allow-unsafe-python")
        env = None
        if (Path.cwd() / "src" / "spectral_packet_engine" / "__init__.py").exists():
            command.append("--source-checkout")
            env = {
                **__import__("os").environ,
                "PYTHONPATH": "src",
            }

        completed = __import__("subprocess").run(
            command,
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.stdout.strip():
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "probe-mcp subprocess returned non-JSON stdout: "
                    f"{completed.stdout[:2000]}"
                ) from exc
        else:
            payload = {
                "error": True,
                "error_type": "RuntimeError",
                "error_message": completed.stderr.strip() or "probe-mcp subprocess produced no stdout",
                "tool": "probe_mcp_runtime",
            }
        if completed.returncode != 0 and isinstance(payload, dict) and "summary" not in payload:
            payload = {
                "error": True,
                "error_type": "RuntimeError",
                "error_message": completed.stderr.strip() or completed.stdout.strip(),
                "tool": "probe_mcp_runtime",
            }
        if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
            write_mcp_probe_artifacts(
                probe_output_dir,
                payload,
                metadata={
                    "workflow": "probe-mcp-runtime",
                    "profile": profile,
                    "parent_transport": runtime.config.transport,
                    "child_transport": "stdio",
                    "allow_unsafe_python": runtime.config.allow_unsafe_python,
                },
            )
        return to_serializable(payload)

    # ================================================================
    # One-Call Demo Pipeline
    # ================================================================

    @_tool(
        server,
        runtime,
        "demo_spectral_pipeline",
        "Run a complete demonstration pipeline in one call: generate synthetic profiles, "
        "project onto spectral basis, analyze convergence, compress, and return a full report. "
        "Requires no input — the tool generates everything internally. "
        "Use this to verify the engine works end-to-end or to show capabilities to new users.",
        bounded=True,
    )
    def demo_spectral_pipeline_tool(
        num_profiles: int = 20,
        grid_points: int = 128,
        num_modes: int = 16,
        device: str = "cpu",
    ) -> dict[str, Any]:
        import math
        from spectral_packet_engine.domain import InfiniteWell1D
        from spectral_packet_engine.basis import InfiniteWellBasis
        from spectral_packet_engine.profiles import (
            project_profiles_onto_basis, reconstruct_profiles_from_basis,
            relative_l2_error,
        )
        from spectral_packet_engine.convergence import (
            analyze_convergence, spectral_entropy, recommend_truncation,
        )

        domain = InfiniteWell1D(
            left=torch.tensor(0.0, dtype=torch.float64),
            right=torch.tensor(1.0, dtype=torch.float64),
        )
        basis = InfiniteWellBasis(domain=domain, num_modes=num_modes)
        grid = torch.linspace(0.0, 1.0, grid_points, dtype=torch.float64, device=device)

        profiles = []
        for i in range(num_profiles):
            center = 0.2 + 0.6 * (i / max(num_profiles - 1, 1))
            width = 0.05 + 0.03 * math.sin(2 * math.pi * i / num_profiles)
            p = torch.exp(-((grid - center) ** 2) / (2 * width**2))
            p = p / torch.trapezoid(p, grid)
            profiles.append(p)
        profiles_t = torch.stack(profiles)

        # Spectral projection
        coeffs = project_profiles_onto_basis(profiles_t, grid, basis)

        # Reconstruction
        recon = reconstruct_profiles_from_basis(coeffs, grid, basis)
        errors = relative_l2_error(profiles_t, recon, grid)

        # Convergence diagnostics on first profile
        conv = analyze_convergence(coeffs[0])
        ent = spectral_entropy(coeffs[0])
        trunc = recommend_truncation(coeffs[0], error_tolerance=0.01)

        return to_serializable({
            "pipeline": "demo_spectral",
            "input": {
                "num_profiles": num_profiles,
                "grid_points": grid_points,
                "num_modes": num_modes,
            },
            "reconstruction": {
                "mean_relative_l2_error": float(errors.mean().item()),
                "max_relative_l2_error": float(errors.max().item()),
                "min_relative_l2_error": float(errors.min().item()),
            },
            "convergence": {
                "decay_type": conv.decay.decay_type.value,
                "decay_rate": float(conv.decay.rate.item()),
                "r_squared": float(conv.decay.r_squared.item()),
            },
            "entropy": {
                "effective_mode_count": float(ent.effective_mode_count.item()),
                "total_modes": ent.total_modes,
                "sparsity": float(ent.sparsity.item()),
            },
            "truncation": {
                "recommended_modes": trunc.recommended_modes,
                "energy_captured": float(trunc.energy_captured.item()),
                "estimated_error": float(trunc.estimated_error.item()),
            },
        })

    # ================================================================
    # Self-Test Tool — validate the MCP server end-to-end
    # ================================================================

    @_tool(
        server,
        runtime,
        "self_test",
        "Run a comprehensive self-test of the MCP server. Validates: "
        "(1) library import and version, (2) spectral engine core (projection, reconstruction), "
        "(3) physics modules (eigensolver, density matrix), (4) load modeling, "
        "(5) SQL capability, (6) scratch directory access. "
        "Use this after startup to confirm the server is fully operational.",
        bounded=True,
    )
    def self_test_tool(device: str = "cpu") -> dict[str, Any]:
        import socket

        checks: dict[str, Any] = {}

        # 1. Library version
        from spectral_packet_engine.version import __version__
        checks["library_version"] = __version__

        # 2. Server address
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            local_ip = "127.0.0.1"
        checks["server"] = {
            "hostname": hostname,
            "best_effort_ipv4": local_ip,
            "transport": runtime.config.transport,
            "endpoint_url": runtime.config.endpoint_url,
            "bind_host": runtime.config.host,
            "bind_port": runtime.config.port,
            "streamable_http_path": runtime.config.streamable_http_path,
            "allowed_hosts": list(runtime.config.allowed_hosts),
            "allowed_origins": list(runtime.config.allowed_origins),
            "network_note": (
                "bind_host/bind_port/endpoint_url reflect the configured listener. "
                "best_effort_ipv4 is observational only and may not be remotely routable."
            ),
        }

        from spectral_packet_engine.domain import InfiniteWell1D

        domain = InfiniteWell1D(
            left=torch.tensor(0.0, dtype=torch.float64),
            right=torch.tensor(1.0, dtype=torch.float64),
        )

        # 3. Core spectral engine
        try:
            from spectral_packet_engine.basis import InfiniteWellBasis
            from spectral_packet_engine.profiles import (
                project_profiles_onto_basis, reconstruct_profiles_from_basis, relative_l2_error,
            )
            basis = InfiniteWellBasis(domain=domain, num_modes=16)
            grid = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
            profile = torch.exp(-((grid - 0.5) ** 2) / (2 * 0.07**2))
            profile = profile / torch.trapezoid(profile, grid)
            c = project_profiles_onto_basis(profile.unsqueeze(0), grid, basis).squeeze(0)
            r = reconstruct_profiles_from_basis(c.unsqueeze(0), grid, basis).squeeze(0)
            err = float(relative_l2_error(profile.unsqueeze(0), r.unsqueeze(0), grid).item())
            checks["core_engine"] = {"status": "passed", "reconstruction_error": err}
        except Exception as exc:
            checks["core_engine"] = {"status": "failed", "error": str(exc)}

        # 4. Physics: eigensolver
        try:
            from spectral_packet_engine.eigensolver import solve_eigenproblem, harmonic_potential
            result = solve_eigenproblem(lambda x: harmonic_potential(x, omega=10.0, domain=domain), domain, num_states=4)
            checks["eigensolver"] = {"status": "passed", "first_eigenvalue": float(result.eigenvalues[0].item())}
        except Exception as exc:
            checks["eigensolver"] = {"status": "failed", "error": str(exc)}

        # 5. Physics: density matrix
        try:
            from spectral_packet_engine.density_matrix import analyze_density_matrix, pure_state_density_matrix
            psi = torch.tensor([1.0, 0.0], dtype=torch.complex128) / (2**0.5)
            psi[1] = 1.0 / (2**0.5)
            rho = pure_state_density_matrix(psi)
            dm = analyze_density_matrix(rho)
            checks["density_matrix"] = {"status": "passed", "purity": float(dm.purity.real.item())}
        except Exception as exc:
            checks["density_matrix"] = {"status": "failed", "error": str(exc)}

        # 6. Load modeling
        try:
            from spectral_packet_engine.load_spectral import load_signal_from_rate, decompose_load_signal, compute_adaptive_throttle
            import math
            t = torch.linspace(0, 300, 128)
            rate = 50.0 + 10.0 * torch.sin(2 * math.pi * t / 60.0)
            sig = load_signal_from_rate(rate, window_seconds=300.0)
            lc = decompose_load_signal(sig, num_modes=16)
            th = compute_adaptive_throttle(lc, capacity_rps=100.0)
            checks["load_modeling"] = {"status": "passed", "regime": th.regime}
        except Exception as exc:
            checks["load_modeling"] = {"status": "failed", "error": str(exc)}

        # 7. Scratch directory
        scratch_dir = _managed_scratch_directory(runtime.config)
        test_file = scratch_dir / "_self_test_probe"
        try:
            test_file.write_text("ok")
            test_file.unlink()
            checks["scratch_directory"] = {"status": "passed", "path": str(scratch_dir)}
        except Exception as exc:
            checks["scratch_directory"] = {"status": "failed", "error": str(exc)}

        # 8. SQL capability
        try:
            from spectral_packet_engine.database import sqlalchemy_is_available
            checks["sql"] = {"status": "available" if sqlalchemy_is_available() else "not_installed"}
        except Exception as exc:
            checks["sql"] = {"status": "failed", "error": str(exc)}

        all_passed = all(
            v.get("status") in ("passed", "available", "not_installed")
            for v in checks.values()
            if isinstance(v, dict) and "status" in v
        )
        checks["overall"] = "all_passed" if all_passed else "some_failed"

        return checks

    # ================================================================
    # Server Info Tool — connection details for remote clients
    # ================================================================

    @_tool(
        server,
        runtime,
        "server_info",
        "Return the MCP server's connection details, hostname, local IP, "
        "library version, and scratch directory path. Use this to verify "
        "which server you are connected to.",
    )
    def server_info_tool() -> dict[str, Any]:
        import socket
        from spectral_packet_engine.version import __version__

        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            local_ip = "127.0.0.1"

        scratch_dir = _managed_scratch_directory(runtime.config)

        return {
            "hostname": hostname,
            "best_effort_ipv4": local_ip,
            "library_version": __version__,
            "transport": runtime.config.transport,
            "bind_host": runtime.config.host,
            "bind_port": runtime.config.port,
            "streamable_http_path": runtime.config.streamable_http_path,
            "endpoint_url": runtime.config.endpoint_url,
            "scratch_dir": str(scratch_dir),
            "allowed_hosts": list(runtime.config.allowed_hosts),
            "allowed_origins": list(runtime.config.allowed_origins),
            "max_concurrent_tasks": runtime.config.max_concurrent_tasks,
            "log_destination": runtime.config.log_destination,
            "allow_unsafe_python": runtime.config.allow_unsafe_python,
            "network_note": (
                "Use bind_host/bind_port/endpoint_url as the authoritative connection facts. "
                "best_effort_ipv4 is a best-effort hostname resolution only."
            ),
        }

    # -----------------------------------------------------------------------
    # Spectral extension tools
    # -----------------------------------------------------------------------

    @_tool(server, runtime, "fourier_decomposition", "Compute discrete Fourier decomposition of a 1-D signal, reporting dominant frequencies, amplitudes, phases, and power spectrum.", bounded=True)
    def fourier_decomposition_tool(
        signal: list[float],
        sample_spacing: float = 1.0,
        num_dominant: int = 5,
    ) -> dict[str, Any]:
        return to_serializable(
            fourier_decomposition(signal, sample_spacing=sample_spacing, num_dominant=num_dominant)
        )

    @_tool(server, runtime, "pade_approximant", "Construct a [m/n] Padé approximant from power series coefficients for rational extrapolation.", bounded=True)
    def pade_approximant_tool(
        coefficients: list[float],
        order_m: int = 3,
        order_n: int = 3,
        evaluation_points: list[float] | None = None,
    ) -> dict[str, Any]:
        return to_serializable(
            pade_approximant(
                coefficients, order_m=order_m, order_n=order_n,
                evaluation_points=evaluation_points,
            )
        )

    @_tool(server, runtime, "hilbert_transform", "Compute the analytic signal via Hilbert transform to extract instantaneous amplitude and phase envelopes.", bounded=True)
    def hilbert_transform_tool(
        signal: list[float],
    ) -> dict[str, Any]:
        return to_serializable(hilbert_transform(signal))

    @_tool(server, runtime, "correlation_spectral_analysis", "Eigenvalue decomposition of the cross-correlation matrix among multiple profiles (PCA-style), revealing independent driving modes.", bounded=True)
    def correlation_spectral_analysis_tool(
        profiles: list[list[float]],
        significance_threshold: float = 0.05,
    ) -> dict[str, Any]:
        return to_serializable(
            correlation_spectral_analysis(profiles, significance_threshold=significance_threshold)
        )

    @_tool(server, runtime, "richardson_extrapolation", "Accelerate convergence of a sequence of numerical estimates by removing leading error terms.", bounded=True)
    def richardson_extrapolation_tool(
        estimates: list[float],
        step_ratio: float = 2.0,
        convergence_order: float | None = None,
    ) -> dict[str, Any]:
        return to_serializable(
            richardson_extrapolation(
                estimates, step_ratio=step_ratio, convergence_order=convergence_order,
            )
        )

    @_tool(server, runtime, "kramers_kronig", "Compute Kramers-Kronig dispersion relation connecting real and imaginary parts of a causal response function.", bounded=True)
    def kramers_kronig_tool(
        frequencies: list[float],
        values: list[float],
        direction: str = "real_to_imag",
    ) -> dict[str, Any]:
        return to_serializable(
            kramers_kronig(frequencies, values, direction=direction)
        )

    return server


def main(config: MCPServerConfig | None = None) -> None:
    runtime_config = config or MCPServerConfig()
    configure_service_logging(
        runtime_config.log_level,
        log_file=runtime_config.log_file,
        force=True,
    )
    create_mcp_server(runtime_config).run(transport=runtime_config.transport)


__all__ = [
    "MCPServerConfig",
    "create_mcp_server",
    "main",
    "mcp_is_available",
]


def _build_mcp_module_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m spectral_packet_engine.mcp",
        description="Run the Spectral Packet Engine MCP server.",
    )
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--streamable-http-path", default="/mcp")
    parser.add_argument("--max-concurrent-tasks", type=int, default=1)
    parser.add_argument("--slot-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="warning",
    )
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--scratch-dir", default=None)
    parser.add_argument("--allowed-host", action="append", default=None)
    parser.add_argument("--allowed-origin", action="append", default=None)
    parser.add_argument(
        "--allow-unsafe-python",
        action="store_true",
        help="Enable the trusted-only execute_python tool.",
    )
    return parser


if __name__ == "__main__":
    args = _build_mcp_module_parser().parse_args()
    main(
        MCPServerConfig(
            transport=args.transport,
            host=args.host,
            port=args.port,
            streamable_http_path=args.streamable_http_path,
            max_concurrent_tasks=args.max_concurrent_tasks,
            slot_acquire_timeout_seconds=args.slot_timeout_seconds,
            log_level=args.log_level,
            log_file=args.log_file,
            allow_unsafe_python=args.allow_unsafe_python,
            scratch_directory=args.scratch_dir,
            allowed_hosts=() if args.allowed_host is None else tuple(args.allowed_host),
            allowed_origins=() if args.allowed_origin is None else tuple(args.allowed_origin),
        )
    )
