from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Sequence

from spectral_packet_engine.artifacts import (
    inspect_artifact_directory,
    to_serializable,
    write_compression_artifacts,
    write_compression_sweep_artifacts,
    write_differentiable_artifacts,
    write_feature_table_artifacts,
    write_forward_artifacts,
    write_inverse_artifacts,
    write_modal_evaluation_artifacts,
    write_modal_training_artifacts,
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
from spectral_packet_engine.mcp_runtime import MCPServerConfig
from spectral_packet_engine.ml import ModalSurrogateConfig
from spectral_packet_engine.differentiable_physics import ACCEPTED_CONTROL_OBJECTIVES
from spectral_packet_engine.product import (
    DEFAULT_MCP_LOG_LEVEL,
    DEFAULT_MCP_MAX_CONCURRENT_TASKS,
    DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES,
    DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES,
    DEFAULT_PROFILE_REPORT_OUTPUT_DIR,
    DEFAULT_SQL_PROFILE_REPORT_OUTPUT_DIR,
    PRODUCT_SPINE_STATEMENT,
    RUNTIME_SPINE_STATEMENT,
    guide_workflow,
    inspect_product_identity,
)
from spectral_packet_engine.release_gate import run_release_gate
from spectral_packet_engine.tabular import load_tabular_dataset, save_tabular_dataset
from spectral_packet_engine.table_io import load_profile_table, save_profile_table
from spectral_packet_engine.tf_surrogate import TensorFlowRegressorConfig
from spectral_packet_engine.workflows import (
    analyze_coupled_channel_surfaces,
    analyze_profile_table_spectra,
    analyze_profile_table_from_database_query,
    analyze_separable_tensor_product_spectrum,
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
    derive_urban_microclimate_from_tabular_dataset,
    design_potential_for_target_transition,
    execute_database_script,
    execute_database_statement,
    evaluate_tensorflow_surrogate_on_profile_table,
    execute_database_query,
    export_feature_table_from_database_query,
    export_feature_table_from_profile_table,
    fit_gaussian_packet_to_profile_table,
    fit_gaussian_packet_to_profile_table_from_database_query,
    GradientOptimizationConfig,
    HumanThermalResponseConfig,
    infer_potential_family_from_spectrum,
    inspect_database,
    inspect_environment,
    inspect_ml_backend_support,
    inspect_tree_backend_support,
    ingest_open_meteo_hourly_dataset,
    load_tabular_dataset_from_path,
    materialize_database_query,
    materialize_database_query_to_table,
    optimize_packet_control,
    simulate_packet_sweep,
    project_gaussian_packet,
    run_profile_inference_workflow,
    run_spectroscopy_workflow,
    run_transport_resonance_workflow,
    simulate_gaussian_packet,
    solve_radial_reduction,
    summarize_profile_table,
    summarize_database_query_result,
    summarize_tabular_dataset,
    sweep_profile_table_compression,
    train_tree_model,
    train_modal_surrogate_from_database_query,
    train_modal_surrogate_on_profile_table,
    evaluate_modal_surrogate_from_database_query,
    evaluate_modal_surrogate_on_profile_table,
    train_tensorflow_surrogate_on_profile_table,
    tune_tree_model,
    UrbanBoundaryLayerConfig,
    UrbanRadiativeTransferConfig,
    validate_installation,
    write_tabular_dataset_to_database,
)
from spectral_packet_engine.version import __version__


def _emit(payload) -> None:
    print(json.dumps(to_serializable(payload), indent=2, sort_keys=True))


def _parse_key_value_items(items: Sequence[str] | None) -> dict[str, str]:
    if not items:
        return {}
    values: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"parameters must be passed as key=value, got '{item}'")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("parameter keys must not be empty")
        values[key] = raw_value
    return values


def _parse_float_key_value_items(items: Sequence[str] | None) -> dict[str, float]:
    parsed = _parse_key_value_items(items)
    try:
        return {key: float(value) for key, value in parsed.items()}
    except ValueError as exc:
        raise ValueError(f"parameter values must be numeric: {exc}") from exc


def _parse_json_object(raw_value: str | None, *, flag_name: str) -> dict[str, object]:
    if raw_value is None:
        return {}
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{flag_name} must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{flag_name} must decode to a JSON object")
    return {str(key): value for key, value in payload.items()}


def _coerce_float_mapping(payload: dict[str, object], *, flag_name: str) -> dict[str, float]:
    try:
        return {str(key): float(value) for key, value in payload.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{flag_name} must map parameter names to numeric values") from exc


def _coerce_nested_float_mapping(payload: dict[str, object], *, flag_name: str) -> dict[str, dict[str, float]]:
    nested: dict[str, dict[str, float]] = {}
    for family, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"{flag_name} must map family names to JSON objects")
        nested[str(family)] = _coerce_float_mapping({str(key): item for key, item in value.items()}, flag_name=flag_name)
    return nested


def _add_command_parser(
    subparsers,
    name: str,
    *,
    help_text: str,
    description: str | None = None,
    epilog: str | None = None,
    aliases: Sequence[str] = (),
) -> argparse.ArgumentParser:
    return subparsers.add_parser(
        name,
        aliases=list(aliases),
        help=help_text,
        description=description or help_text,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def _prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    if not sys.stdin.isatty():
        raise ValueError("interactive confirmation requires a TTY; rerun with --yes or --dry-run")
    suffix = "[Y/n]" if default else "[y/N]"
    response = input(f"{prompt} {suffix} ").strip().lower()
    if not response:
        return default
    return response in {"y", "yes"}


def _add_profile_table_materialization_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--time-column",
        default="time",
        help="Column to treat as sample time when materializing non-canonical profile-table inputs.",
    )
    parser.add_argument(
        "--position-column",
        dest="position_columns",
        action="append",
        default=None,
        help="Explicit profile-value column. Repeat to preserve semantic source columns such as h00, h01, ...",
    )
    parser.add_argument(
        "--position-value",
        dest="position_values",
        action="append",
        type=float,
        default=None,
        help="Numeric coordinate for each --position-column. Repeat in the same order as --position-column.",
    )
    parser.add_argument("--sort-by-time", action="store_true", help="Sort rows by the resolved time column before materializing the profile table.")


def _add_sql_profile_table_materialization_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--time-column", default="time", help="Column to treat as sample time when materializing a profile-table-shaped SQL result.")
    parser.add_argument(
        "--position-column",
        action="append",
        dest="position_columns",
        default=None,
        help="Explicit profile-value column from the SQL result. Repeat to preserve semantic source columns.",
    )
    parser.add_argument(
        "--position-value",
        dest="position_values",
        action="append",
        type=float,
        default=None,
        help="Numeric coordinate for each --position-column. Repeat in the same order as --position-column.",
    )
    parser.add_argument(
        "--sort-by-time",
        action="store_true",
        help="Sort materialized query rows by the resolved time column before entering the spectral workflow.",
    )


def _profile_table_load_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "time_column": args.time_column,
        "position_columns": args.position_columns,
        "position_values": args.position_values,
        "sort_by_time": args.sort_by_time,
    }


def _load_profile_table_from_cli_args(path: str | Path, args: argparse.Namespace):
    return load_profile_table(path, **_profile_table_load_kwargs(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectral-packet-engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"{PRODUCT_SPINE_STATEMENT}\n\nRuntime spine: {RUNTIME_SPINE_STATEMENT}",
        epilog=(
            "Start here:\n"
            "  spectral-packet-engine inspect-product\n"
            "  spectral-packet-engine inspect-environment --device cpu\n"
            "  spectral-packet-engine validate-install --device cpu\n\n"
            "Then run the local release gate:\n"
            "  spectral-packet-engine release-gate --device cpu\n\n"
            "Ask the product for the default path when unsure:\n"
            "  spectral-packet-engine guide-workflow\n\n"
            "If you are running from a source checkout, continue with:\n"
            "  spectral-packet-engine profile-report examples/data/synthetic_profiles.csv "
            f"--device cpu --output-dir {DEFAULT_PROFILE_REPORT_OUTPUT_DIR}\n\n"
            "Inspect machine-side surrogate backends with:\n"
            "  spectral-packet-engine ml-backends --device cpu"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    product_parser = _add_command_parser(
        subparsers,
        "inspect-product",
        help_text="Inspect the shared product spine, runtime model, and canonical workflow map.",
        aliases=("product",),
        epilog="Example:\n  spectral-packet-engine inspect-product",
    )

    workflow_guide_parser = _add_command_parser(
        subparsers,
        "guide-workflow",
        aliases=("recommend-workflow",),
        help_text="Recommend the default high-value workflow for a given input shape and surface.",
        description=(
            "Return the product’s opinionated default path for a report-first, inverse-fit, or feature-model "
            "workflow over file-backed or SQL-backed profile data."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine guide-workflow\n"
            "  spectral-packet-engine guide-workflow --input-kind profile-table-sql\n"
            "  spectral-packet-engine guide-workflow --goal inverse-fit\n"
            "  spectral-packet-engine guide-workflow --goal feature-model --input-kind profile-table-sql"
        ),
    )
    workflow_guide_parser.add_argument(
        "--input-kind",
        choices=["profile-table-file", "profile-table-sql"],
        default="profile-table-file",
        help="Type of input you want the product to optimize for.",
    )
    workflow_guide_parser.add_argument(
        "--goal",
        choices=["report", "inverse-fit", "feature-model"],
        default="report",
        help="Outcome you want the product to optimize for.",
    )

    env_parser = _add_command_parser(
        subparsers,
        "inspect-environment",
        aliases=("env",),
        help_text="Inspect runtime capabilities.",
        epilog="Example:\n  spectral-packet-engine inspect-environment --device cpu",
    )
    env_parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or mps.")

    artifact_inspect_parser = _add_command_parser(
        subparsers,
        "inspect-artifacts",
        help_text="Inspect a managed artifact directory and report completion state and metadata.",
        epilog=f"Example:\n  spectral-packet-engine inspect-artifacts {DEFAULT_PROFILE_REPORT_OUTPUT_DIR}",
    )
    artifact_inspect_parser.add_argument("output_dir", type=Path)

    validate_parser = _add_command_parser(
        subparsers,
        "validate-install",
        help_text="Validate the current installation and optional surfaces.",
        epilog="Example:\n  spectral-packet-engine validate-install --device cpu",
    )
    validate_parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or mps.")

    release_gate_parser = _add_command_parser(
        subparsers,
        "release-gate",
        help_text="Run the local in-process release gate for the current environment.",
        description=(
            "Validate the current environment across the shared Python engine, SQL/backend-aware "
            "workflow, and optional API and MCP surfaces."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine release-gate --device cpu\n"
            "  spectral-packet-engine release-gate --device cpu --skip-api --skip-mcp"
        ),
    )
    release_gate_parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or mps.")
    release_gate_parser.add_argument("--skip-api", action="store_true", help="Skip API validation even if FastAPI is installed.")
    release_gate_parser.add_argument("--skip-mcp", action="store_true", help="Skip MCP validation even if MCP is installed.")

    ml_backends_parser = _add_command_parser(
        subparsers,
        "ml-backends",
        help_text="Inspect available ML surrogate backends and their runtime capabilities.",
        description=(
            "Inspect backend-aware modal surrogate support across PyTorch, JAX, and "
            "TensorFlow compatibility paths."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine ml-backends --device cpu\n"
            "  spectral-packet-engine ml-backends --device cuda"
        ),
    )
    ml_backends_parser.add_argument("--device", default="auto", help="Preferred torch device for backend inspection.")

    tree_backends_parser = _add_command_parser(
        subparsers,
        "tree-backends",
        help_text="Inspect available tree-model backends and their runtime capabilities.",
        description=(
            "Inspect tree-model backend support across the scikit-learn baseline and optional "
            "XGBoost, LightGBM, and CatBoost integrations."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine tree-backends\n"
            "  spectral-packet-engine tree-backends --library xgboost"
        ),
    )
    tree_backends_parser.add_argument(
        "--library",
        choices=["auto", "sklearn", "xgboost", "lightgbm", "catboost"],
        default="auto",
        help="Inspect a specific tree-model backend or let the product report the preferred available backend.",
    )

    forward_parser = subparsers.add_parser("forward", help="Run a forward packet simulation.")
    forward_parser.add_argument("--center", type=float, default=0.30)
    forward_parser.add_argument("--width", type=float, default=0.07)
    forward_parser.add_argument("--wavenumber", type=float, default=25.0)
    forward_parser.add_argument("--phase", type=float, default=0.0)
    forward_parser.add_argument("--modes", type=int, default=128)
    forward_parser.add_argument("--quadrature", type=int, default=4096)
    forward_parser.add_argument("--grid", type=int, default=512)
    forward_parser.add_argument("--times", type=float, nargs="*", default=[0.0, 1e-3, 5e-3])
    forward_parser.add_argument("--device", default="auto")
    forward_parser.add_argument("--output-dir", type=Path, default=None)

    project_parser = subparsers.add_parser("project", help="Project a packet into the modal basis.")
    project_parser.add_argument("--center", type=float, default=0.30)
    project_parser.add_argument("--width", type=float, default=0.07)
    project_parser.add_argument("--wavenumber", type=float, default=25.0)
    project_parser.add_argument("--phase", type=float, default=0.0)
    project_parser.add_argument("--modes", type=int, default=128)
    project_parser.add_argument("--quadrature", type=int, default=4096)
    project_parser.add_argument("--grid", type=int, default=2048)
    project_parser.add_argument("--device", default="auto")

    inspect_table_parser = subparsers.add_parser(
        "inspect-profile-table",
        aliases=["inspect-table"],
        help="Inspect a CSV, TSV, JSON, or optionally XLSX profile table.",
    )
    inspect_table_parser.add_argument("table_path", type=Path)
    inspect_table_parser.add_argument("--device", default="auto")
    _add_profile_table_materialization_arguments(inspect_table_parser)

    profile_report_parser = _add_command_parser(
        subparsers,
        "profile-report",
        help_text="Inspect, analyze, and compress a profile table into one reference-grade report.",
        description=(
            "Run the product hero workflow for a profile table: validate the table shape, summarize it, "
            "analyze its modal structure, compress it, and optionally write one inspectable artifact bundle."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine profile-report examples/data/synthetic_profiles.csv "
            f"--device cpu --output-dir {DEFAULT_PROFILE_REPORT_OUTPUT_DIR}"
        ),
    )
    profile_report_parser.add_argument("table_path", type=Path)
    profile_report_parser.add_argument("--analyze-modes", type=int, default=DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES)
    profile_report_parser.add_argument("--compress-modes", type=int, default=DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES)
    profile_report_parser.add_argument("--device", default="auto")
    profile_report_parser.add_argument("--normalize", action="store_true")
    profile_report_parser.add_argument("--output-dir", type=Path, default=None)
    _add_profile_table_materialization_arguments(profile_report_parser)

    export_features_parser = _add_command_parser(
        subparsers,
        "export-features",
        help_text="Export a profile table into a traceable spectral feature table.",
        description=(
            "Project a profile table into modal coefficients and profile moments, then emit one "
            "feature table through the shared workflow and artifact layer."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine export-features examples/data/synthetic_profiles.csv "
            "--modes 16 --device cpu --include coefficients --include moments --output-dir artifacts/features"
        ),
    )
    export_features_parser.add_argument("table_path", type=Path)
    export_features_parser.add_argument("--modes", type=int, default=32)
    export_features_parser.add_argument("--device", default="auto")
    export_features_parser.add_argument("--normalize", action="store_true")
    export_features_parser.add_argument(
        "--include",
        action="append",
        choices=["coefficients", "moments"],
        default=None,
        help="Feature groups to include. Repeat to select multiple groups. Defaults to coefficients and moments.",
    )
    export_features_parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    export_features_parser.add_argument("--output-dir", type=Path, default=None)
    _add_profile_table_materialization_arguments(export_features_parser)

    tabular_inspect_parser = subparsers.add_parser(
        "inspect-tabular-dataset",
        aliases=["tabular-inspect"],
        help="Inspect a CSV, TSV, JSON, and optionally Parquet or XLSX tabular dataset.",
    )
    tabular_inspect_parser.add_argument("dataset_path", type=Path)

    open_meteo_parser = _add_command_parser(
        subparsers,
        "ingest-open-meteo",
        help_text="Fetch hourly Open-Meteo data into the shared tabular/profile-table contract.",
        description=(
            "Retrieve hourly meteorology data from Open-Meteo, normalize it into a shared "
            "TabularDataset, and optionally materialize selected variables into a ProfileTable."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine ingest-open-meteo --latitude 41.01 --longitude 28.97 "
            "--start-date 2026-04-01 --end-date 2026-04-03 --hourly temperature_2m "
            "--hourly relative_humidity_2m --output-path artifacts/weather.csv"
        ),
    )
    open_meteo_parser.add_argument("--latitude", type=float, required=True)
    open_meteo_parser.add_argument("--longitude", type=float, required=True)
    open_meteo_parser.add_argument("--start-date", required=True, help="Inclusive ISO date (YYYY-MM-DD).")
    open_meteo_parser.add_argument("--end-date", required=True, help="Inclusive ISO date (YYYY-MM-DD).")
    open_meteo_parser.add_argument("--hourly", dest="hourly_variables", action="append", required=True, help="Hourly Open-Meteo variable. Repeat to request multiple variables.")
    open_meteo_parser.add_argument(
        "--api-kind",
        choices=["auto", "forecast", "historical-forecast", "historical-weather"],
        default="auto",
        help="Open-Meteo backend family. 'auto' routes by requested date range.",
    )
    open_meteo_parser.add_argument("--timezone-name", default="GMT", help="Open-Meteo timezone parameter. Defaults to GMT for explicit UTC-aligned output.")
    open_meteo_parser.add_argument("--model", dest="models", action="append", default=None, help="Optional Open-Meteo model selector. Repeat to pass multiple models.")
    open_meteo_parser.add_argument("--temperature-unit", default=None)
    open_meteo_parser.add_argument("--wind-speed-unit", default=None)
    open_meteo_parser.add_argument("--precipitation-unit", default=None)
    open_meteo_parser.add_argument("--cell-selection", default=None)
    open_meteo_parser.add_argument("--elevation", type=float, default=None)
    open_meteo_parser.add_argument("--timeout-seconds", type=float, default=30.0)
    open_meteo_parser.add_argument("--output-path", type=Path, default=None, help="Optional path to persist the normalized tabular dataset.")
    open_meteo_parser.add_argument("--profile-output", type=Path, default=None, help="Optional path to persist the materialized profile table.")
    _add_profile_table_materialization_arguments(open_meteo_parser)

    urban_microclimate_parser = _add_command_parser(
        subparsers,
        "derive-urban-microclimate",
        help_text="Derive boundary-layer, radiative, and human thermal metrics from a tabular weather dataset.",
        description=(
            "Run an explicit urban microclimate closure over a shared TabularDataset. "
            "Column alignment stays explicit: either choose a named mapping profile such as "
            "'open-meteo' or pass the required source column names directly."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine derive-urban-microclimate artifacts/weather.csv --mapping-profile open-meteo "
            "--output-path artifacts/weather_microclimate.csv\n"
            "  spectral-packet-engine derive-urban-microclimate data.csv --air-temperature-column temp_c "
            "--relative-humidity-column rh --wind-speed-column wind_m_s"
        ),
    )
    urban_microclimate_parser.add_argument("dataset_path", type=Path)
    urban_microclimate_parser.add_argument("--mapping-profile", choices=["open-meteo"], default=None)
    urban_microclimate_parser.add_argument("--air-temperature-column", default=None)
    urban_microclimate_parser.add_argument("--relative-humidity-column", default=None)
    urban_microclimate_parser.add_argument("--wind-speed-column", default=None)
    urban_microclimate_parser.add_argument("--pressure-column", default="surface_pressure")
    urban_microclimate_parser.add_argument("--shortwave-radiation-column", default=None)
    urban_microclimate_parser.add_argument("--longwave-radiation-column", default=None)
    urban_microclimate_parser.add_argument("--measurement-height-m", type=float, default=10.0)
    urban_microclimate_parser.add_argument("--zero-plane-displacement-m", type=float, default=0.0)
    urban_microclimate_parser.add_argument("--roughness-length-m", type=float, default=0.8)
    urban_microclimate_parser.add_argument("--thermal-roughness-length-m", type=float, default=0.1)
    urban_microclimate_parser.add_argument("--min-wind-speed-m-s", type=float, default=0.1)
    urban_microclimate_parser.add_argument("--reference-pressure-pa", type=float, default=101325.0)
    urban_microclimate_parser.add_argument("--shortwave-absorptivity", type=float, default=0.7)
    urban_microclimate_parser.add_argument("--longwave-emissivity", type=float, default=0.97)
    urban_microclimate_parser.add_argument("--shortwave-projected-area-factor", type=float, default=0.7)
    urban_microclimate_parser.add_argument("--sky-view-factor", type=float, default=1.0)
    urban_microclimate_parser.add_argument("--skin-temperature-c", type=float, default=33.0)
    urban_microclimate_parser.add_argument("--clothing-insulation-clo", type=float, default=0.7)
    urban_microclimate_parser.add_argument("--metabolic-heat-flux-w-m2", type=float, default=80.0)
    urban_microclimate_parser.add_argument("--output-path", type=Path, default=None, help="Optional path to persist the derived dataset.")

    analyze_table_parser = subparsers.add_parser(
        "analyze-profile-table",
        aliases=["analyze-table"],
        help="Project a profile table into the modal basis and summarize its spectral structure.",
    )
    analyze_table_parser.add_argument("table_path", type=Path)
    analyze_table_parser.add_argument("--modes", type=int, default=32)
    analyze_table_parser.add_argument("--device", default="auto")
    analyze_table_parser.add_argument("--normalize", action="store_true")
    analyze_table_parser.add_argument("--output-dir", type=Path, default=None)
    _add_profile_table_materialization_arguments(analyze_table_parser)

    compress_parser = subparsers.add_parser(
        "compress-profile-table",
        aliases=["compress-table", "compress-csv"],
        help="Compress a CSV, TSV, JSON, or optionally XLSX profile table into modal coefficients.",
    )
    compress_parser.add_argument("table_path", type=Path)
    compress_parser.add_argument("--modes", type=int, default=32)
    compress_parser.add_argument("--device", default="auto")
    compress_parser.add_argument("--normalize", action="store_true", help="Normalize each profile before compression.")
    compress_parser.add_argument("--output-dir", type=Path, default=None)
    _add_profile_table_materialization_arguments(compress_parser)

    sweep_parser = subparsers.add_parser("compression-sweep", help="Evaluate compression quality across multiple mode counts.")
    sweep_parser.add_argument("table_path", type=Path)
    sweep_parser.add_argument("--mode-counts", type=int, nargs="+", default=[4, 8, 16, 32])
    sweep_parser.add_argument("--device", default="auto")
    sweep_parser.add_argument("--normalize", action="store_true")
    sweep_parser.add_argument("--output-dir", type=Path, default=None)
    _add_profile_table_materialization_arguments(sweep_parser)

    fit_parser = subparsers.add_parser(
        "fit-profile-table",
        aliases=["fit-table", "fit-csv"],
        help="Fit a Gaussian packet to a CSV, TSV, JSON, or optionally XLSX density table.",
    )
    fit_parser.add_argument("table_path", type=Path)
    fit_parser.add_argument("--center", type=float, default=0.36, help="Initial center guess.")
    fit_parser.add_argument("--width", type=float, default=0.11, help="Initial width guess.")
    fit_parser.add_argument("--wavenumber", type=float, default=22.0, help="Initial wavenumber guess.")
    fit_parser.add_argument("--phase", type=float, default=0.0, help="Initial phase guess.")
    fit_parser.add_argument("--modes", type=int, default=128)
    fit_parser.add_argument("--quadrature", type=int, default=2048)
    fit_parser.add_argument("--steps", type=int, default=200)
    fit_parser.add_argument("--learning-rate", type=float, default=0.05)
    fit_parser.add_argument("--device", default="auto")
    fit_parser.add_argument("--output-dir", type=Path, default=None)
    _add_profile_table_materialization_arguments(fit_parser)

    spectroscopy_parser = _add_command_parser(
        subparsers,
        "infer-potential-spectrum",
        aliases=("spectroscopy-workflow",),
        help_text="Infer which bounded potential family best explains an observed low-lying spectrum.",
        description=(
            "Compare explicit potential families against observed eigenvalues through the shared "
            "differentiable eigensolver and return a family ranking with local uncertainty summaries."
        ),
    )
    spectroscopy_parser.add_argument("target_eigenvalues", type=float, nargs="+")
    spectroscopy_parser.add_argument("--family", action="append", default=None, help="Candidate family name. Repeat to compare multiple families.")
    spectroscopy_parser.add_argument("--initial-guesses", default=None, help="Optional JSON object mapping family -> {parameter: value}.")
    spectroscopy_parser.add_argument("--domain-length", type=float, default=1.0)
    spectroscopy_parser.add_argument("--left", type=float, default=0.0)
    spectroscopy_parser.add_argument("--mass", type=float, default=1.0)
    spectroscopy_parser.add_argument("--hbar", type=float, default=1.0)
    spectroscopy_parser.add_argument("--points", type=int, default=128)
    spectroscopy_parser.add_argument("--steps", type=int, default=200)
    spectroscopy_parser.add_argument("--learning-rate", type=float, default=0.05)
    spectroscopy_parser.add_argument("--device", default="auto")
    spectroscopy_parser.add_argument("--output-dir", type=Path, default=None)

    separable_parser = _add_command_parser(
        subparsers,
        "analyze-separable-spectrum",
        help_text="Analyze a structured separable spectrum from tensor-product 1D components.",
    )
    separable_parser.add_argument("--family-x", required=True)
    separable_parser.add_argument("--params-x", required=True, help="JSON object of x-axis family parameters.")
    separable_parser.add_argument("--family-y", required=True)
    separable_parser.add_argument("--params-y", required=True, help="JSON object of y-axis family parameters.")
    separable_parser.add_argument("--domain-length-x", type=float, default=1.0)
    separable_parser.add_argument("--domain-length-y", type=float, default=1.0)
    separable_parser.add_argument("--points-x", type=int, default=96)
    separable_parser.add_argument("--points-y", type=int, default=96)
    separable_parser.add_argument("--states-x", type=int, default=6)
    separable_parser.add_argument("--states-y", type=int, default=6)
    separable_parser.add_argument("--combined-states", type=int, default=12)
    separable_parser.add_argument("--low-rank-rank", type=int, default=1)
    separable_parser.add_argument("--device", default="auto")
    separable_parser.add_argument("--output-dir", type=Path, default=None)

    coupled_surfaces_parser = _add_command_parser(
        subparsers,
        "analyze-coupled-surfaces",
        help_text="Analyze a reduced two-channel avoided crossing and its adiabatic surfaces.",
    )
    coupled_surfaces_parser.add_argument("--domain-length", type=float, default=1.0)
    coupled_surfaces_parser.add_argument("--grid-points", type=int, default=256)
    coupled_surfaces_parser.add_argument("--slope", type=float, default=30.0)
    coupled_surfaces_parser.add_argument("--bias", type=float, default=0.0)
    coupled_surfaces_parser.add_argument("--coupling", type=float, default=2.0)
    coupled_surfaces_parser.add_argument("--coupling-width", type=float, default=0.12)
    coupled_surfaces_parser.add_argument("--device", default="cpu")
    coupled_surfaces_parser.add_argument("--output-dir", type=Path, default=None)

    radial_parser = _add_command_parser(
        subparsers,
        "solve-radial-reduction",
        help_text="Solve a bounded radial effective-coordinate reduction with centrifugal term.",
    )
    radial_parser.add_argument("--family", required=True)
    radial_parser.add_argument("--params", required=True, help="JSON object of radial base-potential parameters.")
    radial_parser.add_argument("--angular-momentum", type=int, default=0)
    radial_parser.add_argument("--radial-min", type=float, default=0.05)
    radial_parser.add_argument("--radial-max", type=float, default=3.0)
    radial_parser.add_argument("--points", type=int, default=128)
    radial_parser.add_argument("--states", type=int, default=6)
    radial_parser.add_argument("--mass", type=float, default=1.0)
    radial_parser.add_argument("--hbar", type=float, default=1.0)
    radial_parser.add_argument("--device", default="cpu")
    radial_parser.add_argument("--output-dir", type=Path, default=None)

    compare_parser = subparsers.add_parser(
        "compare-profile-tables",
        aliases=["compare-tables"],
        help="Compare a candidate profile table against a reference table.",
    )
    compare_parser.add_argument("reference_table_path", type=Path)
    compare_parser.add_argument("candidate_table_path", type=Path)
    compare_parser.add_argument("--device", default="auto")
    compare_parser.add_argument("--output-dir", type=Path, default=None)
    _add_profile_table_materialization_arguments(compare_parser)

    db_inspect_parser = subparsers.add_parser(
        "inspect-database",
        aliases=["db-inspect"],
        help="Inspect an existing SQLite database path or database URL and list available tables.",
    )
    db_inspect_parser.add_argument("database")

    db_bootstrap_parser = _add_command_parser(
        subparsers,
        "bootstrap-database",
        aliases=("db-bootstrap",),
        help_text="Create or open a local SQLite database path and report its capabilities.",
        description=(
            "Bootstrap a local SQLite database file so file-backed or SQL-backed spectral "
            "and ML workflows have a predictable local storage target."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine db-bootstrap artifacts/local.db"
        ),
    )
    db_bootstrap_parser.add_argument("database", help="SQLite file path to create or open.")

    db_list_parser = subparsers.add_parser(
        "list-database-tables",
        aliases=["db-list-tables"],
        help="List tables in an existing SQLite database path or database URL.",
    )
    db_list_parser.add_argument("database")

    db_describe_parser = subparsers.add_parser(
        "describe-database-table",
        aliases=["db-describe-table"],
        help="Describe a table schema and row count from an existing database.",
    )
    db_describe_parser.add_argument("database")
    db_describe_parser.add_argument("table_name")

    db_query_parser = _add_command_parser(
        subparsers,
        "query-database",
        aliases=("db-query",),
        help_text="Run a read-only parameterized SQL query and materialize the result as a tabular dataset.",
        description=(
            "Execute a safe read-only parameterized query against a SQLite path or database URL, "
            "then return the result through the shared tabular dataset layer."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine db-query artifacts/local.db "
            "\"select * from profiles where time >= :t0\" --param t0=0.5 "
            "--output-dir artifacts/query"
        ),
    )
    db_query_parser.add_argument("database", help="SQLite file path or database URL.")
    db_query_parser.add_argument("query", help="SQL statement to execute.")
    db_query_parser.add_argument("--param", action="append", default=None, help="Bound query parameter in key=value form.")
    db_query_parser.add_argument("--output-dir", type=Path, default=None)

    db_execute_parser = _add_command_parser(
        subparsers,
        "execute-database-statement",
        aliases=("db-exec",),
        help_text="Run a non-query SQL statement such as CREATE, INSERT, UPDATE, or DELETE.",
        description=(
            "Execute one explicit non-query SQL statement through the shared database workflow layer. "
            "Use this for database mutations; keep db-query for read-only result sets."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine db-exec artifacts/local.db "
            "\"create table if not exists metrics (time real, value real)\""
        ),
    )
    db_execute_parser.add_argument("database", help="SQLite file path or database URL.")
    db_execute_parser.add_argument("statement", help="Single SQL statement to execute.")
    db_execute_parser.add_argument("--param", action="append", default=None, help="Bound query parameter in key=value form.")

    db_script_parser = _add_command_parser(
        subparsers,
        "execute-database-script",
        aliases=("db-script",),
        help_text="Run a multi-statement SQL script against a managed SQLite database.",
        description=(
            "Execute an explicit multi-statement SQL script through the shared database workflow layer. "
            "This is intended for SQLite bootstrap or scratch-database setup flows."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine db-script artifacts/local.db "
            "\"create table if not exists metrics (time real, value real); insert into metrics values (0.0, 1.0);\""
        ),
    )
    db_script_parser.add_argument("database", help="SQLite file path or database URL.")
    db_script_parser.add_argument("script", help="SQL script to execute.")

    db_write_parser = subparsers.add_parser(
        "write-database-table",
        aliases=["db-write-table"],
        help="Load a tabular dataset from disk and write it into a database table.",
    )
    db_write_parser.add_argument("database")
    db_write_parser.add_argument("table_name")
    db_write_parser.add_argument("dataset_path", type=Path)
    db_write_parser.add_argument("--if-exists", choices=["fail", "replace", "append"], default="fail")

    db_materialize_parser = _add_command_parser(
        subparsers,
        "materialize-database-query",
        aliases=("db-materialize-query",),
        help_text="Run a query and persist its result as a managed database table.",
        description=(
            "Materialize a query result into a managed database table so later spectral "
            "analysis, inverse fitting, or surrogate jobs can reuse it without rerunning "
            "the original SQL."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine db-materialize-query artifacts/local.db "
            "profiles_subset \"select * from profiles where time >= :t0\" --param t0=0.5"
        ),
    )
    db_materialize_parser.add_argument("database", help="SQLite file path or database URL.")
    db_materialize_parser.add_argument("table_name", help="Destination table name for the materialized result.")
    db_materialize_parser.add_argument("query", help="SQL statement whose result should be persisted.")
    db_materialize_parser.add_argument("--param", action="append", default=None, help="Bound query parameter in key=value form.")
    db_materialize_parser.add_argument("--replace", action="store_true")

    sql_analyze_parser = _add_command_parser(
        subparsers,
        "sql-analyze-table",
        help_text="Run a SQL query that yields a profile table shape, then analyze its spectral structure.",
        description=(
            "Pull a profile-table-shaped result set from SQL, convert it through the shared "
            "library boundary, and summarize modal structure and truncation behavior."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine sql-analyze-table artifacts/local.db "
            "\"select * from profiles\" --modes 8 --device cpu --output-dir artifacts/sql-analysis"
        ),
    )
    sql_analyze_parser.add_argument("database", help="SQLite file path or database URL.")
    sql_analyze_parser.add_argument("query", help="SQL statement that yields a profile-table-shaped result.")
    _add_sql_profile_table_materialization_arguments(sql_analyze_parser)
    sql_analyze_parser.add_argument("--param", action="append", default=None, help="Bound query parameter in key=value form.")
    sql_analyze_parser.add_argument("--modes", type=int, default=32)
    sql_analyze_parser.add_argument("--device", default="auto")
    sql_analyze_parser.add_argument("--normalize", action="store_true")
    sql_analyze_parser.add_argument("--output-dir", type=Path, default=None)

    sql_compress_parser = _add_command_parser(
        subparsers,
        "sql-compress-table",
        help_text="Run a profile-table-shaped SQL query, then compress it into modal coefficients and reconstructions.",
        description=(
            "Materialize a query result through the shared profile-table contract, then run the same "
            "modal compression workflow used by Python and file-backed CLI jobs."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine sql-compress-table artifacts/local.db "
            "\"select * from profiles order by time\" --modes 8 --device cpu --output-dir artifacts/sql-compression"
        ),
    )
    sql_compress_parser.add_argument("database", help="SQLite file path or database URL.")
    sql_compress_parser.add_argument("query", help="SQL statement that yields a profile-table-shaped result.")
    _add_sql_profile_table_materialization_arguments(sql_compress_parser)
    sql_compress_parser.add_argument("--param", action="append", default=None, help="Bound query parameter in key=value form.")
    sql_compress_parser.add_argument("--modes", type=int, default=32)
    sql_compress_parser.add_argument("--device", default="auto")
    sql_compress_parser.add_argument("--normalize", action="store_true")
    sql_compress_parser.add_argument("--output-dir", type=Path, default=None)

    sql_export_features_parser = _add_command_parser(
        subparsers,
        "sql-export-features",
        help_text="Run a profile-table-shaped SQL query and export a spectral feature table from it.",
        description=(
            "Materialize a query result through the shared profile-table contract, then export "
            "modal coefficients and profile moments as a traceable feature table."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine sql-export-features artifacts/local.db "
            "\"select * from profiles order by time\" --modes 16 --device cpu --output-dir artifacts/sql-features"
        ),
    )
    sql_export_features_parser.add_argument("database", help="SQLite file path or database URL.")
    sql_export_features_parser.add_argument("query", help="SQL statement that yields a profile-table-shaped result.")
    _add_sql_profile_table_materialization_arguments(sql_export_features_parser)
    sql_export_features_parser.add_argument("--param", action="append", default=None, help="Bound query parameter in key=value form.")
    sql_export_features_parser.add_argument("--modes", type=int, default=32)
    sql_export_features_parser.add_argument("--device", default="auto")
    sql_export_features_parser.add_argument("--normalize", action="store_true")
    sql_export_features_parser.add_argument(
        "--include",
        action="append",
        choices=["coefficients", "moments"],
        default=None,
        help="Feature groups to include. Repeat to select multiple groups. Defaults to coefficients and moments.",
    )
    sql_export_features_parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    sql_export_features_parser.add_argument("--output-dir", type=Path, default=None)

    sql_profile_report_parser = _add_command_parser(
        subparsers,
        "sql-profile-report",
        help_text="Run a profile-table-shaped SQL query through the shared inspect-analyze-compress report workflow.",
        description=(
            "Materialize a SQL result through the explicit profile-table contract, then run the same "
            "profile report workflow used by direct Python and file-backed CLI jobs."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine sql-profile-report artifacts/local.db "
            f"\"select * from profiles order by time\" --device cpu --output-dir {DEFAULT_SQL_PROFILE_REPORT_OUTPUT_DIR}"
        ),
    )
    sql_profile_report_parser.add_argument("database", help="SQLite file path or database URL.")
    sql_profile_report_parser.add_argument("query", help="SQL statement that yields a profile-table-shaped result.")
    _add_sql_profile_table_materialization_arguments(sql_profile_report_parser)
    sql_profile_report_parser.add_argument("--param", action="append", default=None, help="Bound query parameter in key=value form.")
    sql_profile_report_parser.add_argument("--analyze-modes", type=int, default=DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES)
    sql_profile_report_parser.add_argument("--compress-modes", type=int, default=DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES)
    sql_profile_report_parser.add_argument("--device", default="auto")
    sql_profile_report_parser.add_argument("--normalize", action="store_true")
    sql_profile_report_parser.add_argument("--output-dir", type=Path, default=None)

    sql_fit_parser = _add_command_parser(
        subparsers,
        "sql-fit-table",
        help_text="Run a profile-table-shaped SQL query, then fit a bounded Gaussian packet to the observed densities.",
        description=(
            "Materialize a query result through the shared profile-table contract, then run the same "
            "inverse fitting workflow used by direct Python and file-backed jobs."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine sql-fit-table artifacts/local.db "
            "\"select * from profiles order by time\" --device cpu --output-dir artifacts/sql-fit"
        ),
    )
    sql_fit_parser.add_argument("database", help="SQLite file path or database URL.")
    sql_fit_parser.add_argument("query", help="SQL statement that yields a profile-table-shaped result.")
    _add_sql_profile_table_materialization_arguments(sql_fit_parser)
    sql_fit_parser.add_argument("--param", action="append", default=None, help="Bound query parameter in key=value form.")
    sql_fit_parser.add_argument("--center", type=float, default=0.36, help="Initial center guess.")
    sql_fit_parser.add_argument("--width", type=float, default=0.11, help="Initial width guess.")
    sql_fit_parser.add_argument("--wavenumber", type=float, default=22.0, help="Initial wavenumber guess.")
    sql_fit_parser.add_argument("--phase", type=float, default=0.0, help="Initial phase guess.")
    sql_fit_parser.add_argument("--modes", type=int, default=128)
    sql_fit_parser.add_argument("--quadrature", type=int, default=2048)
    sql_fit_parser.add_argument("--steps", type=int, default=200)
    sql_fit_parser.add_argument("--learning-rate", type=float, default=0.05)
    sql_fit_parser.add_argument("--device", default="auto")
    sql_fit_parser.add_argument("--output-dir", type=Path, default=None)

    packet_sweep_parser = subparsers.add_parser("packet-sweep", help="Run a batch of packet simulations with shared settings.")
    packet_sweep_parser.add_argument("--centers", type=float, nargs="+", required=True)
    packet_sweep_parser.add_argument("--widths", type=float, nargs="+", required=True)
    packet_sweep_parser.add_argument("--wavenumbers", type=float, nargs="+", required=True)
    packet_sweep_parser.add_argument("--phases", type=float, nargs="*", default=None)
    packet_sweep_parser.add_argument("--times", type=float, nargs="*", default=[0.0, 1e-3, 5e-3])
    packet_sweep_parser.add_argument("--modes", type=int, default=128)
    packet_sweep_parser.add_argument("--quadrature", type=int, default=4096)
    packet_sweep_parser.add_argument("--grid", type=int, default=512)
    packet_sweep_parser.add_argument("--device", default="auto")
    packet_sweep_parser.add_argument("--output-dir", type=Path, default=None)

    transport_parser = subparsers.add_parser("transport-benchmark", help="Benchmark the published transport scan.")
    transport_parser.add_argument("--scan-id", default="scan11879_56")
    transport_parser.add_argument("--mode-counts", type=int, nargs="*", default=[8, 16, 32, 64])
    transport_parser.add_argument("--device", default="auto")
    transport_parser.add_argument("--output-dir", type=Path, default=None)

    design_transition_parser = _add_command_parser(
        subparsers,
        "design-transition",
        help_text="Optimize potential-family parameters so a selected spectral transition hits a target value.",
    )
    design_transition_parser.add_argument("--family", required=True)
    design_transition_parser.add_argument("--target-transition", type=float, required=True)
    design_transition_parser.add_argument("--initial-guess", required=True, help="JSON object of family parameter seeds.")
    design_transition_parser.add_argument("--transition-indices", type=int, nargs=2, default=(0, 1))
    design_transition_parser.add_argument("--domain-length", type=float, default=1.0)
    design_transition_parser.add_argument("--left", type=float, default=0.0)
    design_transition_parser.add_argument("--mass", type=float, default=1.0)
    design_transition_parser.add_argument("--hbar", type=float, default=1.0)
    design_transition_parser.add_argument("--points", type=int, default=128)
    design_transition_parser.add_argument("--states", type=int, default=4)
    design_transition_parser.add_argument("--steps", type=int, default=200)
    design_transition_parser.add_argument("--learning-rate", type=float, default=0.05)
    design_transition_parser.add_argument("--device", default="auto")
    design_transition_parser.add_argument("--output-dir", type=Path, default=None)

    control_parser = _add_command_parser(
        subparsers,
        "optimize-packet-control",
        help_text="Optimize Gaussian packet preparation parameters for a target observable.",
    )
    control_parser.add_argument("--center", type=float, default=0.30)
    control_parser.add_argument("--width", type=float, default=0.07)
    control_parser.add_argument("--wavenumber", type=float, default=25.0)
    control_parser.add_argument("--phase", type=float, default=0.0)
    control_parser.add_argument(
        "--objective",
        choices=list(ACCEPTED_CONTROL_OBJECTIVES),
        default="position",
    )
    control_parser.add_argument("--target-value", type=float, required=True)
    control_parser.add_argument("--final-time", type=float, required=True)
    control_parser.add_argument("--interval", type=float, nargs=2, default=None)
    control_parser.add_argument("--modes", type=int, default=96)
    control_parser.add_argument("--quadrature", type=int, default=2048)
    control_parser.add_argument("--grid", type=int, default=128)
    control_parser.add_argument("--steps", type=int, default=200)
    control_parser.add_argument("--learning-rate", type=float, default=0.05)
    control_parser.add_argument("--device", default="auto")
    control_parser.add_argument("--output-dir", type=Path, default=None)

    transport_workflow_parser = _add_command_parser(
        subparsers,
        "transport-workflow",
        help_text="Run the barrier/resonance vertical workflow over the shared tunneling experiment stack.",
    )
    transport_workflow_parser.add_argument("--barrier-height", type=float, default=50.0)
    transport_workflow_parser.add_argument("--barrier-width-sigma", type=float, default=0.03)
    transport_workflow_parser.add_argument("--domain-length", type=float, default=1.0)
    transport_workflow_parser.add_argument("--grid-points", type=int, default=512)
    transport_workflow_parser.add_argument("--modes", type=int, default=128)
    transport_workflow_parser.add_argument("--energies", type=int, default=500)
    transport_workflow_parser.add_argument("--packet-center", type=float, default=0.25)
    transport_workflow_parser.add_argument("--packet-width", type=float, default=0.04)
    transport_workflow_parser.add_argument("--packet-wavenumber", type=float, default=40.0)
    transport_workflow_parser.add_argument("--device", default="cpu")
    transport_workflow_parser.add_argument("--output-dir", type=Path, default=None)

    profile_inference_parser = _add_command_parser(
        subparsers,
        "profile-inference-workflow",
        help_text="Run the report-first tabular vertical: report, inverse fit, and spectral feature export.",
    )
    profile_inference_parser.add_argument("table_path", type=Path)
    profile_inference_parser.add_argument("--center", type=float, default=0.36)
    profile_inference_parser.add_argument("--width", type=float, default=0.11)
    profile_inference_parser.add_argument("--wavenumber", type=float, default=22.0)
    profile_inference_parser.add_argument("--phase", type=float, default=0.0)
    profile_inference_parser.add_argument("--analyze-modes", type=int, default=16)
    profile_inference_parser.add_argument("--compress-modes", type=int, default=8)
    profile_inference_parser.add_argument("--inverse-modes", type=int, default=96)
    profile_inference_parser.add_argument("--feature-modes", type=int, default=16)
    profile_inference_parser.add_argument("--quadrature", type=int, default=2048)
    profile_inference_parser.add_argument("--device", default="auto")
    profile_inference_parser.add_argument("--normalize", action="store_true")
    profile_inference_parser.add_argument("--output-dir", type=Path, default=None)
    _add_profile_table_materialization_arguments(profile_inference_parser)

    tf_parser = subparsers.add_parser(
        "tf-train-table",
        aliases=["tf-train-csv"],
        help="Train the TensorFlow surrogate on a CSV, TSV, JSON, or XLSX profile table.",
    )
    tf_parser.add_argument("table_path", type=Path)
    tf_parser.add_argument("--modes", type=int, default=16)
    tf_parser.add_argument("--epochs", type=int, default=20)
    tf_parser.add_argument("--batch-size", type=int, default=64)
    tf_parser.add_argument("--normalize", action="store_true")
    tf_parser.add_argument("--export-dir", type=Path, default=None)
    tf_parser.add_argument("--output-dir", type=Path, default=None)
    _add_profile_table_materialization_arguments(tf_parser)

    tf_eval_parser = subparsers.add_parser(
        "tf-evaluate-table",
        help="Train the TensorFlow surrogate and evaluate it on the same profile table.",
    )
    tf_eval_parser.add_argument("table_path", type=Path)
    tf_eval_parser.add_argument("--modes", type=int, default=16)
    tf_eval_parser.add_argument("--epochs", type=int, default=20)
    tf_eval_parser.add_argument("--batch-size", type=int, default=64)
    tf_eval_parser.add_argument("--normalize", action="store_true")
    tf_eval_parser.add_argument("--export-dir", type=Path, default=None)
    tf_eval_parser.add_argument("--output-dir", type=Path, default=None)
    _add_profile_table_materialization_arguments(tf_eval_parser)

    ml_train_parser = _add_command_parser(
        subparsers,
        "ml-train-table",
        help_text="Train a backend-aware modal surrogate on a profile table.",
        description=(
            "Train the shared modal surrogate workflow on a file-backed profile table using "
            "the requested backend or the best available backend."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine ml-train-table examples/data/synthetic_profiles.csv "
            "--backend torch --modes 8 --epochs 20 --device cpu\n"
            "  spectral-packet-engine ml-train-table examples/data/synthetic_profiles.csv "
            "--backend jax --modes 8 --epochs 20 --device cpu"
        ),
    )
    ml_train_parser.add_argument("table_path", type=Path, help="Path to a profile table file.")
    ml_train_parser.add_argument("--backend", choices=["auto", "torch", "jax", "tensorflow"], default="auto")
    ml_train_parser.add_argument("--modes", type=int, default=16)
    ml_train_parser.add_argument("--epochs", type=int, default=20)
    ml_train_parser.add_argument("--batch-size", type=int, default=64)
    ml_train_parser.add_argument("--normalize", action="store_true")
    ml_train_parser.add_argument("--device", default="auto")
    ml_train_parser.add_argument("--export-dir", type=Path, default=None)
    ml_train_parser.add_argument("--output-dir", type=Path, default=None)
    _add_profile_table_materialization_arguments(ml_train_parser)

    ml_eval_parser = _add_command_parser(
        subparsers,
        "ml-evaluate-table",
        help_text="Train and evaluate a backend-aware modal surrogate on a profile table.",
        description=(
            "Train the shared modal surrogate workflow, reconstruct held-out profiles, and "
            "emit backend-aware metrics and artifact bundles."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine ml-evaluate-table examples/data/synthetic_profiles.csv "
            "--backend torch --modes 8 --epochs 20 --device cpu --output-dir artifacts/ml-eval"
        ),
    )
    ml_eval_parser.add_argument("table_path", type=Path, help="Path to a profile table file.")
    ml_eval_parser.add_argument("--backend", choices=["auto", "torch", "jax", "tensorflow"], default="auto")
    ml_eval_parser.add_argument("--modes", type=int, default=16)
    ml_eval_parser.add_argument("--epochs", type=int, default=20)
    ml_eval_parser.add_argument("--batch-size", type=int, default=64)
    ml_eval_parser.add_argument("--normalize", action="store_true")
    ml_eval_parser.add_argument("--device", default="auto")
    ml_eval_parser.add_argument("--export-dir", type=Path, default=None)
    ml_eval_parser.add_argument("--output-dir", type=Path, default=None)
    _add_profile_table_materialization_arguments(ml_eval_parser)

    tree_train_parser = _add_command_parser(
        subparsers,
        "tree-train",
        help_text="Train a tree model on a feature table.",
        description=(
            "Load a generic tabular feature table, select explicit predictors and a target column, "
            "then train one tree-model backend through the shared workflow layer."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine tree-train artifacts/features/features.csv "
            "--target-column y --library sklearn --task regression --output-dir artifacts/tree-train"
        ),
    )
    tree_train_parser.add_argument("features_path", type=Path)
    tree_train_parser.add_argument("--target-column", required=True)
    tree_train_parser.add_argument("--feature-column", action="append", dest="feature_columns", default=None)
    tree_train_parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    tree_train_parser.add_argument(
        "--library",
        choices=["auto", "sklearn", "xgboost", "lightgbm", "catboost"],
        default="auto",
    )
    tree_train_parser.add_argument("--model", default=None)
    tree_train_parser.add_argument("--params", default=None, help="JSON object of model constructor parameters.")
    tree_train_parser.add_argument("--test-fraction", type=float, default=0.2)
    tree_train_parser.add_argument("--random-state", type=int, default=0)
    tree_train_parser.add_argument("--export-dir", type=Path, default=None)
    tree_train_parser.add_argument("--output-dir", type=Path, default=None)

    tree_tune_parser = _add_command_parser(
        subparsers,
        "tree-tune",
        help_text="Tune a tree model on a feature table with cross-validation.",
        description=(
            "Load a generic feature table, search an explicit hyperparameter space, and keep one "
            "holdout test split separate from the cross-validation search."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine tree-tune artifacts/features/features.csv "
            "--target-column y --library sklearn --search-space '{\"n_estimators\":[50,100]}' "
            "--output-dir artifacts/tree-tune"
        ),
    )
    tree_tune_parser.add_argument("features_path", type=Path)
    tree_tune_parser.add_argument("--target-column", required=True)
    tree_tune_parser.add_argument("--feature-column", action="append", dest="feature_columns", default=None)
    tree_tune_parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    tree_tune_parser.add_argument(
        "--library",
        choices=["auto", "sklearn", "xgboost", "lightgbm", "catboost"],
        default="auto",
    )
    tree_tune_parser.add_argument("--model", default=None)
    tree_tune_parser.add_argument("--search-space", required=True, help="JSON object mapping parameter names to candidate value lists.")
    tree_tune_parser.add_argument("--search-kind", choices=["grid", "random"], default="random")
    tree_tune_parser.add_argument("--n-iter", type=int, default=30)
    tree_tune_parser.add_argument("--cv", type=int, default=5)
    tree_tune_parser.add_argument("--scoring", default=None)
    tree_tune_parser.add_argument("--test-fraction", type=float, default=0.2)
    tree_tune_parser.add_argument("--random-state", type=int, default=0)
    tree_tune_parser.add_argument("--export-dir", type=Path, default=None)
    tree_tune_parser.add_argument("--output-dir", type=Path, default=None)

    sql_ml_train_parser = _add_command_parser(
        subparsers,
        "sql-ml-train-table",
        help_text="Materialize a profile-table-shaped SQL query and train a backend-aware modal surrogate on it.",
        description=(
            "Use a profile-table-shaped SQL result as the data source for backend-aware "
            "modal surrogate training without rewriting the dataset outside the engine."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine sql-ml-train-table artifacts/local.db "
            "\"select * from profiles\" --backend torch --modes 8 --epochs 20 --device cpu"
        ),
    )
    sql_ml_train_parser.add_argument("database", help="SQLite file path or database URL.")
    sql_ml_train_parser.add_argument("query", help="SQL statement that yields a profile-table-shaped result.")
    sql_ml_train_parser.add_argument("--backend", choices=["auto", "torch", "jax", "tensorflow"], default="auto")
    _add_sql_profile_table_materialization_arguments(sql_ml_train_parser)
    sql_ml_train_parser.add_argument("--param", action="append", default=None, help="Bound query parameter in key=value form.")
    sql_ml_train_parser.add_argument("--modes", type=int, default=16)
    sql_ml_train_parser.add_argument("--epochs", type=int, default=20)
    sql_ml_train_parser.add_argument("--batch-size", type=int, default=64)
    sql_ml_train_parser.add_argument("--normalize", action="store_true")
    sql_ml_train_parser.add_argument("--device", default="auto")
    sql_ml_train_parser.add_argument("--export-dir", type=Path, default=None)
    sql_ml_train_parser.add_argument("--output-dir", type=Path, default=None)

    sql_ml_eval_parser = _add_command_parser(
        subparsers,
        "sql-ml-evaluate-table",
        help_text="Materialize a profile-table-shaped SQL query and evaluate a backend-aware modal surrogate on it.",
        description=(
            "Run the shared SQL-to-profile-table-to-modal-surrogate path and emit evaluation "
            "metrics, reconstructions, and backend-tagged artifacts."
        ),
        epilog=(
            "Example:\n"
            "  spectral-packet-engine sql-ml-evaluate-table artifacts/local.db "
            "\"select * from profiles\" --backend torch --modes 8 --epochs 20 --device cpu "
            "--output-dir artifacts/sql-ml-eval"
        ),
    )
    sql_ml_eval_parser.add_argument("database", help="SQLite file path or database URL.")
    sql_ml_eval_parser.add_argument("query", help="SQL statement that yields a profile-table-shaped result.")
    sql_ml_eval_parser.add_argument("--backend", choices=["auto", "torch", "jax", "tensorflow"], default="auto")
    _add_sql_profile_table_materialization_arguments(sql_ml_eval_parser)
    sql_ml_eval_parser.add_argument("--param", action="append", default=None, help="Bound query parameter in key=value form.")
    sql_ml_eval_parser.add_argument("--modes", type=int, default=16)
    sql_ml_eval_parser.add_argument("--epochs", type=int, default=20)
    sql_ml_eval_parser.add_argument("--batch-size", type=int, default=64)
    sql_ml_eval_parser.add_argument("--normalize", action="store_true")
    sql_ml_eval_parser.add_argument("--device", default="auto")
    sql_ml_eval_parser.add_argument("--export-dir", type=Path, default=None)
    sql_ml_eval_parser.add_argument("--output-dir", type=Path, default=None)

    api_parser = subparsers.add_parser("serve-api", help="Run the optional FastAPI service.")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8000)

    mcp_parser = _add_command_parser(
        subparsers,
        "serve-mcp",
        help_text="Run the MCP server over stdio or streamable HTTP with bounded in-process execution.",
        description=(
            "Serve the MCP surface over stdio or streamable HTTP. "
            "Stdout is reserved for protocol traffic only when stdio transport is used."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine serve-mcp\n"
            "  spectral-packet-engine serve-mcp --max-concurrent-tasks 1 --log-level warning\n"
            "  spectral-packet-engine serve-mcp --transport streamable-http --host 127.0.0.1 --port 8765\n"
            "  spectral-packet-engine serve-mcp --log-file logs/mcp.log"
        ),
    )
    mcp_parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio", help="MCP transport to expose.")
    mcp_parser.add_argument("--host", default="127.0.0.1", help="Bind host for streamable HTTP transport.")
    mcp_parser.add_argument("--port", type=int, default=8000, help="Bind port for streamable HTTP transport.")
    mcp_parser.add_argument("--streamable-http-path", default="/mcp", help="Path to mount when using streamable HTTP transport.")
    mcp_parser.add_argument("--max-concurrent-tasks", type=int, default=DEFAULT_MCP_MAX_CONCURRENT_TASKS, help="Maximum number of compute-heavy MCP tools that may execute concurrently in-process.")
    mcp_parser.add_argument("--slot-timeout-seconds", type=float, default=60.0, help="How long a queued MCP request waits for an execution slot before failing clearly.")
    mcp_parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default=DEFAULT_MCP_LOG_LEVEL, help="Repository-managed MCP log level. Logs are written to stderr unless --log-file is provided.")
    mcp_parser.add_argument("--log-file", type=Path, default=None, help="Optional log file for MCP runtime events and failures.")
    mcp_parser.add_argument("--scratch-dir", type=Path, default=None, help="Optional managed scratch directory for MCP helper tools.")
    mcp_parser.add_argument("--allowed-host", action="append", default=None, help="Additional Host header value to accept for streamable HTTP deployments. Repeat as needed.")
    mcp_parser.add_argument("--allowed-origin", action="append", default=None, help="Additional Origin value to accept for streamable HTTP deployments. Repeat as needed.")
    mcp_parser.add_argument(
        "--allow-unsafe-python",
        action="store_true",
        help="Enable the trusted-only execute_python MCP tool. Disabled by default.",
    )
    # Security limits
    mcp_parser.add_argument("--max-database-size-mb", type=float, default=256.0, help="Max SQLite database file size in MB.")
    mcp_parser.add_argument("--max-scratch-databases", type=int, default=20, help="Max number of scratch databases.")
    mcp_parser.add_argument("--max-script-length", type=int, default=500000, help="Max SQL script length in characters.")
    mcp_parser.add_argument("--max-script-statements", type=int, default=5000, help="Max statements per SQL script.")
    mcp_parser.add_argument("--max-query-seconds", type=float, default=30.0, help="Max seconds per SQL query.")
    mcp_parser.add_argument("--max-pivot-cardinality", type=int, default=500, help="Max distinct values for pivot.")
    mcp_parser.add_argument("--max-interpolation-steps", type=int, default=100000, help="Max steps for time series interpolation.")
    mcp_parser.add_argument("--rate-limit-per-minute", type=int, default=120, help="Max tool calls per minute.")
    mcp_parser.add_argument("--storage-protection-window-seconds", type=float, default=86400.0, help="Seconds needed to refill one protected-database worth of mutation budget.")
    mcp_parser.add_argument("--storage-seed-bytes", type=int, default=8 * 1024 * 1024, help="Bootstrap mutation budget when no protected database mass exists yet.")
    mcp_parser.add_argument("--storage-minimum-mutation-cost-bytes", type=int, default=4096, help="Minimum charged mutation size for managed SQLite writes.")
    mcp_parser.add_argument("--storage-snapshot-retention", type=int, default=8, help="How many historical managed-database snapshots to keep.")
    mcp_parser.add_argument("--disable-managed-db-restore", action="store_true", help="Disable startup restore of managed SQLite databases from snapshots.")

    _add_command_parser(
        subparsers,
        "diagnose",
        help_text="Pre-flight check: detect hosting issues before they happen.",
        description=(
            "Run platform-aware diagnostics to catch common errors that would "
            "break MCP hosting, API serving, or heavy computation. Reports "
            "missing dependencies, encoding issues, permission problems, and "
            "platform-specific gotchas with suggested fixes."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine diagnose\n"
            "  python -m spectral_packet_engine.cli diagnose"
        ),
    )

    _add_command_parser(
        subparsers,
        "generate-mcp-config",
        help_text="Generate an MCP client config block for Claude Desktop or VS Code.",
        description=(
            "Prints a JSON configuration block you can paste into claude_desktop_config.json "
            "or .vscode/mcp.json to register this server with an AI client."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine generate-mcp-config\n"
            "  spectral-packet-engine generate-mcp-config >> claude_desktop_config.json"
        ),
    )
    generate_mcp_config_parser = subparsers.choices["generate-mcp-config"]
    generate_mcp_config_parser.add_argument("--transport", choices=["stdio", "ssh"], default="stdio")
    generate_mcp_config_parser.add_argument("--python-executable", default="python3")
    generate_mcp_config_parser.add_argument("--host", default=None, help="Remote SSH host when --transport ssh is used.")
    generate_mcp_config_parser.add_argument("--remote-cwd", default=None, help="Remote working directory for the SSH target.")
    generate_mcp_config_parser.add_argument("--max-concurrent-tasks", type=int, default=DEFAULT_MCP_MAX_CONCURRENT_TASKS)
    generate_mcp_config_parser.add_argument("--slot-timeout-seconds", type=float, default=60.0)
    generate_mcp_config_parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default=DEFAULT_MCP_LOG_LEVEL)
    generate_mcp_config_parser.add_argument("--allow-unsafe-python", action="store_true")
    generate_mcp_config_parser.add_argument(
        "--source-checkout",
        action="store_true",
        help="Force PYTHONPATH=src in the generated config instead of relying on an installed package.",
    )

    _add_command_parser(
        subparsers,
        "plan-mcp-tunnel",
        help_text="Print a reproducible SSH tunnel plan for a remote streamable-HTTP MCP endpoint.",
        description=(
            "Build the exact ssh -L command and local/remote endpoint URLs for connecting "
            "to a remote streamable-HTTP MCP deployment through an SSH tunnel."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine plan-mcp-tunnel --host user@example-host\n"
            "  spectral-packet-engine plan-mcp-tunnel --host user@example-host --local-port 9876 --remote-port 8765"
        ),
    )
    plan_mcp_tunnel_parser = subparsers.choices["plan-mcp-tunnel"]
    plan_mcp_tunnel_parser.add_argument("--host", required=True, help="SSH host that can reach the remote MCP service.")
    plan_mcp_tunnel_parser.add_argument("--local-port", type=int, default=8765, help="Local forwarded port.")
    plan_mcp_tunnel_parser.add_argument("--remote-port", type=int, default=8765, help="Remote MCP service port.")
    plan_mcp_tunnel_parser.add_argument("--remote-host", default="127.0.0.1", help="Remote bind host seen from the SSH target.")
    plan_mcp_tunnel_parser.add_argument("--streamable-http-path", default="/mcp", help="Remote MCP HTTP path.")

    _add_command_parser(
        subparsers,
        "plan-mcp-ingress",
        help_text="Render a public-hosting plan that keeps MCP bind/origin/nginx settings aligned.",
        description=(
            "Build one shared deployment contract for public hostnames, allowed Host/Origin values, "
            "the canonical public endpoint, and the nginx reverse-proxy snippet that should front the "
            "private MCP listener."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine plan-mcp-ingress --public-host lightcap.ai\n"
            "  spectral-packet-engine plan-mcp-ingress --public-host lightcap.ai --public-host www.lightcap.ai "
            "--upstream-port 8765 --root-redirect-path /mcp/server_info"
        ),
    )
    plan_mcp_ingress_parser = subparsers.choices["plan-mcp-ingress"]
    plan_mcp_ingress_parser.add_argument("--public-host", action="append", required=True, help="Public hostname served by the reverse proxy. Repeat as needed.")
    plan_mcp_ingress_parser.add_argument("--public-scheme", choices=["http", "https"], default="https", help="External scheme clients use when reaching the MCP endpoint.")
    plan_mcp_ingress_parser.add_argument("--public-port", type=int, default=None, help="External port if it differs from the scheme default.")
    plan_mcp_ingress_parser.add_argument("--streamable-http-path", default="/mcp", help="Published MCP HTTP mount path.")
    plan_mcp_ingress_parser.add_argument("--upstream-host", default="127.0.0.1", help="Private upstream host that nginx should proxy to.")
    plan_mcp_ingress_parser.add_argument("--upstream-port", type=int, default=8765, help="Private upstream port that nginx should proxy to.")
    plan_mcp_ingress_parser.add_argument(
        "--root-redirect-path",
        default="/mcp/server_info",
        help="Optional site-root redirect target. Pass empty string to leave / unmanaged.",
    )

    probe_mcp_parser = _add_command_parser(
        subparsers,
        "probe-mcp",
        help_text="Run a controlled MCP self-probe suite and write structured artifacts.",
        description=(
            "Starts the local MCP server through the documented stdio entrypoint, runs a "
            "small adversarial-but-safe probe suite, and optionally writes a reproducible "
            "artifact bundle with tool-call logs and the server log."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine probe-mcp --output-dir artifacts/mcp_probe\n"
            "  spectral-packet-engine probe-mcp --allow-unsafe-python --output-dir artifacts/mcp_probe_trusted"
        ),
    )
    probe_mcp_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mcp_probe"))
    probe_mcp_parser.add_argument("--python-executable", default="python3")
    probe_mcp_parser.add_argument("--max-concurrent-tasks", type=int, default=1)
    probe_mcp_parser.add_argument("--slot-timeout-seconds", type=float, default=60.0)
    probe_mcp_parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default=DEFAULT_MCP_LOG_LEVEL)
    probe_mcp_parser.add_argument("--allow-unsafe-python", action="store_true")
    probe_mcp_parser.add_argument(
        "--profile",
        choices=["smoke", "stress", "audit"],
        default="smoke",
        help="Probe depth: smoke for fast checks, stress for repeated/heavier checks, audit for both.",
    )
    probe_mcp_parser.add_argument(
        "--skip-nested-probe",
        action="store_true",
        help="Skip the nested probe_mcp_runtime check. Useful when probe-mcp is itself running under probe_mcp_runtime.",
    )
    probe_mcp_parser.add_argument(
        "--source-checkout",
        action="store_true",
        help="Force PYTHONPATH=src for the probe target instead of relying on an installed package.",
    )

    install_mcp_service_parser = _add_command_parser(
        subparsers,
        "install-mcp-service",
        help_text="Install an auto-restarting user service manifest for the MCP server.",
        description=(
            "Writes a user-level systemd or launchd manifest that restarts the MCP server "
            "after crashes or machine restarts. This does not modify existing CLI or Python "
            "workflows; it only adds an opt-in supervised runtime path."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine install-mcp-service --dry-run\n"
            "  spectral-packet-engine install-mcp-service --yes\n"
            "  spectral-packet-engine install-mcp-service --yes --enable"
        ),
    )
    install_mcp_service_parser.add_argument("--label", default="dev.spectral-packet-engine.mcp")
    install_mcp_service_parser.add_argument("--working-directory", type=Path, default=Path.cwd())
    install_mcp_service_parser.add_argument("--python-executable", default=sys.executable)
    install_mcp_service_parser.add_argument("--host", default="127.0.0.1")
    install_mcp_service_parser.add_argument("--port", type=int, default=8765)
    install_mcp_service_parser.add_argument("--streamable-http-path", default="/mcp")
    install_mcp_service_parser.add_argument("--max-concurrent-tasks", type=int, default=DEFAULT_MCP_MAX_CONCURRENT_TASKS)
    install_mcp_service_parser.add_argument("--slot-timeout-seconds", type=float, default=60.0)
    install_mcp_service_parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default=DEFAULT_MCP_LOG_LEVEL)
    install_mcp_service_parser.add_argument("--log-file", type=Path, default=None)
    install_mcp_service_parser.add_argument("--scratch-dir", type=Path, default=None)
    install_mcp_service_parser.add_argument("--allowed-host", action="append", default=None)
    install_mcp_service_parser.add_argument("--allowed-origin", action="append", default=None)
    install_mcp_service_parser.add_argument("--allow-unsafe-python", action="store_true")
    install_mcp_service_parser.add_argument("--source-checkout", action="store_true")
    install_mcp_service_parser.add_argument("--dry-run", action="store_true")
    install_mcp_service_parser.add_argument("--enable", action="store_true")
    install_mcp_service_parser.add_argument("--yes", action="store_true")

    # --- Spectral load modeling commands ---

    load_analyze_parser = _add_command_parser(
        subparsers,
        "load-analyze",
        help_text="Analyze server request load using spectral decomposition.",
        description=(
            "Read request timestamps from a file (one per line) or stdin and run a full "
            "spectral load analysis: decomposition, traffic classification, adaptive throttling, "
            "and capacity estimation."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine load-analyze timestamps.txt --capacity 100\n"
            "  spectral-packet-engine load-analyze timestamps.txt --window 600 --modes 128\n"
            "  spectral-packet-engine load-analyze timestamps.txt --baseline baseline.txt"
        ),
    )
    load_analyze_parser.add_argument("timestamps_file", type=Path, help="File with one timestamp per line.")
    load_analyze_parser.add_argument("--window", type=float, default=300.0, help="Observation window in seconds.")
    load_analyze_parser.add_argument("--resolution", type=int, default=256, help="Grid resolution.")
    load_analyze_parser.add_argument("--modes", type=int, default=64, help="Number of spectral modes.")
    load_analyze_parser.add_argument("--capacity", type=float, default=100.0, help="Server capacity in req/s.")
    load_analyze_parser.add_argument("--baseline", type=Path, default=None, help="Baseline timestamps file for anomaly detection.")
    load_analyze_parser.add_argument("--device", default="cpu")

    load_throttle_parser = _add_command_parser(
        subparsers,
        "load-throttle",
        help_text="Compute adaptive throttling parameters from a rate time series.",
        description=(
            "Read a rate time series from a file (one rate value per line) and compute "
            "spectral-derived throttling parameters: cooldown, interval, concurrency."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine load-throttle rates.txt --capacity 100\n"
            "  spectral-packet-engine load-throttle rates.txt --window 600 --modes 64"
        ),
    )
    load_throttle_parser.add_argument("rate_file", type=Path, help="File with one rate value per line.")
    load_throttle_parser.add_argument("--window", type=float, default=300.0, help="Window duration in seconds.")
    load_throttle_parser.add_argument("--modes", type=int, default=64, help="Number of spectral modes.")
    load_throttle_parser.add_argument("--capacity", type=float, default=100.0, help="Server capacity in req/s.")
    load_throttle_parser.add_argument("--device", default="cpu")

    load_capacity_parser = _add_command_parser(
        subparsers,
        "load-capacity",
        help_text="Estimate sustainable server capacity from a rate time series.",
        description=(
            "Separate sustained load (low-frequency modes) from burst spikes (high-frequency "
            "modes) using spectral decomposition."
        ),
        epilog=(
            "Examples:\n"
            "  spectral-packet-engine load-capacity rates.txt --modes 64"
        ),
    )
    load_capacity_parser.add_argument("rate_file", type=Path, help="File with one rate value per line.")
    load_capacity_parser.add_argument("--window", type=float, default=300.0, help="Window duration in seconds.")
    load_capacity_parser.add_argument("--modes", type=int, default=64, help="Number of spectral modes.")
    load_capacity_parser.add_argument("--device", default="cpu")

    return parser


def _run(args, parser: argparse.ArgumentParser) -> int:
    del parser

    if args.command in {"inspect-product", "product"}:
        _emit(inspect_product_identity())
        return 0

    if args.command in {"guide-workflow", "recommend-workflow"}:
        _emit(guide_workflow(surface="cli", input_kind=args.input_kind, goal=args.goal))
        return 0

    if args.command in {"inspect-environment", "env"}:
        _emit(inspect_environment(args.device))
        return 0

    if args.command == "inspect-artifacts":
        _emit(inspect_artifact_directory(args.output_dir))
        return 0

    if args.command == "validate-install":
        _emit(validate_installation(args.device))
        return 0

    if args.command == "release-gate":
        _emit(
            run_release_gate(
                device=args.device,
                include_api=not args.skip_api,
                include_mcp=not args.skip_mcp,
            )
        )
        return 0

    if args.command == "ml-backends":
        _emit(inspect_ml_backend_support(args.device))
        return 0

    if args.command == "tree-backends":
        _emit(inspect_tree_backend_support(requested_library=args.library))
        return 0

    if args.command == "forward":
        summary = simulate_gaussian_packet(
            center=args.center,
            width=args.width,
            wavenumber=args.wavenumber,
            phase=args.phase,
            times=args.times,
            num_modes=args.modes,
            quadrature_points=args.quadrature,
            grid_points=args.grid,
            device=args.device,
        )
        if args.output_dir is not None:
            write_forward_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "project":
        _emit(
            project_gaussian_packet(
                center=args.center,
                width=args.width,
                wavenumber=args.wavenumber,
                phase=args.phase,
                num_modes=args.modes,
                quadrature_points=args.quadrature,
                grid_points=args.grid,
                device=args.device,
            )
        )
        return 0

    if args.command in {"inspect-profile-table", "inspect-table"}:
        _emit(summarize_profile_table(_load_profile_table_from_cli_args(args.table_path, args), device=args.device))
        return 0

    if args.command == "profile-report":
        report = build_profile_table_report(
            _load_profile_table_from_cli_args(args.table_path, args),
            analyze_num_modes=args.analyze_modes,
            compress_num_modes=args.compress_modes,
            device=args.device,
            normalize_each_profile=args.normalize,
        )
        if args.output_dir is not None:
            report.write_artifacts(
                args.output_dir,
                metadata={"input": {"table_path": str(args.table_path)}},
            )
        _emit(report)
        return 0

    if args.command == "export-features":
        requested_includes = {"coefficients", "moments"} if not args.include else set(args.include)
        summary = export_feature_table_from_profile_table(
            _load_profile_table_from_cli_args(args.table_path, args),
            num_modes=args.modes,
            device=args.device,
            normalize_each_profile=args.normalize,
            include_coefficients="coefficients" in requested_includes,
            include_moments="moments" in requested_includes,
            format=args.format,
        )
        if args.output_dir is not None:
            write_feature_table_artifacts(
                args.output_dir,
                summary,
                metadata={"input": {"table_path": str(args.table_path)}},
            )
            summary = replace(summary, output_path=str(args.output_dir / f"features.{args.format}"))
        _emit(summary)
        return 0

    if args.command in {"inspect-tabular-dataset", "tabular-inspect"}:
        _emit(load_tabular_dataset_from_path(args.dataset_path))
        return 0

    if args.command == "ingest-open-meteo":
        result = ingest_open_meteo_hourly_dataset(
            latitude=args.latitude,
            longitude=args.longitude,
            start_date=args.start_date,
            end_date=args.end_date,
            hourly_variables=args.hourly_variables,
            api_kind=args.api_kind,
            timezone_name=args.timezone_name,
            models=args.models,
            temperature_unit=args.temperature_unit,
            wind_speed_unit=args.wind_speed_unit,
            precipitation_unit=args.precipitation_unit,
            cell_selection=args.cell_selection,
            elevation=args.elevation,
            timeout_seconds=args.timeout_seconds,
            **_profile_table_load_kwargs(args),
        )
        if args.profile_output is not None and result.profile_table is None:
            raise ValueError("--profile-output requires explicit profile-table materialization arguments")
        output_path = None
        profile_output = None
        if args.output_path is not None:
            output_path = str(save_tabular_dataset(result.dataset, args.output_path))
        if args.profile_output is not None:
            profile_output = str(save_profile_table(result.profile_table, args.profile_output))
        _emit(
            {
                "api_kind": result.api_kind,
                "request_url": result.request_url,
                "latitude": result.latitude,
                "longitude": result.longitude,
                "elevation": result.elevation,
                "timezone": result.timezone,
                "utc_offset_seconds": result.utc_offset_seconds,
                "hourly_variables": result.hourly_variables,
                "hourly_units": result.hourly_units,
                "dataset": result.dataset_summary,
                "profile_materialization": result.profile_materialization,
                "profile_table": result.profile_summary,
                "output_path": output_path,
                "profile_output": profile_output,
            }
        )
        return 0

    if args.command == "derive-urban-microclimate":
        result = derive_urban_microclimate_from_tabular_dataset(
            args.dataset_path,
            mapping_profile=args.mapping_profile,
            air_temperature_column=args.air_temperature_column,
            relative_humidity_column=args.relative_humidity_column,
            wind_speed_column=args.wind_speed_column,
            pressure_column=args.pressure_column,
            shortwave_radiation_column=args.shortwave_radiation_column,
            longwave_radiation_column=args.longwave_radiation_column,
            boundary_layer=UrbanBoundaryLayerConfig(
                measurement_height_m=args.measurement_height_m,
                zero_plane_displacement_m=args.zero_plane_displacement_m,
                roughness_length_m=args.roughness_length_m,
                thermal_roughness_length_m=args.thermal_roughness_length_m,
                min_wind_speed_m_s=args.min_wind_speed_m_s,
                reference_pressure_pa=args.reference_pressure_pa,
            ),
            radiative_transfer=UrbanRadiativeTransferConfig(
                shortwave_absorptivity=args.shortwave_absorptivity,
                longwave_emissivity=args.longwave_emissivity,
                shortwave_projected_area_factor=args.shortwave_projected_area_factor,
                sky_view_factor=args.sky_view_factor,
            ),
            human_thermal_response=HumanThermalResponseConfig(
                skin_temperature_c=args.skin_temperature_c,
                clothing_insulation_clo=args.clothing_insulation_clo,
                metabolic_heat_flux_w_m2=args.metabolic_heat_flux_w_m2,
            ),
        )
        output_path = None if args.output_path is None else str(save_tabular_dataset(result.dataset, args.output_path))
        _emit(
            {
                "summary": result.summary,
                "mapping": result.mapping,
                "boundary_layer": result.boundary_layer,
                "radiative_transfer": result.radiative_transfer,
                "human_thermal_response": result.human_thermal_response,
                "output_path": output_path,
            }
        )
        return 0

    if args.command in {"analyze-profile-table", "analyze-table"}:
        summary = analyze_profile_table_spectra(
            _load_profile_table_from_cli_args(args.table_path, args),
            num_modes=args.modes,
            device=args.device,
            normalize_each_profile=args.normalize,
        )
        if args.output_dir is not None:
            write_spectral_analysis_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command in {"compress-profile-table", "compress-table", "compress-csv"}:
        table = _load_profile_table_from_cli_args(args.table_path, args)
        summary = compress_profile_table(
            table,
            num_modes=args.modes,
            device=args.device,
            normalize_each_profile=args.normalize,
        )
        if args.output_dir is not None:
            write_compression_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "compression-sweep":
        summary = sweep_profile_table_compression(
            _load_profile_table_from_cli_args(args.table_path, args),
            mode_counts=args.mode_counts,
            device=args.device,
            normalize_each_profile=args.normalize,
        )
        if args.output_dir is not None:
            write_compression_sweep_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command in {"fit-profile-table", "fit-table", "fit-csv"}:
        table = _load_profile_table_from_cli_args(args.table_path, args)
        summary = fit_gaussian_packet_to_profile_table(
            table,
            initial_guess={
                "center": args.center,
                "width": args.width,
                "wavenumber": args.wavenumber,
                "phase": args.phase,
            },
            num_modes=args.modes,
            quadrature_points=args.quadrature,
            device=args.device,
            steps=args.steps,
            learning_rate=args.learning_rate,
        )
        if args.output_dir is not None:
            write_inverse_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command in {"infer-potential-spectrum", "spectroscopy-workflow"}:
        raw_initial_guesses = _parse_json_object(args.initial_guesses, flag_name="--initial-guesses")
        summary = run_spectroscopy_workflow(
            target_eigenvalues=args.target_eigenvalues,
            families=args.family,
            initial_guesses=None if not raw_initial_guesses else _coerce_nested_float_mapping(raw_initial_guesses, flag_name="--initial-guesses"),
            domain_length=args.domain_length,
            left=args.left,
            mass=args.mass,
            hbar=args.hbar,
            num_points=args.points,
            optimization_config=GradientOptimizationConfig(
                steps=args.steps,
                learning_rate=args.learning_rate,
            ),
            device=args.device,
        )
        if args.output_dir is not None:
            write_vertical_workflow_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "analyze-separable-spectrum":
        summary = analyze_separable_tensor_product_spectrum(
            family_x=args.family_x,
            parameters_x=_coerce_float_mapping(_parse_json_object(args.params_x, flag_name="--params-x"), flag_name="--params-x"),
            family_y=args.family_y,
            parameters_y=_coerce_float_mapping(_parse_json_object(args.params_y, flag_name="--params-y"), flag_name="--params-y"),
            domain_length_x=args.domain_length_x,
            domain_length_y=args.domain_length_y,
            num_points_x=args.points_x,
            num_points_y=args.points_y,
            num_states_x=args.states_x,
            num_states_y=args.states_y,
            num_combined_states=args.combined_states,
            low_rank_rank=args.low_rank_rank,
            device=args.device,
        )
        if args.output_dir is not None:
            write_reduced_model_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "analyze-coupled-surfaces":
        summary = analyze_coupled_channel_surfaces(
            domain_length=args.domain_length,
            grid_points=args.grid_points,
            slope=args.slope,
            bias=args.bias,
            coupling=args.coupling,
            coupling_width=args.coupling_width,
            device=args.device,
        )
        if args.output_dir is not None:
            write_reduced_model_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "solve-radial-reduction":
        summary = solve_radial_reduction(
            family=args.family,
            parameters=_coerce_float_mapping(_parse_json_object(args.params, flag_name="--params"), flag_name="--params"),
            angular_momentum=args.angular_momentum,
            radial_min=args.radial_min,
            radial_max=args.radial_max,
            num_points=args.points,
            num_states=args.states,
            mass=args.mass,
            hbar=args.hbar,
            device=args.device,
        )
        if args.output_dir is not None:
            write_reduced_model_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command in {"compare-profile-tables", "compare-tables"}:
        summary = compare_profile_tables(
            _load_profile_table_from_cli_args(args.reference_table_path, args),
            _load_profile_table_from_cli_args(args.candidate_table_path, args),
            device=args.device,
        )
        if args.output_dir is not None:
            write_profile_comparison_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command in {"inspect-database", "db-inspect"}:
        _emit(inspect_database(args.database))
        return 0

    if args.command in {"bootstrap-database", "db-bootstrap"}:
        _emit(bootstrap_local_database(args.database))
        return 0

    if args.command in {"list-database-tables", "db-list-tables"}:
        inspection = inspect_database(args.database)
        _emit({"redacted_url": inspection.capability.redacted_url, "tables": inspection.tables})
        return 0

    if args.command in {"describe-database-table", "db-describe-table"}:
        _emit(describe_database_table(args.database, args.table_name))
        return 0

    if args.command in {"query-database", "db-query"}:
        parameters = _parse_key_value_items(args.param)
        result = materialize_database_query(args.database, args.query, parameters=parameters)
        if args.output_dir is not None:
            write_tabular_artifacts(
                args.output_dir,
                result.dataset,
                summary_name="db_query_summary.json",
                table_name="query_result.csv",
                metadata=database_query_workflow_artifact_metadata(
                    "db-query",
                    args.database,
                    args.query,
                    parameters=parameters,
                ),
            )
        _emit(summarize_database_query_result(args.database, args.query, result, parameters=parameters))
        return 0

    if args.command in {"execute-database-statement", "db-exec"}:
        parameters = _parse_key_value_items(args.param)
        _emit(
            execute_database_statement(
                args.database,
                args.statement,
                parameters=parameters,
                create_if_missing=True,
            )
        )
        return 0

    if args.command in {"execute-database-script", "db-script"}:
        _emit(
            execute_database_script(
                args.database,
                args.script,
                create_if_missing=True,
            )
        )
        return 0

    if args.command in {"write-database-table", "db-write-table"}:
        summary = write_tabular_dataset_to_database(
            args.database,
            args.table_name,
            load_tabular_dataset(args.dataset_path),
            if_exists=args.if_exists,
        )
        _emit(summary)
        return 0

    if args.command in {"materialize-database-query", "db-materialize-query"}:
        parameters = _parse_key_value_items(args.param)
        summary = materialize_database_query_to_table(
            args.database,
            args.table_name,
            args.query,
            parameters=parameters,
            replace=args.replace,
        )
        _emit(summary)
        return 0

    if args.command == "sql-analyze-table":
        parameters = _parse_key_value_items(args.param)
        summary = analyze_profile_table_from_database_query(
            args.database,
            args.query,
            parameters=parameters,
            **_profile_table_load_kwargs(args),
            num_modes=args.modes,
            device=args.device,
            normalize_each_profile=args.normalize,
        )
        if args.output_dir is not None:
            write_spectral_analysis_artifacts(
                args.output_dir,
                summary,
                metadata=database_profile_query_workflow_artifact_metadata(
                    "sql-analyze-table",
                    args.database,
                    args.query,
                    parameters=parameters,
                    **_profile_table_load_kwargs(args),
                ),
            )
        _emit(summary)
        return 0

    if args.command == "sql-compress-table":
        parameters = _parse_key_value_items(args.param)
        summary = compress_profile_table_from_database_query(
            args.database,
            args.query,
            parameters=parameters,
            **_profile_table_load_kwargs(args),
            num_modes=args.modes,
            device=args.device,
            normalize_each_profile=args.normalize,
        )
        if args.output_dir is not None:
            write_compression_artifacts(
                args.output_dir,
                summary,
                metadata=database_profile_query_workflow_artifact_metadata(
                    "sql-compress-table",
                    args.database,
                    args.query,
                    parameters=parameters,
                    **_profile_table_load_kwargs(args),
                ),
            )
        _emit(summary)
        return 0

    if args.command == "sql-export-features":
        parameters = _parse_key_value_items(args.param)
        requested_includes = {"coefficients", "moments"} if not args.include else set(args.include)
        summary = export_feature_table_from_database_query(
            args.database,
            args.query,
            parameters=parameters,
            **_profile_table_load_kwargs(args),
            num_modes=args.modes,
            device=args.device,
            normalize_each_profile=args.normalize,
            include_coefficients="coefficients" in requested_includes,
            include_moments="moments" in requested_includes,
            format=args.format,
        )
        if args.output_dir is not None:
            write_feature_table_artifacts(
                args.output_dir,
                summary,
                metadata=database_profile_query_workflow_artifact_metadata(
                    "export-features",
                    args.database,
                    args.query,
                    parameters=parameters,
                    **_profile_table_load_kwargs(args),
                ),
            )
            summary = replace(summary, output_path=str(args.output_dir / f"features.{args.format}"))
        _emit(summary)
        return 0

    if args.command == "sql-profile-report":
        parameters = _parse_key_value_items(args.param)
        report = build_profile_table_report_from_database_query(
            args.database,
            args.query,
            parameters=parameters,
            **_profile_table_load_kwargs(args),
            analyze_num_modes=args.analyze_modes,
            compress_num_modes=args.compress_modes,
            device=args.device,
            normalize_each_profile=args.normalize,
        )
        if args.output_dir is not None:
            report.write_artifacts(
                args.output_dir,
                metadata=database_profile_query_workflow_artifact_metadata(
                    "profile-report",
                    args.database,
                    args.query,
                    parameters=parameters,
                    **_profile_table_load_kwargs(args),
                ),
            )
        _emit(report)
        return 0

    if args.command == "sql-fit-table":
        parameters = _parse_key_value_items(args.param)
        summary = fit_gaussian_packet_to_profile_table_from_database_query(
            args.database,
            args.query,
            parameters=parameters,
            initial_guess={
                "center": args.center,
                "width": args.width,
                "wavenumber": args.wavenumber,
                "phase": args.phase,
            },
            **_profile_table_load_kwargs(args),
            num_modes=args.modes,
            quadrature_points=args.quadrature,
            device=args.device,
            steps=args.steps,
            learning_rate=args.learning_rate,
        )
        if args.output_dir is not None:
            write_inverse_artifacts(
                args.output_dir,
                summary,
                metadata=database_profile_query_workflow_artifact_metadata(
                    "sql-fit-table",
                    args.database,
                    args.query,
                    parameters=parameters,
                    **_profile_table_load_kwargs(args),
                ),
            )
        _emit(summary)
        return 0

    if args.command == "packet-sweep":
        if not (len(args.centers) == len(args.widths) == len(args.wavenumbers)):
            raise ValueError("--centers, --widths, and --wavenumbers must have the same length")
        if args.phases is None:
            phases = [0.0] * len(args.centers)
        else:
            phases = args.phases
        if len(phases) != len(args.centers):
            raise ValueError("--phases must be omitted or have the same length as --centers")
        packet_specs = [
            {
                "center": center,
                "width": width,
                "wavenumber": wavenumber,
                "phase": phase,
            }
            for center, width, wavenumber, phase in zip(args.centers, args.widths, args.wavenumbers, phases)
        ]
        summary = simulate_packet_sweep(
            packet_specs,
            times=args.times,
            num_modes=args.modes,
            quadrature_points=args.quadrature,
            grid_points=args.grid,
            device=args.device,
        )
        if args.output_dir is not None:
            write_packet_sweep_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "transport-benchmark":
        summary = benchmark_transport_scan(
            scan_id=args.scan_id,
            mode_counts=args.mode_counts,
            device=args.device,
        )
        if args.output_dir is not None:
            write_transport_benchmark_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "design-transition":
        summary = design_potential_for_target_transition(
            family=args.family,
            target_transition=args.target_transition,
            transition_indices=tuple(args.transition_indices),
            initial_guess=_coerce_float_mapping(
                _parse_json_object(args.initial_guess, flag_name="--initial-guess"),
                flag_name="--initial-guess",
            ),
            domain_length=args.domain_length,
            left=args.left,
            mass=args.mass,
            hbar=args.hbar,
            num_points=args.points,
            num_states=args.states,
            optimization_config=GradientOptimizationConfig(
                steps=args.steps,
                learning_rate=args.learning_rate,
            ),
            device=args.device,
        )
        if args.output_dir is not None:
            write_differentiable_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "optimize-packet-control":
        summary = optimize_packet_control(
            initial_guess={
                "center": args.center,
                "width": args.width,
                "wavenumber": args.wavenumber,
                "phase": args.phase,
            },
            objective=args.objective,
            target_value=args.target_value,
            final_time=args.final_time,
            interval=None if args.interval is None else (args.interval[0], args.interval[1]),
            num_modes=args.modes,
            quadrature_points=args.quadrature,
            grid_points=args.grid,
            optimization_config=GradientOptimizationConfig(
                steps=args.steps,
                learning_rate=args.learning_rate,
            ),
            device=args.device,
        )
        if args.output_dir is not None:
            write_differentiable_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "transport-workflow":
        summary = run_transport_resonance_workflow(
            barrier_height=args.barrier_height,
            barrier_width_sigma=args.barrier_width_sigma,
            domain_length=args.domain_length,
            grid_points=args.grid_points,
            num_modes=args.modes,
            num_energies=args.energies,
            packet_center=args.packet_center,
            packet_width=args.packet_width,
            packet_wavenumber=args.packet_wavenumber,
            device=args.device,
        )
        if args.output_dir is not None:
            write_vertical_workflow_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "profile-inference-workflow":
        summary = run_profile_inference_workflow(
            _load_profile_table_from_cli_args(args.table_path, args),
            initial_guess={
                "center": args.center,
                "width": args.width,
                "wavenumber": args.wavenumber,
                "phase": args.phase,
            },
            analyze_num_modes=args.analyze_modes,
            compress_num_modes=args.compress_modes,
            inverse_num_modes=args.inverse_modes,
            feature_num_modes=args.feature_modes,
            quadrature_points=args.quadrature,
            normalize_each_profile=args.normalize,
            device=args.device,
        )
        if args.output_dir is not None:
            write_vertical_workflow_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command in {"tf-train-table", "tf-train-csv"}:
        table = _load_profile_table_from_cli_args(args.table_path, args)
        summary = train_tensorflow_surrogate_on_profile_table(
            table,
            num_modes=args.modes,
            normalize_each_profile=args.normalize,
            config=TensorFlowRegressorConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
            ),
            export_dir=None if args.export_dir is None else str(args.export_dir),
        )
        if args.output_dir is not None:
            write_tensorflow_training_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "tf-evaluate-table":
        table = _load_profile_table_from_cli_args(args.table_path, args)
        summary = evaluate_tensorflow_surrogate_on_profile_table(
            table,
            num_modes=args.modes,
            normalize_each_profile=args.normalize,
            config=TensorFlowRegressorConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
            ),
            export_dir=None if args.export_dir is None else str(args.export_dir),
        )
        if args.output_dir is not None:
            write_tensorflow_evaluation_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "ml-train-table":
        table = _load_profile_table_from_cli_args(args.table_path, args)
        summary = train_modal_surrogate_on_profile_table(
            table,
            backend=args.backend,
            num_modes=args.modes,
            normalize_each_profile=args.normalize,
            config=ModalSurrogateConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
            ),
            export_dir=None if args.export_dir is None else str(args.export_dir),
        )
        if args.output_dir is not None:
            write_modal_training_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "ml-evaluate-table":
        table = _load_profile_table_from_cli_args(args.table_path, args)
        summary = evaluate_modal_surrogate_on_profile_table(
            table,
            backend=args.backend,
            num_modes=args.modes,
            normalize_each_profile=args.normalize,
            config=ModalSurrogateConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
            ),
            export_dir=None if args.export_dir is None else str(args.export_dir),
        )
        if args.output_dir is not None:
            write_modal_evaluation_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "tree-train":
        export_dir = args.export_dir
        if export_dir is None and args.output_dir is not None:
            export_dir = args.output_dir
        summary = train_tree_model(
            args.features_path,
            target_column=args.target_column,
            feature_columns=args.feature_columns,
            task=args.task,
            library=args.library,
            model=args.model,
            params=_parse_json_object(args.params, flag_name="--params"),
            test_fraction=args.test_fraction,
            random_state=args.random_state,
            export_dir=None if export_dir is None else str(export_dir),
        )
        if args.output_dir is not None:
            write_tree_training_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "tree-tune":
        export_dir = args.export_dir
        if export_dir is None and args.output_dir is not None:
            export_dir = args.output_dir / "best_model"
        summary = tune_tree_model(
            args.features_path,
            target_column=args.target_column,
            feature_columns=args.feature_columns,
            task=args.task,
            library=args.library,
            model=args.model,
            search_space=_parse_json_object(args.search_space, flag_name="--search-space"),
            search_kind=args.search_kind,
            n_iter=args.n_iter,
            cv=args.cv,
            scoring=args.scoring,
            test_fraction=args.test_fraction,
            random_state=args.random_state,
            export_dir=None if export_dir is None else str(export_dir),
        )
        if args.output_dir is not None:
            write_tree_tuning_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "sql-ml-train-table":
        parameters = _parse_key_value_items(args.param)
        summary = train_modal_surrogate_from_database_query(
            args.database,
            args.query,
            backend=args.backend,
            parameters=parameters,
            **_profile_table_load_kwargs(args),
            num_modes=args.modes,
            normalize_each_profile=args.normalize,
            config=ModalSurrogateConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
            ),
            export_dir=None if args.export_dir is None else str(args.export_dir),
        )
        if args.output_dir is not None:
            write_modal_training_artifacts(
                args.output_dir,
                summary,
                metadata=database_profile_query_workflow_artifact_metadata(
                    "sql-ml-train-table",
                    args.database,
                    args.query,
                    parameters=parameters,
                    **_profile_table_load_kwargs(args),
                ),
            )
        _emit(summary)
        return 0

    if args.command == "sql-ml-evaluate-table":
        parameters = _parse_key_value_items(args.param)
        summary = evaluate_modal_surrogate_from_database_query(
            args.database,
            args.query,
            backend=args.backend,
            parameters=parameters,
            **_profile_table_load_kwargs(args),
            num_modes=args.modes,
            normalize_each_profile=args.normalize,
            config=ModalSurrogateConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
            ),
            export_dir=None if args.export_dir is None else str(args.export_dir),
        )
        if args.output_dir is not None:
            write_modal_evaluation_artifacts(
                args.output_dir,
                summary,
                metadata=database_profile_query_workflow_artifact_metadata(
                    "sql-ml-evaluate-table",
                    args.database,
                    args.query,
                    parameters=parameters,
                    **_profile_table_load_kwargs(args),
                ),
            )
        _emit(summary)
        return 0

    if args.command == "serve-api":
        try:
            import uvicorn
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("FastAPI serving requires the 'api' extra.") from exc
        from spectral_packet_engine.api import create_api_app

        uvicorn.run(create_api_app(), host=args.host, port=args.port)
        return 0

    if args.command == "serve-mcp":
        from spectral_packet_engine.mcp import main as mcp_main

        mcp_main(
            MCPServerConfig(
                transport=args.transport,
                host=args.host,
                port=args.port,
                streamable_http_path=args.streamable_http_path,
                max_concurrent_tasks=args.max_concurrent_tasks,
                slot_acquire_timeout_seconds=args.slot_timeout_seconds,
                log_level=args.log_level,
                log_file=None if args.log_file is None else str(args.log_file),
                scratch_directory=None if args.scratch_dir is None else str(args.scratch_dir),
                allowed_hosts=() if args.allowed_host is None else tuple(args.allowed_host),
                allowed_origins=() if args.allowed_origin is None else tuple(args.allowed_origin),
                allow_unsafe_python=args.allow_unsafe_python,
                max_database_size_mb=getattr(args, "max_database_size_mb", 256.0),
                max_scratch_databases=getattr(args, "max_scratch_databases", 20),
                max_script_length_chars=getattr(args, "max_script_length", 500000),
                max_script_statements=getattr(args, "max_script_statements", 5000),
                max_query_seconds=getattr(args, "max_query_seconds", 30.0),
                max_pivot_cardinality=getattr(args, "max_pivot_cardinality", 500),
                max_interpolation_steps=getattr(args, "max_interpolation_steps", 100000),
                rate_limit_per_minute=getattr(args, "rate_limit_per_minute", 120),
                storage_protection_window_seconds=getattr(args, "storage_protection_window_seconds", 86400.0),
                storage_seed_bytes=getattr(args, "storage_seed_bytes", 8 * 1024 * 1024),
                storage_minimum_mutation_cost_bytes=getattr(args, "storage_minimum_mutation_cost_bytes", 4096),
                storage_snapshot_retention=getattr(args, "storage_snapshot_retention", 8),
                restore_managed_databases_on_startup=not getattr(args, "disable_managed_db_restore", False),
            )
        )
        return 0

    if args.command == "diagnose":
        from spectral_packet_engine.config import diagnose_hosting_readiness

        report = diagnose_hosting_readiness()
        result = {
            "ready": report.ready,
            "errors": list(report.errors),
            "warnings": list(report.warnings),
            "fixes": list(report.fixes),
            "platform_notes": list(report.platform_notes),
        }
        _emit(result)
        if not report.ready:
            print("\n--- ERRORS ---", file=__import__("sys").stderr)
            for err in report.errors:
                print(f"  ✗ {err}", file=__import__("sys").stderr)
            if report.fixes:
                print("\n--- SUGGESTED FIXES ---", file=__import__("sys").stderr)
                for fix in report.fixes:
                    print(f"  → {fix}", file=__import__("sys").stderr)
        return 0 if report.ready else 1

    if args.command == "generate-mcp-config":
        from spectral_packet_engine.mcp_deployment import (
            build_local_mcp_client_configuration,
            build_ssh_mcp_client_configuration,
        )

        if args.transport == "ssh":
            if args.host is None or args.remote_cwd is None:
                raise ValueError("--host and --remote-cwd are required when --transport ssh is used")
            config = build_ssh_mcp_client_configuration(
                host=args.host,
                remote_working_directory=args.remote_cwd,
                remote_python_executable=args.python_executable,
                max_concurrent_tasks=args.max_concurrent_tasks,
                slot_timeout_seconds=args.slot_timeout_seconds,
                log_level=args.log_level,
                allow_unsafe_python=args.allow_unsafe_python,
                source_checkout=True if args.source_checkout else None,
            ).to_dict()
        else:
            config = build_local_mcp_client_configuration(
                working_directory=Path.cwd(),
                python_executable=args.python_executable,
                max_concurrent_tasks=args.max_concurrent_tasks,
                slot_timeout_seconds=args.slot_timeout_seconds,
                log_level=args.log_level,
                allow_unsafe_python=args.allow_unsafe_python,
                source_checkout=True if args.source_checkout else None,
            ).to_dict()
        print(json.dumps(config, indent=2))
        return 0

    if args.command == "plan-mcp-tunnel":
        from spectral_packet_engine.mcp_deployment import build_mcp_tunnel_plan

        _emit(
            build_mcp_tunnel_plan(
                host=args.host,
                local_port=args.local_port,
                remote_port=args.remote_port,
                remote_host=args.remote_host,
                streamable_http_path=args.streamable_http_path,
            ).to_dict()
        )
        return 0

    if args.command == "plan-mcp-ingress":
        from spectral_packet_engine.mcp_deployment import build_mcp_public_ingress_plan

        _emit(
            build_mcp_public_ingress_plan(
                public_hosts=tuple(args.public_host),
                public_scheme=args.public_scheme,
                public_port=args.public_port,
                streamable_http_path=args.streamable_http_path,
                upstream_host=args.upstream_host,
                upstream_port=args.upstream_port,
                root_redirect_path=None if args.root_redirect_path == "" else args.root_redirect_path,
            ).to_dict()
        )
        return 0

    if args.command == "probe-mcp":
        from spectral_packet_engine.mcp_probe import (
            build_local_probe_server_spec,
            run_mcp_probe_suite,
            write_mcp_probe_artifacts,
        )

        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        server = build_local_probe_server_spec(
            working_directory=Path.cwd(),
            python_executable=args.python_executable,
            max_concurrent_tasks=args.max_concurrent_tasks,
            slot_timeout_seconds=args.slot_timeout_seconds,
            log_level=args.log_level,
            log_file=output_dir / "server.log",
            allow_unsafe_python=args.allow_unsafe_python,
            source_checkout=True if args.source_checkout else None,
        )
        report = run_mcp_probe_suite(
            server,
            expect_unsafe_python_enabled=args.allow_unsafe_python,
            profile=args.profile,
            skip_nested_probe=args.skip_nested_probe,
        )
        write_mcp_probe_artifacts(report, output_dir)
        _emit(report.to_dict())
        return 0

    if args.command == "install-mcp-service":
        from spectral_packet_engine.mcp_deployment import build_mcp_service_install_plan, install_mcp_service

        proceed = args.yes or args.dry_run
        if not proceed:
            proceed = _prompt_yes_no("Install an auto-restarting user service manifest for the MCP server?")
        if not proceed:
            _emit({"installed": False, "reason": "declined"})
            return 0

        plan = build_mcp_service_install_plan(
            working_directory=args.working_directory,
            label=args.label,
            python_executable=args.python_executable,
            host=args.host,
            port=args.port,
            streamable_http_path=args.streamable_http_path,
            max_concurrent_tasks=args.max_concurrent_tasks,
            slot_timeout_seconds=args.slot_timeout_seconds,
            log_level=args.log_level,
            log_file=args.log_file,
            scratch_directory=args.scratch_dir,
            allowed_hosts=() if args.allowed_host is None else tuple(args.allowed_host),
            allowed_origins=() if args.allowed_origin is None else tuple(args.allowed_origin),
            allow_unsafe_python=args.allow_unsafe_python,
            source_checkout=True if args.source_checkout else None,
        )
        if args.dry_run:
            _emit({"installed": False, "dry_run": True, "plan": plan.to_dict(), "manifest": plan.manifest})
            return 0
        _emit(install_mcp_service(plan, enable=args.enable))
        return 0

    # --- Spectral load modeling commands ---

    if args.command == "load-analyze":
        from spectral_packet_engine.load_spectral import analyze_request_load

        timestamps = [float(line.strip()) for line in args.timestamps_file.read_text().splitlines() if line.strip()]
        baseline_ts = None
        if args.baseline is not None:
            baseline_ts = [float(line.strip()) for line in args.baseline.read_text().splitlines() if line.strip()]
        report = analyze_request_load(
            timestamps,
            window_seconds=args.window,
            resolution=args.resolution,
            num_modes=args.modes,
            capacity_rps=args.capacity,
            baseline_timestamps=baseline_ts,
            device=args.device,
        )
        result = {
            "signal": {"window_seconds": report.signal.window_seconds, "total_requests": report.signal.total_requests},
            "spectrum": {
                "decay_type": report.spectrum.decay.decay_type.value,
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
                "headroom_fraction": report.throttle.headroom_fraction,
            },
            "capacity": {
                "sustained_rps": report.capacity.sustained_rps,
                "peak_rps": report.capacity.peak_rps,
                "burst_ratio": report.capacity.burst_ratio,
                "stable": report.capacity.stable,
            },
        }
        if report.anomaly is not None:
            result["anomaly"] = {
                "is_anomalous": report.anomaly.is_anomalous,
                "spectral_distance": report.anomaly.spectral_distance,
                "reason": report.anomaly.reason,
            }
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "load-throttle":
        from spectral_packet_engine.load_spectral import (
            load_signal_from_rate, decompose_load_signal, compute_adaptive_throttle,
        )

        rates = [float(line.strip()) for line in args.rate_file.read_text().splitlines() if line.strip()]
        signal = load_signal_from_rate(rates, window_seconds=args.window, device=args.device)
        coeffs = decompose_load_signal(signal, num_modes=args.modes)
        throttle = compute_adaptive_throttle(coeffs, capacity_rps=args.capacity)
        print(json.dumps({
            "regime": throttle.regime,
            "recommended_cooldown_seconds": throttle.recommended_cooldown_seconds,
            "recommended_min_interval_seconds": throttle.recommended_min_interval_seconds,
            "recommended_max_concurrent": throttle.recommended_max_concurrent,
            "capacity_utilization": throttle.capacity_utilization,
            "spectral_load_factor": throttle.spectral_load_factor,
            "headroom_fraction": throttle.headroom_fraction,
        }, indent=2))
        return 0

    if args.command == "load-capacity":
        from spectral_packet_engine.load_spectral import (
            load_signal_from_rate, decompose_load_signal, estimate_capacity,
        )

        rates = [float(line.strip()) for line in args.rate_file.read_text().splitlines() if line.strip()]
        signal = load_signal_from_rate(rates, window_seconds=args.window, device=args.device)
        coeffs = decompose_load_signal(signal, num_modes=args.modes)
        cap = estimate_capacity(coeffs)
        print(json.dumps({
            "sustained_rps": cap.sustained_rps,
            "peak_rps": cap.peak_rps,
            "burst_ratio": cap.burst_ratio,
            "spectral_headroom_modes": cap.spectral_headroom_modes,
            "stable": cap.stable,
        }, indent=2))
        return 0

    raise ValueError(f"unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return _run(args, parser)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted.\n")
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


__all__ = [
    "build_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
