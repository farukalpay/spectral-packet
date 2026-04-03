from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from spectral_packet_engine.artifacts import (
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
from spectral_packet_engine.tabular import load_tabular_dataset
from spectral_packet_engine.table_io import load_profile_table
from spectral_packet_engine.tf_surrogate import TensorFlowRegressorConfig
from spectral_packet_engine.workflows import (
    analyze_profile_table_spectra,
    analyze_profile_table_from_database_query,
    benchmark_transport_scan,
    bootstrap_local_database,
    compare_profile_tables,
    compress_profile_table,
    describe_database_table,
    evaluate_tensorflow_surrogate_on_profile_table,
    execute_database_query,
    fit_gaussian_packet_to_profile_table,
    inspect_database,
    inspect_environment,
    inspect_ml_backend_support,
    load_tabular_dataset_from_path,
    materialize_database_query,
    materialize_database_query_to_table,
    simulate_packet_sweep,
    project_gaussian_packet,
    simulate_gaussian_packet,
    summarize_profile_table,
    summarize_database_query_result,
    summarize_tabular_dataset,
    sweep_profile_table_compression,
    train_modal_surrogate_from_database_query,
    train_modal_surrogate_on_profile_table,
    evaluate_modal_surrogate_from_database_query,
    evaluate_modal_surrogate_on_profile_table,
    train_tensorflow_surrogate_on_profile_table,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectral-packet-engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Cross-platform spectral packet compute workflows for bounded-domain simulation, "
            "modal decomposition, inverse reconstruction, SQL-backed data ingress, and "
            "backend-aware surrogate evaluation."
        ),
        epilog=(
            "Start here:\n"
            "  spectral-packet-engine validate-install --device cpu\n\n"
            "If you are running from a source checkout, continue with:\n"
            "  spectral-packet-engine compress-table examples/data/synthetic_profiles.csv "
            "--modes 8 --device cpu --output-dir artifacts/compression\n\n"
            "Inspect machine-side surrogate backends with:\n"
            "  spectral-packet-engine ml-backends --device cpu"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = _add_command_parser(
        subparsers,
        "env",
        help_text="Inspect runtime capabilities.",
        epilog="Example:\n  spectral-packet-engine env --device cpu",
    )
    env_parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or mps.")

    validate_parser = _add_command_parser(
        subparsers,
        "validate-install",
        help_text="Validate the current installation and optional surfaces.",
        epilog="Example:\n  spectral-packet-engine validate-install --device cpu",
    )
    validate_parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or mps.")

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

    inspect_table_parser = subparsers.add_parser("inspect-table", help="Inspect a CSV, TSV, JSON, or optionally XLSX profile table.")
    inspect_table_parser.add_argument("table_path", type=Path)
    inspect_table_parser.add_argument("--device", default="auto")

    tabular_inspect_parser = subparsers.add_parser(
        "tabular-inspect",
        help="Inspect a CSV, TSV, JSON, and optionally Parquet or XLSX tabular dataset.",
    )
    tabular_inspect_parser.add_argument("dataset_path", type=Path)

    analyze_table_parser = subparsers.add_parser(
        "analyze-table",
        help="Project a profile table into the modal basis and summarize its spectral structure.",
    )
    analyze_table_parser.add_argument("table_path", type=Path)
    analyze_table_parser.add_argument("--modes", type=int, default=32)
    analyze_table_parser.add_argument("--device", default="auto")
    analyze_table_parser.add_argument("--normalize", action="store_true")
    analyze_table_parser.add_argument("--output-dir", type=Path, default=None)

    compress_parser = subparsers.add_parser(
        "compress-table",
        aliases=["compress-csv"],
        help="Compress a CSV, TSV, JSON, or optionally XLSX profile table into modal coefficients.",
    )
    compress_parser.add_argument("table_path", type=Path)
    compress_parser.add_argument("--modes", type=int, default=32)
    compress_parser.add_argument("--device", default="auto")
    compress_parser.add_argument("--normalize", action="store_true", help="Normalize each profile before compression.")
    compress_parser.add_argument("--output-dir", type=Path, default=None)

    sweep_parser = subparsers.add_parser("compression-sweep", help="Evaluate compression quality across multiple mode counts.")
    sweep_parser.add_argument("table_path", type=Path)
    sweep_parser.add_argument("--mode-counts", type=int, nargs="+", default=[4, 8, 16, 32])
    sweep_parser.add_argument("--device", default="auto")
    sweep_parser.add_argument("--normalize", action="store_true")
    sweep_parser.add_argument("--output-dir", type=Path, default=None)

    fit_parser = subparsers.add_parser(
        "fit-table",
        aliases=["fit-csv"],
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

    compare_parser = subparsers.add_parser(
        "compare-tables",
        help="Compare a candidate profile table against a reference table.",
    )
    compare_parser.add_argument("reference_table_path", type=Path)
    compare_parser.add_argument("candidate_table_path", type=Path)
    compare_parser.add_argument("--device", default="auto")
    compare_parser.add_argument("--output-dir", type=Path, default=None)

    db_inspect_parser = subparsers.add_parser(
        "db-inspect",
        help="Inspect an existing SQLite database path or database URL and list available tables.",
    )
    db_inspect_parser.add_argument("database")

    db_bootstrap_parser = _add_command_parser(
        subparsers,
        "db-bootstrap",
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
        "db-list-tables",
        help="List tables in an existing SQLite database path or database URL.",
    )
    db_list_parser.add_argument("database")

    db_describe_parser = subparsers.add_parser(
        "db-describe-table",
        help="Describe a table schema and row count from an existing database.",
    )
    db_describe_parser.add_argument("database")
    db_describe_parser.add_argument("table_name")

    db_query_parser = _add_command_parser(
        subparsers,
        "db-query",
        help_text="Run a parameterized SQL query and materialize the result as a tabular dataset.",
        description=(
            "Execute a safe parameterized query against a SQLite path or database URL, then "
            "return the result through the shared tabular dataset layer."
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

    db_write_parser = subparsers.add_parser(
        "db-write-table",
        help="Load a tabular dataset from disk and write it into a database table.",
    )
    db_write_parser.add_argument("database")
    db_write_parser.add_argument("table_name")
    db_write_parser.add_argument("dataset_path", type=Path)
    db_write_parser.add_argument("--if-exists", choices=["fail", "replace", "append"], default="fail")

    db_materialize_parser = _add_command_parser(
        subparsers,
        "db-materialize-query",
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
    sql_analyze_parser.add_argument("--time-column", default="time", help="Column to use as the profile-table time axis.")
    sql_analyze_parser.add_argument("--param", action="append", default=None, help="Bound query parameter in key=value form.")
    sql_analyze_parser.add_argument("--modes", type=int, default=32)
    sql_analyze_parser.add_argument("--device", default="auto")
    sql_analyze_parser.add_argument("--normalize", action="store_true")
    sql_analyze_parser.add_argument("--output-dir", type=Path, default=None)

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
    sql_ml_train_parser.add_argument("--time-column", default="time", help="Column to use as the profile-table time axis.")
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
    sql_ml_eval_parser.add_argument("--time-column", default="time", help="Column to use as the profile-table time axis.")
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

    subparsers.add_parser("serve-mcp", help="Run the MCP server over stdio.")
    return parser


def _run(args, parser: argparse.ArgumentParser) -> int:
    del parser

    if args.command == "env":
        _emit(inspect_environment(args.device))
        return 0

    if args.command == "validate-install":
        _emit(validate_installation(args.device))
        return 0

    if args.command == "ml-backends":
        _emit(inspect_ml_backend_support(args.device))
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

    if args.command == "inspect-table":
        _emit(summarize_profile_table(load_profile_table(args.table_path), device=args.device))
        return 0

    if args.command == "tabular-inspect":
        _emit(load_tabular_dataset_from_path(args.dataset_path))
        return 0

    if args.command == "analyze-table":
        summary = analyze_profile_table_spectra(
            load_profile_table(args.table_path),
            num_modes=args.modes,
            device=args.device,
            normalize_each_profile=args.normalize,
        )
        if args.output_dir is not None:
            write_spectral_analysis_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command in {"compress-table", "compress-csv"}:
        table = load_profile_table(args.table_path)
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
            load_profile_table(args.table_path),
            mode_counts=args.mode_counts,
            device=args.device,
            normalize_each_profile=args.normalize,
        )
        if args.output_dir is not None:
            write_compression_sweep_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command in {"fit-table", "fit-csv"}:
        table = load_profile_table(args.table_path)
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

    if args.command == "compare-tables":
        summary = compare_profile_tables(
            load_profile_table(args.reference_table_path),
            load_profile_table(args.candidate_table_path),
            device=args.device,
        )
        if args.output_dir is not None:
            write_profile_comparison_artifacts(args.output_dir, summary)
        _emit(summary)
        return 0

    if args.command == "db-inspect":
        _emit(inspect_database(args.database))
        return 0

    if args.command == "db-bootstrap":
        _emit(bootstrap_local_database(args.database))
        return 0

    if args.command == "db-list-tables":
        inspection = inspect_database(args.database)
        _emit({"redacted_url": inspection.capability.redacted_url, "tables": inspection.tables})
        return 0

    if args.command == "db-describe-table":
        _emit(describe_database_table(args.database, args.table_name))
        return 0

    if args.command == "db-query":
        parameters = _parse_key_value_items(args.param)
        result = materialize_database_query(args.database, args.query, parameters=parameters)
        if args.output_dir is not None:
            write_tabular_artifacts(
                args.output_dir,
                result.dataset,
                summary_name="db_query_summary.json",
                table_name="query_result.csv",
                metadata={"workflow": "db-query"},
            )
        _emit(summarize_database_query_result(args.database, args.query, result, parameters=parameters))
        return 0

    if args.command == "db-write-table":
        summary = write_tabular_dataset_to_database(
            args.database,
            args.table_name,
            load_tabular_dataset(args.dataset_path),
            if_exists=args.if_exists,
        )
        _emit(summary)
        return 0

    if args.command == "db-materialize-query":
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
            time_column=args.time_column,
            num_modes=args.modes,
            device=args.device,
            normalize_each_profile=args.normalize,
        )
        if args.output_dir is not None:
            write_spectral_analysis_artifacts(args.output_dir, summary)
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

    if args.command in {"tf-train-table", "tf-train-csv"}:
        table = load_profile_table(args.table_path)
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
        table = load_profile_table(args.table_path)
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
        table = load_profile_table(args.table_path)
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
        table = load_profile_table(args.table_path)
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

    if args.command == "sql-ml-train-table":
        parameters = _parse_key_value_items(args.param)
        summary = train_modal_surrogate_from_database_query(
            args.database,
            args.query,
            backend=args.backend,
            parameters=parameters,
            time_column=args.time_column,
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

    if args.command == "sql-ml-evaluate-table":
        parameters = _parse_key_value_items(args.param)
        summary = evaluate_modal_surrogate_from_database_query(
            args.database,
            args.query,
            backend=args.backend,
            parameters=parameters,
            time_column=args.time_column,
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

        mcp_main()
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
