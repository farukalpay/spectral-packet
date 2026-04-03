from __future__ import annotations

import argparse
from pathlib import Path

from spectral_packet_engine import (
    ModalSurrogateConfig,
    evaluate_modal_surrogate_on_profile_table,
    inspect_ml_backends,
    load_profile_table,
    write_modal_evaluation_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the backend-aware modal surrogate on a profile table.",
    )
    parser.add_argument(
        "--table-path",
        type=Path,
        default=Path("examples/data/synthetic_profiles.csv"),
        help="Profile table to load: CSV, TSV, JSON, or optionally XLSX.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "torch", "jax", "tensorflow"],
        default="auto",
        help="Modal surrogate backend to use.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Preferred execution device for backend inspection and torch-backed runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/ml_backend_workflow"),
        help="Directory for written artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend_report = inspect_ml_backends(args.device)
    selected_backend = backend_report.preferred_backend if args.backend == "auto" else args.backend
    if selected_backend is None:
        raise RuntimeError("No modal surrogate backend is available in this environment.")

    table = load_profile_table(args.table_path)
    summary = evaluate_modal_surrogate_on_profile_table(
        table,
        backend=selected_backend,
        num_modes=8,
        config=ModalSurrogateConfig(
            profile_hidden_units=(128, 64),
            time_hidden_units=(8,),
            residual_blocks=1,
            epochs=12,
            batch_size=8,
            validation_fraction=0.25,
            device=args.device,
        ),
    )
    write_modal_evaluation_artifacts(args.output_dir, summary)
    print(f"Backend: {summary.backend}")
    print(f"Wrote artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
