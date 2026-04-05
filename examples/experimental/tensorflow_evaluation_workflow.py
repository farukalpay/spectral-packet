from __future__ import annotations

import argparse
from pathlib import Path

from spectral_packet_engine import (
    TensorFlowRegressorConfig,
    evaluate_tensorflow_surrogate_on_profile_table,
    load_profile_table,
    write_tensorflow_evaluation_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the TensorFlow surrogate on a profile table.",
    )
    parser.add_argument(
        "--table-path",
        type=Path,
        default=Path("examples/data/synthetic_profiles.csv"),
        help="Profile table to load: CSV, TSV, JSON, or optionally XLSX.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tensorflow_evaluation"),
        help="Directory for written artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table = load_profile_table(args.table_path)
    summary = evaluate_tensorflow_surrogate_on_profile_table(
        table,
        num_modes=8,
        config=TensorFlowRegressorConfig(
            profile_hidden_units=(128, 64),
            time_hidden_units=(8,),
            residual_blocks=1,
            dropout_rate=0.0,
            epochs=12,
            batch_size=8,
            validation_fraction=0.25,
        ),
    )
    write_tensorflow_evaluation_artifacts(args.output_dir, summary)
    print(f"Wrote artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
