from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectral_packet_engine import (
    DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES,
    DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES,
    DEFAULT_PROFILE_REPORT_OUTPUT_DIR,
    inspect_artifact_directory,
    load_profile_table_report,
    to_serializable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Python-first inspect-analyze-compress hero workflow for a profile table."
    )
    parser.add_argument(
        "--table-path",
        "--csv-path",
        dest="table_path",
        type=Path,
        default=Path("examples/data/synthetic_profiles.csv"),
        help="Profile table to load: CSV, TSV, JSON, or optionally XLSX.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_PROFILE_REPORT_OUTPUT_DIR),
        help="Directory for written artifacts.",
    )
    parser.add_argument(
        "--analyze-modes",
        type=int,
        default=DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES,
        help="Mode count for spectral analysis.",
    )
    parser.add_argument(
        "--compress-modes",
        type=int,
        default=DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES,
        help="Mode count for modal compression.",
    )
    parser.add_argument("--normalize", action="store_true", help="Normalize each profile before analysis and compression.")
    parser.add_argument("--device", default="cpu", help="Torch device: auto, cpu, cuda, or mps.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = load_profile_table_report(
        args.table_path,
        analyze_num_modes=args.analyze_modes,
        compress_num_modes=args.compress_modes,
        device=args.device,
        normalize_each_profile=args.normalize,
    )
    report.write_artifacts(
        args.output_dir,
        metadata={"input": {"table_path": str(args.table_path)}},
    )
    print(json.dumps(to_serializable(report.overview), sort_keys=True))
    print(json.dumps(to_serializable(inspect_artifact_directory(args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
