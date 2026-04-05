from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectral_packet_engine import analyze_coupled_channel_surfaces, build_separable_2d_report, to_serializable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the artifact-backed separable 2D reduced-model report and one coupled-surface summary."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reduced_model_workflow"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_separable_2d_report(device=args.device)
    artifact_report = report.write_artifacts(args.output_dir)
    coupled = analyze_coupled_channel_surfaces(device=args.device)
    print(json.dumps(to_serializable(report.overview), sort_keys=True))
    print(json.dumps(artifact_report.to_dict(), sort_keys=True))
    print(json.dumps(to_serializable(coupled), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
