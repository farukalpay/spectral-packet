from __future__ import annotations

import argparse
from pathlib import Path

from spectral_packet_engine import (
    project_gaussian_packet,
    simulate_gaussian_packet,
    write_forward_artifacts,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a direct Python workflow and write artifacts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/python_workflow"),
        help="Directory for written artifacts.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device: auto, cpu, cuda, or mps.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    forward = simulate_gaussian_packet(
        center=0.30,
        width=0.07,
        wavenumber=25.0,
        times=[0.0, 1e-3, 5e-3],
        num_modes=128,
        grid_points=512,
        device=args.device,
    )
    projection = project_gaussian_packet(
        center=0.30,
        width=0.07,
        wavenumber=25.0,
        num_modes=128,
        grid_points=2048,
        device=args.device,
    )

    write_forward_artifacts(args.output_dir, forward)
    write_json(args.output_dir / "projection_summary.json", projection)

    print(f"Wrote artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
