from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spectral_packet_engine import (
    InfiniteWell1D,
    harmonic_potential,
    inspect_artifact_directory,
    run_spectroscopy_workflow,
    solve_eigenproblem,
    to_serializable,
    write_vertical_workflow_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the spectroscopy-style inverse-physics workflow and write one vertical artifact bundle."
    )
    parser.add_argument("--omega", type=float, default=8.0, help="Ground-truth harmonic frequency used to synthesize the target spectrum.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/inverse_physics"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device=args.device)
    target = solve_eigenproblem(
        lambda x: harmonic_potential(x, omega=args.omega, domain=domain),
        domain,
        num_points=128,
        num_states=3,
    ).eigenvalues
    summary = run_spectroscopy_workflow(
        target_eigenvalues=target,
        families=("harmonic", "double-well"),
        initial_guesses={
            "harmonic": {"omega": max(1.0, args.omega * 0.6)},
            "double-well": {"a_param": 1.5, "b_param": 1.0},
        },
        device=args.device,
    )
    write_vertical_workflow_artifacts(args.output_dir, summary)
    print(json.dumps(to_serializable(summary.family_inference), sort_keys=True))
    print(json.dumps(to_serializable(inspect_artifact_directory(args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
