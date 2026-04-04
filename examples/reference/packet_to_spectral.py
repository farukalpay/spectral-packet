from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from reference_case import build_reference_problem

from spectral_packet_engine import inspect_torch_runtime, l2_norm
from spectral_packet_engine.plotting import plot_mode_weights, plot_truncation_tail, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project the reference packet onto the spectral basis.")
    parser.add_argument("--modes", type=int, default=200, help="Number of retained sine modes.")
    parser.add_argument("--quadrature", type=int, default=8192, help="Quadrature points used for projection.")
    parser.add_argument("--grid", type=int, default=4000, help="Spatial samples used for reconstruction.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or mps.")
    parser.add_argument("--plot", type=Path, default=None, help="Optional output path for a summary figure.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = inspect_torch_runtime(args.device)
    problem = build_reference_problem(
        num_modes=args.modes,
        quadrature_points=args.quadrature,
        dtype=runtime.preferred_real_dtype,
        device=runtime.device,
    )
    grid = problem.domain.grid(args.grid)
    initial_state = problem.projector.project_packet(problem.packet)
    weights = torch.abs(initial_state.coefficients) ** 2

    cutoffs = sorted({n for n in (20, 40, 80, 120, args.modes) if 0 < n <= args.modes})
    print("Spectral convergence")
    print(f"  torch backend = {runtime.backend}")
    print(f"  torch device = {runtime.device}")
    print("  N        sum|c_n|^2     tail")
    for cutoff in cutoffs:
        partial = float(torch.sum(weights[:cutoff]))
        print(f"  {cutoff:3d}      {partial:0.12f}   {1.0 - partial:0.3e}")

    dominant = torch.argsort(weights, descending=True)[:10] + 1
    print()
    print("Top 10 modes")
    for mode in dominant:
        weight = float(weights[mode - 1])
        print(f"  n = {int(mode):3d}   |c_n|^2 = {weight:0.6f}")

    reconstructed = problem.projector.reconstruct(initial_state, grid)
    reference = problem.packet.wavefunction(grid)
    reconstruction_error = torch.sqrt(l2_norm(reconstructed - reference, grid))
    print()
    print(f"Truncated reconstruction L2 error: {float(reconstruction_error):0.6e}")

    if args.plot is not None:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        plot_mode_weights(
            axes[0],
            problem.basis.mode_numbers,
            weights,
            title="Spectral weights",
            label=r"$|c_n|^2$",
        )
        plot_truncation_tail(
            axes[1],
            torch.arange(1, args.modes + 1, dtype=problem.domain.real_dtype),
            1.0 - torch.cumsum(weights, dim=0),
            title="Truncation tail",
            label="tail",
        )
        save_figure(figure, args.plot)
        print()
        print(f"Wrote figure to {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
