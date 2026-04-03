from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from _reference_case import build_reference_problem, reference_times

from spectral_packet_engine import inspect_torch_runtime, simulate
from spectral_packet_engine.plotting import (
    TrajectorySummary,
    plot_density,
    plot_mode_weights,
    plot_trajectory_summary,
    plot_truncation_tail,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report figures for the reference problem.")
    parser.add_argument("--modes", type=int, default=200, help="Number of retained sine modes.")
    parser.add_argument("--quadrature", type=int, default=8192, help="Quadrature points used for projection.")
    parser.add_argument("--grid", type=int, default=4000, help="Spatial samples used for reconstruction.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or mps.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory used for generated figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = inspect_torch_runtime(args.device)

    problem = build_reference_problem(
        num_modes=args.modes,
        quadrature_points=args.quadrature,
        dtype=runtime.preferred_real_dtype,
        device=runtime.device,
    )
    grid = problem.domain.grid(args.grid)
    times = reference_times(dtype=problem.domain.real_dtype, device=problem.domain.device)

    initial_state = problem.projector.project_packet(problem.packet)
    record = simulate(
        problem.packet,
        times,
        projector=problem.projector,
        propagator=problem.propagator,
        grid=grid,
    )
    weights = torch.abs(initial_state.coefficients) ** 2
    truncation_tail = 1.0 - torch.cumsum(weights, dim=0)

    density_figure, density_axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True, sharex=True)
    plot_density(density_axes[0], grid, record.densities[0], title="Initial density", label="t = 0")
    plot_density(
        density_axes[1],
        grid,
        record.densities[-1],
        title=f"Density at t = {times[-1].item():g}",
        label=f"t = {times[-1].item():g}",
    )
    density_path = save_figure(density_figure, args.output_dir / "forward_density_evolution.png")

    mode_figure, mode_axis = plt.subplots(1, 1, figsize=(9, 4.5), constrained_layout=True)
    plot_mode_weights(
        mode_axis,
        problem.basis.mode_numbers,
        weights,
        title="Mode weights",
        label=r"$|c_n|^2$",
    )
    mode_path = save_figure(mode_figure, args.output_dir / "forward_mode_weights.png")

    tail_figure, tail_axis = plt.subplots(1, 1, figsize=(9, 4.5), constrained_layout=True)
    plot_truncation_tail(
        tail_axis,
        torch.arange(1, args.modes + 1, dtype=problem.domain.real_dtype),
        truncation_tail,
        title="Spectral truncation tail",
        label="tail",
    )
    tail_path = save_figure(tail_figure, args.output_dir / "forward_truncation_tail.png")

    summary_figure, summary_axis = plt.subplots(1, 1, figsize=(9, 4.8), constrained_layout=True)
    summary = TrajectorySummary(
        times=record.times,
        expectation_position=record.expectation_position(),
        left_probability=record.interval_probability(0.0, 0.5),
        right_probability=record.interval_probability(0.5, 1.0),
        total_probability=record.total_probability(),
    )
    plot_trajectory_summary(summary_axis, summary, title="Trajectory summary")
    summary_path = save_figure(summary_figure, args.output_dir / "forward_trajectory_summary.png")

    print("Wrote figures")
    print(f"  torch backend: {runtime.backend}")
    print(f"  torch device: {runtime.device}")
    print(f"  {density_path}")
    print(f"  {mode_path}")
    print(f"  {tail_path}")
    print(f"  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
