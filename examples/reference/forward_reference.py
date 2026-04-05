from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from reference_case import build_reference_problem, reference_times

from spectral_packet_engine import inspect_torch_runtime, simulate, total_probability
from spectral_packet_engine.plotting import plot_density, plot_mode_weights, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the forward spectral reference problem.")
    parser.add_argument("--modes", type=int, default=200, help="Number of retained sine modes.")
    parser.add_argument("--quadrature", type=int, default=8192, help="Quadrature points used for the packet projection.")
    parser.add_argument("--grid", type=int, default=4000, help="Number of spatial samples for reconstruction.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or mps.")
    parser.add_argument(
        "--times",
        type=float,
        nargs="*",
        default=list(reference_times().tolist()),
        help="Times at which the solution is reported.",
    )
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
    times = torch.tensor(args.times, dtype=problem.domain.real_dtype, device=problem.domain.device)

    initial_state = problem.projector.project_packet(problem.packet)
    initial_wavefunction = problem.packet.wavefunction(grid)
    record = simulate(
        problem.packet,
        times,
        projector=problem.projector,
        propagator=problem.propagator,
        grid=grid,
    )

    spatial_norm = float(total_probability(initial_wavefunction, grid))
    spectral_norm = float(initial_state.norm_squared)

    print("Reference parameters")
    print(f"  torch backend = {runtime.backend}")
    print(f"  torch device = {runtime.device}")
    print("  center = 0.300000")
    print("  width = 0.070000")
    print("  wavenumber = 25.000000")
    print(f"  spatial norm = {spatial_norm:.12f}")
    print(f"  spectral norm (N={args.modes}) = {spectral_norm:.12f}")
    print(f"  spectral tail = {1.0 - spectral_norm:.3e}")
    print()
    print("Time evolution")
    print("  t        <x>         P_left      P_right     P_total")

    expectation = record.expectation_position()
    left_probability = record.interval_probability(0.0, 0.5)
    right_probability = record.interval_probability(0.5, 1.0)
    total = record.total_probability()

    for index, time in enumerate(times):
        print(
            f"  {float(time):0.4f}   "
            f"{float(expectation[index]):0.6f}   "
            f"{float(left_probability[index]):0.6f}   "
            f"{float(right_probability[index]):0.6f}   "
            f"{float(total[index]):0.6f}"
        )

    if args.plot is not None:
        figure, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
        plot_density(
            axes[0],
            grid,
            record.densities[0],
            label="t = 0",
            title="Density evolution",
        )
        plot_density(
            axes[0],
            grid,
            record.densities[-1],
            label=f"t = {float(times[-1]):g}",
            fill=False,
        )
        plot_mode_weights(
            axes[1],
            problem.basis.mode_numbers,
            torch.abs(initial_state.coefficients) ** 2,
            title="Mode weights",
            label=r"$|c_n|^2$",
        )
        save_figure(figure, args.plot)
        print()
        print(f"Wrote figure to {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
