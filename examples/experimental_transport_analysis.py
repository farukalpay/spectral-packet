from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from spectral_packet_engine import (
    DensityPreprocessingConfig,
    InfiniteWell1D,
    InfiniteWellBasis,
    download_and_prepare_quantum_gas_transport_scan,
    inspect_torch_runtime,
    profile_mean,
    profile_variance,
    project_profiles_onto_basis,
    reconstruct_profiles_from_basis,
    relative_l2_error,
    summarize_profile_compression,
)
from spectral_packet_engine.plotting import (
    plot_metric_curve,
    plot_profile_comparison,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the sine-basis compression pipeline on real quantum transport data.")
    parser.add_argument("--scan-id", default="scan11879_56", help="Published transport scan identifier.")
    parser.add_argument("--mode-counts", type=int, nargs="*", default=[8, 16, 32, 64], help="Mode counts evaluated in the compression benchmark.")
    parser.add_argument("--reconstruction-modes", type=int, default=32, help="Mode count used for the snapshot reconstruction plot.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional cache directory for downloaded datasets.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or mps.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Optional output directory for generated figures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = inspect_torch_runtime(args.device)
    prepared = download_and_prepare_quantum_gas_transport_scan(
        scan_id=args.scan_id,
        cache_dir=args.cache_dir,
        preprocessing=DensityPreprocessingConfig(
            aggregate="mean",
            nan_fill_value=0.0,
            clip_negative=True,
            normalize_each_profile=True,
            drop_nonpositive_mass=True,
        ),
    )

    grid, times, profiles = prepared.to_torch(
        dtype=runtime.preferred_real_dtype,
        device=runtime.device,
    )
    domain = InfiniteWell1D(left=grid[0], right=grid[-1])
    compression = summarize_profile_compression(
        profiles,
        grid,
        mode_counts=torch.tensor(args.mode_counts, dtype=torch.int64),
        domain=domain,
    )

    mean_position = profile_mean(profiles, grid)
    width = torch.sqrt(profile_variance(profiles, grid))

    print("Experimental benchmark")
    print(f"  torch backend: {runtime.backend}")
    print(f"  torch device: {runtime.device}")
    print(f"  dataset: {prepared.title}")
    print(f"  doi: {prepared.doi_url}")
    print(f"  scan: {prepared.scan_name}")
    print(f"  positions: {prepared.position_axis_m.shape[0]}")
    print(f"  times: {prepared.time_axis_s.shape[0]}")
    print(f"  shots per time: {prepared.shots_per_time}")
    print(f"  temperature: {prepared.temperature_nK:.2f} +/- {prepared.temperature_std_nK:.2f} nK")
    print()
    print("Compression summary")
    print("  modes    mean rel-L2    max rel-L2")
    for mode, mean_error, max_error in zip(
        compression.mode_counts.tolist(),
        compression.mean_relative_l2_error.tolist(),
        compression.max_relative_l2_error.tolist(),
    ):
        print(f"  {int(mode):4d}    {mean_error:0.6f}      {max_error:0.6f}")
    print()
    print("Profile moments")
    print(f"  center-of-mass range: {float(torch.min(mean_position)):.6e} m -> {float(torch.max(mean_position)):.6e} m")
    print(f"  width range: {float(torch.min(width)):.6e} m -> {float(torch.max(width)):.6e} m")

    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        basis = InfiniteWellBasis(domain, args.reconstruction_modes)
        coefficients = project_profiles_onto_basis(profiles, grid, basis)
        reconstructions = reconstruct_profiles_from_basis(coefficients, grid, basis)
        reconstruction_error = relative_l2_error(profiles, reconstructions, grid)

        selected_time_indices = np.linspace(0, profiles.shape[0] - 1, num=4, dtype=int)
        comparison_figure, comparison_axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True, sharey=True)
        for axis, time_index in zip(comparison_axes.flatten(), selected_time_indices):
            plot_profile_comparison(
                axis,
                grid,
                profiles[time_index],
                reconstructions[time_index],
                reference_label=f"experiment, t = {float(times[time_index]):.3f} s",
                approximation_label=f"{args.reconstruction_modes}-mode reconstruction",
                title=f"Relative L2 = {float(reconstruction_error[time_index]):.4f}",
                ylabel="normalized density",
            )
        comparison_path = save_figure(comparison_figure, args.plot_dir / "experimental_profile_reconstruction.png")

        error_figure, error_axis = plt.subplots(1, 1, figsize=(8, 4.8), constrained_layout=True)
        plot_metric_curve(
            error_axis,
            compression.mode_counts,
            compression.mean_relative_l2_error,
            title="Mean profile reconstruction error",
            xlabel="mode count",
            ylabel="mean relative L2 error",
        )
        error_path = save_figure(error_figure, args.plot_dir / "experimental_compression_curve.png")

        moment_figure, moment_axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True, sharex=True)
        plot_metric_curve(
            moment_axes[0],
            times,
            mean_position,
            title="Experimental center of mass",
            xlabel="time (s)",
            ylabel="mean position (m)",
        )
        plot_metric_curve(
            moment_axes[1],
            times,
            width,
            title="Experimental width",
            xlabel="time (s)",
            ylabel="width (m)",
        )
        moment_path = save_figure(moment_figure, args.plot_dir / "experimental_moments.png")

        print()
        print("Wrote figures")
        print(f"  {comparison_path}")
        print(f"  {error_path}")
        print(f"  {moment_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
