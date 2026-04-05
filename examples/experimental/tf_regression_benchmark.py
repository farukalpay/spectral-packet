from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from spectral_packet_engine import (
    DensityPreprocessingConfig,
    InfiniteWell1D,
    InfiniteWellBasis,
    PreparedShotDensityProfiles,
    TensorFlowModalRegressionResult,
    TensorFlowModalRegressor,
    TensorFlowRegressorConfig,
    download_and_prepare_quantum_gas_transport_shots,
    inspect_torch_runtime,
    tensorflow_is_available,
)
from spectral_packet_engine.plotting import plot_metric_curve, plot_profile_comparison, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a time-conditioned TensorFlow regressor on published quantum-gas transport shots."
    )
    parser.add_argument("--scan-id", default="scan11879_56", help="Published transport scan identifier.")
    parser.add_argument("--modes", type=int, default=32, help="Number of spectral coefficients predicted by the network.")
    parser.add_argument(
        "--benchmark-modes",
        type=int,
        nargs="*",
        default=(),
        help="Additional mode counts evaluated in the same run.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--profile-widths", type=int, nargs="*", default=[512, 256], help="Hidden widths for the profile encoder.")
    parser.add_argument("--time-widths", type=int, nargs="*", default=[64, 32], help="Hidden widths for the time encoder.")
    parser.add_argument("--residual-blocks", type=int, default=4, help="Residual blocks in the shared trunk.")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate in the shared trunk.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--disable-xla", action="store_true", help="Disable XLA compilation.")
    parser.add_argument("--disable-mixed-precision", action="store_true", help="Disable mixed precision when a TensorFlow GPU is visible.")
    parser.add_argument("--torch-device", default="auto", help="Torch device for spectral target generation: auto, cpu, cuda, or mps.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional cache directory for downloaded datasets.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory for benchmark artifacts.")
    parser.add_argument("--plot", type=Path, default=None, help="Optional output path for validation reconstructions.")
    parser.add_argument("--history-plot", type=Path, default=None, help="Optional output path for a training-history plot.")
    parser.add_argument("--export-dir", type=Path, default=None, help="Optional SavedModel export directory.")
    parser.add_argument("--tensorboard-logdir", type=Path, default=None, help="Optional TensorBoard log directory.")
    parser.add_argument("--tensorboard-histograms", type=int, default=0, help="TensorBoard histogram frequency in epochs.")
    parser.add_argument("--profile-batch-start", type=int, default=None, help="First mini-batch to profile with TensorBoard.")
    parser.add_argument("--profile-batch-stop", type=int, default=None, help="Last mini-batch to profile with TensorBoard.")
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"object of type {type(value)!r} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    return path


def write_history_csv(path: Path, history: dict[str, list[float]]) -> Path:
    metric_names = list(history)
    row_count = max((len(values) for values in history.values()), default=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *metric_names])
        for row_index in range(row_count):
            row = [row_index + 1]
            for metric_name in metric_names:
                values = history[metric_name]
                row.append(values[row_index] if row_index < len(values) else "")
            writer.writerow(row)
    return path


def summarize_mode_result(mode_count: int, result: TensorFlowModalRegressionResult) -> dict[str, Any]:
    return {
        "mode_count": mode_count,
        "epochs_ran": result.epochs_ran,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "parameter_count": result.parameter_count,
        "training_seconds": result.training_seconds,
        "training_profiles_per_second": result.training_profiles_per_second,
        "validation_prediction_seconds": result.validation_prediction_seconds,
        "validation_inference_profiles_per_second": result.validation_inference_profiles_per_second,
        "validation_coefficient_mse": result.validation_coefficient_mse,
        "validation_moment_mae": result.validation_moment_mae,
        "validation_profile_relative_l2": result.validation_profile_relative_l2,
        "runtime": result.runtime.to_dict(),
    }


def plot_training_history(history: dict[str, list[float]], output_path: Path) -> None:
    keys = [
        ("loss", "val_loss", "total loss"),
        ("coefficients_loss", "val_coefficients_loss", "coefficient loss"),
    ]
    if "moments_loss" in history:
        keys.append(("moments_loss", "val_moments_loss", "moment loss"))

    figure, axes = plt.subplots(len(keys), 1, figsize=(9, 3.2 * len(keys)), constrained_layout=True)
    if len(keys) == 1:
        axes = [axes]

    for axis, (train_key, validation_key, title) in zip(axes, keys):
        axis.plot(history[train_key], label="train", linewidth=2.0)
        if validation_key in history:
            axis.plot(history[validation_key], label="validation", linewidth=2.0)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.grid(alpha=0.25)
        axis.legend()

    save_figure(figure, output_path)


def plot_mode_sweep(mode_summaries: list[dict[str, Any]], output_path: Path) -> None:
    mode_counts = np.asarray([entry["mode_count"] for entry in mode_summaries], dtype=float)
    relative_l2 = np.asarray([entry["validation_profile_relative_l2"] for entry in mode_summaries], dtype=float)
    coefficient_mse = np.asarray([entry["validation_coefficient_mse"] for entry in mode_summaries], dtype=float)
    throughput = np.asarray([entry["validation_inference_profiles_per_second"] for entry in mode_summaries], dtype=float)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    plot_metric_curve(
        axes[0],
        mode_counts,
        relative_l2,
        title="Validation profile error",
        xlabel="mode count",
        ylabel="relative L2",
    )
    plot_metric_curve(
        axes[1],
        mode_counts,
        coefficient_mse,
        title="Validation coefficient error",
        xlabel="mode count",
        ylabel="MSE",
    )
    plot_metric_curve(
        axes[2],
        mode_counts,
        throughput,
        title="Validation inference throughput",
        xlabel="mode count",
        ylabel="profiles per second",
    )
    save_figure(figure, output_path)


def plot_validation_reconstructions(
    prepared: PreparedShotDensityProfiles,
    regressor: TensorFlowModalRegressor,
    grid: torch.Tensor,
    output_path: Path,
) -> None:
    validation_indices = regressor.validation_indices
    if validation_indices is None or validation_indices.size == 0:
        raise RuntimeError("validation indices are unavailable; call fit before plotting")

    sample_indices = validation_indices[np.linspace(0, validation_indices.size - 1, num=min(4, validation_indices.size), dtype=int)]
    sample_profiles = prepared.density_profiles[sample_indices]
    sample_times = prepared.sample_times_s[sample_indices]
    predicted_profiles = regressor.reconstruct_profiles(sample_profiles, grid, sample_times=sample_times)

    figure, axes = plt.subplots(
        sample_indices.shape[0],
        1,
        figsize=(9, 3.0 * sample_indices.shape[0]),
        constrained_layout=True,
        sharex=True,
    )
    if sample_indices.shape[0] == 1:
        axes = [axes]

    for axis, row_index, profile, prediction, sample_time in zip(
        axes,
        sample_indices.tolist(),
        sample_profiles,
        predicted_profiles,
        sample_times.tolist(),
    ):
        plot_profile_comparison(
            axis,
            grid,
            profile,
            prediction,
            reference_label=f"experimental shot {row_index}",
            approximation_label="time-conditioned surrogate",
            title=f"time = {sample_time:.3f} s",
            ylabel="normalized density",
        )

    save_figure(figure, output_path)


def main(args: argparse.Namespace) -> int:
    if not tensorflow_is_available():
        raise SystemExit(
            "TensorFlow is not installed in this environment. Use a supported TensorFlow wheel environment and install the 'ml' extra."
        )

    torch_runtime = inspect_torch_runtime(args.torch_device)
    prepared = download_and_prepare_quantum_gas_transport_shots(
        scan_id=args.scan_id,
        cache_dir=args.cache_dir,
        preprocessing=DensityPreprocessingConfig(
            nan_fill_value=0.0,
            clip_negative=True,
            normalize_each_profile=True,
            drop_nonpositive_mass=True,
        ),
    )

    grid, sample_times, profiles = prepared.to_torch(
        dtype=torch_runtime.preferred_real_dtype,
        device=torch_runtime.device,
    )
    domain = InfiniteWell1D(left=grid[0], right=grid[-1])
    mode_counts = tuple(sorted({args.modes, *args.benchmark_modes}))
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    primary_regressor: TensorFlowModalRegressor | None = None
    primary_result: TensorFlowModalRegressionResult | None = None
    mode_summaries: list[dict[str, Any]] = []

    for mode_count in mode_counts:
        tensorboard_logdir = None
        if args.tensorboard_logdir is not None:
            tensorboard_logdir = args.tensorboard_logdir / f"mode_{mode_count}"

        regressor = TensorFlowModalRegressor(
            InfiniteWellBasis(domain, mode_count),
            config=TensorFlowRegressorConfig(
                profile_hidden_units=tuple(args.profile_widths),
                time_hidden_units=tuple(args.time_widths),
                residual_blocks=args.residual_blocks,
                dropout_rate=args.dropout,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                batch_size=args.batch_size,
                enable_xla=not args.disable_xla,
                enable_mixed_precision=not args.disable_mixed_precision,
                tensorboard_log_dir=None if tensorboard_logdir is None else str(tensorboard_logdir),
                tensorboard_histogram_frequency=args.tensorboard_histograms,
                tensorboard_profile_batch_start=args.profile_batch_start,
                tensorboard_profile_batch_stop=args.profile_batch_stop,
            ),
        )
        result = regressor.fit(profiles, grid, sample_times=sample_times)
        mode_summaries.append(summarize_mode_result(mode_count, result))

        print(f"[modes={mode_count}]")
        print(f"  epochs ran: {result.epochs_ran}")
        print(f"  best epoch: {result.best_epoch}")
        print(f"  best validation loss: {result.best_validation_loss:.6e}")
        print(f"  parameters: {result.parameter_count}")
        print(f"  training throughput: {result.training_profiles_per_second:.2f} profiles/s")
        print(f"  validation inference throughput: {result.validation_inference_profiles_per_second:.2f} profiles/s")
        print(f"  validation coefficient MSE: {result.validation_coefficient_mse:.6e}")
        print(f"  validation moment MAE: {result.validation_moment_mae:.6e}")
        print(f"  validation profile rel-L2: {result.validation_profile_relative_l2:.6e}")

        if args.output_dir is not None:
            mode_dir = args.output_dir / f"mode_{mode_count}"
            write_json(mode_dir / "report.json", result.to_dict())
            write_history_csv(mode_dir / "history.csv", result.history)

        if mode_count == args.modes:
            primary_regressor = regressor
            primary_result = result

    if primary_regressor is None or primary_result is None:
        raise RuntimeError("primary mode was not evaluated")

    runtime = primary_result.runtime
    print()
    print("TensorFlow modal regressor")
    print(f"  host system: {runtime.host.system} ({runtime.host.machine})")
    print(f"  recommended accelerator: {runtime.host.recommended_accelerator}")
    print(f"  recommended runtime: {runtime.host.recommended_runtime}")
    print(f"  TensorFlow version: {runtime.version}")
    print(f"  visible GPUs: {len(runtime.gpu_devices)}")
    for device in runtime.gpu_devices:
        print(f"    {device}")
    print(f"  visible device types: {', '.join(runtime.visible_device_types) if runtime.visible_device_types else 'none'}")
    print(f"  mixed precision policy: {runtime.mixed_precision_policy}")
    print(f"  XLA enabled: {runtime.xla_enabled}")
    print(f"  torch target backend: {torch_runtime.backend}")
    print(f"  torch target device: {torch_runtime.device}")
    print(f"  samples: {profiles.shape[0]}")
    print(f"  train samples: {primary_result.train_size}")
    print(f"  validation samples: {primary_result.validation_size}")

    artifact_paths: dict[str, str] = {}

    if args.output_dir is not None:
        primary_dir = args.output_dir / f"mode_{args.modes}"
        summary_path = write_json(
            args.output_dir / "benchmark_summary.json",
            {
                "dataset": {
                    "scan_id": prepared.scan_id,
                    "scan_name": prepared.scan_name,
                    "doi_url": prepared.doi_url,
                    "title": prepared.title,
                    "num_samples": int(profiles.shape[0]),
                    "num_positions": int(grid.shape[0]),
                    "num_times": int(prepared.time_axis_s.shape[0]),
                },
                "torch_runtime": {
                    "system": torch_runtime.system,
                    "machine": torch_runtime.machine,
                    "torch_version": torch_runtime.torch_version,
                    "backend": torch_runtime.backend,
                    "device": str(torch_runtime.device),
                    "accelerator": torch_runtime.accelerator,
                    "preferred_real_dtype": str(torch_runtime.preferred_real_dtype),
                    "supports_float64": torch_runtime.supports_float64,
                },
                "primary_mode": args.modes,
                "mode_summaries": mode_summaries,
                "command_config": vars(args),
            },
        )
        artifact_paths["benchmark_summary"] = str(summary_path)
        artifact_paths["history_csv"] = str(primary_dir / "history.csv")

        if len(mode_summaries) > 1:
            mode_sweep_path = args.output_dir / "mode_sweep.png"
            plot_mode_sweep(mode_summaries, mode_sweep_path)
            artifact_paths["mode_sweep_plot"] = str(mode_sweep_path)

        if args.plot is None:
            args.plot = primary_dir / "validation_reconstruction.png"
        if args.history_plot is None:
            args.history_plot = primary_dir / "history.png"

    if args.plot is not None:
        plot_validation_reconstructions(prepared, primary_regressor, grid, args.plot)
        artifact_paths["validation_reconstruction_plot"] = str(args.plot)

    if args.history_plot is not None:
        plot_training_history(primary_result.history, args.history_plot)
        artifact_paths["history_plot"] = str(args.history_plot)

    if args.export_dir is not None:
        export_path = primary_regressor.export(args.export_dir)
        artifact_paths["saved_model"] = str(export_path)

    if artifact_paths:
        print()
        print("Artifacts")
        for name, path in artifact_paths.items():
            print(f"  {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
