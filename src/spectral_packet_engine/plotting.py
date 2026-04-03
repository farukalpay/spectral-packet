from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


ArrayLike = Sequence[float] | Sequence[int] | np.ndarray


@dataclass(frozen=True, slots=True)
class TrajectorySummary:
    """Time series used for compact forward-propagation reports."""

    times: ArrayLike
    expectation_position: ArrayLike
    left_probability: ArrayLike
    right_probability: ArrayLike
    total_probability: ArrayLike


def _to_numpy(values: ArrayLike) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    return array


def _coerce_1d(name: str, values: ArrayLike) -> np.ndarray:
    array = np.asarray(_to_numpy(values))
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(float, copy=False)


def _set_common_axes(ax: Axes, *, title: str | None, xlabel: str, ylabel: str) -> Axes:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.8)
    return ax


def plot_density(
    ax: Axes,
    x: ArrayLike,
    density: ArrayLike,
    *,
    label: str | None = None,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = r"$|\psi(x)|^2$",
    color=None,
    fill: bool = True,
    fill_alpha: float = 0.18,
    linewidth: float = 2.0,
) -> Axes:
    """Plot a one-dimensional probability density."""

    x_values = _coerce_1d("x", x)
    density_values = _coerce_1d("density", density)
    if x_values.shape != density_values.shape:
        raise ValueError("x and density must have the same shape")

    (line,) = ax.plot(x_values, density_values, label=label, color=color, linewidth=linewidth)
    if fill:
        ax.fill_between(x_values, density_values, color=line.get_color(), alpha=fill_alpha)
    if label is not None:
        ax.legend(loc="best")
    return _set_common_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)


def plot_profile_comparison(
    ax: Axes,
    x: ArrayLike,
    reference: ArrayLike,
    approximation: ArrayLike,
    *,
    reference_label: str = "reference",
    approximation_label: str = "approximation",
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "profile",
    linewidth: float = 2.0,
) -> Axes:
    """Overlay a reference profile and its approximation on one axis."""

    x_values = _coerce_1d("x", x)
    reference_values = _coerce_1d("reference", reference)
    approximation_values = _coerce_1d("approximation", approximation)
    if not (x_values.shape == reference_values.shape == approximation_values.shape):
        raise ValueError("x, reference, and approximation must have the same shape")

    ax.plot(x_values, reference_values, label=reference_label, linewidth=linewidth)
    ax.plot(x_values, approximation_values, label=approximation_label, linewidth=linewidth, linestyle="--")
    ax.legend(loc="best")
    return _set_common_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)


def plot_mode_weights(
    ax: Axes,
    modes: ArrayLike,
    weights: ArrayLike,
    *,
    label: str | None = None,
    title: str | None = None,
    xlabel: str = "mode n",
    ylabel: str = r"$|c_n|^2$",
    color=None,
    log_scale: bool = True,
    marker: str = "o",
    linewidth: float = 1.8,
) -> Axes:
    """Plot the modal weight distribution of a spectral state."""

    mode_values = _coerce_1d("modes", modes)
    weight_values = _coerce_1d("weights", weights)
    if mode_values.shape != weight_values.shape:
        raise ValueError("modes and weights must have the same shape")

    plot_fn = ax.semilogy if log_scale else ax.plot
    (line,) = plot_fn(
        mode_values,
        np.maximum(weight_values, np.finfo(weight_values.dtype).tiny),
        label=label,
        color=color,
        marker=marker,
        linewidth=linewidth,
    )
    if label is not None:
        ax.legend(loc="best")
    return _set_common_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)


def plot_truncation_tail(
    ax: Axes,
    truncation_sizes: ArrayLike,
    tail_values: ArrayLike,
    *,
    label: str | None = None,
    title: str | None = None,
    xlabel: str = "N",
    ylabel: str = r"$1 - \sum_{n=1}^{N} |c_n|^2$",
    color=None,
    log_scale: bool = True,
    marker: str = "o",
    linewidth: float = 1.8,
) -> Axes:
    """Plot the convergence tail left after truncating a spectral expansion."""

    n_values = _coerce_1d("truncation_sizes", truncation_sizes)
    tail_array = _coerce_1d("tail_values", tail_values)
    if n_values.shape != tail_array.shape:
        raise ValueError("truncation_sizes and tail_values must have the same shape")

    plot_fn = ax.semilogy if log_scale else ax.plot
    plot_fn(
        n_values,
        np.maximum(tail_array, np.finfo(tail_array.dtype).tiny),
        label=label,
        color=color,
        marker=marker,
        linewidth=linewidth,
    )
    if label is not None:
        ax.legend(loc="best")
    return _set_common_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)


def plot_metric_curve(
    ax: Axes,
    x: ArrayLike,
    y: ArrayLike,
    *,
    label: str | None = None,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "metric",
    marker: str = "o",
    linewidth: float = 1.8,
    color=None,
) -> Axes:
    """Plot a generic one-dimensional metric curve."""

    x_values = _coerce_1d("x", x)
    y_values = _coerce_1d("y", y)
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same shape")

    ax.plot(x_values, y_values, label=label, marker=marker, linewidth=linewidth, color=color)
    if label is not None:
        ax.legend(loc="best")
    return _set_common_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)


def plot_trajectory_summary(
    ax: Axes,
    summary: TrajectorySummary,
    *,
    probability_ax: Axes | None = None,
    title: str | None = None,
    position_label: str = r"$\langle x \rangle_t$",
    probability_labels: tuple[str, str, str] = ("P_left", "P_right", "P_total"),
    position_color=None,
    probability_colors: tuple = (None, None, None),
    linewidth: float = 2.0,
) -> tuple[Axes, Axes]:
    """Plot a compact summary of the packet trajectory over time."""

    times = _coerce_1d("times", summary.times)
    expectation = _coerce_1d("expectation_position", summary.expectation_position)
    left = _coerce_1d("left_probability", summary.left_probability)
    right = _coerce_1d("right_probability", summary.right_probability)
    total = _coerce_1d("total_probability", summary.total_probability)

    if not (times.shape == expectation.shape == left.shape == right.shape == total.shape):
        raise ValueError("trajectory summary arrays must have the same shape")

    if probability_ax is None:
        probability_ax = ax.twinx()

    ax.plot(times, expectation, label=position_label, color=position_color, linewidth=linewidth)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\langle x \rangle_t$")

    probability_ax.plot(
        times,
        left,
        label=probability_labels[0],
        color=probability_colors[0],
        linewidth=linewidth,
        linestyle="-",
    )
    probability_ax.plot(
        times,
        right,
        label=probability_labels[1],
        color=probability_colors[1],
        linewidth=linewidth,
        linestyle="--",
    )
    probability_ax.plot(
        times,
        total,
        label=probability_labels[2],
        color=probability_colors[2],
        linewidth=linewidth,
        linestyle=":",
    )
    probability_ax.set_ylabel("probability")

    ax.grid(True, which="both", alpha=0.25, linewidth=0.8)
    probability_ax.grid(False)

    if title is not None:
        ax.set_title(title)

    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = probability_ax.get_legend_handles_labels()
    ax.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best")
    return ax, probability_ax


def plot_metric_curve(
    ax: Axes,
    x: ArrayLike,
    y: ArrayLike,
    *,
    label: str | None = None,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "value",
    color=None,
    marker: str = "o",
    linewidth: float = 2.0,
    log_scale: bool = False,
) -> Axes:
    """Plot a generic one-dimensional metric curve."""

    x_values = _coerce_1d("x", x)
    y_values = _coerce_1d("y", y)
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same shape")

    plot_fn = ax.semilogy if log_scale else ax.plot
    plot_fn(
        x_values,
        y_values,
        label=label,
        color=color,
        marker=marker,
        linewidth=linewidth,
    )
    if label is not None:
        ax.legend(loc="best")
    return _set_common_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 200,
    close: bool = True,
) -> Path:
    """Save a figure and optionally close it."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return output_path


__all__ = [
    "TrajectorySummary",
    "plot_density",
    "plot_metric_curve",
    "plot_mode_weights",
    "plot_trajectory_summary",
    "plot_truncation_tail",
    "save_figure",
]
