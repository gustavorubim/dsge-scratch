from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from sgelabs.plotting.labels import format_shock_display, format_state_display


def plot_irfs(
    irfs: NDArray[np.floating],
    state_names: Sequence[str],
    shock_names: Sequence[str],
    figsize: tuple[float, float] = (12.0, 8.0),
) -> plt.Figure:
    horizon = irfs.shape[0] - 1
    n = len(state_names)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    time = np.arange(horizon + 1)
    state_labels = [format_state_display(name) for name in state_names]
    shock_labels = [format_shock_display(shock) for shock in shock_names]

    for idx, label in enumerate(state_labels):
        ax = axes[idx // cols, idx % cols]
        for j, shock in enumerate(shock_names):
            ax.plot(time, irfs[:, idx, j], label=shock_labels[j])
        ax.set_title(label)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")
    axes[0, 0].legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    return fig


def plot_fevd(
    fevd: NDArray[np.floating],
    state_names: Sequence[str],
    shock_names: Sequence[str],
    figsize: tuple[float, float] = (12.0, 8.0),
) -> plt.Figure:
    n = len(state_names)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    x = np.arange(len(shock_names))
    state_labels = [format_state_display(name) for name in state_names]
    shock_labels = [format_shock_display(shock) for shock in shock_names]
    for idx, label in enumerate(state_labels):
        ax = axes[idx // cols, idx % cols]
        ax.bar(x, fevd[idx], color=plt.cm.tab20c(range(len(shock_names))))
        ax.set_xticks(x, shock_labels, rotation=45, ha="right", fontsize="small")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(label)
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")
    fig.tight_layout()
    return fig


def plot_fevd_timeseries(
    fevd_ts: NDArray[np.floating],
    state_names: Sequence[str],
    shock_names: Sequence[str],
    figsize: tuple[float, float] = (12.0, 8.0),
) -> plt.Figure:
    """Grid plot of FEVD shares as a function of horizon.

    fevd_ts: (H, n, k) cumulative shares up to each horizon.
    """
    H, n, k = fevd_ts.shape
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    time = np.arange(1, H + 1)
    state_labels = [format_state_display(name) for name in state_names]
    shock_labels = [format_shock_display(shock) for shock in shock_names]
    for idx, label in enumerate(state_labels):
        ax = axes[idx // cols, idx % cols]
        for j, shock in enumerate(shock_names):
            ax.plot(time, fevd_ts[:, idx, j], label=shock_labels[j])
        ax.set_title(label)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(1, H)
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")
    axes[0, 0].legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    return fig


def plot_historical(
    contributions: NDArray[np.floating],
    state_names: Sequence[str],
    shock_names: Sequence[str],
    time_index: Iterable[float] | None = None,
    figsize: tuple[float, float] = (12.0, 8.0),
) -> plt.Figure:
    shocks, T, n = contributions.shape
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    time = np.arange(T) if time_index is None else np.asarray(list(time_index))
    state_labels = [format_state_display(name) for name in state_names]
    shock_labels = [format_shock_display(shock) for shock in shock_names]
    for idx, label in enumerate(state_labels):
        ax = axes[idx // cols, idx % cols]
        series = contributions[:, :, idx]
        ax.stackplot(time, series, labels=shock_labels)
        ax.set_title(label)
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")
    axes[0, 0].legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    return fig


__all__ = ["plot_irfs", "plot_fevd", "plot_fevd_timeseries", "plot_historical"]
