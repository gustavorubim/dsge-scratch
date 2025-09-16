from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


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
    for idx, name in enumerate(state_names):
        ax = axes[idx // cols, idx % cols]
        for j, shock in enumerate(shock_names):
            ax.plot(time, irfs[:, idx, j], label=shock)
        ax.set_title(name)
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
    for idx, name in enumerate(state_names):
        ax = axes[idx // cols, idx % cols]
        ax.bar(x, fevd[idx], color=plt.cm.tab20c(range(len(shock_names))))
        ax.set_xticks(x, shock_names, rotation=45, ha="right", fontsize="small")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(name)
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")
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
    for idx, name in enumerate(state_names):
        ax = axes[idx // cols, idx % cols]
        series = contributions[:, :, idx]
        ax.stackplot(time, series, labels=shock_names)
        ax.set_title(name)
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")
    axes[0, 0].legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    return fig


__all__ = ["plot_irfs", "plot_fevd", "plot_historical"]
