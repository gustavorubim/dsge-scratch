from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from sgelabs.analysis import compute_fevd, compute_fevd_ts, compute_irfs
from sgelabs.io import parse_mod_file
from sgelabs.ir import linearize
from sgelabs.plotting.labels import format_shock_display, format_state_display
from sgelabs.solve import solve_gensys


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate IRF and FEVD plots from a .mod model")
    parser.add_argument("mod", type=Path, help="Path to Dynare-style .mod file")
    parser.add_argument(
        "output",
        type=Path,
        help="Directory where plots and diagnostics will be written",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=40,
        help="Number of periods for IRFs/FEVD (default: 40)",
    )
    parser.add_argument(
        "--include-lags",
        action="store_true",
        help="Plot companion lag states as well as contemporaneous variables",
    )
    return parser


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _filter_states(
    state_names: list[str],
    keep_all: bool,
) -> tuple[list[str], list[int]]:
    if keep_all:
        indices = list(range(len(state_names)))
        return state_names, indices
    filtered: list[str] = []
    indices: list[int] = []
    for idx, name in enumerate(state_names):
        if "_lag" in name:
            continue
        filtered.append(name)
        indices.append(idx)
    return filtered, indices


def _save_irf_plots(
    irfs: np.ndarray,
    state_names: list[str],
    shock_names: list[str],
    indices: list[int],
    out_dir: Path,
) -> None:
    time = np.arange(irfs.shape[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    shock_labels = [format_shock_display(shock) for shock in shock_names]
    for idx in indices:
        state = state_names[idx]
        state_label = format_state_display(state)
        state_dir = out_dir / _safe_name(state_label)
        state_dir.mkdir(parents=True, exist_ok=True)
        for j, shock in enumerate(shock_names):
            shock_label = shock_labels[j]
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.plot(time, irfs[:, idx, j], label=shock_label)
            ax.set_title(f"{state_label} - {shock_label}")
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Response")
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
            fig.tight_layout()
            fig.savefig(
                state_dir / f"{_safe_name(shock_label)}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


def _save_fevd_plots_timeseries(
    fevd_ts: np.ndarray,
    state_names: list[str],
    shock_names: list[str],
    indices: list[int],
    out_dir: Path,
) -> None:
    H = fevd_ts.shape[0]
    time = np.arange(1, H + 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    shock_labels = [format_shock_display(shock) for shock in shock_names]
    for idx in indices:
        state = state_names[idx]
        state_label = format_state_display(state)
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for j, shock in enumerate(shock_names):
            ax.plot(time, fevd_ts[:, idx, j], label=shock_labels[j])
        ax.set_title(state_label)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("FEVD share")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(1, H)
        if shock_names:
            ax.legend(loc="upper right", fontsize="small")
        fig.tight_layout()
        fig.savefig(out_dir / f"{_safe_name(state_label)}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = parse_mod_file(args.mod)
    linearized = linearize(model)
    solution = solve_gensys(linearized)

    irfs = compute_irfs(solution.g, solution.impact, args.horizon)
    shock_std = np.array(
        [model.shocks.get(name, 1.0) for name in solution.shock_names],
        dtype=float,
    )
    sigma = shock_std ** 2
    fevd = compute_fevd(solution.g, solution.impact, sigma, args.horizon)
    fevd_ts = compute_fevd_ts(solution.g, solution.impact, sigma, args.horizon)

    state_names = solution.state_names
    filtered_names, indices = _filter_states(state_names, keep_all=args.include_lags)

    model_name = _safe_name(args.mod.stem)
    model_dir = output_dir / model_name
    irf_dir = model_dir / "irfs"
    fevd_dir = model_dir / "fevds"

    _save_irf_plots(irfs, state_names, solution.shock_names, indices, irf_dir)
    _save_fevd_plots_timeseries(fevd_ts, state_names, solution.shock_names, indices, fevd_dir)

    np.savez(
        model_dir / "diagnostics.npz",
        G=solution.g,
        C=solution.c,
        impact=solution.impact,
        irfs=irfs,
        fevd=fevd,
        fevd_ts=fevd_ts,
        state_names=np.array(state_names, dtype=object),
        shock_names=np.array(solution.shock_names, dtype=object),
        horizon=args.horizon,
        eu=np.array(solution.eu, dtype=int),
    )

    print(f"Saved IRFs to {irf_dir}")
    print(f"Saved FEVD time series to {fevd_dir}")
    print(f"Saved diagnostics npz to {model_dir / 'diagnostics.npz'}")
    if not args.include_lags:
        print("(Lagged companion states omitted; use --include-lags to plot them.)")


if __name__ == "__main__":
    main()
