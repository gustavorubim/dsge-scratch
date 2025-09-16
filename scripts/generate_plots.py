from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from sgelabs.analysis import compute_fevd, compute_irfs
from sgelabs.io import parse_mod_file
from sgelabs.ir import linearize
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
    return parser


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _save_irf_plots(
    irfs: np.ndarray,
    state_names: list[str],
    shock_names: list[str],
    out_dir: Path,
) -> None:
    time = np.arange(irfs.shape[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, state in enumerate(state_names):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for j, shock in enumerate(shock_names):
            ax.plot(time, irfs[:, idx, j], label=shock)
        ax.set_title(state)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Response")
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
        if shock_names:
            ax.legend(loc="upper right", fontsize="small")
        fig.tight_layout()
        fig.savefig(out_dir / f"{_safe_name(state)}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _save_fevd_plots(
    fevd: np.ndarray,
    state_names: list[str],
    shock_names: list[str],
    out_dir: Path,
) -> None:
    x = np.arange(len(shock_names))
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, state in enumerate(state_names):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.bar(x, fevd[idx])
        ax.set_title(state)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Share")
        if shock_names:
            ax.set_xticks(x, shock_names, rotation=45, ha="right", fontsize="small")
        ax.set_xlabel("Shock")
        fig.tight_layout()
        fig.savefig(out_dir / f"{_safe_name(state)}.png", dpi=300, bbox_inches="tight")
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

    model_name = _safe_name(args.mod.stem)
    model_dir = output_dir / model_name
    irf_dir = model_dir / "irfs"
    fevd_dir = model_dir / "fevds"

    _save_irf_plots(irfs, solution.state_names, solution.shock_names, irf_dir)
    _save_fevd_plots(fevd, solution.state_names, solution.shock_names, fevd_dir)

    np.savez(
        model_dir / "diagnostics.npz",
        G=solution.g,
        C=solution.c,
        impact=solution.impact,
        irfs=irfs,
        fevd=fevd,
        state_names=np.array(solution.state_names, dtype=object),
        shock_names=np.array(solution.shock_names, dtype=object),
        horizon=args.horizon,
        eu=np.array(solution.eu, dtype=int),
    )

    print(f"Saved IRFs to {irf_dir}")
    print(f"Saved FEVDs to {fevd_dir}")
    print(f"Saved diagnostics npz to {model_dir / 'diagnostics.npz'}")


if __name__ == "__main__":
    main()
