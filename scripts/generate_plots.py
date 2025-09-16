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
from sgelabs.plotting import plot_fevd, plot_irfs
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

    irf_fig = plot_irfs(irfs, solution.state_names, solution.shock_names)
    irf_path = output_dir / "irfs.png"
    irf_fig.savefig(irf_path, dpi=300, bbox_inches="tight")
    plt.close(irf_fig)

    fevd_fig = plot_fevd(fevd, solution.state_names, solution.shock_names)
    fevd_path = output_dir / "fevd.png"
    fevd_fig.savefig(fevd_path, dpi=300, bbox_inches="tight")
    plt.close(fevd_fig)

    np.savez(
        output_dir / "diagnostics.npz",
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

    print(f"Saved IRF plot to {irf_path}")
    print(f"Saved FEVD plot to {fevd_path}")
    print(f"Saved diagnostics npz to {output_dir / 'diagnostics.npz'}")


if __name__ == "__main__":
    main()
