from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from sgelabs.analysis import compute_fevd, compute_irfs, historical_decomp
from sgelabs.io import parse_mod_file
from sgelabs.ir import LinearizedModel, ModelIR, linearize
from sgelabs.solve import solve_gensys


def _load_ir(path: Path) -> ModelIR:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelIR.from_json_dict(data)


def _save_ir(model: ModelIR, path: Path) -> None:
    path.write_text(json.dumps(model.to_json_dict(), indent=2), encoding="utf-8")


def _save_solution(
    linearized: LinearizedModel, solver_result, model: ModelIR, path: Path
) -> None:
    shock_names = solver_result.shock_names
    shock_std = np.array([model.shocks[name] for name in shock_names], dtype=float)
    np.savez(
        path,
        G=solver_result.g,
        C=solver_result.c,
        impact=solver_result.impact,
        state_names=np.array(solver_result.state_names, dtype=object),
        shock_names=np.array(shock_names, dtype=object),
        shock_std=shock_std,
        eu=np.array(solver_result.eu, dtype=int),
    )


def _load_solution(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def cmd_load(args: argparse.Namespace) -> None:
    model = parse_mod_file(args.mod)
    _save_ir(model, args.output)
    print(f"Wrote IR to {args.output}")


def cmd_solve(args: argparse.Namespace) -> None:
    if args.mod is None and args.ir is None:
        raise ValueError("Either --mod or --ir must be provided")
    if args.mod is not None:
        model = parse_mod_file(args.mod)
    else:
        model = _load_ir(Path(args.ir))
    linearized = linearize(model)
    result = solve_gensys(linearized)
    output = args.output or Path(args.mod or args.ir).with_suffix(".npz")
    _save_solution(linearized, result, model, output)
    eu = result.eu
    print(f"Solver eu={eu}, saved solution to {output}")


def cmd_irf(args: argparse.Namespace) -> None:
    solution = _load_solution(Path(args.solution))
    G = solution["G"]
    impact = solution["impact"]
    irfs = compute_irfs(G, impact, args.horizon)
    output = args.output or Path(args.solution).with_suffix(".irf.npz")
    np.savez(
        output,
        irfs=irfs,
        state_names=solution["state_names"],
        shock_names=solution["shock_names"],
    )
    print(f"Saved IRFs to {output}")


def cmd_fevd(args: argparse.Namespace) -> None:
    solution = _load_solution(Path(args.solution))
    G = solution["G"]
    impact = solution["impact"]
    shock_std = solution["shock_std"]
    sigma = shock_std ** 2
    fevd = compute_fevd(G, impact, sigma, args.horizon)
    output = args.output or Path(args.solution).with_suffix(".fevd.npy")
    np.save(output, fevd)
    print(f"Saved FEVD to {output}")


def cmd_hist(args: argparse.Namespace) -> None:
    solution = _load_solution(Path(args.solution))
    G = solution["G"]
    impact = solution["impact"]
    shocks = np.load(args.shocks)
    contributions = historical_decomp(G, impact, shocks)
    output = args.output or Path(args.shocks).with_suffix(".hist.npz")
    np.savez(
        output,
        contributions=contributions,
        state_names=solution["state_names"],
        shock_names=solution["shock_names"],
    )
    print(f"Saved historical decomposition to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sgelabs")
    sub = parser.add_subparsers(dest="command", required=True)

    load_p = sub.add_parser("load", help="Parse a Dynare .mod file")
    load_p.add_argument("--mod", type=Path, required=True, help="Path to .mod file")
    load_p.add_argument(
        "--output", type=Path, required=True, help="Output path for JSON IR"
    )
    load_p.set_defaults(func=cmd_load)

    solve_p = sub.add_parser("solve", help="Linearize and solve a model")
    solve_p.add_argument("--mod", type=Path, help="Path to .mod file")
    solve_p.add_argument("--ir", type=Path, help="Path to IR JSON")
    solve_p.add_argument("--output", type=Path, help="Path to save solution npz")
    solve_p.set_defaults(func=cmd_solve)

    irf_p = sub.add_parser("irf", help="Compute impulse responses")
    irf_p.add_argument("--solution", type=Path, required=True, help="Solution npz")
    irf_p.add_argument("--horizon", type=int, default=40)
    irf_p.add_argument("--output", type=Path, help="Output npz file")
    irf_p.set_defaults(func=cmd_irf)

    fevd_p = sub.add_parser("fevd", help="Compute FEVD")
    fevd_p.add_argument("--solution", type=Path, required=True)
    fevd_p.add_argument("--horizon", type=int, default=40)
    fevd_p.add_argument("--output", type=Path)
    fevd_p.set_defaults(func=cmd_fevd)

    hist_p = sub.add_parser("hist", help="Historical decomposition")
    hist_p.add_argument("--solution", type=Path, required=True)
    hist_p.add_argument("--shocks", type=Path, required=True, help="Shocks npy/npz")
    hist_p.add_argument("--output", type=Path)
    hist_p.set_defaults(func=cmd_hist)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
