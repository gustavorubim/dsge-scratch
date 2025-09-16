from __future__ import annotations

import math
from typing import Dict, List, Tuple

import sympy as sp

from sgelabs.ir.model import LinearizedModel, ModelIR
from sgelabs.utils.timing import alias_for_shift, parse_alias


def _collect_symbol_shifts(model: ModelIR) -> Dict[str, set[int]]:
    endo_names = [var.name for var in model.endo]
    exo_names = [shock.name for shock in model.exo]
    all_names = set(endo_names + exo_names)
    shifts: Dict[str, set[int]] = {name: {0} for name in all_names}
    for equation in model.equations:
        residual = equation.residual()
        for symbol in residual.free_symbols:
            if not isinstance(symbol, sp.Symbol):
                continue
            base, shift = parse_alias(symbol.name)
            if base in all_names:
                shifts.setdefault(base, set()).add(shift)
    return shifts


def _build_steady_state_subs(
    model: ModelIR, shifts: Dict[str, set[int]]
) -> Tuple[Dict[sp.Symbol, float], Dict[sp.Symbol, float]]:
    steady_subs: Dict[sp.Symbol, float] = {}
    shock_subs: Dict[sp.Symbol, float] = {}
    for var in model.endo:
        value = float(model.initvals.get(var.name, 0.0))
        for shift in shifts.get(var.name, {0}):
            steady_subs[sp.Symbol(alias_for_shift(var.name, shift))] = value
    for shock in model.exo:
        for shift in shifts.get(shock.name, {0}):
            shock_subs[sp.Symbol(alias_for_shift(shock.name, shift))] = 0.0
    return steady_subs, shock_subs


def linearize(model: ModelIR) -> LinearizedModel:
    missing = [name for name, value in model.params.items() if math.isnan(value)]
    if missing:
        raise ValueError(f"Parameters missing numeric values: {missing}")

    shifts = _collect_symbol_shifts(model)

    endo_names = [var.name for var in model.endo]
    exo_names = [shock.name for shock in model.exo]
    n = len(endo_names)
    k = len(exo_names)

    steady_subs, shock_subs = _build_steady_state_subs(model, shifts)
    param_subs: Dict[sp.Symbol, float] = {
        sp.Symbol(name): float(value) for name, value in model.params.items()
    }

    subs_eval = steady_subs | shock_subs
    var_index = {name: idx for idx, name in enumerate(endo_names)}

    shift_values = sorted(
        {shift for name in endo_names for shift in shifts.get(name, {0})}
    )
    jacobians: Dict[int, sp.Matrix] = {
        shift: sp.zeros(n, n) for shift in shift_values
    }

    psi = sp.zeros(n, k)
    c = sp.zeros(n, 1)

    for i, equation in enumerate(model.equations):
        residual = sp.simplify(equation.residual().subs(param_subs))
        res_eval = sp.N(residual.subs(subs_eval))
        c[i, 0] = -res_eval

        for name in endo_names:
            available_shifts = shifts.get(name, {0})
            for shift in available_shifts:
                symbol = sp.Symbol(alias_for_shift(name, shift))
                deriv_val = sp.N(residual.diff(symbol).subs(subs_eval))
                jacobians.setdefault(shift, sp.zeros(n, n))
                jacobians[shift][i, var_index[name]] = deriv_val

        for j, shock_name in enumerate(exo_names):
            shock_symbol = sp.Symbol(alias_for_shift(shock_name, 0))
            d_shock = sp.N(residual.diff(shock_symbol).subs(subs_eval))
            if d_shock != 0:
                psi[i, j] = -d_shock

    lead_pairs: List[Tuple[str, int]] = []
    for shift in sorted(s for s in jacobians.keys() if s > 0):
        for name in endo_names:
            if shift in shifts.get(name, {0}):
                column = jacobians[shift][:, var_index[name]]
                if any(val != 0 for val in column):
                    lead_pairs.append((name, shift))
    lead_indices = {pair: idx for idx, pair in enumerate(lead_pairs)}

    max_lag = max((-shift for shift in jacobians if shift < 0), default=0)
    n_ext = n * (max_lag + 1)

    gam0_ext = sp.zeros(n_ext, n_ext)
    gam1_ext = sp.zeros(n_ext, n_ext)
    psi_ext = sp.zeros(n_ext, k)
    pi_ext = sp.zeros(n_ext, len(lead_pairs))
    c_ext = sp.zeros(n_ext, 1)

    gam0_ext[:n, :n] = jacobians.get(0, sp.zeros(n, n))
    c_ext[:n, :] = c
    psi_ext[:n, :] = psi

    for shift, matrix in jacobians.items():
        if shift < 0:
            lag = -shift
            if lag == 0:
                continue
            if lag > max_lag:
                continue
            start = (lag - 1) * n
            stop = start + n
            gam1_ext[:n, start:stop] -= matrix
        elif shift > 0:
            for name in endo_names:
                if shift not in shifts.get(name, {0}):
                    continue
                key = (name, shift)
                idx = lead_indices.get(key)
                if idx is None:
                    continue
                col = matrix[:, var_index[name]]
                pi_ext[:n, idx] = col

    for lag in range(1, max_lag + 1):
        row_start = n + (lag - 1) * n
        row_end = row_start + n
        gam0_ext[row_start:row_end, lag * n : (lag + 1) * n] = sp.eye(n)
        gam1_ext[row_start:row_end, (lag - 1) * n : lag * n] = sp.eye(n)

    state_names: List[str] = []
    for lag in range(max_lag + 1):
        suffix = "" if lag == 0 else f"_lag{lag}"
        for name in endo_names:
            state_names.append(name if not suffix else f"{name}{suffix}")

    return LinearizedModel(
        gam0=sp.Matrix(gam0_ext),
        gam1=sp.Matrix(gam1_ext),
        c=sp.Matrix(c_ext),
        psi=sp.Matrix(psi_ext),
        pi=sp.Matrix(pi_ext),
        state_names=state_names,
        shock_names=exo_names,
    )


__all__ = ["linearize"]
