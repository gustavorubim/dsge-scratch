from __future__ import annotations

import sympy as sp

from sgelabs.ir.model import LinearizedModel
from sgelabs.solve.gensys import solve_gensys


def test_solve_gensys_handles_zero_state_system() -> None:
    linearized = LinearizedModel(
        gam0=sp.zeros(0, 0),
        gam1=sp.zeros(0, 0),
        c=sp.zeros(0, 1),
        psi=sp.zeros(0, 0),
        pi=sp.zeros(0, 0),
        state_names=[],
        shock_names=[],
    )

    result = solve_gensys(linearized)

    assert result.g.shape == (0, 0)
    assert result.impact.shape == (0, 0)
    assert result.c.shape == (0, 1)
    assert result.eu == (1, 1)
    assert result.eigenvalues.size == 0
    assert result.state_names == []
    assert result.shock_names == []


def test_solve_gensys_zero_state_keeps_shock_dimension() -> None:
    linearized = LinearizedModel(
        gam0=sp.zeros(0, 0),
        gam1=sp.zeros(0, 0),
        c=sp.zeros(0, 1),
        psi=sp.zeros(0, 2),
        pi=sp.zeros(0, 0),
        state_names=[],
        shock_names=["eps_a", "eps_b"],
    )

    result = solve_gensys(linearized)

    assert result.impact.shape == (0, 2)
    assert result.shock_names == ["eps_a", "eps_b"]
