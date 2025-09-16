import numpy as np

from sgelabs.io import parse_mod_file
from sgelabs.ir import linearize
from sgelabs.solve import solve_gensys


def test_linearize_companion_shape() -> None:
    model = parse_mod_file('examples/rbc_basic/rbc.mod')
    lin = linearize(model)
    n_states = len(lin.state_names)
    assert lin.gam0.shape == (n_states, n_states)
    assert lin.gam1.shape == (n_states, n_states)


def test_solver_returns_gensys_result() -> None:
    model = parse_mod_file('examples/rbc_basic/rbc.mod')
    lin = linearize(model)
    sol = solve_gensys(lin)
    assert sol.g.shape[0] == sol.g.shape[1] == len(lin.state_names)
    assert sol.impact.shape[0] == len(lin.state_names)
