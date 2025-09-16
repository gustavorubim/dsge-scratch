import numpy as np

from sgelabs.analysis import compute_fevd, compute_irfs, historical_decomp
from sgelabs.io import parse_mod_file
from sgelabs.ir import linearize
from sgelabs.solve import solve_gensys


def _load_solution():
    model = parse_mod_file('examples/rbc_basic/rbc.mod')
    lin = linearize(model)
    sol = solve_gensys(lin)
    return lin, sol


def test_compute_irfs_matches_transition() -> None:
    lin, sol = _load_solution()
    irfs = compute_irfs(sol.g, sol.impact, horizon=5)
    expected = sol.g @ sol.impact
    np.testing.assert_allclose(irfs[1], expected, rtol=1e-6, atol=1e-9)


def test_compute_fevd_sum_to_one() -> None:
    lin, sol = _load_solution()
    sigma = np.ones(sol.impact.shape[1]) * 0.01 ** 2
    fevd = compute_fevd(sol.g, sol.impact, np.array(sigma), horizon=5)
    sums = fevd.sum(axis=1)
    mask = sums > 1e-8
    np.testing.assert_allclose(sums[mask], np.ones_like(sums[mask]), atol=1e-6)


def test_historical_decomp_shape() -> None:
    lin, sol = _load_solution()
    eps = np.ones((4, sol.impact.shape[1]))
    contrib = historical_decomp(sol.g, sol.impact, eps)
    assert contrib.shape == (sol.impact.shape[1], eps.shape[0], sol.g.shape[0])
