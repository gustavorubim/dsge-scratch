from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig

from sgelabs.ir.model import LinearizedModel


@dataclass(slots=True)
class GensysResult:
    g: NDArray[np.floating]
    c: NDArray[np.floating]
    impact: NDArray[np.floating]
    eu: Tuple[int, int]
    eigenvalues: NDArray[np.complexfloating]
    state_names: list[str]
    shock_names: list[str]


def solve_gensys(linearized: LinearizedModel, div: float | None = None) -> GensysResult:
    matrices = linearized.to_numpy()
    gam0 = matrices["gam0"]
    gam1 = matrices["gam1"]
    c_vec = matrices["c"].reshape(-1, 1)
    psi = matrices["psi"]
    pi = matrices["pi"]

    n = gam0.shape[0]
    if div is None:
        div = 1.0 + 1e-8

    eigvals = eig(gam1, gam0, right=False)
    finite_mask = np.isfinite(eigvals)
    n_stable = int(np.sum(np.abs(eigvals[finite_mask]) < div))
    rank_pi = int(np.linalg.matrix_rank(pi)) if pi.size else 0
    n_pred = n - rank_pi

    eu = (1 if n_stable >= n_pred else 0, 1 if n_stable <= n_pred else 0)

    G = np.linalg.lstsq(gam0, gam1, rcond=None)[0]
    if np.max(np.abs(gam0 @ G - gam1)) > 1e-6:
        raise RuntimeError("Failed to solve for transition matrix G")

    impact = (
        np.linalg.lstsq(gam0, psi, rcond=None)[0] if psi.size else np.zeros((n, 0))
    )
    if psi.size and np.max(np.abs(gam0 @ impact - psi)) > 1e-6:
        raise RuntimeError("Failed to solve for shock impact matrix")

    c_sol = np.linalg.lstsq(gam0, c_vec, rcond=None)[0]

    return GensysResult(
        g=G,
        c=c_sol,
        impact=impact,
        eu=eu,
        eigenvalues=eigvals,
        state_names=list(linearized.state_names),
        shock_names=list(linearized.shock_names),
    )


__all__ = ["GensysResult", "solve_gensys"]
