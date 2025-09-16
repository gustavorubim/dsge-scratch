from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_irfs(
    g: NDArray[np.floating],
    impact: NDArray[np.floating],
    horizon: int = 40,
) -> NDArray[np.floating]:
    g = np.asarray(g, dtype=float)
    impact = np.asarray(impact, dtype=float)
    n, k = impact.shape
    irfs = np.zeros((horizon + 1, n, k), dtype=float)
    irfs[0] = impact
    current = impact.copy()
    for h in range(1, horizon + 1):
        current = g @ current
        irfs[h] = current
    return irfs


def compute_fevd(
    g: NDArray[np.floating],
    impact: NDArray[np.floating],
    sigma_e: NDArray[np.floating],
    horizon: int = 40,
) -> NDArray[np.floating]:
    g = np.asarray(g, dtype=float)
    impact = np.asarray(impact, dtype=float)
    sigma_e = np.asarray(sigma_e, dtype=float)
    n, k = impact.shape
    if sigma_e.shape == (k,):
        shock_std = np.sqrt(sigma_e)
    else:
        shock_std = np.sqrt(np.diag(sigma_e))
    irfs = compute_irfs(g, impact, horizon - 1)
    fevd = np.zeros((n, k), dtype=float)
    total = np.zeros(n, dtype=float)
    for h in range(horizon):
        scaled = irfs[h] * shock_std[np.newaxis, :]
        fevd += scaled ** 2
        total += np.sum(scaled ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.divide(fevd, total[:, None], out=np.zeros_like(fevd), where=total[:, None] != 0.0)
    return shares


def compute_fevd_ts(
    g: NDArray[np.floating],
    impact: NDArray[np.floating],
    sigma_e: NDArray[np.floating],
    horizon: int = 40,
) -> NDArray[np.floating]:
    """FEVD time series by horizon.

    Returns an array of shape (horizon, n, k), where each t index contains the
    cumulative FEVD shares up to horizon t (1-based horizons).
    """
    g = np.asarray(g, dtype=float)
    impact = np.asarray(impact, dtype=float)
    sigma_e = np.asarray(sigma_e, dtype=float)
    n, k = impact.shape
    if sigma_e.shape == (k,):
        shock_std = np.sqrt(sigma_e)
    else:
        shock_std = np.sqrt(np.diag(sigma_e))

    irfs = compute_irfs(g, impact, horizon - 1)  # (horizon, n, k)
    scaled = irfs * shock_std[np.newaxis, np.newaxis, :]  # scale shocks
    sq = scaled ** 2
    cum = np.cumsum(sq, axis=0)  # (horizon, n, k)
    total = np.sum(cum, axis=2)  # (horizon, n)

    shares = np.zeros_like(cum)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.divide(cum, total[:, :, None], out=np.zeros_like(cum), where=total[:, :, None] != 0.0)
    return shares


def historical_decomp(
    g: NDArray[np.floating],
    impact: NDArray[np.floating],
    shocks: NDArray[np.floating],
) -> NDArray[np.floating]:
    g = np.asarray(g, dtype=float)
    impact = np.asarray(impact, dtype=float)
    shocks = np.asarray(shocks, dtype=float)
    n, k = impact.shape
    T = shocks.shape[0]
    contributions = np.zeros((k, T, n), dtype=float)
    state_contrib = np.zeros((k, n), dtype=float)
    for t in range(T):
        state_contrib = state_contrib @ g.T
        for j in range(k):
            state_contrib[j] += impact[:, j] * shocks[t, j]
        contributions[:, t, :] = state_contrib
    return contributions


__all__ = ["compute_irfs", "compute_fevd", "compute_fevd_ts", "historical_decomp"]
