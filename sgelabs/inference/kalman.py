from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class KalmanFilterResult:
    filtered: NDArray[np.floating]
    predicted: NDArray[np.floating]
    cov_filtered: NDArray[np.floating]
    cov_predicted: NDArray[np.floating]
    innovations: NDArray[np.floating]
    innovation_cov: NDArray[np.floating]
    loglik: float


@dataclass(slots=True)
class KalmanSmootherResult:
    smoothed: NDArray[np.floating]
    cov_smoothed: NDArray[np.floating]
    smoothed_shocks: NDArray[np.floating]


def kalman_filter(
    transition: NDArray[np.floating],
    impact: NDArray[np.floating],
    shock_cov: NDArray[np.floating],
    measurement: NDArray[np.floating],
    meas_cov: NDArray[np.floating],
    observations: NDArray[np.floating],
    initial_state: Optional[NDArray[np.floating]] = None,
    initial_cov: Optional[NDArray[np.floating]] = None,
    transition_const: Optional[NDArray[np.floating]] = None,
    measurement_const: Optional[NDArray[np.floating]] = None,
) -> KalmanFilterResult:
    transition = np.asarray(transition, dtype=float)
    impact = np.asarray(impact, dtype=float)
    shock_cov = np.asarray(shock_cov, dtype=float)
    measurement = np.asarray(measurement, dtype=float)
    meas_cov = np.asarray(meas_cov, dtype=float)
    observations = np.asarray(observations, dtype=float)

    n = transition.shape[0]
    T = observations.shape[0]
    m = observations.shape[1]

    state = np.zeros(n, dtype=float) if initial_state is None else np.asarray(initial_state, dtype=float)
    cov = np.eye(n, dtype=float) * 1e4 if initial_cov is None else np.asarray(initial_cov, dtype=float)
    const_t = np.zeros(n, dtype=float) if transition_const is None else np.asarray(transition_const, dtype=float)
    const_m = np.zeros(m, dtype=float) if measurement_const is None else np.asarray(measurement_const, dtype=float)

    Q = impact @ shock_cov @ impact.T

    filtered = np.zeros((T, n), dtype=float)
    predicted = np.zeros((T + 1, n), dtype=float)
    cov_filtered = np.zeros((T, n, n), dtype=float)
    cov_predicted = np.zeros((T + 1, n, n), dtype=float)
    innovations = np.zeros((T, m), dtype=float)
    innovation_cov = np.zeros((T, m, m), dtype=float)

    predicted[0] = state
    cov_predicted[0] = cov
    loglik = 0.0

    I_n = np.eye(n)

    for t in range(T):
        y_pred = measurement @ state + const_m
        S = measurement @ cov @ measurement.T + meas_cov
        innovation = observations[t] - y_pred
        innovations[t] = innovation
        innovation_cov[t] = S

        K = cov @ measurement.T
        S_inv = np.linalg.pinv(S)
        K = K @ S_inv
        state = state + K @ innovation
        cov = (I_n - K @ measurement) @ cov

        filtered[t] = state
        cov_filtered[t] = cov

        loglik -= 0.5 * (
            m * np.log(2.0 * np.pi)
            + np.linalg.slogdet(S)[1]
            + innovation.T @ S_inv @ innovation
        )

        state = transition @ state + const_t
        cov = transition @ cov @ transition.T + Q

        predicted[t + 1] = state
        cov_predicted[t + 1] = cov

    return KalmanFilterResult(
        filtered=filtered,
        predicted=predicted,
        cov_filtered=cov_filtered,
        cov_predicted=cov_predicted,
        innovations=innovations,
        innovation_cov=innovation_cov,
        loglik=loglik,
    )


def kalman_smoother(
    result: KalmanFilterResult,
    transition: NDArray[np.floating],
    impact: NDArray[np.floating],
    shocks: Optional[NDArray[np.floating]] = None,
    transition_const: Optional[NDArray[np.floating]] = None,
) -> KalmanSmootherResult:
    transition = np.asarray(transition, dtype=float)
    impact = np.asarray(impact, dtype=float)
    const_t = (
        np.zeros(transition.shape[0], dtype=float)
        if transition_const is None
        else np.asarray(transition_const, dtype=float)
    )

    filtered = result.filtered
    predicted = result.predicted
    cov_filtered = result.cov_filtered
    cov_pred = result.cov_predicted

    T, n = filtered.shape
    smoothed = filtered.copy()
    cov_smoothed = cov_filtered.copy()

    for t in range(T - 2, -1, -1):
        C = cov_filtered[t] @ transition.T
        C = C @ np.linalg.pinv(cov_pred[t + 1])
        smoothed[t] = filtered[t] + C @ (smoothed[t + 1] - predicted[t + 1])
        cov_smoothed[t] = cov_filtered[t] + C @ (cov_smoothed[t + 1] - cov_pred[t + 1]) @ C.T

    if shocks is None:
        shocks = np.zeros((T, impact.shape[1]), dtype=float)
    else:
        shocks = np.asarray(shocks, dtype=float)

    smoothed_shocks = np.zeros((T, impact.shape[1]), dtype=float)
    state_prev = predicted[0]
    for t in range(T):
        expected_state = transition @ state_prev + const_t
        residual = smoothed[t] - expected_state
        shock_solution, *_ = np.linalg.lstsq(impact, residual, rcond=None)
        smoothed_shocks[t] = shock_solution
        state_prev = smoothed[t]

    return KalmanSmootherResult(
        smoothed=smoothed,
        cov_smoothed=cov_smoothed,
        smoothed_shocks=smoothed_shocks,
    )


__all__ = ["KalmanFilterResult", "KalmanSmootherResult", "kalman_filter", "kalman_smoother"]
