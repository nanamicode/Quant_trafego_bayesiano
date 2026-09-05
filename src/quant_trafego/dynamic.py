from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from .temporal import TemporalSignal, TrendEstimate, analyze_temporal


@dataclass(frozen=True)
class DynamicRateState:
    level_mean: float
    level_sd: float
    trend_mean: float
    trend_sd: float
    process_level_var: float
    process_trend_var: float
    predictive_log_score: float
    n_observations: int
    confidence: float


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1.0 - p))


def _rate_observations(
    successes: np.ndarray,
    trials: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    valid = np.isfinite(successes) & np.isfinite(trials) & (trials > 0)
    s = successes[valid]
    n = trials[valid]
    p = (s + 0.5) / (n + 1.0)
    z = _logit(p)
    # Delta-method observation variance on the logit scale.
    r = 1.0 / np.maximum(n * p * (1.0 - p), 1e-6)
    r = np.clip(r, 1e-5, 8.0)
    return z, r, n


def _filter_local_linear(
    z: np.ndarray,
    r: np.ndarray,
    *,
    q_level: float,
    q_trend: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(z) == 0:
        return np.zeros(2), np.eye(2), float("-inf")

    f = np.array([[1.0, 1.0], [0.0, 1.0]])
    h = np.array([1.0, 0.0])
    q = np.diag([q_level, q_trend])

    m = np.array([float(z[0]), 0.0])
    p_cov = np.diag([max(float(r[0]), 0.25), 0.04])
    log_score = 0.0

    for i in range(1, len(z)):
        m_pred = f @ m
        p_pred = f @ p_cov @ f.T + q

        s_var = float(h @ p_pred @ h + r[i])
        innovation = float(z[i] - h @ m_pred)
        if i >= 3:
            log_score += -0.5 * (
                math.log(2.0 * math.pi * s_var)
                + (innovation * innovation) / s_var
            )

        k = (p_pred @ h) / s_var
        m = m_pred + k * innovation
        p_cov = p_pred - np.outer(k, h) @ p_pred
        p_cov = (p_cov + p_cov.T) / 2.0

    return m, p_cov, float(log_score)


def fit_dynamic_rate(
    successes: np.ndarray,
    trials: np.ndarray,
) -> DynamicRateState:
    z, r, information = _rate_observations(successes, trials)
    n = len(z)
    if n < 5:
        return DynamicRateState(
            level_mean=float(z[-1]) if n else 0.0,
            level_sd=float(np.sqrt(r[-1])) if n else 1.0,
            trend_mean=0.0,
            trend_sd=0.15,
            process_level_var=0.0,
            process_trend_var=0.0,
            predictive_log_score=float("nan"),
            n_observations=n,
            confidence=0.0,
        )

    level_grid = np.array(
        [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
        dtype=float,
    )
    trend_ratios = (0.005, 0.02, 0.08)

    best = None
    for q_level in level_grid:
        for ratio in trend_ratios:
            q_trend = q_level * ratio
            m, cov, score = _filter_local_linear(
                z,
                r,
                q_level=float(q_level),
                q_trend=float(q_trend),
            )
            if best is None or score > best[0]:
                best = (score, q_level, q_trend, m, cov)

    assert best is not None
    score, q_level, q_trend, m, cov = best

    trend_mean = float(np.clip(m[1], -0.35, 0.35))
    trend_sd = float(
        np.clip(np.sqrt(max(float(cov[1, 1]), 1e-12)), 1e-5, 0.30)
    )
    level_sd = float(
        np.clip(np.sqrt(max(float(cov[0, 0]), 1e-12)), 1e-5, 2.0)
    )
    p_positive = float(norm.cdf(trend_mean / max(trend_sd, 1e-9)))

    n_factor = float(np.clip((n - 4) / 20.0, 0.0, 1.0))
    information_factor = float(
        np.clip(np.log1p(np.median(information)) / np.log1p(5000.0), 0.0, 1.0)
    )
    certainty = float(np.clip(2.0 * abs(p_positive - 0.5), 0.0, 1.0))
    confidence = n_factor * (0.4 + 0.6 * information_factor) * certainty

    return DynamicRateState(
        level_mean=float(m[0]),
        level_sd=level_sd,
        trend_mean=trend_mean,
        trend_sd=trend_sd,
        process_level_var=float(q_level),
        process_trend_var=float(q_trend),
        predictive_log_score=float(score),
        n_observations=n,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


def analyze_state_space_temporal(
    df: pd.DataFrame,
    *,
    recent_days: int = 7,
    seed: int = 42,
) -> TemporalSignal:
    daily = (
        df.groupby("date", as_index=False)
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    baseline = analyze_temporal(
        df,
        recent_days=recent_days,
        seed=seed,
    )
    ctr_state = fit_dynamic_rate(
        daily["clicks"].to_numpy(),
        daily["impressions"].to_numpy(),
    )
    cvr_state = fit_dynamic_rate(
        daily["conversions"].to_numpy(),
        daily["clicks"].to_numpy(),
    )

    def trend(state: DynamicRateState) -> TrendEstimate:
        p_positive = float(
            norm.cdf(state.trend_mean / max(state.trend_sd, 1e-9))
        )
        return TrendEstimate(
            mean=state.trend_mean,
            sd=state.trend_sd,
            p_positive=p_positive,
            confidence=state.confidence,
        )

    return TemporalSignal(
        ctr=trend(ctr_state),
        cvr=trend(cvr_state),
        p_recent_ctr_better=baseline.p_recent_ctr_better,
        p_recent_cvr_better=baseline.p_recent_cvr_better,
        regime_change_score=baseline.regime_change_score,
        instability_score=baseline.instability_score,
        recent_days=baseline.recent_days,
        history_days=baseline.history_days,
        ctr_current_logit_shift=baseline.ctr_current_logit_shift,
        cvr_current_logit_shift=baseline.cvr_current_logit_shift,
        ctr_current_shift_confidence=baseline.ctr_current_shift_confidence,
        cvr_current_shift_confidence=baseline.cvr_current_shift_confidence,
    )
