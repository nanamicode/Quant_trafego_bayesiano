from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class TrendEstimate:
    mean: float
    sd: float
    p_positive: float
    confidence: float

    @property
    def effective_mean(self) -> float:
        return self.mean * self.confidence

    @property
    def effective_sd(self) -> float:
        return self.sd * self.confidence


@dataclass(frozen=True)
class TemporalSignal:
    ctr: TrendEstimate
    cvr: TrendEstimate
    p_recent_ctr_better: float
    p_recent_cvr_better: float
    regime_change_score: float
    instability_score: float
    recent_days: int
    history_days: int


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def _weighted_trend(
    successes: np.ndarray,
    trials: np.ndarray,
    dates: pd.Series,
    half_life_days: float,
) -> TrendEstimate:
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    valid = np.isfinite(successes) & np.isfinite(trials) & (trials > 0)

    if valid.sum() < 4:
        return TrendEstimate(0.0, 0.0, 0.5, 0.0)

    successes = successes[valid]
    trials = trials[valid]
    d = pd.to_datetime(dates.iloc[np.flatnonzero(valid)])
    latest = d.max()
    t = (d - latest).dt.total_seconds().to_numpy() / 86400.0

    # Jeffreys-style smoothing before the logit transform.
    rate = (successes + 0.5) / (trials + 1.0)
    y = _logit(rate)

    age = -t
    recency = np.exp(-math.log(2.0) * age / max(half_life_days, 1.0))
    median_trials = max(float(np.median(trials)), 1.0)
    information = np.sqrt(np.clip(trials / median_trials, 0.15, 12.0))
    w = recency * information

    X = np.column_stack([np.ones_like(t), t, t * t])
    # Weak shrinkage on slope and stronger shrinkage on curvature prevents
    # tiny datasets from extrapolating absurdly.
    ridge = np.diag([1e-6, 2.0, 20.0])
    xtwx = X.T @ (w[:, None] * X)
    xtwy = X.T @ (w * y)

    try:
        precision = xtwx + ridge
        beta = np.linalg.solve(precision, xtwy)
        resid = y - X @ beta
        dof = max(int(valid.sum()) - X.shape[1], 1)
        sigma2 = max(float(np.sum(w * resid * resid) / dof), 1e-6)
        cov = sigma2 * np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        return TrendEstimate(0.0, 0.0, 0.5, 0.0)

    # t=0 is the most recent date; beta[1] is therefore the current derivative.
    slope_mean = float(beta[1])
    slope_sd = float(np.sqrt(max(cov[1, 1], 1e-12)))
    z = slope_mean / max(slope_sd, 1e-9)
    p_positive = float(norm.cdf(z))

    n_factor = float(np.clip((valid.sum() - 3) / 14.0, 0.0, 1.0))
    certainty = float(np.clip(2.0 * abs(p_positive - 0.5), 0.0, 1.0))
    confidence = n_factor * certainty

    # Guardrail: a one-day logit shift above ~0.35 is almost always
    # extrapolation from noise in paid-media data.
    slope_mean = float(np.clip(slope_mean, -0.35, 0.35))
    slope_sd = float(np.clip(slope_sd, 0.0, 0.25))

    return TrendEstimate(slope_mean, slope_sd, p_positive, confidence)


def _beta_recent_probability(
    successes: np.ndarray,
    trials: np.ndarray,
    recent_mask: np.ndarray,
    rng: np.random.Generator,
    draws: int = 6000,
) -> float:
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)

    recent_s = float(successes[recent_mask].sum())
    recent_n = float(trials[recent_mask].sum())
    base_s = float(successes[~recent_mask].sum())
    base_n = float(trials[~recent_mask].sum())

    if recent_n <= 0 or base_n <= 0:
        return 0.5

    r = rng.beta(0.5 + recent_s, 0.5 + recent_n - recent_s, size=draws)
    b = rng.beta(0.5 + base_s, 0.5 + base_n - base_s, size=draws)
    return float(np.mean(r > b))


def analyze_temporal(
    df: pd.DataFrame,
    *,
    half_life_days: float = 14.0,
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
    daily["date"] = pd.to_datetime(daily["date"])

    n = len(daily)
    if n < 2:
        neutral = TrendEstimate(0.0, 0.0, 0.5, 0.0)
        return TemporalSignal(neutral, neutral, 0.5, 0.5, 0.0, 0.0, n, n)

    ctr_trend = _weighted_trend(
        daily["clicks"].to_numpy(),
        daily["impressions"].to_numpy(),
        daily["date"],
        half_life_days,
    )
    cvr_trend = _weighted_trend(
        daily["conversions"].to_numpy(),
        daily["clicks"].to_numpy(),
        daily["date"],
        half_life_days,
    )

    rdays = int(min(max(recent_days, 2), max(n // 2, 2)))
    cutoff = daily["date"].max() - pd.Timedelta(days=rdays - 1)
    recent_mask = (daily["date"] >= cutoff).to_numpy()

    # Need observations on both sides; if calendar gaps make the split empty,
    # fall back to the last third of rows.
    if recent_mask.all() or (~recent_mask).all():
        recent_mask = np.zeros(n, dtype=bool)
        recent_mask[max(1, n - max(2, n // 3)) :] = True

    rng = np.random.default_rng(seed)
    p_ctr = _beta_recent_probability(
        daily["clicks"].to_numpy(),
        daily["impressions"].to_numpy(),
        recent_mask,
        rng,
    )
    p_cvr = _beta_recent_probability(
        daily["conversions"].to_numpy(),
        daily["clicks"].to_numpy(),
        recent_mask,
        rng,
    )

    sample_factor = float(np.clip((n - 3) / 14.0, 0.0, 1.0))
    regime = sample_factor * max(
        2.0 * abs(p_ctr - 0.5),
        2.0 * abs(p_cvr - 0.5),
    )

    ctr_daily = (daily["clicks"].to_numpy() + 0.5) / (
        daily["impressions"].to_numpy() + 1.0
    )
    cvr_daily = (daily["conversions"].to_numpy() + 0.5) / (
        daily["clicks"].to_numpy() + 1.0
    )
    logits = np.concatenate([_logit(ctr_daily), _logit(cvr_daily)])
    instability = float(np.clip(np.nanstd(logits) / 1.5, 0.0, 1.0))

    return TemporalSignal(
        ctr=ctr_trend,
        cvr=cvr_trend,
        p_recent_ctr_better=p_ctr,
        p_recent_cvr_better=p_cvr,
        regime_change_score=float(np.clip(regime, 0.0, 1.0)),
        instability_score=instability,
        recent_days=int(recent_mask.sum()),
        history_days=n,
    )
