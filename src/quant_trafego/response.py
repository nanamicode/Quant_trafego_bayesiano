from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResponseEstimate:
    elasticity_mean: float
    elasticity_sd: float
    confidence: float
    n_days: int
    independent_spend_sd: float = 0.0
    effective_days: float = 0.0
    controls: str = "trend"

    @property
    def diminishing_returns_probability_proxy(self) -> float:
        if self.elasticity_sd <= 0:
            return float(self.elasticity_mean < 1.0)
        z = (1.0 - self.elasticity_mean) / self.elasticity_sd
        return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / max(float(np.sum(w)), 1e-12))


def _weighted_sd(x: np.ndarray, w: np.ndarray) -> float:
    mean = _weighted_mean(x, w)
    var = float(
        np.sum(w * np.square(x - mean))
        / max(float(np.sum(w)), 1e-12)
    )
    return float(np.sqrt(max(var, 0.0)))


def _nuisance_matrix(
    dates: pd.Series,
    *,
    include_weekday: bool,
) -> tuple[np.ndarray, str]:
    d = pd.to_datetime(dates).reset_index(drop=True)
    latest = d.max()
    span = max(
        float((latest - d.min()).total_seconds() / 86400.0),
        1.0,
    )
    t = (
        (d - latest).dt.total_seconds().to_numpy(dtype=float)
        / 86400.0
        / span
    )

    cols = [
        np.ones(len(d), dtype=float),
        t,
        np.square(t),
    ]
    controls = "trend_quadratic"

    if include_weekday:
        dow = d.dt.dayofweek.to_numpy(dtype=int)
        for value in range(1, 7):
            cols.append((dow == value).astype(float))
        controls += "+weekday"

    return np.column_stack(cols), controls


def _residualized_spend_sd(
    x: np.ndarray,
    nuisance: np.ndarray,
    w: np.ndarray,
) -> float:
    penalty = np.eye(nuisance.shape[1], dtype=float)
    penalty[0, 0] = 1e-8
    if nuisance.shape[1] >= 3:
        penalty[1, 1] = 0.10
        penalty[2, 2] = 0.25
    if nuisance.shape[1] > 3:
        penalty[3:, 3:] *= 2.0

    try:
        precision = (
            nuisance.T @ (w[:, None] * nuisance)
            + penalty
        )
        beta = np.linalg.solve(
            precision,
            nuisance.T @ (w * x),
        )
        residual = x - nuisance @ beta
    except np.linalg.LinAlgError:
        residual = x - _weighted_mean(x, w)

    return _weighted_sd(residual, w)


def estimate_response(
    df: pd.DataFrame,
    *,
    parent: ResponseEstimate | None = None,
    half_life_days: float = 21.0,
) -> ResponseEstimate:
    """
    Observational spend-response estimate.

    The spend coefficient is estimated after controlling for smooth calendar
    trend and, when enough history exists, weekday effects. This reduces
    obvious time/seasonality confounding but does not turn observational
    history into a causal intervention estimate.
    """
    daily = (
        df.groupby("date", as_index=False)
        .agg(
            spend=("spend", "sum"),
            conversions=("conversions", "sum"),
            revenue=("revenue", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[
        (daily["spend"] > 0)
        & np.isfinite(daily["spend"])
    ].reset_index(drop=True)

    prior_mean = (
        parent.elasticity_mean
        if parent is not None
        else 0.75
    )
    prior_sd = (
        max(parent.elasticity_sd, 0.15)
        if parent is not None
        else 0.30
    )

    n_days = len(daily)
    if n_days < 6:
        return ResponseEstimate(
            prior_mean,
            prior_sd,
            0.0,
            n_days,
            independent_spend_sd=0.0,
            effective_days=float(n_days),
            controls="insufficient",
        )

    x_raw = np.log(
        daily["spend"].to_numpy(dtype=float)
    )
    y = np.log(
        daily["conversions"].to_numpy(dtype=float)
        + 0.5
    )

    latest = daily["date"].max()
    age = (
        (latest - daily["date"])
        .dt.total_seconds()
        .to_numpy(dtype=float)
        / 86400.0
    )
    w = np.exp(
        -math.log(2.0)
        * age
        / max(half_life_days, 1.0)
    )
    effective_days = float(
        np.square(np.sum(w))
        / max(float(np.sum(np.square(w))), 1e-12)
    )

    include_weekday = (
        n_days >= 21
        and daily["date"].dt.dayofweek.nunique() >= 6
    )
    nuisance, controls = _nuisance_matrix(
        daily["date"],
        include_weekday=include_weekday,
    )

    x_center = x_raw - _weighted_mean(x_raw, w)
    independent_spend_sd = _residualized_spend_sd(
        x_center,
        nuisance,
        w,
    )

    if independent_spend_sd < 0.04:
        return ResponseEstimate(
            prior_mean,
            prior_sd,
            0.0,
            n_days,
            independent_spend_sd=independent_spend_sd,
            effective_days=effective_days,
            controls=controls,
        )

    X = np.column_stack(
        [nuisance[:, 0], x_center, nuisance[:, 1:]]
    )
    p = X.shape[1]

    initial_penalty = np.eye(p, dtype=float)
    initial_penalty[0, 0] = 1e-8
    initial_penalty[1, 1] = 1e-3
    if p > 2:
        initial_penalty[2, 2] = 0.10
    if p > 3:
        initial_penalty[3, 3] = 0.25
    if p > 4:
        initial_penalty[4:, 4:] *= 1.0

    try:
        beta0 = np.linalg.solve(
            X.T @ (w[:, None] * X)
            + initial_penalty,
            X.T @ (w * y),
        )
        resid = y - X @ beta0
        dof = max(
            effective_days - min(p, effective_days - 1.0),
            1.0,
        )
        sigma2 = max(
            float(np.sum(w * resid * resid) / dof),
            0.03**2,
        )

        prior_precision = np.zeros((p, p), dtype=float)
        prior_rhs = np.zeros(p, dtype=float)

        prior_precision[0, 0] = 1e-6
        prior_precision[1, 1] = 1.0 / (prior_sd**2)
        prior_rhs[1] = (
            prior_mean / (prior_sd**2)
        )

        if p > 2:
            prior_precision[2, 2] = 1.0
        if p > 3:
            prior_precision[3, 3] = 1.0 / (0.75**2)
        if p > 4:
            for j in range(4, p):
                prior_precision[j, j] = 1.0 / (0.50**2)

        likelihood_precision = (
            X.T @ (w[:, None] * X)
            / sigma2
        )
        precision = (
            likelihood_precision
            + prior_precision
        )
        rhs = (
            X.T @ (w * y) / sigma2
            + prior_rhs
        )
        beta = np.linalg.solve(
            precision,
            rhs,
        )
        cov = np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        return ResponseEstimate(
            prior_mean,
            prior_sd,
            0.0,
            n_days,
            independent_spend_sd=independent_spend_sd,
            effective_days=effective_days,
            controls=controls,
        )

    mean = float(
        np.clip(beta[1], 0.05, 1.50)
    )
    sd = float(
        np.clip(
            np.sqrt(max(cov[1, 1], 1e-12)),
            0.04,
            0.60,
        )
    )

    n_factor = float(
        np.clip(
            (effective_days - 5.0) / 24.0,
            0.0,
            1.0,
        )
    )
    spread_factor = float(
        np.clip(
            independent_spend_sd / 0.30,
            0.0,
            1.0,
        )
    )
    precision_factor = float(
        np.clip(
            0.25 / sd,
            0.0,
            1.0,
        )
    )
    confidence = (
        n_factor
        * spread_factor
        * precision_factor
    )

    return ResponseEstimate(
        elasticity_mean=mean,
        elasticity_sd=sd,
        confidence=float(
            np.clip(confidence, 0.0, 1.0)
        ),
        n_days=n_days,
        independent_spend_sd=independent_spend_sd,
        effective_days=effective_days,
        controls=controls,
    )
