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

    @property
    def diminishing_returns_probability_proxy(self) -> float:
        # Gaussian approximation: P(elasticity < 1).
        if self.elasticity_sd <= 0:
            return float(self.elasticity_mean < 1.0)
        z = (1.0 - self.elasticity_mean) / self.elasticity_sd
        # Normal CDF without adding another dependency.
        return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def estimate_response(
    df: pd.DataFrame,
    *,
    parent: ResponseEstimate | None = None,
    half_life_days: float = 21.0,
) -> ResponseEstimate:
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
    daily = daily[(daily["spend"] > 0) & np.isfinite(daily["spend"])]

    prior_mean = parent.elasticity_mean if parent else 0.75
    prior_sd = max(parent.elasticity_sd, 0.15) if parent else 0.30

    if len(daily) < 6:
        return ResponseEstimate(prior_mean, prior_sd, 0.0, len(daily))

    x = np.log(daily["spend"].to_numpy(dtype=float))
    # Conversions are used instead of revenue so ticket variation is modeled
    # separately in the Monte Carlo stage.
    y = np.log(daily["conversions"].to_numpy(dtype=float) + 0.5)

    if float(np.std(x)) < 0.08:
        return ResponseEstimate(prior_mean, prior_sd, 0.0, len(daily))

    latest = daily["date"].max()
    age = (latest - daily["date"]).dt.total_seconds().to_numpy() / 86400.0
    w = np.exp(-math.log(2.0) * age / max(half_life_days, 1.0))

    x_center = x - np.average(x, weights=w)
    X = np.column_stack([np.ones_like(x_center), x_center])

    # Initial weighted fit only to estimate observation noise.
    weak = np.diag([1e-6, 1e-3])
    try:
        beta0 = np.linalg.solve(X.T @ (w[:, None] * X) + weak, X.T @ (w * y))
        resid = y - X @ beta0
        sigma2 = max(float(np.sum(w * resid * resid) / max(len(y) - 2, 1)), 0.03**2)

        prior_precision = np.diag([1e-6, 1.0 / (prior_sd**2)])
        likelihood_precision = X.T @ (w[:, None] * X) / sigma2
        precision = likelihood_precision + prior_precision
        rhs = X.T @ (w * y) / sigma2 + np.array([0.0, prior_mean / (prior_sd**2)])
        beta = np.linalg.solve(precision, rhs)
        cov = np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        return ResponseEstimate(prior_mean, prior_sd, 0.0, len(daily))

    mean = float(np.clip(beta[1], 0.05, 1.50))
    sd = float(np.clip(np.sqrt(max(cov[1, 1], 1e-12)), 0.04, 0.60))

    n_factor = float(np.clip((len(daily) - 5) / 24.0, 0.0, 1.0))
    spread_factor = float(np.clip(np.std(x) / 0.45, 0.0, 1.0))
    precision_factor = float(np.clip(0.25 / sd, 0.0, 1.0))
    confidence = n_factor * spread_factor * precision_factor

    return ResponseEstimate(mean, sd, confidence, len(daily))
