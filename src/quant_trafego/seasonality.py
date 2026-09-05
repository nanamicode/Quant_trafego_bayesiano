from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeeklyRateEffect:
    means: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    confidence: float
    n_days: int
    effective_days: float

    def future_shift(
        self,
        last_date,
        horizon_days: int,
        *,
        from_last_day: bool = False,
    ) -> tuple[float, float]:
        if (
            horizon_days <= 0
            or self.confidence <= 0
        ):
            return 0.0, 0.0

        future = pd.date_range(
            pd.Timestamp(last_date)
            + pd.Timedelta(days=1),
            periods=int(horizon_days),
            freq="D",
        )
        counts = np.bincount(
            future.dayofweek,
            minlength=7,
        ).astype(float)
        weights = counts / max(
            float(counts.sum()),
            1.0,
        )

        means = np.asarray(
            self.means,
            dtype=float,
        )
        cov = np.asarray(
            self.covariance,
            dtype=float,
        )

        contrast = weights.copy()
        if from_last_day:
            contrast[
                pd.Timestamp(last_date).dayofweek
            ] -= 1.0

        mean = float(
            contrast @ means
        )
        var = float(
            contrast @ cov @ contrast
        )

        return (
            mean * self.confidence,
            float(
                np.sqrt(max(var, 0.0))
                * self.confidence
            ),
        )


@dataclass(frozen=True)
class WeeklySeasonalitySignal:
    ctr: WeeklyRateEffect
    cvr: WeeklyRateEffect

    @property
    def confidence(self) -> float:
        return max(
            self.ctr.confidence,
            self.cvr.confidence,
        )


def _neutral(
    n_days: int,
) -> WeeklyRateEffect:
    return WeeklyRateEffect(
        means=(0.0,) * 7,
        covariance=tuple(
            tuple(
                0.0
                for _ in range(7)
            )
            for _ in range(7)
        ),
        confidence=0.0,
        n_days=n_days,
        effective_days=float(n_days),
    )


def _logit(
    p: np.ndarray,
) -> np.ndarray:
    p = np.clip(
        p,
        1e-8,
        1 - 1e-8,
    )
    return np.log(
        p / (1.0 - p)
    )


def _fit_weekly_rate(
    successes: np.ndarray,
    trials: np.ndarray,
    dates: pd.Series,
    *,
    half_life_days: float,
    min_days: int,
) -> WeeklyRateEffect:
    successes = np.asarray(
        successes,
        dtype=float,
    )
    trials = np.asarray(
        trials,
        dtype=float,
    )
    dates = pd.to_datetime(
        dates
    ).reset_index(drop=True)

    valid = (
        np.isfinite(successes)
        & np.isfinite(trials)
        & (trials > 0)
        & dates.notna().to_numpy()
    )
    if int(valid.sum()) < min_days:
        return _neutral(
            int(valid.sum())
        )

    successes = successes[valid]
    trials = trials[valid]
    d = dates.iloc[
        np.flatnonzero(valid)
    ].reset_index(drop=True)
    dow = d.dt.dayofweek.to_numpy(
        dtype=int
    )

    if len(np.unique(dow)) < 6:
        return _neutral(
            len(d)
        )

    rate = (
        successes + 0.5
    ) / (
        trials + 1.0
    )
    y = _logit(rate)

    latest = d.max()
    span_days = max(
        float(
            (
                latest - d.min()
            ).total_seconds()
            / 86400.0
        ),
        1.0,
    )
    t = (
        (
            d - latest
        )
        .dt.total_seconds()
        .to_numpy(dtype=float)
        / 86400.0
        / span_days
    )

    age = (
        (
            latest - d
        )
        .dt.total_seconds()
        .to_numpy(dtype=float)
        / 86400.0
    )
    recency = np.exp(
        -math.log(2.0)
        * age
        / max(
            half_life_days,
            1.0,
        )
    )
    median_trials = max(
        float(
            np.median(trials)
        ),
        1.0,
    )
    information = np.sqrt(
        np.clip(
            trials / median_trials,
            0.20,
            10.0,
        )
    )
    w = recency * information

    columns = [
        np.ones(len(d)),
        t,
        np.square(t),
    ]
    for day in range(1, 7):
        columns.append(
            (dow == day).astype(float)
        )
    X = np.column_stack(columns)
    p = X.shape[1]

    initial_penalty = np.eye(
        p,
        dtype=float,
    )
    initial_penalty[0, 0] = 1e-8
    initial_penalty[1, 1] = 0.25
    initial_penalty[2, 2] = 0.75
    initial_penalty[3:, 3:] *= 2.0

    try:
        beta0 = np.linalg.solve(
            X.T @ (
                w[:, None] * X
            )
            + initial_penalty,
            X.T @ (w * y),
        )
        resid = y - X @ beta0

        effective_days = float(
            np.square(np.sum(w))
            / max(
                float(
                    np.sum(
                        np.square(w)
                    )
                ),
                1e-12,
            )
        )
        dof = max(
            effective_days
            - min(
                p,
                effective_days - 1,
            ),
            1.0,
        )
        sigma2 = max(
            float(
                np.sum(
                    w
                    * np.square(resid)
                )
                / dof
            ),
            0.02**2,
        )

        prior_precision = np.eye(
            p,
            dtype=float,
        )
        prior_precision[0, 0] = 1e-6
        prior_precision[1, 1] = 1.0
        prior_precision[2, 2] = (
            1.0 / (0.75**2)
        )
        prior_precision[3:, 3:] *= (
            1.0 / (0.50**2)
        )

        precision = (
            X.T @ (
                w[:, None] * X
            )
            / sigma2
            + prior_precision
        )
        beta = np.linalg.solve(
            precision,
            X.T @ (w * y) / sigma2,
        )
        cov_beta = np.linalg.inv(
            precision
        )
    except np.linalg.LinAlgError:
        return _neutral(
            len(d)
        )

    raw_design = np.zeros(
        (7, p),
        dtype=float,
    )
    for day in range(1, 7):
        raw_design[
            day,
            2 + day,
        ] = 1.0

    reference_weights = np.array(
        [
            float(
                w[dow == day].sum()
            )
            for day in range(7)
        ],
        dtype=float,
    )
    reference_weights /= max(
        float(
            reference_weights.sum()
        ),
        1e-12,
    )
    reference_contrast = (
        reference_weights
        @ raw_design
    )
    contrasts = (
        raw_design
        - reference_contrast[None, :]
    )

    means = contrasts @ beta
    means = np.clip(
        means,
        -0.80,
        0.80,
    )
    cov = (
        contrasts
        @ cov_beta
        @ contrasts.T
    )
    cov = (
        cov + cov.T
    ) / 2.0

    raw_counts = np.bincount(
        dow,
        minlength=7,
    )
    coverage_factor = float(
        np.clip(
            np.min(raw_counts)
            / 3.0,
            0.0,
            1.0,
        )
    )
    n_factor = float(
        np.clip(
            (
                effective_days
                - min_days
            )
            / 42.0,
            0.0,
            1.0,
        )
    )
    typical_sd = float(
        np.sqrt(
            max(
                np.mean(
                    np.clip(
                        np.diag(cov),
                        0.0,
                        None,
                    )
                ),
                1e-12,
            )
        )
    )
    precision_factor = float(
        np.clip(
            0.30
            / max(
                typical_sd,
                1e-9,
            ),
            0.0,
            1.0,
        )
    )
    confidence = (
        n_factor
        * coverage_factor
        * precision_factor
    )

    return WeeklyRateEffect(
        means=tuple(
            float(x)
            for x in means
        ),
        covariance=tuple(
            tuple(
                float(v)
                for v in row
            )
            for row in cov
        ),
        confidence=float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        ),
        n_days=len(d),
        effective_days=effective_days,
    )


def analyze_weekly_seasonality(
    df: pd.DataFrame,
    *,
    half_life_days: float = 56.0,
    min_days: int = 21,
) -> WeeklySeasonalitySignal:
    daily = (
        df.groupby(
            "date",
            as_index=False,
        )
        .agg(
            impressions=(
                "impressions",
                "sum",
            ),
            clicks=(
                "clicks",
                "sum",
            ),
            conversions=(
                "conversions",
                "sum",
            ),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    ctr = _fit_weekly_rate(
        daily["clicks"].to_numpy(),
        daily["impressions"].to_numpy(),
        daily["date"],
        half_life_days=half_life_days,
        min_days=min_days,
    )
    cvr = _fit_weekly_rate(
        daily["conversions"].to_numpy(),
        daily["clicks"].to_numpy(),
        daily["date"],
        half_life_days=half_life_days,
        min_days=min_days,
    )
    return WeeklySeasonalitySignal(
        ctr=ctr,
        cvr=cvr,
    )
