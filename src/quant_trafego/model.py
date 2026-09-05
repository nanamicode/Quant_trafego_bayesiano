from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BetaPosterior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (
            self.alpha + self.beta
        )

    @property
    def strength(self) -> float:
        return self.alpha + self.beta

    def sample(
        self,
        rng: np.random.Generator,
        n: int,
    ) -> np.ndarray:
        return rng.beta(
            self.alpha,
            self.beta,
            size=n,
        )


@dataclass(frozen=True)
class SimulationContext:
    """
    Latent future worlds shared by every candidate intervention for one entity.

    Sharing these draws prevents action comparisons from accidentally comparing
    different posterior worlds. Observation noise remains action-specific.
    """

    cpm: np.ndarray
    aov: np.ndarray
    ctr: np.ndarray
    cvr: np.ndarray
    elasticity: np.ndarray

    @property
    def draws(self) -> int:
        return int(len(self.ctr))


def beta_from_mean(
    mean: float,
    strength: float,
) -> BetaPosterior:
    mean = float(
        np.clip(
            mean,
            1e-9,
            1 - 1e-9,
        )
    )
    strength = max(
        float(strength),
        2.0,
    )
    return BetaPosterior(
        alpha=1.0 + mean * strength,
        beta=1.0 + (1.0 - mean) * strength,
    )


def update_beta(
    prior: BetaPosterior,
    successes: float,
    trials: float,
) -> BetaPosterior:
    s = max(
        float(successes),
        0.0,
    )
    n = max(
        float(trials),
        s,
    )
    return BetaPosterior(
        prior.alpha + s,
        prior.beta + n - s,
    )


def shrink_to(
    parent: BetaPosterior,
    strength: float,
) -> BetaPosterior:
    return beta_from_mean(
        parent.mean,
        strength,
    )


def aggregate(
    df: pd.DataFrame,
) -> dict:
    impressions = float(
        df["impressions"].sum()
    )
    clicks = float(
        df["clicks"].sum()
    )
    conversions = float(
        df["conversions"].sum()
    )
    spend = float(
        df["spend"].sum()
    )
    revenue = float(
        df["revenue"].sum()
    )
    days = max(
        int(df["date"].nunique()),
        1,
    )

    daily = (
        df.groupby(
            "date",
            as_index=False,
        )
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

    daily["cpm"] = np.where(
        daily["impressions"] > 0,
        daily["spend"]
        / daily["impressions"]
        * 1000.0,
        np.nan,
    )
    daily["aov"] = np.where(
        daily["conversions"] > 0,
        daily["revenue"]
        / daily["conversions"],
        np.nan,
    )

    return {
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "spend": spend,
        "revenue": revenue,
        "days": days,
        "daily": daily,
        "ctr": (
            clicks / impressions
            if impressions
            else 0.0
        ),
        "cvr": (
            conversions / clicks
            if clicks
            else 0.0
        ),
        "cpc": (
            spend / clicks
            if clicks
            else np.nan
        ),
        "cpa": (
            spend / conversions
            if conversions
            else np.nan
        ),
        "roas": (
            revenue / spend
            if spend
            else 0.0
        ),
        "cpm": (
            spend / impressions * 1000.0
            if impressions
            else np.nan
        ),
        "aov": (
            revenue / conversions
            if conversions
            else np.nan
        ),
    }


def _lognormal_params(
    values: np.ndarray,
    default_mean: float,
    default_cv: float,
):
    values = np.asarray(
        values,
        dtype=float,
    )
    values = values[
        np.isfinite(values)
        & (values > 0)
    ]

    if len(values) >= 2:
        logs = np.log(values)
        return (
            float(logs.mean()),
            float(
                max(
                    logs.std(ddof=1),
                    0.05,
                )
            ),
        )

    mean = max(
        float(default_mean),
        1e-6,
    )
    sigma = float(
        np.sqrt(
            np.log(
                1.0 + default_cv**2
            )
        )
    )
    mu = float(
        np.log(mean)
        - 0.5 * sigma**2
    )
    return mu, sigma


def _logit(
    p: np.ndarray,
) -> np.ndarray:
    p = np.clip(
        p,
        1e-8,
        1 - 1e-8,
    )
    return np.log(
        p / (1 - p)
    )


def _sigmoid(
    x: np.ndarray,
) -> np.ndarray:
    x = np.clip(
        x,
        -30.0,
        30.0,
    )
    return 1.0 / (
        1.0 + np.exp(-x)
    )


def hill_efficiency(
    multiplier: float,
    half: float = 1.5,
    slope: float = 1.3,
) -> float:
    m = max(
        float(multiplier),
        0.0,
    )
    if m == 0.0:
        return 0.0

    def hill(x: float) -> float:
        return (
            x**slope
            / (
                half**slope
                + x**slope
            )
        )

    response_ratio = (
        hill(m) / hill(1.0)
    )
    return response_ratio / m


def sample_simulation_context(
    *,
    stats: dict,
    ctr_post: BetaPosterior,
    cvr_post: BetaPosterior,
    draws: int,
    horizon_days: int,
    rng: np.random.Generator,
    temporal_ctr_slope_mean: float = 0.0,
    temporal_ctr_slope_sd: float = 0.0,
    temporal_cvr_slope_mean: float = 0.0,
    temporal_cvr_slope_sd: float = 0.0,
    response_elasticity_mean: float = 0.75,
    response_elasticity_sd: float = 0.25,
    current_ctr_logit_shift: float = 0.0,
    current_cvr_logit_shift: float = 0.0,
    temporal_projection_days: float = 2.0,
    seasonal_ctr_shift_mean: float = 0.0,
    seasonal_ctr_shift_sd: float = 0.0,
    seasonal_cvr_shift_mean: float = 0.0,
    seasonal_cvr_shift_sd: float = 0.0,
) -> SimulationContext:
    daily = stats["daily"]
    default_cpm = (
        stats["cpm"]
        if (
            np.isfinite(stats["cpm"])
            and stats["cpm"] > 0
        )
        else 20.0
    )
    default_aov = (
        stats["aov"]
        if (
            np.isfinite(stats["aov"])
            and stats["aov"] > 0
        )
        else 100.0
    )

    cpm_mu, cpm_sigma = _lognormal_params(
        daily["cpm"].to_numpy(),
        default_cpm,
        0.20,
    )
    aov_mu, aov_sigma = _lognormal_params(
        daily["aov"].to_numpy(),
        default_aov,
        0.25,
    )

    cpm = rng.lognormal(
        cpm_mu,
        cpm_sigma,
        size=draws,
    )
    aov = rng.lognormal(
        aov_mu,
        aov_sigma,
        size=draws,
    )
    ctr = ctr_post.sample(
        rng,
        draws,
    )
    cvr = cvr_post.sample(
        rng,
        draws,
    )

    if current_ctr_logit_shift != 0.0:
        ctr = _sigmoid(
            _logit(ctr)
            + float(
                np.clip(
                    current_ctr_logit_shift,
                    -0.75,
                    0.75,
                )
            )
        )
    if current_cvr_logit_shift != 0.0:
        cvr = _sigmoid(
            _logit(cvr)
            + float(
                np.clip(
                    current_cvr_logit_shift,
                    -0.75,
                    0.75,
                )
            )
        )

    avg_future_day = min(
        (horizon_days + 1.0) / 2.0,
        max(float(temporal_projection_days), 0.0),
    )

    if (
        temporal_ctr_slope_mean != 0.0
        or temporal_ctr_slope_sd != 0.0
    ):
        ctr_slope = rng.normal(
            temporal_ctr_slope_mean,
            max(
                temporal_ctr_slope_sd,
                1e-9,
            ),
            size=draws,
        )
        ctr_shift = np.clip(
            ctr_slope
            * avg_future_day,
            -0.75,
            0.75,
        )
        ctr = _sigmoid(
            _logit(ctr)
            + ctr_shift
        )

    if (
        temporal_cvr_slope_mean != 0.0
        or temporal_cvr_slope_sd != 0.0
    ):
        cvr_slope = rng.normal(
            temporal_cvr_slope_mean,
            max(
                temporal_cvr_slope_sd,
                1e-9,
            ),
            size=draws,
        )
        cvr_shift = np.clip(
            cvr_slope
            * avg_future_day,
            -1.5,
            1.5,
        )
        cvr = _sigmoid(
            _logit(cvr)
            + cvr_shift
        )

    if (
        seasonal_ctr_shift_mean != 0.0
        or seasonal_ctr_shift_sd != 0.0
    ):
        ctr_weekly_shift = rng.normal(
            seasonal_ctr_shift_mean,
            max(
                seasonal_ctr_shift_sd,
                1e-9,
            ),
            size=draws,
        )
        ctr = _sigmoid(
            _logit(ctr)
            + np.clip(
                ctr_weekly_shift,
                -1.0,
                1.0,
            )
        )

    if (
        seasonal_cvr_shift_mean != 0.0
        or seasonal_cvr_shift_sd != 0.0
    ):
        cvr_weekly_shift = rng.normal(
            seasonal_cvr_shift_mean,
            max(
                seasonal_cvr_shift_sd,
                1e-9,
            ),
            size=draws,
        )
        cvr = _sigmoid(
            _logit(cvr)
            + np.clip(
                cvr_weekly_shift,
                -1.0,
                1.0,
            )
        )

    elasticity = rng.normal(
        response_elasticity_mean,
        max(
            response_elasticity_sd,
            1e-6,
        ),
        size=draws,
    )
    elasticity = np.clip(
        elasticity,
        0.05,
        1.50,
    )

    return SimulationContext(
        cpm=np.asarray(cpm, dtype=float),
        aov=np.asarray(aov, dtype=float),
        ctr=np.asarray(ctr, dtype=float),
        cvr=np.asarray(cvr, dtype=float),
        elasticity=np.asarray(
            elasticity,
            dtype=float,
        ),
    )


def simulate_action(
    *,
    stats: dict,
    ctr_post: BetaPosterior,
    cvr_post: BetaPosterior,
    multiplier: float,
    draws: int,
    horizon_days: int,
    target_roas: float,
    contribution_margin: float,
    rng: np.random.Generator,
    saturation_half: float = 1.5,
    saturation_slope: float = 1.3,
    temporal_ctr_slope_mean: float = 0.0,
    temporal_ctr_slope_sd: float = 0.0,
    temporal_cvr_slope_mean: float = 0.0,
    temporal_cvr_slope_sd: float = 0.0,
    response_elasticity_mean: float = 0.75,
    response_elasticity_sd: float = 0.25,
    response_confidence: float = 0.0,
    base_daily_spend: float | None = None,
    current_ctr_logit_shift: float = 0.0,
    current_cvr_logit_shift: float = 0.0,
    temporal_projection_days: float = 2.0,
    seasonal_ctr_shift_mean: float = 0.0,
    seasonal_ctr_shift_sd: float = 0.0,
    seasonal_cvr_shift_mean: float = 0.0,
    seasonal_cvr_shift_sd: float = 0.0,
    context: SimulationContext | None = None,
) -> dict:
    if base_daily_spend is None:
        base_daily_spend = (
            stats["spend"]
            / max(
                stats["days"],
                1,
            )
        )
    base_daily_spend = max(
        float(base_daily_spend),
        0.0,
    )
    spend = float(
        base_daily_spend
        * horizon_days
        * multiplier
    )
    contribution_margin = float(
        np.clip(
            contribution_margin,
            0.0,
            1.0,
        )
    )

    if context is None:
        context = sample_simulation_context(
            stats=stats,
            ctr_post=ctr_post,
            cvr_post=cvr_post,
            draws=draws,
            horizon_days=horizon_days,
            rng=rng,
            temporal_ctr_slope_mean=temporal_ctr_slope_mean,
            temporal_ctr_slope_sd=temporal_ctr_slope_sd,
            temporal_cvr_slope_mean=temporal_cvr_slope_mean,
            temporal_cvr_slope_sd=temporal_cvr_slope_sd,
            response_elasticity_mean=response_elasticity_mean,
            response_elasticity_sd=response_elasticity_sd,
            current_ctr_logit_shift=current_ctr_logit_shift,
            current_cvr_logit_shift=current_cvr_logit_shift,
            temporal_projection_days=temporal_projection_days,
            seasonal_ctr_shift_mean=seasonal_ctr_shift_mean,
            seasonal_ctr_shift_sd=seasonal_ctr_shift_sd,
            seasonal_cvr_shift_mean=seasonal_cvr_shift_mean,
            seasonal_cvr_shift_sd=seasonal_cvr_shift_sd,
        )
    elif context.draws != draws:
        raise ValueError(
            "SimulationContext possui número de draws incompatível."
        )

    if spend <= 0:
        revenue = np.zeros(
            draws,
            dtype=float,
        )
        profit = np.zeros(
            draws,
            dtype=float,
        )
        roas = np.zeros(
            draws,
            dtype=float,
        )
        decision_profit = np.zeros(
            draws,
            dtype=float,
        )
    else:
        impressions = np.maximum(
            np.round(
                (
                    spend
                    / np.maximum(
                        context.cpm,
                        1e-6,
                    )
                )
                * 1000.0
            ),
            0,
        ).astype(int)

        fixed_eff = max(
            hill_efficiency(
                multiplier,
                saturation_half,
                saturation_slope,
            ),
            1e-6,
        )

        response_confidence = float(
            np.clip(
                response_confidence,
                0.0,
                1.0,
            )
        )
        if response_confidence > 0:
            empirical_eff = np.power(
                max(
                    multiplier,
                    1e-6,
                ),
                context.elasticity - 1.0,
            )
            eff = np.exp(
                (
                    1.0
                    - response_confidence
                )
                * np.log(fixed_eff)
                + response_confidence
                * np.log(
                    np.clip(
                        empirical_eff,
                        1e-6,
                        10.0,
                    )
                )
            )
            eff = np.clip(
                eff,
                0.25,
                2.50,
            )
        else:
            eff = fixed_eff

        cvr_scaled = np.clip(
            context.cvr * eff,
            1e-9,
            1 - 1e-9,
        )

        conditional_clicks = (
            impressions
            * context.ctr
        )
        conditional_conversions = (
            conditional_clicks
            * cvr_scaled
        )
        conditional_revenue = (
            conditional_conversions
            * context.aov
        )
        decision_profit = (
            conditional_revenue
            * contribution_margin
            - spend
        )

        clicks = rng.binomial(
            impressions,
            context.ctr,
        )
        conversions = rng.binomial(
            clicks,
            cvr_scaled,
        )
        revenue = (
            conversions
            * context.aov
        )
        profit = (
            revenue
            * contribution_margin
            - spend
        )
        roas = revenue / spend

    q05, q10, q50, q95 = np.quantile(
        profit,
        [
            0.05,
            0.10,
            0.50,
            0.95,
        ],
    )
    tail = profit[
        profit <= q10
    ]
    cvar10 = (
        float(tail.mean())
        if len(tail)
        else float(q10)
    )
    revenue_q05, revenue_q50, revenue_q95 = (
        np.quantile(
            revenue,
            [
                0.05,
                0.50,
                0.95,
            ],
        )
    )
    roas_q05, roas_q50, roas_q95 = (
        np.quantile(
            roas,
            [
                0.05,
                0.50,
                0.95,
            ],
        )
    )

    return {
        "multiplier": float(multiplier),
        "expected_spend": spend,
        "expected_revenue": float(
            revenue.mean()
        ),
        "expected_profit": float(
            profit.mean()
        ),
        "expected_roas": float(
            roas.mean()
        ),
        "profit_p05": float(q05),
        "profit_p50": float(q50),
        "profit_p95": float(q95),
        "revenue_p05": float(
            revenue_q05
        ),
        "revenue_p50": float(
            revenue_q50
        ),
        "revenue_p95": float(
            revenue_q95
        ),
        "roas_p05": float(
            roas_q05
        ),
        "roas_p50": float(
            roas_q50
        ),
        "roas_p95": float(
            roas_q95
        ),
        "p_profit": float(
            np.mean(
                profit > 0
            )
        ),
        "p_roas_target": float(
            np.mean(
                roas
                >= target_roas
            )
        ),
        "var10_profit": float(q10),
        "cvar10_profit": cvar10,
        "_profit_draws": profit,
        "_decision_profit_draws": decision_profit,
    }
