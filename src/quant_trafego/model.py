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
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self) -> float:
        return self.alpha + self.beta

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.beta(self.alpha, self.beta, size=n)


def beta_from_mean(mean: float, strength: float) -> BetaPosterior:
    mean = float(np.clip(mean, 1e-9, 1 - 1e-9))
    strength = max(float(strength), 2.0)
    return BetaPosterior(
        alpha=1.0 + mean * strength,
        beta=1.0 + (1.0 - mean) * strength,
    )


def update_beta(prior: BetaPosterior, successes: float, trials: float) -> BetaPosterior:
    s = max(float(successes), 0.0)
    n = max(float(trials), s)
    return BetaPosterior(prior.alpha + s, prior.beta + n - s)


def shrink_to(parent: BetaPosterior, strength: float) -> BetaPosterior:
    return beta_from_mean(parent.mean, strength)


def aggregate(df: pd.DataFrame) -> dict:
    impressions = float(df["impressions"].sum())
    clicks = float(df["clicks"].sum())
    conversions = float(df["conversions"].sum())
    spend = float(df["spend"].sum())
    revenue = float(df["revenue"].sum())
    days = max(int(df["date"].nunique()), 1)

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
        )
    )

    daily["cpm"] = np.where(
        daily["impressions"] > 0,
        daily["spend"] / daily["impressions"] * 1000.0,
        np.nan,
    )
    daily["aov"] = np.where(
        daily["conversions"] > 0,
        daily["revenue"] / daily["conversions"],
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
        "ctr": clicks / impressions if impressions else 0.0,
        "cvr": conversions / clicks if clicks else 0.0,
        "cpc": spend / clicks if clicks else np.nan,
        "cpa": spend / conversions if conversions else np.nan,
        "roas": revenue / spend if spend else 0.0,
        "cpm": spend / impressions * 1000.0 if impressions else np.nan,
        "aov": revenue / conversions if conversions else np.nan,
    }


def _lognormal_params(values: np.ndarray, default_mean: float, default_cv: float):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]

    if len(values) >= 2:
        logs = np.log(values)
        return float(logs.mean()), float(max(logs.std(ddof=1), 0.05))

    mean = max(float(default_mean), 1e-6)
    sigma = float(np.sqrt(np.log(1.0 + default_cv ** 2)))
    mu = float(np.log(mean) - 0.5 * sigma ** 2)
    return mu, sigma


def hill_efficiency(multiplier: float, half: float = 1.5, slope: float = 1.3) -> float:
    m = max(float(multiplier), 0.0)
    if m == 0.0:
        return 0.0

    def hill(x: float) -> float:
        return (x ** slope) / (half ** slope + x ** slope)

    response_ratio = hill(m) / hill(1.0)
    return response_ratio / m


def simulate_action(
    *,
    stats: dict,
    ctr_post: BetaPosterior,
    cvr_post: BetaPosterior,
    multiplier: float,
    draws: int,
    horizon_days: int,
    target_roas: float,
    rng: np.random.Generator,
    saturation_half: float = 1.5,
    saturation_slope: float = 1.3,
) -> dict:
    base_daily_spend = stats["spend"] / max(stats["days"], 1)
    spend = float(base_daily_spend * horizon_days * multiplier)

    if spend <= 0:
        revenue = np.zeros(draws)
        profit = np.zeros(draws)
        roas = np.zeros(draws)
    else:
        daily = stats["daily"]
        default_cpm = stats["cpm"] if np.isfinite(stats["cpm"]) and stats["cpm"] > 0 else 20.0
        default_aov = stats["aov"] if np.isfinite(stats["aov"]) and stats["aov"] > 0 else 100.0

        cpm_mu, cpm_sigma = _lognormal_params(daily["cpm"].to_numpy(), default_cpm, 0.20)
        aov_mu, aov_sigma = _lognormal_params(daily["aov"].to_numpy(), default_aov, 0.25)

        cpm = rng.lognormal(cpm_mu, cpm_sigma, size=draws)
        aov = rng.lognormal(aov_mu, aov_sigma, size=draws)
        impressions = np.maximum(np.round((spend / np.maximum(cpm, 1e-6)) * 1000), 0).astype(int)

        ctr = ctr_post.sample(rng, draws)
        cvr = cvr_post.sample(rng, draws)

        eff = hill_efficiency(multiplier, saturation_half, saturation_slope)
        cvr_scaled = np.clip(cvr * eff, 1e-9, 1 - 1e-9)

        clicks = rng.binomial(impressions, ctr)
        conversions = rng.binomial(clicks, cvr_scaled)
        revenue = conversions * aov
        profit = revenue - spend
        roas = revenue / spend

    q10 = float(np.quantile(profit, 0.10))
    tail = profit[profit <= q10]
    cvar10 = float(tail.mean()) if len(tail) else q10

    return {
        "multiplier": float(multiplier),
        "expected_spend": spend,
        "expected_revenue": float(revenue.mean()),
        "expected_profit": float(profit.mean()),
        "expected_roas": float(roas.mean()),
        "p_profit": float(np.mean(profit > 0)),
        "p_roas_target": float(np.mean(roas >= target_roas)),
        "var10_profit": q10,
        "cvar10_profit": cvar10,
        "_profit_draws": profit,
    }
