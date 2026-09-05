from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PPCSummary:
    n_ads: int
    click_90_coverage: float
    conversion_90_coverage: float
    click_extreme_fraction: float
    conversion_extreme_fraction: float
    mean_abs_click_z: float
    mean_abs_conversion_z: float
    status: str


def _two_sided_tail(simulated: np.ndarray, observed: float) -> float:
    lo = float(np.mean(simulated <= observed))
    hi = float(np.mean(simulated >= observed))
    return float(np.clip(2.0 * min(lo, hi), 0.0, 1.0))


def posterior_predictive_checks(
    idata,
    df: pd.DataFrame,
    mapping: dict,
    *,
    draws: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, PPCSummary]:
    grouped = (
        df.groupby(["campaign_id", "adset_id", "ad_id"], as_index=False)
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
        )
        .copy()
    )

    by_ad = {
        str(row.ad_id): row
        for row in grouped.itertuples(index=False)
    }
    ctr = idata.posterior["ctr_p_ad"]
    cvr = idata.posterior["cvr_p_ad"]
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for i, entity_id in enumerate(mapping["ad"]):
        entity_id = str(entity_id)
        if entity_id not in by_ad:
            continue

        row = by_ad[entity_id]
        ctr_samples = np.asarray(ctr.isel(ad=i).values).ravel()
        cvr_samples = np.asarray(cvr.isel(ad=i).values).ravel()
        n_available = min(len(ctr_samples), len(cvr_samples))
        if n_available == 0:
            continue

        if n_available > draws:
            idx = rng.choice(n_available, size=draws, replace=False)
        else:
            idx = rng.choice(n_available, size=draws, replace=True)

        p_ctr = np.clip(ctr_samples[idx], 1e-9, 1 - 1e-9)
        p_cvr = np.clip(cvr_samples[idx], 1e-9, 1 - 1e-9)

        sim_clicks = rng.binomial(int(row.impressions), p_ctr)
        sim_conversions = rng.binomial(sim_clicks, p_cvr)

        click_q05, click_q50, click_q95 = np.quantile(
            sim_clicks, [0.05, 0.50, 0.95]
        )
        conv_q05, conv_q50, conv_q95 = np.quantile(
            sim_conversions, [0.05, 0.50, 0.95]
        )

        click_sd = max(float(np.std(sim_clicks, ddof=1)), 1.0)
        conv_sd = max(float(np.std(sim_conversions, ddof=1)), 1.0)
        click_tail = _two_sided_tail(sim_clicks, float(row.clicks))
        conv_tail = _two_sided_tail(
            sim_conversions, float(row.conversions)
        )

        records.append(
            {
                "ad_id": entity_id,
                "impressions": int(row.impressions),
                "observed_clicks": int(row.clicks),
                "pred_clicks_p05": float(click_q05),
                "pred_clicks_p50": float(click_q50),
                "pred_clicks_p95": float(click_q95),
                "click_90_covered": float(
                    click_q05 <= row.clicks <= click_q95
                ),
                "click_two_sided_tail_p": click_tail,
                "click_standardized_residual": float(
                    (row.clicks - np.mean(sim_clicks)) / click_sd
                ),
                "observed_conversions": int(row.conversions),
                "pred_conversions_p05": float(conv_q05),
                "pred_conversions_p50": float(conv_q50),
                "pred_conversions_p95": float(conv_q95),
                "conversion_90_covered": float(
                    conv_q05 <= row.conversions <= conv_q95
                ),
                "conversion_two_sided_tail_p": conv_tail,
                "conversion_standardized_residual": float(
                    (row.conversions - np.mean(sim_conversions)) / conv_sd
                ),
            }
        )

    detail = pd.DataFrame(records)
    if detail.empty:
        return detail, PPCSummary(
            n_ads=0,
            click_90_coverage=float("nan"),
            conversion_90_coverage=float("nan"),
            click_extreme_fraction=float("nan"),
            conversion_extreme_fraction=float("nan"),
            mean_abs_click_z=float("nan"),
            mean_abs_conversion_z=float("nan"),
            status="insufficient",
        )

    click_cov = float(detail["click_90_covered"].mean())
    conv_cov = float(detail["conversion_90_covered"].mean())
    click_extreme = float(
        (detail["click_two_sided_tail_p"] < 0.05).mean()
    )
    conv_extreme = float(
        (detail["conversion_two_sided_tail_p"] < 0.05).mean()
    )
    click_z = float(
        detail["click_standardized_residual"].abs().mean()
    )
    conv_z = float(
        detail["conversion_standardized_residual"].abs().mean()
    )

    # These are operational warning gates, not universal statistical laws.
    warning = (
        click_cov < 0.80
        or conv_cov < 0.80
        or click_extreme > 0.15
        or conv_extreme > 0.15
    )

    summary = PPCSummary(
        n_ads=int(len(detail)),
        click_90_coverage=click_cov,
        conversion_90_coverage=conv_cov,
        click_extreme_fraction=click_extreme,
        conversion_extreme_fraction=conv_extreme,
        mean_abs_click_z=click_z,
        mean_abs_conversion_z=conv_z,
        status="warning" if warning else "pass",
    )
    return detail, summary
