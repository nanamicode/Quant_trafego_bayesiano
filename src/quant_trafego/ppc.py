from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PPCSummary:
    n_observations: int
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
    has_date = "date" in df.columns
    if has_date:
        daily = (
            df.groupby(
                ["date", "campaign_id", "adset_id", "ad_id"],
                as_index=False,
            )
            .agg(
                impressions=("impressions", "sum"),
                clicks=("clicks", "sum"),
                conversions=("conversions", "sum"),
            )
            .copy()
        )
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values(
            ["date", "campaign_id", "adset_id", "ad_id"]
        ).reset_index(drop=True)
    else:
        daily = (
            df.groupby(
                ["campaign_id", "adset_id", "ad_id"],
                as_index=False,
            )
            .agg(
                impressions=("impressions", "sum"),
                clicks=("clicks", "sum"),
                conversions=("conversions", "sum"),
            )
            .copy()
        )

    rng = np.random.default_rng(seed)
    records: list[dict] = []

    try:
        ctr_entity = idata.posterior["ctr_p_entity"]
        cvr_entity = idata.posterior["cvr_p_entity"]
        entity_size = ctr_entity.sizes.get("entity", -1)
        use_entity = has_date and len(daily) == entity_size
    except Exception:
        ctr_entity = None
        cvr_entity = None
        use_entity = False

    if use_entity:
        ctr_source = ctr_entity
        cvr_source = cvr_entity
        iterator = [
            (i, row)
            for i, row in enumerate(daily.itertuples(index=False))
        ]
    else:
        # Backward-compatible fallback for older/static idata objects.
        grouped = (
            daily.groupby(
                ["campaign_id", "adset_id", "ad_id"],
                as_index=False,
            )
            .agg(
                impressions=("impressions", "sum"),
                clicks=("clicks", "sum"),
                conversions=("conversions", "sum"),
            )
        )
        by_ad = {
            str(row.ad_id): row
            for row in grouped.itertuples(index=False)
        }
        ctr_source = idata.posterior["ctr_p_ad"]
        cvr_source = idata.posterior["cvr_p_ad"]
        iterator = []
        for i, entity_id in enumerate(mapping["ad"]):
            row = by_ad.get(str(entity_id))
            if row is not None:
                iterator.append((i, row))

    for i, row in iterator:
        if use_entity:
            ctr_samples = np.asarray(
                ctr_source.isel(entity=i).values
            ).ravel()
            cvr_samples = np.asarray(
                cvr_source.isel(entity=i).values
            ).ravel()
            date_value = str(pd.Timestamp(row.date))
            ad_value = str(row.ad_id)
        else:
            ctr_samples = np.asarray(
                ctr_source.isel(ad=i).values
            ).ravel()
            cvr_samples = np.asarray(
                cvr_source.isel(ad=i).values
            ).ravel()
            date_value = None
            ad_value = str(row.ad_id)

        n_available = min(
            len(ctr_samples),
            len(cvr_samples),
        )
        if n_available == 0:
            continue

        idx = rng.choice(
            n_available,
            size=draws,
            replace=n_available < draws,
        )
        p_ctr = np.clip(
            ctr_samples[idx],
            1e-9,
            1 - 1e-9,
        )
        p_cvr = np.clip(
            cvr_samples[idx],
            1e-9,
            1 - 1e-9,
        )

        sim_clicks = rng.binomial(
            int(row.impressions),
            p_ctr,
        )
        sim_conversions = rng.binomial(
            sim_clicks,
            p_cvr,
        )

        click_q05, click_q50, click_q95 = np.quantile(
            sim_clicks,
            [0.05, 0.50, 0.95],
        )
        conv_q05, conv_q50, conv_q95 = np.quantile(
            sim_conversions,
            [0.05, 0.50, 0.95],
        )

        click_sd = max(
            float(np.std(sim_clicks, ddof=1)),
            1.0,
        )
        conv_sd = max(
            float(np.std(sim_conversions, ddof=1)),
            1.0,
        )
        click_tail = _two_sided_tail(
            sim_clicks,
            float(row.clicks),
        )
        conv_tail = _two_sided_tail(
            sim_conversions,
            float(row.conversions),
        )

        records.append(
            {
                "date": date_value,
                "ad_id": ad_value,
                "impressions": int(row.impressions),
                "observed_clicks": int(row.clicks),
                "pred_clicks_p05": float(click_q05),
                "pred_clicks_p50": float(click_q50),
                "pred_clicks_p95": float(click_q95),
                "click_90_covered": float(
                    click_q05
                    <= row.clicks
                    <= click_q95
                ),
                "click_two_sided_tail_p": click_tail,
                "click_standardized_residual": float(
                    (
                        row.clicks
                        - np.mean(sim_clicks)
                    )
                    / click_sd
                ),
                "observed_conversions": int(row.conversions),
                "pred_conversions_p05": float(conv_q05),
                "pred_conversions_p50": float(conv_q50),
                "pred_conversions_p95": float(conv_q95),
                "conversion_90_covered": float(
                    conv_q05
                    <= row.conversions
                    <= conv_q95
                ),
                "conversion_two_sided_tail_p": conv_tail,
                "conversion_standardized_residual": float(
                    (
                        row.conversions
                        - np.mean(sim_conversions)
                    )
                    / conv_sd
                ),
            }
        )

    detail = pd.DataFrame(records)
    if detail.empty:
        return detail, PPCSummary(
            n_observations=0,
            n_ads=0,
            click_90_coverage=float("nan"),
            conversion_90_coverage=float("nan"),
            click_extreme_fraction=float("nan"),
            conversion_extreme_fraction=float("nan"),
            mean_abs_click_z=float("nan"),
            mean_abs_conversion_z=float("nan"),
            status="insufficient",
        )

    click_cov = float(
        detail["click_90_covered"].mean()
    )
    conv_cov = float(
        detail["conversion_90_covered"].mean()
    )
    click_extreme = float(
        (
            detail["click_two_sided_tail_p"]
            < 0.05
        ).mean()
    )
    conv_extreme = float(
        (
            detail["conversion_two_sided_tail_p"]
            < 0.05
        ).mean()
    )
    click_z = float(
        detail[
            "click_standardized_residual"
        ].abs().mean()
    )
    conv_z = float(
        detail[
            "conversion_standardized_residual"
        ].abs().mean()
    )

    warning = (
        click_cov < 0.80
        or conv_cov < 0.80
        or click_extreme > 0.15
        or conv_extreme > 0.15
    )
    return detail, PPCSummary(
        n_observations=int(len(detail)),
        n_ads=int(detail["ad_id"].nunique()),
        click_90_coverage=click_cov,
        conversion_90_coverage=conv_cov,
        click_extreme_fraction=click_extreme,
        conversion_extreme_fraction=conv_extreme,
        mean_abs_click_z=click_z,
        mean_abs_conversion_z=conv_z,
        status="warning" if warning else "pass",
    )
