from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .calibration import expected_calibration_error
from .engine import BayesTrafficEngine, EngineConfig


_LEVEL_GROUPS = {
    "account": [],
    "campaign": ["campaign_id"],
    "adset": ["adset_id"],
    "ad": ["ad_id"],
}


def _interval_score(
    observed: float,
    lower: float,
    upper: float,
    alpha: float = 0.10,
) -> float:
    width = upper - lower
    penalty = 0.0
    if observed < lower:
        penalty += (2.0 / alpha) * (lower - observed)
    elif observed > upper:
        penalty += (2.0 / alpha) * (observed - upper)
    return float(width + penalty)


def _future_entities(df: pd.DataFrame, level: str):
    keys = _LEVEL_GROUPS[level]
    if not keys:
        yield "ALL", df
        return

    for key, group in df.groupby(keys[0], sort=False):
        yield str(key), group


def rolling_origin_backtest(
    df: pd.DataFrame,
    *,
    config: EngineConfig | None = None,
    min_train_days: int = 21,
    horizon_days: int | None = None,
    step_days: int = 7,
    levels: tuple[str, ...] = ("account", "campaign", "adset", "ad"),
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling-origin predictive validation under the approximately observed
    future spend level.

    This validates probabilistic forecasting under the observed policy. It
    cannot identify the counterfactual result of actions that were not taken,
    so it must not be interpreted as causal validation of recommendations.
    """
    base_config = config or EngineConfig()
    engine = BayesTrafficEngine(base_config)
    clean = engine.validate(df).sort_values("date").reset_index(drop=True)

    dates = pd.Index(sorted(pd.to_datetime(clean["date"]).unique()))
    horizon = int(horizon_days or base_config.horizon_days)

    if len(dates) < min_train_days + horizon:
        return pd.DataFrame(), pd.DataFrame()

    records: list[dict] = []

    last_origin_index = len(dates) - horizon - 1
    origin_indices = list(
        range(
            min_train_days - 1,
            last_origin_index + 1,
            step_days,
        )
    )
    total_origins = len(origin_indices)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "backtest",
                "event": "start",
                "completed": 0,
                "total": total_origins,
                "progress": 0.0,
            }
        )

    for fold_number, origin_index in enumerate(origin_indices, start=1):
        train_dates = dates[: origin_index + 1]
        future_dates = dates[origin_index + 1 : origin_index + 1 + horizon]
        if len(future_dates) < horizon:
            continue

        train = clean[clean["date"].isin(train_dates)].copy()
        future = clean[clean["date"].isin(future_dates)].copy()
        if train.empty or future.empty:
            continue

        cfg = replace(
            base_config,
            horizon_days=horizon,
            seed=base_config.seed + origin_index,
        )
        fold_engine = BayesTrafficEngine(cfg)
        origin_date = pd.Timestamp(train_dates[-1])
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "backtest",
                    "event": "fold_start",
                    "fold": fold_number,
                    "completed": fold_number - 1,
                    "total": total_origins,
                    "progress": (
                        (fold_number - 1) / max(total_origins, 1)
                    ),
                    "origin_date": origin_date,
                }
            )
        all_actions, _ = fold_engine.run(
            train,
            evaluation_levels=set(levels),
        )

        for level in levels:
            level_actions = all_actions[all_actions["level"] == level]
            if level_actions.empty:
                continue

            for entity_id, observed_df in _future_entities(future, level):
                candidates = level_actions[
                    level_actions["entity_id"].astype(str) == str(entity_id)
                ].copy()
                if candidates.empty:
                    continue

                actual_spend = float(observed_df["spend"].sum())
                actual_revenue = float(observed_df["revenue"].sum())
                actual_profit = actual_revenue * cfg.contribution_margin - actual_spend
                actual_roas = actual_revenue / actual_spend if actual_spend > 0 else 0.0

                hold = candidates[candidates["action_multiplier"] == 1.0]
                if hold.empty:
                    continue
                hold_spend = float(hold.iloc[0]["expected_spend"])

                if actual_spend <= 0:
                    multipliers = candidates["action_multiplier"].to_numpy(dtype=float)
                    chosen = candidates.iloc[int(np.argmin(np.abs(multipliers)))]
                    actual_multiplier = 0.0
                else:
                    actual_multiplier = actual_spend / hold_spend if hold_spend > 0 else 1.0
                    positive = candidates[candidates["action_multiplier"] > 0].copy()
                    distances = np.abs(
                        np.log(positive["action_multiplier"].to_numpy(dtype=float))
                        - np.log(max(actual_multiplier, 1e-9))
                    )
                    chosen = positive.iloc[int(np.argmin(distances))]

                profit_event = float(actual_profit > 0)
                roas_event = float(actual_roas >= cfg.target_roas)
                p_profit = float(chosen["p_profit"])
                p_roas = float(chosen["p_roas_target"])
                lower = float(chosen["profit_p05"])
                upper = float(chosen["profit_p95"])

                records.append(
                    {
                        "origin_date": origin_date,
                        "forecast_end_date": pd.Timestamp(future_dates[-1]),
                        "level": level,
                        "entity_id": str(entity_id),
                        "actual_multiplier": float(actual_multiplier),
                        "matched_action_multiplier": float(chosen["action_multiplier"]),
                        "actual_spend": actual_spend,
                        "actual_revenue": actual_revenue,
                        "actual_profit": actual_profit,
                        "actual_roas": actual_roas,
                        "predicted_profit_mean": float(chosen["expected_profit"]),
                        "predicted_profit_p05": lower,
                        "predicted_profit_p50": float(chosen["profit_p50"]),
                        "predicted_profit_p95": upper,
                        "predicted_p_profit": p_profit,
                        "predicted_p_roas_target": p_roas,
                        "profit_brier": (p_profit - profit_event) ** 2,
                        "roas_target_brier": (p_roas - roas_event) ** 2,
                        "profit_90_covered": float(lower <= actual_profit <= upper),
                        "profit_90_width": upper - lower,
                        "profit_interval_score_90": _interval_score(
                            actual_profit, lower, upper, alpha=0.10
                        ),
                        "profit_error": float(chosen["expected_profit"] - actual_profit),
                        "absolute_profit_error": abs(
                            float(chosen["expected_profit"] - actual_profit)
                        ),
                        "posterior_source": chosen["posterior_source"],
                    }
                )

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "backtest",
                    "event": "fold_done",
                    "fold": fold_number,
                    "completed": fold_number,
                    "total": total_origins,
                    "progress": (
                        fold_number / max(total_origins, 1)
                    ),
                    "origin_date": origin_date,
                }
            )

    detail = pd.DataFrame(records)
    if detail.empty:
        return detail, pd.DataFrame()

    summaries = []
    for level, group in detail.groupby("level", sort=False):
        summaries.append(
            {
                "level": level,
                "n_forecasts": int(len(group)),
                "n_origins": int(group["origin_date"].nunique()),
                "profit_brier": float(group["profit_brier"].mean()),
                "roas_target_brier": float(group["roas_target_brier"].mean()),
                "profit_event_calibration_gap": float(
                    group["predicted_p_profit"].mean()
                    - (group["actual_profit"] > 0).mean()
                ),
                "profit_ece": expected_calibration_error(
                    group["predicted_p_profit"].to_numpy(),
                    (group["actual_profit"] > 0).to_numpy(dtype=float),
                ),
                "roas_event_calibration_gap": float(
                    group["predicted_p_roas_target"].mean()
                    - (group["actual_roas"] >= base_config.target_roas).mean()
                ),
                "roas_ece": expected_calibration_error(
                    group["predicted_p_roas_target"].to_numpy(),
                    (
                        group["actual_roas"] >= base_config.target_roas
                    ).to_numpy(dtype=float),
                ),
                "profit_90_coverage": float(group["profit_90_covered"].mean()),
                "profit_90_mean_width": float(group["profit_90_width"].mean()),
                "profit_interval_score_90": float(
                    group["profit_interval_score_90"].mean()
                ),
                "profit_bias": float(group["profit_error"].mean()),
                "profit_mae": float(group["absolute_profit_error"].mean()),
            }
        )
    return detail, pd.DataFrame(summaries)