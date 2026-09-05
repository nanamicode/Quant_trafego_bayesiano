from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .backtest import rolling_origin_backtest
from .engine import EngineConfig


def _weighted_metric(
    summary: pd.DataFrame,
    column: str,
) -> float:
    if summary.empty or column not in summary:
        return float("nan")
    weights = summary["n_forecasts"].to_numpy(dtype=float)
    values = summary[column].to_numpy(dtype=float)
    valid = np.isfinite(values) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def compare_temporal_models(
    df: pd.DataFrame,
    *,
    config: EngineConfig | None = None,
    min_train_days: int = 21,
    horizon_days: int | None = None,
    step_days: int = 7,
    minimum_improvement: float = 0.02,
    levels: tuple[str, ...] = ("account", "campaign"),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = config or EngineConfig()

    summaries: dict[str, pd.DataFrame] = {}
    details: dict[str, pd.DataFrame] = {}
    for model in ("derivative", "state_space"):
        cfg = replace(base, temporal_model=model)
        detail, summary = rolling_origin_backtest(
            df,
            config=cfg,
            min_train_days=min_train_days,
            horizon_days=horizon_days,
            step_days=step_days,
            levels=levels,
        )
        details[model] = detail
        summaries[model] = summary

    baseline = summaries["derivative"]
    candidate = summaries["state_space"]
    if baseline.empty or candidate.empty:
        decision = {
            "promote_state_space": False,
            "reason": "insufficient_backtest_data",
        }
        return pd.DataFrame(), decision

    metrics = [
        "profit_brier",
        "roas_target_brier",
        "profit_ece",
        "roas_ece",
        "profit_interval_score_90",
        "profit_mae",
    ]
    rows = []
    improvements = []
    for metric in metrics:
        b = _weighted_metric(baseline, metric)
        c = _weighted_metric(candidate, metric)
        relative = (b - c) / max(abs(b), 1e-9)
        improvements.append(relative)
        rows.append(
            {
                "metric": metric,
                "derivative": b,
                "state_space": c,
                "relative_improvement": relative,
                "lower_is_better": True,
            }
        )

    baseline_cal = abs(
        _weighted_metric(baseline, "profit_event_calibration_gap")
    )
    candidate_cal = abs(
        _weighted_metric(candidate, "profit_event_calibration_gap")
    )
    baseline_cov_dev = abs(
        _weighted_metric(baseline, "profit_90_coverage") - 0.90
    )
    candidate_cov_dev = abs(
        _weighted_metric(candidate, "profit_90_coverage") - 0.90
    )

    composite_improvement = float(np.nanmean(improvements))
    no_calibration_regression = candidate_cal <= baseline_cal + 0.03
    no_coverage_regression = candidate_cov_dev <= baseline_cov_dev + 0.03
    enough_forecasts = int(candidate["n_forecasts"].sum()) >= int(
        baseline["n_forecasts"].sum()
    )

    promote = bool(
        composite_improvement >= minimum_improvement
        and no_calibration_regression
        and no_coverage_regression
        and enough_forecasts
    )

    decision = {
        "promote_state_space": promote,
        "composite_relative_improvement": composite_improvement,
        "baseline_abs_profit_calibration_gap": baseline_cal,
        "candidate_abs_profit_calibration_gap": candidate_cal,
        "baseline_profit_coverage_deviation": baseline_cov_dev,
        "candidate_profit_coverage_deviation": candidate_cov_dev,
        "minimum_improvement_required": minimum_improvement,
        "reason": (
            "promotion_gates_passed"
            if promote
            else "promotion_gates_not_passed"
        ),
    }
    return pd.DataFrame(rows), decision
