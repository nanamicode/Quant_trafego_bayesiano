from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import numpy as np
import pandas as pd

from .backtest import rolling_origin_backtest
from .engine import EngineConfig


_METRICS = (
    "profit_brier",
    "roas_target_brier",
    "profit_ece",
    "roas_ece",
    "profit_interval_score_90",
    "profit_mae",
)

# Floors prevent a baseline metric extremely close to zero from creating an
# arbitrarily large relative ratio and dominating model promotion.
_SCALE_FLOORS = {
    "profit_brier": 0.02,
    "roas_target_brier": 0.02,
    "profit_ece": 0.02,
    "roas_ece": 0.02,
    "profit_interval_score_90": 1.0,
    "profit_mae": 1.0,
}


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
    return float(
        np.average(
            values[valid],
            weights=weights[valid],
        )
    )


def _relative_improvement(
    metric: str,
    baseline: float,
    candidate: float,
) -> float:
    if not (
        np.isfinite(baseline)
        and np.isfinite(candidate)
    ):
        return float("nan")

    denominator = max(
        abs(float(baseline)),
        _SCALE_FLOORS.get(metric, 1e-9),
    )
    relative = (
        float(baseline) - float(candidate)
    ) / denominator
    return float(np.clip(relative, -2.0, 2.0))


def _promotion_from_summaries(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    minimum_improvement: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if baseline.empty or candidate.empty:
        return pd.DataFrame(), {
            "promote_state_space": False,
            "reason": "insufficient_backtest_data",
        }

    rows: list[dict[str, Any]] = []
    valid_improvements: list[float] = []

    for metric in _METRICS:
        b = _weighted_metric(
            baseline,
            metric,
        )
        c = _weighted_metric(
            candidate,
            metric,
        )
        relative = _relative_improvement(
            metric,
            b,
            c,
        )
        if np.isfinite(relative):
            valid_improvements.append(relative)

        rows.append(
            {
                "metric": metric,
                "derivative": b,
                "state_space": c,
                "relative_improvement": relative,
                "lower_is_better": True,
            }
        )

    if len(valid_improvements) < 4:
        return pd.DataFrame(rows), {
            "promote_state_space": False,
            "reason": "insufficient_valid_metrics",
            "valid_metric_count": len(valid_improvements),
        }

    improvements = np.asarray(
        valid_improvements,
        dtype=float,
    )
    robust_improvement = float(
        np.median(improvements)
    )
    improved_count = int(
        np.sum(improvements > 0)
    )
    minimum_metrics_improved = max(
        3,
        int(math.ceil(len(improvements) * 0.60)),
    )
    worst_relative_improvement = float(
        np.min(improvements)
    )

    baseline_cal = abs(
        _weighted_metric(
            baseline,
            "profit_event_calibration_gap",
        )
    )
    candidate_cal = abs(
        _weighted_metric(
            candidate,
            "profit_event_calibration_gap",
        )
    )
    baseline_ece = _weighted_metric(
        baseline,
        "profit_ece",
    )
    candidate_ece = _weighted_metric(
        candidate,
        "profit_ece",
    )
    baseline_cov_dev = abs(
        _weighted_metric(
            baseline,
            "profit_90_coverage",
        )
        - 0.90
    )
    candidate_cov_dev = abs(
        _weighted_metric(
            candidate,
            "profit_90_coverage",
        )
        - 0.90
    )

    no_calibration_regression = (
        candidate_cal
        <= baseline_cal + 0.03
    )
    no_ece_regression = (
        candidate_ece
        <= baseline_ece + 0.02
    )
    no_coverage_regression = (
        candidate_cov_dev
        <= baseline_cov_dev + 0.03
    )
    no_material_metric_regression = (
        worst_relative_improvement >= -0.15
    )
    enough_metrics_improved = (
        improved_count
        >= minimum_metrics_improved
    )
    enough_forecasts = int(
        candidate["n_forecasts"].sum()
    ) >= int(
        baseline["n_forecasts"].sum()
    )

    promote = bool(
        robust_improvement
        >= minimum_improvement
        and enough_metrics_improved
        and no_material_metric_regression
        and no_calibration_regression
        and no_ece_regression
        and no_coverage_regression
        and enough_forecasts
    )

    decision = {
        "promote_state_space": promote,
        "robust_median_relative_improvement": robust_improvement,
        "metrics_improved": improved_count,
        "minimum_metrics_improved": minimum_metrics_improved,
        "worst_relative_improvement": worst_relative_improvement,
        "baseline_abs_profit_calibration_gap": baseline_cal,
        "candidate_abs_profit_calibration_gap": candidate_cal,
        "baseline_profit_ece": baseline_ece,
        "candidate_profit_ece": candidate_ece,
        "baseline_profit_coverage_deviation": baseline_cov_dev,
        "candidate_profit_coverage_deviation": candidate_cov_dev,
        "minimum_improvement_required": minimum_improvement,
        "no_material_metric_regression": no_material_metric_regression,
        "reason": (
            "promotion_gates_passed"
            if promote
            else "promotion_gates_not_passed"
        ),
    }
    return pd.DataFrame(rows), decision


def compare_temporal_models(
    df: pd.DataFrame,
    *,
    config: EngineConfig | None = None,
    min_train_days: int = 21,
    horizon_days: int | None = None,
    step_days: int = 7,
    minimum_improvement: float = 0.02,
    levels: tuple[str, ...] = (
        "account",
        "campaign",
    ),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = config or EngineConfig()

    summaries: dict[
        str,
        pd.DataFrame,
    ] = {}

    for model in (
        "derivative",
        "state_space",
    ):
        cfg = replace(
            base,
            temporal_model=model,
        )
        _, summary = rolling_origin_backtest(
            df,
            config=cfg,
            min_train_days=min_train_days,
            horizon_days=horizon_days,
            step_days=step_days,
            levels=levels,
        )
        summaries[model] = summary

    return _promotion_from_summaries(
        summaries["derivative"],
        summaries["state_space"],
        minimum_improvement=minimum_improvement,
    )
