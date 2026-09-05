from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .observability import RunTelemetry


def _counts(series: pd.Series | None) -> dict[str, int]:
    if series is None:
        return {}
    return {
        str(key): int(value)
        for key, value in series.fillna("NA").astype(str).value_counts().items()
    }


def build_development_diagnostics(
    *,
    full_df: pd.DataFrame,
    operational_df: pd.DataFrame,
    all_actions: pd.DataFrame,
    best_actions: pd.DataFrame,
    quality,
    telemetry: RunTelemetry,
    inference_mode: str,
    config,
    model_decision: dict[str, Any] | None = None,
    allocation_summary: dict[str, Any] | None = None,
    adset_allocation_summary: dict[str, Any] | None = None,
    diagnostics=None,
    ppc_summary=None,
    deep_decision_source: str | None = None,
    deep_guardrail: str | None = None,
) -> dict[str, Any]:
    best = best_actions.copy()
    actions = all_actions.copy()

    scale = (
        best["action_multiplier"] > 1.0
        if "action_multiplier" in best
        else pd.Series(False, index=best.index)
    )
    constrained = (
        best["policy_constrained"].fillna(False).astype(bool)
        if "policy_constrained" in best
        else pd.Series(False, index=best.index)
    )

    response_confidence = (
        pd.to_numeric(
            best.get(
                "response_confidence",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )
    decision_score = (
        pd.to_numeric(
            best.get(
                "decision_score",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    stage_summary = telemetry.stage_summary()
    slowest = []
    if not stage_summary.empty:
        slowest = (
            stage_summary.sort_values(
                "duration_seconds",
                ascending=False,
            )
            .head(5)
            .to_dict(orient="records")
        )

    total_elapsed = max(
        telemetry.elapsed_seconds,
        1e-9,
    )

    return {
        "purpose": (
            "Development-only diagnostics. Not an additional decision signal."
        ),
        "input": {
            "rows_full_history": int(len(full_df)),
            "rows_operational_universe": int(len(operational_df)),
            "days_full_history": int(full_df["date"].nunique()),
            "campaigns_full_history": int(full_df["campaign_id"].nunique()),
            "adsets_full_history": int(full_df["adset_id"].nunique()),
            "ads_full_history": int(full_df["ad_id"].nunique()),
            "campaigns_active": int(operational_df["campaign_id"].nunique()),
            "adsets_active": int(operational_df["adset_id"].nunique()),
            "ads_active": int(operational_df["ad_id"].nunique()),
            "quality_score": float(quality.score),
            "quality_warnings": list(quality.warnings),
        },
        "configuration": {
            "inference_mode": inference_mode,
            "draws": int(config.draws),
            "horizon_days": int(config.horizon_days),
            "target_roas": float(config.target_roas),
            "contribution_margin": float(config.contribution_margin),
            "risk_aversion": float(config.risk_aversion),
            "temporal_model": str(config.temporal_model),
            "weekly_seasonality": bool(config.use_weekly_seasonality),
            "actions": [
                float(x)
                for x in config.actions
            ],
        },
        "computation": {
            "elapsed_seconds_so_far": float(total_elapsed),
            "all_action_rows": int(len(actions)),
            "best_action_rows": int(len(best)),
            "action_rows_per_second": float(
                len(actions) / total_elapsed
            ),
            "telemetry_event_count": int(len(telemetry.events)),
            "slowest_stages_so_far": slowest,
            "bottleneck_stage_so_far": (
                telemetry.developer_snapshot().get(
                    "bottleneck_stage_so_far"
                )
            ),
        },
        "decision_behavior": {
            "selected_actions": _counts(
                best.get("action_multiplier")
            ),
            "evidence_tiers": _counts(
                best.get("evidence_tier")
            ),
            "policy_constrained_count": int(constrained.sum()),
            "policy_constrained_fraction": float(
                constrained.mean()
                if len(constrained)
                else 0.0
            ),
            "scale_selected_count": int(scale.sum()),
            "median_response_confidence": (
                float(np.median(response_confidence))
                if len(response_confidence)
                else None
            ),
            "median_decision_score": (
                float(np.median(decision_score))
                if len(decision_score)
                else None
            ),
        },
        "validation": {
            "temporal_model_decision": model_decision,
            "deep_decision_source": deep_decision_source,
            "deep_guardrail": deep_guardrail,
            "mcmc_diagnostics": (
                diagnostics.__dict__
                if diagnostics is not None
                else None
            ),
            "ppc_summary": (
                ppc_summary.__dict__
                if ppc_summary is not None
                else None
            ),
        },
        "allocation": {
            "campaign": allocation_summary,
            "adset": adset_allocation_summary,
        },
    }
