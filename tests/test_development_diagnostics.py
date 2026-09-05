import pandas as pd

from quant_trafego.development_diagnostics import (
    build_development_diagnostics,
)
from quant_trafego.engine import EngineConfig
from quant_trafego.observability import RunTelemetry
from quant_trafego.quality import assess_data_quality


def test_development_diagnostics_summarizes_run_behavior():
    df = pd.DataFrame(
        [
            {
                "date": "2026-09-01",
                "campaign_id": "c1",
                "adset_id": "s1",
                "ad_id": "a1",
                "impressions": 1000,
                "clicks": 20,
                "conversions": 2,
                "spend": 50.0,
                "revenue": 200.0,
            }
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "level": "campaign",
                "entity_id": "c1",
                "action_multiplier": 1.2,
                "evidence_tier": "predictive",
                "policy_constrained": True,
                "response_confidence": 0.2,
                "decision_score": 0.7,
            }
        ]
    )
    telemetry = RunTelemetry()
    telemetry.emit(
        "inference",
        "stage_start",
        message="running",
        progress=0.2,
    )
    quality = assess_data_quality(df)

    diagnostics = build_development_diagnostics(
        full_df=df,
        operational_df=df,
        all_actions=actions,
        best_actions=actions,
        quality=quality,
        telemetry=telemetry,
        inference_mode="empirical_bayes",
        config=EngineConfig(
            contribution_margin=0.7,
            draws=100,
        ),
    )

    assert diagnostics["input"]["campaigns_active"] == 1
    assert diagnostics["decision_behavior"]["policy_constrained_count"] == 1
    assert diagnostics["decision_behavior"]["scale_selected_count"] == 1
    assert diagnostics["computation"]["all_action_rows"] == 1
