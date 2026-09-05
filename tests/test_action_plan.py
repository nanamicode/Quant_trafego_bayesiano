import pandas as pd

from quant_trafego.action_plan import build_operational_action_plan


def _best():
    return pd.DataFrame(
        [
            {
                "level": "campaign",
                "entity_id": "c1",
                "campaign_id": "c1",
                "campaign_name": "Campanha 1",
                "action_multiplier": 1.5,
                "historical_days": 14,
                "historical_spend": 1400.0,
                "expected_profit": 900.0,
                "expected_revenue": 3000.0,
                "expected_incremental_profit_vs_hold": 250.0,
                "expected_incremental_revenue_vs_hold": 500.0,
                "p_profit": 0.92,
                "p_roas_target": 0.80,
                "p_beats_hold": 0.75,
                "p_incremental_profit_positive": 0.82,
                "p_action_optimal": 0.60,
                "cvar10_profit": 200.0,
                "expected_regret": 30.0,
                "response_confidence": 0.30,
                "evidence_tier": "observational_intervention",
                "policy_eligible": True,
                "policy_constrained": False,
                "data_quality_score": 95.0,
                "decision_score": 0.83,
            },
            {
                "level": "ad",
                "entity_id": "a1",
                "campaign_id": "c1",
                "campaign_name": "Campanha 1",
                "adset_id": "s1",
                "adset_name": "Conjunto 1",
                "ad_id": "a1",
                "ad_name": "Criativo ruim",
                "action_multiplier": 0.0,
                "historical_days": 14,
                "historical_spend": 700.0,
                "expected_profit": 0.0,
                "expected_revenue": 0.0,
                "expected_incremental_profit_vs_hold": 180.0,
                "expected_incremental_revenue_vs_hold": -400.0,
                "p_profit": 1.0,
                "p_roas_target": 0.0,
                "p_beats_hold": 0.86,
                "p_incremental_profit_positive": 0.86,
                "p_action_optimal": 0.75,
                "cvar10_profit": 0.0,
                "expected_regret": 10.0,
                "response_confidence": 0.0,
                "evidence_tier": "predictive",
                "policy_eligible": True,
                "policy_constrained": False,
                "data_quality_score": 95.0,
                "decision_score": 0.90,
            },
        ]
    )


def test_plan_answers_capital_amount_and_duplication_candidate():
    source = pd.DataFrame(
        [
            {
                "date": "2026-09-04",
                "campaign_id": "c1",
                "campaign_daily_budget": 120.0,
            },
            {
                "date": "2026-09-05",
                "campaign_id": "c1",
                "campaign_daily_budget": 140.0,
            },
        ]
    )
    plan = build_operational_action_plan(
        _best(),
        source_df=source,
        horizon_days=7,
    )

    campaign = plan[plan["level"] == "campaign"].iloc[0]
    assert campaign["capital_action"] == "AUMENTAR"
    assert campaign["amount_basis"] == "campaign_daily_budget"
    assert campaign["current_daily_amount"] == 140.0
    assert campaign["recommended_daily_amount"] == 210.0
    assert campaign["daily_amount_change"] == 70.0
    assert campaign["duplicate_action"] == "TESTAR_DUPLICACAO"
    assert campaign["suggested_additional_copies"] == 1

    ad = plan[plan["level"] == "ad"].iloc[0]
    assert ad["capital_action"] == "DESLIGAR"
    assert ad["recommended_daily_amount"] == 0.0


def test_portfolio_allocation_replaces_independent_campaign_action():
    best = _best()
    allocation = best[best["level"] == "campaign"].copy()
    allocation["action_multiplier"] = 0.8
    allocation["expected_incremental_profit_vs_hold"] = 100.0
    allocation["expected_incremental_revenue_vs_hold"] = -100.0

    plan = build_operational_action_plan(
        best,
        allocation=allocation,
        horizon_days=7,
    )
    campaign = plan[plan["level"] == "campaign"].iloc[0]
    assert campaign["capital_action"] == "REDUZIR"
    assert campaign["action_multiplier"] == 0.8
