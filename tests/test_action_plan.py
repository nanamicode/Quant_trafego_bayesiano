import pandas as pd

from quant_trafego.action_plan import build_operational_action_plan, derive_account_budget_target


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
    assert campaign["amount_basis"] == "campaign_daily_budget_fallback"
    assert campaign["configured_daily_budget"] == 140.0
    assert campaign["current_daily_amount"] == 140.0
    assert campaign["recommended_daily_amount"] == 210.0
    assert campaign["daily_amount_change"] == 70.0
    assert campaign["duplicate_action"] == "TESTAR_DUPLICACAO"
    assert campaign["suggested_additional_copies"] == 1

    ad = plan[plan["level"] == "ad"].iloc[0]
    assert ad["capital_action"] == "DESLIGAR"
    assert pd.isna(ad["recommended_daily_amount"])


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


def test_plan_uses_recent_spend_when_budget_is_absent():
    source = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-08-20") + pd.Timedelta(days=day),
                "campaign_id": "c1",
                "spend": 20.0 if day < 3 else 100.0,
            }
            for day in range(10)
        ]
    )
    best = _best()
    plan = build_operational_action_plan(
        best[best["level"] == "campaign"],
        source_df=source,
        horizon_days=7,
    )
    row = plan.iloc[0]
    assert row["amount_basis"] == "recent_7d_avg_daily_spend"
    assert row["current_daily_amount"] == 100.0
    assert row["recommended_daily_amount"] == 150.0


def test_adset_allocation_replaces_independent_adset_action():
    best = pd.concat(
        [
            _best(),
            pd.DataFrame(
                [
                    {
                        "level": "adset",
                        "entity_id": "s1",
                        "campaign_id": "c1",
                        "campaign_name": "Campanha 1",
                        "adset_id": "s1",
                        "adset_name": "Conjunto 1",
                        "action_multiplier": 1.5,
                        "historical_days": 14,
                        "historical_spend": 1400.0,
                        "expected_profit": 500.0,
                        "expected_revenue": 1500.0,
                        "expected_incremental_profit_vs_hold": 100.0,
                        "expected_incremental_revenue_vs_hold": 300.0,
                        "p_profit": 0.9,
                        "p_roas_target": 0.7,
                        "p_beats_hold": 0.7,
                        "p_incremental_profit_positive": 0.75,
                        "p_action_optimal": 0.6,
                        "cvar10_profit": 100.0,
                        "expected_regret": 20.0,
                        "response_confidence": 0.3,
                        "evidence_tier": "observational_intervention",
                        "policy_eligible": True,
                        "policy_constrained": False,
                        "data_quality_score": 95.0,
                        "decision_score": 0.8,
                    }
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    nested = best[best["level"] == "adset"].copy()
    nested["action_multiplier"] = 0.8
    nested["expected_incremental_profit_vs_hold"] = 40.0

    plan = build_operational_action_plan(
        best,
        adset_allocation=nested,
        horizon_days=7,
    )
    adset = plan[plan["level"] == "adset"].iloc[0]
    assert adset["capital_action"] == "REDUZIR"
    assert adset["action_multiplier"] == 0.8


def test_account_action_defines_absolute_capital_envelope():
    best = pd.DataFrame(
        [
            {
                "level": "account",
                "entity_id": "ALL",
                "action_multiplier": 1.2,
                "historical_days": 14,
                "historical_spend": 1400.0,
                "p_profit": 0.9,
                "p_incremental_profit_positive": 0.8,
                "decision_score": 0.8,
            }
        ]
    )
    source = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-08-30") + pd.Timedelta(days=day),
                "campaign_id": campaign_id,
                "campaign_daily_budget": budget,
                "spend": spend,
            }
            for day in range(7)
            for campaign_id, budget, spend in [
                ("c1", 100.0, 90.0),
                ("c2", 50.0, 45.0),
            ]
        ]
    )
    target = derive_account_budget_target(
        best,
        source_df=source,
        horizon_days=7,
    )
    assert target["amount_basis"] == "recent_7d_account_daily_spend"
    assert target["configured_daily_budget"] == 150.0
    assert target["current_daily_amount"] == 135.0
    assert target["recommended_daily_amount"] == 162.0
    assert target["recommended_horizon_amount"] == 1134.0


def test_campaign_and_adset_amounts_close_to_parent_envelopes():
    best = pd.DataFrame(
        [
            {
                "level": "account",
                "entity_id": "ALL",
                "action_multiplier": 1.2,
                "historical_days": 7,
                "historical_spend": 700.0,
                "p_profit": 0.9,
                "p_incremental_profit_positive": 0.8,
                "decision_score": 0.8,
            },
            {
                "level": "campaign",
                "entity_id": "c1",
                "campaign_id": "c1",
                "campaign_name": "C1",
                "action_multiplier": 1.2,
                "historical_days": 7,
                "historical_spend": 700.0,
                "expected_profit": 100.0,
                "expected_revenue": 300.0,
                "expected_incremental_profit_vs_hold": 20.0,
                "expected_incremental_revenue_vs_hold": 50.0,
                "p_profit": 0.9,
                "p_roas_target": 0.7,
                "p_beats_hold": 0.7,
                "p_incremental_profit_positive": 0.75,
                "p_action_optimal": 0.6,
                "cvar10_profit": 10.0,
                "expected_regret": 2.0,
                "response_confidence": 0.3,
                "evidence_tier": "observational_intervention",
                "policy_eligible": True,
                "data_quality_score": 95.0,
                "decision_score": 0.8,
            },
            {
                "level": "adset",
                "entity_id": "s1",
                "campaign_id": "c1",
                "campaign_name": "C1",
                "adset_id": "s1",
                "adset_name": "S1",
                "action_multiplier": 1.2,
                "historical_days": 7,
                "historical_spend": 420.0,
                "expected_profit": 60.0,
                "expected_revenue": 180.0,
                "expected_incremental_profit_vs_hold": 10.0,
                "expected_incremental_revenue_vs_hold": 20.0,
                "p_profit": 0.9,
                "p_roas_target": 0.7,
                "p_beats_hold": 0.7,
                "p_incremental_profit_positive": 0.75,
                "p_action_optimal": 0.6,
                "cvar10_profit": 5.0,
                "expected_regret": 1.0,
                "response_confidence": 0.3,
                "evidence_tier": "observational_intervention",
                "policy_eligible": True,
                "data_quality_score": 95.0,
                "decision_score": 0.8,
            },
            {
                "level": "adset",
                "entity_id": "s2",
                "campaign_id": "c1",
                "campaign_name": "C1",
                "adset_id": "s2",
                "adset_name": "S2",
                "action_multiplier": 0.8,
                "historical_days": 7,
                "historical_spend": 280.0,
                "expected_profit": 40.0,
                "expected_revenue": 120.0,
                "expected_incremental_profit_vs_hold": 8.0,
                "expected_incremental_revenue_vs_hold": -5.0,
                "p_profit": 0.9,
                "p_roas_target": 0.7,
                "p_beats_hold": 0.7,
                "p_incremental_profit_positive": 0.75,
                "p_action_optimal": 0.6,
                "cvar10_profit": 5.0,
                "expected_regret": 1.0,
                "response_confidence": 0.3,
                "evidence_tier": "observational_intervention",
                "policy_eligible": True,
                "data_quality_score": 95.0,
                "decision_score": 0.8,
            },
        ]
    )
    source = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-08-30") + pd.Timedelta(days=day),
                "campaign_id": "c1",
                "campaign_daily_budget": 100.0,
                "adset_id": adset_id,
                "adset_daily_budget": budget,
                "spend": spend,
            }
            for day in range(7)
            for adset_id, budget, spend in [
                ("s1", 60.0, 60.0),
                ("s2", 40.0, 40.0),
            ]
        ]
    )
    target = derive_account_budget_target(
        best,
        source_df=source,
        horizon_days=7,
    )
    campaign_alloc = best[best["level"] == "campaign"].copy()
    campaign_alloc["expected_spend"] = 840.0
    campaign_alloc["parent_account_budget_limit"] = 840.0

    nested = best[best["level"] == "adset"].copy()
    nested.loc[nested["entity_id"] == "s1", "expected_spend"] = 504.0
    nested.loc[nested["entity_id"] == "s2", "expected_spend"] = 336.0
    nested["parent_campaign_budget_limit"] = 840.0

    plan = build_operational_action_plan(
        best,
        allocation=campaign_alloc,
        adset_allocation=nested,
        account_budget_target=target,
        source_df=source,
        horizon_days=7,
    )

    campaign = plan[plan["level"] == "campaign"]
    adsets = plan[plan["level"] == "adset"]
    assert abs(campaign["recommended_daily_amount"].sum() - 120.0) < 1e-9
    assert abs(adsets["recommended_daily_amount"].sum() - 120.0) < 1e-9
    assert adsets["nested_budget_reconciled"].all()


def test_campaign_amount_uses_exact_selected_scenario_spend():
    best = _best()
    allocation = best[best["level"] == "campaign"].copy()
    allocation["action_multiplier"] = 1.0
    allocation["expected_spend"] = 700.0

    source = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-08-30") + pd.Timedelta(days=day),
                "campaign_id": "c1",
                "campaign_daily_budget": 500.0,
                "spend": 100.0,
            }
            for day in range(7)
        ]
    )
    plan = build_operational_action_plan(
        best,
        allocation=allocation,
        source_df=source,
        horizon_days=7,
    )
    campaign = plan[plan["level"] == "campaign"].iloc[0]
    assert campaign["action_multiplier"] == 1.0
    assert campaign["current_daily_amount"] == 100.0
    assert campaign["configured_daily_budget"] == 500.0
    assert campaign["recommended_daily_amount"] == 100.0
    assert campaign["capital_action"] == "MANTER"


def test_ad_is_blocked_when_parent_campaign_is_off_and_has_no_fake_budget():
    best = pd.DataFrame(
        [
            {
                "level": "campaign",
                "entity_id": "c1",
                "campaign_id": "c1",
                "action_multiplier": 0.0,
                "historical_days": 7,
                "historical_spend": 700.0,
                "expected_spend": 0.0,
                "expected_profit": 0.0,
                "expected_revenue": 0.0,
                "expected_incremental_profit_vs_hold": 100.0,
                "expected_incremental_revenue_vs_hold": -200.0,
                "p_profit": 1.0,
                "p_roas_target": 0.0,
                "p_beats_hold": 0.9,
                "p_incremental_profit_positive": 0.9,
                "p_action_optimal": 0.8,
                "cvar10_profit": 0.0,
                "expected_regret": 0.0,
                "response_confidence": 0.2,
                "evidence_tier": "observational_intervention",
                "policy_eligible": True,
            },
            {
                "level": "ad",
                "entity_id": "a1",
                "campaign_id": "c1",
                "adset_id": "s1",
                "ad_id": "a1",
                "ad_name": "Ad otimista isolado",
                "action_multiplier": 1.2,
                "historical_days": 7,
                "historical_spend": 700.0,
                "expected_spend": 840.0,
                "expected_profit": 300.0,
                "expected_revenue": 1500.0,
                "expected_incremental_profit_vs_hold": 80.0,
                "expected_incremental_revenue_vs_hold": 200.0,
                "p_profit": 0.9,
                "p_roas_target": 0.8,
                "p_beats_hold": 0.8,
                "p_incremental_profit_positive": 0.8,
                "p_action_optimal": 0.7,
                "cvar10_profit": 100.0,
                "expected_regret": 10.0,
                "response_confidence": 0.3,
                "evidence_tier": "observational_intervention",
                "policy_eligible": True,
            },
        ]
    )
    plan = build_operational_action_plan(
        best,
        horizon_days=7,
    )
    ad = plan[plan["level"] == "ad"].iloc[0]
    assert ad["model_suggested_action"] == "PRIORIZAR_MAIS"
    assert ad["capital_action"] == "BLOQUEADO_PELO_PAI"
    assert bool(ad["blocked_by_parent"])
    assert ad["duplicate_action"] == "NAO"
    assert pd.isna(ad["recommended_daily_amount"])
