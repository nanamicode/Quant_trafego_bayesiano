import pandas as pd

from quant_trafego.optimization import (
    AllocationConfig,
    optimize_campaign_allocation,
)


def _actions():
    rows = []
    for campaign, base_profit in [("c1", 100.0), ("c2", 80.0)]:
        for multiplier, spend, gain in [
            (0.8, 80.0, 0.7),
            (1.0, 100.0, 1.0),
            (1.2, 120.0, 1.35),
            (1.5, 150.0, 1.45),
        ]:
            rows.append(
                {
                    "level": "campaign",
                    "entity_id": campaign,
                    "action_multiplier": multiplier,
                    "expected_spend": spend,
                    "expected_profit": base_profit * gain,
                    "risk_adjusted_utility": base_profit * gain,
                    "p_profit": 0.9,
                    "p_incremental_profit_positive": 0.85,
                    "cvar10_profit": 20.0,
                    "expected_regret": 5.0,
                    "response_confidence": 0.30,
                }
            )
    return pd.DataFrame(rows)


def test_optimizer_respects_budget_and_one_action_per_campaign():
    selected, summary = optimize_campaign_allocation(
        _actions(),
        total_budget=220.0,
    )
    assert len(selected) == 2
    assert selected["entity_id"].nunique() == 2
    assert selected["expected_spend"].sum() <= 220.0 + 1e-9
    assert summary["selected_spend"] <= 220.0 + 1e-9


def test_predictive_evidence_caps_scale_up():
    selected, _ = optimize_campaign_allocation(
        _actions().assign(response_confidence=0.0),
        total_budget=300.0,
        config=AllocationConfig(predictive_max_multiplier=1.2),
    )
    assert (selected["action_multiplier"] <= 1.2).all()


def test_optimizer_cannot_override_engine_policy_eligibility():
    actions = _actions()
    actions["policy_eligible"] = actions["action_multiplier"] <= 1.0
    selected, _ = optimize_campaign_allocation(
        actions,
        total_budget=300.0,
    )
    assert (selected["action_multiplier"] <= 1.0).all()
