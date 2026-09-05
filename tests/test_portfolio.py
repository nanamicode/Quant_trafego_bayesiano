import pandas as pd

from quant_trafego.portfolio import (
    PortfolioRiskConfig,
    estimate_campaign_correlation,
    optimize_campaign_portfolio,
)


def _history():
    rows = []
    for day in range(30):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
        common = (day % 5) * 10
        for campaign, base in [("c1", 300), ("c2", 250)]:
            rows.append(
                {
                    "date": date,
                    "campaign_id": campaign,
                    "spend": 100.0,
                    "revenue": base + common,
                }
            )
    return pd.DataFrame(rows)


def _actions():
    rows = []
    for campaign, base in [("c1", 100.0), ("c2", 80.0)]:
        for mult, spend, factor in [
            (0.8, 80, 0.8),
            (1.0, 100, 1.0),
            (1.2, 120, 1.3),
            (1.5, 150, 1.45),
        ]:
            expected = base * factor
            rows.append(
                {
                    "level": "campaign",
                    "entity_id": campaign,
                    "action_multiplier": mult,
                    "expected_spend": spend,
                    "expected_profit": expected,
                    "profit_p05": expected - 80,
                    "profit_p50": expected,
                    "profit_p95": expected + 100,
                    "p_profit": 0.90,
                    "p_incremental_profit_positive": 0.80,
                    "response_confidence": 0.30,
                }
            )
    return pd.DataFrame(rows)


def test_portfolio_optimizer_respects_budget_and_reports_joint_cvar():
    selected, summary = optimize_campaign_portfolio(
        _actions(),
        _history(),
        contribution_margin=1.0,
        total_budget=220.0,
        risk_config=PortfolioRiskConfig(
            scenarios=200,
            seed=8,
        ),
    )
    assert len(selected) == 2
    assert selected["expected_spend"].sum() <= 220.0 + 1e-9
    assert summary["scenario_count"] == 200
    assert 0 <= summary["scenario_p_profit"] <= 1
    assert summary["scenario_portfolio_cvar"] <= summary["scenario_profit_p50"]


def test_pairwise_correlation_is_shrunk_by_actual_overlap():
    rows = []
    for day in range(100):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
        rows.append(
            {
                "date": date,
                "campaign_id": "c1",
                "spend": 100.0,
                "revenue": 200.0 + day,
            }
        )
        if day >= 95:
            rows.append(
                {
                    "date": date,
                    "campaign_id": "c2",
                    "spend": 100.0,
                    "revenue": 300.0 + 2 * day,
                }
            )

    corr, n_days = estimate_campaign_correlation(
        pd.DataFrame(rows),
        ["c1", "c2"],
        contribution_margin=1.0,
        shrinkage_days=20.0,
    )
    assert n_days == 100
    # Raw overlap correlation is nearly one, but only five shared days
    # exist, so pair-specific shrinkage must keep dependence conservative.
    assert 0.0 < corr[0, 1] < 0.35


def test_portfolio_cannot_override_engine_policy_eligibility():
    actions = _actions()
    actions["policy_eligible"] = actions["action_multiplier"] <= 1.0
    selected, _ = optimize_campaign_portfolio(
        actions,
        _history(),
        contribution_margin=1.0,
        total_budget=300.0,
        risk_config=PortfolioRiskConfig(
            scenarios=150,
            seed=12,
        ),
    )
    assert (selected["action_multiplier"] <= 1.0).all()
