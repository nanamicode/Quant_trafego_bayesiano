import pandas as pd

from quant_trafego.optimization import AllocationConfig
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


def test_portfolio_uses_revenue_only_as_near_optimal_tiebreak():
    actions = pd.DataFrame(
        [
            {
                "level": "campaign",
                "entity_id": "c1",
                "action_multiplier": 1.0,
                "expected_spend": 100.0,
                "expected_profit": 100.0,
                "expected_revenue": 500.0,
                "profit_p05": 80.0,
                "profit_p50": 100.0,
                "profit_p95": 120.0,
                "p_profit": 0.95,
                "p_incremental_profit_positive": 0.8,
                "response_confidence": 0.30,
            },
            {
                "level": "campaign",
                "entity_id": "c1",
                "action_multiplier": 1.2,
                "expected_spend": 120.0,
                "expected_profit": 99.0,
                "expected_revenue": 900.0,
                "profit_p05": 79.0,
                "profit_p50": 99.0,
                "profit_p95": 119.0,
                "p_profit": 0.95,
                "p_incremental_profit_positive": 0.8,
                "response_confidence": 0.30,
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                "campaign_id": "c1",
                "spend": 100.0,
                "revenue": 220.0,
            }
            for day in range(30)
        ]
    )
    selected, summary = optimize_campaign_portfolio(
        actions,
        history,
        contribution_margin=1.0,
        total_budget=120.0,
        allocation_config=AllocationConfig(
            revenue_tiebreak=True,
            revenue_tiebreak_tolerance=0.02,
        ),
        risk_config=PortfolioRiskConfig(
            scenarios=120,
            seed=9,
            cvar_weight=0.0,
        ),
    )
    assert selected.iloc[0]["action_multiplier"] == 1.2
    assert summary["expected_portfolio_revenue"] == 900.0


def test_cvar_portfolio_does_not_collapse_to_zero_when_one_campaign_is_profitable():
    actions = pd.DataFrame(
        [
            {
                "level": "campaign",
                "entity_id": "winner",
                "action_multiplier": 0.0,
                "expected_spend": 0.0,
                "expected_profit": 0.0,
                "expected_revenue": 0.0,
                "profit_p05": 0.0,
                "profit_p50": 0.0,
                "profit_p95": 0.0,
                "p_profit": 0.0,
                "p_incremental_profit_positive": 0.0,
                "response_confidence": 0.0,
                "policy_eligible": True,
            },
            {
                "level": "campaign",
                "entity_id": "winner",
                "action_multiplier": 1.0,
                "expected_spend": 100.0,
                "expected_profit": 50.0,
                "expected_revenue": 200.0,
                "profit_p05": 30.0,
                "profit_p50": 50.0,
                "profit_p95": 70.0,
                "p_profit": 0.95,
                "p_incremental_profit_positive": 0.5,
                "response_confidence": 0.0,
                "policy_eligible": True,
            },
            {
                "level": "campaign",
                "entity_id": "loser",
                "action_multiplier": 0.0,
                "expected_spend": 0.0,
                "expected_profit": 0.0,
                "expected_revenue": 0.0,
                "profit_p05": 0.0,
                "profit_p50": 0.0,
                "profit_p95": 0.0,
                "p_profit": 0.0,
                "p_incremental_profit_positive": 0.9,
                "response_confidence": 0.0,
                "policy_eligible": True,
            },
            {
                "level": "campaign",
                "entity_id": "loser",
                "action_multiplier": 1.0,
                "expected_spend": 100.0,
                "expected_profit": -40.0,
                "expected_revenue": 60.0,
                "profit_p05": -80.0,
                "profit_p50": -40.0,
                "profit_p95": -10.0,
                "p_profit": 0.05,
                "p_incremental_profit_positive": 0.5,
                "response_confidence": 0.0,
                "policy_eligible": True,
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                "campaign_id": campaign,
                "spend": 100.0,
                "revenue": revenue,
            }
            for day in range(30)
            for campaign, revenue in [
                ("winner", 180.0),
                ("loser", 60.0),
            ]
        ]
    )
    selected, summary = optimize_campaign_portfolio(
        actions,
        history,
        contribution_margin=1.0,
        total_budget=200.0,
        risk_config=PortfolioRiskConfig(
            scenarios=200,
            seed=17,
            cvar_weight=0.25,
        ),
    )
    chosen = dict(
        zip(
            selected["entity_id"],
            selected["action_multiplier"],
        )
    )
    assert chosen["winner"] == 1.0
    assert chosen["loser"] == 0.0
    assert summary["selected_spend"] == 100.0


def test_structural_zero_panel_rows_do_not_fake_campaign_overlap():
    rows = []
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    for date in dates:
        for campaign in ["c1", "c2"]:
            if campaign == "c1":
                delivering = date <= pd.Timestamp("2026-01-10")
            else:
                delivering = (
                    pd.Timestamp("2026-01-06")
                    <= date
                    <= pd.Timestamp("2026-01-15")
                )
            rows.append(
                {
                    "date": date,
                    "campaign_id": campaign,
                    "spend": 100.0 if delivering else 0.0,
                    "revenue": 180.0 if delivering else 0.0,
                    "impressions": 1000 if delivering else 0,
                }
            )

    corr, n_days = estimate_campaign_correlation(
        pd.DataFrame(rows),
        ["c1", "c2"],
        contribution_margin=1.0,
        shrinkage_days=20.0,
    )
    assert n_days == 30
    # True simultaneous delivery is only Jan 6-10 (5 days). Structural
    # zero rows outside delivery must not create 30 days of overlap.
    assert abs(corr[0, 1]) < 0.35
