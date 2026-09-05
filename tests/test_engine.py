import numpy as np
import pandas as pd
import pytest

from quant_trafego.engine import BayesTrafficEngine, EngineConfig
from quant_trafego.model import BetaPosterior


def test_engine_runs():
    df = pd.read_csv("examples/example_data.csv")
    engine = BayesTrafficEngine(EngineConfig(draws=1000, seed=123))
    all_actions, best = engine.run(df)

    assert not all_actions.empty
    assert not best.empty
    assert {"account", "campaign", "adset", "ad"} <= set(all_actions["level"])
    assert all_actions["p_profit"].between(0, 1).all()
    assert all_actions["p_roas_target"].between(0, 1).all()
    assert all_actions["p_action_optimal"].between(0, 1).all()
    assert best["decision_confidence"].between(0, 1).all()

    sums = all_actions.groupby(["level", "entity_id"])["p_action_optimal"].sum()
    assert ((sums - 1.0).abs() < 1e-9).all()


def test_zero_contribution_margin_cannot_generate_profit_with_spend():
    df = pd.read_csv("examples/example_data.csv")
    engine = BayesTrafficEngine(
        EngineConfig(draws=500, seed=123, contribution_margin=0.0)
    )
    all_actions, _ = engine.run(df)

    spending_actions = all_actions[all_actions["expected_spend"] > 0]
    assert (spending_actions["p_profit"] == 0.0).all()
    assert (spending_actions["expected_profit"] < 0.0).all()


def test_engine_accepts_mcmc_posterior_override():
    df = pd.read_csv("examples/example_data.csv")
    engine = BayesTrafficEngine(EngineConfig(draws=300, seed=123))
    overrides = {
        ("account", "ALL"): (
            BetaPosterior(2500, 97500),
            BetaPosterior(80, 920),
        )
    }
    all_actions, _ = engine.run(df, posterior_overrides=overrides)
    account = all_actions[all_actions["level"] == "account"]
    assert set(account["posterior_source"]) == {"mcmc"}


def test_policy_distinguishes_unconstrained_and_approved_actions():
    df = pd.read_csv("examples/example_data.csv")
    engine = BayesTrafficEngine(
        EngineConfig(
            draws=400,
            seed=8,
            predictive_max_multiplier=1.0,
            observational_max_multiplier=1.0,
            experiment_max_multiplier=1.0,
        )
    )
    _, best = engine.run(df)
    assert (best["action_multiplier"] <= 1.0).all()
    assert "unconstrained_best_multiplier" in best.columns
    assert "policy_constrained" in best.columns


def test_engine_rejects_non_numeric_core_values():
    df = pd.read_csv("examples/example_data.csv").head(5).copy()
    df.loc[df.index[0], "spend"] = "invalid-number"
    engine = BayesTrafficEngine(EngineConfig(draws=100))
    with pytest.raises(ValueError, match="não numéricos"):
        engine.run(df)


def test_low_quality_data_blocks_scale_up_policy():
    rows = [
        {
            "date": "2026-01-01",
            "campaign_id": "c1",
            "adset_id": "s1",
            "ad_id": "a1",
            "impressions": 1000,
            "clicks": 50,
            "conversions": 5,
            "spend": 100,
            "revenue": 800,
        },
        {
            "date": "2026-01-01",
            "campaign_id": "c1",
            "adset_id": "s1",
            "ad_id": "a1",
            "impressions": 1000,
            "clicks": 50,
            "conversions": 5,
            "spend": 100,
            "revenue": 800,
        },
        {
            "date": "2026-01-02",
            "campaign_id": "c1",
            "adset_id": "s1",
            "ad_id": "a1",
            "impressions": 0,
            "clicks": 0,
            "conversions": 0,
            "spend": 0,
            "revenue": 0,
        },
        {
            "date": "2026-01-03",
            "campaign_id": "c1",
            "adset_id": "s1",
            "ad_id": "a1",
            "impressions": 0,
            "clicks": 0,
            "conversions": 0,
            "spend": 0,
            "revenue": 0,
        },
    ]
    engine = BayesTrafficEngine(
        EngineConfig(
            draws=200,
            seed=5,
        )
    )
    all_actions, best = engine.run(pd.DataFrame(rows))
    assert float(best["data_quality_score"].iloc[0]) < 55.0
    scale = all_actions["action_multiplier"] > 1.0
    assert (~all_actions.loc[scale, "policy_eligible"]).all()
    assert (
        best["decision_confidence"]
        <= best["decision_confidence_raw"]
    ).all()


def test_action_results_are_invariant_to_action_grid_order():
    df = pd.read_csv("examples/example_data.csv")
    actions_a = (0.8, 1.0, 1.2)
    actions_b = (1.2, 0.8, 1.0)

    a, _ = BayesTrafficEngine(
        EngineConfig(
            draws=350,
            seed=77,
            actions=actions_a,
        )
    ).run(df)
    b, _ = BayesTrafficEngine(
        EngineConfig(
            draws=350,
            seed=77,
            actions=actions_b,
        )
    ).run(df)

    keys = [
        "level",
        "entity_id",
        "action_multiplier",
    ]
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)

    for col in [
        "expected_profit",
        "p_profit",
        "p_roas_target",
        "p_action_optimal",
        "expected_regret",
        "cvar10_profit",
    ]:
        assert np.allclose(
            a[col].to_numpy(),
            b[col].to_numpy(),
            rtol=0,
            atol=1e-12,
        )


def test_action_grid_requires_hold_baseline():
    with pytest.raises(ValueError, match="1.0x"):
        BayesTrafficEngine(
            EngineConfig(
                actions=(0.0, 0.8, 1.2),
            )
        )


def test_entity_selection_uses_revenue_only_as_secondary_objective():
    engine = BayesTrafficEngine(
        EngineConfig(
            revenue_tiebreak=True,
            revenue_tiebreak_tolerance=0.02,
        )
    )
    actions = pd.DataFrame(
        [
            {
                "level": "campaign",
                "entity_id": "c1",
                "risk_adjusted_utility": 100.0,
                "expected_revenue": 500.0,
            },
            {
                "level": "campaign",
                "entity_id": "c1",
                "risk_adjusted_utility": 99.0,
                "expected_revenue": 900.0,
            },
            {
                "level": "campaign",
                "entity_id": "c1",
                "risk_adjusted_utility": 90.0,
                "expected_revenue": 5000.0,
            },
        ]
    )
    idx = engine._select_profit_first_growth_indices(actions)
    assert idx == [1]


def test_inactive_history_informs_active_prior_without_receiving_actions():
    rows = []
    for day in range(14):
        date = pd.Timestamp("2026-08-01") + pd.Timedelta(days=day)
        rows.extend(
            [
                {
                    "date": date,
                    "campaign_id": "c_active",
                    "adset_id": "s_active",
                    "ad_id": "a_active",
                    "status": "active",
                    "impressions": 1000,
                    "clicks": 10,
                    "conversions": 1,
                    "spend": 50.0,
                    "revenue": 100.0,
                },
                {
                    "date": date,
                    "campaign_id": "c_paused",
                    "adset_id": "s_paused",
                    "ad_id": "a_paused",
                    "status": "paused",
                    "impressions": 1000,
                    "clicks": 200,
                    "conversions": 20,
                    "spend": 50.0,
                    "revenue": 2000.0,
                },
            ]
        )

    full = pd.DataFrame(rows)
    decision_entities = {
        "campaign": {"c_active"},
        "adset": {"s_active"},
        "ad": {"a_active"},
    }
    cfg = EngineConfig(
        draws=200,
        seed=123,
    )
    full_actions, _ = BayesTrafficEngine(cfg).run(
        full,
        decision_entities=decision_entities,
    )
    active_only = full[
        full["campaign_id"] == "c_active"
    ].copy()
    active_actions, _ = BayesTrafficEngine(cfg).run(
        active_only
    )

    assert set(full_actions["entity_id"]) == {
        "ALL",
        "c_active",
        "s_active",
        "a_active",
    }
    full_ctr = full_actions.loc[
        (full_actions["level"] == "campaign")
        & (full_actions["entity_id"] == "c_active"),
        "posterior_ctr_mean",
    ].iloc[0]
    active_ctr = active_actions.loc[
        (active_actions["level"] == "campaign")
        & (active_actions["entity_id"] == "c_active"),
        "posterior_ctr_mean",
    ].iloc[0]
    assert full_ctr > active_ctr
