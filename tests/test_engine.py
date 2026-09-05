import pandas as pd

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
