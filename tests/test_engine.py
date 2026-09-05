import pandas as pd

from quant_trafego.engine import BayesTrafficEngine, EngineConfig


def test_engine_runs():
    df = pd.read_csv("examples/example_data.csv")
    engine = BayesTrafficEngine(EngineConfig(draws=1000, seed=123))
    all_actions, best = engine.run(df)

    assert not all_actions.empty
    assert not best.empty
    assert {"account", "campaign", "adset", "ad"} <= set(all_actions["level"])
    assert all_actions["p_profit"].between(0, 1).all()
    assert all_actions["p_roas_target"].between(0, 1).all()
