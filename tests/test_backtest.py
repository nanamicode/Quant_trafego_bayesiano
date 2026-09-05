import numpy as np
import pandas as pd

from quant_trafego.backtest import rolling_origin_backtest
from quant_trafego.engine import EngineConfig


def _synthetic_panel():
    rows = []
    rng = np.random.default_rng(7)
    for d in range(42):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)
        for ad_id, cvr in [("a1", 0.05), ("a2", 0.03)]:
            impressions = 5000
            clicks = 120 + int(rng.integers(-5, 6))
            conversions = max(
                0, int(round(clicks * cvr + rng.normal(0, 0.8)))
            )
            spend = 180 + float(rng.normal(0, 5))
            revenue = conversions * 100.0
            rows.append(
                {
                    "date": date,
                    "campaign_id": "c1",
                    "adset_id": "s1",
                    "ad_id": ad_id,
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "spend": max(spend, 1),
                    "revenue": revenue,
                }
            )
    return pd.DataFrame(rows)


def test_rolling_origin_backtest_produces_probabilistic_scores():
    detail, summary = rolling_origin_backtest(
        _synthetic_panel(),
        config=EngineConfig(draws=250, seed=9),
        min_train_days=21,
        horizon_days=7,
        step_days=7,
        levels=("account", "ad"),
    )
    assert not detail.empty
    assert not summary.empty
    assert detail["profit_brier"].between(0, 1).all()
    assert detail["profit_90_covered"].between(0, 1).all()
    assert set(summary["level"]) == {"account", "ad"}