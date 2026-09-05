import numpy as np
import pandas as pd

from quant_trafego.dynamic import fit_dynamic_rate, analyze_state_space_temporal


def test_dynamic_rate_detects_sustained_improvement():
    trials = np.full(40, 5000.0)
    p = np.linspace(0.015, 0.035, 40)
    successes = np.round(trials * p)
    state = fit_dynamic_rate(successes, trials)
    assert state.trend_mean > 0
    assert state.confidence > 0
    assert state.n_observations == 40


def test_state_space_temporal_returns_valid_signal():
    rows = []
    for day in range(35):
        impressions = 5000
        clicks = int(round(impressions * (0.02 + day * 0.0002)))
        conversions = int(round(clicks * 0.05))
        rows.append(
            {
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "spend": 200.0,
                "revenue": conversions * 100.0,
            }
        )
    signal = analyze_state_space_temporal(pd.DataFrame(rows), seed=2)
    assert 0 <= signal.ctr.p_positive <= 1
    assert 0 <= signal.ctr.confidence <= 1
