import numpy as np
import pandas as pd

from quant_trafego.temporal import analyze_temporal


def test_temporal_detects_improving_cvr():
    rows = []
    rng = np.random.default_rng(1)
    for day in range(30):
        impressions = 10000
        ctr = 0.02
        clicks = int(impressions * ctr)
        cvr = 0.02 + day * 0.0015
        conversions = int(clicks * cvr)
        rows.append({
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "spend": 400 + day,
            "revenue": conversions * 100,
        })

    signal = analyze_temporal(pd.DataFrame(rows), seed=7)
    assert signal.p_recent_cvr_better > 0.70
    assert signal.cvr.p_positive > 0.70
    assert signal.cvr.mean > 0


def test_temporal_signal_exposes_current_level_anchor_when_recent_cvr_degrades():
    rows = []
    for day in range(35):
        impressions = 10000
        clicks = 300
        cvr = 0.08 if day < 28 else 0.02
        rows.append(
            {
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                "impressions": impressions,
                "clicks": clicks,
                "conversions": int(round(clicks * cvr)),
                "spend": 500.0,
                "revenue": int(round(clicks * cvr)) * 100.0,
            }
        )

    signal = analyze_temporal(pd.DataFrame(rows), seed=8)
    assert signal.cvr_current_logit_shift < 0
    assert signal.cvr_current_shift_confidence > 0.5
