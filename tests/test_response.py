import numpy as np
import pandas as pd

from quant_trafego.response import estimate_response


def test_response_estimates_sublinear_elasticity():
    rng = np.random.default_rng(4)
    rows = []
    true_elasticity = 0.70

    for day in range(60):
        spend = 100 + day * 15
        mean_conv = 2.0 * (spend / 100.0) ** true_elasticity
        conversions = max(1, int(round(mean_conv + rng.normal(0, 0.25))))
        rows.append({
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
            "spend": spend,
            "conversions": conversions,
            "revenue": conversions * 100,
        })

    est = estimate_response(pd.DataFrame(rows))
    assert 0.35 < est.elasticity_mean < 1.05
    assert est.confidence > 0
    assert est.diminishing_returns_probability_proxy > 0.5


def test_response_confidence_collapses_when_spend_is_time_confounded():
    rows = []
    for day in range(45):
        # Spend rises almost deterministically with time.
        spend = 100.0 * np.exp(day * 0.015)
        # Conversions also rise with time, but not because of spend.
        conversions = max(
            1,
            int(round(3.0 * np.exp(day * 0.018))),
        )
        rows.append(
            {
                "date": pd.Timestamp("2026-03-01") + pd.Timedelta(days=day),
                "spend": spend,
                "conversions": conversions,
                "revenue": conversions * 100.0,
            }
        )

    est = estimate_response(pd.DataFrame(rows))
    assert est.independent_spend_sd < 0.08
    assert est.confidence < 0.20
    assert "trend" in est.controls


def test_response_controls_weekday_when_history_supports_it():
    rng = np.random.default_rng(13)
    rows = []
    true_elasticity = 0.65
    weekday_effect = np.array([0.0, 0.20, 0.10, -0.05, 0.15, -0.10, -0.15])

    for day in range(70):
        date = pd.Timestamp("2026-01-05") + pd.Timedelta(days=day)
        # Independent spend variation plus a slow drift.
        spend = 220.0 * np.exp(
            0.18 * np.sin(day * 1.7) + 0.002 * day
        )
        log_mean = (
            np.log(5.0)
            + true_elasticity * np.log(spend / 220.0)
            + 0.006 * day
            + weekday_effect[date.dayofweek]
        )
        conversions = max(
            1,
            int(round(np.exp(log_mean) + rng.normal(0, 0.35))),
        )
        rows.append(
            {
                "date": date,
                "spend": spend,
                "conversions": conversions,
                "revenue": conversions * 100.0,
            }
        )

    est = estimate_response(pd.DataFrame(rows))
    assert "weekday" in est.controls
    assert est.independent_spend_sd > 0.05
    assert 0.25 < est.elasticity_mean < 1.05
    assert est.confidence > 0
