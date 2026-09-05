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
