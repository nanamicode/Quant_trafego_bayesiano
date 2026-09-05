import pandas as pd
import pytest

pytest.importorskip("pymc")

from quant_trafego.mcmc import fit_hierarchical_funnel, posterior_rate_overrides
from quant_trafego.ppc import posterior_predictive_checks


def _tiny_daily_panel():
    rows = []
    for day in range(6):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
        for ad_id, clicks, conversions in [
            ("a1", 45 + day, 4 + (day % 2)),
            ("a2", 30 + day, 2 + (day % 2)),
        ]:
            rows.append(
                {
                    "date": date,
                    "campaign_id": "c1",
                    "adset_id": "s1",
                    "ad_id": ad_id,
                    "impressions": 1000,
                    "clicks": clicks,
                    "conversions": conversions,
                    "spend": 100.0,
                    "revenue": conversions * 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_deep_advi_daily_panel_smoke():
    df = _tiny_daily_panel()
    idata, diagnostics, mapping = fit_hierarchical_funnel(
        df,
        method="advi",
        advi_steps=250,
        draws=100,
        chains=2,
        seed=4,
        return_mapping=True,
    )

    assert diagnostics.method == "advi"
    assert diagnostics.n_days == 6
    assert diagnostics.n_observations == 12
    assert "ctr_p_ad_current" in idata.posterior
    assert "cvr_p_entity" in idata.posterior

    overrides = posterior_rate_overrides(idata, mapping)
    assert ("account", "ALL") in overrides
    assert ("ad", "a1") in overrides

    detail, summary = posterior_predictive_checks(
        idata,
        df,
        mapping,
        draws=100,
        seed=5,
    )
    assert len(detail) == 12
    assert summary.n_observations == 12


def test_deep_nuts_api_smoke():
    df = _tiny_daily_panel()
    idata, diagnostics, mapping = fit_hierarchical_funnel(
        df,
        method="nuts",
        draws=30,
        tune=30,
        chains=2,
        cores=1,
        target_accept=0.85,
        seed=11,
        return_mapping=True,
    )
    assert diagnostics.method == "nuts"
    assert diagnostics.n_days == 6
    assert diagnostics.divergences is not None
    assert "ctr_p_account_current" in idata.posterior
    assert ("campaign", "c1") in posterior_rate_overrides(idata, mapping)
