import numpy as np
import pandas as pd

from quant_trafego.ppc import posterior_predictive_checks


class FakeArray:
    def __init__(self, values):
        self.values = np.asarray(values)

    def isel(self, **kwargs):
        idx = kwargs["ad"]
        return FakeArray(self.values[..., idx])


class FakePosterior:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return FakeArray(self.data[key])


class FakeIdata:
    def __init__(self, data):
        self.posterior = FakePosterior(data)


def test_ppc_produces_coverage_and_tail_checks():
    df = pd.DataFrame(
        [
            {
                "campaign_id": "c1",
                "adset_id": "s1",
                "ad_id": "a1",
                "impressions": 1000,
                "clicks": 50,
                "conversions": 5,
            }
        ]
    )
    # chain x draw x ad
    ctr = np.full((2, 500, 1), 0.05)
    cvr = np.full((2, 500, 1), 0.10)
    idata = FakeIdata({"ctr_p_ad": ctr, "cvr_p_ad": cvr})
    detail, summary = posterior_predictive_checks(
        idata,
        df,
        {"ad": ["a1"]},
        draws=500,
        seed=3,
    )
    assert len(detail) == 1
    assert 0 <= summary.click_90_coverage <= 1
    assert 0 <= summary.conversion_90_coverage <= 1
    assert summary.status in {"pass", "warning"}
