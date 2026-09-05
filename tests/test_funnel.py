import pandas as pd

from quant_trafego.funnel import detect_funnel_schema, hierarchical_funnel_diagnostics


def test_optional_funnel_is_detected_and_quantified():
    df = pd.DataFrame(
        [
            {
                "campaign_id": "c1",
                "adset_id": "s1",
                "ad_id": "a1",
                "impressions": 1000,
                "clicks": 100,
                "landing_page_views": 80,
                "adds_to_cart": 20,
                "checkouts": 10,
                "conversions": 5,
            },
            {
                "campaign_id": "c1",
                "adset_id": "s1",
                "ad_id": "a1",
                "impressions": 1200,
                "clicks": 120,
                "landing_page_views": 95,
                "adds_to_cart": 24,
                "checkouts": 12,
                "conversions": 6,
            },
        ]
    )
    schema = detect_funnel_schema(df)
    assert schema.available_stages[-1] == "conversions"
    detail = hierarchical_funnel_diagnostics(df)
    assert not detail.empty
    assert detail["valid_binomial_transition"].all()
    assert detail["posterior_mean"].between(0, 1).all()
