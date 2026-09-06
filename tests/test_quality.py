import pandas as pd

from quant_trafego.quality import assess_data_quality


def test_quality_report():
    df = pd.read_csv("examples/example_data.csv")
    report = assess_data_quality(df)
    assert 0 <= report.score <= 100
    assert report.rows == len(df)
    assert report.campaigns >= 1


def test_quality_detects_calendar_gaps_and_tracking_violations():
    df = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "campaign_id": "c1",
                "adset_id": "s1",
                "ad_id": "a1",
                "impressions": 100,
                "clicks": 10,
                "conversions": 2,
                "spend": 10,
                "revenue": 40,
            },
            {
                "date": "2026-01-20",
                "campaign_id": "c1",
                "adset_id": "s1",
                "ad_id": "a1",
                "impressions": 100,
                "clicks": 120,
                "conversions": 130,
                "spend": 10,
                "revenue": 40,
            },
        ]
    )
    report = assess_data_quality(df)
    assert report.calendar_span_days == 20
    assert report.calendar_coverage_ratio < 0.20
    assert report.funnel_tracking_violation_rows == 1
    assert report.score < 70


def test_quality_reports_missing_core_columns_without_crashing():
    df = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "campaign_id": ["c1"],
        }
    )
    report = assess_data_quality(df)
    assert "spend" in report.missing_core_columns
    assert report.score < 50


def test_quality_does_not_penalize_valid_meta_daily_attribution_lag():
    df = pd.DataFrame(
        [
            {
                "date": "2026-09-01",
                "campaign_id": "c1",
                "adset_id": "s1",
                "ad_id": "a1",
                "impressions": 1000,
                "clicks": 0,
                "conversions": 1,
                "spend": 10.0,
                "revenue": 100.0,
            },
            {
                "date": "2026-09-02",
                "campaign_id": "c1",
                "adset_id": "s1",
                "ad_id": "a1",
                "impressions": 1000,
                "clicks": 20,
                "conversions": 0,
                "spend": 10.0,
                "revenue": 0.0,
            },
        ]
    )
    report = assess_data_quality(df)
    assert report.funnel_tracking_violation_rows == 0


def test_quality_ignores_structural_zero_rows_outside_ad_delivery_window():
    rows = []
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    for date in dates:
        delivering = pd.Timestamp("2026-01-04") <= date <= pd.Timestamp("2026-01-08")
        rows.append(
            {
                "date": date,
                "campaign_id": "c1",
                "adset_id": "s1",
                "ad_id": "a1",
                "impressions": 1000 if delivering else 0,
                "clicks": 20 if delivering else 0,
                "conversions": 2 if delivering else 0,
                "spend": 50.0 if delivering else 0.0,
                "revenue": 150.0 if delivering else 0.0,
            }
        )

    report = assess_data_quality(pd.DataFrame(rows))
    assert report.delivery_window_rows == 5
    assert report.zero_spend_rows == 0
    assert report.zero_impression_rows == 0
    assert not any(
        "gasto zero" in warning
        for warning in report.warnings
    )
    assert not any(
        "impressões zero" in warning
        for warning in report.warnings
    )
