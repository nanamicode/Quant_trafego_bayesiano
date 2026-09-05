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
