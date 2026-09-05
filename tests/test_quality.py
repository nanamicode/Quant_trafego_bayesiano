import pandas as pd

from quant_trafego.quality import assess_data_quality


def test_quality_report():
    df = pd.read_csv("examples/example_data.csv")
    report = assess_data_quality(df)
    assert 0 <= report.score <= 100
    assert report.rows == len(df)
    assert report.campaigns >= 1
