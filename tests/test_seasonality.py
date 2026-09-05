import numpy as np
import pandas as pd

from quant_trafego.seasonality import analyze_weekly_seasonality


def _weekly_panel(days: int):
    rows = []
    start = pd.Timestamp("2026-01-05")  # Monday
    for day in range(days):
        date = start + pd.Timedelta(days=day)
        weekend = date.dayofweek >= 5
        ctr = 0.020 if weekend else 0.040
        impressions = 10_000
        clicks = int(round(impressions * ctr))
        conversions = int(round(clicks * 0.05))
        rows.append(
            {
                "date": date,
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
            }
        )
    return pd.DataFrame(rows)


def test_weekly_seasonality_detects_weekend_ctr_drop():
    df = _weekly_panel(84)
    signal = analyze_weekly_seasonality(df)

    means = np.asarray(signal.ctr.means)
    weekday_mean = float(means[:5].mean())
    weekend_mean = float(means[5:].mean())

    assert signal.ctr.confidence > 0
    assert weekend_mean < weekday_mean
    assert weekend_mean < 0

    # 84 days starting Monday ends on Sunday. The next five days are
    # Monday-Friday, so their average shift should be positive relative to the
    # historical weekday mix that includes weekends.
    mean_shift, sd_shift = signal.ctr.future_shift(
        df["date"].max(),
        5,
    )
    assert mean_shift > 0
    assert sd_shift >= 0


def test_weekly_seasonality_is_disabled_for_short_history():
    df = _weekly_panel(14)
    signal = analyze_weekly_seasonality(df)
    assert signal.ctr.confidence == 0.0
    assert signal.cvr.confidence == 0.0
    assert signal.ctr.future_shift(df["date"].max(), 7) == (0.0, 0.0)
