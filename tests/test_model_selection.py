import pandas as pd

from quant_trafego.model_selection import _promotion_from_summaries


def _summary(**overrides):
    row = {
        "level": "account",
        "n_forecasts": 100,
        "profit_brier": 0.20,
        "roas_target_brier": 0.18,
        "profit_ece": 0.10,
        "roas_ece": 0.09,
        "profit_interval_score_90": 100.0,
        "profit_mae": 50.0,
        "profit_event_calibration_gap": 0.05,
        "profit_90_coverage": 0.84,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_model_promotion_passes_broad_consistent_improvement():
    baseline = _summary()
    candidate = _summary(
        profit_brier=0.17,
        roas_target_brier=0.16,
        profit_ece=0.075,
        roas_ece=0.075,
        profit_interval_score_90=90.0,
        profit_mae=44.0,
        profit_event_calibration_gap=0.04,
        profit_90_coverage=0.87,
    )
    _, decision = _promotion_from_summaries(
        baseline,
        candidate,
        minimum_improvement=0.02,
    )
    assert decision["promote_state_space"] is True
    assert decision["metrics_improved"] >= 4


def test_model_promotion_blocks_material_single_metric_regression():
    baseline = _summary()
    candidate = _summary(
        profit_brier=0.30,
        roas_target_brier=0.16,
        profit_ece=0.08,
        roas_ece=0.075,
        profit_interval_score_90=90.0,
        profit_mae=44.0,
        profit_event_calibration_gap=0.04,
        profit_90_coverage=0.87,
    )
    _, decision = _promotion_from_summaries(
        baseline,
        candidate,
        minimum_improvement=0.02,
    )
    assert decision["promote_state_space"] is False
    assert decision["no_material_metric_regression"] is False
