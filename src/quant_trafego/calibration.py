from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_table(
    probabilities,
    outcomes,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    valid = np.isfinite(p) & np.isfinite(y)
    p = np.clip(p[valid], 0.0, 1.0)
    y = y[valid]

    if len(p) == 0:
        return pd.DataFrame(
            columns=[
                "bin",
                "lower",
                "upper",
                "count",
                "mean_predicted",
                "observed_frequency",
                "calibration_gap",
                "absolute_gap",
            ]
        )

    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(
        np.digitize(p, edges[1:-1], right=False),
        0,
        bins - 1,
    )
    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        pred = float(p[mask].mean())
        obs = float(y[mask].mean())
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": int(mask.sum()),
                "mean_predicted": pred,
                "observed_frequency": obs,
                "calibration_gap": pred - obs,
                "absolute_gap": abs(pred - obs),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    probabilities,
    outcomes,
    *,
    bins: int = 10,
) -> float:
    table = calibration_table(
        probabilities,
        outcomes,
        bins=bins,
    )
    if table.empty:
        return float("nan")
    weights = table["count"].to_numpy(dtype=float)
    gaps = table["absolute_gap"].to_numpy(dtype=float)
    return float(np.average(gaps, weights=weights))
