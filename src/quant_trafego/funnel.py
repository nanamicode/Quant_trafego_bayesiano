from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


CANONICAL_FUNNEL = (
    "impressions",
    "clicks",
    "landing_page_views",
    "adds_to_cart",
    "checkouts",
    "conversions",
)


@dataclass(frozen=True)
class FunnelSchema:
    available_stages: tuple[str, ...]
    transitions: tuple[tuple[str, str], ...]


def detect_funnel_schema(df: pd.DataFrame) -> FunnelSchema:
    stages = tuple(stage for stage in CANONICAL_FUNNEL if stage in df.columns)
    transitions = tuple(zip(stages[:-1], stages[1:]))
    return FunnelSchema(
        available_stages=stages,
        transitions=transitions,
    )


def _posterior_transition(successes: float, trials: float) -> dict:
    if trials <= 0 or successes < 0 or successes > trials:
        return {
            "posterior_mean": np.nan,
            "posterior_p05": np.nan,
            "posterior_p95": np.nan,
            "valid_binomial_transition": False,
        }
    alpha = 0.5 + successes
    beta = 0.5 + trials - successes
    return {
        "posterior_mean": float(alpha / (alpha + beta)),
        "posterior_p05": float(beta_dist.ppf(0.05, alpha, beta)),
        "posterior_p95": float(beta_dist.ppf(0.95, alpha, beta)),
        "valid_binomial_transition": True,
    }


def hierarchical_funnel_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    schema = detect_funnel_schema(df)
    if len(schema.transitions) == 0:
        return pd.DataFrame()

    levels = [
        ("account", [], "ALL"),
        ("campaign", ["campaign_id"], None),
        ("adset", ["campaign_id", "adset_id"], None),
        ("ad", ["campaign_id", "adset_id", "ad_id"], None),
    ]
    rows: list[dict] = []

    for level, keys, fixed_id in levels:
        groups = [(fixed_id, df)] if not keys else df.groupby(keys, sort=False)
        for group_key, group in groups:
            if keys:
                key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
                entity_id = str(key_tuple[-1])
            else:
                entity_id = str(group_key)

            for parent, child in schema.transitions:
                parent_values = pd.to_numeric(
                    group[parent], errors="coerce"
                ).fillna(0.0)
                child_values = pd.to_numeric(
                    group[child], errors="coerce"
                ).fillna(0.0)
                trials = float(parent_values.sum())
                successes = float(child_values.sum())
                violations = int((child_values > parent_values).sum())
                raw_rate = successes / trials if trials > 0 else np.nan
                posterior = _posterior_transition(successes, trials)

                rows.append(
                    {
                        "level": level,
                        "entity_id": entity_id,
                        "from_stage": parent,
                        "to_stage": child,
                        "trials": trials,
                        "successes": successes,
                        "raw_rate": raw_rate,
                        "tracking_violation_rows": violations,
                        **posterior,
                    }
                )

    return pd.DataFrame(rows)
