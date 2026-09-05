from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    score: float
    rows: int
    days: int
    campaigns: int
    adsets: int
    ads: int
    zero_spend_rows: int
    zero_impression_rows: int
    duplicate_entity_day_rows: int
    warnings: tuple[str, ...]


def assess_data_quality(df: pd.DataFrame) -> DataQualityReport:
    warnings: list[str] = []
    rows = len(df)
    days = int(pd.to_datetime(df["date"]).nunique())

    zero_spend = int((pd.to_numeric(df["spend"], errors="coerce").fillna(0) == 0).sum())
    zero_impr = int((pd.to_numeric(df["impressions"], errors="coerce").fillna(0) == 0).sum())

    key = ["date", "campaign_id", "adset_id", "ad_id"]
    dupes = int(df.duplicated(key, keep=False).sum()) if all(c in df for c in key) else 0

    penalty = 0.0
    if days < 7:
        warnings.append("Menos de 7 dias: tendências temporais têm baixa sustentação.")
        penalty += 20
    elif days < 14:
        warnings.append("Menos de 14 dias: derivadas e regime ainda têm incerteza alta.")
        penalty += 10

    if rows and zero_spend / rows > 0.10:
        warnings.append("Mais de 10% das linhas têm gasto zero.")
        penalty += 10
    if rows and zero_impr / rows > 0.05:
        warnings.append("Mais de 5% das linhas têm impressões zero.")
        penalty += 10
    if dupes:
        warnings.append(
            "Há múltiplas linhas por anúncio/dia; o motor agrega essas linhas, mas confirme se isso é esperado."
        )
        penalty += min(10, 100 * dupes / max(rows, 1))

    score = float(np.clip(100.0 - penalty, 0.0, 100.0))
    return DataQualityReport(
        score=score,
        rows=rows,
        days=days,
        campaigns=int(df["campaign_id"].nunique()),
        adsets=int(df["adset_id"].nunique()),
        ads=int(df["ad_id"].nunique()),
        zero_spend_rows=zero_spend,
        zero_impression_rows=zero_impr,
        duplicate_entity_day_rows=dupes,
        warnings=tuple(warnings),
    )
