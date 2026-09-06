from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


_CORE_COLUMNS = (
    "date",
    "campaign_id",
    "adset_id",
    "ad_id",
    "impressions",
    "clicks",
    "conversions",
    "spend",
    "revenue",
)
_NUMERIC_COLUMNS = (
    "impressions",
    "clicks",
    "conversions",
    "spend",
    "revenue",
)


@dataclass(frozen=True)
class DataQualityReport:
    score: float
    rows: int
    days: int
    calendar_span_days: int
    calendar_coverage_ratio: float
    campaigns: int
    adsets: int
    ads: int
    median_campaign_active_days: float
    delivery_window_rows: int
    zero_spend_rows: int
    zero_impression_rows: int
    duplicate_entity_day_rows: int
    invalid_date_rows: int
    invalid_numeric_rows: int
    negative_numeric_rows: int
    funnel_tracking_violation_rows: int
    low_overlap_campaign_pair_fraction: float
    missing_core_columns: tuple[str, ...]
    warnings: tuple[str, ...]


def _numeric_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    numeric = pd.DataFrame(index=df.index)
    invalid_mask = np.zeros(len(df), dtype=bool)

    for col in _NUMERIC_COLUMNS:
        if col not in df.columns:
            numeric[col] = np.nan
            invalid_mask[:] = True
            continue

        raw = df[col]
        parsed = pd.to_numeric(
            raw,
            errors="coerce",
        )
        invalid = (
            parsed.isna()
            & raw.notna()
            & raw.astype(str).str.strip().ne("")
        )
        invalid_mask |= invalid.to_numpy()
        numeric[col] = parsed

    return numeric, int(invalid_mask.sum())


def _campaign_overlap_fraction(
    df: pd.DataFrame,
    valid_dates: pd.Series,
    delivery_mask: pd.Series | None = None,
) -> tuple[float, float]:
    if (
        "campaign_id" not in df.columns
        or valid_dates.notna().sum() == 0
    ):
        return 0.0, 0.0

    panel = pd.DataFrame(
        {
            "date": valid_dates,
            "campaign_id": df["campaign_id"].astype(str),
            "delivering": (
                delivery_mask.astype(bool)
                if delivery_mask is not None
                else True
            ),
        }
    ).dropna(subset=["date"])
    panel = panel[
        panel["delivering"]
    ].copy()

    if panel.empty:
        return 0.0, 0.0

    active_days = (
        panel.groupby("campaign_id")["date"]
        .nunique()
        .astype(float)
    )
    median_active = float(
        active_days.median()
        if len(active_days)
        else 0.0
    )

    campaigns = active_days.index.tolist()
    if len(campaigns) < 2:
        return median_active, 0.0

    presence = pd.crosstab(
        panel["date"],
        panel["campaign_id"],
    )
    presence = (
        presence.reindex(columns=campaigns)
        .gt(0)
        .astype(np.int16)
    )
    matrix = presence.to_numpy(dtype=np.int16)
    overlap = matrix.T @ matrix

    upper = overlap[
        np.triu_indices(
            len(campaigns),
            k=1,
        )
    ]
    if len(upper) == 0:
        return median_active, 0.0

    low_fraction = float(
        np.mean(upper < 5)
    )
    return median_active, low_fraction


def assess_data_quality(
    df: pd.DataFrame,
) -> DataQualityReport:
    warnings: list[str] = []
    rows = int(len(df))
    missing = tuple(
        col
        for col in _CORE_COLUMNS
        if col not in df.columns
    )

    penalty = 0.0
    if missing:
        warnings.append(
            "Colunas obrigatórias ausentes: "
            + ", ".join(missing)
            + "."
        )
        penalty += min(
            70.0,
            15.0 * len(missing),
        )

    if "date" in df.columns:
        parsed_dates = pd.to_datetime(
            df["date"],
            errors="coerce",
        )
    else:
        parsed_dates = pd.Series(
            pd.NaT,
            index=df.index,
            dtype="datetime64[ns]",
        )

    invalid_date_rows = int(
        parsed_dates.isna().sum()
    )
    valid_unique_dates = (
        parsed_dates.dropna().drop_duplicates()
    )
    days = int(len(valid_unique_dates))

    if days:
        min_date = valid_unique_dates.min()
        max_date = valid_unique_dates.max()
        calendar_span_days = int(
            (max_date - min_date).days + 1
        )
        calendar_coverage_ratio = float(
            days / max(calendar_span_days, 1)
        )
    else:
        calendar_span_days = 0
        calendar_coverage_ratio = 0.0

    numeric, invalid_numeric_rows = _numeric_frame(
        df
    )
    negative_mask = (
        numeric.lt(0).any(axis=1)
    )
    negative_numeric_rows = int(
        negative_mask.sum()
    )

    delivering_now = (
        numeric["spend"].fillna(0).gt(0)
        | numeric["impressions"].fillna(0).gt(0)
    )

    # Meta exports frequently materialize a complete ad×day panel, including
    # structural zero rows before an ad starts delivering. Quality penalties
    # must be computed only inside each ad's observed delivery window.
    if (
        "ad_id" in df.columns
        and parsed_dates.notna().any()
    ):
        delivery_frame = pd.DataFrame(
            {
                "ad_id": df["ad_id"].astype(str),
                "date": parsed_dates,
                "delivering": delivering_now.to_numpy(dtype=bool),
            },
            index=df.index,
        )
        delivered_dates = delivery_frame[
            delivery_frame["delivering"]
            & delivery_frame["date"].notna()
        ]
        if delivered_dates.empty:
            delivery_window_mask = pd.Series(
                False,
                index=df.index,
            )
        else:
            bounds = (
                delivered_dates.groupby("ad_id")["date"]
                .agg(["min", "max"])
            )
            first = delivery_frame["ad_id"].map(
                bounds["min"]
            )
            last = delivery_frame["ad_id"].map(
                bounds["max"]
            )
            delivery_window_mask = (
                delivery_frame["date"].notna()
                & first.notna()
                & last.notna()
                & (delivery_frame["date"] >= first)
                & (delivery_frame["date"] <= last)
            )
    else:
        delivery_window_mask = pd.Series(
            True,
            index=df.index,
        )

    delivery_window_rows = int(
        delivery_window_mask.sum()
    )
    zero_spend = int(
        (
            delivery_window_mask
            & numeric["spend"].fillna(0).eq(0)
        ).sum()
    )
    zero_impr = int(
        (
            delivery_window_mask
            & numeric["impressions"].fillna(0).eq(0)
        ).sum()
    )

    # Meta Ads attribution is not a same-day literal funnel:
    # purchases can be credited today for earlier clicks or view-through
    # exposure. Daily conversions > daily clicks is therefore not itself a
    # tracking violation. Clicks > impressions is impossible, while
    # conversions > clicks is checked over the ad's full observed history.
    impossible_click_mask = (
        numeric["clicks"]
        > numeric["impressions"]
    ).fillna(False)
    impossible_click_rows = int(
        impossible_click_mask.sum()
    )

    aggregate_invalid_ads: set[str] = set()
    if "ad_id" in df.columns:
        aggregate_check = pd.DataFrame(
            {
                "ad_id": df["ad_id"].astype(str),
                "clicks": numeric["clicks"].fillna(0.0),
                "conversions": numeric["conversions"].fillna(0.0),
            }
        ).groupby("ad_id", as_index=False).sum()
        aggregate_invalid_ads = set(
            aggregate_check.loc[
                aggregate_check["conversions"]
                > aggregate_check["clicks"],
                "ad_id",
            ].astype(str)
        )

        represented_ads = set(
            df.loc[
                impossible_click_mask,
                "ad_id",
            ].astype(str)
        )
        aggregate_only_count = len(
            aggregate_invalid_ads
            - represented_ads
        )
    else:
        total_clicks = float(
            numeric["clicks"].fillna(0.0).sum()
        )
        total_conversions = float(
            numeric["conversions"].fillna(0.0).sum()
        )
        aggregate_only_count = int(
            total_conversions > total_clicks
            and impossible_click_rows == 0
        )

    funnel_tracking_violation_rows = (
        impossible_click_rows
        + aggregate_only_count
    )

    key = [
        "date",
        "campaign_id",
        "adset_id",
        "ad_id",
    ]
    dupes = (
        int(
            df.duplicated(
                key,
                keep=False,
            ).sum()
        )
        if all(c in df.columns for c in key)
        else 0
    )

    median_campaign_active_days, low_overlap_fraction = (
        _campaign_overlap_fraction(
            df,
            parsed_dates,
            delivering_now,
        )
    )

    def nunique(col: str) -> int:
        return (
            int(df[col].nunique(dropna=True))
            if col in df.columns
            else 0
        )

    campaigns = nunique("campaign_id")
    adsets = nunique("adset_id")
    ads = nunique("ad_id")

    if invalid_date_rows:
        warnings.append(
            f"{invalid_date_rows} linhas possuem data inválida/ausente."
        )
        penalty += min(
            20.0,
            5.0
            + 30.0
            * invalid_date_rows
            / max(rows, 1),
        )

    if invalid_numeric_rows:
        warnings.append(
            f"{invalid_numeric_rows} linhas possuem valor numérico inválido."
        )
        penalty += min(
            20.0,
            5.0
            + 30.0
            * invalid_numeric_rows
            / max(rows, 1),
        )

    if negative_numeric_rows:
        warnings.append(
            f"{negative_numeric_rows} linhas possuem métricas negativas."
        )
        penalty += min(
            25.0,
            10.0
            + 40.0
            * negative_numeric_rows
            / max(rows, 1),
        )

    if funnel_tracking_violation_rows:
        warnings.append(
            f"{funnel_tracking_violation_rows} anomalias de tracking foram "
            "detectadas (cliques > impressões ou conversões agregadas > cliques)."
        )
        penalty += min(
            30.0,
            10.0
            + 50.0
            * funnel_tracking_violation_rows
            / max(rows, 1),
        )

    if days < 7:
        warnings.append(
            "Menos de 7 dias: tendências temporais têm baixa sustentação."
        )
        penalty += 20
    elif days < 14:
        warnings.append(
            "Menos de 14 dias: derivadas e regime ainda têm incerteza alta."
        )
        penalty += 10

    if (
        calendar_span_days >= 14
        and calendar_coverage_ratio < 0.70
    ):
        warnings.append(
            "A série cobre menos de 70% dos dias do intervalo calendário; "
            "gaps podem degradar tendência e backtesting."
        )
        penalty += (
            15
            if calendar_coverage_ratio < 0.40
            else 8
        )

    if (
        delivery_window_rows
        and zero_spend / delivery_window_rows > 0.10
    ):
        warnings.append(
            "Mais de 10% das linhas dentro da janela real de entrega têm gasto zero."
        )
        penalty += 10

    if (
        delivery_window_rows
        and zero_impr / delivery_window_rows > 0.05
    ):
        warnings.append(
            "Mais de 5% das linhas dentro da janela real de entrega têm impressões zero."
        )
        penalty += 10

    if dupes:
        warnings.append(
            "Há múltiplas linhas por anúncio/dia; o motor agrega essas "
            "linhas, mas confirme se isso é esperado."
        )
        penalty += min(
            10.0,
            100.0 * dupes / max(rows, 1),
        )

    if (
        campaigns >= 2
        and median_campaign_active_days < 7
    ):
        warnings.append(
            "A campanha mediana possui menos de 7 dias ativos; "
            "efeitos por campanha terão forte shrinkage."
        )
        penalty += 7

    if (
        campaigns >= 2
        and low_overlap_fraction > 0.50
    ):
        warnings.append(
            "Mais da metade dos pares de campanhas possui menos de "
            "5 dias de sobreposição; correlação de portfólio será "
            "fortemente encolhida."
        )

    score = float(
        np.clip(
            100.0 - penalty,
            0.0,
            100.0,
        )
    )

    return DataQualityReport(
        score=score,
        rows=rows,
        days=days,
        calendar_span_days=calendar_span_days,
        calendar_coverage_ratio=calendar_coverage_ratio,
        campaigns=campaigns,
        adsets=adsets,
        ads=ads,
        median_campaign_active_days=median_campaign_active_days,
        delivery_window_rows=delivery_window_rows,
        zero_spend_rows=zero_spend,
        zero_impression_rows=zero_impr,
        duplicate_entity_day_rows=dupes,
        invalid_date_rows=invalid_date_rows,
        invalid_numeric_rows=invalid_numeric_rows,
        negative_numeric_rows=negative_numeric_rows,
        funnel_tracking_violation_rows=funnel_tracking_violation_rows,
        low_overlap_campaign_pair_fraction=low_overlap_fraction,
        missing_core_columns=missing,
        warnings=tuple(warnings),
    )
