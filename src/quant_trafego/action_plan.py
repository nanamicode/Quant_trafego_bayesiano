from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OperationalPlanConfig:
    hold_tolerance: float = 0.05
    duplicate_min_multiplier: float = 1.50
    duplicate_min_p_profit: float = 0.80
    duplicate_min_p_incremental: float = 0.70
    duplicate_min_p_beats_hold: float = 0.65
    duplicate_min_response_confidence: float = 0.15
    recent_spend_days: int = 7


def _latest_numeric_for_entity(
    df: pd.DataFrame | None,
    *,
    level: str,
    entity_id: str,
    column: str,
) -> float | None:
    if df is None or column not in df.columns:
        return None

    id_col = {
        "campaign": "campaign_id",
        "adset": "adset_id",
        "ad": "ad_id",
    }.get(level)
    if id_col is None or id_col not in df.columns:
        return None

    subset = df[
        df[id_col].astype(str) == str(entity_id)
    ].copy()
    if subset.empty:
        return None

    if "date" in subset.columns:
        subset["date"] = pd.to_datetime(
            subset["date"],
            errors="coerce",
        )
        latest = subset["date"].max()
        if pd.notna(latest):
            subset = subset[
                subset["date"] == latest
            ]

    values = pd.to_numeric(
        subset[column],
        errors="coerce",
    )
    values = values[
        np.isfinite(values)
        & (values >= 0)
    ]
    if values.empty:
        return None

    # Budget columns are commonly repeated down ad rows. Median avoids
    # multiplying the same budget by the number of children.
    return float(values.median())


def _capital_action(
    level: str,
    multiplier: float,
    *,
    tolerance: float,
) -> str:
    if multiplier <= 1e-9:
        return "DESLIGAR"

    if multiplier < 1.0 - tolerance:
        if level == "ad":
            return "REDUZIR_EXPOSICAO"
        return "REDUZIR"

    if multiplier > 1.0 + tolerance:
        if level == "ad":
            return "PRIORIZAR_MAIS"
        return "AUMENTAR"

    return "MANTER"


def _duplicate_decision(
    row: pd.Series,
    cfg: OperationalPlanConfig,
) -> tuple[str, int, float, str]:
    level = str(row["level"])
    multiplier = float(row["action_multiplier"])

    if level == "account" or multiplier < cfg.duplicate_min_multiplier:
        return "NAO", 0, 0.0, "Sem escala suficiente para justificar duplicação."

    strong = (
        float(row.get("p_profit", 0.0))
        >= cfg.duplicate_min_p_profit
        and float(
            row.get(
                "p_incremental_profit_positive",
                0.0,
            )
        )
        >= cfg.duplicate_min_p_incremental
        and float(row.get("p_beats_hold", 0.0))
        >= cfg.duplicate_min_p_beats_hold
        and float(
            row.get(
                "response_confidence",
                0.0,
            )
        )
        >= cfg.duplicate_min_response_confidence
        and bool(row.get("policy_eligible", True))
    )

    if not strong:
        return (
            "NAO",
            0,
            0.0,
            "Scale-up ainda não possui evidência suficiente para recomendar clone.",
        )

    extra_capacity = max(
        multiplier - 1.0,
        0.0,
    )
    copies = max(
        1,
        int(np.ceil(extra_capacity)),
    )

    tier = str(
        row.get(
            "evidence_tier",
            "predictive",
        )
    )
    if tier == "experiment_calibrated":
        decision = "DUPLICAR"
        note = (
            "Resposta calibrada por experimento: duplicação pode ser usada "
            "como mecanismo de expansão, respeitando o orçamento recomendado."
        )
    else:
        decision = "TESTAR_DUPLICACAO"
        note = (
            "Há evidência de escala, mas a base não identifica que duplicar "
            "é superior a escalar o ativo existente. Trate o clone como teste "
            "e compare o resultado incremental."
        )

    return (
        decision,
        copies,
        float(extra_capacity),
        note,
    )


def _recent_daily_spend_for_entity(
    df: pd.DataFrame | None,
    *,
    level: str,
    entity_id: str,
    recent_days: int,
) -> float | None:
    if (
        df is None
        or "spend" not in df.columns
        or "date" not in df.columns
    ):
        return None

    id_col = {
        "campaign": "campaign_id",
        "adset": "adset_id",
        "ad": "ad_id",
    }.get(level)
    if id_col is None or id_col not in df.columns:
        return None

    subset = df[
        df[id_col].astype(str) == str(entity_id)
    ].copy()
    if subset.empty:
        return None

    subset["date"] = pd.to_datetime(
        subset["date"],
        errors="coerce",
    )
    subset["spend"] = pd.to_numeric(
        subset["spend"],
        errors="coerce",
    )
    subset = subset.dropna(
        subset=["date", "spend"]
    )
    if subset.empty:
        return None

    daily = (
        subset.groupby(
            "date",
            as_index=False,
        )["spend"]
        .sum()
        .sort_values("date")
    )
    daily = daily.tail(
        max(int(recent_days), 1)
    )
    if daily.empty:
        return None
    return float(
        daily["spend"].mean()
    )


def _budget_reference(
    row: pd.Series,
    source_df: pd.DataFrame | None,
    *,
    recent_spend_days: int,
) -> tuple[float, str, bool]:
    level = str(row["level"])
    entity_id = str(row["entity_id"])

    if level == "campaign":
        budget = _latest_numeric_for_entity(
            source_df,
            level=level,
            entity_id=entity_id,
            column="campaign_daily_budget",
        )
        if budget is not None:
            return budget, "campaign_daily_budget", True

    if level == "adset":
        budget = _latest_numeric_for_entity(
            source_df,
            level=level,
            entity_id=entity_id,
            column="adset_daily_budget",
        )
        if budget is not None:
            return budget, "adset_daily_budget", True

    recent_spend = _recent_daily_spend_for_entity(
        source_df,
        level=level,
        entity_id=entity_id,
        recent_days=recent_spend_days,
    )
    if recent_spend is not None:
        return (
            recent_spend,
            (
                f"recent_{int(recent_spend_days)}d_attributed_daily_spend"
                if level == "ad"
                else f"recent_{int(recent_spend_days)}d_avg_daily_spend"
            ),
            False,
        )

    historical_days = max(
        float(row.get("historical_days", 1.0)),
        1.0,
    )
    historical_spend = max(
        float(row.get("historical_spend", 0.0)),
        0.0,
    )
    return (
        historical_spend / historical_days,
        (
            "historical_attributed_avg_daily_spend"
            if level == "ad"
            else "historical_avg_daily_spend"
        ),
        False,
    )


def build_operational_action_plan(
    best_actions: pd.DataFrame,
    *,
    allocation: pd.DataFrame | None = None,
    source_df: pd.DataFrame | None = None,
    horizon_days: int = 7,
    config: OperationalPlanConfig | None = None,
) -> pd.DataFrame:
    """
    Convert posterior decisions into an execution-first plan.

    Campaign decisions from the constrained portfolio allocation replace
    independent campaign optima when an allocation is supplied. Lower
    hierarchy levels retain their own diagnostic recommendations.
    """
    cfg = config or OperationalPlanConfig()

    best = best_actions.copy()
    if best.empty:
        return pd.DataFrame()

    if allocation is not None and not allocation.empty:
        alloc = allocation.copy()
        campaign_best = best[
            best["level"] == "campaign"
        ].copy()

        # Keep columns computed only after best-action selection, while
        # replacing the selected campaign action and posterior metrics by the
        # globally constrained allocation.
        post_selection = [
            col
            for col in [
                "unconstrained_best_multiplier",
                "unconstrained_best_utility",
                "policy_constrained",
                "policy_utility_gap",
                "decision_score_raw",
                "decision_score",
                "decision_score_kind",
                "decision_confidence_raw",
                "decision_confidence",
                "opportunity_score",
            ]
            if col in campaign_best.columns
        ]
        if post_selection:
            meta = campaign_best[
                ["entity_id", *post_selection]
            ].drop_duplicates("entity_id")
            alloc = alloc.drop(
                columns=[
                    col
                    for col in post_selection
                    if col in alloc.columns
                ],
                errors="ignore",
            ).merge(
                meta,
                on="entity_id",
                how="left",
            )

        best = pd.concat(
            [
                best[
                    best["level"] != "campaign"
                ],
                alloc,
            ],
            ignore_index=True,
            sort=False,
        )

    records: list[dict] = []

    for _, row in best.iterrows():
        level = str(row["level"])
        if level == "account":
            continue

        multiplier = float(
            row["action_multiplier"]
        )
        current_daily, amount_basis, direct_budget = (
            _budget_reference(
                row,
                source_df,
                recent_spend_days=cfg.recent_spend_days,
            )
        )

        recommended_daily = (
            current_daily * multiplier
        )
        delta_daily = (
            recommended_daily
            - current_daily
        )
        current_horizon = (
            current_daily
            * int(horizon_days)
        )
        recommended_horizon = (
            recommended_daily
            * int(horizon_days)
        )

        capital_action = _capital_action(
            level,
            multiplier,
            tolerance=cfg.hold_tolerance,
        )
        (
            duplicate_action,
            additional_copies,
            extra_capacity_fraction,
            duplicate_note,
        ) = _duplicate_decision(
            row,
            cfg,
        )

        execution_note = ""
        if level == "ad":
            if capital_action == "REDUZIR_EXPOSICAO":
                execution_note = (
                    "Anúncio não possui orçamento próprio no Meta. Reduza sua "
                    "exposição via conjunto/campanha ou desligue se a estrutura "
                    "não permitir controle de entrega."
                )
            elif capital_action == "PRIORIZAR_MAIS":
                execution_note = (
                    "Anúncio não possui orçamento próprio. Direcione mais "
                    "capacidade ao conjunto/campanha que o contém ou use "
                    "duplicação apenas conforme o campo de duplicação."
                )
        elif direct_budget:
            execution_note = (
                "Valor calculado sobre orçamento diário informado na planilha."
            )
        else:
            execution_note = (
                "Valor calculado sobre gasto diário médio observado; trate como "
                "meta de spend se a planilha não contiver orçamento diário."
            )

        expected_incremental_profit = float(
            row.get(
                "expected_incremental_profit_vs_hold",
                0.0,
            )
        )
        expected_incremental_revenue = float(
            row.get(
                "expected_incremental_revenue_vs_hold",
                0.0,
            )
        )
        p_incremental = float(
            row.get(
                "p_incremental_profit_positive",
                0.0,
            )
        )

        priority_score = (
            abs(
                expected_incremental_profit
            )
            * (
                0.35
                + 0.65 * p_incremental
            )
        )

        records.append(
            {
                "level": level,
                "campaign_id": row.get("campaign_id"),
                "campaign_name": row.get("campaign_name"),
                "adset_id": row.get("adset_id"),
                "adset_name": row.get("adset_name"),
                "ad_id": row.get("ad_id"),
                "ad_name": row.get("ad_name"),
                "entity_id": str(row["entity_id"]),
                "capital_action": capital_action,
                "action_multiplier": multiplier,
                "amount_basis": amount_basis,
                "direct_budget_available": direct_budget,
                "current_daily_amount": current_daily,
                "recommended_daily_amount": recommended_daily,
                "daily_amount_change": delta_daily,
                "daily_amount_change_pct": (
                    multiplier - 1.0
                ),
                "current_horizon_amount": current_horizon,
                "recommended_horizon_amount": recommended_horizon,
                "horizon_amount_change": (
                    recommended_horizon
                    - current_horizon
                ),
                "horizon_days": int(horizon_days),
                "duplicate_action": duplicate_action,
                "suggested_additional_copies": additional_copies,
                "duplicate_extra_capacity_fraction": extra_capacity_fraction,
                "duplicate_note": duplicate_note,
                "expected_incremental_profit": expected_incremental_profit,
                "expected_incremental_revenue": expected_incremental_revenue,
                "expected_profit": float(
                    row.get(
                        "expected_profit",
                        0.0,
                    )
                ),
                "expected_revenue": float(
                    row.get(
                        "expected_revenue",
                        0.0,
                    )
                ),
                "p_profit": float(
                    row.get(
                        "p_profit",
                        0.0,
                    )
                ),
                "p_roas_target": float(
                    row.get(
                        "p_roas_target",
                        0.0,
                    )
                ),
                "p_beats_hold": float(
                    row.get(
                        "p_beats_hold",
                        0.0,
                    )
                ),
                "p_incremental_profit_positive": p_incremental,
                "p_action_optimal": float(
                    row.get(
                        "p_action_optimal",
                        0.0,
                    )
                ),
                "cvar10_profit": float(
                    row.get(
                        "cvar10_profit",
                        0.0,
                    )
                ),
                "expected_regret": float(
                    row.get(
                        "expected_regret",
                        0.0,
                    )
                ),
                "evidence_tier": row.get("evidence_tier"),
                "policy_eligible": bool(
                    row.get(
                        "policy_eligible",
                        True,
                    )
                ),
                "policy_constrained": bool(
                    row.get(
                        "policy_constrained",
                        False,
                    )
                ),
                "data_quality_score": float(
                    row.get(
                        "data_quality_score",
                        np.nan,
                    )
                ),
                "decision_score": float(
                    row.get(
                        "decision_score",
                        np.nan,
                    )
                ),
                "execution_priority_score": priority_score,
                "execution_note": execution_note,
            }
        )

    plan = pd.DataFrame(records)
    if plan.empty:
        return plan

    action_rank = {
        "DESLIGAR": 0,
        "REDUZIR": 1,
        "REDUZIR_EXPOSICAO": 1,
        "AUMENTAR": 2,
        "PRIORIZAR_MAIS": 2,
        "MANTER": 3,
    }
    plan["_action_rank"] = (
        plan["capital_action"]
        .map(action_rank)
        .fillna(9)
    )
    plan = (
        plan.sort_values(
            [
                "_action_rank",
                "execution_priority_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop(
            columns="_action_rank"
        )
        .reset_index(drop=True)
    )
    return plan



def write_operational_action_plan(
    plan: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plan.to_csv(
        out / "operational_action_plan.csv",
        index=False,
    )

    if plan.empty:
        (
            out / "operational_action_plan.md"
        ).write_text(
            "# Plano operacional\n\nNenhuma ação disponível.\n",
            encoding="utf-8",
        )
        return

    lines = [
        "# Plano operacional",
        "",
        "Este arquivo traduz a inferência quantitativa em ações executáveis.",
        "",
    ]
    for action in [
        "DESLIGAR",
        "REDUZIR",
        "REDUZIR_EXPOSICAO",
        "AUMENTAR",
        "PRIORIZAR_MAIS",
        "MANTER",
    ]:
        subset = plan[
            plan["capital_action"] == action
        ]
        if subset.empty:
            continue

        lines.append(f"## {action}")
        lines.append("")
        for _, row in subset.iterrows():
            name = (
                row.get("ad_name")
                or row.get("adset_name")
                or row.get("campaign_name")
                or row.get("entity_id")
            )
            lines.append(
                "- "
                f"{row['level']} | {name} | "
                f"{row['current_daily_amount']:.2f}/dia → "
                f"{row['recommended_daily_amount']:.2f}/dia | "
                f"Δ {row['daily_amount_change']:+.2f}/dia | "
                f"Δ lucro esperado {row['expected_incremental_profit']:+.2f} | "
                f"P(ganho incremental) "
                f"{row['p_incremental_profit_positive']:.1%} | "
                f"duplicação: {row['duplicate_action']}"
            )
        lines.append("")

    (
        out / "operational_action_plan.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
