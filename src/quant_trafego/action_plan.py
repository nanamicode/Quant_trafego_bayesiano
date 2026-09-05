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


def _capital_action_from_amounts(
    level: str,
    current: float,
    recommended: float,
    *,
    tolerance: float,
) -> str:
    current = max(float(current), 0.0)
    recommended = max(float(recommended), 0.0)

    if recommended <= 1e-9:
        return "DESLIGAR" if current > 1e-9 else "MANTER"

    if current <= 1e-9:
        return "PRIORIZAR_MAIS" if level == "ad" else "AUMENTAR"

    return _capital_action(
        level,
        recommended / current,
        tolerance=tolerance,
    )


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


def _configured_daily_budget(
    row: pd.Series,
    source_df: pd.DataFrame | None,
) -> float | None:
    level = str(row["level"])
    entity_id = str(row["entity_id"])
    if level == "campaign":
        return _latest_numeric_for_entity(
            source_df,
            level=level,
            entity_id=entity_id,
            column="campaign_daily_budget",
        )
    if level == "adset":
        return _latest_numeric_for_entity(
            source_df,
            level=level,
            entity_id=entity_id,
            column="adset_daily_budget",
        )
    return None


def _budget_reference(
    row: pd.Series,
    source_df: pd.DataFrame | None,
    *,
    recent_spend_days: int,
) -> tuple[float, str, bool]:
    """
    Operational baseline is actual recent delivery, not configured budget.

    The model forecasts spend; a Meta budget can be much larger than delivered
    spend and must not redefine what 1.0x means.
    """
    level = str(row["level"])
    entity_id = str(row["entity_id"])
    configured_budget = _configured_daily_budget(
        row,
        source_df,
    )

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
            configured_budget is not None,
        )

    if configured_budget is not None:
        return (
            configured_budget,
            (
                "campaign_daily_budget_fallback"
                if level == "campaign"
                else "adset_daily_budget_fallback"
            ),
            True,
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


def _campaign_operational_amounts(
    row: pd.Series,
    *,
    account_budget_target: dict | None,
    source_df: pd.DataFrame | None,
    recent_spend_days: int,
    horizon_days: int,
) -> tuple[float, float, str, bool, bool, float]:
    current_daily, amount_basis, direct_budget = _budget_reference(
        row,
        source_df,
        recent_spend_days=recent_spend_days,
    )

    expected_spend = max(
        float(
            row.get(
                "expected_spend",
                current_daily
                * float(row["action_multiplier"])
                * max(int(horizon_days), 1),
            )
        ),
        0.0,
    )
    recommended_daily = (
        expected_spend
        / max(int(horizon_days), 1)
    )
    reconciled = False
    share = np.nan

    parent_limit = pd.to_numeric(
        pd.Series(
            [
                row.get(
                    "parent_account_budget_limit",
                    np.nan,
                )
            ]
        ),
        errors="coerce",
    ).iloc[0]

    if (
        account_budget_target is not None
        and np.isfinite(parent_limit)
    ):
        share = (
            float(
                np.clip(
                    expected_spend
                    / float(parent_limit),
                    0.0,
                    1.0,
                )
            )
            if parent_limit > 1e-9
            else 0.0
        )
        reconciled = True

    return (
        current_daily,
        recommended_daily,
        amount_basis,
        direct_budget,
        reconciled,
        share,
    )


def derive_account_budget_target(
    best_actions: pd.DataFrame,
    *,
    source_df: pd.DataFrame | None = None,
    horizon_days: int = 7,
    recent_spend_days: int = 7,
) -> dict:
    """
    Translate the approved account action into an absolute capital envelope.

    Explicit campaign budgets are preferred. Otherwise recent observed account
    spend is used. The returned horizon amount is the budget constraint passed
    to the campaign portfolio optimizer.
    """
    account = best_actions[
        best_actions["level"] == "account"
    ]
    if account.empty:
        raise ValueError(
            "A decisão da conta é necessária para definir o capital total."
        )
    row = account.iloc[0]
    multiplier = float(
        row["action_multiplier"]
    )

    current_daily = None
    basis = None
    configured_budget_daily = None

    if source_df is not None:
        if (
            "campaign_daily_budget" in source_df.columns
            and "campaign_id" in source_df.columns
        ):
            tmp_budget = source_df.copy()
            if "date" in tmp_budget.columns:
                tmp_budget["date"] = pd.to_datetime(
                    tmp_budget["date"],
                    errors="coerce",
                )
                latest = tmp_budget["date"].max()
                if pd.notna(latest):
                    tmp_budget = tmp_budget[
                        tmp_budget["date"] == latest
                    ]
            tmp_budget["campaign_daily_budget"] = pd.to_numeric(
                tmp_budget["campaign_daily_budget"],
                errors="coerce",
            )
            budgets = (
                tmp_budget.dropna(
                    subset=[
                        "campaign_id",
                        "campaign_daily_budget",
                    ]
                )
                .groupby("campaign_id")[
                    "campaign_daily_budget"
                ]
                .median()
            )
            budgets = budgets[
                np.isfinite(budgets)
                & (budgets >= 0)
            ]
            if not budgets.empty:
                configured_budget_daily = float(
                    budgets.sum()
                )

        if (
            "date" in source_df.columns
            and "spend" in source_df.columns
        ):
            tmp = source_df.copy()
            tmp["date"] = pd.to_datetime(
                tmp["date"],
                errors="coerce",
            )
            tmp["spend"] = pd.to_numeric(
                tmp["spend"],
                errors="coerce",
            )
            tmp = tmp.dropna(
                subset=["date", "spend"]
            )
            if not tmp.empty:
                latest = tmp["date"].max()
                cutoff = latest - pd.Timedelta(
                    days=max(
                        int(recent_spend_days),
                        1,
                    ) - 1
                )
                recent = tmp[
                    tmp["date"] >= cutoff
                ]
                spend = float(
                    recent["spend"].sum()
                )
                current_daily = (
                    spend
                    / max(
                        int(recent_spend_days),
                        1,
                    )
                )
                basis = (
                    f"recent_{int(recent_spend_days)}d_account_daily_spend"
                )

    if current_daily is None and configured_budget_daily is not None:
        current_daily = configured_budget_daily
        basis = "configured_campaign_budget_fallback"

    if current_daily is None:
        historical_days = max(
            float(
                row.get(
                    "historical_days",
                    1.0,
                )
            ),
            1.0,
        )
        current_daily = (
            max(
                float(
                    row.get(
                        "historical_spend",
                        0.0,
                    )
                ),
                0.0,
            )
            / historical_days
        )
        basis = "historical_account_daily_spend"

    recommended_daily = (
        current_daily
        * multiplier
    )
    horizon = max(
        int(horizon_days),
        1,
    )

    return {
        "action_multiplier": multiplier,
        "amount_basis": basis,
        "configured_daily_budget": configured_budget_daily,
        "current_daily_amount": current_daily,
        "recommended_daily_amount": recommended_daily,
        "daily_amount_change": (
            recommended_daily
            - current_daily
        ),
        "horizon_days": horizon,
        "current_horizon_amount": (
            current_daily
            * horizon
        ),
        "recommended_horizon_amount": (
            recommended_daily
            * horizon
        ),
        "horizon_amount_change": (
            recommended_daily
            - current_daily
        )
        * horizon,
        "p_profit": float(
            row.get(
                "p_profit",
                np.nan,
            )
        ),
        "p_incremental_profit_positive": float(
            row.get(
                "p_incremental_profit_positive",
                np.nan,
            )
        ),
        "decision_score": float(
            row.get(
                "decision_score",
                np.nan,
            )
        ),
    }


def build_operational_action_plan(
    best_actions: pd.DataFrame,
    *,
    allocation: pd.DataFrame | None = None,
    adset_allocation: pd.DataFrame | None = None,
    account_budget_target: dict | None = None,
    source_df: pd.DataFrame | None = None,
    horizon_days: int = 7,
    config: OperationalPlanConfig | None = None,
) -> pd.DataFrame:
    """
    Convert posterior decisions into an execution-first plan.

    Campaign decisions from the constrained portfolio allocation replace
    independent campaign optima. Ad-set decisions can likewise be replaced by
    the nested allocation reconciled to each selected campaign budget.
    """
    cfg = config or OperationalPlanConfig()

    best = best_actions.copy()
    if best.empty:
        return pd.DataFrame()

    if account_budget_target is None:
        try:
            account_budget_target = derive_account_budget_target(
                best,
                source_df=source_df,
                horizon_days=horizon_days,
                recent_spend_days=cfg.recent_spend_days,
            )
        except ValueError:
            account_budget_target = None

    if allocation is not None and not allocation.empty:
        alloc = allocation.copy()
        alloc["entity_id"] = alloc["entity_id"].astype(str)
        campaign_best = best[
            best["level"] == "campaign"
        ].copy()
        campaign_best["entity_id"] = campaign_best["entity_id"].astype(str)

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

    if adset_allocation is not None and not adset_allocation.empty:
        alloc = adset_allocation.copy()
        alloc["entity_id"] = alloc["entity_id"].astype(str)
        adset_best = best[
            best["level"] == "adset"
        ].copy()
        adset_best["entity_id"] = adset_best["entity_id"].astype(str)

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
            if col in adset_best.columns
        ]
        if post_selection:
            meta = adset_best[
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
                    best["level"] != "adset"
                ],
                alloc,
            ],
            ignore_index=True,
            sort=False,
        )

    campaign_rows = best[
        best["level"] == "campaign"
    ].copy()
    campaign_rows["entity_id"] = (
        campaign_rows["entity_id"].astype(str)
    )
    campaign_multiplier_map = {
        str(row["entity_id"]): float(row["action_multiplier"])
        for _, row in campaign_rows.iterrows()
    }

    adset_rows = best[
        best["level"] == "adset"
    ].copy()
    adset_rows["entity_id"] = (
        adset_rows["entity_id"].astype(str)
    )
    adset_multiplier_map = {
        str(row["entity_id"]): float(row["action_multiplier"])
        for _, row in adset_rows.iterrows()
    }

    def selected_expected_spend(row: pd.Series) -> float:
        value = pd.to_numeric(
            pd.Series(
                [row.get("expected_spend", np.nan)]
            ),
            errors="coerce",
        ).iloc[0]
        if np.isfinite(value):
            return max(float(value), 0.0)

        current_daily, _, _ = _budget_reference(
            row,
            source_df,
            recent_spend_days=cfg.recent_spend_days,
        )
        return (
            max(float(current_daily), 0.0)
            * max(float(row["action_multiplier"]), 0.0)
            * max(int(horizon_days), 1)
        )

    if not campaign_rows.empty:
        campaign_rows["selected_expected_spend"] = (
            campaign_rows.apply(
                selected_expected_spend,
                axis=1,
            )
        )
    else:
        campaign_rows["selected_expected_spend"] = pd.Series(
            dtype=float
        )

    if not adset_rows.empty:
        adset_rows["selected_expected_spend"] = (
            adset_rows.apply(
                selected_expected_spend,
                axis=1,
            )
        )
    else:
        adset_rows["selected_expected_spend"] = pd.Series(
            dtype=float
        )

    account_capital_ceiling_daily = (
        float(
            account_budget_target[
                "recommended_daily_amount"
            ]
        )
        if account_budget_target is not None
        else np.nan
    )
    account_deployed_daily = float(
        campaign_rows["selected_expected_spend"].fillna(0.0).sum()
        / max(int(horizon_days), 1)
    )
    account_unallocated_daily = (
        max(
            account_capital_ceiling_daily
            - account_deployed_daily,
            0.0,
        )
        if np.isfinite(
            account_capital_ceiling_daily
        )
        else np.nan
    )

    campaign_selected_horizon_spend = {
        str(row["entity_id"]): max(
            float(row.get("selected_expected_spend", 0.0)),
            0.0,
        )
        for _, row in campaign_rows.iterrows()
    }
    adset_selected_horizon_spend = {
        str(row["entity_id"]): max(
            float(row.get("selected_expected_spend", 0.0)),
            0.0,
        )
        for _, row in adset_rows.iterrows()
    }
    adset_selected_by_campaign = (
        adset_rows.assign(
            campaign_id=adset_rows[
                "campaign_id"
            ].astype(str)
        )
        .groupby("campaign_id")[
            "selected_expected_spend"
        ]
        .sum()
        .to_dict()
        if (
            not adset_rows.empty
            and "campaign_id" in adset_rows.columns
        )
        else {}
    )

    campaign_daily_targets: dict[str, float] = {}
    for _, campaign_row in best[
        best["level"] == "campaign"
    ].iterrows():
        (
            _campaign_current,
            campaign_recommended,
            _campaign_basis,
            _campaign_direct,
            _campaign_reconciled,
            _campaign_share,
        ) = _campaign_operational_amounts(
            campaign_row,
            account_budget_target=account_budget_target,
            source_df=source_df,
            recent_spend_days=cfg.recent_spend_days,
            horizon_days=horizon_days,
        )
        campaign_daily_targets[
            str(campaign_row["entity_id"])
        ] = campaign_recommended

    records: list[dict] = []

    for _, row in best.iterrows():
        level = str(row["level"])
        if level == "account":
            continue

        multiplier = float(
            row["action_multiplier"]
        )
        account_budget_reconciled = False
        parent_account_recommended_daily = np.nan
        parent_account_spend_share = np.nan

        if level == "campaign":
            (
                current_daily,
                recommended_daily,
                amount_basis,
                direct_budget,
                account_budget_reconciled,
                parent_account_spend_share,
            ) = _campaign_operational_amounts(
                row,
                account_budget_target=account_budget_target,
                source_df=source_df,
                recent_spend_days=cfg.recent_spend_days,
                horizon_days=horizon_days,
            )
            if account_budget_target is not None:
                parent_account_recommended_daily = (
                    account_deployed_daily
                )
        else:
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

        nested_budget_reconciled = False
        parent_campaign_recommended_daily = np.nan
        parent_campaign_spend_share = np.nan

        if level == "adset":
            campaign_id = str(
                row.get("campaign_id", "")
            )
            raw_expected_spend = pd.to_numeric(
                pd.Series(
                    [row.get("expected_spend", np.nan)]
                ),
                errors="coerce",
            ).iloc[0]
            expected_spend = (
                max(float(raw_expected_spend), 0.0)
                if np.isfinite(raw_expected_spend)
                else adset_selected_horizon_spend.get(
                    str(row["entity_id"]),
                    0.0,
                )
            )
            parent_campaign_recommended_daily = (
                campaign_daily_targets.get(
                    campaign_id,
                    np.nan,
                )
            )
            parent_horizon = (
                campaign_selected_horizon_spend.get(
                    campaign_id,
                    np.nan,
                )
            )
            if np.isfinite(parent_horizon):
                parent_campaign_spend_share = (
                    float(
                        np.clip(
                            expected_spend
                            / parent_horizon,
                            0.0,
                            1.0,
                        )
                    )
                    if parent_horizon > 1e-9
                    else 0.0
                )
                nested_budget_reconciled = True

            recommended_daily = (
                expected_spend
                / max(int(horizon_days), 1)
            )

        model_suggested_action = _capital_action(
            level,
            multiplier,
            tolerance=cfg.hold_tolerance,
        )
        parent_campaign_action = None
        parent_adset_action = None
        blocked_by_parent = False

        if level in {"adset", "ad"}:
            campaign_key = str(
                row.get("campaign_id", "")
            )
            campaign_multiplier = (
                campaign_multiplier_map.get(
                    campaign_key
                )
            )
            if campaign_multiplier is not None:
                parent_campaign_action = _capital_action(
                    "campaign",
                    campaign_multiplier,
                    tolerance=cfg.hold_tolerance,
                )
                blocked_by_parent = (
                    campaign_multiplier <= 1e-9
                )

        if level == "ad":
            adset_key = str(
                row.get("adset_id", "")
            )
            adset_multiplier = (
                adset_multiplier_map.get(
                    adset_key
                )
            )
            if adset_multiplier is not None:
                parent_adset_action = _capital_action(
                    "adset",
                    adset_multiplier,
                    tolerance=cfg.hold_tolerance,
                )
                blocked_by_parent = (
                    blocked_by_parent
                    or adset_multiplier <= 1e-9
                )

            # Ads do not own a Meta budget. Keep current attributed spend as
            # context, but do not fabricate an executable R$/day target.
            recommended_daily = np.nan

        delta_daily = (
            recommended_daily
            - current_daily
            if np.isfinite(recommended_daily)
            else np.nan
        )
        current_horizon = (
            current_daily
            * int(horizon_days)
        )
        recommended_horizon = (
            recommended_daily
            * int(horizon_days)
        )

        if level == "ad":
            capital_action = (
                "BLOQUEADO_PELO_PAI"
                if blocked_by_parent
                else model_suggested_action
            )
            operational_multiplier = np.nan
        else:
            capital_action = _capital_action_from_amounts(
                level,
                current_daily,
                recommended_daily,
                tolerance=cfg.hold_tolerance,
            )
            operational_multiplier = (
                recommended_daily / current_daily
                if (
                    current_daily > 1e-9
                    and np.isfinite(
                        recommended_daily
                    )
                )
                else np.nan
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
        if blocked_by_parent:
            duplicate_action = "NAO"
            additional_copies = 0
            extra_capacity_fraction = 0.0
            duplicate_note = (
                "Duplicação bloqueada porque o nível pai não recebeu capital."
            )

        configured_daily_budget = (
            _configured_daily_budget(
                row,
                source_df,
            )
        )

        execution_note = ""
        if level == "ad":
            if blocked_by_parent:
                execution_note = (
                    "O modelo do anúncio é apenas diagnóstico: a campanha ou "
                    "o conjunto pai foi desligado pelo plano de capital, então "
                    "nenhum scale individual deve ser executado."
                )
            elif capital_action == "REDUZIR_EXPOSICAO":
                execution_note = (
                    "Anúncio não possui orçamento próprio. Reduza sua participação "
                    "dentro da capacidade aprovada do conjunto/campanha."
                )
            elif capital_action == "PRIORIZAR_MAIS":
                execution_note = (
                    "Priorize relativamente este anúncio dentro do orçamento "
                    "já aprovado para o conjunto/campanha; não interprete como "
                    "um novo orçamento independente."
                )
            else:
                execution_note = (
                    "Decisão de exposição relativa. O dinheiro executável fica "
                    "nos níveis campanha/conjunto."
                )
        elif nested_budget_reconciled:
            execution_note = (
                "Cenário selecionado pelo solver dentro do teto da campanha pai; "
                "o R$/dia corresponde exatamente ao spend usado nas métricas "
                "de lucro/risco desta linha."
            )
        elif account_budget_reconciled:
            execution_note = (
                "Cenário selecionado pelo portfólio dentro do teto de capital "
                "da conta; o R$/dia corresponde ao spend usado nas métricas."
            )
        elif amount_basis.startswith("recent_"):
            execution_note = (
                "Baseline = gasto diário recente efetivamente entregue. O orçamento "
                "configurado, quando disponível, é mostrado separadamente."
            )
        elif direct_budget:
            execution_note = (
                "Sem entrega recente suficiente; orçamento configurado usado apenas "
                "como fallback de baseline."
            )
        else:
            execution_note = (
                "Baseline histórica usada por falta de entrega recente suficiente."
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
                "model_suggested_action": model_suggested_action,
                "blocked_by_parent": blocked_by_parent,
                "parent_campaign_action": parent_campaign_action,
                "parent_adset_action": parent_adset_action,
                "action_multiplier": multiplier,
                "operational_amount_multiplier": operational_multiplier,
                "account_budget_reconciled": account_budget_reconciled,
                "parent_account_recommended_daily_amount": parent_account_recommended_daily,
                "parent_account_capital_ceiling_daily_amount": account_capital_ceiling_daily,
                "parent_account_unallocated_daily_amount": account_unallocated_daily,
                "parent_account_spend_share": parent_account_spend_share,
                "nested_budget_reconciled": nested_budget_reconciled,
                "parent_campaign_recommended_daily_amount": parent_campaign_recommended_daily,
                "parent_campaign_spend_share": parent_campaign_spend_share,
                "amount_basis": amount_basis,
                "configured_daily_budget": configured_daily_budget,
                "direct_budget_available": direct_budget,
                "current_daily_amount": current_daily,
                "recommended_daily_amount": recommended_daily,
                "daily_amount_change": delta_daily,
                "daily_amount_change_pct": (
                    delta_daily / current_daily
                    if current_daily > 1e-9
                    else np.nan
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
        "BLOQUEADO_PELO_PAI": 0,
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
