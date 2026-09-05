from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class AllocationConfig:
    objective_column: str = "risk_adjusted_utility"
    min_p_profit_for_scale: float = 0.60
    min_p_incremental_for_scale: float = 0.55
    predictive_max_multiplier: float = 1.20
    observational_max_multiplier: float = 1.50
    experiment_max_multiplier: float = 2.00
    max_total_downside_proxy: float | None = None
    time_limit_seconds: float = 30.0
    revenue_tiebreak: bool = True
    revenue_tiebreak_tolerance: float = 0.02


def infer_evidence_tier(
    row: pd.Series,
    override: str | None = None,
) -> str:
    if override in {
        "predictive",
        "observational_intervention",
        "experiment_calibrated",
    }:
        return override
    if float(row.get("response_confidence", 0.0)) >= 0.15:
        return "observational_intervention"
    return "predictive"


def _max_multiplier(tier: str, cfg: AllocationConfig) -> float:
    if tier == "experiment_calibrated":
        return cfg.experiment_max_multiplier
    if tier == "observational_intervention":
        return cfg.observational_max_multiplier
    return cfg.predictive_max_multiplier


def optimize_campaign_allocation(
    all_actions: pd.DataFrame,
    *,
    total_budget: float | None = None,
    config: AllocationConfig | None = None,
    evidence_overrides: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    cfg = config or AllocationConfig()
    actions = all_actions[all_actions["level"] == "campaign"].copy()
    if actions.empty:
        raise ValueError("Não há ações de campanha para otimizar.")

    original_campaigns = actions["entity_id"].astype(str).unique().tolist()
    evidence_overrides = evidence_overrides or {}
    actions["evidence_tier"] = actions.apply(
        lambda row: infer_evidence_tier(
            row,
            evidence_overrides.get(str(row["entity_id"])),
        ),
        axis=1,
    )
    actions["max_allowed_multiplier"] = actions["evidence_tier"].map(
        lambda tier: _max_multiplier(tier, cfg)
    )

    eligible = (
        actions["action_multiplier"]
        <= actions["max_allowed_multiplier"] + 1e-12
    )
    scale = actions["action_multiplier"] > 1.0
    eligible &= (~scale) | (
        (actions["p_profit"] >= cfg.min_p_profit_for_scale)
        & (
            actions["p_incremental_profit_positive"]
            >= cfg.min_p_incremental_for_scale
        )
    )
    if "policy_eligible" in actions.columns:
        eligible &= (
            actions["policy_eligible"]
            .fillna(False)
            .astype(bool)
        )

    actions = actions[eligible].copy().reset_index(drop=True)

    campaigns = actions["entity_id"].astype(str).unique().tolist()
    if not campaigns:
        raise ValueError("Nenhuma ação elegível após restrições de evidência.")

    missing_campaigns = sorted(set(original_campaigns) - set(campaigns))
    if missing_campaigns:
        raise ValueError(
            "Campanhas ficaram sem ação elegível: "
            + ", ".join(missing_campaigns)
        )

    if total_budget is None:
        holds = actions[np.isclose(actions["action_multiplier"], 1.0)]
        if len(holds["entity_id"].unique()) == len(campaigns):
            total_budget = float(holds["expected_spend"].sum())
        else:
            total_budget = float(
                actions.groupby("entity_id")["expected_spend"].median().sum()
            )
    total_budget = max(float(total_budget), 0.0)

    n = len(actions)
    campaign_index = {c: i for i, c in enumerate(campaigns)}
    aeq = lil_matrix((len(campaigns), n), dtype=float)
    for j, row in actions.iterrows():
        aeq[
            campaign_index[str(row["entity_id"])],
            j,
        ] = 1.0

    constraints = [
        LinearConstraint(aeq.tocsr(), np.ones(len(campaigns)), np.ones(len(campaigns))),
        LinearConstraint(
            actions["expected_spend"].to_numpy(dtype=float)[None, :],
            -np.inf,
            total_budget,
        ),
    ]

    if cfg.max_total_downside_proxy is not None:
        downside = np.maximum(
            0.0,
            -actions["cvar10_profit"].to_numpy(dtype=float),
        )
        constraints.append(
            LinearConstraint(
                downside[None, :],
                -np.inf,
                float(cfg.max_total_downside_proxy),
            )
        )

    objective = actions[cfg.objective_column].to_numpy(dtype=float)
    primary_c = -objective
    result = milp(
        c=primary_c,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraints,
        options={"time_limit": cfg.time_limit_seconds},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"Otimização inteira não encontrou solução: {result.message}"
        )

    primary_result = result
    primary_optimum = float(result.fun)

    if (
        cfg.revenue_tiebreak
        and "expected_revenue" in actions.columns
        and cfg.revenue_tiebreak_tolerance >= 0
    ):
        allowed_deterioration = (
            float(cfg.revenue_tiebreak_tolerance)
            * max(abs(primary_optimum), 1.0)
        )
        secondary_constraints = [
            *constraints,
            LinearConstraint(
                primary_c[None, :],
                -np.inf,
                primary_optimum + allowed_deterioration,
            ),
        ]
        revenue_c = -actions["expected_revenue"].to_numpy(dtype=float)
        secondary = milp(
            c=revenue_c,
            integrality=np.ones(n, dtype=int),
            bounds=Bounds(np.zeros(n), np.ones(n)),
            constraints=secondary_constraints,
            options={"time_limit": cfg.time_limit_seconds},
        )
        if secondary.success and secondary.x is not None:
            result = secondary

    chosen = actions[np.asarray(result.x) > 0.5].copy()
    chosen["parent_account_budget_limit"] = float(total_budget)
    chosen["allocation_source"] = "account_capital_envelope"
    chosen = chosen.sort_values("entity_id").reset_index(drop=True)

    summary = {
        "solver": "scipy_milp_highs",
        "n_campaigns": len(campaigns),
        "total_budget_limit": total_budget,
        "selected_spend": float(chosen["expected_spend"].sum()),
        "expected_portfolio_profit_additive": float(
            chosen["expected_profit"].sum()
        ),
        "risk_adjusted_objective": float(
            chosen[cfg.objective_column].sum()
        ),
        "primary_risk_objective_optimum": float(-primary_optimum),
        "expected_portfolio_revenue_additive": float(
            chosen["expected_revenue"].sum()
        ) if "expected_revenue" in chosen.columns else None,
        "revenue_tiebreak": bool(cfg.revenue_tiebreak),
        "revenue_tiebreak_tolerance": float(cfg.revenue_tiebreak_tolerance),
        "additive_cvar10_proxy": float(
            chosen["cvar10_profit"].sum()
        ),
        "expected_regret_additive": float(
            chosen["expected_regret"].sum()
        ),
        "important_limitation": (
            "A distribuição conjunta entre campanhas ainda não modela "
            "covariância/canibalização; CVaR agregado é apenas proxy aditivo."
        ),
    }
    return chosen, summary



def optimize_adset_allocation(
    all_actions: pd.DataFrame,
    campaign_allocation: pd.DataFrame,
    *,
    config: AllocationConfig | None = None,
    evidence_overrides: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Reconcile ad-set actions with the budget selected for each parent campaign.

    One action is chosen per ad set and the sum of expected spend cannot exceed
    the selected campaign spend for the same forecast horizon.
    """
    cfg = config or AllocationConfig()
    evidence_overrides = evidence_overrides or {}

    if campaign_allocation is None or campaign_allocation.empty:
        raise ValueError("A alocação de campanhas é obrigatória.")

    actions = all_actions[
        all_actions["level"] == "adset"
    ].copy()
    if actions.empty:
        return pd.DataFrame(), {
            "solver": "scipy_milp_highs_nested",
            "n_campaigns": 0,
            "n_adsets": 0,
            "selected_spend": 0.0,
            "campaigns": {},
        }

    if "campaign_id" not in actions.columns:
        raise ValueError(
            "Ações de conjunto precisam carregar campaign_id."
        )

    campaign_budget = {
        str(row["entity_id"]): float(row["expected_spend"])
        for _, row in campaign_allocation.iterrows()
    }

    selected_groups: list[pd.DataFrame] = []
    details: dict[str, dict] = {}

    for campaign_id, budget_limit in campaign_budget.items():
        group = actions[
            actions["campaign_id"].astype(str) == campaign_id
        ].copy()
        if group.empty:
            details[campaign_id] = {
                "status": "no_adsets",
                "budget_limit": budget_limit,
                "selected_spend": 0.0,
                "unallocated_spend": budget_limit,
            }
            continue

        original_adsets = (
            group["entity_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        group["evidence_tier"] = group.apply(
            lambda row: infer_evidence_tier(
                row,
                evidence_overrides.get(
                    str(row["entity_id"])
                ),
            ),
            axis=1,
        )
        group["max_allowed_multiplier"] = (
            group["evidence_tier"].map(
                lambda tier: _max_multiplier(
                    tier,
                    cfg,
                )
            )
        )

        eligible = (
            group["action_multiplier"]
            <= group["max_allowed_multiplier"] + 1e-12
        )
        scale = group["action_multiplier"] > 1.0
        eligible &= (~scale) | (
            (
                group["p_profit"]
                >= cfg.min_p_profit_for_scale
            )
            & (
                group["p_incremental_profit_positive"]
                >= cfg.min_p_incremental_for_scale
            )
        )
        if "policy_eligible" in group.columns:
            eligible &= (
                group["policy_eligible"]
                .fillna(False)
                .astype(bool)
            )

        group = group[
            eligible
        ].copy().reset_index(drop=True)

        remaining = (
            group["entity_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        missing = sorted(
            set(original_adsets) - set(remaining)
        )
        if missing:
            raise ValueError(
                "Conjuntos ficaram sem ação elegível na campanha "
                f"{campaign_id}: "
                + ", ".join(missing)
            )

        adsets = remaining
        n = len(group)
        adset_index = {
            adset: i
            for i, adset in enumerate(adsets)
        }
        choose = lil_matrix(
            (len(adsets), n),
            dtype=float,
        )
        for j, row in group.iterrows():
            choose[
                adset_index[
                    str(row["entity_id"])
                ],
                j,
            ] = 1.0

        constraints: list[LinearConstraint] = [
            LinearConstraint(
                choose.tocsr(),
                np.ones(len(adsets)),
                np.ones(len(adsets)),
            ),
            LinearConstraint(
                group["expected_spend"]
                .to_numpy(dtype=float)[None, :],
                -np.inf,
                max(float(budget_limit), 0.0),
            ),
        ]

        if cfg.max_total_downside_proxy is not None:
            downside = np.maximum(
                0.0,
                -group["cvar10_profit"]
                .to_numpy(dtype=float),
            )
            constraints.append(
                LinearConstraint(
                    downside[None, :],
                    -np.inf,
                    float(
                        cfg.max_total_downside_proxy
                    ),
                )
            )

        objective = group[
            cfg.objective_column
        ].to_numpy(dtype=float)
        primary_c = -objective

        result = milp(
            c=primary_c,
            integrality=np.ones(n, dtype=int),
            bounds=Bounds(
                np.zeros(n),
                np.ones(n),
            ),
            constraints=constraints,
            options={
                "time_limit": cfg.time_limit_seconds
            },
        )
        if not result.success or result.x is None:
            raise RuntimeError(
                "Alocação de conjuntos não encontrou solução para "
                f"{campaign_id}: {result.message}"
            )

        primary_optimum = float(result.fun)

        if (
            cfg.revenue_tiebreak
            and "expected_revenue" in group.columns
            and cfg.revenue_tiebreak_tolerance >= 0
        ):
            allowed_deterioration = (
                float(
                    cfg.revenue_tiebreak_tolerance
                )
                * max(
                    abs(primary_optimum),
                    1.0,
                )
            )
            secondary_constraints = [
                *constraints,
                LinearConstraint(
                    primary_c[None, :],
                    -np.inf,
                    (
                        primary_optimum
                        + allowed_deterioration
                    ),
                ),
            ]
            revenue_c = -group[
                "expected_revenue"
            ].to_numpy(dtype=float)
            secondary = milp(
                c=revenue_c,
                integrality=np.ones(n, dtype=int),
                bounds=Bounds(
                    np.zeros(n),
                    np.ones(n),
                ),
                constraints=secondary_constraints,
                options={
                    "time_limit": cfg.time_limit_seconds
                },
            )
            if (
                secondary.success
                and secondary.x is not None
            ):
                result = secondary

        chosen = group[
            np.asarray(result.x) > 0.5
        ].copy()
        chosen[
            "parent_campaign_budget_limit"
        ] = float(budget_limit)
        chosen[
            "nested_allocation_source"
        ] = "campaign_budget_reconciliation"

        selected_spend = float(
            chosen["expected_spend"].sum()
        )
        details[campaign_id] = {
            "status": "ok",
            "budget_limit": float(budget_limit),
            "selected_spend": selected_spend,
            "unallocated_spend": max(
                float(budget_limit)
                - selected_spend,
                0.0,
            ),
            "n_adsets": len(adsets),
            "risk_adjusted_objective": float(
                chosen[
                    cfg.objective_column
                ].sum()
            ),
            "expected_revenue": (
                float(
                    chosen[
                        "expected_revenue"
                    ].sum()
                )
                if "expected_revenue"
                in chosen.columns
                else None
            ),
        }
        selected_groups.append(chosen)

    selected = (
        pd.concat(
            selected_groups,
            ignore_index=True,
            sort=False,
        )
        if selected_groups
        else pd.DataFrame()
    )

    summary = {
        "solver": "scipy_milp_highs_nested",
        "n_campaigns": len(campaign_budget),
        "n_adsets": (
            int(selected["entity_id"].nunique())
            if not selected.empty
            else 0
        ),
        "selected_spend": (
            float(selected["expected_spend"].sum())
            if not selected.empty
            else 0.0
        ),
        "campaign_budget_total": float(
            sum(campaign_budget.values())
        ),
        "unallocated_spend_total": float(
            sum(
                detail.get(
                    "unallocated_spend",
                    0.0,
                )
                for detail in details.values()
            )
        ),
        "campaigns": details,
    }
    return selected, summary
