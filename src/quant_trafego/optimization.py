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
    result = milp(
        c=-objective,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraints,
        options={"time_limit": cfg.time_limit_seconds},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"Otimização inteira não encontrou solução: {result.message}"
        )

    chosen = actions[np.asarray(result.x) > 0.5].copy()
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
