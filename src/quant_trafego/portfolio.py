from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .optimization import AllocationConfig, infer_evidence_tier


@dataclass(frozen=True)
class PortfolioRiskConfig:
    cvar_alpha: float = 0.10
    cvar_weight: float = 0.25
    min_portfolio_cvar: float | None = None
    scenarios: int = 1000
    correlation_shrinkage_days: float = 20.0
    seed: int = 42
    time_limit_seconds: float = 45.0


def _nearest_psd_correlation(corr: np.ndarray) -> np.ndarray:
    corr = np.asarray(corr, dtype=float)
    corr = (corr + corr.T) / 2.0
    values, vectors = np.linalg.eigh(corr)
    values = np.clip(values, 1e-6, None)
    psd = vectors @ np.diag(values) @ vectors.T
    scale = np.sqrt(np.clip(np.diag(psd), 1e-12, None))
    psd = psd / np.outer(scale, scale)
    np.fill_diagonal(psd, 1.0)
    return psd


def _estimate_campaign_correlation_details(
    df: pd.DataFrame,
    campaign_ids: list[str],
    *,
    contribution_margin: float,
    shrinkage_days: float = 20.0,
) -> tuple[np.ndarray, int, np.ndarray]:
    daily = (
        df.groupby(["date", "campaign_id"], as_index=False)
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
        )
        .copy()
    )
    daily["campaign_id"] = daily["campaign_id"].astype(str)
    daily["profit"] = (
        daily["revenue"] * float(contribution_margin)
        - daily["spend"]
    )
    pivot = daily.pivot(
        index="date",
        columns="campaign_id",
        values="profit",
    ).reindex(columns=campaign_ids)

    n_days = int(len(pivot))
    n_campaigns = len(campaign_ids)
    if n_campaigns <= 1 or n_days < 5:
        return (
            np.eye(n_campaigns),
            n_days,
            np.full((n_campaigns, n_campaigns), n_days, dtype=int),
        )

    standardized = pivot.copy()
    for col in standardized.columns:
        x = standardized[col]
        mean = x.mean()
        sd = x.std(ddof=1)
        if not np.isfinite(sd) or sd <= 1e-9:
            standardized[col] = np.nan
        else:
            standardized[col] = (x - mean) / sd

    values = standardized.to_numpy(dtype=float)
    corr = np.eye(n_campaigns, dtype=float)
    overlap = np.zeros((n_campaigns, n_campaigns), dtype=int)

    for i in range(n_campaigns):
        valid_i = np.isfinite(values[:, i])
        overlap[i, i] = int(valid_i.sum())

        for j in range(i + 1, n_campaigns):
            valid = (
                np.isfinite(values[:, i])
                & np.isfinite(values[:, j])
            )
            n_overlap = int(valid.sum())
            overlap[i, j] = n_overlap
            overlap[j, i] = n_overlap

            if n_overlap < 5:
                shrunk = 0.0
            else:
                raw = float(
                    np.corrcoef(
                        values[valid, i],
                        values[valid, j],
                    )[0, 1]
                )
                if not np.isfinite(raw):
                    raw = 0.0

                # Pair-specific empirical shrinkage. A correlation estimated
                # from 5 overlapping days should never receive the same trust
                # as one estimated from 60 overlapping days.
                shrink = float(
                    np.clip(
                        n_overlap
                        / (
                            n_overlap
                            + max(shrinkage_days, 1.0)
                        ),
                        0.0,
                        0.95,
                    )
                )
                shrunk = shrink * raw

            corr[i, j] = shrunk
            corr[j, i] = shrunk

    return (
        _nearest_psd_correlation(corr),
        n_days,
        overlap,
    )


def estimate_campaign_correlation(
    df: pd.DataFrame,
    campaign_ids: list[str],
    *,
    contribution_margin: float,
    shrinkage_days: float = 20.0,
) -> tuple[np.ndarray, int]:
    corr, n_days, _ = _estimate_campaign_correlation_details(
        df,
        campaign_ids,
        contribution_margin=contribution_margin,
        shrinkage_days=shrinkage_days,
    )
    return corr, n_days


def _action_scenarios(
    actions: pd.DataFrame,
    campaign_ids: list[str],
    corr: np.ndarray,
    *,
    scenarios: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.multivariate_normal(
        mean=np.zeros(len(campaign_ids)),
        cov=corr,
        size=scenarios,
    )
    campaign_pos = {c: i for i, c in enumerate(campaign_ids)}
    result = np.zeros((scenarios, len(actions)), dtype=float)

    normal_q = 1.6448536269514722
    for j, row in actions.iterrows():
        pos = campaign_pos[str(row["entity_id"])]
        z = latent[:, pos]

        median = float(row["profit_p50"])
        lower_scale = max(
            (median - float(row["profit_p05"])) / normal_q,
            1e-6,
        )
        upper_scale = max(
            (float(row["profit_p95"]) - median) / normal_q,
            1e-6,
        )
        simulated = np.where(
            z < 0,
            median + z * lower_scale,
            median + z * upper_scale,
        )
        # Preserve the Monte Carlo expected value while using a Gaussian copula
        # only for dependence across campaigns.
        simulated += float(row["expected_profit"]) - float(simulated.mean())
        result[:, j] = simulated

    return result


def optimize_campaign_portfolio(
    all_actions: pd.DataFrame,
    historical_df: pd.DataFrame,
    *,
    contribution_margin: float,
    total_budget: float | None = None,
    allocation_config: AllocationConfig | None = None,
    risk_config: PortfolioRiskConfig | None = None,
    evidence_overrides: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    alloc_cfg = allocation_config or AllocationConfig()
    risk_cfg = risk_config or PortfolioRiskConfig()

    if not 0.0 < risk_cfg.cvar_alpha < 0.5:
        raise ValueError("cvar_alpha deve estar entre 0 e 0.5.")

    actions = all_actions[all_actions["level"] == "campaign"].copy()
    if actions.empty:
        raise ValueError("Não há campanhas para otimizar.")

    original_campaigns = (
        actions["entity_id"].astype(str).drop_duplicates().tolist()
    )
    evidence_overrides = evidence_overrides or {}
    actions["evidence_tier"] = actions.apply(
        lambda row: infer_evidence_tier(
            row,
            evidence_overrides.get(str(row["entity_id"])),
        ),
        axis=1,
    )

    tier_caps = {
        "predictive": alloc_cfg.predictive_max_multiplier,
        "observational_intervention": alloc_cfg.observational_max_multiplier,
        "experiment_calibrated": alloc_cfg.experiment_max_multiplier,
    }
    actions["max_allowed_multiplier"] = actions["evidence_tier"].map(tier_caps)

    scale = actions["action_multiplier"] > 1.0
    eligible = (
        actions["action_multiplier"]
        <= actions["max_allowed_multiplier"] + 1e-12
    )
    eligible &= (~scale) | (
        (actions["p_profit"] >= alloc_cfg.min_p_profit_for_scale)
        & (
            actions["p_incremental_profit_positive"]
            >= alloc_cfg.min_p_incremental_for_scale
        )
    )
    actions = actions[eligible].copy().reset_index(drop=True)

    remaining = actions["entity_id"].astype(str).unique().tolist()
    missing = sorted(set(original_campaigns) - set(remaining))
    if missing:
        raise ValueError(
            "Campanhas sem ação elegível: " + ", ".join(missing)
        )
    campaign_ids = original_campaigns

    if total_budget is None:
        holds = actions[np.isclose(actions["action_multiplier"], 1.0)]
        if len(holds["entity_id"].unique()) == len(campaign_ids):
            total_budget = float(holds["expected_spend"].sum())
        else:
            total_budget = float(
                actions.groupby("entity_id")["expected_spend"].median().sum()
            )
    total_budget = max(float(total_budget), 0.0)

    corr, correlation_days, pair_overlap = _estimate_campaign_correlation_details(
        historical_df,
        campaign_ids,
        contribution_margin=contribution_margin,
        shrinkage_days=risk_cfg.correlation_shrinkage_days,
    )
    scenario_profit = _action_scenarios(
        actions,
        campaign_ids,
        corr,
        scenarios=risk_cfg.scenarios,
        seed=risk_cfg.seed,
    )

    n_actions = len(actions)
    n_scenarios = risk_cfg.scenarios
    eta_idx = n_actions
    u_start = n_actions + 1
    n_vars = n_actions + 1 + n_scenarios

    objective = np.zeros(n_vars, dtype=float)
    objective[:n_actions] = -actions["expected_profit"].to_numpy(dtype=float)
    objective[eta_idx] = -risk_cfg.cvar_weight
    objective[u_start:] = (
        risk_cfg.cvar_weight
        / (risk_cfg.cvar_alpha * n_scenarios)
    )

    constraints: list[LinearConstraint] = []

    campaign_index = {c: i for i, c in enumerate(campaign_ids)}
    choose = lil_matrix((len(campaign_ids), n_vars), dtype=float)
    for j, row in actions.iterrows():
        choose[campaign_index[str(row["entity_id"])], j] = 1.0
    constraints.append(
        LinearConstraint(
            choose.tocsr(),
            np.ones(len(campaign_ids)),
            np.ones(len(campaign_ids)),
        )
    )

    budget = np.zeros((1, n_vars), dtype=float)
    budget[0, :n_actions] = actions["expected_spend"].to_numpy(dtype=float)
    constraints.append(
        LinearConstraint(budget, -np.inf, total_budget)
    )

    # u_s >= eta - portfolio_profit_s
    tail = lil_matrix((n_scenarios, n_vars), dtype=float)
    for s in range(n_scenarios):
        tail[s, :n_actions] = -scenario_profit[s, :]
        tail[s, eta_idx] = 1.0
        tail[s, u_start + s] = -1.0
    constraints.append(
        LinearConstraint(
            tail.tocsr(),
            -np.inf * np.ones(n_scenarios),
            np.zeros(n_scenarios),
        )
    )

    if risk_cfg.min_portfolio_cvar is not None:
        floor = np.zeros((1, n_vars), dtype=float)
        floor[0, eta_idx] = -1.0
        floor[0, u_start:] = (
            1.0 / (risk_cfg.cvar_alpha * n_scenarios)
        )
        constraints.append(
            LinearConstraint(
                floor,
                -np.inf,
                -float(risk_cfg.min_portfolio_cvar),
            )
        )

    lower = np.concatenate(
        [np.zeros(n_actions), np.array([-np.inf]), np.zeros(n_scenarios)]
    )
    upper = np.concatenate(
        [np.ones(n_actions), np.array([np.inf]), np.full(n_scenarios, np.inf)]
    )
    integrality = np.concatenate(
        [np.ones(n_actions, dtype=int), np.zeros(1 + n_scenarios, dtype=int)]
    )

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"time_limit": risk_cfg.time_limit_seconds},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"Otimização CVaR não encontrou solução: {result.message}"
        )

    chosen_mask = result.x[:n_actions] > 0.5
    chosen = actions[chosen_mask].copy().sort_values("entity_id").reset_index(drop=True)
    chosen_indices = np.flatnonzero(chosen_mask)
    portfolio_scenarios = scenario_profit[:, chosen_indices].sum(axis=1)

    q = float(np.quantile(portfolio_scenarios, risk_cfg.cvar_alpha))
    tail_values = portfolio_scenarios[portfolio_scenarios <= q]
    cvar = float(tail_values.mean()) if len(tail_values) else q

    off_diag_overlap = pair_overlap[
        ~np.eye(len(campaign_ids), dtype=bool)
    ]
    positive_overlap = off_diag_overlap[off_diag_overlap > 0]

    summary = {
        "solver": "scipy_milp_highs_cvar",
        "dependence_model": "shrunk_historical_correlation_gaussian_copula",
        "correlation_history_days": correlation_days,
        "correlation_pair_overlap_min_days": (
            int(positive_overlap.min())
            if len(positive_overlap)
            else correlation_days
        ),
        "correlation_pair_overlap_median_days": (
            float(np.median(positive_overlap))
            if len(positive_overlap)
            else float(correlation_days)
        ),
        "scenario_count": n_scenarios,
        "cvar_alpha": risk_cfg.cvar_alpha,
        "cvar_weight": risk_cfg.cvar_weight,
        "total_budget_limit": total_budget,
        "selected_spend": float(chosen["expected_spend"].sum()),
        "expected_portfolio_profit": float(chosen["expected_profit"].sum()),
        "scenario_profit_p10": q,
        "scenario_portfolio_cvar": cvar,
        "scenario_p_profit": float(np.mean(portfolio_scenarios > 0)),
        "scenario_profit_p05": float(np.quantile(portfolio_scenarios, 0.05)),
        "scenario_profit_p50": float(np.quantile(portfolio_scenarios, 0.50)),
        "scenario_profit_p95": float(np.quantile(portfolio_scenarios, 0.95)),
        "important_limitation": (
            "A dependência usa correlação histórica encolhida e cópula Gaussiana; "
            "não identifica causalmente canibalização de leilão/audiência."
        ),
    }
    return chosen, summary
