from __future__ import annotations

import argparse
from dataclasses import asdict

from .action_plan import build_operational_action_plan, derive_account_budget_target, write_operational_action_plan
from .deep_analysis import run_deep_analysis
from .engine import EngineConfig
from .funnel import hierarchical_funnel_diagnostics
from .hardware import detect_hardware
from .io import filter_decision_rows, infer_decision_universe, load_ads_file
from .optimization import optimize_adset_allocation, optimize_campaign_allocation
from .portfolio import optimize_campaign_portfolio
from .quality import assess_data_quality
from .reproducibility import build_run_manifest, write_run_manifest
from .storage import LocalWarehouse


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Análise quantitativa profunda: PyMC hierárquico + Monte Carlo "
            "econômico + derivadas temporais."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output_mcmc")
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--target-roas", type=float, default=2.0)
    parser.add_argument("--contribution-margin", type=float, required=True)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument(
        "--mc-draws",
        type=int,
        default=0,
        help="Amostras Monte Carlo por ação. 0 = automático pelo hardware.",
    )
    parser.add_argument("--risk-aversion", type=float, default=0.25)
    parser.add_argument(
        "--temporal-model",
        choices=["derivative", "state_space"],
        default="derivative",
    )
    parser.add_argument("--mcmc-draws", type=int, default=1200)
    parser.add_argument("--mcmc-tune", type=int, default=1200)
    parser.add_argument("--chains", type=int)
    parser.add_argument("--cores", type=int)
    parser.add_argument("--target-accept", type=float, default=0.92)
    parser.add_argument("--advi-steps", type=int, default=40000)
    parser.add_argument("--method", choices=["auto", "nuts", "advi"], default="auto")
    parser.add_argument(
        "--disable-weekly-seasonality",
        action="store_true",
        help="Desativa o ajuste semanal encolhido de CTR/CVR.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-inactive", action="store_true")
    args = parser.parse_args()

    df = load_ads_file(args.input)
    decision_entities = None
    operational_df = df
    if not args.include_inactive:
        universe = infer_decision_universe(df)
        decision_entities = {
            "campaign": universe.campaign_ids,
            "adset": universe.adset_ids,
            "ad": universe.ad_ids,
        }
        operational_df = filter_decision_rows(
            df,
            universe,
        )
        print(
            "Contexto completo preservado | "
            f"ativos detectados por {universe.detection_method}: "
            f"{len(universe.campaign_ids)} campanhas, "
            f"{len(universe.adset_ids)} conjuntos, "
            f"{len(universe.ad_ids)} anúncios."
        )

    quality = assess_data_quality(operational_df)
    funnel_detail = hierarchical_funnel_diagnostics(df)
    hw = detect_hardware()
    chains = args.chains or hw.recommended_mcmc_chains
    cores = args.cores or hw.recommended_mcmc_cores
    mc_draws = args.mc_draws or hw.recommended_draws

    print(
        f"Hardware: {hw.cpu_threads} threads | perfil {hw.label} | "
        f"Monte Carlo {mc_draws:,} | MCMC {chains} chains / {cores} cores."
    )
    print(f"Qualidade estrutural: {quality.score:.0f}/100")
    for warning in quality.warnings:
        print(f"AVISO: {warning}")

    config = EngineConfig(
            target_roas=args.target_roas,
            contribution_margin=args.contribution_margin,
            horizon_days=args.horizon_days,
            draws=mc_draws,
            risk_aversion=args.risk_aversion,
            seed=args.seed,
            temporal_model=args.temporal_model,
            use_weekly_seasonality=not args.disable_weekly_seasonality,
        )

    result = run_deep_analysis(
        df,
        engine_config=config,
        mcmc_draws=args.mcmc_draws,
        mcmc_tune=args.mcmc_tune,
        mcmc_chains=chains,
        mcmc_cores=cores,
        mcmc_method=args.method,
        target_accept=args.target_accept,
        advi_steps=args.advi_steps,
        seed=args.seed,
        output_dir=args.output,
        decision_entities=decision_entities,
    )

    account_budget_target = derive_account_budget_target(
        result.best_actions,
        source_df=operational_df,
        horizon_days=config.horizon_days,
    )
    allocation = None
    allocation_summary = None
    try:
        allocation, allocation_summary = optimize_campaign_portfolio(
            result.all_actions,
            df,
            contribution_margin=config.contribution_margin,
            total_budget=account_budget_target["recommended_horizon_amount"],
        )
    except Exception as portfolio_exc:
        try:
            allocation, allocation_summary = optimize_campaign_allocation(
                result.all_actions,
                total_budget=account_budget_target["recommended_horizon_amount"],
            )
            allocation_summary["fallback_reason"] = str(portfolio_exc)
        except Exception as allocation_exc:
            allocation_summary = {
                "status": "unavailable",
                "reason": str(allocation_exc),
                "portfolio_reason": str(portfolio_exc),
            }

    adset_allocation = None
    adset_allocation_summary = None
    if allocation is not None:
        try:
            adset_allocation, adset_allocation_summary = optimize_adset_allocation(
                result.all_actions,
                allocation,
            )
        except Exception as adset_exc:
            adset_allocation_summary = {
                "status": "unavailable",
                "reason": str(adset_exc),
            }

    operational_plan = build_operational_action_plan(
        result.best_actions,
        allocation=allocation,
        adset_allocation=adset_allocation,
        account_budget_target=account_budget_target,
        source_df=df,
        horizon_days=config.horizon_days,
    )
    write_operational_action_plan(
        operational_plan,
        args.output,
    )

    manifest = build_run_manifest(
        df,
        config=config,
        inference_mode=f"mcmc_{result.diagnostics.method}",
        seed=args.seed,
        extra={
            "quality_score": quality.score,
            "quality_warnings": list(quality.warnings),
            "quality_report": asdict(quality),
            "mcmc_diagnostics": result.diagnostics.__dict__,
            "ppc_summary": result.ppc_summary.__dict__,
            "deep_decision_source": result.decision_source,
            "deep_guardrail": result.guardrail,
            "account_budget_target": account_budget_target,
                    "allocation_summary": allocation_summary,
            "adset_allocation_summary": adset_allocation_summary,
        },
    )
    write_run_manifest(manifest, args.output)
    workspace = LocalWarehouse(args.workspace)
    run_dir = workspace.persist_run(
        df,
        manifest,
        result.all_actions,
        result.best_actions,
        extra_tables={
            "posterior_predictive_checks": result.ppc_detail,
            **(
                {"mcmc_candidate_actions": result.candidate_mcmc_actions}
                if result.guardrail != "none"
                else {}
            ),
            **(
                {"funnel_diagnostics": funnel_detail}
                if not funnel_detail.empty
                else {}
            ),
            **(
                {"allocation": allocation}
                if allocation is not None
                else {}
            ),
            **(
                {"adset_allocation": adset_allocation}
                if adset_allocation is not None and not adset_allocation.empty
                else {}
            ),
            **(
                {"operational_action_plan": operational_plan}
                if not operational_plan.empty
                else {}
            ),
        },
        extra_json={
            "posterior_predictive_summary": result.ppc_summary.__dict__,
            "account_budget_target": account_budget_target,
            "allocation_summary": allocation_summary or {},
            "adset_allocation_summary": adset_allocation_summary or {},
        },
    )
    print(f"Run auditável: {run_dir}")
    print(result.diagnostics)
    print(result.ppc_summary)
    print(f"Fonte decisória profunda: {result.decision_source}")
    print(f"Guardrail profundo: {result.guardrail}")
    print("\nPLANO OPERACIONAL")
    operational_cols = [
        "level",
        "campaign_name",
        "adset_name",
        "ad_name",
        "capital_action",
        "current_daily_amount",
        "recommended_daily_amount",
        "daily_amount_change",
        "duplicate_action",
        "expected_incremental_profit",
        "expected_incremental_revenue",
        "p_incremental_profit_positive",
    ]
    print(
        operational_plan[operational_cols].to_string(index=False)
        if not operational_plan.empty
        else "Nenhuma ação operacional disponível."
    )
    cols = [
        "level",
        "entity_id",
        "posterior_source",
        "action_multiplier",
        "expected_profit",
        "expected_incremental_profit_vs_hold",
        "p_profit",
        "p_action_optimal",
        "decision_score",
        "expected_regret",
    ]
    print(result.best_actions[cols].to_string(index=False))


if __name__ == "__main__":
    main()
