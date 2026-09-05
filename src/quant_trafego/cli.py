from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .action_plan import build_operational_action_plan, write_operational_action_plan
from .engine import BayesTrafficEngine, EngineConfig
from .funnel import hierarchical_funnel_diagnostics
from .hardware import detect_hardware
from .io import filter_active, load_ads_file
from .optimization import optimize_adset_allocation, optimize_campaign_allocation
from .portfolio import optimize_campaign_portfolio
from .quality import assess_data_quality
from .report import write_reports
from .reproducibility import build_run_manifest, write_run_manifest
from .storage import LocalWarehouse


def main():
    parser = argparse.ArgumentParser(
        description="Analisador quantitativo Bayesiano local para tráfego pago."
    )
    parser.add_argument("--input", required=True, help="Arquivo .csv ou .xlsx")
    parser.add_argument("--output", default="output", help="Pasta de saída")
    parser.add_argument("--workspace", default="workspace", help="Warehouse local auditável")
    parser.add_argument("--target-roas", type=float, default=2.0)
    parser.add_argument(
        "--contribution-margin",
        type=float,
        required=True,
        help="Obrigatória. Margem de contribuição antes da mídia, de 0 a 1. Ex.: 0.40 = 40%%.",
    )
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--draws", type=int, default=0, help="0 = automático pelo hardware.")
    parser.add_argument("--risk-aversion", type=float, default=0.25)
    parser.add_argument(
        "--temporal-model",
        choices=["derivative", "state_space"],
        default="derivative",
    )
    parser.add_argument(
        "--disable-weekly-seasonality",
        action="store_true",
        help="Desativa o ajuste semanal encolhido de CTR/CVR.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Inclui entidades inativas quando houver coluna de status.",
    )
    args = parser.parse_args()

    df = load_ads_file(args.input)
    if not args.include_inactive:
        df = filter_active(df)

    quality = assess_data_quality(df)
    hw = detect_hardware()
    draws = args.draws if args.draws > 0 else hw.recommended_draws

    print(
        f"Hardware: {hw.cpu_threads} threads | perfil {hw.label} | "
        f"Monte Carlo: {draws:,} amostras."
    )
    print(f"Qualidade estrutural: {quality.score:.0f}/100")
    for warning in quality.warnings:
        print(f"AVISO: {warning}")

    config = EngineConfig(
            target_roas=args.target_roas,
            contribution_margin=args.contribution_margin,
            horizon_days=args.horizon_days,
            draws=draws,
            risk_aversion=args.risk_aversion,
            seed=args.seed,
            temporal_model=args.temporal_model,
            use_weekly_seasonality=not args.disable_weekly_seasonality,
        )
    engine = BayesTrafficEngine(config)
    all_actions, best = engine.run(df)
    allocation = None
    allocation_summary = None
    try:
        allocation, allocation_summary = optimize_campaign_portfolio(
            all_actions,
            df,
            contribution_margin=config.contribution_margin,
        )
    except Exception as portfolio_exc:
        try:
            allocation, allocation_summary = optimize_campaign_allocation(
                all_actions
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
                all_actions,
                allocation,
            )
        except Exception as adset_exc:
            adset_allocation_summary = {
                "status": "unavailable",
                "reason": str(adset_exc),
            }

    operational_plan = build_operational_action_plan(
        best,
        allocation=allocation,
        adset_allocation=adset_allocation,
        source_df=df,
        horizon_days=config.horizon_days,
    )
    funnel_detail = hierarchical_funnel_diagnostics(df)
    write_reports(all_actions, best, args.output)
    write_operational_action_plan(operational_plan, args.output)
    if not funnel_detail.empty:
        funnel_detail.to_csv(
            Path(args.output) / "funnel_diagnostics.csv",
            index=False,
        )

    manifest = build_run_manifest(
        df,
        config=config,
        inference_mode="empirical_bayes",
        seed=args.seed,
        extra={
            "quality_score": quality.score,
            "quality_warnings": list(quality.warnings),
            "quality_report": asdict(quality),
            "allocation_summary": allocation_summary,
            "adset_allocation_summary": adset_allocation_summary,
        },
    )
    write_run_manifest(manifest, args.output)
    workspace = LocalWarehouse(args.workspace)
    run_dir = workspace.persist_run(
        df,
        manifest,
        all_actions,
        best,
        extra_tables={
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
            "allocation_summary": allocation_summary or {}
        },
    )
    print(f"Run auditável: {run_dir}")

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
    print()

    cols = [
        "level",
        "entity_id",
        "action_multiplier",
        "expected_profit",
        "expected_incremental_profit_vs_hold",
        "expected_roas",
        "p_profit",
        "p_roas_target",
        "p_action_optimal",
        "decision_score",
        "cvar10_profit",
    ]
    print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()
