from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .engine import BayesTrafficEngine, EngineConfig
from .funnel import hierarchical_funnel_diagnostics
from .hardware import detect_hardware
from .io import filter_active, load_ads_file
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
        default=1.0,
        help="Margem de contribuição antes da mídia, de 0 a 1. Ex.: 0.40 = 40%%.",
    )
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--draws", type=int, default=0, help="0 = automático pelo hardware.")
    parser.add_argument("--risk-aversion", type=float, default=0.25)
    parser.add_argument(
        "--temporal-model",
        choices=["derivative", "state_space"],
        default="derivative",
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
        )
    engine = BayesTrafficEngine(config)
    all_actions, best = engine.run(df)
    funnel_detail = hierarchical_funnel_diagnostics(df)
    write_reports(all_actions, best, args.output)
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
        },
    )
    write_run_manifest(manifest, args.output)
    workspace = LocalWarehouse(args.workspace)
    run_dir = workspace.persist_run(
        df,
        manifest,
        all_actions,
        best,
        extra_tables=(
            {"funnel_diagnostics": funnel_detail}
            if not funnel_detail.empty
            else None
        ),
    )
    print(f"Run auditável: {run_dir}")

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
        "decision_confidence",
        "cvar10_profit",
    ]
    print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()
