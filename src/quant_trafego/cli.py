from __future__ import annotations

import argparse

from .engine import BayesTrafficEngine, EngineConfig
from .hardware import detect_hardware
from .io import filter_active, load_ads_file
from .quality import assess_data_quality
from .report import write_reports


def main():
    parser = argparse.ArgumentParser(
        description="Analisador quantitativo Bayesiano local para tráfego pago."
    )
    parser.add_argument("--input", required=True, help="Arquivo .csv ou .xlsx")
    parser.add_argument("--output", default="output", help="Pasta de saída")
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

    engine = BayesTrafficEngine(
        EngineConfig(
            target_roas=args.target_roas,
            contribution_margin=args.contribution_margin,
            horizon_days=args.horizon_days,
            draws=draws,
            risk_aversion=args.risk_aversion,
            seed=args.seed,
        )
    )
    all_actions, best = engine.run(df)
    write_reports(all_actions, best, args.output)

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
