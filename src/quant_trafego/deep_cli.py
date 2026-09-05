from __future__ import annotations

import argparse

from .deep_analysis import run_deep_analysis
from .engine import EngineConfig
from .hardware import detect_hardware
from .io import filter_active, load_ads_file
from .quality import assess_data_quality


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Análise quantitativa profunda: PyMC hierárquico + Monte Carlo "
            "econômico + derivadas temporais."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output_mcmc")
    parser.add_argument("--target-roas", type=float, default=2.0)
    parser.add_argument("--contribution-margin", type=float, default=1.0)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument(
        "--mc-draws",
        type=int,
        default=0,
        help="Amostras Monte Carlo por ação. 0 = automático pelo hardware.",
    )
    parser.add_argument("--risk-aversion", type=float, default=0.25)
    parser.add_argument("--mcmc-draws", type=int, default=1200)
    parser.add_argument("--mcmc-tune", type=int, default=1200)
    parser.add_argument("--chains", type=int)
    parser.add_argument("--cores", type=int)
    parser.add_argument("--target-accept", type=float, default=0.92)
    parser.add_argument("--advi-steps", type=int, default=40000)
    parser.add_argument("--method", choices=["auto", "nuts", "advi"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-inactive", action="store_true")
    args = parser.parse_args()

    df = load_ads_file(args.input)
    if not args.include_inactive:
        df = filter_active(df)

    quality = assess_data_quality(df)
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

    result = run_deep_analysis(
        df,
        engine_config=EngineConfig(
            target_roas=args.target_roas,
            contribution_margin=args.contribution_margin,
            horizon_days=args.horizon_days,
            draws=mc_draws,
            risk_aversion=args.risk_aversion,
            seed=args.seed,
        ),
        mcmc_draws=args.mcmc_draws,
        mcmc_tune=args.mcmc_tune,
        mcmc_chains=chains,
        mcmc_cores=cores,
        mcmc_method=args.method,
        target_accept=args.target_accept,
        advi_steps=args.advi_steps,
        seed=args.seed,
        output_dir=args.output,
    )

    print(result.diagnostics)
    cols = [
        "level",
        "entity_id",
        "posterior_source",
        "action_multiplier",
        "expected_profit",
        "expected_incremental_profit_vs_hold",
        "p_profit",
        "p_action_optimal",
        "decision_confidence",
        "expected_regret",
    ]
    print(result.best_actions[cols].to_string(index=False))


if __name__ == "__main__":
    main()
