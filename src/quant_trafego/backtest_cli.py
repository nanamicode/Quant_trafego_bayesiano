from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import rolling_origin_backtest
from .calibration import calibration_table
from .engine import EngineConfig
from .hardware import detect_hardware
from .io import filter_active, load_ads_file


def main():
    parser = argparse.ArgumentParser(
        description="Rolling-origin backtesting probabilístico."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="backtest_output")
    parser.add_argument("--target-roas", type=float, default=2.0)
    parser.add_argument("--contribution-margin", type=float, default=1.0)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--min-train-days", type=int, default=21)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument(
        "--temporal-model",
        choices=["derivative", "state_space"],
        default="derivative",
    )
    parser.add_argument("--draws", type=int, default=0)
    parser.add_argument(
        "--disable-weekly-seasonality",
        action="store_true",
        help="Desativa o ajuste semanal encolhido de CTR/CVR.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-inactive", action="store_true")
    args = parser.parse_args()

    df = load_ads_file(args.input)
    if not args.include_inactive:
        df = filter_active(df)

    hw = detect_hardware()
    draws = args.draws or min(hw.recommended_draws, 30000)
    config = EngineConfig(
        target_roas=args.target_roas,
        contribution_margin=args.contribution_margin,
        horizon_days=args.horizon_days,
        draws=draws,
        seed=args.seed,
            temporal_model=args.temporal_model,
            use_weekly_seasonality=not args.disable_weekly_seasonality,
    )

    detail, summary = rolling_origin_backtest(
        df,
        config=config,
        min_train_days=args.min_train_days,
        horizon_days=args.horizon_days,
        step_days=args.step_days,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "backtest_detail.csv", index=False)
    summary.to_csv(out / "backtest_summary.csv", index=False)
    if not detail.empty:
        profit_calibration = calibration_table(
            detail["predicted_p_profit"],
            (detail["actual_profit"] > 0).astype(float),
        )
        roas_calibration = calibration_table(
            detail["predicted_p_roas_target"],
            (detail["actual_roas"] >= args.target_roas).astype(float),
        )
        profit_calibration.to_csv(out / "profit_calibration.csv", index=False)
        roas_calibration.to_csv(out / "roas_calibration.csv", index=False)

    if summary.empty:
        print("Dados insuficientes para o protocolo solicitado.")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()