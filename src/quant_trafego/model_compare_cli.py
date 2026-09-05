from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import EngineConfig
from .hardware import detect_hardware
from .io import filter_active, load_ads_file
from .model_selection import compare_temporal_models


def main():
    parser = argparse.ArgumentParser(
        description="Compara modelos temporais por rolling-origin backtesting."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="model_comparison")
    parser.add_argument("--target-roas", type=float, default=2.0)
    parser.add_argument("--contribution-margin", type=float, default=1.0)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--min-train-days", type=int, default=21)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--draws", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-inactive", action="store_true")
    args = parser.parse_args()

    df = load_ads_file(args.input)
    if not args.include_inactive:
        df = filter_active(df)

    hw = detect_hardware()
    draws = args.draws or min(hw.recommended_draws, 20000)
    comparison, decision = compare_temporal_models(
        df,
        config=EngineConfig(
            target_roas=args.target_roas,
            contribution_margin=args.contribution_margin,
            horizon_days=args.horizon_days,
            draws=draws,
            seed=args.seed,
        ),
        min_train_days=args.min_train_days,
        horizon_days=args.horizon_days,
        step_days=args.step_days,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out / "temporal_model_comparison.csv", index=False)
    (out / "temporal_model_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
