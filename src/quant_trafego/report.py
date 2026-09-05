from __future__ import annotations

from pathlib import Path
import pandas as pd


def write_reports(all_actions: pd.DataFrame, best: pd.DataFrame, output_dir: str | Path):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_actions.to_csv(out / "all_actions.csv", index=False)
    best.to_csv(out / "best_actions.csv", index=False)

    summary = []
    for level in ["account", "campaign", "adset", "ad"]:
        subset = best[best["level"] == level].copy()
        if subset.empty:
            continue
        summary.append(f"# {level.upper()}\n")
        for _, row in subset.iterrows():
            summary.append(
                f"- **{row['entity_id']}** → ação {row['action_multiplier']:.1f}x | "
                f"lucro esperado {row['expected_profit']:.2f} | "
                f"P(lucro) {row['p_profit']:.1%} | "
                f"P(ROAS alvo) {row['p_roas_target']:.1%} | "
                f"P(superar manter) {row['p_beats_hold']:.1%} | "
                f"CVaR10 {row['cvar10_profit']:.2f}"
            )
        summary.append("")

    (out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
