from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .io import filter_active, load_ads_file
from .optimization import AllocationConfig
from .portfolio import PortfolioRiskConfig, optimize_campaign_portfolio


def main():
    parser = argparse.ArgumentParser(
        description="Alocação global de campanhas com cenários correlacionados e CVaR."
    )
    parser.add_argument("--actions", required=True, help="all_actions.csv")
    parser.add_argument("--history", required=True, help="CSV/XLSX histórico original")
    parser.add_argument("--output", default="portfolio_output")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--contribution-margin", type=float, required=True)
    parser.add_argument("--cvar-alpha", type=float, default=0.10)
    parser.add_argument("--cvar-weight", type=float, default=0.25)
    parser.add_argument("--min-portfolio-cvar", type=float)
    parser.add_argument("--scenarios", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-inactive", action="store_true")
    args = parser.parse_args()

    actions = pd.read_csv(args.actions)
    history = load_ads_file(args.history)
    if not args.include_inactive:
        history = filter_active(history)

    selected, summary = optimize_campaign_portfolio(
        actions,
        history,
        contribution_margin=args.contribution_margin,
        total_budget=args.budget,
        allocation_config=AllocationConfig(),
        risk_config=PortfolioRiskConfig(
            cvar_alpha=args.cvar_alpha,
            cvar_weight=args.cvar_weight,
            min_portfolio_cvar=args.min_portfolio_cvar,
            scenarios=args.scenarios,
            seed=args.seed,
        ),
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out / "portfolio_allocation.csv", index=False)
    (out / "portfolio_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(selected.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
