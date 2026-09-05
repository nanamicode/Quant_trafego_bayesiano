from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .optimization import AllocationConfig, optimize_campaign_allocation


def main():
    parser = argparse.ArgumentParser(
        description="Alocação discreta exata de orçamento entre campanhas."
    )
    parser.add_argument("--actions", required=True, help="all_actions.csv")
    parser.add_argument("--output", default="allocation_output")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--min-p-profit-scale", type=float, default=0.60)
    parser.add_argument("--min-p-incremental-scale", type=float, default=0.55)
    parser.add_argument("--max-downside-proxy", type=float)
    args = parser.parse_args()

    actions = pd.read_csv(args.actions)
    selected, summary = optimize_campaign_allocation(
        actions,
        total_budget=args.budget,
        config=AllocationConfig(
            min_p_profit_for_scale=args.min_p_profit_scale,
            min_p_incremental_for_scale=args.min_p_incremental_scale,
            max_total_downside_proxy=args.max_downside_proxy,
        ),
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out / "allocation.csv", index=False)
    (out / "allocation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(selected.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
