from __future__ import annotations

import argparse

from .hardware import detect_hardware
from .io import filter_active, load_ads_file
from .mcmc import fit_hierarchical_funnel, save_mcmc_result


def main():
    parser = argparse.ArgumentParser(
        description="Ajuste Bayesiano hierárquico profundo de CTR/CVR via PyMC."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output_mcmc")
    parser.add_argument("--draws", type=int, default=1200)
    parser.add_argument("--tune", type=int, default=1200)
    parser.add_argument("--chains", type=int)
    parser.add_argument("--cores", type=int)
    parser.add_argument("--method", choices=["auto", "nuts", "advi"], default="auto")
    parser.add_argument("--include-inactive", action="store_true")
    args = parser.parse_args()

    df = load_ads_file(args.input)
    if not args.include_inactive:
        df = filter_active(df)

    hw = detect_hardware()
    chains = args.chains or hw.recommended_mcmc_chains
    cores = args.cores or hw.recommended_mcmc_cores

    idata, diagnostics = fit_hierarchical_funnel(
        df,
        draws=args.draws,
        tune=args.tune,
        chains=chains,
        cores=cores,
        method=args.method,
    )
    save_mcmc_result(idata, diagnostics, args.output)
    print(diagnostics)


if __name__ == "__main__":
    main()
