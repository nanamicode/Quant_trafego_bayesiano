from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .engine import BayesTrafficEngine, EngineConfig
from .mcmc import (
    MCMCDiagnostics,
    fit_hierarchical_funnel,
    posterior_rate_overrides,
    save_mcmc_result,
)
from .ppc import PPCSummary, posterior_predictive_checks
from .report import write_reports


@dataclass
class DeepAnalysisResult:
    all_actions: pd.DataFrame
    best_actions: pd.DataFrame
    diagnostics: MCMCDiagnostics
    idata: object
    mapping: dict
    ppc_detail: pd.DataFrame
    ppc_summary: PPCSummary


def run_deep_analysis(
    df: pd.DataFrame,
    *,
    engine_config: EngineConfig | None = None,
    mcmc_draws: int = 1200,
    mcmc_tune: int = 1200,
    mcmc_chains: int = 4,
    mcmc_cores: int | None = None,
    mcmc_method: str = "auto",
    target_accept: float = 0.92,
    advi_steps: int = 40_000,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> DeepAnalysisResult:
    engine = BayesTrafficEngine(engine_config or EngineConfig(seed=seed))
    clean = engine.validate(df)

    idata, diagnostics, mapping = fit_hierarchical_funnel(
        clean,
        draws=mcmc_draws,
        tune=mcmc_tune,
        chains=mcmc_chains,
        cores=mcmc_cores,
        target_accept=target_accept,
        seed=seed,
        method=mcmc_method,
        advi_steps=advi_steps,
        return_mapping=True,
    )

    ppc_detail, ppc_summary = posterior_predictive_checks(
        idata,
        clean,
        mapping,
        seed=seed + 1000,
    )

    overrides = posterior_rate_overrides(idata, mapping)
    all_actions, best_actions = engine.run(
        clean,
        posterior_overrides=overrides,
    )

    all_actions = all_actions.copy()
    best_actions = best_actions.copy()
    all_actions["deep_inference_method"] = diagnostics.method
    best_actions["deep_inference_method"] = diagnostics.method
    best_actions["mcmc_converged"] = diagnostics.converged
    best_actions["mcmc_max_rhat"] = diagnostics.max_rhat
    best_actions["mcmc_min_ess_bulk"] = diagnostics.min_ess_bulk
    best_actions["mcmc_divergences"] = diagnostics.divergences
    best_actions["ppc_status"] = ppc_summary.status
    best_actions["ppc_click_90_coverage"] = ppc_summary.click_90_coverage
    best_actions["ppc_conversion_90_coverage"] = ppc_summary.conversion_90_coverage

    if output_dir is not None:
        out = Path(output_dir)
        write_reports(all_actions, best_actions, out)
        save_mcmc_result(idata, diagnostics, out, mapping=mapping)
        ppc_detail.to_csv(out / "posterior_predictive_checks.csv", index=False)
        pd.DataFrame([ppc_summary.__dict__]).to_csv(
            out / "posterior_predictive_summary.csv", index=False
        )

    return DeepAnalysisResult(
        all_actions=all_actions,
        best_actions=best_actions,
        diagnostics=diagnostics,
        idata=idata,
        mapping=mapping,
        ppc_detail=ppc_detail,
        ppc_summary=ppc_summary,
    )
