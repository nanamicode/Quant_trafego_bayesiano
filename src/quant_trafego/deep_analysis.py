from __future__ import annotations

from dataclasses import dataclass, replace
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
    decision_source: str
    guardrail: str
    candidate_mcmc_actions: pd.DataFrame


def deep_decision_policy(
    diagnostics: MCMCDiagnostics,
    ppc_summary: PPCSummary,
) -> tuple[str, str]:
    """
    Decide whether deep posterior uncertainty is allowed to drive actions.

    Returns (decision_source, guardrail).
    """
    if (
        diagnostics.method == "nuts"
        and diagnostics.converged is not True
    ):
        return (
            "empirical_bayes_fallback",
            "nuts_diagnostics_failed",
        )

    if ppc_summary.status != "pass":
        return (
            "mcmc_posterior_guarded",
            "posterior_predictive_check_not_passed",
        )

    if diagnostics.method == "advi":
        return (
            "mcmc_advi_approximate",
            "advi_approximation_scale_cap",
        )

    return (
        "mcmc_validated",
        "none",
    )


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
    base_config = engine_config or EngineConfig(
        seed=seed
    )
    engine = BayesTrafficEngine(base_config)
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

    ppc_detail, ppc_summary = (
        posterior_predictive_checks(
            idata,
            clean,
            mapping,
            seed=seed + 1000,
        )
    )

    overrides = posterior_rate_overrides(
        idata,
        mapping,
    )
    candidate_all, candidate_best = engine.run(
        clean,
        posterior_overrides=overrides,
    )
    candidate_all = candidate_all.copy()
    candidate_best = candidate_best.copy()

    decision_source, guardrail = deep_decision_policy(
        diagnostics,
        ppc_summary,
    )

    if decision_source == "empirical_bayes_fallback":
        # A numerically invalid NUTS posterior is retained for diagnosis only.
        final_engine = BayesTrafficEngine(
            base_config
        )
        all_actions, best_actions = final_engine.run(
            clean
        )
    elif guardrail != "none":
        if guardrail == "posterior_predictive_check_not_passed":
            max_scale = 1.0
        elif guardrail == "advi_approximation_scale_cap":
            max_scale = 1.20
        else:
            max_scale = 1.0

        # Approximate or misspecified deep posteriors may still inform
        # direction, but they cannot increase exposure beyond the guardrail.
        guarded_config = replace(
            base_config,
            predictive_max_multiplier=min(
                base_config.predictive_max_multiplier,
                max_scale,
            ),
            observational_max_multiplier=min(
                base_config.observational_max_multiplier,
                max_scale,
            ),
            experiment_max_multiplier=min(
                base_config.experiment_max_multiplier,
                max_scale,
            ),
        )
        final_engine = BayesTrafficEngine(
            guarded_config
        )
        all_actions, best_actions = final_engine.run(
            clean,
            posterior_overrides=overrides,
        )
    else:
        all_actions = candidate_all.copy()
        best_actions = candidate_best.copy()

    all_actions = all_actions.copy()
    best_actions = best_actions.copy()

    all_actions["deep_inference_method"] = diagnostics.method
    all_actions["deep_decision_source"] = decision_source
    all_actions["deep_guardrail"] = guardrail
    all_actions["deep_posterior_used"] = (
        decision_source
        != "empirical_bayes_fallback"
    )

    best_actions["deep_inference_method"] = diagnostics.method
    best_actions["deep_decision_source"] = decision_source
    best_actions["deep_guardrail"] = guardrail
    best_actions["deep_posterior_used"] = (
        decision_source
        != "empirical_bayes_fallback"
    )
    best_actions["mcmc_converged"] = diagnostics.converged
    best_actions["mcmc_max_rhat"] = diagnostics.max_rhat
    best_actions["mcmc_min_ess_bulk"] = diagnostics.min_ess_bulk
    best_actions["mcmc_divergences"] = diagnostics.divergences
    best_actions["ppc_status"] = ppc_summary.status
    best_actions["ppc_click_90_coverage"] = (
        ppc_summary.click_90_coverage
    )
    best_actions["ppc_conversion_90_coverage"] = (
        ppc_summary.conversion_90_coverage
    )

    candidate_all["deep_candidate_only"] = (
        decision_source
        != "mcmc_validated"
        and decision_source
        != "mcmc_advi_approximate"
    )

    if output_dir is not None:
        out = Path(output_dir)
        write_reports(
            all_actions,
            best_actions,
            out,
        )
        save_mcmc_result(
            idata,
            diagnostics,
            out,
            mapping=mapping,
        )
        ppc_detail.to_csv(
            out / "posterior_predictive_checks.csv",
            index=False,
        )
        pd.DataFrame(
            [ppc_summary.__dict__]
        ).to_csv(
            out / "posterior_predictive_summary.csv",
            index=False,
        )
        if decision_source != "mcmc_validated":
            candidate_all.to_csv(
                out / "mcmc_candidate_actions.csv",
                index=False,
            )

    return DeepAnalysisResult(
        all_actions=all_actions,
        best_actions=best_actions,
        diagnostics=diagnostics,
        idata=idata,
        mapping=mapping,
        ppc_detail=ppc_detail,
        ppc_summary=ppc_summary,
        decision_source=decision_source,
        guardrail=guardrail,
        candidate_mcmc_actions=candidate_all,
    )
