from quant_trafego.deep_analysis import deep_decision_policy
from quant_trafego.mcmc import MCMCDiagnostics
from quant_trafego.ppc import PPCSummary


def _diag(method: str, converged):
    return MCMCDiagnostics(
        method=method,
        n_campaigns=1,
        n_adsets=1,
        n_ads=2,
        n_days=30,
        n_observations=60,
        max_rhat=1.0 if converged else 1.2,
        min_ess_bulk=500 if converged else 50,
        divergences=0 if converged else 5,
        converged=converged,
    )


def _ppc(status: str):
    return PPCSummary(
        n_observations=60,
        n_ads=2,
        click_90_coverage=0.90,
        conversion_90_coverage=0.90,
        click_extreme_fraction=0.05,
        conversion_extreme_fraction=0.05,
        mean_abs_click_z=0.8,
        mean_abs_conversion_z=0.8,
        status=status,
    )


def test_failed_nuts_falls_back_to_empirical_bayes():
    source, guardrail = deep_decision_policy(
        _diag("nuts", False),
        _ppc("pass"),
    )
    assert source == "empirical_bayes_fallback"
    assert guardrail == "nuts_diagnostics_failed"


def test_ppc_warning_blocks_scale_even_when_posterior_exists():
    source, guardrail = deep_decision_policy(
        _diag("advi", None),
        _ppc("warning"),
    )
    assert source == "mcmc_posterior_guarded"
    assert guardrail == "posterior_predictive_check_not_passed"


def test_valid_nuts_is_allowed_to_drive_decisions():
    source, guardrail = deep_decision_policy(
        _diag("nuts", True),
        _ppc("pass"),
    )
    assert source == "mcmc_validated"
    assert guardrail == "none"
