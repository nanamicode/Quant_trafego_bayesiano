from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import pandas as pd

from .model import BetaPosterior


@dataclass(frozen=True)
class MCMCDiagnostics:
    method: str
    n_campaigns: int
    n_adsets: int
    n_ads: int
    max_rhat: float | None
    min_ess_bulk: float | None
    divergences: int | None
    converged: bool | None


def _safe_logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _prepare(df: pd.DataFrame):
    data = (
        df.groupby(["campaign_id", "adset_id", "ad_id"], as_index=False)
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
        )
        .copy()
    )

    campaign_codes, campaign_uniques = pd.factorize(data["campaign_id"], sort=True)
    adset_key = data["campaign_id"].astype(str) + "::" + data["adset_id"].astype(str)
    adset_codes, adset_uniques = pd.factorize(adset_key, sort=True)
    ad_key = adset_key + "::" + data["ad_id"].astype(str)
    ad_codes, ad_uniques = pd.factorize(ad_key, sort=True)

    data["campaign_idx"] = campaign_codes
    data["adset_idx"] = adset_codes
    data["ad_idx"] = ad_codes

    campaign_map = (
        data[["campaign_idx", "campaign_id"]]
        .drop_duplicates()
        .sort_values("campaign_idx")
        .reset_index(drop=True)
    )
    adset_map = (
        data[["adset_idx", "campaign_idx", "adset_id"]]
        .drop_duplicates()
        .sort_values("adset_idx")
        .reset_index(drop=True)
    )
    ad_map = (
        data[["ad_idx", "campaign_idx", "adset_idx", "ad_id"]]
        .drop_duplicates()
        .sort_values("ad_idx")
        .reset_index(drop=True)
    )

    mapping = {
        "account": ["ALL"],
        "campaign": campaign_map["campaign_id"].astype(str).tolist(),
        "adset": adset_map["adset_id"].astype(str).tolist(),
        "ad": ad_map["ad_id"].astype(str).tolist(),
        "adset_campaign_idx": adset_map["campaign_idx"].to_numpy(dtype=int),
        "ad_adset_idx": ad_map["adset_idx"].to_numpy(dtype=int),
    }
    return data, campaign_uniques, adset_uniques, ad_uniques, mapping


def fit_hierarchical_funnel(
    df: pd.DataFrame,
    *,
    draws: int = 1200,
    tune: int = 1200,
    chains: int = 4,
    cores: int | None = None,
    target_accept: float = 0.92,
    seed: int = 42,
    method: str = "auto",
    advi_steps: int = 40_000,
    return_mapping: bool = False,
):
    """
    Full hierarchical Bayesian funnel for CTR and CVR.

    Non-centered random effects are estimated at account, campaign, ad-set and
    ad level. The model emits posterior rates for every hierarchy level so the
    decision engine can consume MCMC uncertainty directly.

    method='auto' uses NUTS for up to 300 ads and ADVI above that threshold.
    """
    try:
        import pymc as pm
        import arviz as az
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Modo MCMC requer dependências profundas. Instale com: pip install -e .[deep]"
        ) from exc

    data, campaigns, adsets, ads, mapping = _prepare(df)

    n_campaigns = len(campaigns)
    n_adsets = len(adsets)
    n_ads = len(ads)

    impressions = data["impressions"].to_numpy(dtype=int)
    clicks = data["clicks"].to_numpy(dtype=int)
    conversions = data["conversions"].to_numpy(dtype=int)
    campaign_idx = data["campaign_idx"].to_numpy(dtype=int)
    adset_idx = data["adset_idx"].to_numpy(dtype=int)
    ad_idx = data["ad_idx"].to_numpy(dtype=int)

    adset_campaign_idx = mapping["adset_campaign_idx"]
    ad_adset_idx = mapping["ad_adset_idx"]

    global_ctr = clicks.sum() / max(impressions.sum(), 1)
    global_cvr = conversions.sum() / max(clicks.sum(), 1)

    coords = {
        "campaign": campaigns.astype(str).tolist(),
        "adset": adsets.astype(str).tolist(),
        "ad": ads.astype(str).tolist(),
        "entity": np.arange(len(data)),
    }

    with pm.Model(coords=coords) as model:
        campaign_i = pm.Data("campaign_i", campaign_idx, dims="entity")
        adset_i = pm.Data("adset_i", adset_idx, dims="entity")
        ad_i = pm.Data("ad_i", ad_idx, dims="entity")
        adset_campaign_i = pm.Data(
            "adset_campaign_i",
            adset_campaign_idx,
            dims="adset",
        )
        ad_adset_i = pm.Data(
            "ad_adset_i",
            ad_adset_idx,
            dims="ad",
        )

        def funnel(prefix: str, base_rate: float):
            mu = pm.Normal(f"{prefix}_mu", mu=_safe_logit(base_rate), sigma=1.0)

            sigma_campaign = pm.HalfNormal(f"{prefix}_sigma_campaign", sigma=0.7)
            sigma_adset = pm.HalfNormal(f"{prefix}_sigma_adset", sigma=0.6)
            sigma_ad = pm.HalfNormal(f"{prefix}_sigma_ad", sigma=0.6)

            z_campaign = pm.Normal(f"{prefix}_z_campaign", 0, 1, dims="campaign")
            z_adset = pm.Normal(f"{prefix}_z_adset", 0, 1, dims="adset")
            z_ad = pm.Normal(f"{prefix}_z_ad", 0, 1, dims="ad")

            eta_campaign = mu + sigma_campaign * z_campaign
            eta_adset = (
                eta_campaign[adset_campaign_i]
                + sigma_adset * z_adset
            )
            eta_ad = (
                eta_adset[ad_adset_i]
                + sigma_ad * z_ad
            )
            eta_entity = eta_ad[ad_i]

            pm.Deterministic(
                f"{prefix}_p_account",
                pm.math.sigmoid(mu),
            )
            pm.Deterministic(
                f"{prefix}_p_campaign",
                pm.math.sigmoid(eta_campaign),
                dims="campaign",
            )
            pm.Deterministic(
                f"{prefix}_p_adset",
                pm.math.sigmoid(eta_adset),
                dims="adset",
            )
            pm.Deterministic(
                f"{prefix}_p_ad",
                pm.math.sigmoid(eta_ad),
                dims="ad",
            )
            return pm.math.sigmoid(eta_entity)

        p_ctr_entity = funnel("ctr", global_ctr)
        p_cvr_entity = funnel("cvr", global_cvr)

        pm.Binomial(
            "clicks_obs",
            n=impressions,
            p=p_ctr_entity,
            observed=clicks,
            dims="entity",
        )
        pm.Binomial(
            "conversions_obs",
            n=clicks,
            p=p_cvr_entity,
            observed=conversions,
            dims="entity",
        )

        selected = method
        if method == "auto":
            selected = "nuts" if n_ads <= 300 else "advi"

        if selected == "nuts":
            use_cores = cores or max(1, min(chains, (os.cpu_count() or 2) - 1))
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=use_cores,
                target_accept=target_accept,
                random_seed=seed,
                return_inferencedata=True,
            )
        elif selected == "advi":
            approx = pm.fit(
                n=advi_steps,
                method="advi",
                random_seed=seed,
            )
            idata = approx.sample(draws=draws, return_inferencedata=True)
        else:
            raise ValueError("method deve ser 'auto', 'nuts' ou 'advi'.")

    max_rhat = None
    min_ess = None
    divergences = None
    converged = None

    if selected == "nuts":
        summary = az.summary(
            idata,
            var_names=[
                "ctr_mu",
                "cvr_mu",
                "ctr_sigma_campaign",
                "ctr_sigma_adset",
                "ctr_sigma_ad",
                "cvr_sigma_campaign",
                "cvr_sigma_adset",
                "cvr_sigma_ad",
            ],
            kind="diagnostics",
        )
        if "r_hat" in summary:
            max_rhat = float(np.nanmax(summary["r_hat"].to_numpy()))
        if "ess_bulk" in summary:
            min_ess = float(np.nanmin(summary["ess_bulk"].to_numpy()))
        if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
            divergences = int(idata.sample_stats["diverging"].sum().item())
        converged = bool(
            (max_rhat is None or max_rhat <= 1.01)
            and (min_ess is None or min_ess >= 400)
            and (divergences is None or divergences == 0)
        )

    diagnostics = MCMCDiagnostics(
        method=selected,
        n_campaigns=n_campaigns,
        n_adsets=n_adsets,
        n_ads=n_ads,
        max_rhat=max_rhat,
        min_ess_bulk=min_ess,
        divergences=divergences,
        converged=converged,
    )
    if return_mapping:
        return idata, diagnostics, mapping
    return idata, diagnostics


def _moment_match_beta(samples: np.ndarray) -> BetaPosterior:
    x = np.asarray(samples, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return BetaPosterior(1.0, 1.0)

    mean = float(np.clip(x.mean(), 1e-8, 1 - 1e-8))
    var = float(max(x.var(ddof=1) if len(x) > 1 else 0.0, 1e-12))
    max_var = mean * (1 - mean)
    if var >= max_var:
        strength = 2.0
    else:
        strength = max(mean * (1 - mean) / var - 1.0, 2.0)

    strength = float(np.clip(strength, 2.0, 10_000_000.0))
    return BetaPosterior(
        alpha=max(mean * strength, 1e-6),
        beta=max((1 - mean) * strength, 1e-6),
    )


def posterior_rate_overrides(idata, mapping) -> dict:
    """
    Convert MCMC posterior rate samples into Beta moment-matched distributions
    that can be consumed by the Monte Carlo decision engine.
    """
    overrides: dict[tuple[str, str], tuple[BetaPosterior, BetaPosterior]] = {}

    account_ctr = idata.posterior["ctr_p_account"].values
    account_cvr = idata.posterior["cvr_p_account"].values
    overrides[("account", "ALL")] = (
        _moment_match_beta(account_ctr),
        _moment_match_beta(account_cvr),
    )

    for level, dim in [("campaign", "campaign"), ("adset", "adset"), ("ad", "ad")]:
        ctr = idata.posterior[f"ctr_p_{level}"]
        cvr = idata.posterior[f"cvr_p_{level}"]
        ids = mapping[level]

        for i, entity_id in enumerate(ids):
            ctr_samples = ctr.isel({dim: i}).values
            cvr_samples = cvr.isel({dim: i}).values
            overrides[(level, str(entity_id))] = (
                _moment_match_beta(ctr_samples),
                _moment_match_beta(cvr_samples),
            )

    return overrides


def save_mcmc_result(
    idata,
    diagnostics: MCMCDiagnostics,
    output_dir: str | Path,
    mapping: dict | None = None,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(out / "hierarchical_funnel.nc")
    pd.DataFrame([diagnostics.__dict__]).to_csv(
        out / "mcmc_diagnostics.csv",
        index=False,
    )
    if mapping is not None:
        rows = []
        for level in ["account", "campaign", "adset", "ad"]:
            for entity_id in mapping[level]:
                rows.append({"level": level, "entity_id": str(entity_id)})
        pd.DataFrame(rows).to_csv(out / "mcmc_entity_mapping.csv", index=False)
