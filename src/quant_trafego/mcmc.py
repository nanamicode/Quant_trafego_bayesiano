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
    n_days: int
    n_observations: int
    max_rhat: float | None
    min_ess_bulk: float | None
    divergences: int | None
    converged: bool | None


def _safe_logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _prepare(df: pd.DataFrame):
    data = (
        df.groupby(
            ["date", "campaign_id", "adset_id", "ad_id"],
            as_index=False,
        )
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
        )
        .copy()
    )
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(
        ["date", "campaign_id", "adset_id", "ad_id"]
    ).reset_index(drop=True)

    campaign_codes, campaign_uniques = pd.factorize(
        data["campaign_id"], sort=True
    )
    adset_key = (
        data["campaign_id"].astype(str)
        + "::"
        + data["adset_id"].astype(str)
    )
    adset_codes, adset_uniques = pd.factorize(adset_key, sort=True)
    ad_key = adset_key + "::" + data["ad_id"].astype(str)
    ad_codes, ad_uniques = pd.factorize(ad_key, sort=True)
    date_codes, date_uniques = pd.factorize(data["date"], sort=True)

    data["campaign_idx"] = campaign_codes
    data["adset_idx"] = adset_codes
    data["ad_idx"] = ad_codes
    data["date_idx"] = date_codes

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
        "date": [str(pd.Timestamp(x)) for x in date_uniques],
        "adset_campaign_idx": adset_map["campaign_idx"].to_numpy(dtype=int),
        "ad_adset_idx": ad_map["adset_idx"].to_numpy(dtype=int),
        "observation_date": data["date"].astype(str).tolist(),
        "observation_ad": data["ad_id"].astype(str).tolist(),
    }
    return (
        data,
        campaign_uniques,
        adset_uniques,
        ad_uniques,
        date_uniques,
        mapping,
    )


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
    include_global_time_effect: bool = True,
    return_mapping: bool = False,
):
    """
    Hierarchical Bayesian daily funnel for CTR and CVR.

    Observations remain daily. Campaign, ad-set and ad effects are non-centered.
    An optional global local-level time process captures account-wide temporal
    movement without pretending each low-volume ad has an independently
    identifiable daily latent state.

    method='auto' uses NUTS for moderate structures and ADVI otherwise.
    """
    try:
        import pymc as pm
        import arviz as az
        import pytensor.tensor as pt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Modo MCMC requer dependências profundas. "
            "Instale com: uv sync --extra deep"
        ) from exc

    (
        data,
        campaigns,
        adsets,
        ads,
        dates,
        mapping,
    ) = _prepare(df)

    n_campaigns = len(campaigns)
    n_adsets = len(adsets)
    n_ads = len(ads)
    n_days = len(dates)
    n_observations = len(data)

    impressions = data["impressions"].to_numpy(dtype=int)
    clicks = data["clicks"].to_numpy(dtype=int)
    conversions = data["conversions"].to_numpy(dtype=int)
    campaign_idx = data["campaign_idx"].to_numpy(dtype=int)
    adset_idx = data["adset_idx"].to_numpy(dtype=int)
    ad_idx = data["ad_idx"].to_numpy(dtype=int)
    date_idx = data["date_idx"].to_numpy(dtype=int)

    ad_clicks = np.bincount(
        ad_idx,
        weights=clicks,
        minlength=n_ads,
    ).astype(int)
    ad_conversions = np.bincount(
        ad_idx,
        weights=conversions,
        minlength=n_ads,
    ).astype(int)
    if np.any(ad_conversions > ad_clicks):
        raise ValueError(
            "Há anúncios em que conversões agregadas excedem cliques "
            "agregados; o MCMC de propensão clique→compra não é válido "
            "para essa base sem um modelo de atribuição por exposição."
        )

    adset_campaign_idx = mapping["adset_campaign_idx"]
    ad_adset_idx = mapping["ad_adset_idx"]

    global_ctr = clicks.sum() / max(impressions.sum(), 1)
    global_cvr = conversions.sum() / max(clicks.sum(), 1)

    coords = {
        "campaign": campaigns.astype(str).tolist(),
        "adset": adsets.astype(str).tolist(),
        "ad": ads.astype(str).tolist(),
        "date": [str(pd.Timestamp(x)) for x in dates],
        "entity": np.arange(n_observations),
    }

    with pm.Model(coords=coords) as model:
        campaign_i = pm.Data(
            "campaign_i", campaign_idx, dims="entity"
        )
        adset_i = pm.Data(
            "adset_i", adset_idx, dims="entity"
        )
        ad_i = pm.Data(
            "ad_i", ad_idx, dims="entity"
        )
        date_i = pm.Data(
            "date_i", date_idx, dims="entity"
        )
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

        def funnel(prefix: str, base_rate: float, *, use_time_effect: bool = True):
            mu = pm.Normal(
                f"{prefix}_mu",
                mu=_safe_logit(base_rate),
                sigma=1.0,
            )
            sigma_campaign = pm.HalfNormal(
                f"{prefix}_sigma_campaign", sigma=0.7
            )
            sigma_adset = pm.HalfNormal(
                f"{prefix}_sigma_adset", sigma=0.6
            )
            sigma_ad = pm.HalfNormal(
                f"{prefix}_sigma_ad", sigma=0.6
            )

            z_campaign = pm.Normal(
                f"{prefix}_z_campaign", 0, 1, dims="campaign"
            )
            z_adset = pm.Normal(
                f"{prefix}_z_adset", 0, 1, dims="adset"
            )
            z_ad = pm.Normal(
                f"{prefix}_z_ad", 0, 1, dims="ad"
            )

            eta_campaign = mu + sigma_campaign * z_campaign
            eta_adset = (
                eta_campaign[adset_campaign_i]
                + sigma_adset * z_adset
            )
            eta_ad = (
                eta_adset[ad_adset_i]
                + sigma_ad * z_ad
            )

            if include_global_time_effect and use_time_effect and n_days >= 4:
                sigma_time = pm.HalfNormal(
                    f"{prefix}_sigma_time",
                    sigma=0.12,
                )
                raw_time = pm.GaussianRandomWalk(
                    f"{prefix}_time_raw",
                    sigma=sigma_time,
                    init_dist=pm.Normal.dist(0.0, 0.10),
                    dims="date",
                )
                centered_time = raw_time - pt.mean(raw_time)
                time_effect = pm.Deterministic(
                    f"{prefix}_time_effect",
                    centered_time,
                    dims="date",
                )
            else:
                time_effect = pt.zeros(n_days)

            eta_entity = eta_ad[ad_i] + time_effect[date_i]
            current_time = time_effect[-1]

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

            pm.Deterministic(
                f"{prefix}_p_account_current",
                pm.math.sigmoid(mu + current_time),
            )
            pm.Deterministic(
                f"{prefix}_p_campaign_current",
                pm.math.sigmoid(eta_campaign + current_time),
                dims="campaign",
            )
            pm.Deterministic(
                f"{prefix}_p_adset_current",
                pm.math.sigmoid(eta_adset + current_time),
                dims="adset",
            )
            pm.Deterministic(
                f"{prefix}_p_ad_current",
                pm.math.sigmoid(eta_ad + current_time),
                dims="ad",
            )
            pm.Deterministic(
                f"{prefix}_p_entity",
                pm.math.sigmoid(eta_entity),
                dims="entity",
            )
            return pm.math.sigmoid(eta_entity)

        p_ctr_entity = funnel(
            "ctr",
            global_ctr,
            use_time_effect=True,
        )
        # Meta Ads daily conversions can be attributed to earlier clicks or
        # view-through exposure. We therefore estimate the hierarchical CVR
        # likelihood on each ad's full-period click/conversion totals instead
        # of forcing a false same-day Binomial funnel.
        funnel(
            "cvr",
            global_cvr,
            use_time_effect=False,
        )

        pm.Binomial(
            "clicks_obs",
            n=impressions,
            p=p_ctr_entity,
            observed=clicks,
            dims="entity",
        )
        pm.Binomial(
            "conversions_obs",
            n=ad_clicks,
            p=model["cvr_p_ad"],
            observed=ad_conversions,
            dims="ad",
        )

        selected = method
        if method == "auto":
            moderate = (
                n_ads <= 250
                and n_days <= 180
                and n_observations <= 20_000
            )
            selected = "nuts" if moderate else "advi"

        if selected == "nuts":
            use_cores = cores or max(
                1,
                min(chains, (os.cpu_count() or 2) - 1),
            )
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
            idata = approx.sample(
                draws=draws,
                return_inferencedata=True,
            )
        else:
            raise ValueError(
                "method deve ser 'auto', 'nuts' ou 'advi'."
            )

    max_rhat = None
    min_ess = None
    divergences = None
    converged = None

    if selected == "nuts":
        var_names = [
            "ctr_mu",
            "cvr_mu",
            "ctr_sigma_campaign",
            "ctr_sigma_adset",
            "ctr_sigma_ad",
            "cvr_sigma_campaign",
            "cvr_sigma_adset",
            "cvr_sigma_ad",
        ]
        if include_global_time_effect and n_days >= 4:
            var_names += ["ctr_sigma_time"]

        summary = az.summary(
            idata,
            var_names=var_names,
            kind="diagnostics",
        )
        if "r_hat" in summary:
            max_rhat = float(
                np.nanmax(summary["r_hat"].to_numpy())
            )
        if "ess_bulk" in summary:
            min_ess = float(
                np.nanmin(summary["ess_bulk"].to_numpy())
            )
        if (
            hasattr(idata, "sample_stats")
            and "diverging" in idata.sample_stats
        ):
            divergences = int(
                idata.sample_stats["diverging"].sum().item()
            )
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
        n_days=n_days,
        n_observations=n_observations,
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
    var = float(
        max(
            x.var(ddof=1) if len(x) > 1 else 0.0,
            1e-12,
        )
    )
    max_var = mean * (1 - mean)
    if var >= max_var:
        strength = 2.0
    else:
        strength = max(
            mean * (1 - mean) / var - 1.0,
            2.0,
        )

    strength = float(
        np.clip(strength, 2.0, 10_000_000.0)
    )
    return BetaPosterior(
        alpha=max(mean * strength, 1e-6),
        beta=max((1 - mean) * strength, 1e-6),
    )


def posterior_rate_overrides(idata, mapping) -> dict:
    """
    Convert current-state MCMC posterior rate samples into Beta
    moment-matched distributions consumed by the decision engine.
    """
    overrides: dict[
        tuple[str, str],
        tuple[BetaPosterior, BetaPosterior],
    ] = {}

    def variable(prefix: str, level: str):
        current = f"{prefix}_p_{level}_current"
        fallback = f"{prefix}_p_{level}"
        return (
            idata.posterior[current]
            if current in idata.posterior
            else idata.posterior[fallback]
        )

    account_ctr = variable("ctr", "account").values
    account_cvr = variable("cvr", "account").values
    overrides[("account", "ALL")] = (
        _moment_match_beta(account_ctr),
        _moment_match_beta(account_cvr),
    )

    for level, dim in [
        ("campaign", "campaign"),
        ("adset", "adset"),
        ("ad", "ad"),
    ]:
        ctr = variable("ctr", level)
        cvr = variable("cvr", level)
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
        for level in [
            "account",
            "campaign",
            "adset",
            "ad",
        ]:
            for entity_id in mapping[level]:
                rows.append(
                    {
                        "level": level,
                        "entity_id": str(entity_id),
                    }
                )
        pd.DataFrame(rows).to_csv(
            out / "mcmc_entity_mapping.csv",
            index=False,
        )
