from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd

from .model import BetaPosterior, aggregate, shrink_to, simulate_action, update_beta
from .response import ResponseEstimate, estimate_response
from .temporal import analyze_temporal


REQUIRED = [
    "date",
    "campaign_id",
    "adset_id",
    "ad_id",
    "impressions",
    "clicks",
    "conversions",
    "spend",
    "revenue",
]


@dataclass
class EngineConfig:
    target_roas: float = 2.0
    contribution_margin: float = 1.0
    horizon_days: int = 7
    draws: int = 30000
    seed: int = 42
    risk_aversion: float = 0.25
    actions: tuple[float, ...] = (0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0)
    saturation_half: float = 1.5
    saturation_slope: float = 1.3
    temporal_half_life_days: float = 14.0
    temporal_recent_days: int = 7
    use_temporal: bool = True
    use_empirical_response: bool = True

    global_ctr_strength: float = 2500
    global_cvr_strength: float = 250
    campaign_ctr_strength: float = 1600
    campaign_cvr_strength: float = 160
    adset_ctr_strength: float = 900
    adset_cvr_strength: float = 100
    ad_ctr_strength: float = 350
    ad_cvr_strength: float = 60


class BayesTrafficEngine:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()
        if not 0.0 <= self.config.contribution_margin <= 1.0:
            raise ValueError("contribution_margin deve estar entre 0 e 1.")
        self.rng = np.random.default_rng(self.config.seed)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(
                "Colunas obrigatórias não identificadas: "
                + ", ".join(missing)
                + ". Renomeie-as ou ajuste os aliases em src/quant_trafego/io.py."
            )

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        if out["date"].isna().any():
            raise ValueError("Há datas inválidas na planilha.")

        for col in ["impressions", "clicks", "conversions", "spend", "revenue"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
            if (out[col] < 0).any():
                raise ValueError(f"A coluna {col} contém valores negativos.")

        if (out["clicks"] > out["impressions"]).any():
            raise ValueError("Existem linhas onde cliques > impressões.")
        if (out["conversions"] > out["clicks"]).any():
            raise ValueError("Neste modelo de funil, conversões não podem exceder cliques.")

        return out

    @staticmethod
    def _stable_offset(level: str, entity_id: str) -> int:
        digest = hashlib.blake2b(
            f"{level}|{entity_id}".encode("utf-8"),
            digest_size=4,
        ).digest()
        return int.from_bytes(digest, "little") % 100_000

    @staticmethod
    def _get_override(overrides, level, entity_id, fallback):
        if not overrides:
            return fallback, "empirical_bayes"
        key = (level, str(entity_id))
        if key in overrides:
            return overrides[key], "mcmc"
        return fallback, "empirical_bayes"

    def _global_posteriors(self, df: pd.DataFrame):
        s = aggregate(df)
        ctr_prior = BetaPosterior(
            1 + s["ctr"] * self.config.global_ctr_strength,
            1 + (1 - s["ctr"]) * self.config.global_ctr_strength,
        )
        cvr_prior = BetaPosterior(
            1 + s["cvr"] * self.config.global_cvr_strength,
            1 + (1 - s["cvr"]) * self.config.global_cvr_strength,
        )
        return (
            update_beta(ctr_prior, s["clicks"], s["impressions"]),
            update_beta(cvr_prior, s["conversions"], s["clicks"]),
        )

    def _posterior(self, df, parent_ctr, parent_cvr, ctr_strength, cvr_strength):
        s = aggregate(df)
        ctr = update_beta(
            shrink_to(parent_ctr, ctr_strength),
            s["clicks"],
            s["impressions"],
        )
        cvr = update_beta(
            shrink_to(parent_cvr, cvr_strength),
            s["conversions"],
            s["clicks"],
        )
        return ctr, cvr

    def _evaluate_entity(
        self,
        level,
        entity_id,
        df,
        ctr_post,
        cvr_post,
        response_estimate: ResponseEstimate,
        posterior_source: str,
    ):
        stats = aggregate(df)
        temporal = analyze_temporal(
            df,
            half_life_days=self.config.temporal_half_life_days,
            recent_days=self.config.temporal_recent_days,
            seed=(
                self.config.seed
                + self._stable_offset(level, str(entity_id))
            ),
        )

        ctr_mean = temporal.ctr.effective_mean if self.config.use_temporal else 0.0
        ctr_sd = temporal.ctr.effective_sd if self.config.use_temporal else 0.0
        cvr_mean = temporal.cvr.effective_mean if self.config.use_temporal else 0.0
        cvr_sd = temporal.cvr.effective_sd if self.config.use_temporal else 0.0

        response_confidence = (
            response_estimate.confidence if self.config.use_empirical_response else 0.0
        )

        sims = []
        for action in self.config.actions:
            sims.append(
                simulate_action(
                    stats=stats,
                    ctr_post=ctr_post,
                    cvr_post=cvr_post,
                    multiplier=action,
                    draws=self.config.draws,
                    horizon_days=self.config.horizon_days,
                    target_roas=self.config.target_roas,
                    contribution_margin=self.config.contribution_margin,
                    rng=self.rng,
                    saturation_half=self.config.saturation_half,
                    saturation_slope=self.config.saturation_slope,
                    temporal_ctr_slope_mean=ctr_mean,
                    temporal_ctr_slope_sd=ctr_sd,
                    temporal_cvr_slope_mean=cvr_mean,
                    temporal_cvr_slope_sd=cvr_sd,
                    response_elasticity_mean=response_estimate.elasticity_mean,
                    response_elasticity_sd=response_estimate.elasticity_sd,
                    response_confidence=response_confidence,
                )
            )

        hold = next(x for x in sims if x["multiplier"] == 1.0)
        hold_draws = hold["_profit_draws"]
        profit_matrix = np.vstack([x["_profit_draws"] for x in sims])
        best_idx = np.argmax(profit_matrix, axis=0)
        best_draws = np.max(profit_matrix, axis=0)

        rows = []
        for i, sim in enumerate(sims):
            profit_draws = sim["_profit_draws"]
            regret = float(np.mean(best_draws - profit_draws))
            downside = max(0.0, -sim["cvar10_profit"])
            instability_penalty = (
                temporal.instability_score * abs(sim["expected_profit"]) * 0.10
            )
            utility = (
                sim["expected_profit"]
                - self.config.risk_aversion * downside
                - self.config.risk_aversion * 0.15 * regret
                - self.config.risk_aversion * instability_penalty
            )

            incremental = profit_draws - hold_draws
            rows.append({
                "level": level,
                "entity_id": str(entity_id),
                "posterior_source": posterior_source,
                "historical_days": stats["days"],
                "historical_spend": stats["spend"],
                "historical_revenue": stats["revenue"],
                "historical_roas": stats["roas"],
                "historical_ctr": stats["ctr"],
                "historical_cvr": stats["cvr"],
                "posterior_ctr_mean": ctr_post.mean,
                "posterior_cvr_mean": cvr_post.mean,
                "posterior_ctr_strength": ctr_post.strength,
                "posterior_cvr_strength": cvr_post.strength,
                "ctr_logit_derivative_per_day": temporal.ctr.mean,
                "ctr_trend_confidence": temporal.ctr.confidence,
                "cvr_logit_derivative_per_day": temporal.cvr.mean,
                "cvr_trend_confidence": temporal.cvr.confidence,
                "p_recent_ctr_better": temporal.p_recent_ctr_better,
                "p_recent_cvr_better": temporal.p_recent_cvr_better,
                "regime_change_score": temporal.regime_change_score,
                "instability_score": temporal.instability_score,
                "response_elasticity": response_estimate.elasticity_mean,
                "response_elasticity_sd": response_estimate.elasticity_sd,
                "response_confidence": response_estimate.confidence,
                "p_diminishing_returns_proxy": (
                    response_estimate.diminishing_returns_probability_proxy
                ),
                "contribution_margin": self.config.contribution_margin,
                "action_multiplier": sim["multiplier"],
                "expected_spend": sim["expected_spend"],
                "expected_revenue": sim["expected_revenue"],
                "expected_profit": sim["expected_profit"],
                "expected_roas": sim["expected_roas"],
                "p_profit": sim["p_profit"],
                "p_ruin": 1.0 - sim["p_profit"],
                "p_roas_target": sim["p_roas_target"],
                "p_beats_hold": float(np.mean(profit_draws > hold_draws)),
                "p_action_optimal": float(np.mean(best_idx == i)),
                "expected_incremental_profit_vs_hold": float(np.mean(incremental)),
                "p_incremental_profit_positive": float(np.mean(incremental > 0)),
                "var10_profit": sim["var10_profit"],
                "cvar10_profit": sim["cvar10_profit"],
                "expected_regret": regret,
                "risk_adjusted_utility": float(utility),
            })
        return rows

    def run(
        self,
        df: pd.DataFrame,
        *,
        posterior_overrides: dict | None = None,
    ):
        df = self.validate(df)
        rows = []

        global_fallback = self._global_posteriors(df)
        (global_ctr, global_cvr), global_source = self._get_override(
            posterior_overrides,
            "account",
            "ALL",
            global_fallback,
        )
        global_response = estimate_response(df)
        rows.extend(
            self._evaluate_entity(
                "account",
                "ALL",
                df,
                global_ctr,
                global_cvr,
                global_response,
                global_source,
            )
        )

        campaign_posts = {}
        campaign_response = {}
        campaign_source = {}
        for campaign_id, cdf in df.groupby("campaign_id", sort=False):
            fallback = self._posterior(
                cdf,
                global_ctr,
                global_cvr,
                self.config.campaign_ctr_strength,
                self.config.campaign_cvr_strength,
            )
            posts, source = self._get_override(
                posterior_overrides,
                "campaign",
                campaign_id,
                fallback,
            )
            response = estimate_response(cdf, parent=global_response)
            campaign_posts[campaign_id] = posts
            campaign_response[campaign_id] = response
            campaign_source[campaign_id] = source
            rows.extend(
                self._evaluate_entity(
                    "campaign",
                    campaign_id,
                    cdf,
                    *posts,
                    response,
                    source,
                )
            )

        adset_posts = {}
        adset_response = {}
        for (campaign_id, adset_id), sdf in df.groupby(
            ["campaign_id", "adset_id"],
            sort=False,
        ):
            fallback = self._posterior(
                sdf,
                *campaign_posts[campaign_id],
                self.config.adset_ctr_strength,
                self.config.adset_cvr_strength,
            )
            posts, source = self._get_override(
                posterior_overrides,
                "adset",
                adset_id,
                fallback,
            )
            response = estimate_response(
                sdf,
                parent=campaign_response[campaign_id],
            )
            adset_posts[(campaign_id, adset_id)] = posts
            adset_response[(campaign_id, adset_id)] = response
            rows.extend(
                self._evaluate_entity(
                    "adset",
                    adset_id,
                    sdf,
                    *posts,
                    response,
                    source,
                )
            )

        for (campaign_id, adset_id, ad_id), adf in df.groupby(
            ["campaign_id", "adset_id", "ad_id"],
            sort=False,
        ):
            fallback = self._posterior(
                adf,
                *adset_posts[(campaign_id, adset_id)],
                self.config.ad_ctr_strength,
                self.config.ad_cvr_strength,
            )
            posts, source = self._get_override(
                posterior_overrides,
                "ad",
                ad_id,
                fallback,
            )
            response = estimate_response(
                adf,
                parent=adset_response[(campaign_id, adset_id)],
            )
            rows.extend(
                self._evaluate_entity(
                    "ad",
                    ad_id,
                    adf,
                    *posts,
                    response,
                    source,
                )
            )

        all_actions = pd.DataFrame(rows)
        idx = all_actions.groupby(["level", "entity_id"])["risk_adjusted_utility"].idxmax()
        best = all_actions.loc[idx].copy().reset_index(drop=True)

        confidence = (
            0.40 * best["p_profit"]
            + 0.25 * best["p_action_optimal"]
            + 0.20 * best["p_roas_target"]
            + 0.15 * (1.0 - best["instability_score"])
        )
        best["decision_confidence"] = np.clip(confidence, 0.0, 1.0)
        best["opportunity_score"] = (
            best["risk_adjusted_utility"]
            * (0.25 + 0.75 * best["decision_confidence"])
            / (1.0 + np.maximum(best["expected_regret"], 0.0))
        )
        best = best.sort_values(
            ["level", "opportunity_score"],
            ascending=[True, False],
        ).reset_index(drop=True)

        return all_actions, best
