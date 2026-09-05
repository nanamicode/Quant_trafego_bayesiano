from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .model import BetaPosterior, aggregate, shrink_to, simulate_action, update_beta


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
            raise ValueError("Neste MVP, conversões não podem exceder cliques.")

        return out

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
        ctr = update_beta(shrink_to(parent_ctr, ctr_strength), s["clicks"], s["impressions"])
        cvr = update_beta(shrink_to(parent_cvr, cvr_strength), s["conversions"], s["clicks"])
        return ctr, cvr

    def _evaluate_entity(self, level, entity_id, df, ctr_post, cvr_post):
        stats = aggregate(df)
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
                )
            )

        hold = next(x for x in sims if x["multiplier"] == 1.0)
        hold_draws = hold["_profit_draws"]
        best_draws = np.maximum.reduce([x["_profit_draws"] for x in sims])

        rows = []
        for sim in sims:
            profit_draws = sim["_profit_draws"]
            downside = max(0.0, -sim["cvar10_profit"])
            utility = sim["expected_profit"] - self.config.risk_aversion * downside
            rows.append({
                "level": level,
                "entity_id": str(entity_id),
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
                "contribution_margin": self.config.contribution_margin,
                "action_multiplier": sim["multiplier"],
                "expected_spend": sim["expected_spend"],
                "expected_revenue": sim["expected_revenue"],
                "expected_profit": sim["expected_profit"],
                "expected_roas": sim["expected_roas"],
                "p_profit": sim["p_profit"],
                "p_roas_target": sim["p_roas_target"],
                "p_beats_hold": float(np.mean(profit_draws > hold_draws)),
                "var10_profit": sim["var10_profit"],
                "cvar10_profit": sim["cvar10_profit"],
                "expected_regret": float(np.mean(best_draws - profit_draws)),
                "risk_adjusted_utility": float(utility),
            })
        return rows

    def run(self, df: pd.DataFrame):
        df = self.validate(df)
        rows = []

        global_ctr, global_cvr = self._global_posteriors(df)
        rows.extend(self._evaluate_entity("account", "ALL", df, global_ctr, global_cvr))

        campaign_posts = {}
        for campaign_id, cdf in df.groupby("campaign_id", sort=False):
            posts = self._posterior(
                cdf,
                global_ctr,
                global_cvr,
                self.config.campaign_ctr_strength,
                self.config.campaign_cvr_strength,
            )
            campaign_posts[campaign_id] = posts
            rows.extend(self._evaluate_entity("campaign", campaign_id, cdf, *posts))

        adset_posts = {}
        for (campaign_id, adset_id), sdf in df.groupby(["campaign_id", "adset_id"], sort=False):
            posts = self._posterior(
                sdf,
                *campaign_posts[campaign_id],
                self.config.adset_ctr_strength,
                self.config.adset_cvr_strength,
            )
            adset_posts[(campaign_id, adset_id)] = posts
            rows.extend(self._evaluate_entity("adset", adset_id, sdf, *posts))

        for (campaign_id, adset_id, ad_id), adf in df.groupby(
            ["campaign_id", "adset_id", "ad_id"], sort=False
        ):
            posts = self._posterior(
                adf,
                *adset_posts[(campaign_id, adset_id)],
                self.config.ad_ctr_strength,
                self.config.ad_cvr_strength,
            )
            rows.extend(self._evaluate_entity("ad", ad_id, adf, *posts))

        all_actions = pd.DataFrame(rows)
        idx = all_actions.groupby(["level", "entity_id"])["risk_adjusted_utility"].idxmax()
        best = all_actions.loc[idx].copy().reset_index(drop=True)
        best["opportunity_score"] = (
            best["expected_profit"]
            * (0.5 + 0.5 * best["p_profit"])
            * (0.5 + 0.5 * best["p_roas_target"])
            / (1.0 + np.maximum(best["expected_regret"], 0.0))
        )
        best = best.sort_values(
            ["level", "opportunity_score"],
            ascending=[True, False],
        ).reset_index(drop=True)

        return all_actions, best
