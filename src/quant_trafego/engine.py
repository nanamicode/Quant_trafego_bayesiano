from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd

from .dynamic import analyze_state_space_temporal
from .model import (
    BetaPosterior,
    aggregate,
    beta_from_mean,
    sample_simulation_context,
    shrink_to,
    simulate_action,
    update_beta,
)
from .quality import assess_data_quality
from .response import ResponseEstimate, estimate_response
from .seasonality import analyze_weekly_seasonality
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
    action_baseline_recent_days: int = 7
    temporal_projection_days: float = 2.0
    use_temporal: bool = True
    temporal_model: str = "derivative"
    use_empirical_response: bool = True
    use_weekly_seasonality: bool = True
    attribution_safe_cvr_temporal: bool = True
    seasonality_half_life_days: float = 56.0
    seasonality_min_days: int = 21
    predictive_max_multiplier: float = 1.20
    observational_max_multiplier: float = 1.50
    experiment_max_multiplier: float = 2.00
    min_p_profit_for_scale: float = 0.60
    min_p_incremental_for_scale: float = 0.55
    min_quality_for_scale: float = 0.55
    require_recent_contribution_profit_for_scale: bool = True
    protect_recent_profitable_from_hard_pause: bool = True
    min_p_beats_hold_for_profitable_pause: float = 0.80
    min_p_action_optimal_for_profitable_pause: float = 0.60
    max_instability_for_profitable_pause: float = 0.60
    cautious_quality_threshold: float = 0.75
    cautious_quality_max_multiplier: float = 1.20
    quality_confidence_weight: float = 0.35
    revenue_tiebreak: bool = True
    revenue_tiebreak_tolerance: float = 0.02

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
        if self.config.temporal_model not in {"derivative", "state_space"}:
            raise ValueError("temporal_model deve ser derivative ou state_space.")
        actions = np.asarray(self.config.actions, dtype=float)
        if np.any(actions < 0):
            raise ValueError("Multiplicadores de ação não podem ser negativos.")
        if len(np.unique(np.round(actions, 12))) != len(actions):
            raise ValueError("A grade de ações contém multiplicadores duplicados.")
        if not np.any(np.isclose(actions, 1.0)):
            raise ValueError("A grade de ações deve conter 1.0x como baseline hold.")

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
            raw = out[col]
            parsed = pd.to_numeric(raw, errors="coerce")
            invalid = (
                parsed.isna()
                & raw.notna()
                & raw.astype(str).str.strip().ne("")
            )
            if invalid.any():
                raise ValueError(
                    f"A coluna {col} contém valores não numéricos inválidos."
                )
            out[col] = parsed.fillna(0.0)
            if (out[col] < 0).any():
                raise ValueError(f"A coluna {col} contém valores negativos.")

        if (out["clicks"] > out["impressions"]).any():
            raise ValueError("Existem linhas onde cliques > impressões.")
        # Meta Ads attribution is not a literal same-day funnel:
        # purchases can be credited today for earlier clicks or view-through
        # exposure. Daily conversions may therefore exceed same-day clicks.
        # The fast click-to-conversion posterior only requires coherence over
        # the full history of each ad.
        aggregated = (
            out.groupby("ad_id", as_index=False)
            .agg(
                clicks=("clicks", "sum"),
                conversions=("conversions", "sum"),
            )
        )
        if (aggregated["conversions"] > aggregated["clicks"]).any():
            raise ValueError(
                "Há anúncios em que conversões agregadas no período excedem "
                "cliques agregados. Essa base exige um modelo de atribuição "
                "por exposição antes de estimar CVR por clique."
            )

        return out

    @staticmethod
    def _entity_context(level: str, entity_id: str, df: pd.DataFrame) -> dict:
        def first_value(column: str):
            if column not in df.columns:
                return None
            values = df[column].dropna()
            if values.empty:
                return None
            return str(values.iloc[-1])

        context = {
            "campaign_id": first_value("campaign_id"),
            "campaign_name": first_value("campaign_name"),
            "adset_id": first_value("adset_id"),
            "adset_name": first_value("adset_name"),
            "ad_id": first_value("ad_id"),
            "ad_name": first_value("ad_name"),
        }
        if level == "account":
            context.update(
                campaign_id=None,
                campaign_name=None,
                adset_id=None,
                adset_name=None,
                ad_id=None,
                ad_name=None,
            )
        elif level == "campaign":
            context["campaign_id"] = str(entity_id)
            context["adset_id"] = None
            context["adset_name"] = None
            context["ad_id"] = None
            context["ad_name"] = None
        elif level == "adset":
            context["adset_id"] = str(entity_id)
            context["ad_id"] = None
            context["ad_name"] = None
        elif level == "ad":
            context["ad_id"] = str(entity_id)
        return context

    @staticmethod
    def _stable_offset(level: str, entity_id: str) -> int:
        digest = hashlib.blake2b(
            f"{level}|{entity_id}".encode("utf-8"),
            digest_size=4,
        ).digest()
        return int.from_bytes(digest, "little") % 100_000

    def _select_profit_first_growth_indices(
        self,
        actions: pd.DataFrame,
    ) -> list[int]:
        selected: list[int] = []
        for _, group in actions.groupby(
            ["level", "entity_id"],
            sort=False,
        ):
            max_utility = float(
                group["risk_adjusted_utility"].max()
            )
            if not self.config.revenue_tiebreak:
                selected.append(
                    int(
                        group["risk_adjusted_utility"].idxmax()
                    )
                )
                continue

            tolerance = (
                float(
                    self.config.revenue_tiebreak_tolerance
                )
                * max(
                    abs(max_utility),
                    1.0,
                )
            )
            near = group[
                group["risk_adjusted_utility"]
                >= max_utility - tolerance
            ]
            chosen = near.sort_values(
                [
                    "expected_revenue",
                    "risk_adjusted_utility",
                ],
                ascending=[
                    False,
                    False,
                ],
            ).index[0]
            selected.append(int(chosen))
        return selected

    @staticmethod
    def _action_offset(multiplier: float) -> int:
        digest = hashlib.blake2b(
            f"{float(multiplier):.8f}".encode("utf-8"),
            digest_size=4,
        ).digest()
        return int.from_bytes(digest, "little") % 1_000_000

    @staticmethod
    def _get_override(overrides, level, entity_id, fallback):
        if not overrides:
            return fallback, "empirical_bayes"
        key = (level, str(entity_id))
        if key in overrides:
            return overrides[key], "mcmc"
        return fallback, "empirical_bayes"

    def _global_posteriors(self, df: pd.DataFrame):
        """
        Global empirical posterior with a weak Jeffreys prior.

        The previous implementation centered a strong prior on the same data
        and then updated with those observations again. For hierarchy means we
        need the data to enter the likelihood once.
        """
        s = aggregate(df)
        ctr_prior = BetaPosterior(0.5, 0.5)
        cvr_prior = BetaPosterior(0.5, 0.5)
        return (
            update_beta(
                ctr_prior,
                s["clicks"],
                s["impressions"],
            ),
            update_beta(
                cvr_prior,
                s["conversions"],
                s["clicks"],
            ),
        )

    def _global_hyperprior(
        self,
        df: pd.DataFrame,
    ) -> tuple[BetaPosterior, BetaPosterior]:
        s = aggregate(df)
        return (
            beta_from_mean(
                s["ctr"],
                self.config.global_ctr_strength,
            ),
            beta_from_mean(
                s["cvr"],
                self.config.global_cvr_strength,
            ),
        )

    def _external_parent(
        self,
        reference_df: pd.DataFrame,
        fallback_df: pd.DataFrame,
    ) -> tuple[BetaPosterior, BetaPosterior]:
        if (
            not reference_df.empty
            and float(reference_df["impressions"].sum()) > 0
            and float(reference_df["clicks"].sum()) > 0
        ):
            return self._global_posteriors(reference_df)
        return self._global_hyperprior(fallback_df)

    def _recent_operating_stats(
        self,
        df: pd.DataFrame,
    ) -> dict:
        latest = pd.to_datetime(df["date"]).max()
        cutoff = latest - pd.Timedelta(
            days=max(
                int(
                    self.config.action_baseline_recent_days
                ),
                1,
            )
            - 1
        )
        recent = df[
            pd.to_datetime(df["date"]) >= cutoff
        ].copy()
        calendar_days = max(
            int(
                self.config.action_baseline_recent_days
            ),
            1,
        )
        spend = float(recent["spend"].sum())
        revenue = float(recent["revenue"].sum())
        conversions = float(
            recent["conversions"].sum()
        )
        impressions = float(
            recent["impressions"].sum()
        )
        clicks = float(recent["clicks"].sum())
        daily_spend = spend / calendar_days
        contribution_profit = (
            revenue
            * self.config.contribution_margin
            - spend
        )
        return {
            "recent_days": calendar_days,
            "recent_spend": spend,
            "recent_revenue": revenue,
            "recent_conversions": conversions,
            "recent_impressions": impressions,
            "recent_clicks": clicks,
            "recent_daily_spend": daily_spend,
            "recent_roas": (
                revenue / spend
                if spend > 0
                else 0.0
            ),
            "recent_cpa": (
                spend / conversions
                if conversions > 0
                else np.nan
            ),
            "recent_contribution_profit": contribution_profit,
            "recent_contribution_roas": (
                revenue
                * self.config.contribution_margin
                / spend
                if spend > 0
                else 0.0
            ),
        }

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

    def _temporal_signal(self, df, level, entity_id):
        seed = self.config.seed + self._stable_offset(level, str(entity_id))
        if self.config.temporal_model == "state_space":
            return analyze_state_space_temporal(
                df,
                recent_days=self.config.temporal_recent_days,
                seed=seed,
            )
        return analyze_temporal(
            df,
            half_life_days=self.config.temporal_half_life_days,
            recent_days=self.config.temporal_recent_days,
            seed=seed,
        )

    @staticmethod
    def _evidence_tier(response_estimate: ResponseEstimate) -> str:
        if response_estimate.confidence >= 0.15:
            return "observational_intervention"
        return "predictive"

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
        recent = self._recent_operating_stats(df)
        entity_context = self._entity_context(level, entity_id, df)
        temporal = self._temporal_signal(df, level, entity_id)

        if self.config.use_weekly_seasonality:
            weekly = analyze_weekly_seasonality(
                df,
                half_life_days=self.config.seasonality_half_life_days,
                min_days=self.config.seasonality_min_days,
            )
            last_date = stats["daily"]["date"].max()
            from_last_day = posterior_source == "mcmc"
            ctr_weekly_mean, ctr_weekly_sd = weekly.ctr.future_shift(
                last_date,
                self.config.horizon_days,
                from_last_day=from_last_day,
            )
            cvr_weekly_mean, cvr_weekly_sd = weekly.cvr.future_shift(
                last_date,
                self.config.horizon_days,
                from_last_day=from_last_day,
            )
            weekly_confidence = weekly.confidence
        else:
            ctr_weekly_mean = 0.0
            ctr_weekly_sd = 0.0
            cvr_weekly_mean = 0.0
            cvr_weekly_sd = 0.0
            weekly_confidence = 0.0

        ctr_mean = temporal.ctr.effective_mean if self.config.use_temporal else 0.0
        ctr_sd = temporal.ctr.effective_sd if self.config.use_temporal else 0.0

        if (
            self.config.use_temporal
            and not self.config.attribution_safe_cvr_temporal
        ):
            cvr_mean = temporal.cvr.effective_mean
            cvr_sd = temporal.cvr.effective_sd
        else:
            # Meta's attributed purchases do not belong to a literal same-day
            # click funnel. Recent aggregate level is useful, but extrapolating
            # a daily CVR derivative is not sufficiently identified.
            cvr_mean = 0.0
            cvr_sd = 0.0
            if self.config.attribution_safe_cvr_temporal:
                cvr_weekly_mean = 0.0
                cvr_weekly_sd = 0.0

        if (
            self.config.use_temporal
            and posterior_source != "mcmc"
        ):
            current_ctr_shift = (
                temporal.ctr_current_logit_shift
                * temporal.ctr_current_shift_confidence
            )
        else:
            # Deep CTR posterior already carries the current time state.
            current_ctr_shift = 0.0

        if (
            self.config.use_temporal
            and self.config.attribution_safe_cvr_temporal
        ):
            # Deep CVR likelihood is intentionally aggregated by ad because
            # Meta attribution is asynchronous, so it still needs a recent
            # aggregate level anchor just like the fast posterior.
            current_cvr_shift = (
                temporal.cvr_current_logit_shift
                * temporal.cvr_current_shift_confidence
            )
        elif (
            self.config.use_temporal
            and posterior_source != "mcmc"
        ):
            current_cvr_shift = (
                temporal.cvr_current_logit_shift
                * temporal.cvr_current_shift_confidence
            )
        else:
            current_cvr_shift = 0.0

        response_confidence = (
            response_estimate.confidence if self.config.use_empirical_response else 0.0
        )

        entity_seed = (
            self.config.seed
            + self._stable_offset(
                level,
                str(entity_id),
            )
        )
        context = sample_simulation_context(
            stats=stats,
            ctr_post=ctr_post,
            cvr_post=cvr_post,
            draws=self.config.draws,
            horizon_days=self.config.horizon_days,
            rng=np.random.default_rng(
                entity_seed + 1_000_003
            ),
            temporal_ctr_slope_mean=ctr_mean,
            temporal_ctr_slope_sd=ctr_sd,
            temporal_cvr_slope_mean=cvr_mean,
            temporal_cvr_slope_sd=cvr_sd,
            response_elasticity_mean=response_estimate.elasticity_mean,
            response_elasticity_sd=response_estimate.elasticity_sd,
            current_ctr_logit_shift=current_ctr_shift,
            current_cvr_logit_shift=current_cvr_shift,
            temporal_projection_days=self.config.temporal_projection_days,
            seasonal_ctr_shift_mean=ctr_weekly_mean,
            seasonal_ctr_shift_sd=ctr_weekly_sd,
            seasonal_cvr_shift_mean=cvr_weekly_mean,
            seasonal_cvr_shift_sd=cvr_weekly_sd,
        )

        sims = []
        for action in self.config.actions:
            action_rng = np.random.default_rng(
                entity_seed
                + 2_000_003
                + self._action_offset(action)
            )
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
                    rng=action_rng,
                    saturation_half=self.config.saturation_half,
                    saturation_slope=self.config.saturation_slope,
                    temporal_ctr_slope_mean=ctr_mean,
                    temporal_ctr_slope_sd=ctr_sd,
                    temporal_cvr_slope_mean=cvr_mean,
                    temporal_cvr_slope_sd=cvr_sd,
                    response_elasticity_mean=response_estimate.elasticity_mean,
                    response_elasticity_sd=response_estimate.elasticity_sd,
                    response_confidence=response_confidence,
                    base_daily_spend=recent["recent_daily_spend"],
                    current_ctr_logit_shift=current_ctr_shift,
                    current_cvr_logit_shift=current_cvr_shift,
                    temporal_projection_days=self.config.temporal_projection_days,
                    seasonal_ctr_shift_mean=ctr_weekly_mean,
                    seasonal_ctr_shift_sd=ctr_weekly_sd,
                    seasonal_cvr_shift_mean=cvr_weekly_mean,
                    seasonal_cvr_shift_sd=cvr_weekly_sd,
                    context=context,
                )
            )

        hold = next(x for x in sims if x["multiplier"] == 1.0)
        hold_draws = hold["_decision_profit_draws"]
        hold_expected_revenue = float(hold["expected_revenue"])
        hold_expected_profit = float(hold["expected_profit"])
        profit_matrix = np.vstack(
            [x["_decision_profit_draws"] for x in sims]
        )
        best_idx = np.argmax(profit_matrix, axis=0)
        best_draws = np.max(profit_matrix, axis=0)

        rows = []
        for i, sim in enumerate(sims):
            profit_draws = sim["_profit_draws"]
            decision_draws = sim["_decision_profit_draws"]
            regret = float(
                np.mean(
                    best_draws
                    - decision_draws
                )
            )
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

            incremental = (
                decision_draws
                - hold_draws
            )
            rows.append({
                "level": level,
                "entity_id": str(entity_id),
                **entity_context,
                "posterior_source": posterior_source,
                "temporal_model": self.config.temporal_model,
                "evidence_tier": self._evidence_tier(response_estimate),
                "historical_days": stats["days"],
                "historical_spend": stats["spend"],
                "historical_revenue": stats["revenue"],
                "historical_roas": stats["roas"],
                "historical_ctr": stats["ctr"],
                "historical_cvr": stats["cvr"],
                "action_baseline_daily_spend": recent["recent_daily_spend"],
                "action_baseline_recent_days": recent["recent_days"],
                "recent_spend": recent["recent_spend"],
                "recent_revenue": recent["recent_revenue"],
                "recent_conversions": recent["recent_conversions"],
                "recent_roas": recent["recent_roas"],
                "recent_cpa": recent["recent_cpa"],
                "recent_contribution_profit": recent["recent_contribution_profit"],
                "recent_contribution_roas": recent["recent_contribution_roas"],
                "posterior_ctr_mean": ctr_post.mean,
                "posterior_cvr_mean": cvr_post.mean,
                "posterior_ctr_strength": ctr_post.strength,
                "posterior_cvr_strength": cvr_post.strength,
                "ctr_logit_derivative_per_day": temporal.ctr.mean,
                "ctr_trend_confidence": temporal.ctr.confidence,
                "cvr_logit_derivative_per_day": temporal.cvr.mean,
                "cvr_trend_confidence": temporal.cvr.confidence,
                "ctr_current_logit_shift": current_ctr_shift,
                "cvr_current_logit_shift": current_cvr_shift,
                "temporal_projection_days": float(
                    self.config.temporal_projection_days
                ),
                "cvr_temporal_mode": (
                    "recent_level_only"
                    if self.config.attribution_safe_cvr_temporal
                    else "daily_derivative"
                ),
                "p_recent_ctr_better": temporal.p_recent_ctr_better,
                "p_recent_cvr_better": temporal.p_recent_cvr_better,
                "regime_change_score": temporal.regime_change_score,
                "instability_score": temporal.instability_score,
                "response_elasticity": response_estimate.elasticity_mean,
                "response_elasticity_sd": response_estimate.elasticity_sd,
                "response_confidence": response_estimate.confidence,
                "response_independent_spend_sd": response_estimate.independent_spend_sd,
                "response_effective_days": response_estimate.effective_days,
                "response_controls": response_estimate.controls,
                "weekly_seasonality_confidence": weekly_confidence,
                "future_ctr_weekly_logit_shift": ctr_weekly_mean,
                "future_ctr_weekly_shift_sd": ctr_weekly_sd,
                "future_cvr_weekly_logit_shift": cvr_weekly_mean,
                "future_cvr_weekly_shift_sd": cvr_weekly_sd,
                "p_diminishing_returns_proxy": (
                    response_estimate.diminishing_returns_probability_proxy
                ),
                "contribution_margin": self.config.contribution_margin,
                "action_multiplier": sim["multiplier"],
                "expected_spend": sim["expected_spend"],
                "expected_revenue": sim["expected_revenue"],
                "expected_profit": sim["expected_profit"],
                "expected_roas": sim["expected_roas"],
                "profit_p05": sim["profit_p05"],
                "profit_p50": sim["profit_p50"],
                "profit_p95": sim["profit_p95"],
                "revenue_p05": sim["revenue_p05"],
                "revenue_p50": sim["revenue_p50"],
                "revenue_p95": sim["revenue_p95"],
                "roas_p05": sim["roas_p05"],
                "roas_p50": sim["roas_p50"],
                "roas_p95": sim["roas_p95"],
                "p_profit": sim["p_profit"],
                "p_ruin": 1.0 - sim["p_profit"],
                "p_roas_target": sim["p_roas_target"],
                "p_beats_hold": float(
                    np.mean(
                        decision_draws
                        > hold_draws
                    )
                ),
                "p_action_optimal": float(np.mean(best_idx == i)),
                "expected_hold_profit": hold_expected_profit,
                "expected_hold_revenue": hold_expected_revenue,
                "expected_incremental_profit_vs_hold": float(np.mean(incremental)),
                "expected_incremental_revenue_vs_hold": float(
                    sim["expected_revenue"] - hold_expected_revenue
                ),
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
        decision_entities: dict[str, set[str] | frozenset[str]] | None = None,
        evaluation_levels: set[str] | tuple[str, ...] | None = None,
        progress_callback=None,
    ):
        df = self.validate(df)
        requested_levels = (
            {"account", "campaign", "adset", "ad"}
            if evaluation_levels is None
            else {str(level) for level in evaluation_levels}
        )
        invalid_levels = requested_levels - {
            "account",
            "campaign",
            "adset",
            "ad",
        }
        if invalid_levels:
            raise ValueError(
                "Níveis de avaliação inválidos: "
                + ", ".join(sorted(invalid_levels))
            )
        # Parent levels are required internally to build hierarchical priors.
        if "ad" in requested_levels:
            requested_levels.update(
                {"account", "campaign", "adset"}
            )
        elif "adset" in requested_levels:
            requested_levels.update(
                {"account", "campaign"}
            )
        elif "campaign" in requested_levels:
            requested_levels.add("account")

        active_campaigns = (
            {str(x) for x in decision_entities.get("campaign", set())}
            if decision_entities
            else None
        )
        active_adsets = (
            {str(x) for x in decision_entities.get("adset", set())}
            if decision_entities
            else None
        )
        active_ads = (
            {str(x) for x in decision_entities.get("ad", set())}
            if decision_entities
            else None
        )

        progress_total = 0
        if "account" in requested_levels:
            progress_total += 1
        if "campaign" in requested_levels:
            progress_total += (
                len(active_campaigns)
                if active_campaigns is not None
                else int(df["campaign_id"].nunique())
            )
        if "adset" in requested_levels:
            progress_total += (
                len(active_adsets)
                if active_adsets is not None
                else int(df["adset_id"].nunique())
            )
        if "ad" in requested_levels:
            progress_total += (
                len(active_ads)
                if active_ads is not None
                else int(df["ad_id"].nunique())
            )
        progress_done = 0

        def emit_progress(level: str, entity_id: str):
            nonlocal progress_done
            progress_done += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "engine",
                        "level": level,
                        "entity_id": str(entity_id),
                        "completed": progress_done,
                        "total": max(progress_total, 1),
                        "progress": min(
                            progress_done / max(progress_total, 1),
                            1.0,
                        ),
                    }
                )

        if decision_entities:
            if active_ads:
                account_df = df[
                    df["ad_id"].astype(str).isin(active_ads)
                ].copy()
            elif active_adsets:
                account_df = df[
                    df["adset_id"].astype(str).isin(active_adsets)
                ].copy()
            elif active_campaigns:
                account_df = df[
                    df["campaign_id"].astype(str).isin(active_campaigns)
                ].copy()
            else:
                account_df = df.iloc[0:0].copy()

            if account_df.empty:
                raise ValueError(
                    "Nenhuma entidade ativa foi identificada para gerar decisões."
                )
        else:
            account_df = df

        quality_report = assess_data_quality(account_df)
        quality_factor = float(np.clip(quality_report.score / 100.0, 0.0, 1.0))
        rows = []

        # Full history, including currently inactive entities, remains the
        # statistical context used for hierarchical shrinkage.
        context_global_ctr, context_global_cvr = self._global_posteriors(df)
        context_response = estimate_response(df)

        if decision_entities:
            account_ctr, account_cvr = self._global_posteriors(account_df)
            account_source = "empirical_bayes_active_portfolio"
            account_response = estimate_response(
                account_df,
                parent=context_response,
            )
        else:
            (account_ctr, account_cvr), account_source = self._get_override(
                posterior_overrides,
                "account",
                "ALL",
                (context_global_ctr, context_global_cvr),
            )
            account_response = context_response

        if "account" in requested_levels:
            rows.extend(
                self._evaluate_entity(
                    "account",
                    "ALL",
                    account_df,
                    account_ctr,
                    account_cvr,
                    account_response,
                    account_source,
                )
            )
            emit_progress("account", "ALL")

        campaign_posts = {}
        campaign_response = {}
        campaign_source = {}
        campaign_external_parent = {}
        campaign_external_response = {}

        for campaign_id, cdf in df.groupby("campaign_id", sort=False):
            if (
                active_campaigns is not None
                and str(campaign_id) not in active_campaigns
            ):
                continue

            other_campaigns = df[
                df["campaign_id"].astype(str)
                != str(campaign_id)
            ].copy()
            external_parent = self._external_parent(
                other_campaigns,
                df,
            )
            external_response = (
                estimate_response(other_campaigns)
                if not other_campaigns.empty
                else context_response
            )

            fallback = self._posterior(
                cdf,
                *external_parent,
                self.config.campaign_ctr_strength,
                self.config.campaign_cvr_strength,
            )
            posts, source = self._get_override(
                posterior_overrides,
                "campaign",
                campaign_id,
                fallback,
            )
            response = estimate_response(
                cdf,
                parent=external_response,
            )
            campaign_posts[campaign_id] = posts
            campaign_response[campaign_id] = response
            campaign_source[campaign_id] = source
            campaign_external_parent[campaign_id] = external_parent
            campaign_external_response[campaign_id] = external_response

            if "campaign" in requested_levels:
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
                emit_progress("campaign", campaign_id)

        adset_posts = {}
        adset_response = {}
        adset_external_parent = {}
        adset_external_response = {}
        adset_groups = (
            df.groupby(
                ["campaign_id", "adset_id"],
                sort=False,
            )
            if (
                "adset" in requested_levels
                or "ad" in requested_levels
            )
            else []
        )
        for (campaign_id, adset_id), sdf in adset_groups:
            if (
                active_adsets is not None
                and str(adset_id) not in active_adsets
            ):
                continue
            if campaign_id not in campaign_posts:
                continue

            campaign_df = df[
                df["campaign_id"].astype(str)
                == str(campaign_id)
            ]
            sibling_adsets = campaign_df[
                campaign_df["adset_id"].astype(str)
                != str(adset_id)
            ].copy()

            if not sibling_adsets.empty:
                parent_for_adset = self._posterior(
                    sibling_adsets,
                    *campaign_external_parent[campaign_id],
                    self.config.campaign_ctr_strength,
                    self.config.campaign_cvr_strength,
                )
                parent_response = estimate_response(
                    sibling_adsets,
                    parent=campaign_external_response[campaign_id],
                )
            else:
                parent_for_adset = campaign_external_parent[
                    campaign_id
                ]
                parent_response = campaign_external_response[
                    campaign_id
                ]

            fallback = self._posterior(
                sdf,
                *parent_for_adset,
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
                parent=parent_response,
            )
            adset_posts[(campaign_id, adset_id)] = posts
            adset_response[(campaign_id, adset_id)] = response
            adset_external_parent[(campaign_id, adset_id)] = (
                parent_for_adset
            )
            adset_external_response[(campaign_id, adset_id)] = (
                parent_response
            )
            if "adset" in requested_levels:
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
                emit_progress("adset", adset_id)

        ad_groups = (
            df.groupby(
                ["campaign_id", "adset_id", "ad_id"],
                sort=False,
            )
            if "ad" in requested_levels
            else []
        )
        for (campaign_id, adset_id, ad_id), adf in ad_groups:
            if (
                active_ads is not None
                and str(ad_id) not in active_ads
            ):
                continue
            if (campaign_id, adset_id) not in adset_posts:
                continue

            adset_df = df[
                (
                    df["campaign_id"].astype(str)
                    == str(campaign_id)
                )
                & (
                    df["adset_id"].astype(str)
                    == str(adset_id)
                )
            ]
            sibling_ads = adset_df[
                adset_df["ad_id"].astype(str)
                != str(ad_id)
            ].copy()

            if not sibling_ads.empty:
                parent_for_ad = self._posterior(
                    sibling_ads,
                    *adset_external_parent[
                        (campaign_id, adset_id)
                    ],
                    self.config.adset_ctr_strength,
                    self.config.adset_cvr_strength,
                )
                parent_response = estimate_response(
                    sibling_ads,
                    parent=adset_external_response[
                        (campaign_id, adset_id)
                    ],
                )
            else:
                parent_for_ad = adset_external_parent[
                    (campaign_id, adset_id)
                ]
                parent_response = adset_external_response[
                    (campaign_id, adset_id)
                ]

            fallback = self._posterior(
                adf,
                *parent_for_ad,
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
                parent=parent_response,
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
            emit_progress("ad", ad_id)

        all_actions = pd.DataFrame(rows)
        all_actions["data_quality_score"] = quality_report.score
        all_actions["data_quality_factor"] = quality_factor

        tier_caps = {
            "predictive": self.config.predictive_max_multiplier,
            "observational_intervention": self.config.observational_max_multiplier,
            "experiment_calibrated": self.config.experiment_max_multiplier,
        }
        caps = all_actions["evidence_tier"].map(tier_caps).fillna(
            self.config.predictive_max_multiplier
        ).to_numpy(dtype=float)

        if quality_factor < self.config.cautious_quality_threshold:
            caps = np.minimum(
                caps,
                self.config.cautious_quality_max_multiplier,
            )

        scale = all_actions["action_multiplier"] > 1.0
        recent_profit_ok = (
            all_actions["recent_contribution_profit"]
            > 0.0
        )
        if not self.config.require_recent_contribution_profit_for_scale:
            recent_profit_ok = pd.Series(
                True,
                index=all_actions.index,
            )

        all_actions["recent_scale_sanity_ok"] = recent_profit_ok

        hard_pause = np.isclose(
            all_actions["action_multiplier"].to_numpy(dtype=float),
            0.0,
        )
        recent_profitable = (
            all_actions["recent_contribution_profit"].to_numpy(dtype=float)
            > 0.0
        )
        pause_evidence_ok = (
            (
                all_actions["p_beats_hold"].to_numpy(dtype=float)
                >= self.config.min_p_beats_hold_for_profitable_pause
            )
            & (
                all_actions["p_action_optimal"].to_numpy(dtype=float)
                >= self.config.min_p_action_optimal_for_profitable_pause
            )
            & (
                all_actions["instability_score"].to_numpy(dtype=float)
                <= self.config.max_instability_for_profitable_pause
            )
        )
        if self.config.protect_recent_profitable_from_hard_pause:
            profitable_pause_ok = (
                (~hard_pause)
                | (~recent_profitable)
                | pause_evidence_ok
            )
        else:
            profitable_pause_ok = np.ones(
                len(all_actions),
                dtype=bool,
            )
        all_actions["hard_pause_guardrail_ok"] = profitable_pause_ok
        all_actions["hard_pause_guardrail_triggered"] = (
            hard_pause
            & recent_profitable
            & (~pause_evidence_ok)
        )

        all_actions["policy_max_multiplier"] = caps
        all_actions["policy_eligible"] = (
            all_actions["action_multiplier"].to_numpy(dtype=float)
            <= caps + 1e-12
        ) & (
            (~scale)
            | (
                (quality_factor >= self.config.min_quality_for_scale)
                & recent_profit_ok.to_numpy(dtype=bool)
                & (
                    all_actions["p_profit"]
                    >= self.config.min_p_profit_for_scale
                )
                & (
                    all_actions["p_incremental_profit_positive"]
                    >= self.config.min_p_incremental_for_scale
                )
            )
        ) & profitable_pause_ok

        all_actions["selection_objective"] = (
            "risk_adjusted_profit_first_revenue_secondary"
            if self.config.revenue_tiebreak
            else "risk_adjusted_profit"
        )
        all_actions["revenue_tiebreak_tolerance"] = float(
            self.config.revenue_tiebreak_tolerance
        )

        raw_idx = self._select_profit_first_growth_indices(
            all_actions
        )
        raw_best = all_actions.loc[
            raw_idx,
            [
                "level",
                "entity_id",
                "action_multiplier",
                "risk_adjusted_utility",
                "expected_revenue",
            ],
        ].rename(
            columns={
                "action_multiplier": "unconstrained_best_multiplier",
                "risk_adjusted_utility": "unconstrained_best_utility",
                "expected_revenue": "unconstrained_best_revenue",
            }
        )

        eligible = all_actions[
            all_actions["policy_eligible"]
        ].copy()
        idx = self._select_profit_first_growth_indices(
            eligible
        )
        best = all_actions.loc[idx].copy().reset_index(drop=True)
        best = best.merge(
            raw_best,
            on=["level", "entity_id"],
            how="left",
        )
        best["policy_constrained"] = (
            best["action_multiplier"]
            != best["unconstrained_best_multiplier"]
        )
        best["policy_utility_gap"] = (
            best["unconstrained_best_utility"]
            - best["risk_adjusted_utility"]
        )

        confidence = (
            0.40 * best["p_profit"]
            + 0.25 * best["p_action_optimal"]
            + 0.20 * best["p_roas_target"]
            + 0.15 * (1.0 - best["instability_score"])
        )
        quality_multiplier = (
            1.0
            - self.config.quality_confidence_weight
            * (1.0 - quality_factor)
        )
        best["decision_score_raw"] = np.clip(
            confidence,
            0.0,
            1.0,
        )
        best["decision_score"] = np.clip(
            confidence * quality_multiplier,
            0.0,
            1.0,
        )
        best["decision_score_kind"] = (
            "heuristic_composite_not_calibrated_probability"
        )

        # Backward-compatible aliases. These must not be interpreted as
        # calibrated probabilities.
        best["decision_confidence_raw"] = best["decision_score_raw"]
        best["decision_confidence"] = best["decision_score"]
        best["data_quality_score"] = quality_report.score
        best["opportunity_score"] = (
            best["risk_adjusted_utility"]
            * (0.25 + 0.75 * best["decision_score"])
            / (1.0 + np.maximum(best["expected_regret"], 0.0))
        )
        best = best.sort_values(
            ["level", "opportunity_score"],
            ascending=[True, False],
        ).reset_index(drop=True)

        return all_actions, best
