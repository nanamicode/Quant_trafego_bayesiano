from __future__ import annotations

import math
import time
from typing import Any

import pandas as pd
import streamlit as st

from .observability import RunTelemetry


_STAGE_DEFINITIONS = (
    ("load", "Leitura e normalização"),
    ("scope", "Escopo ativo e qualidade"),
    ("inference", "Inferência hierárquica"),
    ("validation", "Validação temporal"),
    ("allocation", "Otimização de capital"),
    ("plan", "Plano operacional"),
    ("persist", "Auditoria local"),
)

_MATH_BY_STAGE = {
    "load": (
        "Normalização de schema, aliases Meta PT-BR, datas, IDs e métricas."
    ),
    "scope": (
        "Separação entre contexto histórico e universo ativo; auditoria de "
        "qualidade, cobertura, gaps e integridade do funil."
    ),
    "inference": (
        "Partial pooling Bayesiano conta→campanha→conjunto→anúncio; CTR/CVR, "
        "CPM/AOV, derivadas temporais, regime, sazonalidade, elasticidade e "
        "Monte Carlo contrafactual pareado."
    ),
    "validation": (
        "Rolling-origin fora da amostra: derivative vs state-space; Brier, "
        "ECE, cobertura de intervalo, MAE e gates de promoção."
    ),
    "allocation": (
        "Otimização restrita de capital: portfólio correlacionado/CVaR com "
        "fallback MILP, envelope conta→campanhas→conjuntos."
    ),
    "plan": (
        "Tradução do ótimo estatístico para AUMENTAR/REDUZIR/MANTER/DESLIGAR "
        "e candidatos a duplicação."
    ),
    "persist": (
        "Snapshot, manifest, ações, diagnósticos, telemetria e artefatos "
        "auditáveis no DuckDB/workspace local."
    ),
}


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "calculando"
    seconds = max(int(round(seconds)), 0)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}min {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}min"


class RunMonitor:
    """Live Streamlit observability without affecting the statistical model."""

    def __init__(
        self,
        *,
        validate_temporal: bool,
        deep_mode: bool,
        telemetry: RunTelemetry | None = None,
    ):
        self.telemetry = telemetry or RunTelemetry()
        self.deep_mode = bool(deep_mode)
        self.validate_temporal = bool(validate_temporal)

        raw = [
            ("load", 0.04),
            ("scope", 0.06),
            ("inference", 0.46 if deep_mode else 0.28),
            ("validation", 0.18 if deep_mode else 0.38),
            ("allocation", 0.14),
            ("plan", 0.05),
            ("persist", 0.07),
        ]
        if not validate_temporal:
            raw = [
                item
                for item in raw
                if item[0] != "validation"
            ]

        total = sum(weight for _, weight in raw)
        cursor = 0.0
        self.stage_weights: dict[str, tuple[float, float]] = {}
        for key, weight in raw:
            normalized = weight / total
            self.stage_weights[key] = (
                cursor,
                normalized,
            )
            cursor += normalized

        self.stage_state = {
            key: "PENDENTE"
            for key, _ in _STAGE_DEFINITIONS
            if key in self.stage_weights
        }
        self.current_stage: str | None = None
        self.current_stage_started = time.perf_counter()
        self.logs: list[str] = []

        st.markdown("#### Observabilidade da execução")
        self.progress = st.progress(
            0.0,
            text="Preparando análise...",
        )
        m1, m2, m3, m4 = st.columns(4)
        self.overall_metric = m1.empty()
        self.elapsed_metric = m2.empty()
        self.stage_metric = m3.empty()
        self.eta_metric = m4.empty()

        left, right = st.columns([1.25, 1.0])
        with left:
            st.markdown("**Mapa de exploração**")
            self.exploration_box = st.empty()
            st.markdown("**Processos matemáticos**")
            self.pipeline_box = st.empty()
        with right:
            st.markdown("**Log ao vivo**")
            self.log_box = st.empty()
            with st.expander(
                "Telemetria de desenvolvimento",
                expanded=False,
            ):
                self.dev_box = st.empty()

        self.exploration_context: dict[str, Any] = {}
        self._render_pipeline()
        self._render_logs()
        self._render_dev()

    def configure_exploration(
        self,
        *,
        historical_days: int,
        historical_campaigns: int,
        historical_adsets: int,
        historical_ads: int,
        active_campaigns: int,
        active_adsets: int,
        active_ads: int,
        horizon_days: int,
        actions: tuple[float, ...],
        draws: int,
        rolling_origins: int,
        temporal_models: int,
        temporal_model: str,
        weekly_seasonality: bool,
    ):
        entities = (
            1
            + active_campaigns
            + active_adsets
            + active_ads
        )
        action_worlds = (
            entities
            * len(actions)
            * int(draws)
        )
        self.exploration_context = {
            "historical_days": historical_days,
            "historical_campaigns": historical_campaigns,
            "historical_adsets": historical_adsets,
            "historical_ads": historical_ads,
            "active_campaigns": active_campaigns,
            "active_adsets": active_adsets,
            "active_ads": active_ads,
            "horizon_days": horizon_days,
            "actions": actions,
            "draws": draws,
            "action_worlds": action_worlds,
            "rolling_origins": rolling_origins,
            "temporal_models": temporal_models,
            "temporal_model": temporal_model,
            "weekly_seasonality": weekly_seasonality,
        }
        self._render_exploration()

    def _overall_progress(
        self,
        stage: str,
        fraction: float,
    ) -> float:
        start, weight = self.stage_weights[stage]
        return min(
            max(
                start
                + weight
                * min(max(float(fraction), 0.0), 1.0),
                0.0,
            ),
            1.0,
        )

    def update(
        self,
        stage: str,
        fraction: float,
        message: str,
        *,
        event: str = "progress",
        log: bool = False,
        **meta: Any,
    ):
        if stage not in self.stage_weights:
            return

        now = time.perf_counter()
        if self.current_stage != stage:
            if (
                self.current_stage is not None
                and self.stage_state.get(self.current_stage)
                == "RODANDO"
            ):
                self.stage_state[self.current_stage] = "CONCLUÍDO"
            self.current_stage = stage
            self.current_stage_started = now
            self.stage_state[stage] = "RODANDO"
            log = True
            event = (
                "stage_start"
                if event == "progress"
                else event
            )

        fraction = min(
            max(float(fraction), 0.0),
            1.0,
        )
        overall = self._overall_progress(
            stage,
            fraction,
        )
        elapsed = self.telemetry.elapsed_seconds
        stage_elapsed = now - self.current_stage_started

        eta = None
        if overall >= 0.07 and elapsed >= 2.0:
            eta = (
                elapsed
                * (1.0 - overall)
                / max(overall, 1e-6)
            )

        row = self.telemetry.emit(
            stage,
            event,
            message=message,
            progress=overall,
            stage_fraction=fraction,
            eta_seconds=eta,
            **meta,
        )

        self.progress.progress(
            overall,
            text=(
                f"{overall:.0%} · "
                f"{dict(_STAGE_DEFINITIONS)[stage]} · "
                f"{message}"
            ),
        )
        self.overall_metric.metric(
            "Progresso",
            f"{overall:.0%}",
        )
        self.elapsed_metric.metric(
            "Tempo total",
            _fmt_duration(elapsed),
        )
        self.stage_metric.metric(
            "Etapa atual",
            dict(_STAGE_DEFINITIONS)[stage],
        )
        self.eta_metric.metric(
            "ETA estimada",
            (
                _fmt_duration(eta)
                if eta is not None
                else "calculando"
            ),
        )

        if log:
            self.logs.append(
                f"[+{_fmt_duration(row['elapsed_seconds'])}] "
                f"{message}"
            )
        self.logs = self.logs[-24:]
        self._render_pipeline()
        self._render_logs()
        self._render_dev()

    def finish_stage(
        self,
        stage: str,
        message: str,
        **meta: Any,
    ):
        self.update(
            stage,
            1.0,
            message,
            event="stage_done",
            log=True,
            **meta,
        )
        self.stage_state[stage] = "CONCLUÍDO"
        self._render_pipeline()

    def engine_callback(
        self,
        event: dict[str, Any],
    ):
        progress = float(
            event.get("progress", 0.0)
        )
        level = str(
            event.get("level", "")
        )
        entity_id = str(
            event.get("entity_id", "")
        )
        completed = int(
            event.get("completed", 0)
        )
        total = int(
            event.get("total", 0)
        )
        label = {
            "account": "conta",
            "campaign": "campanha",
            "adset": "conjunto",
            "ad": "anúncio",
        }.get(level, level)
        self.update(
            "inference",
            progress,
            (
                f"{label} {completed}/{total}"
                + (
                    f" · {entity_id}"
                    if entity_id and level != "account"
                    else ""
                )
            ),
            event="entity_done",
            log=(
                completed == 1
                or completed == total
                or completed % 10 == 0
            ),
            hierarchy_level=level,
            entity_id=entity_id,
            entities_completed=completed,
            entities_total=total,
        )

    def temporal_callback(
        self,
        event: dict[str, Any],
    ):
        progress = float(
            event.get("progress", 0.0)
        )
        model = str(
            event.get("model", "")
        )
        fold = event.get("fold")
        total = event.get("total")
        origin = event.get("origin_date")

        if fold is not None and total is not None:
            message = (
                f"{model} · janela {fold}/{total}"
                + (
                    f" · origem {pd.Timestamp(origin).date()}"
                    if origin is not None
                    else ""
                )
            )
            log = (
                int(fold) == 1
                or int(fold) == int(total)
            )
        else:
            message = (
                f"{model} · preparando modelo temporal"
                if model
                else "preparando validação temporal"
            )
            log = True

        self.update(
            "validation",
            progress,
            message,
            event=str(
                event.get(
                    "event",
                    "temporal_progress",
                )
            ),
            log=log,
            temporal_model=model,
            fold=fold,
            folds_total=total,
            origin_date=(
                str(origin)
                if origin is not None
                else None
            ),
        )

    def _render_exploration(self):
        if not self.exploration_context:
            self.exploration_box.info(
                "O mapa aparecerá assim que a base for normalizada."
            )
            return

        c = self.exploration_context
        actions = " · ".join(
            f"{float(action):g}x"
            for action in c["actions"]
        )
        seasonality = (
            "ativa"
            if c["weekly_seasonality"]
            else "desativada"
        )
        text = (
            f"**Contexto:** {c['historical_days']} dias · "
            f"{c['historical_campaigns']} campanhas · "
            f"{c['historical_adsets']} conjuntos · "
            f"{c['historical_ads']} anúncios  \n"
            f"**Árvore ativa:** CONTA → "
            f"{c['active_campaigns']} campanhas → "
            f"{c['active_adsets']} conjuntos → "
            f"{c['active_ads']} anúncios  \n"
            f"**Ações por entidade:** {actions}  \n"
            f"**Monte Carlo:** {c['draws']:,} mundos latentes por entidade · "
            f"~{c['action_worlds']:,} avaliações ação×mundo  \n"
            f"**Horizonte decisório:** {c['horizon_days']} dias  \n"
            f"**Rolling-origin:** {c['rolling_origins']} janelas × "
            f"{c['temporal_models']} modelos temporais  \n"
            f"**Temporal:** {c['temporal_model']} · "
            f"sazonalidade semanal {seasonality}  \n"
            "**Variáveis:** CTR · CVR · CPM · AOV · spend · receita · "
            "lucro · ROAS · elasticidade · tendência · regime · sazonalidade · "
            "VaR/CVaR · regret."
        )
        self.exploration_box.markdown(text)

    def _render_pipeline(self):
        rows = []
        for key, label in _STAGE_DEFINITIONS:
            if key not in self.stage_weights:
                continue
            state = self.stage_state[key]
            icon = {
                "PENDENTE": "○",
                "RODANDO": "◉",
                "CONCLUÍDO": "✓",
            }[state]
            rows.append(
                {
                    "": icon,
                    "Etapa": label,
                    "Estado": state,
                    "Matemática": _MATH_BY_STAGE[key],
                }
            )
        self.pipeline_box.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    def _render_logs(self):
        if not self.logs:
            self.log_box.code(
                "Aguardando início...",
                language=None,
            )
            return
        self.log_box.code(
            "\n".join(self.logs),
            language=None,
        )

    def _render_dev(self):
        snapshot = self.telemetry.developer_snapshot()
        recent = self.telemetry.events[-8:]
        self.dev_box.json(
            {
                "runtime": snapshot,
                "recent_events": recent,
                "purpose": (
                    "Diagnóstico de gargalos, regressões, calibração e "
                    "planejamento de otimizações futuras."
                ),
            },
            expanded=False,
        )

    def done(self):
        if self.current_stage is not None:
            self.stage_state[self.current_stage] = "CONCLUÍDO"
        elapsed = self.telemetry.elapsed_seconds
        self.progress.progress(
            1.0,
            text="100% · análise concluída",
        )
        self.overall_metric.metric(
            "Progresso",
            "100%",
        )
        self.elapsed_metric.metric(
            "Tempo total",
            _fmt_duration(elapsed),
        )
        self.stage_metric.metric(
            "Etapa atual",
            "Concluída",
        )
        self.eta_metric.metric(
            "ETA estimada",
            "0s",
        )
        self.logs.append(
            f"[+{_fmt_duration(elapsed)}] Análise concluída."
        )
        self.telemetry.emit(
            "persist",
            "complete",
            message="Análise concluída",
            progress=1.0,
        )
        self._render_pipeline()
        self._render_logs()
        self._render_dev()
