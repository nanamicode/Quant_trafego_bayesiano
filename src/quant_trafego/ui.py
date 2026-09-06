from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile

import streamlit as st

from quant_trafego.action_plan import build_operational_action_plan, derive_account_budget_target
from quant_trafego.development_diagnostics import build_development_diagnostics
from quant_trafego.engine import BayesTrafficEngine, EngineConfig
from quant_trafego.funnel import detect_funnel_schema, hierarchical_funnel_diagnostics
from quant_trafego.hardware import detect_hardware
from quant_trafego.io import filter_decision_rows, infer_decision_universe, load_ads_file
from quant_trafego.model_selection import compare_temporal_models
from quant_trafego.observability import RunTelemetry
from quant_trafego.optimization import optimize_adset_allocation, optimize_campaign_allocation
from quant_trafego.portfolio import optimize_campaign_portfolio
from quant_trafego.quality import assess_data_quality
from quant_trafego.reproducibility import build_run_manifest
from quant_trafego.storage import LocalWarehouse
from quant_trafego.ui_observability import RunMonitor


DEPTHS = {
    "Automática": None,
    "Rápida": 5_000,
    "Completa": 30_000,
    "Profunda": 100_000,
}


def _fmt_money(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def main():
    st.set_page_config(page_title="Quant Tráfego Bayesiano", layout="wide")
    st.title("Quant Tráfego Bayesiano")
    st.caption(
        "Motor quantitativo local. A planilha e os cálculos permanecem no computador; "
        "a interface usa apenas localhost."
    )

    hw = detect_hardware()
    ram = f"{hw.ram_gb:.1f} GB" if hw.ram_gb is not None else "não detectada"
    st.info(
        f"Hardware: {hw.cpu_threads} threads | RAM {ram} | perfil {hw.label} | "
        f"Monte Carlo automático {hw.recommended_draws:,}."
    )

    uploaded = st.file_uploader(
        "Planilha completa de tráfego pago",
        type=["csv", "xlsx", "xlsm", "xltx"],
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        inference_mode = st.selectbox(
            "Motor de inferência",
            ["Hierárquico rápido", "MCMC hierárquico profundo"],
            index=0,
        )
        temporal_label = st.selectbox(
            "Modelo temporal",
            ["Derivada local (padrão)", "State-space (candidato)"],
            index=0,
        )
    with c2:
        depth = st.selectbox("Monte Carlo", list(DEPTHS), index=0)
    with c3:
        horizon_days = st.number_input("Horizonte futuro (dias)", 1, 90, 7)
    with c4:
        target_roas = st.number_input("ROAS alvo", min_value=0.0, value=2.0, step=0.1)

    c5, c6, c7 = st.columns(3)
    with c5:
        margin_pct = st.number_input(
            "Margem de contribuição (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            help="Obrigatória para calcular lucro econômico. Informe a margem real antes da mídia.",
        )
    with c6:
        risk_aversion = st.slider(
            "Aversão a risco",
            min_value=0.0,
            max_value=2.0,
            value=0.25,
            step=0.05,
        )
    with c7:
        include_inactive = st.checkbox("Gerar decisões também para entidades inativas", value=False, help="O histórico inativo sempre entra como contexto estatístico. Marque apenas se quiser receber ações também para entidades hoje inativas.")
        validate_temporal = st.checkbox(
            "Validar modelos temporais fora da amostra",
            value=True,
        )
        weekly_seasonality = st.checkbox(
            "Modelar sazonalidade semanal",
            value=True,
        )

    mcmc_method = "auto"
    mcmc_draws = 1200
    mcmc_tune = 1200
    if inference_mode == "MCMC hierárquico profundo":
        with st.expander("Configuração MCMC", expanded=True):
            st.warning(
                "Esse modo pode consumir bastante CPU/RAM. Instale as dependências uma vez "
                "com install_deep_windows.bat."
            )
            d1, d2, d3 = st.columns(3)
            with d1:
                mcmc_method = st.selectbox("Método", ["auto", "nuts", "advi"], index=0)
            with d2:
                mcmc_draws = st.number_input(
                    "Amostras posteriores",
                    min_value=400,
                    max_value=10000,
                    value=1200,
                    step=200,
                )
            with d3:
                mcmc_tune = st.number_input(
                    "Amostras de adaptação",
                    min_value=400,
                    max_value=10000,
                    value=1200,
                    step=200,
                )

    if not uploaded:
        st.info("Envie um CSV ou XLSX para começar.")
        return

    if not st.button("Executar análise quantitativa", type="primary", use_container_width=True):
        return

    if float(margin_pct) <= 0.0:
        st.error(
            "Informe uma margem de contribuição maior que 0%. "
            "O motor não assume 100% de margem automaticamente."
        )
        return

    try:
        with st.status("Executando análise local...", expanded=True) as status:
            telemetry = RunTelemetry()
            monitor = RunMonitor(
                validate_temporal=bool(validate_temporal),
                deep_mode=(
                    inference_mode
                    == "MCMC hierárquico profundo"
                ),
                telemetry=telemetry,
            )
            monitor.update(
                "load",
                0.05,
                "Lendo arquivo enviado...",
                log=True,
            )
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / uploaded.name
                path.write_bytes(uploaded.getbuffer())
                df = load_ads_file(path)
            monitor.finish_stage(
                "load",
                f"{len(df):,} linhas normalizadas.",
                rows=int(len(df)),
                columns=int(len(df.columns)),
            )
            monitor.update(
                "scope",
                0.10,
                "Identificando estrutura ativa e auditando a base...",
                log=True,
            )

            decision_entities = None
            operational_df = df
            if not include_inactive:
                monitor.update(
                    "scope",
                    0.30,
                    "Mantendo histórico completo como contexto e separando ativos...",
                    log=True,
                )
                universe = infer_decision_universe(df)
                decision_entities = {
                    "campaign": universe.campaign_ids,
                    "adset": universe.adset_ids,
                    "ad": universe.ad_ids,
                }
                operational_df = filter_decision_rows(
                    df,
                    universe,
                )
                st.caption(
                    f"Detecção de ativos: {universe.detection_method} | "
                    f"{len(universe.campaign_ids)} campanhas | "
                    f"{len(universe.adset_ids)} conjuntos | "
                    f"{len(universe.ad_ids)} anúncios."
                )

            quality = assess_data_quality(operational_df)
            funnel_schema = detect_funnel_schema(df)
            funnel_detail = hierarchical_funnel_diagnostics(df)
            st.write(
                f"Base: {quality.rows:,} linhas | {quality.days} dias | "
                f"{quality.campaigns} campanhas | {quality.adsets} conjuntos | "
                f"{quality.ads} anúncios | qualidade {quality.score:.0f}/100."
            )
            st.caption(
                f"Cobertura calendário: {quality.calendar_coverage_ratio:.1%} | "
                f"campanha mediana ativa: {quality.median_campaign_active_days:.0f} dias | "
                f"pares com <5 dias de sobreposição: "
                f"{quality.low_overlap_campaign_pair_fraction:.1%}."
            )
            for warning in quality.warnings:
                st.warning(warning)

            draws = DEPTHS[depth] or hw.recommended_draws
            temporal_model = (
                "state_space"
                if temporal_label.startswith("State-space")
                else "derivative"
            )
            config = EngineConfig(
                target_roas=float(target_roas),
                contribution_margin=float(margin_pct) / 100.0,
                horizon_days=int(horizon_days),
                draws=int(draws),
                risk_aversion=float(risk_aversion),
                temporal_model=temporal_model,
                use_weekly_seasonality=weekly_seasonality,
            )

            historical_days = int(
                df["date"].nunique()
            )
            active_campaigns_count = int(
                operational_df["campaign_id"].nunique()
            )
            active_adsets_count = int(
                operational_df["adset_id"].nunique()
            )
            active_ads_count = int(
                operational_df["ad_id"].nunique()
            )
            min_train_days = 21
            validation_step = max(
                1,
                int(horizon_days),
            )
            if (
                validate_temporal
                and historical_days
                >= min_train_days + int(horizon_days)
            ):
                last_origin_index = (
                    historical_days
                    - int(horizon_days)
                    - 1
                )
                rolling_origins = len(
                    range(
                        min_train_days - 1,
                        last_origin_index + 1,
                        validation_step,
                    )
                )
            else:
                rolling_origins = 0

            monitor.configure_exploration(
                historical_days=historical_days,
                historical_campaigns=int(
                    df["campaign_id"].nunique()
                ),
                historical_adsets=int(
                    df["adset_id"].nunique()
                ),
                historical_ads=int(
                    df["ad_id"].nunique()
                ),
                active_campaigns=active_campaigns_count,
                active_adsets=active_adsets_count,
                active_ads=active_ads_count,
                horizon_days=int(horizon_days),
                actions=config.actions,
                draws=int(draws),
                rolling_origins=rolling_origins,
                temporal_models=(
                    2 if rolling_origins > 0 else 0
                ),
                temporal_model=temporal_model,
                weekly_seasonality=bool(
                    weekly_seasonality
                ),
            )
            monitor.finish_stage(
                "scope",
                (
                    f"{active_campaigns_count} campanhas, "
                    f"{active_adsets_count} conjuntos e "
                    f"{active_ads_count} anúncios ativos; "
                    f"qualidade {quality.score:.0f}/100."
                ),
                active_campaigns=active_campaigns_count,
                active_adsets=active_adsets_count,
                active_ads=active_ads_count,
                quality_score=float(quality.score),
            )

            diagnostics = None
            ppc_summary = None
            deep_decision_source = None
            deep_guardrail = None
            if inference_mode == "MCMC hierárquico profundo":
                monitor.update(
                    "inference",
                    0.02,
                    "Ajustando posterior hierárquico profundo...",
                    log=True,
                )
                from quant_trafego.deep_analysis import run_deep_analysis

                result = run_deep_analysis(
                    df,
                    engine_config=config,
                    decision_entities=decision_entities,
                    mcmc_draws=int(mcmc_draws),
                    mcmc_tune=int(mcmc_tune),
                    mcmc_chains=hw.recommended_mcmc_chains,
                    mcmc_cores=hw.recommended_mcmc_cores,
                    mcmc_method=mcmc_method,
                )
                all_actions = result.all_actions
                best = result.best_actions
                diagnostics = result.diagnostics
                ppc_summary = result.ppc_summary
                deep_decision_source = result.decision_source
                deep_guardrail = result.guardrail
                monitor.finish_stage(
                    "inference",
                    (
                        f"Inferência profunda concluída via "
                        f"{diagnostics.method.upper()}."
                    ),
                    inference_method=diagnostics.method,
                    deep_guardrail=deep_guardrail,
                )
            else:
                monitor.update(
                    "inference",
                    0.01,
                    "Iniciando árvore hierárquica e cenários contrafactuais...",
                    log=True,
                )
                engine = BayesTrafficEngine(config)
                all_actions, best = engine.run(
                    df,
                    decision_entities=decision_entities,
                    progress_callback=monitor.engine_callback,
                )
                monitor.finish_stage(
                    "inference",
                    (
                        f"{len(best)} entidades com decisão calculada; "
                        f"{len(all_actions):,} ações avaliadas."
                    ),
                    decision_entities=int(len(best)),
                    action_rows=int(len(all_actions)),
                    draws=int(draws),
                )

            model_comparison = None
            model_decision = None
            if (
                validate_temporal
                and quality.days >= 21 + int(horizon_days)
            ):
                monitor.update(
                    "validation",
                    0.0,
                    (
                        f"Comparando derivative vs state-space em "
                        f"{rolling_origins} janelas rolling-origin..."
                    ),
                    log=True,
                )
                comparison_config = replace(
                    config,
                    draws=min(int(draws), 5000),
                )
                model_comparison, model_decision = compare_temporal_models(
                    operational_df,
                    config=comparison_config,
                    min_train_days=21,
                    horizon_days=int(horizon_days),
                    step_days=max(1, int(horizon_days)),
                    progress_callback=monitor.temporal_callback,
                )
                monitor.finish_stage(
                    "validation",
                    (
                        "Validação temporal concluída; "
                        + (
                            "state-space promovido."
                            if model_decision.get(
                                "promote_state_space"
                            )
                            else "derivada local permanece referência."
                        )
                    ),
                    rolling_origins=rolling_origins,
                    temporal_model_decision=model_decision,
                )
            elif validate_temporal:
                monitor.finish_stage(
                    "validation",
                    "Histórico insuficiente para rolling-origin; etapa ignorada.",
                    rolling_origins=0,
                )

            monitor.update(
                "allocation",
                0.05,
                "Definindo envelope total de capital...",
                log=True,
            )
            account_budget_target = derive_account_budget_target(
                best,
                source_df=operational_df,
                horizon_days=config.horizon_days,
            )
            allocation = None
            allocation_summary = None
            try:
                allocation, allocation_summary = optimize_campaign_portfolio(
                    all_actions,
                    df,
                    contribution_margin=config.contribution_margin,
                    total_budget=account_budget_target["portfolio_budget_cap_horizon"],
                )
            except Exception as portfolio_exc:
                try:
                    allocation, allocation_summary = optimize_campaign_allocation(
                        all_actions,
                        total_budget=account_budget_target["recommended_horizon_amount"],
                    )
                    allocation_summary["fallback_reason"] = str(portfolio_exc)
                except Exception as allocation_exc:
                    allocation_summary = {
                        "status": "unavailable",
                        "reason": str(allocation_exc),
                        "portfolio_reason": str(portfolio_exc),
                    }

            monitor.update(
                "allocation",
                0.55,
                "Capital da conta distribuído entre campanhas; reconciliando conjuntos...",
                log=True,
                account_budget_horizon=float(
                    account_budget_target[
                        "recommended_horizon_amount"
                    ]
                ),
            )
            adset_allocation = None
            adset_allocation_summary = None
            if allocation is not None:
                try:
                    adset_allocation, adset_allocation_summary = optimize_adset_allocation(
                        all_actions,
                        allocation,
                    )
                except Exception as adset_exc:
                    adset_allocation_summary = {
                        "status": "unavailable",
                        "reason": str(adset_exc),
                    }

            monitor.finish_stage(
                "allocation",
                (
                    f"Alocação concluída para "
                    f"{0 if allocation is None else len(allocation)} campanhas "
                    f"e {0 if adset_allocation is None else len(adset_allocation)} conjuntos."
                ),
                campaigns_allocated=(
                    0 if allocation is None else int(len(allocation))
                ),
                adsets_allocated=(
                    0 if adset_allocation is None else int(len(adset_allocation))
                ),
            )
            monitor.update(
                "plan",
                0.10,
                "Convertendo decisões quantitativas em ações operacionais...",
                log=True,
            )

            operational_plan = build_operational_action_plan(
                best,
                allocation=allocation,
                adset_allocation=adset_allocation,
                account_budget_target=account_budget_target,
                source_df=operational_df,
                horizon_days=config.horizon_days,
            )
            monitor.finish_stage(
                "plan",
                f"{len(operational_plan)} linhas no plano operacional.",
                operational_rows=int(len(operational_plan)),
            )
            monitor.update(
                "persist",
                0.10,
                "Salvando snapshot, manifest e diagnósticos...",
                log=True,
            )

            development_diagnostics = build_development_diagnostics(
                full_df=df,
                operational_df=operational_df,
                all_actions=all_actions,
                best_actions=best,
                quality=quality,
                telemetry=telemetry,
                inference_mode=(
                    f"mcmc_{diagnostics.method}"
                    if diagnostics is not None
                    else "empirical_bayes"
                ),
                config=config,
                model_decision=model_decision,
                allocation_summary=allocation_summary,
                adset_allocation_summary=adset_allocation_summary,
                diagnostics=diagnostics,
                ppc_summary=ppc_summary,
                deep_decision_source=deep_decision_source,
                deep_guardrail=deep_guardrail,
            )

            sanity = development_diagnostics.get(
                "sanity_checks",
                {},
            )
            if sanity.get(
                "zero_portfolio_despite_recent_profitable_campaigns"
            ):
                st.error(
                    "SANITY CHECK: o portfólio selecionou spend zero apesar de "
                    "existirem campanhas recentemente lucrativas. Não execute "
                    "esse plano sem revisão; o diagnóstico foi marcado para "
                    "investigação."
                )

            manifest = build_run_manifest(
                df,
                config=config,
                inference_mode=(
                    f"mcmc_{diagnostics.method}"
                    if diagnostics is not None
                    else "empirical_bayes"
                ),
                seed=config.seed,
                extra={
                    "quality_score": quality.score,
                    "quality_warnings": list(quality.warnings),
                    "quality_report": asdict(quality),
                    "mcmc_diagnostics": (
                        diagnostics.__dict__ if diagnostics is not None else None
                    ),
                    "ppc_summary": (
                        ppc_summary.__dict__ if ppc_summary is not None else None
                    ),
                    "temporal_model_decision": model_decision,
                    "account_budget_target": account_budget_target,
                    "allocation_summary": allocation_summary,
                    "adset_allocation_summary": adset_allocation_summary,
                    "deep_decision_source": deep_decision_source,
                    "deep_guardrail": deep_guardrail,
                    "runtime_telemetry": telemetry.developer_snapshot(),
                    "development_diagnostics": development_diagnostics,
                },
            )
            workspace = LocalWarehouse("workspace")
            extra_tables = {}
            extra_json = {}
            if allocation is not None:
                extra_tables["allocation"] = allocation
            if adset_allocation is not None and not adset_allocation.empty:
                extra_tables["adset_allocation"] = adset_allocation
            if operational_plan is not None and not operational_plan.empty:
                extra_tables["operational_action_plan"] = operational_plan
            if funnel_detail is not None and not funnel_detail.empty:
                extra_tables["funnel_diagnostics"] = funnel_detail
            if model_comparison is not None and not model_comparison.empty:
                extra_tables["temporal_model_comparison"] = model_comparison
            if model_decision is not None:
                extra_json["temporal_model_decision"] = model_decision
            extra_json["account_budget_target"] = account_budget_target
            if allocation_summary is not None:
                extra_json["allocation_summary"] = allocation_summary
            if adset_allocation_summary is not None:
                extra_json["adset_allocation_summary"] = adset_allocation_summary
            extra_tables["runtime_telemetry"] = telemetry.dataframe()
            extra_tables["runtime_stage_summary"] = telemetry.stage_summary()
            extra_json["runtime_telemetry"] = telemetry.developer_snapshot()
            extra_json["development_diagnostics"] = development_diagnostics
            if diagnostics is not None:
                extra_tables["posterior_predictive_checks"] = result.ppc_detail
                if result.guardrail != "none":
                    extra_tables["mcmc_candidate_actions"] = result.candidate_mcmc_actions
            run_dir = workspace.persist_run(
                df,
                manifest,
                all_actions,
                best,
                extra_tables=extra_tables,
                extra_json=extra_json,
            )
            monitor.finish_stage(
                "persist",
                f"Execução auditável salva em {run_dir}.",
                run_dir=str(run_dir),
            )
            monitor.done()
            telemetry.dataframe().to_csv(
                run_dir / "runtime_telemetry.csv",
                index=False,
            )
            telemetry.stage_summary().to_csv(
                run_dir / "runtime_stage_summary.csv",
                index=False,
            )
            (
                run_dir / "runtime_telemetry.json"
            ).write_text(
                json.dumps(
                    telemetry.developer_snapshot(),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )
            development_diagnostics = build_development_diagnostics(
                full_df=df,
                operational_df=operational_df,
                all_actions=all_actions,
                best_actions=best,
                quality=quality,
                telemetry=telemetry,
                inference_mode=(
                    f"mcmc_{diagnostics.method}"
                    if diagnostics is not None
                    else "empirical_bayes"
                ),
                config=config,
                model_decision=model_decision,
                allocation_summary=allocation_summary,
                adset_allocation_summary=adset_allocation_summary,
                diagnostics=diagnostics,
                ppc_summary=ppc_summary,
                deep_decision_source=deep_decision_source,
                deep_guardrail=deep_guardrail,
            )
            (
                run_dir / "development_diagnostics.json"
            ).write_text(
                json.dumps(
                    development_diagnostics,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )
            status.update(label="Análise concluída.", state="complete", expanded=False)

        st.caption(f"Execução auditável salva em: {run_dir}")

        if diagnostics is not None:
            st.subheader("Diagnóstico MCMC")
            if deep_guardrail != "none":
                st.warning(
                    f"Guardrail profundo ativo: {deep_guardrail}. "
                    f"Fonte decisória: {deep_decision_source}."
                )
            else:
                st.caption(f"Fonte decisória profunda: {deep_decision_source}")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Método", diagnostics.method.upper())
            d2.metric(
                "R-hat máx.",
                "n/a" if diagnostics.max_rhat is None else f"{diagnostics.max_rhat:.3f}",
            )
            d3.metric(
                "ESS bulk mín.",
                "n/a" if diagnostics.min_ess_bulk is None else f"{diagnostics.min_ess_bulk:.0f}",
            )
            d4.metric(
                "Divergências",
                "n/a" if diagnostics.divergences is None else str(diagnostics.divergences),
            )
            if diagnostics.converged is False:
                st.error("O NUTS não atingiu todos os critérios de convergência definidos.")
            elif diagnostics.converged is True:
                st.success("Diagnósticos principais de convergência aprovados.")

            if ppc_summary is not None:
                st.subheader("Posterior predictive checks")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Status PPC", ppc_summary.status.upper())
                p2.metric(
                    "Cobertura cliques 90%",
                    f"{ppc_summary.click_90_coverage:.1%}",
                )
                p3.metric(
                    "Cobertura conversões 90%",
                    f"{ppc_summary.conversion_90_coverage:.1%}",
                )
                p4.metric(
                    "Conversões extremas",
                    f"{ppc_summary.conversion_extreme_fraction:.1%}",
                )

        account = best[best["level"] == "account"]
        st.subheader("Visão geral")
        if not account.empty:
            row = account.iloc[0]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Escala uniforme macro", f"{row['action_multiplier']:.1f}x")
            m2.metric("Lucro esperado", _fmt_money(row["expected_profit"]))
            m3.metric("P(lucro)", f"{row['p_profit']:.1%}")
            m4.metric("P(ação ótima)", f"{row['p_action_optimal']:.1%}")
            m5.metric("Score decisão", f"{row['decision_score']:.1%}")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("P(ROAS alvo)", f"{row['p_roas_target']:.1%}")
            s2.metric("Mudança de regime", f"{row['regime_change_score']:.1%}")
            s3.metric("Instabilidade", f"{row['instability_score']:.1%}")
            s4.metric("Elasticidade resposta", f"{row['response_elasticity']:.3f}")
            if (
                float(row.get("instability_score", 0.0)) >= 0.80
                or float(row.get("decision_score", 1.0)) < 0.40
            ):
                st.warning(
                    "O cenário macro da conta está em regime altamente instável "
                    "ou com score decisório baixo. Ele é tratado como sinal de "
                    "risco e não como ordem para zerar/escalar todo o portfólio."
                )

        display_cols = [
            "level",
            "entity_id",
            "posterior_source",
            "action_multiplier",
            "unconstrained_best_multiplier",
            "policy_constrained",
            "historical_spend",
            "historical_roas",
            "expected_profit",
            "expected_incremental_profit_vs_hold",
            "expected_roas",
            "p_profit",
            "p_roas_target",
            "p_beats_hold",
            "p_action_optimal",
            "cvar10_profit",
            "expected_regret",
            "regime_change_score",
            "instability_score",
            "response_elasticity",
            "response_confidence",
            "decision_score",
            "risk_adjusted_utility",
            "opportunity_score",
        ]

        tab_plan, tab_overall, tab_campaign, tab_adset, tab_ad, tab_funnel, tab_alloc, tab_validation, tab_all = st.tabs(
            [
                "Plano operacional",
                "Decisões",
                "Campanhas",
                "Conjuntos",
                "Anúncios",
                "Funil",
                "Alocação global",
                "Validação",
                "Todas as simulações",
            ]
        )
        with tab_plan:
            campaign_plan = operational_plan[
                operational_plan["level"] == "campaign"
            ]
            deployed_daily = float(
                campaign_plan[
                    "recommended_daily_amount"
                ].fillna(0.0).sum()
            )
            current_deployed_daily = float(
                campaign_plan[
                    "current_daily_amount"
                ].fillna(0.0).sum()
            )
            capital_ceiling_daily = float(
                account_budget_target.get(
                    "portfolio_budget_cap_daily",
                    account_budget_target[
                        "recommended_daily_amount"
                    ],
                )
            )
            unallocated_daily = max(
                capital_ceiling_daily
                - deployed_daily,
                0.0,
            )

            b1, b2, b3, b4 = st.columns(4)
            b1.metric(
                "Spend atual/dia",
                _fmt_money(current_deployed_daily),
            )
            b2.metric(
                "Spend recomendado/dia",
                _fmt_money(deployed_daily),
            )
            b3.metric(
                "Mudança executável/dia",
                _fmt_money(
                    deployed_daily
                    - current_deployed_daily
                ),
            )
            b4.metric(
                "Capital não alocado/dia",
                _fmt_money(unallocated_daily),
                help=(
                    "Parte do teto de capital da conta que o portfólio "
                    "não encontrou justificativa estatística para empregar."
                ),
            )
            macro_uniform_daily = float(
                account_budget_target[
                    "uniform_account_scenario_daily_amount"
                ]
            )
            st.caption(
                "Cenário macro de escala uniforme da conta: "
                f"{_fmt_money(macro_uniform_daily)}/dia. "
                "Esse cenário é apenas um sinal de risco e NÃO define o "
                "orçamento do portfólio. Teto seletivo permitido ao solver: "
                f"{_fmt_money(capital_ceiling_daily)}/dia. "
                "O spend recomendado é o capital que o solver realmente "
                "encontrou justificativa para empregar nas campanhas."
            )
            if operational_plan.empty:
                st.info("Nenhuma ação operacional disponível.")
            else:
                counts = operational_plan["capital_action"].value_counts()
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Aumentar/priorizar", int(
                    counts.get("AUMENTAR", 0) + counts.get("PRIORIZAR_MAIS", 0)
                ))
                p2.metric("Reduzir", int(
                    counts.get("REDUZIR", 0) + counts.get("REDUZIR_EXPOSICAO", 0)
                ))
                p3.metric("Desligar", int(counts.get("DESLIGAR", 0)))
                p4.metric(
                    "Duplicação/teste",
                    int(
                        operational_plan["duplicate_action"]
                        .isin(["DUPLICAR", "TESTAR_DUPLICACAO"])
                        .sum()
                    ),
                )
                st.dataframe(
                    operational_plan[
                        [
                            "level",
                            "campaign_name",
                            "adset_name",
                            "ad_name",
                            "capital_action",
                            "model_suggested_action",
                            "blocked_by_parent",
                            "current_daily_amount",
                            "configured_daily_budget",
                            "recommended_daily_amount",
                            "daily_amount_change",
                            "duplicate_action",
                            "suggested_additional_copies",
                            "expected_incremental_profit",
                            "expected_incremental_revenue",
                            "p_incremental_profit_positive",
                            "p_profit",
                            "p_roas_target",
                            "decision_score",
                            "execution_note",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
        with tab_overall:
            st.dataframe(best[display_cols], use_container_width=True, hide_index=True)
        with tab_campaign:
            st.dataframe(
                best[best["level"] == "campaign"][display_cols],
                use_container_width=True,
                hide_index=True,
            )
        with tab_adset:
            st.dataframe(
                best[best["level"] == "adset"][display_cols],
                use_container_width=True,
                hide_index=True,
            )
        with tab_ad:
            st.dataframe(
                best[best["level"] == "ad"][display_cols],
                use_container_width=True,
                hide_index=True,
            )
        with tab_funnel:
            if funnel_detail.empty:
                st.info("Nenhuma transição adicional de funil disponível.")
            else:
                st.caption(
                    "Etapas detectadas: "
                    + " → ".join(funnel_schema.available_stages)
                )
                st.dataframe(
                    funnel_detail,
                    use_container_width=True,
                    hide_index=True,
                )
                violations = int(funnel_detail["tracking_violation_rows"].sum())
                if violations > 0:
                    st.warning(
                        f"Foram detectadas {violations} violações de monotonicidade do funil; "
                        "essas transições não recebem posterior Binomial quando inválidas."
                    )
        with tab_alloc:
            if allocation is None:
                st.warning(allocation_summary.get("reason", "Alocação indisponível."))
            else:
                a1, a2, a3 = st.columns(3)
                a1.metric(
                    "Budget selecionado",
                    _fmt_money(allocation_summary["selected_spend"]),
                )
                expected_portfolio_profit = allocation_summary.get(
                    "expected_portfolio_profit",
                    allocation_summary.get("expected_portfolio_profit_additive", 0.0),
                )
                a2.metric(
                    "Lucro esperado do portfólio",
                    _fmt_money(expected_portfolio_profit),
                )
                if "scenario_portfolio_cvar" in allocation_summary:
                    a3.metric(
                        "CVaR do portfólio",
                        _fmt_money(allocation_summary["scenario_portfolio_cvar"]),
                    )
                else:
                    a3.metric(
                        "Regret aditivo",
                        _fmt_money(allocation_summary.get("expected_regret_additive", 0.0)),
                    )
                st.dataframe(
                    allocation[
                        [
                            "entity_id",
                            "evidence_tier",
                            "action_multiplier",
                            "expected_spend",
                            "expected_profit",
                            "p_profit",
                            "p_incremental_profit_positive",
                            "cvar10_profit",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(allocation_summary["important_limitation"])
                if adset_allocation is not None and not adset_allocation.empty:
                    st.markdown("**Alocação reconciliada de conjuntos**")
                    st.dataframe(
                        adset_allocation[
                            [
                                "campaign_id",
                                "entity_id",
                                "adset_name",
                                "action_multiplier",
                                "expected_spend",
                                "parent_campaign_budget_limit",
                                "expected_profit",
                                "expected_revenue",
                                "p_incremental_profit_positive",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
        with tab_validation:
            if model_comparison is None:
                st.info(
                    "Comparação temporal não executada ou histórico insuficiente."
                )
            elif model_comparison.empty:
                st.info("Histórico insuficiente para comparação rolling-origin.")
            else:
                st.dataframe(
                    model_comparison,
                    use_container_width=True,
                    hide_index=True,
                )
                if model_decision.get("promote_state_space"):
                    st.success(
                        "State-space passou os gates de promoção nesta base."
                    )
                else:
                    st.warning(
                        "State-space não passou todos os gates; derivada local permanece referência."
                    )
                st.json(model_decision)
        with tab_all:
            st.dataframe(all_actions, use_container_width=True, hide_index=True)

        st.download_button(
            "Baixar diagnóstico de desenvolvimento (JSON)",
            json.dumps(
                development_diagnostics,
                indent=2,
                ensure_ascii=False,
                default=str,
            ).encode("utf-8"),
            file_name="development_diagnostics.json",
            mime="application/json",
        )
        st.download_button(
            "Baixar plano operacional (CSV)",
            operational_plan.to_csv(index=False).encode("utf-8-sig"),
            file_name="operational_action_plan.csv",
            mime="text/csv",
        )
        st.download_button(
            "Baixar melhores ações (CSV)",
            best.to_csv(index=False).encode("utf-8-sig"),
            file_name="best_actions.csv",
            mime="text/csv",
        )
        st.download_button(
            "Baixar todas as simulações (CSV)",
            all_actions.to_csv(index=False).encode("utf-8-sig"),
            file_name="all_actions.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.exception(exc)


if __name__ == "__main__":
    main()
