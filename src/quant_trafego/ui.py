from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import tempfile

import streamlit as st

from quant_trafego.engine import BayesTrafficEngine, EngineConfig
from quant_trafego.funnel import detect_funnel_schema, hierarchical_funnel_diagnostics
from quant_trafego.hardware import detect_hardware
from quant_trafego.io import filter_active, load_ads_file
from quant_trafego.model_selection import compare_temporal_models
from quant_trafego.optimization import optimize_campaign_allocation
from quant_trafego.portfolio import optimize_campaign_portfolio
from quant_trafego.quality import assess_data_quality
from quant_trafego.reproducibility import build_run_manifest
from quant_trafego.storage import LocalWarehouse


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
            value=100.0,
            step=1.0,
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
        include_inactive = st.checkbox("Incluir entidades inativas", value=False)
        validate_temporal = st.checkbox(
            "Validar modelos temporais fora da amostra",
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

    try:
        with st.status("Executando análise local...", expanded=True) as status:
            st.write("Lendo e normalizando a planilha...")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / uploaded.name
                path.write_bytes(uploaded.getbuffer())
                df = load_ads_file(path)

            if not include_inactive:
                st.write("Filtrando entidades ativas...")
                df = filter_active(df)

            quality = assess_data_quality(df)
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
            )

            diagnostics = None
            ppc_summary = None
            if inference_mode == "MCMC hierárquico profundo":
                st.write(
                    "Ajustando posterior hierárquico completo e conectando-o "
                    "à árvore de decisões..."
                )
                from quant_trafego.deep_analysis import run_deep_analysis

                result = run_deep_analysis(
                    df,
                    engine_config=config,
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
            else:
                st.write(
                    "Estimando posteriores hierárquicos, derivadas temporais, "
                    "regime e resposta observacional ao gasto..."
                )
                engine = BayesTrafficEngine(config)
                all_actions, best = engine.run(df)

            model_comparison = None
            model_decision = None
            if (
                validate_temporal
                and quality.days >= 21 + int(horizon_days)
            ):
                st.write(
                    "Executando comparação temporal rolling-origin "
                    "em conta e campanhas..."
                )
                comparison_config = replace(
                    config,
                    draws=min(int(draws), 5000),
                )
                model_comparison, model_decision = compare_temporal_models(
                    df,
                    config=comparison_config,
                    min_train_days=21,
                    horizon_days=int(horizon_days),
                    step_days=max(1, int(horizon_days)),
                )

            allocation = None
            allocation_summary = None
            try:
                allocation, allocation_summary = optimize_campaign_portfolio(
                    all_actions,
                    df,
                    contribution_margin=config.contribution_margin,
                )
            except Exception as portfolio_exc:
                try:
                    allocation, allocation_summary = optimize_campaign_allocation(
                        all_actions
                    )
                    allocation_summary["fallback_reason"] = str(portfolio_exc)
                except Exception as allocation_exc:
                    allocation_summary = {
                        "status": "unavailable",
                        "reason": str(allocation_exc),
                        "portfolio_reason": str(portfolio_exc),
                    }

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
                    "allocation_summary": allocation_summary,
                },
            )
            workspace = LocalWarehouse("workspace")
            extra_tables = {}
            extra_json = {}
            if allocation is not None:
                extra_tables["allocation"] = allocation
            if funnel_detail is not None and not funnel_detail.empty:
                extra_tables["funnel_diagnostics"] = funnel_detail
            if model_comparison is not None and not model_comparison.empty:
                extra_tables["temporal_model_comparison"] = model_comparison
            if model_decision is not None:
                extra_json["temporal_model_decision"] = model_decision
            if allocation_summary is not None:
                extra_json["allocation_summary"] = allocation_summary
            if diagnostics is not None:
                extra_tables["posterior_predictive_checks"] = result.ppc_detail
            run_dir = workspace.persist_run(
                df,
                manifest,
                all_actions,
                best,
                extra_tables=extra_tables,
                extra_json=extra_json,
            )
            status.update(label="Análise concluída.", state="complete", expanded=False)

        st.caption(f"Execução auditável salva em: {run_dir}")

        if diagnostics is not None:
            st.subheader("Diagnóstico MCMC")
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
            m1.metric("Ação global", f"{row['action_multiplier']:.1f}x")
            m2.metric("Lucro esperado", _fmt_money(row["expected_profit"]))
            m3.metric("P(lucro)", f"{row['p_profit']:.1%}")
            m4.metric("P(ação ótima)", f"{row['p_action_optimal']:.1%}")
            m5.metric("Confiança decisão", f"{row['decision_confidence']:.1%}")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("P(ROAS alvo)", f"{row['p_roas_target']:.1%}")
            s2.metric("Mudança de regime", f"{row['regime_change_score']:.1%}")
            s3.metric("Instabilidade", f"{row['instability_score']:.1%}")
            s4.metric("Elasticidade resposta", f"{row['response_elasticity']:.3f}")

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
            "decision_confidence",
            "risk_adjusted_utility",
            "opportunity_score",
        ]

        tab_overall, tab_campaign, tab_adset, tab_ad, tab_funnel, tab_alloc, tab_validation, tab_all = st.tabs(
            [
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
