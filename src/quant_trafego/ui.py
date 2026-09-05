from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from quant_trafego.engine import BayesTrafficEngine, EngineConfig
from quant_trafego.hardware import detect_hardware
from quant_trafego.io import filter_active, load_ads_file
from quant_trafego.quality import assess_data_quality


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
            st.write(
                f"Base: {quality.rows:,} linhas | {quality.days} dias | "
                f"{quality.campaigns} campanhas | {quality.adsets} conjuntos | "
                f"{quality.ads} anúncios | qualidade {quality.score:.0f}/100."
            )
            for warning in quality.warnings:
                st.warning(warning)

            draws = DEPTHS[depth] or hw.recommended_draws
            config = EngineConfig(
                target_roas=float(target_roas),
                contribution_margin=float(margin_pct) / 100.0,
                horizon_days=int(horizon_days),
                draws=int(draws),
                risk_aversion=float(risk_aversion),
            )

            diagnostics = None
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
            else:
                st.write(
                    "Estimando posteriores hierárquicos, derivadas temporais, "
                    "regime e resposta observacional ao gasto..."
                )
                engine = BayesTrafficEngine(config)
                all_actions, best = engine.run(df)

            status.update(label="Análise concluída.", state="complete", expanded=False)

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

        tab_overall, tab_campaign, tab_adset, tab_ad, tab_all = st.tabs(
            ["Decisões", "Campanhas", "Conjuntos", "Anúncios", "Todas as simulações"]
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
