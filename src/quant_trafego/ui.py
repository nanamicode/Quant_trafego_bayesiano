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
        f"Hardware detectado: {hw.cpu_threads} threads de CPU | RAM {ram} | "
        f"perfil {hw.label} | Monte Carlo automático: {hw.recommended_draws:,} amostras."
    )

    uploaded = st.file_uploader(
        "Planilha completa de tráfego pago",
        type=["csv", "xlsx", "xlsm", "xltx"],
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        depth = st.selectbox("Profundidade Monte Carlo", list(DEPTHS), index=0)
    with c2:
        horizon_days = st.number_input("Horizonte futuro (dias)", 1, 90, 7)
    with c3:
        target_roas = st.number_input("ROAS alvo", min_value=0.0, value=2.0, step=0.1)
    with c4:
        margin_pct = st.number_input(
            "Margem de contribuição (%)",
            min_value=0.0,
            max_value=100.0,
            value=100.0,
            step=1.0,
            help="Margem antes da mídia. Ex.: 40 significa lucro = receita × 40% − mídia.",
        )

    c5, c6 = st.columns(2)
    with c5:
        risk_aversion = st.slider(
            "Aversão a risco",
            min_value=0.0,
            max_value=2.0,
            value=0.25,
            step=0.05,
        )
    with c6:
        include_inactive = st.checkbox("Incluir entidades inativas", value=False)

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
                f"{quality.ads} anúncios | qualidade estrutural {quality.score:.0f}/100."
            )
            for warning in quality.warnings:
                st.warning(warning)

            draws = DEPTHS[depth] or hw.recommended_draws
            st.write(f"Monte Carlo: {draws:,} amostras por ação/entidade.")
            st.write(
                "Estimando posteriores hierárquicos, derivadas temporais, "
                "mudança de regime e resposta observacional ao gasto..."
            )

            engine = BayesTrafficEngine(
                EngineConfig(
                    target_roas=float(target_roas),
                    contribution_margin=float(margin_pct) / 100.0,
                    horizon_days=int(horizon_days),
                    draws=int(draws),
                    risk_aversion=float(risk_aversion),
                )
            )
            all_actions, best = engine.run(df)
            status.update(label="Análise concluída.", state="complete", expanded=False)

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
