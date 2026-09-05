from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from quant_trafego.engine import BayesTrafficEngine, EngineConfig
from quant_trafego.io import filter_active, load_ads_file


DEPTHS = {
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
        "Análise local: a planilha é processada no próprio computador. "
        "A interface usa localhost; não é necessário servidor pago."
    )

    uploaded = st.file_uploader(
        "Planilha completa de tráfego pago",
        type=["csv", "xlsx", "xlsm", "xltx"],
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        depth = st.selectbox("Profundidade", list(DEPTHS), index=1)
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
            help="Margem antes do gasto de mídia. Ex.: 40 significa lucro = receita × 40% − mídia.",
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

    if not st.button("Analisar profundamente", type="primary", use_container_width=True):
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

            st.write(
                f"Base carregada: {len(df):,} linhas | "
                f"{df['campaign_id'].nunique() if 'campaign_id' in df else 0} campanhas | "
                f"{df['adset_id'].nunique() if 'adset_id' in df else 0} conjuntos | "
                f"{df['ad_id'].nunique() if 'ad_id' in df else 0} anúncios."
            )

            st.write("Aprendendo distribuição global e descendo a hierarquia...")
            engine = BayesTrafficEngine(
                EngineConfig(
                    target_roas=float(target_roas),
                    contribution_margin=float(margin_pct) / 100.0,
                    horizon_days=int(horizon_days),
                    draws=DEPTHS[depth],
                    risk_aversion=float(risk_aversion),
                )
            )
            all_actions, best = engine.run(df)
            status.update(label="Análise concluída.", state="complete", expanded=False)

        st.subheader("Visão geral")
        account = best[best["level"] == "account"]
        if not account.empty:
            row = account.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Ação global sugerida", f"{row['action_multiplier']:.1f}x")
            m2.metric("Lucro esperado", _fmt_money(row["expected_profit"]))
            m3.metric("P(lucro)", f"{row['p_profit']:.1%}")
            m4.metric("P(ROAS alvo)", f"{row['p_roas_target']:.1%}")

        display_cols = [
            "level", "entity_id", "action_multiplier", "historical_spend",
            "historical_roas", "expected_profit", "expected_roas", "p_profit",
            "p_roas_target", "p_beats_hold", "cvar10_profit",
            "expected_regret", "risk_adjusted_utility", "opportunity_score",
        ]

        st.subheader("Melhores ações por entidade")
        st.dataframe(
            best[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Todas as ações simuladas")
        level = st.selectbox("Filtrar nível", ["Todos", "account", "campaign", "adset", "ad"])
        shown = all_actions if level == "Todos" else all_actions[all_actions["level"] == level]
        st.dataframe(shown, use_container_width=True, hide_index=True)

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
