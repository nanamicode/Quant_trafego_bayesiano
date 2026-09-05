from __future__ import annotations

from pathlib import Path
import pandas as pd


ALIASES = {
    "date": ["date", "day", "data", "date_start", "dia", "início dos relatórios", "inicio dos relatorios", "início dos relatorios", "inicio dos relatórios"],
    "campaign_id": ["campaign_id", "id_da_campanha", "id campanha", "campaign id", "identificação da campanha", "identificacao da campanha"],
    "campaign_name": ["campaign_name", "nome_da_campanha", "nome campanha", "campaign name"],
    "campaign_daily_budget": ["campaign_daily_budget", "campaign budget", "daily campaign budget", "orcamento_diario_da_campanha", "orçamento diário da campanha", "orcamento_da_campanha", "orçamento da campanha"],
    "adset_id": ["adset_id", "id_do_conjunto", "id conjunto", "ad set id", "adset id", "identificação do conjunto de anúncios", "identificacao do conjunto de anuncios", "id do conjunto de anúncios", "id do conjunto de anuncios"],
    "adset_name": ["adset_name", "nome_do_conjunto", "nome conjunto", "ad set name", "adset name", "nome do conjunto de anúncios", "nome do conjunto de anuncios"],
    "adset_daily_budget": ["adset_daily_budget", "ad set budget", "adset budget", "daily ad set budget", "orcamento_diario_do_conjunto", "orçamento diário do conjunto", "orcamento_do_conjunto", "orçamento do conjunto", "orçamento do conjunto de anúncios", "orcamento do conjunto de anuncios"],
    "ad_id": ["ad_id", "id_do_anuncio", "id anúncio", "id anuncio", "ad id", "identificação do anúncio", "identificacao do anuncio", "id do anúncio", "id do anuncio"],
    "ad_name": ["ad_name", "nome_do_anuncio", "nome anúncio", "nome anuncio", "ad name", "nome do anúncio", "nome do anuncio"],
    "status": ["status", "delivery", "veiculacao", "veiculação", "veiculação do anúncio", "veiculacao do anuncio", "veiculação do conjunto de anúncios", "veiculacao do conjunto de anuncios", "veiculação da campanha", "veiculacao da campanha"],
    "impressions": ["impressions", "impressoes", "impressões"],
    "reach": ["reach", "alcance"],
    "frequency": ["frequency", "frequencia", "frequência"],
    "clicks": ["clicks", "link_clicks", "cliques", "cliques_no_link", "cliques no link"],
    "landing_page_views": ["landing_page_views", "landing page views", "visualizacoes_da_pagina_de_destino", "visualizações da página de destino", "lpv"],
    "adds_to_cart": ["adds_to_cart", "add_to_cart", "adicoes_ao_carrinho", "adições ao carrinho", "atc"],
    "checkouts": ["checkouts", "initiate_checkout", "checkouts_iniciados", "inicios_de_finalizacao", "inícios de finalização", "finalizações de compra iniciadas", "finalizacoes de compra iniciadas"],
    "conversions": ["conversions", "purchases", "compras", "purchase", "results"],
    "spend": ["spend", "amount_spent", "valor_gasto", "valor gasto", "gasto", "valor gasto c_ imposto", "valor gasto c/ imposto", "valor gasto c_ Imposto", "valor gasto c/ Imposto"],
    "revenue": ["revenue", "purchase_conversion_value", "valor_de_conversao", "valor de conversão", "faturamento", "valor de conversão das compras diretas no site", "valor de conversao das compras diretas no site", "valor de conversão das compras", "valor de conversao das compras"],
}


def _norm(s: str) -> str:
    return (
        str(s)
        .strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
    )


def _infer_mapping(columns) -> dict[str, str]:
    normalized = {_norm(c): c for c in columns}
    mapping = {}
    for target, aliases in ALIASES.items():
        for alias in aliases:
            key = _norm(alias)
            if key in normalized:
                mapping[normalized[key]] = target
                break
    return mapping


def load_ads_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xltx"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
    else:
        raise ValueError("Formato não suportado. Use .csv ou .xlsx.")

    df = df.rename(columns=_infer_mapping(df.columns))
    return df


def filter_active(df: pd.DataFrame) -> pd.DataFrame:
    if "status" not in df.columns:
        return df.copy()

    active_terms = {
        "active", "ativo", "ativa", "learning", "aprendizado",
        "learning limited", "aprendizado limitado", "enabled"
    }
    status = df["status"].astype(str).str.strip().str.lower()
    mask = status.isin(active_terms)

    # Se nenhum termo conhecido for encontrado, não descartamos a base.
    return df.loc[mask].copy() if mask.any() else df.copy()
