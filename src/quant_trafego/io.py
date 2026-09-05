from __future__ import annotations

from pathlib import Path
import pandas as pd


ALIASES = {
    "date": ["date", "day", "data", "date_start", "dia"],
    "campaign_id": ["campaign_id", "id_da_campanha", "id campanha", "campaign id"],
    "campaign_name": ["campaign_name", "nome_da_campanha", "nome campanha", "campaign name"],
    "adset_id": ["adset_id", "id_do_conjunto", "id conjunto", "ad set id", "adset id"],
    "adset_name": ["adset_name", "nome_do_conjunto", "nome conjunto", "ad set name", "adset name"],
    "ad_id": ["ad_id", "id_do_anuncio", "id anúncio", "id anuncio", "ad id"],
    "ad_name": ["ad_name", "nome_do_anuncio", "nome anúncio", "nome anuncio", "ad name"],
    "status": ["status", "delivery", "veiculacao", "veiculação"],
    "impressions": ["impressions", "impressoes", "impressões"],
    "clicks": ["clicks", "link_clicks", "cliques", "cliques_no_link", "cliques no link"],
    "conversions": ["conversions", "purchases", "compras", "purchase", "results"],
    "spend": ["spend", "amount_spent", "valor_gasto", "valor gasto", "gasto"],
    "revenue": ["revenue", "purchase_conversion_value", "valor_de_conversao", "valor de conversão", "faturamento"],
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
