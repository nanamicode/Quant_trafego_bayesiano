from quant_trafego.io import load_ads_file


def test_native_meta_ptbr_export_aliases_are_recognized(tmp_path):
    import pandas as pd

    path = tmp_path / "meta.csv"
    pd.DataFrame(
        [
            {
                "Início dos relatórios": "2026-09-01",
                "Nome da campanha": "C1",
                "Identificação da campanha": "1001",
                "Nome do conjunto de anúncios": "S1",
                "Identificação do conjunto de anúncios": "2001",
                "Nome do anúncio": "A1",
                "Identificação do anúncio": "3001",
                "Veiculação do anúncio": "active",
                "Impressões": 1000,
                "Cliques no link": 50,
                "Compras": 5,
                "Valor gasto c/ Imposto": 100.0,
                "Valor de conversão das compras diretas no site": 500.0,
                "Finalizações de compra iniciadas": 10,
                "Orçamento do conjunto de anúncios": 120.0,
            }
        ]
    ).to_csv(path, index=False)

    out = load_ads_file(path)
    for col in [
        "date",
        "campaign_id",
        "campaign_name",
        "adset_id",
        "adset_name",
        "ad_id",
        "ad_name",
        "status",
        "impressions",
        "clicks",
        "conversions",
        "spend",
        "revenue",
        "checkouts",
        "adset_daily_budget",
    ]:
        assert col in out.columns
