from quant_trafego.io import infer_decision_universe, load_ads_file


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


def test_decision_universe_prefers_current_delivery_status():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "date": "2026-09-05",
                "campaign_id": "c_active",
                "adset_id": "s_active",
                "ad_id": "a_active",
                "status": "active",
                "spend": 10.0,
                "impressions": 1000,
            },
            {
                "date": "2026-09-05",
                "campaign_id": "c_paused",
                "adset_id": "s_paused",
                "ad_id": "a_paused",
                "status": "paused",
                "spend": 100.0,
                "impressions": 10000,
            },
        ]
    )
    universe = infer_decision_universe(df)
    assert universe.detection_method == "delivery_status"
    assert universe.campaign_ids == frozenset({"c_active"})
    assert universe.adset_ids == frozenset({"s_active"})
    assert universe.ad_ids == frozenset({"a_active"})


def test_decision_universe_falls_back_to_recent_delivery():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "date": "2026-09-01",
                "campaign_id": "old",
                "adset_id": "old_s",
                "ad_id": "old_a",
                "spend": 100.0,
                "impressions": 10000,
            },
            {
                "date": "2026-09-05",
                "campaign_id": "current",
                "adset_id": "current_s",
                "ad_id": "current_a",
                "spend": 20.0,
                "impressions": 2000,
            },
        ]
    )
    universe = infer_decision_universe(
        df,
        recent_days=3,
    )
    assert universe.detection_method == "recent_delivery_activity"
    assert universe.campaign_ids == frozenset({"current"})
