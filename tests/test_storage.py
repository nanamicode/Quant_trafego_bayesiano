import pandas as pd

from quant_trafego.storage import LocalWarehouse


def test_local_warehouse_deduplicates_snapshot(tmp_path):
    df = pd.read_csv("examples/example_data.csv")
    warehouse = LocalWarehouse(tmp_path)
    digest1, path1 = warehouse.store_snapshot(df)
    digest2, path2 = warehouse.store_snapshot(df)
    assert digest1 == digest2
    assert path1 == path2
    assert path1.exists()