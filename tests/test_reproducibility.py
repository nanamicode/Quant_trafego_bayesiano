import pandas as pd

from quant_trafego.engine import EngineConfig
from quant_trafego.reproducibility import build_run_manifest, dataframe_fingerprint


def test_dataframe_fingerprint_is_order_invariant_for_standard_keys():
    df = pd.read_csv("examples/example_data.csv")
    a = dataframe_fingerprint(df)
    b = dataframe_fingerprint(df.sample(frac=1.0, random_state=1))
    assert a == b


def test_manifest_has_core_reproducibility_fields():
    df = pd.read_csv("examples/example_data.csv")
    manifest = build_run_manifest(
        df,
        config=EngineConfig(draws=100),
        inference_mode="test",
        seed=42,
    )
    assert manifest["data_sha256"]
    assert manifest["run_id"]
    assert manifest["python"]
    assert "hardware" in manifest