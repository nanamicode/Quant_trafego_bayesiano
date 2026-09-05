import numpy as np

from quant_trafego.calibration import calibration_table, expected_calibration_error


def test_calibration_table_and_ece():
    p = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    table = calibration_table(p, y, bins=5)
    ece = expected_calibration_error(p, y, bins=5)
    assert not table.empty
    assert 0 <= ece <= 1
    assert ece < 0.25
