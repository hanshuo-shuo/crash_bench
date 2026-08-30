from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/run_scoped_training.py"


def test_scoped_training_orders_selection_refit_then_calibration():
    source = SCRIPT.read_text()
    assert source.index('"--fit-on", "train"') < source.index('"--fit-on", "train_development"')
    assert source.index('"--fit-on", "train_development"') < source.index('"scripts/expansion/calibrate_selector.py"')
    assert '"test_rows_read": 0' in source
