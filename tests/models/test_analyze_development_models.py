from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/analyze_development_models.py"
SPEC = importlib.util.spec_from_file_location("analyze_development_models", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_choices_group_predictions_by_block():
    rows = [
        {"row_id": "b0:base_continue", "option_id": "base_continue", "predicted_u0": 0.1},
        {"row_id": "b0:safe_stop", "option_id": "safe_stop", "predicted_u0": 0.2},
        {"row_id": "b1:base_continue", "option_id": "base_continue", "predicted_u0": 0.3},
        {"row_id": "b1:safe_stop", "option_id": "safe_stop", "predicted_u0": 0.2},
    ]
    assert MODULE.choices_from_prediction_rows(rows) == {
        "b0": "safe_stop",
        "b1": "base_continue",
    }


def test_development_analyzer_rejects_non_train_fit(tmp_path):
    seed_dirs = []
    for seed in range(5):
        directory = tmp_path / f"seed_{seed}"
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({
            "seed": seed, "fit_on": "train_development", "calibration_rows_read": 0,
            "test_rows_read": 0, "development_rows": 1,
        }))
        (directory / "predictions.jsonl").write_text("")
        (directory / "model.pt").write_bytes(b"model")
        seed_dirs.append(directory)
    try:
        MODULE.analyze(branches=[{"split_role": "development"}], seed_dirs=seed_dirs, baseline_dir=tmp_path)
    except ValueError as error:
        assert "train-only" in str(error)
    else:
        raise AssertionError("expected train/development refit artifact rejection")
