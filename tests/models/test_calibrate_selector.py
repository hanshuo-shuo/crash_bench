from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/calibrate_selector.py"
SPEC = importlib.util.spec_from_file_location("calibrate_selector", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_calibration_scores_cluster_by_source_and_use_pairwise_scale_floor():
    options = np.array(["base_continue", "observation_refresh", "safe_stop"] * 2)
    sources = np.array(["s0"] * 3 + ["s1"] * 3)
    row_ids = np.array([f"b{block}:{option}" for block in range(2) for option in options[:3]])
    actual_u = np.array([0.0, 1.0, -0.2, 0.0, 0.2, -0.2])
    actual_cat = np.array([1, 0, 0, 0, 0, 0], dtype=float)
    prediction = np.vstack([actual_u, actual_u + 0.01, actual_u - 0.01, actual_u, actual_u])
    cat_prediction = np.vstack([actual_cat] * 5)
    utility, difference, absolute = MODULE.calibration_source_scores(
        sources=sources, row_ids=row_ids, options=options,
        actual_utility=actual_u, actual_catastrophe=actual_cat,
        predicted_utility_seeds=prediction, predicted_catastrophe_seeds=cat_prediction,
    )
    assert set(utility) == {"s0", "s1"}
    assert max(utility.values()) < 1e-5
    assert max(difference.values()) == 0
    assert max(absolute.values()) == 0


def test_scoped_quantile_payload_has_global_and_mechanism_keys():
    payload = MODULE.quantile_payload({f"s{index}": float(index) for index in range(12)})
    assert payload["__global__"] == payload[MODULE.MECHANISM_ID] == 11.0
