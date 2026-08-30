from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/train_baselines.py"
SPEC = importlib.util.spec_from_file_location("train_baselines", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_logistic_fit_returns_finite_probabilities():
    x = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1], dtype=float)
    coefficients = MODULE.fit_logistic(x, y, np.ones(4), iterations=1000)
    probability = MODULE.logistic_probability(coefficients, x)
    assert np.all(np.isfinite(probability))
    assert probability[-1] > probability[0]


def test_choices_group_by_block_and_pick_max_prediction():
    choices = MODULE.choices_from_predictions(
        np.array(["block:a", "block:b", "other:a", "other:b"]),
        np.array(["a", "b", "a", "b"]),
        np.array([0.1, 0.2, 0.4, 0.3]),
    )
    assert choices == {"block": "b", "other": "a"}
