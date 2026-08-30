from __future__ import annotations

import numpy as np
import pytest

from crashbench.models.advantage import (
    RidgeOptionValue,
    best_fixed_option,
    source_macro_policy_value,
)


def test_option_specific_and_shared_ridge_fit_variable_options():
    x = np.tile(np.array([[0.0], [1.0], [2.0]]), (3, 1))
    options = np.repeat(["base", "refresh", "safe"], 3)
    y = np.concatenate([x[:3, 0], x[:3, 0] + 1, -x[:3, 0]])
    weights = np.ones(9)
    for shared in (False, True):
        model = RidgeOptionValue(("base", "refresh", "safe"), ridge=1e-6, shared=shared)
        model.fit(x, options, y, weights)
        prediction = model.predict(x, options)
        assert prediction.shape == (9,)
        assert np.all(np.isfinite(prediction))


def test_best_fixed_averages_within_source_before_across_sources():
    source = ["a", "a", "a", "b", "b"]
    option = ["base", "base", "refresh", "base", "refresh"]
    utility = [0, 0, 1, 0, 1]
    winner, values = best_fixed_option(source, option, utility)
    assert winner == "refresh"
    assert values["refresh"] == 1


def test_source_macro_policy_value_requires_complete_choices():
    rows = [
        {"physical_source_id": "s1", "block_id": "b1", "option_id": "base", "u0": 1},
        {"physical_source_id": "s1", "block_id": "b1", "option_id": "safe", "u0": 0},
        {"physical_source_id": "s2", "block_id": "b2", "option_id": "base", "u0": -1},
        {"physical_source_id": "s2", "block_id": "b2", "option_id": "safe", "u0": 0},
    ]
    assert source_macro_policy_value(rows, {"b1": "base", "b2": "safe"}) == 0.5
    with pytest.raises(ValueError, match="cover"):
        source_macro_policy_value(rows, {"b1": "base"})
