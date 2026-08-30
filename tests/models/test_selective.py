from __future__ import annotations

import numpy as np
import pytest

from crashbench.models.selective import (
    PairwiseSourceConformalSelector,
    SourceConformalSelector,
    ensemble_mean_std,
    source_conformal_quantile,
)


OPTIONS = ("base_continue", "observation_refresh", "safe_stop")


def calibrated_selector():
    sources, options, predicted, actual, roles = [], [], [], [], []
    for source in ("s1", "s2", "s3", "s4"):
        for option in OPTIONS:
            sources.append(source)
            options.append(option)
            predicted.append(0.0)
            actual.append(0.1 if source not in {"s3", "s4"} else 0.2)
            roles.append("calibration")
    return SourceConformalSelector(OPTIONS, alpha=0.2).fit(
        source_ids=sources,
        option_ids=options,
        predicted_utility=predicted,
        actual_utility=actual,
        roles=roles,
    )


def test_conformal_uses_finite_sample_higher_quantile():
    assert np.isinf(source_conformal_quantile([0.1, 0.2, 0.3], alpha=0.2))
    assert source_conformal_quantile([0.1, 0.2, 0.3, 0.4], alpha=0.2) == 0.4


def test_calibration_rejects_noncalibration_roles():
    selector = SourceConformalSelector(OPTIONS)
    with pytest.raises(ValueError, match="calibration rows only"):
        selector.fit(
            source_ids=["s"] * 3,
            option_ids=OPTIONS,
            predicted_utility=[0] * 3,
            actual_utility=[0] * 3,
            roles=["test"] * 3,
        )


def test_selector_abstains_without_positive_lcb():
    selector = calibrated_selector()
    result = selector.select(
        predicted_utility={option: 0.0 for option in OPTIONS},
        catastrophe_probability={option: 0.1 for option in OPTIONS},
    )
    assert result.option_id == "base_continue"
    assert result.abstained_to_base


def test_selector_intervenes_only_under_catastrophe_constraint():
    selector = calibrated_selector()
    utility = {"base_continue": 0.0, "observation_refresh": 1.0, "safe_stop": 0.2}
    catastrophe = {"base_continue": 0.2, "observation_refresh": 0.21, "safe_stop": 0.1}
    result = selector.select(
        predicted_utility=utility,
        catastrophe_probability=catastrophe,
    )
    assert result.option_id == "observation_refresh"
    catastrophe["observation_refresh"] = 0.5
    result = selector.select(
        predicted_utility=utility,
        catastrophe_probability=catastrophe,
    )
    assert result.option_id != "observation_refresh"


def test_ensemble_mean_std():
    mean, std = ensemble_mean_std([np.array([1, 2]), np.array([3, 4])])
    np.testing.assert_array_equal(mean, [2, 3])
    assert np.all(std > 0)


def test_pairwise_selector_requires_strict_win_over_every_option_and_safety():
    selector = PairwiseSourceConformalSelector(
        OPTIONS,
        utility_quantiles={"__global__": 1.0, "staleness": 1.0},
        catastrophe_difference_quantiles={"__global__": 0.01, "staleness": 0.01},
        catastrophe_absolute_quantiles={"__global__": 0.01, "staleness": 0.01},
    )
    scale = {(a, b): 0.05 for a in OPTIONS for b in OPTIONS if a != b}
    result = selector.select(
        mechanism_id="staleness",
        predicted_utility={"base_continue": 0.0, "observation_refresh": 0.4, "safe_stop": 0.1},
        pairwise_scale=scale,
        predicted_catastrophe={"base_continue": 0.08, "observation_refresh": 0.07, "safe_stop": 0.01},
    )
    assert result.option_id == "observation_refresh"
    assert result.action == "INTERVENE"


def test_pairwise_selector_fails_closed_for_missing_mechanism_quantile():
    selector = PairwiseSourceConformalSelector(
        OPTIONS,
        utility_quantiles={"__global__": 0.0},
        catastrophe_difference_quantiles={"__global__": 0.0},
        catastrophe_absolute_quantiles={"__global__": 0.0},
    )
    scale = {(a, b): 0.05 for a in OPTIONS for b in OPTIONS if a != b}
    result = selector.select(
        mechanism_id="unseen",
        predicted_utility={"base_continue": 0.0, "observation_refresh": 10.0, "safe_stop": 1.0},
        pairwise_scale=scale,
        predicted_catastrophe={option: 0.0 for option in OPTIONS},
    )
    assert result.option_id == "safe_stop"
    assert result.action == "ABSTAIN_TO_SAFE_STOP"
