from __future__ import annotations

import numpy as np
import pytest

from crashbench.data.statistics import (
    clopper_pearson_one_sided,
    paired_cluster_bootstrap,
    paired_sign_flip_pvalue,
    shared_cluster_bootstrap_indices,
    source_macro_paired_effect,
    source_simultaneous_coverage,
)


def test_source_macro_averages_repeats_before_pairing():
    rows = [
        {"physical_source_id": "s1", "method_id": "m", "value": 1.0},
        {"physical_source_id": "s1", "method_id": "m", "value": 3.0},
        {"physical_source_id": "s1", "method_id": "c", "value": 1.0},
        {"physical_source_id": "s2", "method_id": "m", "value": 0.0},
        {"physical_source_id": "s2", "method_id": "c", "value": 1.0},
    ]
    effect = source_macro_paired_effect(rows, method_id="m", comparator_id="c")
    np.testing.assert_array_equal(effect.differences, [1.0, -1.0])
    assert effect.point_estimate == 0.0
    assert len(effect.source_ids) == 2


def test_paired_effect_rejects_unpaired_source():
    with pytest.raises(ValueError, match="missing"):
        source_macro_paired_effect(
            [
                {"physical_source_id": "s1", "method_id": "m", "value": 1},
                {"physical_source_id": "s1", "method_id": "c", "value": 0},
                {"physical_source_id": "s2", "method_id": "m", "value": 1},
            ],
            method_id="m",
            comparator_id="c",
        )


def test_shared_bootstrap_indices_apply_to_multiple_metrics():
    rows = [
        {"physical_source_id": f"s{i}", "method_id": method, "value": value}
        for i, pair in enumerate(((1, 0), (2, 0), (3, 0)))
        for method, value in zip(("m", "c"), pair)
    ]
    effect = source_macro_paired_effect(rows, method_id="m", comparator_id="c")
    indices = shared_cluster_bootstrap_indices(3, draws=50, seed=7)
    first = paired_cluster_bootstrap(effect, indices=indices)
    second = paired_cluster_bootstrap(effect, indices=indices)
    assert first == second
    assert first["n_physical_sources"] == 3


def test_sign_flip_exact_mode_does_not_treat_repeats_as_sources():
    result = paired_sign_flip_pvalue([1, 1, 1], alternative="greater")
    assert result["mode"] == "exact"
    assert result["draws"] == 8
    assert result["pvalue"] == pytest.approx(1 / 8)


def test_clopper_pearson_boundaries_and_known_all_success_lower():
    assert clopper_pearson_one_sided(0, 10)[0] == 0
    lower, upper = clopper_pearson_one_sided(10, 10)
    assert lower == pytest.approx(0.05 ** (1 / 10), rel=1e-8)
    assert upper == 1


def test_source_simultaneous_coverage_requires_all_rows_within_source():
    result = source_simultaneous_coverage(
        {"s1": [True, True], "s2": [True, False], "s3": [True]}
    )
    assert result["covered_sources"] == 2
    assert result["n_physical_sources"] == 3
    assert result["point_estimate"] == pytest.approx(2 / 3)
