import numpy as np
import pytest

from crashbench.counterfactual_router import (
    OPTIONS,
    classify_option_outcome,
    temporal_window,
    validate_decision_rows,
)


def test_option_outcomes_are_exhaustive_and_exclusive():
    assert classify_option_outcome(crashed=False, succeeded=True) == "task_success"
    assert classify_option_outcome(crashed=True, succeeded=False) == "catastrophe"
    assert classify_option_outcome(crashed=False, succeeded=False) == "safe_noncompletion"
    with pytest.raises(ValueError, match="both crash and succeed"):
        classify_option_outcome(crashed=True, succeeded=True)


def test_temporal_window_is_causal_and_left_padded():
    values = [np.asarray([index, index + 10], dtype=np.float32) for index in range(4)]
    window, mask = temporal_window(values, anchor_index=1, length=4)
    np.testing.assert_array_equal(mask, [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(window[:2], np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(window[2:], values[:2])

    full, full_mask = temporal_window(values, anchor_index=3, length=3)
    np.testing.assert_array_equal(full_mask, np.ones(3, dtype=np.float32))
    np.testing.assert_array_equal(full, values[1:4])


def _rows(source="a", split="train"):
    return [{
        "source_state_sha256": source,
        "placement_id": "p0",
        "horizon_actions": 20,
        "condition": "glass",
        "option": option,
        "outcome": (
            "catastrophe" if option == "base_continue"
            else "task_success" if option == "detour_complete"
            else "safe_noncompletion"
        ),
        "split": split,
    } for option in OPTIONS]


def test_option_table_requires_all_options_and_groups_by_source():
    summary = validate_decision_rows(_rows())
    assert summary == {
        "decision_states": 1,
        "option_rollouts": 3,
        "source_states": 1,
        "source_states_by_split": {"train": 1},
        "outcomes": {
            "task_success": 1,
            "catastrophe": 1,
            "safe_noncompletion": 1,
        },
    }
    with pytest.raises(ValueError, match="missing options"):
        validate_decision_rows(_rows()[:-1])


def test_option_table_rejects_source_split_leakage():
    rows = _rows()
    rows.extend(_rows(source="a", split="development"))
    with pytest.raises(ValueError, match="leaks across"):
        validate_decision_rows(rows)
