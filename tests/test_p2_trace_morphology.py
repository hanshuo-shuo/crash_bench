import math

from scripts.analyze_p2_trace_morphology import (
    cumulative_area_above,
    episode_group,
    longest_true_run,
    longest_value_run,
    moving_average_max,
)


def test_moving_average_max_uses_valid_windows():
    values = [0.0, 3.0, 0.0, 6.0]

    assert moving_average_max(values, 3) == 3.0
    assert moving_average_max(values, 5) is None


def test_area_and_consecutive_runs_use_strict_margin():
    values = [0.0, 0.5, 0.5, 1.5, 0.0, 2.0, 2.0]

    assert math.isclose(cumulative_area_above(values, 0.5), 4.0)
    assert longest_true_run(value > 0.5 for value in values) == 2
    assert longest_value_run(
        ["detour_complete", "detour_complete", "retreat_hold", "retreat_hold", "retreat_hold"],
        "retreat_hold",
    ) == 3


def test_episode_grouping_matches_requested_comparisons():
    base = {
        "condition": "glass",
        "missed_recovery_window": False,
        "reference_base_outcome": "catastrophe",
        "outcome": "catastrophe",
        "intervened": False,
    }
    assert episode_group({**base, "missed_recovery_window": True}) == (
        "missed_t20_recoverable_glass"
    )
    assert episode_group({**base, "outcome": "task_success", "intervened": True}) == (
        "successfully_recovered_glass"
    )
    assert episode_group(base) == "other_glass"
    assert episode_group({
        **base,
        "condition": "offpath",
        "reference_base_outcome": "task_success",
        "outcome": "task_success",
    }) == "base_success_offpath_noglass_control"
    assert episode_group({
        **base,
        "condition": "noglass",
        "reference_base_outcome": "safe_noncompletion",
        "outcome": "safe_noncompletion",
    }) == "outside_requested_groups"
