import argparse
import json

import numpy as np
import pytest

from crashbench.counterfactual_router import (
    OPTIONS,
    classify_option_outcome,
    temporal_window,
    validate_decision_rows,
)
from scripts.sweep_counterfactual_detour import build_configs, summarize_sweep
from scripts.collect_counterfactual_option_rollouts import _apply_frozen_detour_config
from scripts.collect_counterfactual_option_rollouts import _run_structured_option
from scripts.collect_counterfactual_option_rollouts import _branch_start_catastrophic
from scripts.audit_counterfactual_smoke import summarize_smoke
from scripts.freeze_partial_detour_sweep import select_stable_success
from scripts.analyze_counterfactual_option_capture import (
    group_decisions,
    summarize_decisions,
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


def test_detour_sweep_recommends_one_common_stable_config():
    configs = build_configs(
        sides=[-1, 1], lane_margins=[0.12], lift_offsets=[0.30],
        descend_offsets=[0.018], grasp_xy_offsets=[[0.009, -0.04]],
        departure_clearance=0.06,
    )
    rows = []
    for config in configs:
        for state in ("a", "b"):
            outcome = (
                "task_success"
                if config["side"] == 1 and state == "a"
                else "safe_noncompletion"
            )
            for replicate in range(2):
                rows.append({
                    "config_id": config["config_id"], "state_id": state,
                    "replicate": replicate, "outcome": outcome,
                })
    summary = summarize_sweep(rows, configs)
    assert summary["go"] is True
    assert summary["recommended_config"]["config"]["side"] == 1.0
    assert summary["recommended_config"]["task_success_states"] == 1


def test_detour_sweep_rejects_replicate_instability():
    configs = build_configs(
        sides=[1], lane_margins=[0.12], lift_offsets=[0.30],
        descend_offsets=[0.018], grasp_xy_offsets=[[0.009, -0.04]],
        departure_clearance=0.06,
    )
    rows = [
        {"config_id": configs[0]["config_id"], "state_id": "a", "replicate": 0,
         "outcome": "task_success"},
        {"config_id": configs[0]["config_id"], "state_id": "a", "replicate": 1,
         "outcome": "safe_noncompletion"},
    ]
    summary = summarize_sweep(rows, configs)
    assert summary["go"] is False


def test_frozen_detour_config_is_applied_without_partial_defaults(tmp_path):
    config = {
        "side": -1.0, "lane_margin": 0.18, "lift_offset": 0.38,
        "descend_offset": 0.04, "grasp_xy_offset": [-0.003, -0.05],
        "departure_clearance": 0.06, "orientation_target": None,
        "path_aligned": True,
    }
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(config))
    args = argparse.Namespace(detour_config=str(path))
    assert _apply_frozen_detour_config(args) == config
    assert args.detour_side == -1.0
    assert args.detour_grasp_xy_offset == [-0.003, -0.05]


def _smoke_rows(patterns):
    rows = []
    for index, (horizon, outcomes) in enumerate(patterns):
        for option, outcome in zip(
            ("base_continue", "detour_complete", "retreat_hold"), outcomes
        ):
            rows.append({
                "decision_id": f"d{index}", "placement_key": "fresh:p0",
                "source_state_sha256": "source", "condition": "glass",
                "horizon_actions": horizon, "option": option, "outcome": outcome,
            })
    return rows


def test_smoke_audit_requires_informative_multi_h_option_ordering():
    summary = summarize_smoke(_smoke_rows([
        (40, ("task_success", "task_success", "safe_noncompletion")),
        (20, ("catastrophe", "task_success", "safe_noncompletion")),
        (5, ("catastrophe", "catastrophe", "safe_noncompletion")),
    ]))
    assert len(summary["option_ordering_patterns"]) == 3
    assert all(summary["diagnostics"].values())


def test_smoke_audit_blocks_fixed_recovery_degeneracy():
    repeated = ("catastrophe", "task_success", "safe_noncompletion")
    summary = summarize_smoke(_smoke_rows([(40, repeated), (20, repeated), (5, repeated)]))
    assert summary["diagnostics"]["at_least_two_option_ordering_patterns"] is False
    assert summary["detour_advantage_states"] == 3


def test_partial_sweep_freezes_lowest_force_replicated_success():
    configs = build_configs(
        sides=[-1, 1], lane_margins=[0.12], lift_offsets=[0.30],
        descend_offsets=[0.018], grasp_xy_offsets=[[0.009, -0.04]],
        departure_clearance=0.06,
    )
    rows = []
    for config, force in zip(configs, [4.0, 1.0]):
        for replicate in range(2):
            rows.append({
                "config_id": config["config_id"], "state_id": "a",
                "replicate": replicate, "outcome": "task_success",
                "peak_force_n": force, "steps": 100,
            })
    winner = select_stable_success(rows, configs, min_replicates=2)
    assert winner["config"]["side"] == 1.0
    assert winner["mean_peak_force_n"] == 1.0


def test_structured_option_labels_internal_horizon_as_noncompletion():
    class FakeEnv:
        def __init__(self):
            self.terminated = False

        def episode_terminated(self):
            return self.terminated

        def step(self, action):
            self.terminated = True
            return {}, 0.0, False, {}

    class FakeController:
        i = 3

        def engage(self, obs):
            pass

        def step(self, obs):
            return np.zeros(7, dtype=np.float32)

    result = _run_structured_option(
        FakeEnv(), {}, [], FakeController(), max_steps=10
    )
    assert result == {
        "crashed": False, "succeeded": False, "steps": 1,
        "peak_force_n": 0.0, "controller_final_stage": 3,
        "termination": "robosuite_episode_horizon",
    }


def test_branch_start_catastrophe_is_an_exclusion(monkeypatch):
    class FakeEnv:
        sim_view = object()

    monkeypatch.setattr(
        "scripts.collect_counterfactual_option_rollouts._restore_anchor",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "scripts.collect_counterfactual_option_rollouts.build_any",
        lambda specs: object(),
    )

    def reject(crash, sim):
        from scripts.collect_glass_recovery_pairs import CandidateRejected
        raise CandidateRejected("invalid_initial_state", "already catastrophic")

    monkeypatch.setattr(
        "scripts.collect_counterfactual_option_rollouts._prime_glass_predicates",
        reject,
    )
    assert _branch_start_catastrophic(
        FakeEnv(), {"glasses": [{"name": "g", "pos": [0, 0, 0], "size": [1, 1]}]},
        0, {}, label="test",
    ) is True


def test_counterfactual_capture_audit_uses_base_tie_break():
    rows = []
    patterns = [
        ("task_success", "task_success", "safe_noncompletion"),
        ("catastrophe", "task_success", "safe_noncompletion"),
        ("catastrophe", "safe_noncompletion", "safe_noncompletion"),
    ]
    for index, outcomes in enumerate(patterns):
        for option, outcome in zip(
            ("base_continue", "detour_complete", "retreat_hold"), outcomes
        ):
            rows.append({
                "decision_id": f"d{index}", "source_state_sha256": f"s{index}",
                "placement_key": f"p{index}", "split": "train", "condition": "glass",
                "horizon_actions": 20, "feature_index": index,
                "option": option, "outcome": outcome,
            })
    summary = summarize_decisions(group_decisions(rows))
    assert summary["counterfactual_oracle"]["choice_counts_tie_break_base"] == {
        "base_continue": 1, "detour_complete": 2, "retreat_hold": 0,
    }
    assert summary["positive_oracle_value_over_base"] == 2
    assert summary["all_options_identical"] == 0
