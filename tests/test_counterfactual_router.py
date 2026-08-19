import argparse
import json

import numpy as np
import pytest

from crashbench.counterfactual_router import (
    FirstCrossingRouter,
    OPTIONS,
    classify_option_outcome,
    conservative_option_choice,
    option_utilities,
    outcome_probabilities,
    split_conformal_upper_boundary,
    temporal_window,
    validate_decision_rows,
)
from scripts.analyze_dynamic_first_crossing_router import summarize_dynamic_rows
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
from scripts.train_minimal_counterfactual_router import (
    calibrate_router_margin,
    choose_router,
    fit_weighted_ridge,
    predict_ridge,
)
from scripts.train_counterfactual_outcome_router import (
    calibrate_margin_for_rate,
)
from scripts.analyze_fresh_counterfactual_router import metric_values, oracle_choice
from crashbench.recovery import FailSafeHold
from scripts.p0_counterfactual_router_audit import (
    calibrate_two_threshold,
    candidate_thresholds,
    evaluate_choices as evaluate_p0_choices,
    two_threshold_choice,
)
from scripts.author_timing_choice_benchmark import (
    build_timing_records,
    correct_option,
    point_signature,
)
from scripts.author_recovery_window_supervision import (
    extract_hard_control_records,
    feature_overlap_summary,
    label_anchor,
    recovery_intervals,
)


def test_option_outcomes_are_exhaustive_and_exclusive():
    assert classify_option_outcome(crashed=False, succeeded=True) == "task_success"
    assert classify_option_outcome(crashed=True, succeeded=False) == "catastrophe"
    assert classify_option_outcome(crashed=False, succeeded=False) == "safe_noncompletion"
    with pytest.raises(ValueError, match="both crash and succeed"):
        classify_option_outcome(crashed=True, succeeded=True)


def test_probabilistic_router_exposes_lambda_and_base_margin():
    logits = np.asarray([
        [2.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, -2.0, 2.0],
    ])
    probabilities = outcome_probabilities(logits)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
    utilities = option_utilities(probabilities, catastrophe_cost=2.0)
    assert utilities[1] > utilities[0]
    assert conservative_option_choice(
        probabilities, catastrophe_cost=2.0, intervention_margin=0.0
    ) == 1
    assert conservative_option_choice(
        probabilities, catastrophe_cost=2.0, intervention_margin=10.0
    ) == 0


def test_dynamic_router_latches_the_first_positive_calibrated_crossing():
    gate = FirstCrossingRouter(catastrophe_cost=1.0, intervention_margin=0.2)
    before = np.asarray([
        [0.6, 0.1, 0.3],
        [0.7, 0.1, 0.2],
        [0.2, 0.1, 0.7],
    ])
    crossing = np.asarray([
        [0.3, 0.5, 0.2],
        [0.9, 0.05, 0.05],
        [0.1, 0.1, 0.8],
    ])
    assert not gate.observe(before, action_index=3)["first_crossing"]
    assert gate.observe(crossing, action_index=4)["first_crossing"]
    assert gate.selected_option_index == 1
    assert gate.trigger_action_index == 4
    assert not gate.observe(crossing, action_index=5)["first_crossing"]
    assert gate.selected_option_index == 1


def test_sequential_boundary_uses_source_level_split_conformal_rank():
    source_maxima = list(range(10))
    assert split_conformal_upper_boundary(source_maxima, alpha=0.1) == 9.0
    assert split_conformal_upper_boundary(source_maxima, alpha=0.2) == 8.0


def test_dynamic_summary_reports_timing_choice_and_failure_modes():
    common = {
        "source_state_sha256": "source-a",
        "first_trigger_to_collision_eef_distance_m": 0.04,
        "intervention_duration_actions": 12,
        "contact_force_p95_n": 2.0,
        "contact_force_max_n": 3.0,
        "t20_oracle_timing_upper_bound": {
            "option_outcomes": {"detour_complete": "task_success"},
        },
    }
    rows = [
        {
            **common,
            "condition": "glass",
            "reference_base_outcome": "catastrophe",
            "selected_option": "retreat_hold",
            "intervened": True,
            "first_trigger_lead_time_actions": 4,
            "missed_recovery_window": True,
            "unnecessary_early_intervention": False,
            "outcome": "safe_noncompletion",
        },
        {
            **common,
            "condition": "noglass",
            "reference_base_outcome": "task_success",
            "selected_option": "detour_complete",
            "intervened": True,
            "first_trigger_lead_time_actions": None,
            "first_trigger_to_collision_eef_distance_m": None,
            "missed_recovery_window": False,
            "unnecessary_early_intervention": True,
            "outcome": "task_success",
        },
    ]
    summary = summarize_dynamic_rows(rows)
    assert summary["selected_option_counts"]["retreat_hold"] == 1
    assert summary["missed_recovery_window_rate"] == 0.5
    assert summary["unnecessary_early_intervention_given_base_success"] == 1.0
    assert summary["first_trigger_lead_time_actions"]["median"] == 4.0


def test_probabilistic_margin_is_calibrated_without_development_rows():
    probabilities = np.asarray([
        [[0.4, 0.5, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]],
        [[0.7, 0.1, 0.2], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]],
        [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]],
        [[0.3, 0.6, 0.1], [0.2, 0.2, 0.6], [0.1, 0.1, 0.8]],
    ])
    outcomes = np.asarray([
        ["catastrophe", "task_success", "safe_noncompletion"],
        ["task_success", "task_success", "safe_noncompletion"],
        ["task_success", "safe_noncompletion", "safe_noncompletion"],
        ["catastrophe", "safe_noncompletion", "safe_noncompletion"],
    ])
    result = calibrate_margin_for_rate(
        probabilities, outcomes, np.asarray(["a", "b", "c", "d"]),
        catastrophe_cost=5.0, target_rate=0.5,
    )
    choice = conservative_option_choice(
        probabilities, catastrophe_cost=5.0,
        intervention_margin=result["delta"],
    )
    assert np.mean(choice != 0) == result["calibration_intervention_rate"]
    assert abs(result["calibration_intervention_rate"] - 0.5) <= 0.25


def test_fresh_metrics_use_realized_base_and_oracle_counterfactuals():
    outcomes = np.asarray([
        ["catastrophe", "task_success", "safe_noncompletion"],
        ["task_success", "safe_noncompletion", "safe_noncompletion"],
    ])
    choice = np.asarray([1, 0])
    selected = outcomes[np.arange(2), choice]
    metrics = metric_values(
        selected, choice != 0, outcomes, catastrophe_cost=5.0,
    )
    assert oracle_choice(outcomes, catastrophe_cost=5.0).tolist() == [1, 0]
    assert metrics["task_success_rate"] == 1.0
    assert metrics["unnecessary_intervention_rate"] == 0.0
    assert metrics["missed_beneficial_intervention_rate"] == 0.0
    assert metrics["mean_oracle_regret"] == 0.0
    assert metrics["oracle_value_recovered"] == 1.0


def test_p0_two_threshold_risk_can_route_detour_and_retreat():
    scores = np.asarray([0.1, 0.4, 0.8])
    outcomes = np.asarray([
        ["task_success", "safe_noncompletion", "safe_noncompletion"],
        ["catastrophe", "task_success", "safe_noncompletion"],
        ["catastrophe", "catastrophe", "safe_noncompletion"],
    ])
    sources = np.asarray(["a", "b", "c"])
    calibration = calibrate_two_threshold(
        scores, outcomes, sources, catastrophe_cost=5.0,
    )
    choice = two_threshold_choice(scores, calibration)
    assert choice.tolist() == [0, 1, 2]
    metrics = evaluate_p0_choices(choice, outcomes, sources, catastrophe_cost=5.0)
    assert metrics["catastrophe_rate"] == 0.0
    assert metrics["task_success_rate"] == pytest.approx(2 / 3)


def test_p0_threshold_candidates_include_all_and_none_intervention():
    thresholds = candidate_thresholds(np.asarray([0.2, 0.8]))
    assert thresholds == [float("-inf"), 0.5, float("inf")]


def test_p1_point_signatures_encode_option_choice_not_hazard_identity():
    assert point_signature([
        "catastrophe", "task_success", "safe_noncompletion",
    ]) == "recoverable"
    assert point_signature([
        "catastrophe", "catastrophe", "safe_noncompletion",
    ]) == "loss_control"
    assert point_signature([
        "task_success", "catastrophe", "safe_noncompletion",
    ]) == "unnecessary"
    assert correct_option("recoverable") == "detour_complete"
    assert correct_option("loss_control") == "retreat_hold"
    assert correct_option("unnecessary") == "base_continue"


def test_p1_timing_pair_requires_earlier_recovery_and_later_loss_control():
    horizons = [30, 20, 10, 5]
    metadata = [{
        "decision_id": f"d{horizon}",
        "source_state_sha256": "source-a",
        "placement_id": "placement-a",
        "placement_key": "fresh:placement-a",
        "split": "development",
        "condition": "glass",
        "horizon_actions": horizon,
    } for horizon in horizons]
    outcomes = np.asarray([
        ["catastrophe", "task_success", "safe_noncompletion"],
        ["catastrophe", "task_success", "safe_noncompletion"],
        ["catastrophe", "catastrophe", "safe_noncompletion"],
        ["catastrophe", "catastrophe", "safe_noncompletion"],
    ])
    built = build_timing_records(metadata, outcomes)
    records = built["records"]
    assert [row["benchmark_class"] for row in records] == [
        "early_recoverable", "late_loss_control",
    ]
    assert records[0]["horizon_actions"] == 30
    assert records[1]["horizon_actions"] == 5
    assert records[0]["timing_pair_id"] == records[1]["timing_pair_id"]


def test_p1_fail_safe_hold_has_no_directional_motion():
    controller = FailSafeHold()
    controller.engage({"robot0_eef_pos": np.asarray([0.1, 0.2, 0.3])})
    np.testing.assert_array_equal(
        controller.step({"robot0_eef_pos": np.asarray([0.1, 0.2, 0.3])}),
        np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
    )


def test_p3_anchor_labels_follow_predeclared_option_priority():
    recoverable = label_anchor(
        "catastrophe", "task_success", "safe_noncompletion"
    )
    assert recoverable["recovery_open"] is True
    assert recoverable["preferred_option"] == "Detour"
    assert recoverable["intervention_needed"] is True

    loss = label_anchor(
        "catastrophe", "catastrophe", "safe_noncompletion"
    )
    assert loss["loss_control"] is True
    assert loss["preferred_option"] == "FailSafeHold"
    assert loss["task_completion_unsolved"] is True

    base_tie = label_anchor(
        "task_success", "task_success", "safe_noncompletion"
    )
    assert base_tie["preferred_option"] == "Base"
    assert base_tie["intervention_needed"] is False

    violation = label_anchor("catastrophe", "catastrophe", "catastrophe")
    assert violation["fail_safe_contract_violation"] is True
    assert violation["preferred_option"] == "Base"


def test_p3_recovery_intervals_preserve_nonmonotonic_windows():
    horizons = [10, 8, 6, 4, 2, 1]
    records = [
        {"horizon_actions": horizon, "recovery_open": horizon in {10, 8, 4, 2}}
        for horizon in horizons
    ]
    intervals = recovery_intervals(records, horizon_grid=horizons)
    assert [row["horizons"] for row in intervals] == [[10, 8], [4, 2]]
    assert intervals[0]["latest_recoverable_anchor"] == 8
    assert intervals[1]["recovery_window_span_actions"] == 2


def test_p3_hard_controls_take_union_of_peak_and_long_regions():
    episodes = [
        {
            "episode_id": "short",
            "placement_id": "p-short",
            "source_state_sha256": "s-short",
            "condition": "noglass",
            "reference_base_outcome": "task_success",
        },
        {
            "episode_id": "long",
            "placement_id": "p-long",
            "source_state_sha256": "s-long",
            "condition": "offpath",
            "reference_base_outcome": "task_success",
        },
        {
            "episode_id": "crash",
            "placement_id": "p-crash",
            "source_state_sha256": "s-crash",
            "condition": "glass",
            "reference_base_outcome": "catastrophe",
        },
    ]

    def trace(episode, values):
        rows = []
        for action, value in enumerate(values):
            probabilities = np.full((3, 3), 1 / 3)
            rows.append({
                "episode_id": episode,
                "action_index": action,
                "advantage_vs_base": [0.0, value, value - 0.1],
                "option_outcome_probabilities": probabilities.tolist(),
                "base_catastrophe_probability": 0.5,
                "utility": [0.0, value, value - 0.1],
            })
        return rows

    traces = (
        trace("short", [0.0, 2.0, 0.0])
        + trace("long", [0.6, 0.7, 0.8, 0.9])
        + trace("crash", [9.0, 9.0])
    )
    records, regions, vectors = extract_hard_control_records(
        episodes, traces, pointwise_margin=0.5,
        top_by_peak=1, top_by_duration=1,
    )
    assert {row["placement_id"] for row in regions} == {"p-short", "p-long"}
    assert len(records) == 5
    assert all(row["hard_negative"] for row in records)
    assert all(row["preferred_option"] == "Base" for row in records)
    assert len(vectors) == len(records)


def test_p3_feature_overlap_reports_router_output_space():
    result = feature_overlap_summary(
        [np.zeros(3), np.ones(3)],
        [np.full(3, 0.1), np.full(3, 0.9)],
    )
    assert result["available"] is True
    assert result["dimensions"] == 3
    assert 0.0 <= result["leave_one_out_1nn_balanced_accuracy"] <= 1.0


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


def test_source_balanced_ridge_and_router_calibration():
    x = np.asarray([[0.0], [0.1], [1.0], [1.1]])
    sources = np.asarray(["a", "a", "b", "b"])
    target = np.asarray([
        [1.0, 0.0, -5.0], [1.0, 0.0, -5.0],
        [-5.0, 1.0, 0.0], [-5.0, 1.0, 0.0],
    ])
    model = fit_weighted_ridge(x, target, sources, alpha=0.01)
    predicted = predict_ridge(model, x)
    assert predicted.shape == target.shape
    assert choose_router(predicted, margin=0.0).tolist() == [0, 0, 1, 1]
    margin, summary = calibrate_router_margin(predicted, target, sources)
    assert margin >= 0.0
    assert summary["source_macro_utility"] == 1.0
