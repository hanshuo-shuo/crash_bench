from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from crashbench.envs.libero_adapter import inject_obstacles_xml
from crashbench.unstable_placement import (
    CONDITIONS,
    OPTIONS,
    branch_hashes_identical,
    build_support_patch,
    build_attempted_grid,
    derive_geometry,
    evaluate_base_repeat_audit,
    patch_surface_upper_z,
    strict_best_action,
    validate_screen_config,
)
from scripts.iclr27.run_non_glass_option_ambiguity_screen import (
    DEFAULT_CONFIG,
    create_immutable_result_root,
    freeze_protocol,
)


def _config() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text())


def _sources() -> list[dict]:
    return [
        {
            "source_id": f"source_{index}",
            "source_state_sha256": hashlib.sha256(f"source_{index}".encode()).hexdigest(),
        }
        for index in range(4)
    ]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_source_parameter_horizon_and_row_caps_are_fail_closed():
    config = _config()
    accounting = validate_screen_config(config)
    assert accounting == {
        "maximum_source_states": 4,
        "physical_parameterizations": 3,
        "decision_horizons": 3,
        "conditions": 3,
        "options": 3,
        "ordinary_option_outcome_rows": 324,
    }
    too_many_sources = copy.deepcopy(config)
    too_many_sources["task"]["select_nominal_successes"] = 5
    with pytest.raises(ValueError, match="exactly four"):
        validate_screen_config(too_many_sources)
    too_many_parameters = copy.deepcopy(config)
    too_many_parameters["parameters"].append(copy.deepcopy(config["parameters"][-1]))
    with pytest.raises(ValueError, match="parameters must be exactly"):
        validate_screen_config(too_many_parameters)
    too_many_horizons = copy.deepcopy(config)
    too_many_horizons["horizons_actions_before_release"].append(2)
    with pytest.raises(ValueError, match="exactly three"):
        validate_screen_config(too_many_horizons)


def test_all_three_options_require_the_identical_complete_branch_hash():
    rows = [{"option": option, "branch_start_sha256": "same"} for option in OPTIONS]
    assert branch_hashes_identical(rows)
    rows[-1]["branch_start_sha256"] = "different"
    assert not branch_hashes_identical(rows)
    assert not branch_hashes_identical(rows[:2])


def test_base_duplicate_restoration_failure_fails_closed():
    trace = [{
        "object_xyz": [0.0, 0.0, 1.0], "tilt_deg": 0.0,
        "grasped": True, "task_success": False, "support_contact": False,
        "robot_hazard_contact": False, "released": False,
    }]
    repeat = {
        "branch_start_sha256": "expected", "first_action": [0.0] * 7,
        "outcome": "task_success", "catastrophe_reason": None, "trace": trace,
    }
    passed = evaluate_base_repeat_audit(
        expected_first_action=[0.0] * 7,
        repeat_results=[copy.deepcopy(repeat), copy.deepcopy(repeat)],
        expected_branch_start_sha256="expected",
        first_action_atol=0.0,
        trace_atol=1e-7,
    )
    assert passed["passed"]
    broken = copy.deepcopy(repeat)
    broken["first_action"][0] = 0.01
    failed = evaluate_base_repeat_audit(
        expected_first_action=[0.0] * 7,
        repeat_results=[repeat, broken],
        expected_branch_start_sha256="expected",
        first_action_atol=0.0,
        trace_atol=1e-7,
    )
    assert not failed["passed"]
    assert "first_nominal_action_mismatch" in failed["reason"]


def test_no_learned_score_or_router_artifact_is_an_accepted_input():
    for prohibited in ("router_artifact", "risk_score", "embedding", "learned_model"):
        config = _config()
        config[prohibited] = "forbidden"
        with pytest.raises(ValueError, match="prohibited"):
            validate_screen_config(config)


def test_full_attempt_grid_retains_controls_and_deduplicates_no_hazard_evidence():
    rows = build_attempted_grid(_sources(), _config())
    assert len(rows) == 108
    assert {
        row["condition"] for row in rows
    } == set(CONDITIONS)
    blocks = {}
    for row in rows:
        key = (
            row["source_state_sha256"], row["parameter_id"], row["horizon_actions"],
        )
        blocks.setdefault(key, set()).add(row["condition"])
    assert all(value == set(CONDITIONS) for value in blocks.values())
    no_hazard = [row for row in rows if row["condition"] == "no_hazard"]
    assert len(no_hazard) == 36
    assert len({row["canonical_decision_id"] for row in no_hazard}) == 12
    assert sum(row["canonical_evidence"] for row in no_hazard) == 12
    assert sum(row["deduplicated_execution"] for row in no_hazard) == 24


def test_strict_ties_never_count_as_support():
    assert strict_best_action({option: "task_success" for option in OPTIONS}) is None
    assert strict_best_action({
        "base_continue": "task_success",
        "stable_offset_place": "safe_noncompletion",
        "safe_setdown": "safe_noncompletion",
    }) == "base_continue"


def test_frozen_geometry_leaves_offset_center_clear_until_severe():
    config = _config()
    geometry = derive_geometry({
        "bowl_footprint_radius_m": 0.05,
        "bowl_half_height_m": 0.03,
        "plate_support_radius_m": 0.10,
        "plate_top_z_m": 0.91,
        "table_top_z_m": 0.89,
    }, config)
    stable_xy = [-geometry["stable_offset_distance_m"], 0.0]
    covered = {}
    for parameter in geometry["parameters"]:
        patch = build_support_patch(
            parameter,
            center_xy=[parameter["patch_center_offset_m"], 0.0],
            plate_top_z_m=geometry["plate_top_z_m"],
        )
        covered[parameter["parameter_id"]] = (
            patch_surface_upper_z(patch, stable_xy) != float("-inf")
        )
    assert covered == {"mild": False, "moderate": False, "severe": True}


def test_engineering_mock_is_complete_source_aware_and_never_scientific(tmp_path):
    root = freeze_protocol(
        output_parent=tmp_path,
        config_path=DEFAULT_CONFIG,
        checkpoint_revision="0" * 40,
        mock=True,
    )
    attempts = _jsonl(root / "attempted_configurations.jsonl")
    outcomes = _jsonl(root / "option_outcomes.jsonl")
    decision = json.loads((root / "screen_decision.json").read_text())
    exclusions = json.loads((root / "future_benchmark_exclusions.json").read_text())
    assert len(attempts) == 108
    assert len(outcomes) == 324
    assert decision["decision"] == "SCREEN_BLOCKED_ENVIRONMENT"
    assert decision["engineering_only"] is True
    # Many strict decisions per source still count as exactly four independent sources.
    assert decision["gate_values"]["strict_base_source_count"] == 4
    assert decision["gate_values"]["strict_stable_offset_place_source_count"] == 4
    assert decision["gate_values"]["strict_safe_setdown_source_count"] == 4
    assert decision["gate_values"]["same_r_different_strict_action_source_count"] == 4
    assert decision["gate_values"]["stable_offset_safe_setdown_flip_source_count"] == 4
    assert len(exclusions["excluded"]) == 4
    assert {row["source_state_sha256"] for row in exclusions["excluded"]} == {
        row["source_state_sha256"] for row in json.loads(
            (root / "source_registry.json").read_text()
        )["selected_sources"]
    }
    for required in (
        "screen_manifest.json", "resolved_config.json", "source_registry.json",
        "future_benchmark_exclusions.json", "attempted_configurations.jsonl",
        "mechanical_validity.jsonl", "restoration_audit.jsonl",
        "decision_metadata.jsonl", "option_outcomes.jsonl",
        "strict_support_by_source.csv", "same_r_option_witnesses.jsonl",
        "screen_decision.json", "REPORT.md",
    ):
        assert (root / required).exists(), required


def test_existing_result_root_cannot_be_overwritten(tmp_path):
    create_immutable_result_root(tmp_path, "a" * 64, "20260830T000000Z")
    with pytest.raises(FileExistsError):
        create_immutable_result_root(tmp_path, "a" * 64, "20260830T000000Z")


def test_static_patch_xml_supports_frozen_pose_and_friction():
    xml = "<mujoco><worldbody></worldbody></mujoco>"
    patched = inject_obstacles_xml(xml, [{
        "name": "patch", "type": "box", "pos": [0, 0, 1],
        "size": [0.1, 0.1, 0.002], "rgba": [0.1, 0.2, 0.3, 1.0],
        "euler": [0.0, 0.2, 0.0], "friction": [0.12, 0.005, 0.0001],
    }])
    assert 'euler="0.0 0.2 0.0"' in patched
    assert 'friction="0.12 0.005 0.0001"' in patched
    with pytest.raises(ValueError, match="both quat and euler"):
        inject_obstacles_xml(xml, [{
            "name": "bad", "pos": [0, 0, 1], "size": [0.1, 0.1, 0.1],
            "euler": [0, 0, 0], "quat": [1, 0, 0, 0],
        }])


def test_mock_run_does_not_touch_frozen_artifacts(tmp_path):
    frozen = [
        Path("results/grasp/build.json"),
        Path("results/grasp_instability.json"),
        Path("results/iclr27/risk_value_decoupling_dc48ff317dad_20260829T154351Z/story_decision.json"),
    ]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen}
    freeze_protocol(
        output_parent=tmp_path,
        config_path=DEFAULT_CONFIG,
        checkpoint_revision="0" * 40,
        mock=True,
    )
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen}
    assert before == after
