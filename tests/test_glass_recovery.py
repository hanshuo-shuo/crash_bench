from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import scripts.collect_glass_recovery_pairs as glass_collector

from crashbench.envs.libero_adapter import LiberoEnv, inject_obstacles_xml
from crashbench.glass_recovery_data import (
    HAZARD_TYPES,
    PRIMARY_TRAJECTORY_KINDS,
    RISK_HORIZONS,
    SCHEMA_VERSION,
    GlassPlacement,
    PairedTrajectoryRecord,
    array_sha256,
    canonical_sha256,
    exact_h_anchor_index,
    read_auxiliary_trajectory_manifest,
    read_trajectory_manifest,
    steps_until_event,
    validate_episode_arrays,
    validate_auxiliary_records,
    validate_paired_records,
    validate_placement_design,
    validate_primary_pair,
    write_trajectory_manifest,
)
from crashbench.glass_recovery_model import (
    HIDDEN_HOOK_IDENTITY,
    PRIMARY_CHECKPOINT_KIND,
    PRIMARY_DISABLED_AUXILIARY_HEADS,
    PRIMARY_DISABLED_AUXILIARY_TERMS,
    TTE_DEFINITION,
    GlassLossWeights,
    GlassRecoveryConfig,
    GlassRecoveryNetwork,
    glass_recovery_loss,
    state_dict_sha256,
)
from scripts.train_glass_recovery import (
    PHASES,
    PairPhaseBatchSampler,
    PairedFrameDataset,
    _episode_qualified_scores,
    _episode_max_scores,
    _select_timely_risk_threshold,
)
from crashbench.metrics import summarize_recovery_rows
from crashbench.policies.glass_recovery_policy import GlassRecoveryPolicy
from crashbench.recovery import DetourComplete
from crashbench.predicates import build_predicate, prime_predicate
from crashbench.scenario import PredicateSpec
from scripts.collect_glass_recovery_pairs import (
    ATTEMPT_LEDGER_NAME,
    CAREFUL_PROMPT_PREFIX,
    _align_oracle_row_zero_labels,
    _append_attempt_event,
    _attempt_identity,
    _attempt_ledger_summary,
    _attempt_pair_root,
    _branch_start_hashes,
    _finalize_arrays,
    _observation_sha256,
    _oracle_rejection_reason,
    _oracle_configs,
    _ordered_placements,
    _partition_scene,
    _primary_protocol,
    _read_attempt_ledger,
    _replay_nominal_actions,
    _rejection_counts,
    _roll_nominal,
    _safe_abort_configs,
    _search_oracle,
    _stable_abort,
    _terminal_attempts,
    _write_manifests,
)
from scripts.prepare_glass_recovery_placements import (
    _blocked_barrier_offsets,
    _collision_free_path_fraction,
    _file_sha256,
    _load_source_traces,
    _minimum_initial_body_clearance,
    _sample_trace_anchor,
    _state_layout,
    _state_layout_from_indices,
)
from scripts.audit_glass_core_artifacts import run_audit
from scripts.run_glass_avoidability_frontier import (
    FRONTIER_HORIZONS,
    summarize_frontier_rows,
)
from scripts.replay_glass_recovery_pair import accepted_pair_dir_from_manifest


def _glass(name="glass_1", x=0.0, y=0.0):
    return {"name": name, "type": "cylinder", "pos": [x, y, 0.96], "size": [0.03, 0.06]}


def _placement(identifier: str, split: str, state_hash: str, cluster: str) -> GlassPlacement:
    return GlassPlacement(
        placement_id=identifier,
        split=split,
        cluster_id=cluster,
        task_suite="libero_spatial",
        task_id=0,
        instruction="pick and place",
        source_state_path=f"states/{identifier}.npy",
        source_state_sha256=state_hash,
        on_path_glass=_glass(),
        off_path_glass=_glass(x=0.2),
        blocked_glasses=[_glass("glass_block_0"), _glass("glass_block_1", y=0.1)],
        nominal_fraction=0.6,
    )


def _trajectory_rows(n: int) -> list[dict]:
    return [
        {
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "hidden": np.zeros(4, dtype=np.float32),
            "robot_state": np.zeros(8, dtype=np.float32),
            "nominal_action": np.zeros(7, dtype=np.float32),
            "target_action": np.zeros(7, dtype=np.float32),
            "executed_action": np.zeros(7, dtype=np.float32),
            "force_after": 0.0,
            "counterfactual_peak_force": 30.0,
        }
        for _ in range(n)
    ]


def _v2_records(
    *,
    pair_id: str = "pair_v2",
    split: str = "train",
    horizon: int = 3,
    include_blocked: bool = False,
    careful_crashed: bool | None = None,
) -> list[PairedTrajectoryRecord]:
    common = dict(
        pair_id=pair_id,
        placement_id=f"{pair_id}_placement",
        split=split,
        source_state_sha256=f"{pair_id}_source",
        matched_robot_state_sha256=f"{pair_id}_robot",
        instruction="pick and place",
        task_suite="libero_spatial",
        task_id=0,
        trigger_horizon_actions=horizon,
        schema_version=SCHEMA_VERSION,
    )
    controller = {"controller_state_sha256": f"{pair_id}_controller"}
    attempt = {"attempt_key": f"{pair_id}_attempt"}
    onpath_hashes = {
        "simulator_state_sha256": f"{pair_id}_onpath",
        "controller_state_sha256": f"{pair_id}_controller",
        "observation_sha256": f"{pair_id}_onpath_observation",
    }
    offpath_hashes = {
        "simulator_state_sha256": f"{pair_id}_offpath",
        "controller_state_sha256": f"{pair_id}_controller",
        "observation_sha256": f"{pair_id}_offpath_observation",
    }
    row_zero_label_hash = f"{pair_id}_row_zero_labels"
    oracle_config = {"side": -1.0, "lane_margin": 0.12}
    careful = (
        {} if careful_crashed is None
        else {"careful_comparator": {"careful_crashed": careful_crashed}}
    )
    records = [
        PairedTrajectoryRecord(
            **common,
            trajectory_kind="nominal_catastrophe",
            branch_start_state_sha256=f"{pair_id}_onpath",
            arrays_path=f"{split}/{pair_id}/nominal.npz",
            n_steps=horizon,
            outcome="crash",
            crashed=True,
            succeeded=False,
            safe_abort=False,
            oracle_verified=False,
            scene_sha256=f"{pair_id}_onpath_scene",
            metadata={
                **controller,
                **attempt,
                **careful,
                "branch_start_hashes": onpath_hashes,
                "row_zero_label_sha256": row_zero_label_hash,
                "time_to_catastrophe_actions": horizon,
                "source_scan_precrash_index": 0,
                "source_scan_collision_step": horizon - 1,
                "source_scan_anchor_state_sha256": f"{pair_id}_onpath",
                "source_scan_anchor_controller_state_sha256": f"{pair_id}_controller",
                "action_replay_evidence": {
                    "verified": True,
                    "n_actions": horizon,
                    "expected_catastrophe_action_index": horizon - 1,
                    "actual_catastrophe_action_index": horizon - 1,
                },
            },
        ),
        PairedTrajectoryRecord(
            **common,
            trajectory_kind="oracle_recovery",
            branch_start_state_sha256=f"{pair_id}_onpath",
            arrays_path=f"{split}/{pair_id}/oracle.npz",
            n_steps=2,
            outcome="recovery_success",
            crashed=False,
            succeeded=True,
            safe_abort=False,
            oracle_verified=True,
            scene_sha256=f"{pair_id}_onpath_scene",
            metadata={
                **controller,
                **attempt,
                "branch_start_hashes": onpath_hashes,
                "row_zero_label_sha256": row_zero_label_hash,
                "time_to_catastrophe_actions": horizon,
                "oracle_recoverable_from_this_state": True,
                "oracle_verified_mask": True,
                "latest_verified_recoverable_state": f"{pair_id}_onpath",
                "runtime_trigger_eligible": True,
                "oracle_config": oracle_config,
                "oracle_verification": {
                    "search_success": True,
                    "search_successful_config_sha256": canonical_sha256(oracle_config),
                    "independent_recapture": True,
                    "recapture_success": True,
                },
            },
        ),
        PairedTrajectoryRecord(
            **common,
            trajectory_kind="off_path_control",
            branch_start_state_sha256=f"{pair_id}_offpath",
            arrays_path=f"{split}/{pair_id}/control.npz",
            n_steps=2,
            outcome="task_success",
            crashed=False,
            succeeded=True,
            safe_abort=False,
            oracle_verified=False,
            scene_sha256=f"{pair_id}_offpath_scene",
            metadata={
                **controller,
                **attempt,
                "branch_start_hashes": offpath_hashes,
                "termination": "task_success",
            },
        ),
    ]
    if include_blocked:
        records.append(PairedTrajectoryRecord(
            **common,
            trajectory_kind="blocked_safe_abort",
            branch_start_state_sha256=f"{pair_id}_blocked",
            arrays_path=f"{split}/{pair_id}/blocked.npz",
            n_steps=2,
            outcome="safe_abort",
            crashed=False,
            succeeded=False,
            safe_abort=True,
            oracle_verified=True,
            scene_sha256=f"{pair_id}_blocked_scene",
            metadata={
                **controller,
                **attempt,
                "branch_start_hashes": {
                    "simulator_state_sha256": f"{pair_id}_blocked",
                    "controller_state_sha256": f"{pair_id}_controller",
                    "observation_sha256": f"{pair_id}_blocked_observation",
                },
                "blocked_evidence": {"controller_class": "finite grid"},
            },
        ))
    return records


def test_glass_placement_design_rejects_split_state_leakage():
    train_hash = array_sha256(np.asarray([1.0, 2.0]))
    heldout_hash = array_sha256(np.asarray([3.0, 4.0]))
    placements = [
        _placement("train", "train", train_hash, "train/nominal"),
        _placement("validation", "validation", heldout_hash, "validation/tall"),
        _placement("heldout", "heldout", array_sha256([5.0]), "heldout/wide"),
    ]
    summary = validate_placement_design(placements)
    assert summary["placements_by_split"] == {"train": 1, "validation": 1, "heldout": 1}
    leaked = placements[:2] + [
        _placement("heldout", "heldout", train_hash, "heldout/wide")
    ]
    with pytest.raises(ValueError, match="leaks across splits"):
        validate_placement_design(leaked)


def test_v2_physical_family_and_scene_fingerprints_fail_closed():
    placements = []
    for split, family, scene in (
        ("train", "family-a", "scene-a"),
        ("validation", "family-b", "scene-b"),
        ("heldout", "family-c", "scene-c"),
    ):
        payload = _placement(split, split, f"state-{split}", f"{split}/family").to_dict()
        payload["metadata"] = {
            "geometry_family_fingerprint": family,
            "physical_scene_sha256": scene,
        }
        placements.append(GlassPlacement.from_dict(payload))
    summary = validate_placement_design(placements)
    assert summary["physical_scene_count"] == 3
    leaked_family = [*placements]
    payload = leaked_family[-1].to_dict()
    payload["metadata"]["geometry_family_fingerprint"] = "family-a"
    leaked_family[-1] = GlassPlacement.from_dict(payload)
    with pytest.raises(ValueError, match="geometry family leaks"):
        validate_placement_design(leaked_family)
    duplicate_scene = [*placements]
    payload = duplicate_scene[-1].to_dict()
    payload["metadata"]["physical_scene_sha256"] = "scene-a"
    duplicate_scene[-1] = GlassPlacement.from_dict(payload)
    with pytest.raises(ValueError, match="duplicate physical scene"):
        validate_placement_design(duplicate_scene)


def test_glass_placement_rejects_non_boolean_mobility_marker():
    payload = _placement("bad", "train", "state", "train/nominal").to_dict()
    payload["blocked_glasses"] = [{**_glass("bad"), "movable": "no"}]
    with pytest.raises(ValueError, match="movable must be boolean"):
        GlassPlacement.from_dict(payload)


def test_glass_placement_allows_no_blocked_auxiliary_design():
    payload = _placement("primary_only", "train", "state", "train/nominal").to_dict()
    payload["blocked_glasses"] = []
    assert GlassPlacement.from_dict(payload).blocked_glasses == []


def test_legacy_v1_pair_remains_readable_but_cannot_enter_v2_primary():
    common = dict(
        pair_id="pair_0", placement_id="placement_0", split="train",
        source_state_sha256="source", matched_robot_state_sha256="robot",
        arrays_path="train/pair/branch.npz", instruction="pick", n_steps=2,
        scene_sha256="scene", schema_version=1,
    )
    records = [
        PairedTrajectoryRecord(
            **common, trajectory_kind="nominal_catastrophe", branch_start_state_sha256="onpath",
            outcome="crash", crashed=True, succeeded=False, safe_abort=False,
            oracle_verified=False,
        ),
        PairedTrajectoryRecord(
            **common, trajectory_kind="oracle_recovery", branch_start_state_sha256="onpath",
            outcome="recovery_success", crashed=False, succeeded=True, safe_abort=False,
            oracle_verified=True,
        ),
        PairedTrajectoryRecord(
            **common, trajectory_kind="off_path_control", branch_start_state_sha256="offpath",
            outcome="recovery_success", crashed=False, succeeded=True, safe_abort=False,
            oracle_verified=False,
        ),
        PairedTrajectoryRecord(
            **common, trajectory_kind="blocked_safe_abort", branch_start_state_sha256="blocked",
            outcome="safe_abort", crashed=False, succeeded=False, safe_abort=True,
            oracle_verified=True,
            metadata={"blocked_evidence": {"controller_class": "finite detour grid"}},
        ),
    ]
    assert validate_paired_records(records)["pairs"] == 1
    with pytest.raises(ValueError, match="legacy schema v1"):
        validate_primary_pair(records)
    changed = [records[0], PairedTrajectoryRecord(
        **{**records[1].to_dict(), "branch_start_state_sha256": "different"}
    ), *records[2:]]
    with pytest.raises(ValueError, match="not exact-state matched"):
        validate_paired_records(changed)

    controller_changed = [
        PairedTrajectoryRecord(**{
            **record.to_dict(),
            "metadata": {
                **record.metadata,
                "controller_state_sha256": "first" if index < 3 else "different",
            },
        })
        for index, record in enumerate(records)
    ]
    with pytest.raises(ValueError, match="controller state"):
        validate_paired_records(controller_changed)


def test_legacy_v1_manifest_is_read_only_compatible(tmp_path):
    records = []
    for record in _v2_records(include_blocked=True):
        payload = record.to_dict()
        payload.pop("schema_version")
        payload.pop("task_suite")
        payload.pop("task_id")
        payload.pop("trigger_horizon_actions")
        records.append(payload)
    manifest = tmp_path / "legacy.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    legacy = read_trajectory_manifest(manifest)
    assert {record.schema_version for record in legacy} == {1}
    with pytest.raises(ValueError, match="read-only"):
        write_trajectory_manifest(tmp_path / "copy.jsonl", legacy)


def test_h1_labels_only_observation_whose_next_action_catastrophizes():
    rows = _trajectory_rows(5)
    arrays = _finalize_arrays(
        rows, kind="nominal_catastrophe", collision_step=4
    )
    assert np.array_equal(arrays["time_to_catastrophe_actions"], [5, 4, 3, 2, 1])
    assert np.array_equal(arrays["risk_targets"][:, 0], [0, 0, 0, 0, 1])


def test_exact_h_anchor_suffix_catastrophizes_after_exactly_h_actions():
    collision_action_index = 7
    horizon = 3
    anchor = exact_h_anchor_index(collision_action_index, horizon)
    assert anchor == 5
    assert steps_until_event(collision_action_index, anchor) == horizon
    suffix = list(range(anchor, collision_action_index + 1))
    assert len(suffix) == horizon
    assert suffix[-1] - anchor == horizon - 1

    class Sim:
        libero_done = False

        def __init__(self):
            self.xy = (0.0, 0.0)

        def object_xy(self, name):
            return self.xy

        def object_tilt_deg(self, name):
            return 0.0

        def max_contact_force(self, bodies, against=None):
            return 0.0

    class Env:
        def __init__(self, move_on_action):
            self.sim_view = Sim()
            self.steps = 0
            self.move_on_action = move_on_action

        def step(self, action):
            self.steps += 1
            if self.steps == self.move_on_action:
                self.sim_view.xy = (0.061, 0.0)
            return {}, 0.0, False, {}

    rows = [{"executed_action": np.zeros(7, dtype=np.float32)} for _ in range(horizon)]
    replay = _replay_nominal_actions(Env(horizon), {}, [_glass()], rows)
    assert replay["crashed"] is True
    assert replay["collision_step"] == horizon - 1
    early = _replay_nominal_actions(Env(horizon), {}, [_glass()], rows[:-1])
    assert early["crashed"] is False


def test_oracle_row_zero_risk_equals_exact_matched_nominal_row():
    horizon = 5
    nominal = _finalize_arrays(
        _trajectory_rows(horizon),
        kind="nominal_catastrophe",
        collision_step=horizon - 1,
    )
    oracle = _finalize_arrays(
        _trajectory_rows(2),
        kind="oracle_recovery",
        counterfactual_collision_step=horizon - 1,
    )
    assert oracle["severity_force"][0] != nominal["severity_force"][0]
    label_hash = _align_oracle_row_zero_labels(nominal, oracle)
    assert np.array_equal(oracle["risk_targets"][0], nominal["risk_targets"][0])
    assert np.array_equal(oracle["risk_mask"][0], nominal["risk_mask"][0])
    assert oracle["severity_force"][0] == nominal["severity_force"][0]
    assert oracle["time_to_catastrophe_actions"][0] == horizon
    assert isinstance(label_hash, str) and len(label_hash) == 64


def test_explicit_reference_detects_first_action_glass_displacement():
    sim = SimpleNamespace(object_xy=lambda name: sim.xy)
    sim.xy = (0.0, 0.0)
    displaced = build_predicate(PredicateSpec("object_displaced", {
        "object_name": "glass",
        "max_disp": 0.06,
        "reference_xy": [0.0, 0.0],
    }))
    prime_predicate(displaced, sim)
    sim.xy = (0.061, 0.0)
    assert displaced(sim) is True


def test_collector_detects_displacement_caused_by_first_action():
    class Sim:
        libero_done = False

        def __init__(self):
            self.xy = (0.0, 0.0)

        def object_xy(self, name):
            return self.xy

        def object_tilt_deg(self, name):
            return 0.0

        def max_contact_force(self, bodies, against=None):
            return 0.0

    class Env:
        def __init__(self):
            self.sim_view = Sim()

        def policy_observation(self, obs, resize_size):
            return {
                "full_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "state": np.zeros(8, dtype=np.float32),
            }

        def step(self, action):
            self.sim_view.xy = (0.061, 0.0)
            return {}, 0.0, False, {}

    policy = SimpleNamespace(
        resize_size=2,
        last_hidden=np.zeros(4, dtype=np.float32),
        act=lambda observation, instruction: np.zeros(7, dtype=np.float32),
    )
    result = _roll_nominal(Env(), policy, {}, "pick", [_glass()], 2)
    assert result["crashed"] is True
    assert result["collision_step"] == 0
    assert result["steps_to_event"] == 1


def test_timeout_control_tail_is_right_censored():
    arrays = _finalize_arrays(
        _trajectory_rows(4), kind="off_path_control", right_censored=True
    )
    assert np.array_equal(arrays["risk_mask"][:, 0], [1, 1, 1, 1])
    assert np.array_equal(arrays["risk_mask"][:, 1], [1, 1, 0, 0])
    assert np.array_equal(arrays["risk_mask"][:, 2], [0, 0, 0, 0])
    assert np.array_equal(arrays["severity_mask"], [0, 0, 0, 0])


def test_v2_arrays_are_default_fail_closed_but_v1_is_explicitly_readable():
    arrays = _finalize_arrays(
        _trajectory_rows(2), kind="nominal_catastrophe", collision_step=1
    )
    for key in (
        "time_to_catastrophe_actions",
        "time_to_catastrophe_mask",
        "oracle_recoverable_from_this_state",
        "oracle_verified_mask",
        "latest_verified_recoverable_state",
        "runtime_trigger_eligible",
    ):
        arrays.pop(key)
    with pytest.raises(ValueError, match="episode arrays missing"):
        validate_episode_arrays(arrays)
    assert validate_episode_arrays(arrays, schema_version=1) == 2


def test_v2_arrays_reject_risk_and_oracle_semantic_contradictions():
    arrays = _finalize_arrays(
        _trajectory_rows(3), kind="nominal_catastrophe", collision_step=2
    )
    arrays["risk_targets"][0] = 0.0
    with pytest.raises(ValueError, match="risk_targets disagree"):
        validate_episode_arrays(arrays)

    arrays = _finalize_arrays(
        _trajectory_rows(3), kind="nominal_catastrophe", collision_step=2
    )
    arrays["oracle_verified_mask"][0] = 0.0
    with pytest.raises(ValueError, match="recoverability must be verified"):
        validate_episode_arrays(arrays)


def test_v2_missing_controller_hash_fails_primary_admission():
    records = _v2_records()
    payload = records[2].to_dict()
    payload["metadata"] = {}
    records[2] = PairedTrajectoryRecord(**payload)
    with pytest.raises(ValueError, match="controller hash"):
        validate_primary_pair(records)


def test_v2_requires_exact_simulator_controller_and_observation_hashes():
    records = _v2_records()
    payload = records[1].to_dict()
    payload["metadata"] = {
        **payload["metadata"],
        "branch_start_hashes": {
            **payload["metadata"]["branch_start_hashes"],
            "observation_sha256": "different_observation",
        },
    }
    records[1] = PairedTrajectoryRecord(**payload)
    with pytest.raises(ValueError, match="hashes differ"):
        validate_primary_pair(records)

    records = _v2_records()
    payload = records[2].to_dict()
    payload["metadata"] = {
        **payload["metadata"],
        "branch_start_hashes": {
            "simulator_state_sha256": payload["branch_start_state_sha256"],
            "controller_state_sha256": payload["metadata"]["controller_state_sha256"],
        },
    }
    records[2] = PairedTrajectoryRecord(**payload)
    with pytest.raises(ValueError, match="incomplete branch-start hashes"):
        validate_primary_pair(records)


def test_v2_requires_oracle_search_and_independent_recapture_evidence():
    for field in ("search_success", "independent_recapture", "recapture_success"):
        records = _v2_records()
        payload = records[1].to_dict()
        payload["metadata"] = {
            **payload["metadata"],
            "oracle_verification": {
                **payload["metadata"]["oracle_verification"],
                field: False,
            },
        }
        records[1] = PairedTrajectoryRecord(**payload)
        with pytest.raises(ValueError, match="independent recapture"):
            validate_primary_pair(records)


def test_v2_requires_oracle_row_zero_label_hash_to_match_nominal_anchor():
    records = _v2_records()
    payload = records[1].to_dict()
    payload["metadata"] = {
        **payload["metadata"],
        "row_zero_label_sha256": "different_labels",
    }
    records[1] = PairedTrajectoryRecord(**payload)
    with pytest.raises(ValueError, match="row-zero labels"):
        validate_primary_pair(records)


@pytest.mark.parametrize("field,value", [
    ("task_id", 1),
    ("split", "validation"),
    ("instruction", "different instruction"),
])
def test_v2_cross_task_split_or_instruction_mismatch_fails(field, value):
    records = _v2_records()
    payload = records[2].to_dict()
    payload[field] = value
    records[2] = PairedTrajectoryRecord(**payload)
    with pytest.raises(ValueError, match=field):
        validate_primary_pair(records)


def test_careful_and_optional_blocked_do_not_change_primary_acceptance():
    without_aux = _v2_records(careful_crashed=False)
    with_aux = _v2_records(include_blocked=True, careful_crashed=True)
    assert validate_primary_pair(without_aux)["pair_id"] == "pair_v2"
    result = validate_primary_pair(with_aux)
    assert result["pair_id"] == "pair_v2"
    assert result["blocked_auxiliary_present"] is True


def test_blocked_appendix_uses_a_separate_manifest(tmp_path):
    primary = _v2_records()
    blocked = _v2_records(include_blocked=True)[-1]
    _write_manifests(
        tmp_path,
        primary,
        {"test": True},
        blocked_appendix_records=[blocked],
    )
    primary_rows = [
        json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()
    ]
    appendix_rows = [
        json.loads(line)
        for line in (tmp_path / "blocked_appendix.jsonl").read_text().splitlines()
    ]
    assert {row["trajectory_kind"] for row in primary_rows} == {
        "nominal_catastrophe", "oracle_recovery", "off_path_control",
    }
    assert [row["trajectory_kind"] for row in appendix_rows] == ["blocked_safe_abort"]
    loaded_appendix = read_auxiliary_trajectory_manifest(
        tmp_path / "blocked_appendix.jsonl"
    )
    assert loaded_appendix == [blocked]
    assert validate_auxiliary_records(loaded_appendix)["pairs"] == 1
    with pytest.raises(ValueError, match="cannot enter primary manifests"):
        _write_manifests(tmp_path, [*primary, blocked], {"test": True})


def test_empty_current_cohort_clears_stale_primary_manifests(tmp_path):
    for split in ("train", "validation", "heldout"):
        (tmp_path / f"{split}.jsonl").write_text("stale\n")
    _write_manifests(tmp_path, [], {"protocol": "current"})
    for split in ("train", "validation", "heldout"):
        assert (tmp_path / f"{split}.jsonl").read_text() == ""
    assert (tmp_path / "blocked_appendix.jsonl").read_text() == ""
    summary = json.loads((tmp_path / "collection_summary.json").read_text())
    assert summary["validation"]["pairs"] == 0
    assert summary["metadata"]["protocol"] == "current"


def test_offpath_timeout_is_not_safe_abort_or_primary_utility_control():
    records = _v2_records()
    payload = records[2].to_dict()
    payload.update(outcome="timeout", succeeded=False, safe_abort=False)
    payload["metadata"] = {
        **payload["metadata"], "termination": "timeout", "right_censored": True,
    }
    records[2] = PairedTrajectoryRecord(**payload)
    with pytest.raises(ValueError, match="complete the original task"):
        validate_primary_pair(records)
    payload["safe_abort"] = True
    with pytest.raises(ValueError, match="timeout cannot be safe_abort"):
        PairedTrajectoryRecord(**payload)


@pytest.mark.parametrize("record_index,bad_outcome", [
    (0, "timeout"),
    (1, "task_success"),
    (2, "timeout"),
])
def test_v2_primary_branches_require_canonical_outcomes(record_index, bad_outcome):
    payload = _v2_records()[record_index].to_dict()
    payload["outcome"] = bad_outcome
    with pytest.raises(ValueError, match="outcome must be"):
        PairedTrajectoryRecord(**payload)


def test_controller_state_roundtrip_restores_osc_interpolators():
    class Controller(SimpleNamespace):
        def update(self, force=False):
            self.updated_with_force = force

    controller = Controller(
        goal_pos=np.asarray([1.0, 2.0, 3.0]), goal_ori=np.eye(3),
        ori_ref=None, relative_ori=np.asarray([0.1, 0.2, 0.3]),
        new_update=True,
        interpolator_pos=SimpleNamespace(
            start=np.asarray([0.0, 1.0, 2.0]), goal=np.asarray([1.0, 2.0, 3.0]), step=2,
        ),
        interpolator_ori=SimpleNamespace(
            start=np.asarray([0.0, 0.1, 0.2]), goal=np.asarray([0.3, 0.4, 0.5]), step=1,
        ),
    )
    env = LiberoEnv.__new__(LiberoEnv)
    env.env = SimpleNamespace(env=SimpleNamespace(
        robots=[SimpleNamespace(controller=controller)]
    ))
    model = SimpleNamespace(nv=2, na=1, nu=2)
    data = SimpleNamespace(
        qacc_warmstart=np.asarray([0.4, 0.5]),
        act=np.asarray([0.6]), ctrl=np.asarray([0.7, 0.8]),
    )
    env.sim_view = SimpleNamespace(_live_mj=lambda: (model, data))
    snapshot = env.controller_state()
    assert snapshot["robot0.controller.ori_ref"].shape == (0,)
    assert all(not value.dtype.hasobject for value in snapshot.values())
    controller.goal_pos[:] = -1
    controller.interpolator_pos.goal[:] = -1
    controller.interpolator_pos.step = 0
    controller.new_update = False
    data.qacc_warmstart[:] = -1
    data.act[:] = -1
    data.ctrl[:] = -1

    env.restore_controller_state(snapshot)

    assert controller.updated_with_force is True
    assert np.array_equal(controller.goal_pos, [1.0, 2.0, 3.0])
    assert np.array_equal(controller.interpolator_pos.goal, [1.0, 2.0, 3.0])
    assert controller.interpolator_pos.step == 2
    assert controller.new_update is True
    assert controller.ori_ref is None
    assert np.array_equal(data.qacc_warmstart, [0.4, 0.5])
    assert np.array_equal(data.act, [0.6])
    assert np.array_equal(data.ctrl, [0.7, 0.8])


def _batch(n=4, hidden_dim=6):
    return {
        "hidden": torch.randn(n, hidden_dim),
        "robot_state": torch.randn(n, 8),
        "nominal_action": torch.zeros(n, 7),
        "target_action": torch.full((n, 7), 0.25),
        "risk_targets": torch.randint(0, 2, (n, len(RISK_HORIZONS))).float(),
        "risk_mask": torch.ones(n, len(RISK_HORIZONS)),
        "hazard_type": torch.tensor([0, 1, 1, 0]),
        "severity_force": torch.tensor([0.0, 20.0, 50.0, 0.0]),
        "severity_mask": torch.ones(n),
        "abort_target": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        "recovery_mask": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "invariance_mask": torch.tensor([1.0, 0.0, 0.0, 1.0]),
        "sensitivity_mask": torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }


def test_joint_critic_recovery_loss_and_checkpoint_roundtrip():
    config = GlassRecoveryConfig(hidden_dim=6, width=12, depth=2, dropout=0.0)
    model = GlassRecoveryNetwork(config)
    batch = _batch()
    outputs = model(batch["hidden"], batch["robot_state"], batch["nominal_action"])
    assert outputs["risk_logits"].shape == (4, len(RISK_HORIZONS))
    assert outputs["hazard_logits"].shape == (4, len(HAZARD_TYPES))
    assert outputs["recovery_action"].shape == (4, 7)
    loss, terms = glass_recovery_loss(outputs, batch)
    assert torch.isfinite(loss) and set(terms) == {
        "recovery_bc", "risk", "hazard", "severity", "abort",
        "control_invariance", "hazard_sensitivity",
    }
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        model.save_checkpoint(path, {"calibration": {"threshold": 0.5}})
        restored, metadata = GlassRecoveryNetwork.load_checkpoint(path)
        assert restored.config.hidden_dim == 6
        assert metadata["calibration"]["threshold"] == 0.5


def test_primary_loss_owns_only_risk_recovery_and_control_invariance():
    weights = GlassLossWeights()
    weights.validate_primary()
    assert asdict(weights) == {
        "recovery_bc": 1.0,
        "risk": 1.0,
        "hazard": 0.0,
        "severity": 0.0,
        "abort": 0.0,
        "control_invariance": 0.5,
        "hazard_sensitivity": 0.0,
    }
    model = GlassRecoveryNetwork(GlassRecoveryConfig(
        hidden_dim=6, width=8, depth=1, dropout=0.0,
    ))
    batch = _batch()
    outputs = model(batch["hidden"], batch["robot_state"], batch["nominal_action"])
    loss, _ = glass_recovery_loss(outputs, batch, weights=weights)
    loss.backward()
    assert model.hazard_head.weight.grad is None
    assert model.severity_head.weight.grad is None
    assert model.abort_head.weight.grad is None
    with pytest.raises(ValueError, match="zero auxiliary loss weights"):
        GlassLossWeights(abort=0.1).validate_primary()


def test_blocked_rows_cannot_enter_recovery_bc_even_with_legacy_mask():
    outputs = {
        "risk_logits": torch.zeros(2, len(RISK_HORIZONS)),
        "hazard_logits": torch.zeros(2, len(HAZARD_TYPES)),
        "severity_log": torch.zeros(2),
        "abort_logit": torch.zeros(2),
        "recovery_action": torch.stack((torch.zeros(7), torch.ones(7))),
        "action_delta": torch.stack((torch.zeros(7), torch.ones(7))),
    }
    batch = {
        "risk_targets": torch.zeros(2, len(RISK_HORIZONS)),
        "risk_mask": torch.ones(2, len(RISK_HORIZONS)),
        "hazard_type": torch.zeros(2, dtype=torch.long),
        "severity_force": torch.zeros(2),
        "severity_mask": torch.ones(2),
        "abort_target": torch.zeros(2),
        "target_action": torch.zeros(2, 7),
        "recovery_mask": torch.ones(2),
        "invariance_mask": torch.zeros(2),
        "sensitivity_mask": torch.zeros(2),
        "trajectory_kind": torch.tensor([1, 3]),
    }
    _, terms = glass_recovery_loss(outputs, batch, weights=GlassLossWeights())
    assert terms["recovery_bc"].item() == pytest.approx(0.0)


def test_direct_recovery_action_can_reverse_saturated_nominal_action():
    model = GlassRecoveryNetwork(GlassRecoveryConfig(
        hidden_dim=6, width=8, depth=1, dropout=0.0,
    ))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    model.action_head.bias.data.fill_(-4.0)
    nominal = torch.ones(1, 7)
    outputs = model(torch.zeros(1, 6), torch.zeros(1, 8), nominal)
    assert torch.all(outputs["recovery_action"] < -0.99)
    assert torch.allclose(
        outputs["action_delta"], outputs["recovery_action"] - nominal
    )


def test_critic_predictions_condition_on_nominal_action():
    model = GlassRecoveryNetwork(GlassRecoveryConfig(
        hidden_dim=6, width=8, depth=1, dropout=0.0,
    ))
    hidden = torch.randn(1, 6).repeat(2, 1)
    state = torch.randn(1, 8).repeat(2, 1)
    nominal = torch.stack((torch.arange(7), torch.arange(6, -1, -1))).float()
    outputs = model(hidden, state, nominal)
    assert not torch.allclose(outputs["risk_logits"][0], outputs["risk_logits"][1])


def test_calibration_maximizes_certified_timely_trigger_not_any_crossing():
    # The nominal episode's late 0.99 crossing is outside the certified window;
    # calibration must optimize its row-zero 0.70 score instead.
    frame_scores = np.asarray([0.10, 0.60, 0.20, 0.55, 0.70, 0.99])
    valid = np.ones(6, dtype=bool)
    qualified = np.asarray([0, 0, 0, 0, 1, 0], dtype=bool)
    episode_ids = np.asarray([0, 0, 1, 1, 2, 2])
    kinds = np.asarray([2, 2, 2, 2, 0, 0])
    controls = _episode_max_scores(frame_scores, valid, episode_ids, kinds, 2)
    timely = _episode_qualified_scores(
        frame_scores, valid, qualified, episode_ids, kinds, 0
    )
    calibration = _select_timely_risk_threshold(controls, timely, 0.0)
    assert np.array_equal(controls, [0.60, 0.55])
    assert np.array_equal(timely, [0.70])
    assert calibration["threshold"] == pytest.approx(0.7)
    assert calibration["control_episode_fpr"] == 0.0
    assert calibration["timely_trigger_rate"] == 1.0
    assert calibration["calibration_unit"] == (
        "control_episode_max_vs_certified_trigger_frame"
    )


def test_episode_array_schema():
    n = 3
    arrays = {
        "hidden": np.zeros((n, 6)), "robot_state": np.zeros((n, 8)),
        "nominal_action": np.zeros((n, 7)), "target_action": np.zeros((n, 7)),
        "executed_action": np.zeros((n, 7)),
        "risk_targets": np.zeros((n, 5)), "risk_mask": np.ones((n, 5)),
        "hazard_type": np.zeros(n), "severity_force": np.zeros(n),
        "severity_mask": np.ones(n), "abort_target": np.zeros(n),
        "recovery_mask": np.zeros(n), "invariance_mask": np.ones(n),
        "sensitivity_mask": np.zeros(n),
    }
    assert validate_episode_arrays(arrays, schema_version=1) == n
    arrays["nominal_action"] = np.zeros((n, 6))
    with pytest.raises(ValueError, match=r"\[N, 7\]"):
        validate_episode_arrays(arrays, schema_version=1)


class _FakeBase:
    capture_hidden = True
    resize_size = 224

    def __init__(self, revision: str = "a" * 40, unnorm_key: str = "libero_spatial"):
        self.last_hidden = None
        self.checkpoint_identity = {"resolved_revision": revision}
        self.cfg = SimpleNamespace(unnorm_key=unnorm_key)

    def act(self, observation, instruction):
        self.last_hidden = np.ones(4, dtype=np.float32)
        return np.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def _primary_checkpoint_metadata(horizon: int = 5) -> dict:
    return {
        "checkpoint_kind": PRIMARY_CHECKPOINT_KIND,
        "base_resolved_revision": "a" * 40,
        "unnorm_key": "libero_spatial",
        "hidden_hook_identity": HIDDEN_HOOK_IDENTITY,
        "train_manifest_sha256": "b" * 64,
        "validation_manifest_sha256": "c" * 64,
        "trajectory_schema_version": SCHEMA_VERSION,
        "protocol_sha256": "d" * 64,
        "trigger_horizon_actions": horizon,
        "tte_definition": TTE_DEFINITION,
        "seed": 17,
        "disabled_auxiliary_heads": list(PRIMARY_DISABLED_AUXILIARY_HEADS),
        "disabled_auxiliary_loss_terms": list(PRIMARY_DISABLED_AUXILIARY_TERMS),
        "loss_weights": asdict(GlassLossWeights()),
        "calibration": {
            "threshold": 0.5,
            "horizon": horizon,
            "objective": (
                "maximize_timely_trigger_rate_subject_to_clean_control_episode_fpr"
            ),
            "ownership": "recovery_latched_until_terminal_or_reset",
            "abort_enabled": False,
        },
    }


def _constant_checkpoint(path: Path, risk_bias: float, abort_bias: float, *, metadata=None):
    model = GlassRecoveryNetwork(GlassRecoveryConfig(
        hidden_dim=4, width=8, depth=1, dropout=0.0,
    ))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    model.risk_head.bias.data.fill_(risk_bias)
    model.abort_head.bias.data.fill_(abort_bias)
    model.action_head.bias.data.fill_(0.5)
    model.save_checkpoint(path, metadata or _primary_checkpoint_metadata())


def test_runtime_ownership_latches_recovery_until_terminal_or_reset_and_disables_abort():
    observation = {"state": np.asarray([0.2, 0.0, 0.9, 0, 0, 0, 0, 0], dtype=np.float32)}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        low = root / "low.pt"; recover = root / "recover.pt"; abort = root / "abort.pt"
        _constant_checkpoint(low, -10.0, -10.0)
        _constant_checkpoint(recover, 10.0, -10.0)
        _constant_checkpoint(abort, 10.0, 10.0)
        nominal_policy = GlassRecoveryPolicy(
            _FakeBase(), str(low), risk_horizon=5, device="cpu"
        )
        assert np.allclose(nominal_policy.act(observation, "pick")[:3], [0.1, 0.0, 0.0])
        assert nominal_policy.mode == "nominal"
        recovery_policy = GlassRecoveryPolicy(
            _FakeBase(), str(recover), risk_horizon=5, device="cpu"
        )
        recovery_action = recovery_policy.act(observation, "pick")
        assert recovery_policy.mode == "recovery_latched"
        assert not np.allclose(recovery_action, [0.1, 0, 0, 0, 0, 0, -1])
        with torch.no_grad():
            recovery_policy.model.risk_head.bias.fill_(-10.0)
        second = recovery_policy.act(observation, "pick")
        assert recovery_policy.mode == "recovery_latched"
        assert np.allclose(second, recovery_action)
        assert recovery_policy.last_decision["ownership_age"] == 1
        assert recovery_policy.decisions[0]["first_trigger"] is True
        assert recovery_policy.decisions[1]["first_trigger"] is False
        for key in (
            "risk_vector", "nominal_action", "recovery_action", "executed_action",
            "mode", "first_trigger_step", "ownership_age",
        ):
            assert key in recovery_policy.last_decision
        recovery_policy.mark_terminal()
        with pytest.raises(RuntimeError, match="after terminal"):
            recovery_policy.act(observation, "pick")
        recovery_policy.reset()
        assert recovery_policy.mode == "nominal"
        assert recovery_policy.ownership_age is None

        abort_policy = GlassRecoveryPolicy(
            _FakeBase(), str(abort), risk_horizon=5, device="cpu"
        )
        abort_policy.act(observation, "pick")
        assert abort_policy.mode == "recovery_latched"
        assert abort_policy.abort_enabled is False
        with pytest.raises(ValueError, match="abort is disabled"):
            GlassRecoveryPolicy(
                _FakeBase(), str(abort), risk_horizon=5,
                abort_controller=object(), device="cpu",
            )


def test_runtime_fails_closed_on_checkpoint_base_h_and_contract_mismatch():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "valid.pt"
        _constant_checkpoint(checkpoint, -10.0, -10.0)
        with pytest.raises(ValueError, match="resolved revision"):
            GlassRecoveryPolicy(
                _FakeBase(revision="e" * 40), str(checkpoint),
                risk_horizon=5, device="cpu",
            )
        with pytest.raises(ValueError, match="runtime H"):
            GlassRecoveryPolicy(
                _FakeBase(), str(checkpoint), risk_horizon=10, device="cpu"
            )

        bad_metadata = _primary_checkpoint_metadata()
        bad_metadata["train_manifest_sha256"] = "not-a-sha"
        malformed = root / "malformed.pt"
        _constant_checkpoint(malformed, -10.0, -10.0, metadata=bad_metadata)
        with pytest.raises(ValueError, match="train_manifest_sha256"):
            GlassRecoveryPolicy(
                _FakeBase(), str(malformed), risk_horizon=5, device="cpu"
            )


def test_detour_compensates_grasp_offset_before_placing():
    controller = DetourComplete(
        {"pos": [0.0, 0.0, 0.96], "size": [0.03, 0.03, 0.06]},
        target_pos=[0.1, 0.2, 0.91], plate_pos=[0.5, 0.6, 0.90],
        transit_z=1.2, target_name="bowl",
    )
    obs = {"robot0_eef_pos": np.asarray([0.0, 0.0, 1.2]),
           "bowl_pos": np.asarray([0.1, 0.2, 1.0])}
    controller.engage(obs)
    controller.i = 7
    controller.step(obs)
    assert controller._carry_adjusted is True
    # Held bowl is +[.1,.2] from the EEF, so the EEF target is plate-offset.
    assert np.allclose(controller.legs[7][1], [0.4, 0.4, 1.2])


def test_oracle_grid_searches_both_grasp_heights():
    configs = _oracle_configs(0.9)
    assert len(configs) == 16
    assert {config["descend_off"] for config in configs} == {0.012, 0.04}


def test_oracle_grid_can_reuse_clean_control_grasp_pose():
    configs = _oracle_configs(
        0.9,
        orientation_targets=[[3.1, 0.0, 0.1], None],
        control_grasp_offset=[-0.002, -0.05, 0.021],
    )
    assert len(configs) == 32
    assert {config["descend_off"] for config in configs} == {0.021, 0.04}
    assert all(config["grasp_xy_offset"] == [-0.002, -0.05] for config in configs)


def test_oracle_success_is_searched_then_independently_recaptured(monkeypatch):
    placement_payload = _placement("oracle", "train", "state", "train/nominal").to_dict()
    placement_payload["metadata"] = {"bowl_xyz": [0.1, 0.2, 0.9]}
    placement = GlassPlacement.from_dict(placement_payload)
    config = {
        "side": -1.0,
        "lane_margin": 0.12,
        "transit_z": 1.2,
        "descend_off": 0.012,
        "orientation_target": None,
        "path_aligned": True,
        "grasp_xy_offset": [0.0, 0.0],
    }
    monkeypatch.setattr(glass_collector, "_oracle_configs", lambda *args: [config])
    monkeypatch.setattr(glass_collector, "DetourComplete", lambda *args, **kwargs: object())
    capture_modes = []

    def fake_run_controller(*args, capture_rows, **kwargs):
        capture_modes.append(capture_rows)
        return {
            "crashed": False,
            "succeeded": True,
            "steps": 4,
            "peak_force": 1.0,
            "controller_final_stage": 8,
            "initial_target_z_m": 0.9,
            "max_target_z_m": 1.0,
            "controller_trace": [],
            "obs": {
                "akita_black_bowl_1_pos": np.asarray([0.5, 0.6, 0.91]),
                "plate_1_pos": np.asarray([0.5, 0.6, 0.90]),
            },
            "rows": _trajectory_rows(2) if capture_rows else [],
        }

    monkeypatch.setattr(glass_collector, "_run_controller", fake_run_controller)
    reset_calls = []

    def reset():
        reset_calls.append(len(reset_calls))
        return {
            "akita_black_bowl_1_pos": np.asarray([0.1, 0.2, 0.9]),
            "plate_1_pos": np.asarray([0.5, 0.6, 0.9]),
        }

    collected, attempts = _search_oracle(
        reset,
        SimpleNamespace(),
        SimpleNamespace(),
        placement,
        [placement.on_path_glass],
        20,
        collect_success=True,
        counterfactual_peak_force=30.0,
    )
    assert len(attempts) == 1
    assert attempts[0]["selected_for_independent_recapture"] is True
    assert capture_modes == [False, True]
    assert len(reset_calls) == 2
    assert collected["verification"]["search_success"] is True
    assert collected["verification"]["independent_recapture"] is True
    assert collected["verification"]["recapture_success"] is True


def test_safe_abort_search_prefers_hold_and_requires_stability():
    configs = _safe_abort_configs(0.14, 0.10)
    assert configs[0] == {"name": "hold", "back": 0.0, "up": 0.0}
    assert {(row["back"], row["up"]) for row in configs} == {
        (0.0, 0.0), (0.14, 0.0), (0.0, 0.10), (0.14, 0.10),
    }
    assert _stable_abort(
        {"crashed": False, "succeeded": False, "peak_force": 0.0}, 25.0
    )
    assert not _stable_abort(
        {"crashed": True, "succeeded": False, "peak_force": 0.0}, 25.0
    )


def test_detour_can_track_an_absolute_wrist_orientation():
    controller = DetourComplete(
        {"pos": [0.0, 0.0, 0.96], "size": [0.03, 0.03, 0.06]},
        target_pos=[0.1, 0.2, 0.91], plate_pos=[0.5, 0.6, 0.90],
        orientation_target=[0.0, 0.0, 0.2],
    )
    obs = {
        "robot0_eef_pos": np.asarray([0.0, 0.0, 1.2]),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
    }
    controller.engage(obs)
    action = controller.step(obs)
    assert action[5] > 0


def test_detour_orientation_uses_short_arc_across_pi_boundary():
    controller = DetourComplete(
        {"pos": [0.0, 0.0, 0.96], "size": [0.03, 0.03, 0.06]},
        target_pos=[0.1, 0.2, 0.91], plate_pos=[0.5, 0.6, 0.90],
        orientation_target=[3.13, 0.0, 0.0],
    )
    obs = {
        "state": np.asarray([0.0, 0.0, 1.2, -3.13, 0.0, 0.0, 0.0, 0.0]),
    }
    controller.engage(obs)
    action = controller.step(obs)
    assert 0.0 < abs(action[3]) < 0.1


def test_glass_detour_approaches_along_glass_target_path():
    controller = DetourComplete(
        {"pos": [0.0, 0.0, 0.96], "size": [0.03, 0.03, 0.06]},
        target_pos=[0.1, 0.0, 0.91], plate_pos=[0.5, 0.6, 0.90],
        lane_margin=0.12, transit_z=1.2, path_aligned=True,
        pregrasp_offset=0.02,
    )
    obs = {"robot0_eef_pos": np.asarray([-0.1, 0.0, 1.2])}
    controller.engage(obs)
    # The third waypoint returns to the nominal path 2 cm before the bowl;
    # the fourth completes only that short approach, rather than forcing +x.
    assert np.allclose(controller.legs[2][1], [0.08, -0.0, 1.2])
    assert np.allclose(controller.legs[3][1], [0.1, 0.0, 1.2])


def test_glass_detour_uses_matched_control_grasp_offset():
    controller = DetourComplete(
        {"pos": [0.0, 0.0, 0.96], "size": [0.03, 0.03, 0.06]},
        target_pos=[0.1, 0.0, 0.91], plate_pos=[0.5, 0.6, 0.90],
        transit_z=1.2, descend_off=0.021, path_aligned=True,
        grasp_xy_offset=[-0.002, -0.05],
    )
    controller.engage({"robot0_eef_pos": np.asarray([-0.1, 0.0, 1.2])})
    assert np.allclose(controller.legs[3][1], [0.098, -0.05, 1.2])
    assert np.allclose(controller.legs[4][1], [0.098, -0.05, 0.931])
    assert np.allclose(controller.legs[6][1], [0.098, -0.05, 1.2])


def test_late_glass_anchor_is_clamped_before_target_overlap():
    fraction, required = _collision_free_path_fraction(
        requested_fraction=0.83,
        path_length=0.23,
        glass_radius=0.03,
        target_radius=0.04,
        clearance_margin=0.005,
        minimum_clearance=0.10,
    )
    assert required == pytest.approx(0.10)
    assert fraction == pytest.approx(1.0 - required / 0.23)
    assert (1.0 - fraction) * 0.23 == pytest.approx(required)


def test_primary_gate_helpers_keep_rejections_auditable():
    assert CAREFUL_PROMPT_PREFIX == (
        "Move carefully and avoid collisions while completing the task."
    )
    assert _oracle_rejection_reason([]) == "no_oracle_recovery"
    assert _oracle_rejection_reason([{"crashed": True}]) == "oracle_collision"
    assert _oracle_rejection_reason([{"crashed": False}]) == "oracle_task_failure"
    assert _rejection_counts([
        {"reason": "no_base_crash"},
        {"reason": "no_base_crash"},
        {"reason": "off_path_catastrophe"},
        {},
    ]) == {
        "no_base_crash": 2,
        "no_oracle_recovery": 0,
        "oracle_collision": 0,
        "oracle_task_failure": 0,
        "oracle_recapture_failure": 0,
        "off_path_catastrophe": 1,
        "off_path_timeout": 0,
        "off_path_no_frames": 0,
        "invalid_initial_state": 0,
        "exact_state_restore_mismatch": 0,
        "other": 1,
    }


def test_exact_branch_start_hashes_include_observation_dtype_and_shape():
    env = SimpleNamespace(
        flat_state=lambda: np.asarray([0.0, 1.0], dtype=np.float64),
        controller_state=lambda: {"goal": np.asarray([1.0], dtype=np.float64)},
    )
    observation = {
        "state": np.asarray([1.0, 2.0], dtype=np.float32),
        "image": np.zeros((2, 2, 3), dtype=np.uint8),
    }
    hashes = _branch_start_hashes(env, observation)
    assert set(hashes) == {
        "simulator_state_sha256", "controller_state_sha256", "observation_sha256",
    }
    assert hashes["observation_sha256"] == _observation_sha256(observation)
    changed_dtype = {**observation, "state": observation["state"].astype(np.float64)}
    assert _observation_sha256(changed_dtype) != hashes["observation_sha256"]


def test_attempt_ledger_is_append_only_and_resume_keeps_terminal_rejection(tmp_path):
    args = Namespace(
        precrash_horizon=20,
        settle_steps=10,
        target_state_threshold=0.03,
        scan_steps=220,
        control_steps=220,
        oracle_steps=900,
        annotate_careful=False,
        appendix_blocked=False,
    )
    protocol_sha256 = canonical_sha256(_primary_protocol(args))
    placement = _placement("ledger", "train", "state", "train/nominal")
    identity = _attempt_identity(
        placement,
        rollout_seed=17,
        checkpoint_revision="a" * 40,
        code_commit="b" * 40,
        protocol_sha256=protocol_sha256,
    )
    ledger = tmp_path / ATTEMPT_LEDGER_NAME
    _append_attempt_event(ledger, identity, "started", split="train")
    first_bytes = ledger.read_bytes()
    _append_attempt_event(
        ledger,
        identity,
        "rejected",
        split="train",
        reason="no_base_crash",
        deterministic=True,
    )
    assert ledger.read_bytes().startswith(first_bytes)
    rows = _read_attempt_ledger(ledger)
    terminal = _terminal_attempts(rows)
    assert terminal[identity["attempt_key"]]["event"] == "rejected"
    _append_attempt_event(
        ledger,
        identity,
        "skipped_deterministic_rejection",
        reason="no_base_crash",
    )
    rows = _read_attempt_ledger(ledger)
    assert [row["event"] for row in rows] == [
        "started", "rejected", "skipped_deterministic_rejection",
    ]
    assert _terminal_attempts(rows)[identity["attempt_key"]]["event"] == "rejected"
    summary = _attempt_ledger_summary(
        rows,
        protocol_sha256=protocol_sha256,
        checkpoint_revision="a" * 40,
        code_commit="b" * 40,
        rollout_seed=17,
    )
    assert summary["unique_attempts"] == 1
    assert summary["deterministic_rejections"] == 1
    assert summary["rejection_counts"]["no_base_crash"] == 1

    different_seed = _attempt_identity(
        placement,
        rollout_seed=18,
        checkpoint_revision="a" * 40,
        code_commit="b" * 40,
        protocol_sha256=protocol_sha256,
    )
    assert different_seed["attempt_key"] != identity["attempt_key"]
    first_root = _attempt_pair_root(tmp_path, placement, identity)
    second_root = _attempt_pair_root(tmp_path, placement, different_seed)
    assert first_root != second_root
    assert identity["attempt_key"] in first_root.name
    assert different_seed["attempt_key"] in second_root.name


def test_primary_protocol_excludes_careful_and_blocked_appendix_flags():
    common = dict(
        precrash_horizon=20,
        settle_steps=10,
        target_state_threshold=0.03,
        scan_steps=220,
        control_steps=220,
        oracle_steps=900,
    )
    base = Namespace(**common, annotate_careful=False, appendix_blocked=False)
    annotated = Namespace(**common, annotate_careful=True, appendix_blocked=True)
    assert _primary_protocol(base) == _primary_protocol(annotated)
    changed_h = Namespace(**{**common, "precrash_horizon": 30})
    assert canonical_sha256(_primary_protocol(base)) != canonical_sha256(
        _primary_protocol(changed_h)
    )


def test_collector_completes_provenance_preflight_before_output_write(
    tmp_path, monkeypatch,
):
    output = tmp_path / "new_output"
    placement_payload = {
        "design": {
            "placements_by_split": {"train": 0, "validation": 0, "heldout": 0}
        }
    }
    monkeypatch.setattr(
        glass_collector,
        "read_placement_manifest",
        lambda path: ([], placement_payload),
    )

    def fake_repository_provenance(root, require_clean):
        assert require_clean is True
        assert not output.exists()
        return {"git_commit": "b" * 40}

    monkeypatch.setattr(
        glass_collector, "repository_provenance", fake_repository_provenance
    )
    monkeypatch.delenv("CB_CODE_COMMIT", raising=False)
    monkeypatch.setattr(glass_collector.sys, "argv", [
        "collect_glass_recovery_pairs.py",
        "--output", str(output),
        "--checkpoint-revision", "a" * 40,
        "--max-train", "0",
        "--max-validation", "0",
        "--max-heldout", "0",
    ])
    glass_collector.main()
    assert output.is_dir()
    assert json.loads((output / "collection_summary.json").read_text())["validation"][
        "pairs"
    ] == 0


def test_incomplete_resume_materializes_recovered_pair_in_primary_manifest(
    tmp_path, monkeypatch,
):
    output = tmp_path / "resume_output"
    placement = _placement("resume_pair", "train", "state", "train/nominal")
    args = Namespace(
        checkpoint="openvla/openvla-7b-finetuned-libero-spatial",
        unnorm_key="libero_spatial",
        precrash_horizon=20,
        settle_steps=10,
        target_state_threshold=0.03,
        scan_steps=220,
        control_steps=220,
        oracle_steps=900,
    )
    protocol_sha256 = canonical_sha256(_primary_protocol(args))
    identity = _attempt_identity(
        placement,
        rollout_seed=0,
        checkpoint_revision="a" * 40,
        code_commit="b" * 40,
        protocol_sha256=protocol_sha256,
    )
    pair_root = _attempt_pair_root(output, placement, identity)
    pair_root.mkdir(parents=True)
    records = []
    for record in _v2_records(pair_id=placement.placement_id, horizon=20):
        payload = record.to_dict()
        payload.update(
            placement_id=placement.placement_id,
            source_state_sha256=placement.source_state_sha256,
            instruction=placement.instruction,
            task_suite=placement.task_suite,
            task_id=placement.task_id,
            arrays_path=(
                Path("train") / pair_root.name / Path(record.arrays_path).name
            ).as_posix(),
        )
        payload["metadata"] = {
            **payload["metadata"],
            "attempt_key": identity["attempt_key"],
        }
        records.append(PairedTrajectoryRecord(**payload))
    (pair_root / "pair.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "attempt_identity": identity,
        "records": [record.to_dict() for record in records],
    }))
    placement_payload = {
        "design": {
            "placements_by_split": {"train": 1, "validation": 0, "heldout": 0}
        }
    }
    monkeypatch.setattr(
        glass_collector,
        "read_placement_manifest",
        lambda path: ([placement], placement_payload),
    )
    monkeypatch.setattr(
        glass_collector,
        "repository_provenance",
        lambda root, require_clean: {"git_commit": "b" * 40},
    )
    monkeypatch.delenv("CB_CODE_COMMIT", raising=False)
    monkeypatch.setattr(glass_collector.sys, "argv", [
        "collect_glass_recovery_pairs.py",
        "--output", str(output),
        "--checkpoint-revision", "a" * 40,
        "--max-train", "2",
        "--max-validation", "0",
        "--max-heldout", "0",
    ])
    with pytest.raises(SystemExit, match="could not meet requested"):
        glass_collector.main()
    manifest_rows = [
        json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()
    ]
    assert len(manifest_rows) == 3
    assert {row["pair_id"] for row in manifest_rows} == {placement.placement_id}


def test_careful_gate_uses_fixed_prefix_and_restores_policy(monkeypatch):
    placement = _placement("careful", "train", "state", "train/nominal")
    policy = SimpleNamespace(prompt_prefix="")
    env = SimpleNamespace(
        reset_to=lambda state, movable_objects: {"source": state},
        dummy_action=lambda: [0.0] * 7,
        step=lambda action: ({"settled": True}, 0.0, False, {}),
    )
    observed = {}

    def fake_roll(env_arg, policy_arg, obs, instruction, glasses, max_steps):
        observed.update({
            "prefix": policy_arg.prompt_prefix,
            "instruction": instruction,
            "glass": glasses[0],
            "max_steps": max_steps,
        })
        return {
            "crashed": True,
            "succeeded": False,
            "peak_force": 40.0,
            "steps_to_event": 7,
        }

    monkeypatch.setattr(glass_collector, "_roll_nominal", fake_roll)
    result = glass_collector._run_careful_gate(
        env, policy, np.asarray([1.0]), placement, settle_steps=1, max_steps=20,
    )
    assert observed == {
        "prefix": CAREFUL_PROMPT_PREFIX,
        "instruction": placement.instruction,
        "glass": placement.on_path_glass,
        "max_steps": 20,
    }
    assert policy.prompt_prefix == ""
    assert result["careful_crashed"] is True
    assert result["careful_succeeded"] is False
    assert result["careful_peak_glass_force_n"] == 40.0
    assert result["careful_steps_to_event"] == 7


def test_blocked_barrier_is_dense_and_nonoverlapping():
    offsets = _blocked_barrier_offsets(0.28, 9, 0.032, 0.20)
    spacing = float(offsets[1] - offsets[0])
    assert len(offsets) == 9
    assert spacing - 2 * 0.032 == pytest.approx(0.006)
    with pytest.raises(ValueError, match="overlap"):
        _blocked_barrier_offsets(0.28, 9, 0.036, 0.20)
    with pytest.raises(ValueError, match="exceeds"):
        _blocked_barrier_offsets(0.28, 9, 0.020, 0.20)


def test_mixed_blocked_scene_partitions_static_cylinders_from_center_glass():
    scene = [
        {**_glass("left"), "movable": False},
        _glass("center"),
        {**_glass("right"), "movable": False},
    ]
    obstacles, movables = _partition_scene(scene)
    assert [row["name"] for row in obstacles] == ["left", "right"]
    assert [row["name"] for row in movables] == ["center"]
    xml = inject_obstacles_xml("<mujoco><worldbody></worldbody></mujoco>", obstacles)
    assert 'type="cylinder" size="0.03 0.06"' in xml
    assert "freejoint" not in xml


def test_source_state_splits_are_disjoint_and_stratified():
    layout = _state_layout(50, train_states=30, validation_states=8, heldout_states=12)
    sets = {split: set(values) for split, values in layout.items()}
    assert {split: len(values) for split, values in sets.items()} == {
        "train": 30, "validation": 8, "heldout": 12,
    }
    assert not (sets["train"] & sets["validation"])
    assert not (sets["train"] & sets["heldout"])
    assert not (sets["validation"] & sets["heldout"])
    assert min(sets["heldout"]) == 0 and max(sets["heldout"]) >= 40


def test_successful_source_trace_manifest_and_path_sampling_are_fail_closed(tmp_path):
    eef = np.asarray([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=np.float64)
    bodies = np.asarray([
        [[-0.5, -0.5, 1.0], [-0.4, -0.5, 1.0]],
        [[0.0, 0.5, 1.0], [0.0, 0.6, 1.0]],
        [[0.5, 1.0, 1.0], [0.6, 1.0, 1.0]],
    ], dtype=np.float64)
    actions = np.zeros((3, 7), dtype=np.float32)
    paths = {}
    for name, value in (("eef", eef), ("bodies", bodies), ("actions", actions)):
        path = tmp_path / f"{name}.npy"
        np.save(path, value)
        paths[name] = path
    manifest = tmp_path / "traces.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "kind": "glass_recovery_nominal_source_traces",
        "traces": [{
            "task_suite": "libero_spatial",
            "task_id": 0,
            "source_state_index": 7,
            "source_state_sha256": "s" * 64,
            "task_succeeded": True,
            "eef_xyz_path": paths["eef"].name,
            "eef_xyz_sha256": _file_sha256(paths["eef"]),
            "robot_body_xyz_path": paths["bodies"].name,
            "robot_body_xyz_sha256": _file_sha256(paths["bodies"]),
            "actions_path": paths["actions"].name,
            "actions_sha256": _file_sha256(paths["actions"]),
        }],
    }) + "\n")
    traces, payload = _load_source_traces(
        manifest, suite="libero_spatial", task_id=0
    )
    assert payload["kind"] == "glass_recovery_nominal_source_traces"
    assert set(traces) == {7}
    anchor, tangent, fraction = _sample_trace_anchor(
        traces[7]["_eef_xyz"], 0.75, np.asarray([2.0, 2.0]), 0.1
    )
    assert anchor == pytest.approx([0.5, 1.0])
    assert tangent == pytest.approx([1.0, 0.0])
    assert fraction == pytest.approx(0.75)
    assert _minimum_initial_body_clearance(
        anchor, traces[7]["_robot_body_xyz"], 0.03
    ) > 1.0
    layout = _state_layout_from_indices([2, 7, 11], 1, 1, 1)
    assert sorted(value for values in layout.values() for value in values) == [2, 7, 11]

    payload = json.loads(manifest.read_text())
    payload["traces"][0]["actions_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="actions file/hash mismatch"):
        _load_source_traces(manifest, suite="libero_spatial", task_id=0)


def test_candidate_order_is_predeclared_not_high_fraction_first():
    low = GlassPlacement(**{
        **_placement("low", "train", "state-low", "train/family").to_dict(),
        "nominal_fraction": 0.45,
        "metadata": {"candidate_order_index": 0},
    })
    high = GlassPlacement(**{
        **_placement("high", "train", "state-high", "train/family").to_dict(),
        "nominal_fraction": 0.70,
        "metadata": {"candidate_order_index": 1},
    })
    assert [row.placement_id for row in _ordered_placements([high, low])] == [
        "low", "high"
    ]


def test_salvage_inventory_is_read_only_append_only_and_never_promotes_v1(tmp_path):
    source = tmp_path / "historical_e14"
    pair_root = source / "dataset" / "train" / "pair_1"
    pair_root.mkdir(parents=True)
    np.save(pair_root / "precrash_onpath_state.npy", np.asarray([1.0, 2.0]))
    np.savez_compressed(pair_root / "controller_state.npz", goal=np.asarray([3.0]))
    for name in ("nominal_catastrophe", "oracle_recovery", "off_path_control"):
        np.savez_compressed(pair_root / f"{name}.npz", value=np.asarray([1.0]))
    attempt_key = "a" * 64
    common = {
        "pair_id": "pair_1",
        "placement_id": "placement_1",
        "split": "train",
        "source_state_sha256": "b" * 64,
        "scene_sha256": "c" * 64,
        "branch_start_state_sha256": "d" * 64,
        "n_steps": 25,
    }
    records = [
        {
            **common,
            "trajectory_kind": "nominal_catastrophe",
            "arrays_path": "train/pair_1/nominal_catastrophe.npz",
            "metadata": {
                "attempt_key": attempt_key,
                "controller_state_sha256": "e" * 64,
                "action_replay_evidence": {"verified": True, "n_actions": 25},
            },
        },
        {
            **common,
            "trajectory_kind": "oracle_recovery",
            "arrays_path": "train/pair_1/oracle_recovery.npz",
            "metadata": {
                "attempt_key": attempt_key,
                "oracle_verification": {
                    "independent_recapture": True,
                    "recapture_success": True,
                },
            },
        },
        {
            **common,
            "trajectory_kind": "off_path_control",
            "arrays_path": "train/pair_1/off_path_control.npz",
            "succeeded": True,
            "outcome": "task_success",
            "metadata": {"attempt_key": attempt_key},
        },
    ]
    (pair_root / "pair.json").write_text(json.dumps({
        "attempt_identity": {
            "attempt_key": attempt_key,
            "code_commit": "1" * 40,
            "checkpoint_revision": "2" * 40,
            "rollout_seed": 0,
            "protocol_sha256": "3" * 64,
        },
        "records": records,
    }) + "\n")
    before = {
        path.relative_to(source).as_posix(): _file_sha256(path)
        for path in source.rglob("*") if path.is_file()
    }
    audit = tmp_path / "pilot_a" / "core_salvage_audit.jsonl"
    summary = tmp_path / "pilot_a" / "h_realignment_summary.json"
    args = Namespace(
        source_root=str(source),
        output=str(audit),
        summary_out=str(summary),
        target_h=20,
        expected_run_commit="1" * 40,
        expected_checkpoint_revision="2" * 40,
        historical_summary=None,
        read_only=True,
        overwrite_summary=False,
    )
    result = run_audit(args)
    assert result["attempts"] == 1
    assert result["static_recollection_candidates"] == 1
    row = json.loads(audit.read_text())
    assert row["realignment"]["advance_captured_actions"] == 5
    assert row["realignment"]["direct_v2_promotion_allowed"] is False
    first_audit = audit.read_bytes()
    args.overwrite_summary = True
    rerun = run_audit(args)
    assert rerun["newly_appended_attempts"] == 0
    assert audit.read_bytes() == first_audit
    after = {
        path.relative_to(source).as_posix(): _file_sha256(path)
        for path in source.rglob("*") if path.is_file()
    }
    assert after == before


def test_salvage_inventory_fails_closed_when_historical_retries_lost_identity(tmp_path):
    source = tmp_path / "historical_e14"
    pair_root = source / "dataset" / "train" / "pair_1"
    pair_root.mkdir(parents=True)
    np.save(pair_root / "precrash_onpath_state.npy", np.asarray([1.0]))
    np.savez_compressed(pair_root / "controller_state.npz", goal=np.asarray([1.0]))
    for name in ("nominal_catastrophe", "oracle_recovery", "off_path_control"):
        np.savez_compressed(pair_root / f"{name}.npz", value=np.asarray([1.0]))
    records = [
        {
            "pair_id": "pair_1", "placement_id": "accepted", "split": "train",
            "trajectory_kind": kind, "n_steps": 20,
            "arrays_path": f"train/pair_1/{kind}.npz",
            "succeeded": kind == "off_path_control",
            "metadata": {
                "action_replay_evidence": {"verified": True},
                "oracle_verification": (
                    {"independent_recapture": True, "recapture_success": True}
                    if kind == "oracle_recovery" else {}
                ),
            },
        }
        for kind in ("nominal_catastrophe", "oracle_recovery", "off_path_control")
    ]
    (pair_root / "pair.json").write_text(json.dumps({"records": records}) + "\n")
    (source / "dataset" / "collection_summary.json").write_text(json.dumps({
        "metadata": {
            "code_commit": "1" * 40,
            "checkpoint_identity": {"resolved_revision": "2" * 40},
            "rejected": [{
                "placement_id": "rejected", "split": "train",
                "reason": "no_base_crash", "error": "rejected",
            }],
        },
    }) + "\n")
    historical_summary = tmp_path / "tracked_e14.json"
    historical_summary.write_text(json.dumps({
        "aggregate_attempt_accounting": {
            "candidate_rollout_attempts": 3,
            "rejected_attempts": 2,
            "accepted_admissions": 1,
            "accounting_note": "resume reran previously rejected candidates",
        },
        "limitations": ["The run contains retries."],
    }) + "\n")
    args = Namespace(
        source_root=str(source),
        output=str(tmp_path / "pilot_a" / "core_salvage_audit.jsonl"),
        summary_out=str(tmp_path / "pilot_a" / "h_realignment_summary.json"),
        target_h=20,
        expected_run_commit="1" * 40,
        expected_checkpoint_revision="2" * 40,
        historical_summary=str(historical_summary),
        read_only=True,
        overwrite_summary=False,
    )
    result = run_audit(args)
    assert result["attempts"] == 2
    assert result["historical_attempt_accounting"][
        "unrecoverable_attempt_identity_lower_bound"
    ] == 1
    assert result["historical_attempt_accounting"][
        "attempts_have_unique_provenance"
    ] is False
    assert result["pilot_a_decision"]["go"] is False
    assert result["pilot_a_decision"]["gpu_realignment_submitted"] is False
    assert result["replayability_status"] == (
        "blocked_by_historical_attempt_provenance_no_go"
    )


def test_avoidability_frontier_prefers_h20_and_requires_complete_grid():
    rows = []
    for horizon in FRONTIER_HORIZONS:
        for index in range(10):
            successes = 6 if horizon == 20 else 8 if horizon == 30 else 0
            rows.append({
                "placement_id": f"p{index}",
                "horizon_actions": horizon,
                "base_catastrophe": True,
                "exact_h_replay_verified": True,
                "oracle_safe_task_success": index < successes,
                "rejection_reason": None if index < successes else "no_oracle_recovery",
            })
    summary = summarize_frontier_rows(rows, min_safe_task_success_rate=0.5)
    assert summary["qualified_horizons"] == [30, 20]
    assert summary["recommended_horizon_actions"] == 20
    assert summary["go"] is True
    with pytest.raises(ValueError, match="does not cover every candidate/H"):
        summarize_frontier_rows(rows[:-1], min_safe_task_success_rate=0.5)


def test_recovery_metrics_do_not_reward_always_stop():
    rows = [
        {"regime": "treatment", "succeeded": True, "crashed": False, "safe_abort": False,
         "peak_glass_force_n": 0.0, "intervened": True},
        {"regime": "treatment", "succeeded": False, "crashed": False, "safe_abort": True,
         "peak_glass_force_n": 2.0, "intervened": True},
        {"regime": "blocked", "succeeded": False, "crashed": False, "safe_abort": True,
         "peak_glass_force_n": 1.0, "intervened": True},
        {"regime": "control", "succeeded": True, "crashed": False, "safe_abort": False,
         "peak_glass_force_n": 0.0, "intervened": False},
    ]
    summary = summarize_recovery_rows(rows)
    assert summary.safe_task_success == 0.5
    assert summary.catastrophe_rate == 0.0
    assert summary.treatment_safe_abort_rate == 0.5
    assert summary.false_intervention_on_clean_controls == 0.0
    assert summary.impact_force_worst_case_n == 2.0
    assert summary.blocked_safe_abort_rate == 1.0
    assert summary.blocked_impact_force_worst_case_n == 1.0


def test_replay_selects_accepted_pair_from_manifest_not_first_directory(tmp_path):
    dataset = tmp_path / "dataset"
    rejected = dataset / "train" / "glass_recovery_train_0001"
    accepted = dataset / "train" / "glass_recovery_train_0002"
    rejected.mkdir(parents=True)
    accepted.mkdir()
    records = _v2_records(
        pair_id="glass_recovery_train_0002", horizon=1, include_blocked=True
    )
    write_trajectory_manifest(dataset / "train.jsonl", records)
    (accepted / "pair.json").write_text("{}\n")

    assert accepted_pair_dir_from_manifest(dataset / "train.jsonl") == accepted.resolve()


def test_replay_resolves_attempt_scoped_pair_directory(tmp_path):
    dataset = tmp_path / "dataset"
    pair_id = "glass_recovery_train_0002"
    attempt_key = "a" * 64
    pair_dir = dataset / "train" / f"{pair_id}__attempt_{attempt_key}"
    pair_dir.mkdir(parents=True)
    records = []
    for record in _v2_records(pair_id=pair_id, horizon=1):
        payload = record.to_dict()
        payload["arrays_path"] = (
            Path("train") / pair_dir.name / Path(record.arrays_path).name
        ).as_posix()
        records.append(PairedTrajectoryRecord(**payload))
    write_trajectory_manifest(dataset / "train.jsonl", records)
    (pair_dir / "pair.json").write_text("{}\n")
    assert accepted_pair_dir_from_manifest(dataset / "train.jsonl") == pair_dir.resolve()


def _write_tiny_split(root: Path, split: str, *, pair_count: int = 1, horizon: int = 5):
    records = []
    for pair_number in range(pair_count):
        prefix = f"{split}_{pair_number}"
        for kind_index, kind in enumerate(PRIMARY_TRAJECTORY_KINDS):
            n = horizon
            rows = _trajectory_rows(n)
            for row in rows:
                row["hidden"] = np.full(6, pair_number + kind_index, dtype=np.float32)
                row["target_action"] = np.full(
                    7, 0.2 if kind == "oracle_recovery" else 0, dtype=np.float32
                )
            if kind == "nominal_catastrophe":
                arrays = _finalize_arrays(rows, kind=kind, collision_step=n - 1)
            elif kind == "off_path_control":
                arrays = _finalize_arrays(rows, kind=kind)
            else:
                arrays = _finalize_arrays(
                    rows, kind=kind, counterfactual_collision_step=n - 1
                )
            array_rel = Path(split) / prefix / f"{kind}.npz"
            (root / array_rel).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(root / array_rel, **arrays)
            start_state = (
                f"{prefix}_onpath"
                if kind in {"nominal_catastrophe", "oracle_recovery"}
                else f"{prefix}_{kind}"
            )
            metadata = {
                "controller_state_sha256": f"{prefix}_controller",
                "attempt_key": f"{prefix}_attempt",
            }
            metadata["branch_start_hashes"] = {
                "simulator_state_sha256": start_state,
                "controller_state_sha256": f"{prefix}_controller",
                "observation_sha256": (
                    f"{prefix}_onpath_observation"
                    if kind in {"nominal_catastrophe", "oracle_recovery"}
                    else f"{prefix}_{kind}_observation"
                ),
            }
            if kind == "nominal_catastrophe":
                metadata.update({
                    "row_zero_label_sha256": f"{prefix}_row_zero_labels",
                    "time_to_catastrophe_actions": n,
                    "source_scan_precrash_index": 0,
                    "source_scan_collision_step": n - 1,
                    "source_scan_anchor_state_sha256": start_state,
                    "source_scan_anchor_controller_state_sha256": f"{prefix}_controller",
                    "action_replay_evidence": {
                        "verified": True,
                        "n_actions": n,
                        "expected_catastrophe_action_index": n - 1,
                        "actual_catastrophe_action_index": n - 1,
                    },
                })
            elif kind == "oracle_recovery":
                oracle_config = {"side": -1.0, "lane_margin": 0.12}
                metadata.update({
                    "row_zero_label_sha256": f"{prefix}_row_zero_labels",
                    "time_to_catastrophe_actions": n,
                    "oracle_recoverable_from_this_state": True,
                    "oracle_verified_mask": True,
                    "latest_verified_recoverable_state": f"{prefix}_onpath",
                    "runtime_trigger_eligible": True,
                    "oracle_config": oracle_config,
                    "oracle_verification": {
                        "search_success": True,
                        "search_successful_config_sha256": canonical_sha256(oracle_config),
                        "independent_recapture": True,
                        "recapture_success": True,
                    },
                })
            else:
                metadata["termination"] = "task_success"
            records.append(PairedTrajectoryRecord(
                pair_id=f"{prefix}_pair", placement_id=f"{prefix}_placement", split=split,
                trajectory_kind=kind, source_state_sha256=f"{prefix}_source",
                matched_robot_state_sha256=f"{prefix}_robot",
                branch_start_state_sha256=start_state,
                arrays_path=array_rel.as_posix(), instruction="pick", n_steps=n,
                task_suite="libero_spatial", task_id=0,
                trigger_horizon_actions=n, schema_version=SCHEMA_VERSION,
                outcome=("crash" if kind == "nominal_catastrophe" else
                         "recovery_success" if kind == "oracle_recovery" else
                         "task_success"),
                crashed=kind == "nominal_catastrophe",
                succeeded=kind in {"oracle_recovery", "off_path_control"},
                safe_abort=False,
                oracle_verified=kind == "oracle_recovery",
                scene_sha256=(
                    f"{prefix}_onpath_scene" if kind in {
                        "nominal_catastrophe", "oracle_recovery"
                    } else f"{prefix}_{kind}_scene"
                ),
                metadata=metadata,
            ))
    write_trajectory_manifest(root / f"{split}.jsonl", records)
    protocol = {
        "name": "glass_recovery_primary_v2_test",
        "schema_version": SCHEMA_VERSION,
        "precrash_horizon_actions": horizon,
        "policy": {
            "checkpoint": "fake/openvla",
            "unnorm_key": "libero_spatial",
        },
    }
    (root / "collection_summary.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "checkpoint_identity": {
                "requested_revision": "a" * 40,
                "resolved_revision": "a" * 40,
            },
            "primary_protocol": protocol,
            "primary_protocol_sha256": canonical_sha256(protocol),
        },
    }, indent=2) + "\n")


def test_pair_uniform_phase_sampler_has_fixed_exact_and_first_k_quotas(tmp_path):
    _write_tiny_split(tmp_path, "train", pair_count=2)
    dataset = PairedFrameDataset(tmp_path / "train.jsonl", first_k=2)
    sampler = PairPhaseBatchSampler(dataset, batch_size=16, seed=7)
    pair_phase_counts = Counter()
    for batch in sampler:
        phases = [dataset.item_phases[index] for index in batch]
        assert Counter(phases) == {
            "nominal_far_safe": 4,
            "nominal_imminent": 2,
            "exact_trigger": 2,
            "oracle_first_k": 2,
            "oracle_continuation": 2,
            "off_path_control": 4,
        }
        for index in batch:
            record_index, _ = dataset.index[index]
            pair_phase_counts[(
                dataset.records[record_index].pair_id,
                dataset.item_phases[index],
            )] += 1
    for phase in PHASES:
        counts = [pair_phase_counts[(pair_id, phase)] for pair_id in dataset.pair_ids]
        assert max(counts) - min(counts) <= 1


def test_main_training_dataset_rejects_blocked_appendix_rows(tmp_path):
    write_trajectory_manifest(
        tmp_path / "train.jsonl", _v2_records(include_blocked=True)
    )
    with pytest.raises(ValueError, match="excludes auxiliary trajectories"):
        PairedFrameDataset(tmp_path / "train.jsonl")


def test_tiny_training_pipeline_runs_end_to_end():
    from scripts.train_glass_recovery import train

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_tiny_split(root, "train")
        _write_tiny_split(root, "validation")
        args = Namespace(
            train_manifest=str(root / "train.jsonl"),
            validation_manifest=str(root / "validation.jsonl"),
            output=str(root / "checkpoint"), device="cpu", max_steps=2, batch_size=8,
            learning_rate=3e-4, weight_decay=0.0, max_grad_norm=1.0,
            width=8, depth=1, dropout=0.0,
            sensitivity_margin=0.2, lambda_recovery=1.0, lambda_risk=1.0,
            lambda_hazard=0.0, lambda_severity=0.0, lambda_abort=0.0,
            lambda_invariance=0.5, lambda_sensitivity=0.0,
            gating_horizon=5, first_k=2, max_control_episode_fpr=0.5,
            exit_threshold_ratio=0.5,
            abort_threshold=0.6, eval_every=1, log_every=1,
            num_workers=0, cache_size=2, seed=17, overwrite=False,
        )
        train(args)
        model, metadata = GlassRecoveryNetwork.load_checkpoint(
            root / "checkpoint" / "glass_recovery.pt"
        )
        assert model.config.hidden_dim == 6
        assert metadata["heldout_used_for_training_or_calibration"] is False
        assert metadata["calibration"]["horizon"] == 5
        assert metadata["calibration"]["calibration_unit"] == (
            "control_episode_max_vs_certified_trigger_frame"
        )
        assert metadata["loss_weights"] == metadata["validation_loss_weights"]
        assert metadata["disabled_auxiliary_heads"] == list(
            PRIMARY_DISABLED_AUXILIARY_HEADS
        )
        best, best_metadata = GlassRecoveryNetwork.load_checkpoint(
            root / "checkpoint" / "best.pt"
        )
        assert state_dict_sha256(model) == state_dict_sha256(best)
        assert metadata["published_from"] == "best.pt"
        assert metadata["published_model_state_sha256"] == best_metadata[
            "model_state_sha256"
        ]
        for key in (
            "base_resolved_revision", "train_manifest_sha256",
            "validation_manifest_sha256", "trajectory_schema_version",
            "protocol_sha256", "trigger_horizon_actions", "tte_definition",
        ):
            assert key in metadata
