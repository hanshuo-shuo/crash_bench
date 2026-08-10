from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import scripts.collect_glass_recovery_pairs as glass_collector

from crashbench.envs.libero_adapter import LiberoEnv, inject_obstacles_xml
from crashbench.glass_recovery_data import (
    HAZARD_TYPES,
    RISK_HORIZONS,
    SCHEMA_VERSION,
    GlassPlacement,
    PairedTrajectoryRecord,
    array_sha256,
    exact_h_anchor_index,
    read_trajectory_manifest,
    steps_until_event,
    validate_episode_arrays,
    validate_paired_records,
    validate_placement_design,
    validate_primary_pair,
    write_trajectory_manifest,
)
from crashbench.glass_recovery_model import (
    GlassRecoveryConfig,
    GlassRecoveryNetwork,
    glass_recovery_loss,
)
from scripts.train_glass_recovery import (
    _episode_max_scores,
    _select_episode_risk_threshold,
)
from crashbench.metrics import summarize_recovery_rows
from crashbench.policies.glass_recovery_policy import GlassRecoveryPolicy
from crashbench.recovery import DetourComplete
from crashbench.predicates import build_predicate, prime_predicate
from crashbench.scenario import PredicateSpec
from scripts.collect_glass_recovery_pairs import (
    CAREFUL_PROMPT_PREFIX,
    _finalize_arrays,
    _oracle_rejection_reason,
    _oracle_configs,
    _partition_scene,
    _replay_nominal_actions,
    _rejection_counts,
    _roll_nominal,
    _safe_abort_configs,
    _stable_abort,
)
from scripts.prepare_glass_recovery_placements import (
    _blocked_barrier_offsets,
    _collision_free_path_fraction,
    _state_layout,
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
                **careful,
                "time_to_catastrophe_actions": horizon,
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
                "time_to_catastrophe_actions": horizon,
                "oracle_recoverable_from_this_state": True,
                "oracle_verified_mask": True,
                "latest_verified_recoverable_state": f"{pair_id}_onpath",
                "runtime_trigger_eligible": True,
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
            metadata={**controller, "termination": "task_success"},
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
    assert np.array_equal(oracle["risk_targets"][0], nominal["risk_targets"][0])
    assert np.array_equal(oracle["risk_mask"][0], nominal["risk_mask"][0])
    assert oracle["time_to_catastrophe_actions"][0] == horizon


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


def test_episode_max_threshold_calibrates_false_interventions_by_episode():
    frame_scores = np.asarray([0.01, 0.10, 0.20, 0.30, 0.80, 0.90])
    valid = np.ones(6, dtype=bool)
    episode_ids = np.asarray([0, 0, 1, 1, 2, 2])
    kinds = np.asarray([2, 2, 2, 2, 0, 0])
    controls = _episode_max_scores(frame_scores, valid, episode_ids, kinds, 2)
    catastrophes = _episode_max_scores(frame_scores, valid, episode_ids, kinds, 0)
    calibration = _select_episode_risk_threshold(controls, catastrophes, 0.0)
    assert np.array_equal(controls, [0.10, 0.30])
    assert np.array_equal(catastrophes, [0.90])
    assert calibration["threshold"] == pytest.approx(0.9)
    assert calibration["control_episode_fpr"] == 0.0
    assert calibration["catastrophe_episode_tpr"] == 1.0
    assert calibration["calibration_unit"] == "episode_max_risk"


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

    def __init__(self):
        self.last_hidden = None

    def act(self, observation, instruction):
        self.last_hidden = np.ones(4, dtype=np.float32)
        return np.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def _constant_checkpoint(path: Path, risk_bias: float, abort_bias: float):
    model = GlassRecoveryNetwork(GlassRecoveryConfig(
        hidden_dim=4, width=8, depth=1, dropout=0.0,
    ))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    model.risk_head.bias.data.fill_(risk_bias)
    model.abort_head.bias.data.fill_(abort_bias)
    model.action_head.bias.data.fill_(0.5)
    model.save_checkpoint(path)


def test_runtime_gate_nominal_recovery_and_abort():
    observation = {"state": np.asarray([0.2, 0.0, 0.9, 0, 0, 0, 0, 0], dtype=np.float32)}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        low = root / "low.pt"; recover = root / "recover.pt"; abort = root / "abort.pt"
        _constant_checkpoint(low, -10.0, -10.0)
        _constant_checkpoint(recover, 10.0, -10.0)
        _constant_checkpoint(abort, 10.0, 10.0)
        nominal_policy = GlassRecoveryPolicy(_FakeBase(), str(low), device="cpu")
        assert np.allclose(nominal_policy.act(observation, "pick")[:3], [0.1, 0.0, 0.0])
        assert nominal_policy.mode == "nominal"
        recovery_policy = GlassRecoveryPolicy(_FakeBase(), str(recover), device="cpu")
        recovery_action = recovery_policy.act(observation, "pick")
        assert recovery_policy.mode == "recovery"
        assert not np.allclose(recovery_action, [0.1, 0, 0, 0, 0, 0, -1])
        abort_policy = GlassRecoveryPolicy(_FakeBase(), str(abort), device="cpu")
        abort_policy.act(observation, "pick")
        assert abort_policy.mode == "abort"


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
        {"reason": "careful_did_not_crash"},
        {},
    ]) == {
        "no_base_crash": 2,
        "no_oracle_recovery": 0,
        "oracle_collision": 0,
        "oracle_task_failure": 0,
        "off_path_timeout": 0,
        "careful_did_not_crash": 1,
        "invalid_initial_state": 0,
        "other": 1,
    }


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


def _write_tiny_split(root: Path, split: str):
    records = []
    for kind_index, kind in enumerate((
        "nominal_catastrophe", "oracle_recovery", "off_path_control", "blocked_safe_abort",
    )):
        n = 2
        rows = _trajectory_rows(n)
        for row in rows:
            row["hidden"] = np.full(6, kind_index, dtype=np.float32)
            row["target_action"] = np.full(
                7, 0.2 if kind != "off_path_control" else 0, dtype=np.float32
            )
        if kind == "nominal_catastrophe":
            arrays = _finalize_arrays(rows, kind=kind, collision_step=n - 1)
        elif kind == "off_path_control":
            arrays = _finalize_arrays(rows, kind=kind)
        else:
            arrays = _finalize_arrays(
                rows, kind=kind, counterfactual_collision_step=n - 1
            )
        array_rel = Path(split) / f"{kind}.npz"
        (root / array_rel).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / array_rel, **arrays)
        metadata = {"controller_state_sha256": f"{split}_controller"}
        if kind == "nominal_catastrophe":
            metadata.update({
                "time_to_catastrophe_actions": n,
                "action_replay_evidence": {
                    "verified": True,
                    "n_actions": n,
                    "expected_catastrophe_action_index": n - 1,
                    "actual_catastrophe_action_index": n - 1,
                },
            })
        elif kind == "oracle_recovery":
            metadata.update({
                "time_to_catastrophe_actions": n,
                "oracle_recoverable_from_this_state": True,
                "oracle_verified_mask": True,
                "latest_verified_recoverable_state": f"{split}_onpath",
                "runtime_trigger_eligible": True,
            })
        elif kind == "off_path_control":
            metadata["termination"] = "task_success"
        else:
            metadata["blocked_evidence"] = {"controller_class": "grid"}
        records.append(PairedTrajectoryRecord(
            pair_id=f"{split}_pair", placement_id=f"{split}_placement", split=split,
            trajectory_kind=kind, source_state_sha256=f"{split}_source",
            matched_robot_state_sha256=f"{split}_robot",
            branch_start_state_sha256=(f"{split}_onpath" if kind in {
                "nominal_catastrophe", "oracle_recovery"} else f"{split}_{kind}"),
            arrays_path=array_rel.as_posix(), instruction="pick", n_steps=n,
            task_suite="libero_spatial", task_id=0,
            trigger_horizon_actions=n, schema_version=SCHEMA_VERSION,
            outcome=("crash" if kind == "nominal_catastrophe" else
                     "recovery_success" if kind == "oracle_recovery" else
                     "task_success" if kind == "off_path_control" else "safe_abort"),
            crashed=kind == "nominal_catastrophe",
            succeeded=kind in {"oracle_recovery", "off_path_control"},
            safe_abort=kind == "blocked_safe_abort",
            oracle_verified=kind in {"oracle_recovery", "blocked_safe_abort"},
            scene_sha256=(
                f"{split}_onpath_scene" if kind in {
                    "nominal_catastrophe", "oracle_recovery"
                } else f"{split}_{kind}_scene"
            ),
            metadata=metadata,
        ))
    write_trajectory_manifest(root / f"{split}.jsonl", records)


def test_tiny_training_pipeline_runs_end_to_end():
    from scripts.train_glass_recovery import train

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_tiny_split(root, "train")
        _write_tiny_split(root, "validation")
        args = Namespace(
            train_manifest=str(root / "train.jsonl"),
            validation_manifest=str(root / "validation.jsonl"),
            output=str(root / "checkpoint"), device="cpu", max_steps=2, batch_size=4,
            learning_rate=3e-4, weight_decay=0.0, max_grad_norm=1.0,
            width=8, depth=1, dropout=0.0,
            sensitivity_margin=0.2, lambda_recovery=1.0, lambda_risk=1.0,
            lambda_hazard=0.25, lambda_severity=0.25, lambda_abort=0.5,
            lambda_invariance=0.5, lambda_sensitivity=0.25,
            gating_horizon=10, max_control_episode_fpr=0.5, exit_threshold_ratio=0.5,
            abort_threshold=0.6, eval_every=1, log_every=1,
            num_workers=0, cache_size=2, seed=17, overwrite=False,
        )
        train(args)
        model, metadata = GlassRecoveryNetwork.load_checkpoint(
            root / "checkpoint" / "glass_recovery.pt"
        )
        assert model.config.hidden_dim == 6
        assert metadata["heldout_used_for_training_or_calibration"] is False
        assert metadata["calibration"]["horizon"] == 10
        assert metadata["calibration"]["calibration_unit"] == "episode_max_risk"
