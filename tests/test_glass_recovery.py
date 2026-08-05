from __future__ import annotations

import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from crashbench.envs.libero_adapter import inject_obstacles_xml
from crashbench.glass_recovery_data import (
    HAZARD_TYPES,
    RISK_HORIZONS,
    GlassPlacement,
    PairedTrajectoryRecord,
    array_sha256,
    validate_episode_arrays,
    validate_paired_records,
    validate_placement_design,
    write_trajectory_manifest,
)
from crashbench.glass_recovery_model import (
    GlassRecoveryConfig,
    GlassRecoveryNetwork,
    glass_recovery_loss,
)
from crashbench.metrics import summarize_recovery_rows
from crashbench.policies.glass_recovery_policy import GlassRecoveryPolicy
from crashbench.recovery import DetourComplete
from scripts.collect_glass_recovery_pairs import (
    _oracle_configs,
    _partition_scene,
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


def test_four_way_pair_requires_exact_nominal_oracle_state():
    common = dict(
        pair_id="pair_0", placement_id="placement_0", split="train",
        source_state_sha256="source", matched_robot_state_sha256="robot",
        arrays_path="train/pair/branch.npz", instruction="pick", n_steps=2,
        scene_sha256="scene",
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
    changed = [records[0], PairedTrajectoryRecord(
        **{**records[1].to_dict(), "branch_start_state_sha256": "different"}
    ), *records[2:]]
    with pytest.raises(ValueError, match="not exact-state matched"):
        validate_paired_records(changed)


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
    assert validate_episode_arrays(arrays) == n
    arrays["nominal_action"] = np.zeros((n, 6))
    with pytest.raises(ValueError, match=r"\[N, 7\]"):
        validate_episode_arrays(arrays)


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
    assert summary.safe_abort_rate == 2 / 3
    assert summary.false_intervention_on_clean_controls == 0.0
    assert summary.impact_force_worst_case_n == 2.0


def test_replay_selects_accepted_pair_from_manifest_not_first_directory(tmp_path):
    dataset = tmp_path / "dataset"
    rejected = dataset / "train" / "glass_recovery_train_0001"
    accepted = dataset / "train" / "glass_recovery_train_0002"
    rejected.mkdir(parents=True)
    accepted.mkdir()
    common = dict(
        pair_id="glass_recovery_train_0002", placement_id="placement", split="train",
        source_state_sha256="source", matched_robot_state_sha256="robot",
        instruction="pick", n_steps=1, scene_sha256="scene",
    )
    records = [
        PairedTrajectoryRecord(
            **common, trajectory_kind="nominal_catastrophe",
            branch_start_state_sha256="onpath",
            arrays_path="train/glass_recovery_train_0002/nominal_catastrophe.npz",
            outcome="crash", crashed=True, succeeded=False, safe_abort=False,
            oracle_verified=False,
        ),
        PairedTrajectoryRecord(
            **common, trajectory_kind="oracle_recovery",
            branch_start_state_sha256="onpath",
            arrays_path="train/glass_recovery_train_0002/oracle_recovery.npz",
            outcome="recovery_success", crashed=False, succeeded=True, safe_abort=False,
            oracle_verified=True,
        ),
        PairedTrajectoryRecord(
            **common, trajectory_kind="off_path_control",
            branch_start_state_sha256="offpath",
            arrays_path="train/glass_recovery_train_0002/off_path_control.npz",
            outcome="recovery_success", crashed=False, succeeded=True, safe_abort=False,
            oracle_verified=False,
        ),
        PairedTrajectoryRecord(
            **common, trajectory_kind="blocked_safe_abort",
            branch_start_state_sha256="blocked",
            arrays_path="train/glass_recovery_train_0002/blocked_safe_abort.npz",
            outcome="safe_abort", crashed=False, succeeded=False, safe_abort=True,
            oracle_verified=True,
            metadata={"blocked_evidence": {"controller_class": "grid"}},
        ),
    ]
    write_trajectory_manifest(dataset / "train.jsonl", records)
    (accepted / "pair.json").write_text("{}\n")

    assert accepted_pair_dir_from_manifest(dataset / "train.jsonl") == accepted.resolve()


def _write_tiny_split(root: Path, split: str):
    records = []
    for kind_index, kind in enumerate((
        "nominal_catastrophe", "oracle_recovery", "off_path_control", "blocked_safe_abort",
    )):
        n = 2
        risk_positive = kind != "off_path_control"
        arrays = {
            "hidden": np.full((n, 6), kind_index, dtype=np.float16),
            "robot_state": np.zeros((n, 8), dtype=np.float32),
            "nominal_action": np.zeros((n, 7), dtype=np.float32),
            "target_action": np.full((n, 7), 0.2 if kind != "off_path_control" else 0,
                                     dtype=np.float32),
            "executed_action": np.zeros((n, 7), dtype=np.float32),
            "risk_targets": np.full((n, 5), float(risk_positive), dtype=np.float32),
            "risk_mask": np.ones((n, 5), dtype=np.float32),
            "hazard_type": np.full(n, int(risk_positive), dtype=np.int64),
            "severity_force": np.full(n, 20.0 if risk_positive else 0, dtype=np.float32),
            "severity_mask": np.ones(n, dtype=np.float32),
            "abort_target": np.full(n, float(kind == "blocked_safe_abort"), dtype=np.float32),
            "recovery_mask": np.full(n, float(kind in {"oracle_recovery", "blocked_safe_abort"}),
                                     dtype=np.float32),
            "invariance_mask": np.full(n, float(kind == "off_path_control"), dtype=np.float32),
            "sensitivity_mask": np.full(n, float(kind == "oracle_recovery"), dtype=np.float32),
        }
        array_rel = Path(split) / f"{kind}.npz"
        (root / array_rel).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / array_rel, **arrays)
        records.append(PairedTrajectoryRecord(
            pair_id=f"{split}_pair", placement_id=f"{split}_placement", split=split,
            trajectory_kind=kind, source_state_sha256=f"{split}_source",
            matched_robot_state_sha256=f"{split}_robot",
            branch_start_state_sha256=(f"{split}_onpath" if kind in {
                "nominal_catastrophe", "oracle_recovery"} else f"{split}_{kind}"),
            arrays_path=array_rel.as_posix(), instruction="pick", n_steps=n,
            outcome=("crash" if kind == "nominal_catastrophe" else
                     "recovery_success" if kind in {"oracle_recovery", "off_path_control"}
                     else "safe_abort"),
            crashed=kind == "nominal_catastrophe",
            succeeded=kind in {"oracle_recovery", "off_path_control"},
            safe_abort=kind == "blocked_safe_abort",
            oracle_verified=kind in {"oracle_recovery", "blocked_safe_abort"},
            scene_sha256=f"scene_{kind}",
            metadata=({"blocked_evidence": {"controller_class": "grid"}}
                      if kind == "blocked_safe_abort" else {}),
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
            width=8, depth=1, dropout=0.0, max_action_residual=1.0,
            sensitivity_margin=0.2, lambda_recovery=1.0, lambda_risk=1.0,
            lambda_hazard=0.25, lambda_severity=0.25, lambda_abort=0.5,
            lambda_invariance=0.5, lambda_sensitivity=0.25,
            gating_horizon=10, max_control_fpr=0.5, exit_threshold_ratio=0.5,
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
