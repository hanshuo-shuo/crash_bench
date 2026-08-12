from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from crashbench.glass_recovery_data import (
    HAZARD_TYPES,
    RISK_HORIZONS,
    SCHEMA_VERSION,
    GlassPlacement,
    PairedTrajectoryRecord,
    array_sha256,
    canonical_sha256,
    write_placement_manifest,
    write_trajectory_manifest,
)
from crashbench.glass_recovery_model import (
    HIDDEN_HOOK_IDENTITY,
    PRIMARY_CHECKPOINT_KIND,
    PRIMARY_DISABLED_AUXILIARY_HEADS,
    PRIMARY_DISABLED_AUXILIARY_TERMS,
    TTE_DEFINITION,
    GlassLossWeights,
    GlassRecoveryNetwork,
)
from crashbench.policies.glass_recovery_policy import GlassRecoveryPolicy
from crashbench.metrics import analyze_recovery_evaluation
from scripts.audit_repo import (
    E15_MAIN_BASELINE_CONDITIONS,
    learned_recovery_semantic_errors,
)
from scripts.collect_glass_recovery_pairs import (
    _controller_state_sha256,
    _observation_sha256,
)
from scripts.eval_glass_recovery import (
    EVALUATION_MODES,
    MAIN_CONDITIONS,
    NONPRIVILEGED_CONDITIONS,
    PRIVILEGED_CONDITIONS,
    _restore_identity_evidence,
    file_sha256,
    load_evaluation_contract,
    run_evaluation_episode,
)
from scripts.summarize_glass_pilot_cdef import (
    _pilot_c,
    _pilot_d,
    _pilot_e,
    _pilot_f,
)


BASE_REVISION = "a" * 40
PROTOCOL_H = 5
CHECKPOINT_SHA256 = "c" * 64


def test_runtime_restore_requires_state_and_controller_but_audits_observation():
    expected = {
        "simulator_state_sha256": "a" * 64,
        "controller_state_sha256": "b" * 64,
        "observation_sha256": "c" * 64,
    }
    evidence = _restore_identity_evidence(
        {**expected, "observation_sha256": "d" * 64},
        expected,
        label="exact anchor",
    )
    assert evidence["simulator_controller_exact"] is True
    assert evidence["observation_exact"] is False
    with pytest.raises(RuntimeError, match="simulator_state_sha256"):
        _restore_identity_evidence(
            {**expected, "simulator_state_sha256": "e" * 64},
            expected,
            label="exact anchor",
        )


def test_pilot_cdef_summaries_apply_frozen_gates():
    def row(mode, condition, regime, **updates):
        return {
            "evaluation_mode": mode,
            "condition": condition,
            "regime": regime,
            "source_state_sha256": updates.pop("source", "s"),
            "task_success": False,
            "catastrophe": False,
            "intervened": False,
            "timely_trigger": False,
            "safe_noncompletion": False,
            **updates,
        }

    c_rows = [row(
        "exact_anchor", "oracle_timed_oracle_recovery", "treatment",
        task_success=True,
        anchor_restore_identity={"simulator_controller_exact": True},
    )]
    assert all(_pilot_c(c_rows)[1].values())

    d_rows = [
        row(
            "source_to_task", "risk_gate_oracle_recovery", "treatment",
            task_success=True, timely_trigger=True, intervened=True,
            first_intervention_step=3, certified_recoverability_deadline_step=5,
        ),
        row("source_to_task", "risk_gate_oracle_recovery", "control", task_success=True),
    ]
    assert all(_pilot_d(d_rows)[1].values())

    e_row = row(
        "exact_anchor", "oracle_timed_learned_recovery", "treatment",
        task_success=True,
    )
    summary = {"validation_metrics": {"gripper_sign_accuracy": 0.96}}
    assert all(_pilot_e([e_row], [e_row], summary)[1].values())

    f_rows = [
        row("source_to_task", "base", "treatment", catastrophe=True),
        row("source_to_task", "full_learned_gate_recovery", "treatment", task_success=True),
        row("source_to_task", "base", "control", task_success=True),
        row("source_to_task", "full_learned_gate_recovery", "control", task_success=True),
    ]
    assert all(_pilot_f(f_rows)[1].values())


def _glass(name: str = "glass_1", x: float = 0.0) -> dict:
    return {
        "name": name,
        "type": "cylinder",
        "pos": [x, 0.0, 0.96],
        "size": [0.03, 0.06],
    }


def _placement(root: Path, identifier: str, split: str, source: np.ndarray) -> GlassPlacement:
    state_path = root / "states" / f"{identifier}.npy"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(state_path, source)
    return GlassPlacement(
        placement_id=identifier,
        split=split,
        cluster_id=f"{split}/wide",
        task_suite="libero_spatial",
        task_id=0,
        instruction="pick and place",
        source_state_path=state_path.relative_to(root).as_posix(),
        source_state_sha256=array_sha256(source),
        on_path_glass=_glass(),
        off_path_glass=_glass(x=0.2),
        blocked_glasses=[],
        nominal_fraction=0.6,
        metadata={
            "geometry_family": "wide",
            "bowl_xyz": [0.1, 0.0, 0.9],
        },
    )


def _arrays(n: int, kind: str, horizon: int) -> dict[str, np.ndarray]:
    values = {
        "hidden": np.zeros((n, 4), dtype=np.float32),
        "robot_state": np.zeros((n, 8), dtype=np.float32),
        "nominal_action": np.zeros((n, 7), dtype=np.float32),
        "target_action": np.zeros((n, 7), dtype=np.float32),
        "executed_action": np.zeros((n, 7), dtype=np.float32),
        "risk_targets": np.zeros((n, len(RISK_HORIZONS)), dtype=np.float32),
        "risk_mask": np.zeros((n, len(RISK_HORIZONS)), dtype=np.float32),
        "hazard_type": np.full(
            n, HAZARD_TYPES.index("none" if kind == "off_path_control" else "glass"),
            dtype=np.int64,
        ),
        "severity_force": np.zeros(n, dtype=np.float32),
        "severity_mask": np.zeros(n, dtype=np.float32),
        "abort_target": np.zeros(n, dtype=np.float32),
        "recovery_mask": np.full(
            n, float(kind == "oracle_recovery"), dtype=np.float32
        ),
        "invariance_mask": np.full(
            n, float(kind == "off_path_control"), dtype=np.float32
        ),
        "sensitivity_mask": np.zeros(n, dtype=np.float32),
        "time_to_catastrophe_actions": np.full(n, -1, dtype=np.int32),
        "time_to_catastrophe_mask": np.zeros(n, dtype=np.float32),
        "oracle_recoverable_from_this_state": np.zeros(n, dtype=np.float32),
        "oracle_verified_mask": np.zeros(n, dtype=np.float32),
        "latest_verified_recoverable_state": np.zeros(n, dtype=np.float32),
        "runtime_trigger_eligible": np.zeros(n, dtype=np.float32),
    }
    if kind == "nominal_catastrophe":
        remaining = np.arange(horizon, 0, -1, dtype=np.int32)
        values["time_to_catastrophe_actions"] = remaining
        values["time_to_catastrophe_mask"][:] = 1
        values["risk_mask"][:] = 1
        values["risk_targets"] = np.asarray([
            [float(1 <= int(tte) <= risk_h) for risk_h in RISK_HORIZONS]
            for tte in remaining
        ], dtype=np.float32)
        values["oracle_recoverable_from_this_state"][0] = 1
        values["oracle_verified_mask"][0] = 1
        values["latest_verified_recoverable_state"][0] = 1
        values["runtime_trigger_eligible"][0] = 1
    elif kind == "oracle_recovery":
        values["time_to_catastrophe_actions"][0] = horizon
        values["time_to_catastrophe_mask"][0] = 1
        values["risk_mask"][0] = 1
        values["risk_targets"][0] = [
            float(1 <= horizon <= risk_h) for risk_h in RISK_HORIZONS
        ]
        values["oracle_recoverable_from_this_state"][0] = 1
        values["oracle_verified_mask"][0] = 1
        values["latest_verified_recoverable_state"][0] = 1
        values["runtime_trigger_eligible"][0] = 1
    elif kind == "off_path_control":
        # This synthetic control terminates in task success, so its observed
        # safe frame is not a right-censored timeout tail.
        values["risk_mask"][:] = 1
    return values


def _records_for_placement(
    root: Path,
    placement: GlassPlacement,
    *,
    write_artifacts: bool,
) -> tuple[list[PairedTrajectoryRecord], list[Path]]:
    pair_root = root / "artifacts" / placement.split / placement.placement_id
    matched = np.asarray([0.0, 10.0, 11.0, 12.0], dtype=np.float64)
    onpath = np.asarray([
        0.0, 10.0, 11.0,
        *placement.on_path_glass["pos"], 1.0, 0.0, 0.0, 0.0,
        12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float64)
    offpath = np.asarray([
        0.0, 10.0, 11.0,
        *placement.off_path_glass["pos"], 1.0, 0.0, 0.0, 0.0,
        12.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float64)
    controller = {"controller.goal": np.asarray([1.0, 2.0], dtype=np.float64)}
    onpath_hash = array_sha256(onpath)
    offpath_hash = array_sha256(offpath)
    controller_hash = _controller_state_sha256(controller)
    attempt = f"attempt-{placement.placement_id}"
    observation_hash = f"observation-{placement.placement_id}"
    onpath_start = {
        "simulator_state_sha256": onpath_hash,
        "controller_state_sha256": controller_hash,
        "observation_sha256": observation_hash,
    }
    offpath_start = {
        "simulator_state_sha256": offpath_hash,
        "controller_state_sha256": controller_hash,
        "observation_sha256": observation_hash,
    }
    oracle_config = {
        "side": -1.0,
        "lane_margin": 0.12,
        "transit_z": 1.2,
        "descend_off": 0.04,
        "orientation_target": None,
        "path_aligned": True,
        "grasp_xy_offset": [0.0, 0.0],
    }
    paths: list[Path] = []
    arrays_by_kind = {
        "nominal_catastrophe": _arrays(PROTOCOL_H, "nominal_catastrophe", PROTOCOL_H),
        "oracle_recovery": _arrays(PROTOCOL_H, "oracle_recovery", PROTOCOL_H),
        "off_path_control": _arrays(1, "off_path_control", PROTOCOL_H),
    }
    if write_artifacts:
        pair_root.mkdir(parents=True, exist_ok=True)
        np.save(pair_root / "precrash_onpath_state.npy", onpath)
        np.save(pair_root / "offpath_start_state.npy", offpath)
        np.save(pair_root / "matched_robot_state.npy", matched)
        np.savez_compressed(pair_root / "controller_state.npz", **controller)
        paths.extend([
            pair_root / "precrash_onpath_state.npy",
            pair_root / "offpath_start_state.npy",
            pair_root / "matched_robot_state.npy",
            pair_root / "controller_state.npz",
        ])
    for kind, arrays in arrays_by_kind.items():
        path = pair_root / f"{kind}.npz"
        if write_artifacts:
            np.savez_compressed(path, **arrays)
            paths.append(path)
    row_zero_hash = "row-zero-shared"
    common = dict(
        pair_id=placement.placement_id,
        placement_id=placement.placement_id,
        split=placement.split,
        source_state_sha256=placement.source_state_sha256,
        matched_robot_state_sha256=array_sha256(matched),
        instruction=placement.instruction,
        scene_sha256=canonical_sha256([placement.on_path_glass]),
        schema_version=SCHEMA_VERSION,
        task_suite=placement.task_suite,
        task_id=placement.task_id,
        trigger_horizon_actions=PROTOCOL_H,
    )
    nominal = PairedTrajectoryRecord(
        **common,
        trajectory_kind="nominal_catastrophe",
        branch_start_state_sha256=onpath_hash,
        arrays_path=(pair_root / "nominal_catastrophe.npz").relative_to(root).as_posix(),
        n_steps=PROTOCOL_H,
        outcome="crash",
        crashed=True,
        succeeded=False,
        safe_abort=False,
        oracle_verified=False,
        metadata={
            "controller_state_sha256": controller_hash,
            "branch_start_hashes": onpath_start,
            "attempt_key": attempt,
            "row_zero_label_sha256": row_zero_hash,
            "time_to_catastrophe_actions": PROTOCOL_H,
            "source_scan_precrash_index": 2,
            "source_scan_collision_step": 6,
            "source_scan_anchor_state_sha256": onpath_hash,
            "source_scan_anchor_controller_state_sha256": controller_hash,
            "action_replay_evidence": {
                "verified": True,
                "n_actions": PROTOCOL_H,
                "expected_catastrophe_action_index": PROTOCOL_H - 1,
                "actual_catastrophe_action_index": PROTOCOL_H - 1,
            },
        },
    )
    oracle = PairedTrajectoryRecord(
        **common,
        trajectory_kind="oracle_recovery",
        branch_start_state_sha256=onpath_hash,
        arrays_path=(pair_root / "oracle_recovery.npz").relative_to(root).as_posix(),
        n_steps=PROTOCOL_H,
        outcome="recovery_success",
        crashed=False,
        succeeded=True,
        safe_abort=False,
        oracle_verified=True,
        metadata={
            "controller_state_sha256": controller_hash,
            "branch_start_hashes": onpath_start,
            "attempt_key": attempt,
            "row_zero_label_sha256": row_zero_hash,
            "time_to_catastrophe_actions": PROTOCOL_H,
            "oracle_recoverable_from_this_state": True,
            "oracle_verified_mask": True,
            "latest_verified_recoverable_state": onpath_hash,
            "runtime_trigger_eligible": True,
            "oracle_config": oracle_config,
            "oracle_verification": {
                "search_success": True,
                "independent_recapture": True,
                "recapture_success": True,
                "search_successful_config_sha256": canonical_sha256(oracle_config),
            },
        },
    )
    control = PairedTrajectoryRecord(
        **{**common, "scene_sha256": canonical_sha256([placement.off_path_glass])},
        trajectory_kind="off_path_control",
        branch_start_state_sha256=offpath_hash,
        arrays_path=(pair_root / "off_path_control.npz").relative_to(root).as_posix(),
        n_steps=1,
        outcome="task_success",
        crashed=False,
        succeeded=True,
        safe_abort=False,
        oracle_verified=False,
        metadata={
            "controller_state_sha256": controller_hash,
            "branch_start_hashes": offpath_start,
            "attempt_key": attempt,
            "termination": "task_success",
        },
    )
    return [nominal, oracle, control], paths


def _primary_protocol() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": {"unnorm_key": "libero_spatial"},
        "precrash_horizon_actions": PROTOCOL_H,
        "settle_steps": 0,
    }


def _checkpoint_metadata(train_sha: str, validation_sha: str) -> dict:
    protocol_sha = canonical_sha256(_primary_protocol())
    return {
        "checkpoint_kind": PRIMARY_CHECKPOINT_KIND,
        "base_resolved_revision": BASE_REVISION,
        "unnorm_key": "libero_spatial",
        "hidden_hook_identity": HIDDEN_HOOK_IDENTITY,
        "train_manifest_sha256": train_sha,
        "validation_manifest_sha256": validation_sha,
        "trajectory_schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol_sha,
        "trigger_horizon_actions": PROTOCOL_H,
        "tte_definition": TTE_DEFINITION,
        "seed": 17,
        "disabled_auxiliary_heads": list(PRIMARY_DISABLED_AUXILIARY_HEADS),
        "disabled_auxiliary_loss_terms": list(PRIMARY_DISABLED_AUXILIARY_TERMS),
        "loss_weights": asdict(GlassLossWeights()),
        "calibration": {
            "horizon": PROTOCOL_H,
            "threshold": 0.5,
            "objective": "maximize_timely_trigger_rate_subject_to_clean_control_episode_fpr",
            "ownership": "recovery_latched_until_terminal_or_reset",
            "abort_enabled": False,
        },
    }


def _build_contract_files(root: Path) -> dict:
    placements = [
        _placement(root, "train-pair", "train", np.asarray([1.0, 2.0])),
        _placement(root, "validation-pair", "validation", np.asarray([3.0, 4.0])),
        _placement(root, "heldout-pair", "heldout", np.asarray([5.0, 6.0])),
    ]
    placement_path = root / "placements.json"
    placement_payload = write_placement_manifest(placement_path, placements)
    records = {}
    heldout_artifacts: list[Path] = []
    for placement in placements:
        records[placement.split], paths = _records_for_placement(
            root, placement, write_artifacts=True
        )
        if placement.split == "heldout":
            heldout_artifacts.extend(paths)
    manifests = {}
    for split in ("train", "validation", "heldout"):
        manifests[split] = root / f"{split}.jsonl"
        write_trajectory_manifest(manifests[split], records[split])
    metadata = _checkpoint_metadata(
        file_sha256(manifests["train"]), file_sha256(manifests["validation"])
    )
    primary = _primary_protocol()
    heldout = placements[-1]
    evaluation = {
        "name": "glass_recovery_eval_p0d_v1",
        "evaluation_modes": list(EVALUATION_MODES),
        "conditions": list(MAIN_CONDITIONS),
        "rollout_seeds": [101, 202],
        "max_steps": 4,
        "checkpoint_sha256_by_training_seed": {"17": CHECKPOINT_SHA256},
        "cohort_contract": {
            "split": "heldout",
            "pair_ids": [heldout.placement_id],
            "placement_manifest_sha256": file_sha256(placement_path),
            "trajectory_manifest_sha256": file_sha256(manifests["heldout"]),
        },
    }
    protocol_payload = {
        "schema_version": 1,
        "kind": "glass_recovery_evaluation_protocol",
        "primary_protocol": primary,
        "primary_protocol_sha256": canonical_sha256(primary),
        "evaluation_protocol": evaluation,
        "evaluation_protocol_sha256": canonical_sha256(evaluation),
    }
    protocol_path = root / "protocol.json"
    protocol_path.write_text(json.dumps(protocol_payload, indent=2) + "\n")
    collection_summary = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "primary_protocol": primary,
            "primary_protocol_sha256": canonical_sha256(primary),
            "placement_design_sha256": canonical_sha256(placement_payload),
            "checkpoint_identity": {"resolved_revision": BASE_REVISION},
        },
    }
    (root / "collection_summary.json").write_text(
        json.dumps(collection_summary, indent=2) + "\n"
    )
    required_files = [root / heldout.source_state_path, *heldout_artifacts]
    file_hashes = {
        path.relative_to(root).as_posix(): file_sha256(path) for path in required_files
    }
    cohort_payload = {
        "schema_version": 1,
        "kind": "glass_recovery_accepted_evaluation_cohort",
        "split": "heldout",
        "placement_manifest_sha256": file_sha256(placement_path),
        "trajectory_manifest_sha256": file_sha256(manifests["heldout"]),
        "primary_protocol_sha256": canonical_sha256(primary),
        "evaluation_protocol_sha256": canonical_sha256(evaluation),
        "pairs": [{
            "pair_id": heldout.placement_id,
            "placement_id": heldout.placement_id,
            "split": heldout.split,
            "task_suite": heldout.task_suite,
            "task_id": heldout.task_id,
            "family": heldout.metadata["geometry_family"],
            "source_state_sha256": heldout.source_state_sha256,
        }],
        "file_sha256": file_hashes,
        "reference_manifests": {
            "train": {"path": "train.jsonl", "sha256": file_sha256(manifests["train"])},
            "validation": {
                "path": "validation.jsonl",
                "sha256": file_sha256(manifests["validation"]),
            },
        },
    }
    cohort_path = root / "cohort.json"
    cohort_path.write_text(json.dumps(cohort_payload, indent=2) + "\n")
    return {
        "placements": placement_path,
        "manifests": manifests,
        "protocol": protocol_path,
        "cohort": cohort_path,
        "cohort_payload": cohort_payload,
        "metadata": metadata,
    }


def _load(files: dict):
    return load_evaluation_contract(
        placement_manifest=files["placements"],
        trajectory_manifest=files["manifests"]["heldout"],
        evaluation_cohort=files["cohort"],
        protocol=files["protocol"],
        checkpoint_metadata=files["metadata"],
        checkpoint_sha256=CHECKPOINT_SHA256,
        base_checkpoint_revision=BASE_REVISION,
        unnorm_key="libero_spatial",
    )


def _refresh_frozen_protocol(files: dict, **cohort_updates) -> None:
    protocol = json.loads(files["protocol"].read_text())
    protocol["evaluation_protocol"]["cohort_contract"].update(cohort_updates)
    protocol["evaluation_protocol_sha256"] = canonical_sha256(
        protocol["evaluation_protocol"]
    )
    files["protocol"].write_text(json.dumps(protocol, indent=2) + "\n")
    cohort = files["cohort_payload"]
    cohort.update({
        key: value for key, value in cohort_updates.items()
        if key in {"placement_manifest_sha256", "trajectory_manifest_sha256"}
    })
    cohort["evaluation_protocol_sha256"] = protocol["evaluation_protocol_sha256"]
    files["cohort"].write_text(json.dumps(cohort, indent=2) + "\n")


def test_accepted_cohort_preflight_loads_only_manifest_accepted_pairs(tmp_path):
    files = _build_contract_files(tmp_path)
    contract = _load(files)
    assert [pair.placement.placement_id for pair in contract.pairs] == ["heldout-pair"]
    assert contract.horizon == PROTOCOL_H
    assert contract.split == "heldout"
    assert set(contract.reference_manifests) == {"train", "validation"}


def test_development_oracle_cohort_can_be_checkpoint_validation_set(tmp_path):
    files = _build_contract_files(tmp_path)
    heldout_sha = file_sha256(files["manifests"]["heldout"])
    files["metadata"]["validation_manifest_sha256"] = heldout_sha
    payload = files["cohort_payload"]
    payload["development_only"] = True
    payload["checkpoint_validation_role"] = "selected_heldout_cohort"
    files["cohort"].write_text(json.dumps(payload) + "\n")

    contract = _load(files)

    assert set(contract.reference_manifests) == {"train"}


def test_frozen_protocol_rejects_a_reselected_cohort(tmp_path):
    files = _build_contract_files(tmp_path)
    payload = files["cohort_payload"]
    payload["pairs"][0]["pair_id"] = "posthoc-reselection"
    files["cohort"].write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="pair selection differs from the frozen protocol"):
        _load(files)


def test_accepted_pair_is_bound_to_authored_scene_geometry(tmp_path):
    files = _build_contract_files(tmp_path)
    manifest = files["manifests"]["heldout"]
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        if row["trajectory_kind"] in {"nominal_catastrophe", "oracle_recovery"}:
            row["scene_sha256"] = "stale-but-internally-matched-scene"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_frozen_protocol(
        files, trajectory_manifest_sha256=file_sha256(manifest)
    )
    with pytest.raises(ValueError, match="placement/trajectory scene mismatch"):
        _load(files)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("placement_id", "train-pair", "placement is unaccepted or mismatched"),
        ("split", "validation", "split/task/family mismatch"),
        ("task_id", 99, "split/task/family mismatch"),
        ("family", "narrow", "split/task/family mismatch"),
    ],
)
def test_rejected_cohort_identity_fails_closed(tmp_path, field, value, message):
    files = _build_contract_files(tmp_path)
    payload = files["cohort_payload"]
    payload["pairs"][0][field] = value
    files["cohort"].write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match=message):
        _load(files)


def test_rejected_cohort_file_hash_and_state_content_fail_closed(tmp_path):
    files = _build_contract_files(tmp_path)
    payload = files["cohort_payload"]
    state_key = next(key for key in payload["file_sha256"] if key.endswith("precrash_onpath_state.npy"))
    payload["file_sha256"][state_key] = "0" * 64
    files["cohort"].write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        _load(files)

    state_path = tmp_path / state_key
    np.save(state_path, np.asarray([999.0]))
    payload["file_sha256"][state_key] = file_sha256(state_path)
    files["cohort"].write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="exact on-path state hash mismatch"):
        _load(files)


def test_rejected_checkpoint_file_hash_fails_closed(tmp_path):
    files = _build_contract_files(tmp_path)
    with pytest.raises(ValueError, match="outside the frozen evaluation protocol"):
        load_evaluation_contract(
            placement_manifest=files["placements"],
            trajectory_manifest=files["manifests"]["heldout"],
            evaluation_cohort=files["cohort"],
            protocol=files["protocol"],
            checkpoint_metadata=files["metadata"],
            checkpoint_sha256="d" * 64,
            base_checkpoint_revision=BASE_REVISION,
            unnorm_key="libero_spatial",
        )


def test_rejected_unverified_or_state_mismatched_oracle_fails_closed(tmp_path):
    files = _build_contract_files(tmp_path)
    manifest = files["manifests"]["heldout"]
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    oracle = next(row for row in rows if row["trajectory_kind"] == "oracle_recovery")
    oracle["oracle_verified"] = False
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    payload = files["cohort_payload"]
    payload["trajectory_manifest_sha256"] = file_sha256(manifest)
    files["cohort"].write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="oracle_recovery must be verified"):
        _load(files)

    files = _build_contract_files(tmp_path / "mismatch")
    manifest = files["manifests"]["heldout"]
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    oracle = next(row for row in rows if row["trajectory_kind"] == "oracle_recovery")
    oracle["branch_start_state_sha256"] = "different"
    oracle["metadata"]["branch_start_hashes"]["simulator_state_sha256"] = "different"
    oracle["metadata"]["latest_verified_recoverable_state"] = "different"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    payload = files["cohort_payload"]
    payload["trajectory_manifest_sha256"] = file_sha256(manifest)
    files["cohort"].write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="not exact-state matched"):
        _load(files)


def test_source_timing_and_oracle_config_evidence_fail_closed(tmp_path):
    files = _build_contract_files(tmp_path / "timing")
    manifest = files["manifests"]["heldout"]
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    nominal = next(row for row in rows if row["trajectory_kind"] == "nominal_catastrophe")
    nominal["metadata"]["source_scan_precrash_index"] += 1
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="source-scan timing"):
        _load(files)

    files = _build_contract_files(tmp_path / "oracle-config")
    manifest = files["manifests"]["heldout"]
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    oracle = next(row for row in rows if row["trajectory_kind"] == "oracle_recovery")
    oracle["metadata"]["oracle_config"]["lane_margin"] = 9.9
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="independent recapture"):
        _load(files)


def test_exact_anchor_requires_the_materialized_matched_robot_state(tmp_path):
    files = _build_contract_files(tmp_path)
    payload = files["cohort_payload"]
    state_key = next(
        key for key in payload["file_sha256"]
        if key.endswith("matched_robot_state.npy")
    )
    state_path = tmp_path / state_key
    np.save(state_path, np.asarray([999.0], dtype=np.float64))
    payload["file_sha256"][state_key] = file_sha256(state_path)
    files["cohort"].write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="matched robot-state hash mismatch"):
        _load(files)


@pytest.mark.parametrize("mismatch", ("base", "horizon", "protocol"))
def test_checkpoint_base_h_and_protocol_mismatches_fail_closed(tmp_path, mismatch):
    files = _build_contract_files(tmp_path)
    metadata = dict(files["metadata"])
    metadata["calibration"] = dict(metadata["calibration"])
    if mismatch == "base":
        with pytest.raises(ValueError, match="configured Base revision"):
            load_evaluation_contract(
                placement_manifest=files["placements"],
                trajectory_manifest=files["manifests"]["heldout"],
                evaluation_cohort=files["cohort"],
                protocol=files["protocol"],
                checkpoint_metadata=metadata,
                checkpoint_sha256=CHECKPOINT_SHA256,
                base_checkpoint_revision="b" * 40,
                unnorm_key="libero_spatial",
            )
        return
    if mismatch == "horizon":
        metadata["trigger_horizon_actions"] = 10
        metadata["calibration"]["horizon"] = 10
        expected = "trajectory manifest H"
    else:
        metadata["protocol_sha256"] = "f" * 64
        expected = "checkpoint protocol"
    with pytest.raises(ValueError, match=expected):
        load_evaluation_contract(
            placement_manifest=files["placements"],
            trajectory_manifest=files["manifests"]["heldout"],
            evaluation_cohort=files["cohort"],
            protocol=files["protocol"],
            checkpoint_metadata=metadata,
            checkpoint_sha256=CHECKPOINT_SHA256,
            base_checkpoint_revision=BASE_REVISION,
            unnorm_key="libero_spatial",
        )


def test_heldout_source_or_id_leakage_fails_closed(tmp_path):
    files = _build_contract_files(tmp_path)
    heldout_rows = [
        json.loads(line) for line in files["manifests"]["heldout"].read_text().splitlines()
    ]
    train_rows = [
        json.loads(line) for line in files["manifests"]["train"].read_text().splitlines()
    ]
    for row in train_rows:
        row["source_state_sha256"] = heldout_rows[0]["source_state_sha256"]
    train = files["manifests"]["train"]
    train.write_text("".join(json.dumps(row) + "\n" for row in train_rows))
    train_sha = file_sha256(train)
    files["metadata"]["train_manifest_sha256"] = train_sha
    payload = files["cohort_payload"]
    payload["reference_manifests"]["train"]["sha256"] = train_sha
    files["cohort"].write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="heldout cohort leaks into train"):
        _load(files)


def test_authored_placement_manifest_alone_cannot_start_evaluation(tmp_path):
    files = _build_contract_files(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "eval_glass_recovery.py"),
            "--placement-manifest",
            str(files["placements"]),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    for required in ("--trajectory-manifest", "--evaluation-cohort", "--protocol"):
        assert required in result.stderr


class _FakeSim:
    def __init__(self):
        self.libero_done = False
        self.peak_force = 0.0
        self.xy = (0.0, 0.0)

    def max_contact_force(self, bodies, against=None):
        return 0.0

    def object_z(self, object_name):
        return 0.96

    def object_xy(self, object_name):
        return self.xy

    def object_tilt_deg(self, object_name):
        return 0.0

    def is_grasped(self, object_name):
        return False


class _FakeModel:
    nq = 9
    nv = 7


class _FakeEnv:
    def __init__(self, *, success_after=None):
        self.sim_view = _FakeSim()
        self.success_after = success_after
        self.steps = 0
        self.state = np.asarray([0.0])
        self.controller = {"controller.goal": np.asarray([1.0, 2.0], dtype=np.float64)}
        self._obs = {
            "robot0_eef_pos": np.asarray([0.0, 0.0, 1.1], dtype=np.float32),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.asarray([0.0, 0.0], dtype=np.float32),
            "akita_black_bowl_1_pos": np.asarray([0.1, 0.0, 0.9], dtype=np.float32),
            "plate_1_pos": np.asarray([0.3, 0.0, 0.9], dtype=np.float32),
        }

    def seed(self, seed):
        self.seed_value = seed

    def reset_to(self, state, movable_objects=None, obstacles=None):
        return self._reset(state, movable_objects)

    def reset_to_exact(self, state, movable_objects=None, obstacles=None):
        return self._reset(state, movable_objects)

    def _reset(self, state, movables):
        self.state = np.asarray(state).copy()
        self.steps = 0
        self.sim_view.libero_done = False
        self.sim_view.peak_force = 0.0
        self.sim_view.xy = tuple((movables or [_glass()])[0]["pos"][:2])
        return {key: value.copy() for key, value in self._obs.items()}

    def restore_controller_state(self, controller):
        self.controller = {key: np.asarray(value).copy() for key, value in controller.items()}

    def controller_state(self):
        return {key: value.copy() for key, value in self.controller.items()}

    def flat_state(self):
        return self.state.copy()

    def _raw_model(self):
        return _FakeModel()

    @staticmethod
    def _strip_movable_state(expanded_state, nq, nv, movable_count):
        state = np.asarray(expanded_state, dtype=np.float64)
        nq0 = int(nq) - 7 * int(movable_count)
        nv0 = int(nv) - 6 * int(movable_count)
        qpos = state[1:1 + int(nq)]
        qvel = state[1 + int(nq):1 + int(nq) + int(nv)]
        return np.concatenate([state[:1], qpos[:nq0], qvel[:nv0]])

    def dummy_action(self):
        return [0.0] * 7

    def step(self, action):
        self.steps += 1
        if self.success_after is not None and self.steps >= self.success_after:
            self.sim_view.libero_done = True
        return {key: value.copy() for key, value in self._obs.items()}, 0.0, False, {}

    def policy_observation(self, obs, resize_size):
        return {
            "full_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "state": np.zeros(8, dtype=np.float32),
        }


class _FakeBase:
    capture_hidden = True
    resize_size = 2

    def __init__(self):
        self.last_hidden = None

    def reset(self):
        self.last_hidden = None

    def act(self, observation, instruction):
        self.last_hidden = np.zeros(4, dtype=np.float32)
        return np.zeros(7, dtype=np.float32)


class _FakeRiskModel:
    risk_horizon = PROTOCOL_H
    risk_index = RISK_HORIZONS.index(PROTOCOL_H)
    checkpoint_metadata = {"calibration": {"threshold": 0.5}}

    def _predict(self, hidden, state, nominal):
        return {
            "risk": np.full(len(RISK_HORIZONS), 0.9, dtype=np.float32),
            "recovery_action": np.full(7, 0.25, dtype=np.float32),
        }


def _prepare_fake_runtime_pair(pair, env):
    observation_hash = _observation_sha256(env._obs)
    controller_hash = _controller_state_sha256(env.controller)
    for kind, record in pair.records.items():
        record.metadata["branch_start_hashes"]["observation_sha256"] = observation_hash
        record.metadata["branch_start_hashes"]["controller_state_sha256"] = controller_hash
        record.metadata["controller_state_sha256"] = controller_hash


def test_fake_env_modes_full_trace_privileged_diagnostics_and_timeout_semantics(tmp_path):
    contract = _load(_build_contract_files(tmp_path / "contract"))
    pair = contract.pairs[0]
    env = _FakeEnv(success_after=None)
    _prepare_fake_runtime_pair(pair, env)
    base = _FakeBase()
    risk = _FakeRiskModel()

    source = run_evaluation_episode(
        env=env,
        base=base,
        risk_model=risk,
        pair=pair,
        mode="source_to_task",
        regime="treatment",
        condition="base",
        rollout_seed=1,
        training_seed=17,
        settle_steps=0,
        max_steps=2,
        trigger_artifact_root=tmp_path / "triggers",
    )
    assert source["estimand_role"] == "paper_primary"
    assert source["outcome"] == "timeout"
    assert source["timeout"] is True
    assert source["safe_noncompletion"] is True
    assert source["safe_abort"] is False
    assert len(source["trace"]) == 2
    assert source["post_hoc_trigger_state_oracle"]["status"] == "not_applicable_no_intervention"
    required_trace = {
        "risk_vector", "nominal_action", "recovery_action", "executed_action",
        "mode", "first_trigger", "trigger_type",
        "threshold_crossing", "first_threshold_crossing",
        "captured_base_time_to_catastrophe_actions",
        "catastrophe_predicates_after_action",
    }
    assert all(required_trace.issubset(step) for step in source["trace"])

    exact = run_evaluation_episode(
        env=env,
        base=base,
        risk_model=risk,
        pair=pair,
        mode="exact_anchor",
        regime="treatment",
        condition="oracle_timed_oracle_recovery",
        rollout_seed=2,
        training_seed=17,
        settle_steps=0,
        max_steps=2,
        trigger_artifact_root=tmp_path / "triggers",
    )
    assert exact["estimand_role"] == "component_diagnostic"
    assert exact["privileged"] is True
    assert exact["first_intervention_step"] == 0
    assert exact["timely_trigger"] is True
    assert exact["post_hoc_trigger_state_oracle"]["status"] == "accepted_anchor_witness"
    assert exact["trace"][0]["mode"] == "recovery_latched_oracle"
    assert exact["trigger_state"]["simulator_state_file_sha256"]
    assert set(NONPRIVILEGED_CONDITIONS + PRIVILEGED_CONDITIONS) == set(MAIN_CONDITIONS)


def test_posthoc_oracle_restore_mismatch_fails_closed(tmp_path):
    class DriftingPosthocEnv(_FakeEnv):
        def reset_to_exact(self, state, movable_objects=None, obstacles=None):
            obs = super().reset_to_exact(state, movable_objects, obstacles)
            self.state = self.state + 1.0
            return obs

    contract = _load(_build_contract_files(tmp_path / "contract"))
    pair = contract.pairs[0]
    env = DriftingPosthocEnv(success_after=None)
    _prepare_fake_runtime_pair(pair, env)
    with pytest.raises(RuntimeError, match="post-hoc oracle trigger-state restore mismatch"):
        run_evaluation_episode(
            env=env,
            base=_FakeBase(),
            risk_model=_FakeRiskModel(),
            pair=pair,
            mode="source_to_task",
            regime="treatment",
            condition="full_learned_gate_recovery",
            rollout_seed=3,
            training_seed=17,
            settle_steps=0,
            max_steps=1,
            trigger_artifact_root=tmp_path / "triggers",
        )


def test_always_stop_uses_environment_noop_and_remains_appendix(tmp_path):
    class NoopAwareEnv(_FakeEnv):
        def dummy_action(self):
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

    contract = _load(_build_contract_files(tmp_path / "contract"))
    pair = contract.pairs[0]
    env = NoopAwareEnv(success_after=None)
    _prepare_fake_runtime_pair(pair, env)
    result = run_evaluation_episode(
        env=env,
        base=_FakeBase(),
        risk_model=_FakeRiskModel(),
        pair=pair,
        mode="exact_anchor",
        regime="treatment",
        condition="always_stop",
        rollout_seed=4,
        training_seed=17,
        settle_steps=0,
        max_steps=1,
        trigger_artifact_root=tmp_path / "triggers",
    )
    assert result["condition_role"] == "sanity_appendix"
    assert result["trace"][0]["executed_action"][-1] == -1.0
    assert result["task_success"] is False
    assert result["safe_noncompletion"] is True


def _analysis_rows() -> list[dict]:
    rows = []
    layout = (("s1", "p1", 5), ("s1", "p2", 1), ("s2", "p3", 1))
    for condition in ("base", "full_learned_gate_recovery"):
        for source, placement, repeats in layout:
            for repeat in range(repeats):
                base_success = source == "s2"
                success = base_success if condition == "base" else True
                rows.append({
                    "evaluation_mode": "source_to_task",
                    "condition": condition,
                    "regime": "treatment",
                    "training_seed": 17,
                    "source_state_sha256": source,
                    "placement_id": placement,
                    "rollout_seed": repeat,
                    "task_success": success,
                    "catastrophe": not success,
                    "timeout": False,
                    "intervened": condition != "base",
                    "timely_trigger": condition != "base",
                    "task_suite": "libero_spatial",
                    "task_id": 0,
                    "family": "wide" if source == "s1" else "narrow",
                })
                rows.append({
                    "evaluation_mode": "source_to_task",
                    "condition": condition,
                    "regime": "control",
                    "training_seed": 17,
                    "source_state_sha256": source,
                    "placement_id": placement,
                    "rollout_seed": repeat,
                    "task_success": True,
                    "catastrophe": False,
                    "timeout": False,
                    "intervened": condition != "base" and source == "s2",
                    "timely_trigger": False,
                    "task_suite": "libero_spatial",
                    "task_id": 0,
                    "family": "wide" if source == "s1" else "narrow",
                })
    return rows


def test_source_state_is_independent_cluster_and_repeats_do_not_increase_n():
    analysis = analyze_recovery_evaluation(
        _analysis_rows(), bootstrap_replicates=0
    )
    summary = analysis["methods"]["base"]["safe_task_success"]
    assert summary["estimate"] == 0.5
    assert summary["n_independent_source_states"] == 2
    assert summary["n_placements"] == 3
    assert summary["n_episode_rows_descriptive_only"] == 7
    assert summary["source_cluster_bootstrap_95_ci"] is None
    assert analysis["counts"]["unique_source_states"] == 2
    assert analysis["counts"]["placements"] == 3
    assert analysis["counts"]["training_seeds"] == 1

    difference = analysis["paired_method_differences"][
        "full_learned_gate_recovery_minus_base"
    ]["safe_task_success"]
    assert difference["estimate"] == 0.5
    assert difference["n_independent_source_states"] == 2
    assert difference["n_paired_placements"] == 3


def test_cluster_bootstrap_is_reproducible_and_modes_cannot_be_pooled():
    first = analyze_recovery_evaluation(
        _analysis_rows(), bootstrap_replicates=100, bootstrap_seed=9
    )
    second = analyze_recovery_evaluation(
        _analysis_rows(), bootstrap_replicates=100, bootstrap_seed=9
    )
    assert first["paired_method_differences"] == second["paired_method_differences"]
    mixed = _analysis_rows()
    mixed[0] = {**mixed[0], "evaluation_mode": "exact_anchor"}
    with pytest.raises(ValueError, match="one explicit evaluation_mode"):
        analyze_recovery_evaluation(mixed, bootstrap_replicates=0)

    unpaired = _analysis_rows()
    unpaired = [
        row for row in unpaired
        if not (
            row["condition"] == "full_learned_gate_recovery"
            and row["regime"] == "treatment"
            and row["placement_id"] == "p1"
            and row["rollout_seed"] == 4
        )
    ]
    with pytest.raises(ValueError, match="paired safe_task_success populations differ"):
        analyze_recovery_evaluation(unpaired, bootstrap_replicates=0)


def test_training_seeds_are_an_outer_variance_layer():
    seed_17 = _analysis_rows()
    seed_23 = []
    for row in seed_17:
        cloned = {**row, "training_seed": 23}
        if cloned["condition"] == "base" and cloned["regime"] == "treatment":
            cloned["task_success"] = True
            cloned["catastrophe"] = False
        seed_23.append(cloned)
    analysis = analyze_recovery_evaluation(
        [*seed_17, *seed_23], bootstrap_replicates=0
    )
    assert analysis["counts"]["training_seeds"] == 2
    assert analysis["methods"]["base"]["safe_task_success"]["estimate"] == 0.75
    paired = analysis["paired_method_differences"][
        "full_learned_gate_recovery_minus_base"
    ]["safe_task_success"]
    assert paired["estimate"] == 0.25


def test_source_bootstrap_draw_is_shared_across_training_seeds():
    rows = []
    for training_seed in (17, 23, 31):
        rows.extend([
            {**row, "training_seed": training_seed}
            for row in _analysis_rows()
        ])
    analysis = analyze_recovery_evaluation(
        rows, bootstrap_replicates=4000, bootstrap_seed=19
    )
    # With two source clusters whose outcomes are 0 and 1, a source-cluster
    # bootstrap must retain the full [0, 1] support even when the same cohort is
    # evaluated under several training seeds.  Independent source draws inside
    # each seed would spuriously narrow this interval.
    assert analysis["methods"]["base"]["safe_task_success"][
        "source_cluster_bootstrap_95_ci"
    ] == [0.0, 1.0]


def test_analysis_cli_combines_distinct_training_seed_outputs(tmp_path):
    identity = {
        "placement_manifest_sha256": "1" * 64,
        "trajectory_manifest_sha256": "2" * 64,
        "evaluation_cohort_sha256": "3" * 64,
        "primary_protocol_sha256": "4" * 64,
        "evaluation_protocol_sha256": "5" * 64,
    }
    evaluations = []
    for seed in (17, 23):
        rows = [{**row, "training_seed": seed} for row in _analysis_rows()]
        rows.extend([
            {
                **row,
                "condition": "always_stop",
                "condition_role": "sanity_appendix",
                "task_success": False,
                "catastrophe": False,
                "timeout": True,
                "intervened": True,
                "timely_trigger": True,
            }
            for row in rows
            if row["condition"] == "base"
        ])
        path = tmp_path / f"eval_{seed}.json"
        path.write_text(json.dumps({
            "kind": "glass_recovery_accepted_cohort_evaluation",
            "inputs": identity,
            "episodes": rows,
        }) + "\n")
        evaluations.append(path)
    out = tmp_path / "analysis.json"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "analyze_glass_recovery_eval.py"),
            "--evaluation",
            *(str(path) for path in evaluations),
            "--mode",
            "source_to_task",
            "--bootstrap-replicates",
            "0",
            "--out",
            str(out),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["analysis"]["counts"]["training_seeds"] == 2
    assert payload["analysis"]["independent_cluster"] == "source_state_sha256"
    assert "full_learned_gate_recovery_minus_base" in payload["analysis"][
        "paired_method_differences"
    ]
    assert "always_stop" not in payload["analysis"]["methods"]


def test_zero_gpu_v2_integration_collection_training_latch_cohort_eval_analysis(tmp_path):
    """Exercise the E15 contract without importing LIBERO or loading OpenVLA."""

    from scripts.train_glass_recovery import train

    files = _build_contract_files(tmp_path / "e15")
    checkpoint_dir = tmp_path / "e15" / "train_seed_17"
    train(Namespace(
        train_manifest=str(files["manifests"]["train"]),
        validation_manifest=str(files["manifests"]["validation"]),
        output=str(checkpoint_dir),
        device="cpu",
        max_steps=1,
        batch_size=8,
        learning_rate=3e-4,
        weight_decay=0.0,
        max_grad_norm=1.0,
        width=8,
        depth=1,
        dropout=0.0,
        sensitivity_margin=0.2,
        lambda_recovery=1.0,
        lambda_risk=1.0,
        lambda_hazard=0.0,
        lambda_severity=0.0,
        lambda_abort=0.0,
        lambda_invariance=0.5,
        lambda_sensitivity=0.0,
        gating_horizon=PROTOCOL_H,
        first_k=1,
        max_control_episode_fpr=1.0,
        base_resolved_revision=BASE_REVISION,
        unnorm_key="libero_spatial",
        protocol_sha256=canonical_sha256(_primary_protocol()),
        trigger_horizon=PROTOCOL_H,
        exit_threshold_ratio=0.5,
        abort_threshold=0.6,
        eval_every=1,
        log_every=1,
        num_workers=0,
        cache_size=2,
        seed=17,
        overwrite=False,
    ))
    checkpoint = checkpoint_dir / "glass_recovery.pt"
    _, checkpoint_metadata = GlassRecoveryNetwork.load_checkpoint(checkpoint)
    checkpoint_sha = file_sha256(checkpoint)

    from scripts.seal_glass_recovery_evaluation import seal

    sealed = seal(Namespace(
        placements=str(files["placements"]),
        dataset=str(files["manifests"]["validation"].parent),
        split="validation",
        checkpoint=[str(checkpoint)],
        pair_ids=None,
        rollout_seeds=[101],
        max_steps=4,
        output=str(tmp_path / "e15" / "sealed_validation"),
    ))
    sealed_contract = load_evaluation_contract(
        placement_manifest=files["placements"],
        trajectory_manifest=files["manifests"]["validation"],
        evaluation_cohort=sealed["cohort"],
        protocol=sealed["protocol"],
        checkpoint_metadata=checkpoint_metadata,
        checkpoint_sha256=checkpoint_sha,
        base_checkpoint_revision=BASE_REVISION,
        unnorm_key="libero_spatial",
    )
    assert len(sealed_contract.pairs) == 1
    assert sealed["source_states"] == 1

    sealed_train = seal(Namespace(
        placements=str(files["placements"]),
        dataset=str(files["manifests"]["train"].parent),
        split="train",
        checkpoint=[str(checkpoint)],
        pair_ids=None,
        rollout_seeds=[101],
        max_steps=4,
        output=str(tmp_path / "e15" / "sealed_train"),
    ))
    train_contract = load_evaluation_contract(
        placement_manifest=files["placements"],
        trajectory_manifest=files["manifests"]["train"],
        evaluation_cohort=sealed_train["cohort"],
        protocol=sealed_train["protocol"],
        checkpoint_metadata=checkpoint_metadata,
        checkpoint_sha256=checkpoint_sha,
        base_checkpoint_revision=BASE_REVISION,
        unnorm_key="libero_spatial",
    )
    assert train_contract.split == "train"

    # Seal the already accepted cohort and produced checkpoint into the final
    # evaluation protocol before any held-out episode is selected or run.
    protocol = json.loads(files["protocol"].read_text())
    protocol["evaluation_protocol"]["checkpoint_sha256_by_training_seed"] = {
        "17": checkpoint_sha
    }
    protocol["evaluation_protocol_sha256"] = canonical_sha256(
        protocol["evaluation_protocol"]
    )
    files["protocol"].write_text(json.dumps(protocol, indent=2) + "\n")
    cohort = json.loads(files["cohort"].read_text())
    cohort["evaluation_protocol_sha256"] = protocol["evaluation_protocol_sha256"]
    files["cohort"].write_text(json.dumps(cohort, indent=2) + "\n")

    contract = load_evaluation_contract(
        placement_manifest=files["placements"],
        trajectory_manifest=files["manifests"]["heldout"],
        evaluation_cohort=files["cohort"],
        protocol=files["protocol"],
        checkpoint_metadata=checkpoint_metadata,
        checkpoint_sha256=checkpoint_sha,
        base_checkpoint_revision=BASE_REVISION,
        unnorm_key="libero_spatial",
    )
    assert [pair.placement.placement_id for pair in contract.pairs] == ["heldout-pair"]

    class IntegratedBase(_FakeBase):
        checkpoint_identity = {"resolved_revision": BASE_REVISION}
        cfg = SimpleNamespace(unnorm_key="libero_spatial")

    base = IntegratedBase()
    policy = GlassRecoveryPolicy(
        base,
        str(checkpoint),
        device="cpu",
        risk_horizon=PROTOCOL_H,
        risk_enter_threshold=0.5,
    )
    predictions = iter((0.9, 0.1))
    policy._predict = lambda hidden, state, nominal: {
        "risk": np.full(len(RISK_HORIZONS), next(predictions), dtype=np.float32),
        "recovery_action": np.full(7, 0.25, dtype=np.float32),
    }
    observation = {"state": np.zeros(8, dtype=np.float32)}
    policy.act(observation, "pick and place")
    policy.act(observation, "pick and place")
    assert policy.mode == "recovery_latched"
    assert policy.decisions[-1]["ownership_age"] == 1

    pair = contract.pairs[0]
    env = _FakeEnv(success_after=1)
    _prepare_fake_runtime_pair(pair, env)
    rows = []
    for condition in ("base", "full_learned_gate_recovery"):
        for regime in ("treatment", "control"):
            score = 0.9 if condition != "base" and regime == "treatment" else 0.1
            policy._predict = lambda hidden, state, nominal, score=score: {
                "risk": np.full(len(RISK_HORIZONS), score, dtype=np.float32),
                "recovery_action": np.full(7, 0.25, dtype=np.float32),
            }
            rows.append(run_evaluation_episode(
                env=env,
                base=base,
                risk_model=policy,
                pair=pair,
                mode="exact_anchor",
                regime=regime,
                condition=condition,
                rollout_seed=101,
                training_seed=17,
                settle_steps=0,
                max_steps=1,
                trigger_artifact_root=tmp_path / "triggers",
            ))
    analysis = analyze_recovery_evaluation(rows, bootstrap_replicates=0)
    assert analysis["independent_cluster"] == "source_state_sha256"
    assert analysis["counts"]["unique_source_states"] == 1
    assert set(analysis["methods"]) == {"base", "full_learned_gate_recovery"}
    assert "full_learned_gate_recovery_minus_base" in analysis[
        "paired_method_differences"
    ]


def test_learned_result_audit_contract_fails_closed():
    metric = {"estimate": 0.5, "denominator": 8}
    payload = {
        "kind": "glass_recovery_learned_result",
        "accepted_cohort": {"sha256": "a" * 64, "pair_ids": ["p1", "p2"]},
        "exact_replay_summary": {"all_passed": True, "n_pairs": 2},
        "source_state_counts": {"train": 8, "validation": 5, "final_heldout": 12},
        "family_counts": {
            "train": {"narrow": 8},
            "validation": {"wide": 5},
            "final_heldout": {"late": 12},
        },
        "leakage_checks": {
            "source_state_overlap_count": 0,
            "family_overlap_count": 0,
            "physical_scene_overlap_count": 0,
        },
        "identities": {
            "checkpoint": {"sha256": "b" * 64, "training_seed": 17},
            "base": {
                "resolved_revision": BASE_REVISION,
                "unnorm_key": "libero_spatial",
            },
            "dataset": {
                "train_manifest_sha256": "c" * 64,
                "validation_manifest_sha256": "d" * 64,
                "final_heldout_manifest_sha256": "e" * 64,
            },
            "protocol": {
                "primary_sha256": "f" * 64,
                "evaluation_sha256": "1" * 64,
            },
        },
        "evaluation": {
            "primary_mode": "source_to_task",
            "conditions": sorted(E15_MAIN_BASELINE_CONDITIONS),
        },
        "source_cluster_analysis": {"independent_cluster": "source_state_sha256"},
        "primary_metrics": {
            "safe_task_success": metric,
            "catastrophe": metric,
            "clean_control_false_intervention": metric,
            "clean_control_task_preservation": metric,
        },
    }
    assert learned_recovery_semantic_errors(payload) == []
    invalid = json.loads(json.dumps(payload))
    invalid["accepted_cohort"]["pair_ids"].append("p1")
    invalid["evaluation"]["conditions"].remove("base")
    invalid["primary_metrics"]["catastrophe"]["denominator"] = 0
    errors = learned_recovery_semantic_errors(invalid)
    assert any("pair_ids" in error for error in errors)
    assert any("base" in error for error in errors)
    assert any("catastrophe" in error for error in errors)
