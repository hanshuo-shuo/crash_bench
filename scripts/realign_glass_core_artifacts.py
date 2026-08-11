#!/usr/bin/env python3
"""Realign provenance-recovered E14 glass pairs to an exact H anchor.

The historical root is read-only.  This diagnostic restores each complete
accepted E14 pair, advances its captured nominal actions to exact H, verifies
the repaired pre-action glass predicate and H-action catastrophe suffix, then
searches and independently recaptures a task-completing oracle.  It never
promotes an E14 pair into a v2 training/evaluation manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import (
    GlassPlacement,
    array_sha256,
    read_placement_manifest,
)
from crashbench.predicates import build_any
from crashbench.provenance import repository_provenance
from scripts.audit_glass_core_artifacts import file_sha256
from scripts.collect_glass_recovery_pairs import (
    CandidateRejected,
    PLATE,
    TARGET,
    _branch_start_hashes,
    _controller_state_sha256,
    _glass_force,
    _glass_predicate_specs,
    _observation_sha256,
    _prime_glass_predicates,
    _search_oracle,
)


REALIGNMENT_SCHEMA_VERSION = 2
REALIGNMENT_KIND = "glass_core_exact_h_realignment"
ORACLE_RESET_OBSERVATION_FIELDS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    f"{TARGET}_pos",
    f"{PLATE}_pos",
)


class PilotARejected(RuntimeError):
    """Scientific no-go for one candidate, distinct from a technical crash."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _source_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }


def _jsonl_rows(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _record_map(pair_payload: Mapping[str, Any]) -> dict[str, dict]:
    return {
        str(row.get("trajectory_kind")): dict(row)
        for row in pair_payload.get("records", [])
        if isinstance(row, Mapping) and row.get("trajectory_kind")
    }


def _load_controller_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _resolve_nominal_arrays(
    source_root: Path,
    pair_root: Path,
    nominal_record: Mapping[str, Any],
) -> Path:
    declared = Path(str(nominal_record.get("arrays_path", "")))
    candidates = (
        pair_root / declared.name,
        source_root / declared,
        source_root / "dataset" / declared,
    )
    matches = [path.resolve() for path in candidates if path.is_file()]
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise PilotARejected(
            "nominal_actions_unavailable",
            f"nominal arrays resolve to {len(unique)} files: {unique}",
        )
    return unique[0]


def _verify_audited_files(source_root: Path, row: Mapping[str, Any]) -> None:
    for relative, metadata in row.get("files", {}).items():
        path = source_root / relative
        if not path.is_file():
            raise PilotARejected("historical_evidence_missing", f"missing {path}")
        if path.stat().st_size != int(metadata["bytes"]):
            raise PilotARejected("historical_evidence_drift", f"size drift for {path}")
        if file_sha256(path) != metadata["sha256"]:
            raise PilotARejected("historical_evidence_drift", f"hash drift for {path}")


def _predicate_start_diagnostics(
    env: LiberoEnv,
    placement: GlassPlacement,
) -> tuple[Any, dict[str, Any]]:
    glass = placement.on_path_glass
    crash = build_any(_glass_predicate_specs([glass]))
    try:
        _prime_glass_predicates(crash, env.sim_view)
    except CandidateRejected as exc:
        raise PilotARejected("first_action_predicate_invalid", str(exc)) from exc
    authored_xy = np.asarray(glass["pos"][:2], dtype=float)
    live_xy = np.asarray(env.sim_view.object_xy(glass["name"]), dtype=float)
    return crash, {
        "reference_xy_explicit": True,
        "predicate_primed_before_first_action": True,
        "pre_action_predicate_false": True,
        "authored_reference_xy": authored_xy.round(8).tolist(),
        "live_pre_action_xy": live_xy.round(8).tolist(),
        "pre_action_displacement_m": float(np.linalg.norm(live_xy - authored_xy)),
        "pre_action_tilt_deg": float(env.sim_view.object_tilt_deg(glass["name"])),
        "pre_action_glass_force_n": _glass_force(
            env.sim_view, [placement.on_path_glass]
        ),
    }


def _require_simulator_controller_hashes(
    actual: Mapping[str, str],
    expected: Mapping[str, str],
    *,
    label: str,
) -> None:
    mismatched = [
        key for key in ("simulator_state_sha256", "controller_state_sha256")
        if actual.get(key) != expected.get(key)
    ]
    if mismatched:
        reason = (
            "exact_controller_state_unavailable"
            if "controller_state_sha256" in mismatched
            else "exact_state_restore_mismatch"
        )
        raise PilotARejected(
            reason, f"{label} did not restore exact hashes: {mismatched}"
        )


def _observation_restore_diagnostics(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> dict[str, Any]:
    differences = []
    for key in sorted(set(expected) | set(actual)):
        if key not in expected or key not in actual:
            differences.append({
                "field": key,
                "reason": "missing_from_expected" if key not in expected else "missing_from_actual",
            })
            continue
        before = np.asarray(expected[key])
        after = np.asarray(actual[key])
        if before.shape != after.shape or before.dtype != after.dtype:
            differences.append({
                "field": key,
                "reason": "shape_or_dtype",
                "expected_shape": list(before.shape),
                "actual_shape": list(after.shape),
                "expected_dtype": str(before.dtype),
                "actual_dtype": str(after.dtype),
            })
            continue
        if np.array_equal(before, after):
            continue
        difference: dict[str, Any] = {"field": key, "reason": "value"}
        if np.issubdtype(before.dtype, np.number):
            delta = np.abs(before.astype(np.float64) - after.astype(np.float64))
            difference.update({
                "max_abs_difference": float(np.max(delta)) if delta.size else 0.0,
                "differing_elements": int(np.count_nonzero(delta)),
                "elements": int(delta.size),
            })
        differences.append(difference)
    expected_hash = _observation_sha256(expected)
    actual_hash = _observation_sha256(actual)
    return {
        "exact": expected_hash == actual_hash,
        "expected_observation_sha256": expected_hash,
        "restored_observation_sha256": actual_hash,
        "differing_field_count": len(differences),
        "differing_fields": differences,
        "interpretation": (
            "diagnostic_only; historical E14 did not capture every observable "
            "delay/history field, so simulator/controller hashes plus empirical "
            "suffix and oracle replay are authoritative"
        ),
    }


def _oracle_reset_relevant_observation(
    observation: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Copy the small observation subset used by the oracle or its diagnostics.

    The full LIBERO observation also contains rendered images and observable
    delay/history buffers that the legacy E14 artifact did not capture.  The
    detour controller consumes EEF pose and target/plate positions; gripper
    position is retained because the rollout trace reports it.
    """

    missing = [
        field for field in ORACLE_RESET_OBSERVATION_FIELDS
        if field not in observation
    ]
    if missing:
        raise PilotARejected(
            "oracle_observation_fields_unavailable",
            f"aligned-H oracle observation is missing fields: {missing}",
        )
    return {
        field: np.asarray(observation[field]).copy()
        for field in ORACLE_RESET_OBSERVATION_FIELDS
    }


def _oracle_reset_observation_diagnostics(
    full_observation_hashes: list[str],
    relevant_observations: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize reset-to-reset observable drift without treating it as a gate.

    Reset success is established by byte-identical simulator/controller hashes.
    Oracle success is established empirically by a successful search rollout and
    a fresh successful recapture.  Observable drift remains important audit
    evidence, but a full-observation hash also covers camera and sensor history
    that was never part of the E14 snapshot contract.
    """

    if not full_observation_hashes:
        raise ValueError("oracle reset diagnostics require at least one reset")
    if len(full_observation_hashes) != len(relevant_observations):
        raise ValueError("oracle reset hash/observation counts differ")
    relevant_hashes = [
        _observation_sha256(observation)
        for observation in relevant_observations
    ]
    baseline = relevant_observations[0]
    field_diagnostics: dict[str, dict[str, Any]] = {}
    for field in ORACLE_RESET_OBSERVATION_FIELDS:
        expected = np.asarray(baseline[field])
        actual_values = [
            np.asarray(observation[field])
            for observation in relevant_observations
        ]
        shape_dtype_stable = all(
            actual.shape == expected.shape and actual.dtype == expected.dtype
            for actual in actual_values
        )
        exact = shape_dtype_stable and all(
            np.array_equal(expected, actual) for actual in actual_values
        )
        diagnostic: dict[str, Any] = {
            "shape": list(expected.shape),
            "dtype": str(expected.dtype),
            "shape_dtype_stable": shape_dtype_stable,
            "exact_across_resets": exact,
            "sha256_by_reset": [array_sha256(actual) for actual in actual_values],
        }
        if shape_dtype_stable and np.issubdtype(expected.dtype, np.number):
            deltas = [
                np.abs(expected.astype(np.float64) - actual.astype(np.float64))
                for actual in actual_values
            ]
            diagnostic["max_abs_difference_from_first"] = max(
                (float(np.max(delta)) if delta.size else 0.0 for delta in deltas),
                default=0.0,
            )
        field_diagnostics[field] = diagnostic
    return {
        "reset_count": len(full_observation_hashes),
        "full_observation_sha256_by_reset": list(full_observation_hashes),
        "unique_full_observation_hash_count": len(set(full_observation_hashes)),
        "full_observation_exact_across_resets": (
            len(set(full_observation_hashes)) == 1
        ),
        "oracle_relevant_fields": list(ORACLE_RESET_OBSERVATION_FIELDS),
        "oracle_relevant_observation_sha256_by_reset": relevant_hashes,
        "unique_oracle_relevant_observation_hash_count": len(set(relevant_hashes)),
        "oracle_relevant_observation_exact_across_resets": (
            len(set(relevant_hashes)) == 1
        ),
        "field_diagnostics": field_diagnostics,
        "interpretation": (
            "diagnostic_only; exact simulator/controller restore is required on "
            "every reset, and a successful search rollout plus fresh successful "
            "recapture is the empirical oracle acceptance gate"
        ),
    }


def _candidate_pair_rows(audit_rows: list[dict]) -> list[dict]:
    candidates = [
        row for row in audit_rows
        if row.get("terminal_event") in {"accepted", "accepted_pair_artifact"}
        and row.get("pair_path")
    ]
    if len({row.get("attempt_id") for row in candidates}) != len(candidates):
        raise ValueError("candidate audit rows contain duplicate attempt IDs")
    if len({row.get("placement_id") for row in candidates}) != len(candidates):
        raise ValueError("candidate audit rows contain duplicate placements")
    return sorted(candidates, key=lambda row: str(row["placement_id"]))


def _run_candidate(
    *,
    source_root: Path,
    output_root: Path,
    audit_row: Mapping[str, Any],
    placement: GlassPlacement,
    target_h: int,
    oracle_steps: int,
    runner_commit: str,
) -> dict:
    placement_id = str(audit_row["placement_id"])
    candidate_root = output_root / "candidates" / placement_id
    result_path = candidate_root / "result.json"
    if result_path.exists():
        raise FileExistsError(
            f"refusing to overwrite completed candidate result {result_path}"
        )
    candidate_root.mkdir(parents=True, exist_ok=True)
    base_result: dict[str, Any] = {
        "schema_version": REALIGNMENT_SCHEMA_VERSION,
        "kind": REALIGNMENT_KIND,
        "placement_id": placement_id,
        "split": placement.split,
        "candidate_role": "development_salvage_only",
        "direct_v2_promotion_allowed": False,
        "historical_attempt_id": audit_row["attempt_id"],
        "historical_attempt_identity_quality": audit_row[
            "attempt_identity_quality"
        ],
        "runner_commit": runner_commit,
        "target_h_actions": target_h,
    }
    progress: dict[str, Any] = {
        "historical_files_verified": False,
        "exact_controller_state_available": False,
        "repaired_first_action_predicate_checks_pass": False,
        "exact_h_nominal_suffix_replay_verified": False,
        "oracle_independent_replay_verified": False,
    }
    evidence: dict[str, Any] = {}
    try:
        _verify_audited_files(source_root, audit_row)
        progress["historical_files_verified"] = True
        pair_path = (source_root / str(audit_row["pair_path"])).resolve()
        pair_root = pair_path.parent
        pair_payload = json.loads(pair_path.read_text())
        records = _record_map(pair_payload)
        nominal_record = records.get("nominal_catastrophe")
        oracle_record = records.get("oracle_recovery")
        if not nominal_record or not oracle_record:
            raise PilotARejected(
                "historical_pair_incomplete", "nominal/oracle record is missing"
            )
        state_path = pair_root / "precrash_onpath_state.npy"
        controller_path = pair_root / "controller_state.npz"
        if not state_path.is_file() or not controller_path.is_file():
            raise PilotARejected(
                "exact_controller_state_unavailable",
                "historical exact simulator/controller state is unavailable",
            )
        old_state = np.load(state_path, allow_pickle=False)
        old_controller = _load_controller_state(controller_path)
        expected_state_hash = nominal_record.get("branch_start_state_sha256")
        expected_controller_hash = nominal_record.get("metadata", {}).get(
            "controller_state_sha256"
        )
        if array_sha256(old_state) != expected_state_hash:
            raise PilotARejected(
                "historical_evidence_drift", "old exact simulator state hash mismatch"
            )
        if _controller_state_sha256(old_controller) != expected_controller_hash:
            raise PilotARejected(
                "historical_evidence_drift", "old controller state hash mismatch"
            )
        nominal_path = _resolve_nominal_arrays(
            source_root, pair_root, nominal_record
        )
        with np.load(nominal_path, allow_pickle=False) as archive:
            actions = np.asarray(archive["executed_action"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise PilotARejected(
                "nominal_actions_invalid", f"executed actions have shape {actions.shape}"
            )
        if len(actions) < target_h:
            raise PilotARejected(
                "old_suffix_shorter_than_target_h",
                f"old suffix has {len(actions)} actions, target H is {target_h}",
            )
        advance_actions = len(actions) - target_h

        env = LiberoEnv(
            placement.task_suite, placement.task_id, seed=0
        )
        obs = env.reset_to_exact(
            old_state, movable_objects=[placement.on_path_glass]
        )
        try:
            env.restore_controller_state(old_controller)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise PilotARejected(
                "exact_controller_state_unavailable", str(exc)
            ) from exc
        old_start_hashes = _branch_start_hashes(env, obs)
        if old_start_hashes["simulator_state_sha256"] != expected_state_hash:
            raise PilotARejected(
                "exact_state_restore_mismatch", "old simulator state did not restore"
            )
        if old_start_hashes["controller_state_sha256"] != expected_controller_hash:
            raise PilotARejected(
                "exact_controller_state_unavailable",
                "old controller state did not restore byte-identically",
            )
        progress["exact_controller_state_available"] = True
        prefix_crash, old_predicate = _predicate_start_diagnostics(env, placement)
        evidence["old_start_hashes"] = old_start_hashes
        evidence["old_start_predicate"] = old_predicate
        for index, action in enumerate(actions[:advance_actions]):
            obs, _, done, _ = env.step(action.astype(float).tolist())
            if prefix_crash(env.sim_view):
                raise PilotARejected(
                    "catastrophe_before_aligned_h",
                    f"captured prefix catastrophized at action {index}",
                )
            if done:
                raise PilotARejected(
                    "task_completed_before_aligned_h",
                    f"task completed during captured prefix at action {index}",
                )

        aligned_state = env.flat_state()
        aligned_controller = env.controller_state()
        aligned_observation = {
            key: np.asarray(value).copy() for key, value in obs.items()
        }
        aligned_state_hash = array_sha256(aligned_state)
        aligned_controller_hash = _controller_state_sha256(aligned_controller)
        aligned_start_hashes = _branch_start_hashes(env, obs)
        aligned_state_path = candidate_root / "exact_h_onpath_state.npy"
        aligned_controller_path = candidate_root / "exact_h_controller_state.npz"
        np.save(aligned_state_path, aligned_state)
        np.savez_compressed(aligned_controller_path, **aligned_controller)

        replay_obs = env.reset_to_exact(
            aligned_state, movable_objects=[placement.on_path_glass]
        )
        try:
            env.restore_controller_state(aligned_controller)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise PilotARejected(
                "exact_controller_state_unavailable", str(exc)
            ) from exc
        replay_start_hashes = _branch_start_hashes(env, replay_obs)
        _require_simulator_controller_hashes(
            replay_start_hashes,
            aligned_start_hashes,
            label=f"{placement_id} aligned-H replay",
        )
        observation_restore = _observation_restore_diagnostics(
            aligned_observation, replay_obs
        )
        replay_crash, aligned_predicate = _predicate_start_diagnostics(env, placement)
        progress["repaired_first_action_predicate_checks_pass"] = True
        evidence["aligned_state"] = {
            "state_path": str(aligned_state_path),
            "state_sha256": aligned_state_hash,
            "controller_state_path": str(aligned_controller_path),
            "controller_state_sha256": aligned_controller_hash,
            "branch_start_hashes": aligned_start_hashes,
        }
        evidence["repaired_first_action_predicate"] = aligned_predicate
        evidence["observation_restore_diagnostic"] = observation_restore
        event_action_index = None
        peak_force = 0.0
        suffix = actions[advance_actions:]
        for index, action in enumerate(suffix):
            replay_obs, _, _, _ = env.step(action.astype(float).tolist())
            peak_force = max(
                peak_force,
                _glass_force(env.sim_view, [placement.on_path_glass]),
            )
            if replay_crash(env.sim_view):
                event_action_index = index
                break
        aligned_predicate.update({
            "first_action_post_step_evaluated": len(suffix) > 0,
            "post_action_predicate_checks": (
                event_action_index + 1
                if event_action_index is not None else len(suffix)
            ),
            "predicate_evaluated_after_every_executed_action": True,
        })
        exact_replay_verified = event_action_index == target_h - 1
        nominal_replay = {
            "old_suffix_actions": len(actions),
            "advance_captured_actions": advance_actions,
            "remaining_actions": len(suffix),
            "expected_catastrophe_action_index": target_h - 1,
            "actual_catastrophe_action_index": event_action_index,
            "catastrophe_on_action_h": exact_replay_verified,
            "peak_glass_force_n": peak_force,
        }
        evidence["nominal_replay"] = nominal_replay
        if not exact_replay_verified:
            raise PilotARejected(
                "exact_h_nominal_suffix_replay_failed",
                f"expected catastrophe at {target_h - 1}, observed {event_action_index}",
            )
        progress["exact_h_nominal_suffix_replay_verified"] = True

        oracle_reset_observation_hashes: list[str] = []
        oracle_reset_relevant_observations: list[dict[str, np.ndarray]] = []

        def reset_aligned_onpath() -> dict:
            reset_obs = env.reset_to_exact(
                aligned_state, movable_objects=[placement.on_path_glass]
            )
            env.restore_controller_state(aligned_controller)
            reset_hashes = _branch_start_hashes(env, reset_obs)
            _require_simulator_controller_hashes(
                reset_hashes,
                aligned_start_hashes,
                label=f"{placement_id} aligned-H oracle reset",
            )
            oracle_reset_observation_hashes.append(
                reset_hashes["observation_sha256"]
            )
            oracle_reset_relevant_observations.append(
                _oracle_reset_relevant_observation(reset_obs)
            )
            return reset_obs

        old_oracle_config = oracle_record.get("metadata", {}).get("oracle_config")
        preferred = [old_oracle_config] if isinstance(old_oracle_config, Mapping) else []
        try:
            oracle, oracle_attempts = _search_oracle(
                reset_aligned_onpath,
                env,
                None,
                placement,
                [placement.on_path_glass],
                oracle_steps,
                collect_success=True,
                counterfactual_peak_force=peak_force,
                preferred_configs=preferred,
                capture_success_rows=False,
            )
        except CandidateRejected as exc:
            evidence["oracle_reset_observation_diagnostic"] = (
                _oracle_reset_observation_diagnostics(
                    oracle_reset_observation_hashes,
                    oracle_reset_relevant_observations,
                )
            )
            raise PilotARejected("oracle_independent_recapture_failed", str(exc)) from exc
        oracle_reset_diagnostic = _oracle_reset_observation_diagnostics(
            oracle_reset_observation_hashes,
            oracle_reset_relevant_observations,
        )
        evidence["oracle_reset_observation_diagnostic"] = oracle_reset_diagnostic
        if oracle is None:
            raise PilotARejected(
                "oracle_success_not_reproduced_from_aligned_h",
                "no safe task-completing oracle was found from aligned H",
            )
        verification = dict(oracle.get("verification", {}))
        oracle_verified = bool(
            not oracle["crashed"]
            and oracle["succeeded"]
            and verification.get("independent_recapture") is True
            and verification.get("recapture_success") is True
        )
        if not oracle_verified:
            raise PilotARejected(
                "oracle_independent_recapture_failed",
                "oracle search did not produce a successful independent recapture",
            )
        progress["oracle_independent_replay_verified"] = True
        _atomic_json(candidate_root / "oracle_search.json", {
            "placement_id": placement_id,
            "attempts": oracle_attempts,
            "verification": verification,
        })
        result = {
            **base_result,
            **progress,
            **evidence,
            "status": "passed",
            "no_go_reason": None,
            "oracle": {
                "search_attempts": len(oracle_attempts),
                "preferred_historical_config_tested_first": bool(preferred),
                "crashed": bool(oracle["crashed"]),
                "task_succeeded": bool(oracle["succeeded"]),
                "steps": int(oracle["steps"]),
                "peak_glass_force_n": float(oracle["peak_force"]),
                "config": oracle["config"],
                "verification": verification,
                "first_reset_observation_sha256": (
                    oracle_reset_observation_hashes[0]
                ),
                "unique_reset_observation_hash_count": (
                    oracle_reset_diagnostic["unique_full_observation_hash_count"]
                ),
                "reset_count": oracle_reset_diagnostic["reset_count"],
            },
        }
    except PilotARejected as exc:
        result = {
            **base_result,
            **progress,
            **evidence,
            "status": "no_go",
            "no_go_reason": exc.reason,
            "error": str(exc),
        }
    _atomic_json(result_path, result)
    return result


def summarize_realignment(
    results: list[Mapping[str, Any]],
    inventory_summary: Mapping[str, Any],
    *,
    target_h: int,
) -> dict:
    candidate_count = len(results)
    accounting = inventory_summary.get("historical_attempt_accounting") or {}
    provenance_ok = accounting.get("attempts_have_unique_provenance") is True
    controller_passes = sum(
        row.get("exact_controller_state_available") is True for row in results
    )
    predicate_passes = sum(
        row.get("repaired_first_action_predicate_checks_pass") is True
        for row in results
    )
    nominal_passes = sum(
        row.get("exact_h_nominal_suffix_replay_verified") is True for row in results
    )
    oracle_passes = sum(
        row.get("oracle_independent_replay_verified") is True for row in results
    )
    rates = {
        "exact_controller_state_available_rate": (
            controller_passes / candidate_count if candidate_count else 0.0
        ),
        "repaired_first_action_predicate_pass_rate": (
            predicate_passes / candidate_count if candidate_count else 0.0
        ),
        "exact_h_nominal_suffix_replay_rate": (
            nominal_passes / candidate_count if candidate_count else 0.0
        ),
        "oracle_independent_replay_rate": (
            oracle_passes / candidate_count if candidate_count else 0.0
        ),
    }
    no_go_reasons = []
    if not provenance_ok:
        no_go_reasons.append("attempts_do_not_have_unique_provenance")
    if candidate_count == 0:
        no_go_reasons.append("no_complete_historical_candidates")
    if controller_passes != candidate_count:
        no_go_reasons.append("exact_controller_state_unavailable")
    if predicate_passes != candidate_count:
        no_go_reasons.append("repaired_first_action_predicate_checks_failed")
    if nominal_passes != candidate_count:
        no_go_reasons.append("exact_h_nominal_suffix_replay_below_100_percent")
    if oracle_passes != candidate_count:
        no_go_reasons.append("oracle_independent_replay_below_100_percent")
    go = not no_go_reasons
    return {
        "schema_version": REALIGNMENT_SCHEMA_VERSION,
        "kind": "glass_core_h_realignment_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_h_actions": target_h,
        "candidate_role": "development_salvage_only",
        "direct_v2_promotions": 0,
        "attempts_have_unique_provenance": provenance_ok,
        "candidate_count": candidate_count,
        "candidate_ids": [str(row["placement_id"]) for row in results],
        "candidate_statuses": dict(Counter(str(row["status"]) for row in results)),
        "rates": rates,
        "candidate_results": [dict(row) for row in results],
        "go": go,
        "decision": "pilot_a_go" if go else "pilot_a_no_go",
        "no_go_reasons": no_go_reasons,
        "pilot_b_allowed": go,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--inventory-summary", required=True)
    parser.add_argument("--placements", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--target-h", type=int, default=20)
    parser.add_argument("--oracle-steps", type=int, default=900)
    parser.add_argument("--expected-candidates", type=int, default=3)
    args = parser.parse_args()
    if args.target_h < 1 or args.oracle_steps < 1 or args.expected_candidates < 1:
        raise SystemExit("target H, oracle steps, and expected candidates must be positive")

    source_root = Path(args.source_root).resolve()
    audit_path = Path(args.audit).resolve()
    inventory_summary_path = Path(args.inventory_summary).resolve()
    placements_path = Path(args.placements).resolve()
    output_root = Path(args.output_root).resolve()
    summary_out = Path(args.summary_out).resolve()
    if not source_root.is_dir():
        raise SystemExit(f"source root does not exist: {source_root}")
    for target in (output_root, summary_out):
        try:
            target.relative_to(source_root)
        except ValueError:
            pass
        else:
            raise SystemExit("realignment outputs must be outside historical source root")
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to reuse nonempty output root {output_root}")
    if summary_out.exists():
        raise SystemExit(f"refusing to overwrite summary {summary_out}")

    repo_root = Path(__file__).resolve().parents[1]
    repo = repository_provenance(repo_root, require_clean=True)
    runner_commit = str(repo["git_commit"])
    declared_commit = os.environ.get("CB_CODE_COMMIT")
    if declared_commit is not None and declared_commit != runner_commit:
        raise SystemExit(
            f"CB_CODE_COMMIT={declared_commit} does not match checkout {runner_commit}"
        )
    before = _source_snapshot(source_root)
    audit_rows = _jsonl_rows(audit_path)
    if len({str(row.get("attempt_id")) for row in audit_rows}) != len(audit_rows):
        raise SystemExit("audit contains duplicate attempt IDs")
    inventory_summary = json.loads(inventory_summary_path.read_text())
    accounting = inventory_summary.get("historical_attempt_accounting") or {}
    if accounting.get("attempts_have_unique_provenance") is not True:
        raise SystemExit("inventory has not established unique attempt provenance")
    if int(inventory_summary.get("attempts", -1)) != len(audit_rows):
        raise SystemExit("inventory attempt count does not match audit JSONL")
    candidates = _candidate_pair_rows(audit_rows)
    if len(candidates) != args.expected_candidates:
        raise SystemExit(
            f"found {len(candidates)} complete candidates, expected {args.expected_candidates}"
        )
    placements, _ = read_placement_manifest(placements_path)
    placements_by_id = {row.placement_id: row for row in placements}
    missing = sorted(
        str(row["placement_id"]) for row in candidates
        if str(row["placement_id"]) not in placements_by_id
    )
    if missing:
        raise SystemExit(f"candidate placements missing from manifest: {missing}")

    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for row in candidates:
        placement_id = str(row["placement_id"])
        print(f"REALIGN {placement_id} H={args.target_h}", flush=True)
        result = _run_candidate(
            source_root=source_root,
            output_root=output_root,
            audit_row=row,
            placement=placements_by_id[placement_id],
            target_h=args.target_h,
            oracle_steps=args.oracle_steps,
            runner_commit=runner_commit,
        )
        results.append(result)
        print(
            f"{result['status'].upper()} {placement_id} "
            f"reason={result.get('no_go_reason')}",
            flush=True,
        )
    summary = summarize_realignment(results, inventory_summary, target_h=args.target_h)
    summary.update({
        "runner_commit": runner_commit,
        "source_root": str(source_root),
        "source_root_read_only": True,
        "audit_jsonl": str(audit_path),
        "audit_jsonl_sha256": file_sha256(audit_path),
        "inventory_summary": str(inventory_summary_path),
        "inventory_summary_sha256": file_sha256(inventory_summary_path),
        "placement_manifest": str(placements_path),
        "placement_manifest_sha256": file_sha256(placements_path),
        "output_root": str(output_root),
    })
    _atomic_json(summary_out, summary)
    after = _source_snapshot(source_root)
    if after != before:
        raise RuntimeError("historical source root changed during realignment")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
