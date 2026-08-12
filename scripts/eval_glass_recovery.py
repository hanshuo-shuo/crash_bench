#!/usr/bin/env python3
"""Accepted-cohort evaluation for P0-D glass recovery.

All manifest, artifact, split, protocol, and checkpoint checks run before an
OpenVLA policy or LIBERO environment is constructed.  ``source_to_task`` is the
paper-primary end-to-end estimand; ``exact_anchor`` is a component diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import (
    RISK_HORIZONS,
    SCHEMA_VERSION,
    GlassPlacement,
    PairedTrajectoryRecord,
    array_sha256,
    canonical_sha256,
    read_placement_manifest,
    read_trajectory_manifest,
    validate_episode_arrays,
    validate_primary_pair,
)
from crashbench.glass_recovery_model import GlassRecoveryNetwork
from crashbench.policies import OpenVLAPolicy
from crashbench.policies.glass_recovery_policy import (
    GlassRecoveryPolicy,
    _validate_primary_checkpoint_metadata,
)
from crashbench.predicates import build_predicate, prime_predicate
from crashbench.prompts import compose_instruction
from crashbench.recovery import DetourComplete, RetreatHold
from scripts.collect_glass_recovery_pairs import (
    CAREFUL_PROMPT_PREFIX,
    PLATE,
    TARGET,
    _branch_start_hashes,
    _controller_glass,
    _controller_state_sha256,
    _glass_force,
    _glass_predicate_specs,
    _observation_sha256,
)


EVALUATION_COHORT_SCHEMA_VERSION = 1
EVALUATION_PROTOCOL_SCHEMA_VERSION = 1
EVALUATION_MODES = ("source_to_task", "exact_anchor")
NONPRIVILEGED_CONDITIONS = (
    "base",
    "generic_careful",
    "hazard_specific_careful",
    "risk_gate_retreat_hold",
    "full_learned_gate_recovery",
)
PRIVILEGED_CONDITIONS = (
    "risk_gate_oracle_recovery",
    "oracle_timed_learned_recovery",
    "oracle_timed_oracle_recovery",
)
SANITY_CONDITIONS = ("always_stop", "legacy_e14_careful", "blocked_safe_abort")
MAIN_CONDITIONS = NONPRIVILEGED_CONDITIONS + PRIVILEGED_CONDITIONS
ALL_CONDITIONS = MAIN_CONDITIONS + SANITY_CONDITIONS


def condition_role(condition: str) -> str:
    if condition in NONPRIVILEGED_CONDITIONS:
        return "nonprivileged_main"
    if condition in PRIVILEGED_CONDITIONS:
        return "privileged_diagnostic"
    if condition in SANITY_CONDITIONS:
        return "sanity_appendix"
    raise ValueError(f"unknown evaluation condition {condition!r}")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _load_protocol(path: Path) -> tuple[dict, str, dict, str, dict]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != EVALUATION_PROTOCOL_SCHEMA_VERSION:
        raise ValueError(
            f"{path} must use evaluation protocol schema "
            f"{EVALUATION_PROTOCOL_SCHEMA_VERSION}"
        )
    if payload.get("kind") != "glass_recovery_evaluation_protocol":
        raise ValueError(f"{path} is not a glass recovery evaluation protocol")
    primary = dict(_require_mapping(payload.get("primary_protocol"), "primary_protocol"))
    primary_sha = str(payload.get("primary_protocol_sha256", ""))
    if canonical_sha256(primary) != primary_sha:
        raise ValueError(f"{path} primary protocol SHA is invalid")
    evaluation = dict(
        _require_mapping(payload.get("evaluation_protocol"), "evaluation_protocol")
    )
    evaluation_sha = str(payload.get("evaluation_protocol_sha256", ""))
    if canonical_sha256(evaluation) != evaluation_sha:
        raise ValueError(f"{path} evaluation protocol SHA is invalid")
    if evaluation.get("name") != "glass_recovery_eval_p0d_v1":
        raise ValueError("evaluation protocol name is not P0-D")
    modes = tuple(evaluation.get("evaluation_modes", ()))
    conditions = tuple(evaluation.get("conditions", ()))
    if set(modes) != set(EVALUATION_MODES) or len(set(modes)) != len(modes):
        raise ValueError(f"evaluation protocol must declare modes {EVALUATION_MODES}")
    if not set(MAIN_CONDITIONS).issubset(conditions):
        missing = sorted(set(MAIN_CONDITIONS) - set(conditions))
        raise ValueError(f"evaluation protocol lacks required conditions {missing}")
    if (
        not conditions
        or len(set(conditions)) != len(conditions)
        or any(condition not in ALL_CONDITIONS for condition in conditions)
    ):
        raise ValueError("evaluation protocol contains an unknown condition")
    seeds = evaluation.get("rollout_seeds")
    try:
        parsed_seeds = [int(seed) for seed in seeds] if isinstance(seeds, list) else []
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation protocol rollout_seeds must be integers") from exc
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(set(parsed_seeds)) != len(seeds)
        or any(seed < 0 or seed >= 2**32 for seed in parsed_seeds)
    ):
        raise ValueError("evaluation protocol needs unique explicit rollout_seeds")
    if int(evaluation.get("max_steps", 0)) < 1:
        raise ValueError("evaluation protocol max_steps must be positive")
    checkpoints = evaluation.get("checkpoint_sha256_by_training_seed")
    if not isinstance(checkpoints, Mapping) or not checkpoints:
        raise ValueError("evaluation protocol needs checkpoint hashes by training seed")
    for training_seed, digest in checkpoints.items():
        try:
            int(training_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint training-seed keys must be integers") from exc
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("evaluation protocol contains an invalid checkpoint SHA")
    cohort_contract = _require_mapping(
        evaluation.get("cohort_contract"), "evaluation_protocol.cohort_contract"
    )
    if cohort_contract.get("split") not in ("train", "validation", "heldout"):
        raise ValueError("evaluation cohort contract split must be train, validation, or heldout")
    pair_ids = cohort_contract.get("pair_ids")
    if (
        not isinstance(pair_ids, list)
        or not pair_ids
        or any(not isinstance(pair_id, str) or not pair_id for pair_id in pair_ids)
        or len(set(pair_ids)) != len(pair_ids)
    ):
        raise ValueError("evaluation cohort contract needs unique nonempty pair_ids")
    for key in ("placement_manifest_sha256", "trajectory_manifest_sha256"):
        digest = cohort_contract.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"evaluation cohort contract has invalid {key}")
    return primary, primary_sha, evaluation, evaluation_sha, payload


def _resolve_declared_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _declared_file_digest(
    file_hashes: Mapping[str, Any],
    path: Path,
    *,
    roots: tuple[Path, ...],
) -> str:
    candidates = {path.resolve().as_posix()}
    for root in roots:
        try:
            candidates.add(path.resolve().relative_to(root.resolve()).as_posix())
        except ValueError:
            pass
    matches = {str(file_hashes[key]) for key in candidates if key in file_hashes}
    if not matches:
        raise ValueError(f"evaluation cohort lacks a file hash for {path}")
    if len(matches) != 1:
        raise ValueError(f"evaluation cohort has conflicting file hashes for {path}")
    return next(iter(matches))


def _verify_declared_file(
    file_hashes: Mapping[str, Any],
    path: Path,
    *,
    roots: tuple[Path, ...],
) -> str:
    if not path.is_file():
        raise ValueError(f"required evaluation artifact is missing: {path}")
    expected = _declared_file_digest(file_hashes, path, roots=roots)
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"file hash mismatch for {path}: expected {expected}, got {actual}")
    return actual


@dataclass(frozen=True)
class AcceptedEvaluationPair:
    placement: GlassPlacement
    records: Mapping[str, PairedTrajectoryRecord]
    source_state_path: Path
    nominal_arrays_path: Path
    oracle_arrays_path: Path
    control_arrays_path: Path
    onpath_state_path: Path
    offpath_state_path: Path
    matched_robot_state_path: Path
    controller_state_path: Path
    onpath_model_xml_path: Path | None = None
    offpath_model_xml_path: Path | None = None

    @property
    def family(self) -> str:
        return str(self.placement.metadata["geometry_family"])

    @property
    def horizon(self) -> int:
        value = self.records["nominal_catastrophe"].trigger_horizon_actions
        assert value is not None
        return int(value)


@dataclass(frozen=True)
class EvaluationContract:
    placement_manifest: Path
    trajectory_manifest: Path
    cohort_path: Path
    protocol_path: Path
    split: str
    primary_protocol_sha256: str
    evaluation_protocol_sha256: str
    primary_protocol: Mapping[str, Any]
    evaluation_protocol: Mapping[str, Any]
    checkpoint_metadata: Mapping[str, Any]
    pairs: tuple[AcceptedEvaluationPair, ...]
    reference_manifests: Mapping[str, Path]

    @property
    def horizon(self) -> int:
        return int(self.checkpoint_metadata["trigger_horizon_actions"])


def _load_reference_manifests(
    cohort: Mapping[str, Any],
    *,
    cohort_path: Path,
    checkpoint_metadata: Mapping[str, Any],
    selected_records: list[PairedTrajectoryRecord],
    selected_split: str,
) -> dict[str, Path]:
    if selected_split != "heldout":
        return {}
    references = _require_mapping(
        cohort.get("reference_manifests"), "heldout reference_manifests"
    )
    selected_pairs = {record.pair_id for record in selected_records}
    selected_placements = {record.placement_id for record in selected_records}
    selected_sources = {record.source_state_sha256 for record in selected_records}
    development_validation_overlap = bool(
        cohort.get("development_only") is True
        and cohort.get("checkpoint_validation_role") == "selected_heldout_cohort"
    )
    resolved: dict[str, Path] = {}
    for split, metadata_key in (
        ("train", "train_manifest_sha256"),
        ("validation", "validation_manifest_sha256"),
    ):
        if split == "validation" and development_validation_overlap:
            selected_sha = str(cohort.get("trajectory_manifest_sha256", ""))
            if checkpoint_metadata.get(metadata_key) != selected_sha:
                raise ValueError(
                    "development checkpoint validation manifest is not the selected "
                    "heldout cohort"
                )
            # Pilot C's oracle-timed controller does not consume learned timing
            # or learned recovery actions.  Permit the explicitly labeled
            # development cohort to have calibrated the otherwise protocol-
            # required checkpoint while retaining the independent train leak
            # check below.  This mode must not be reported as heldout learned
            # policy generalization.
            continue
        entry = _require_mapping(references.get(split), f"reference_manifests.{split}")
        path = _resolve_declared_path(str(entry.get("path", "")), relative_to=cohort_path.parent)
        expected = str(entry.get("sha256", ""))
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"{split} reference manifest is missing or has a hash mismatch")
        if checkpoint_metadata.get(metadata_key) != expected:
            raise ValueError(f"checkpoint {metadata_key} disagrees with the cohort reference")
        records = read_trajectory_manifest(path)
        if {record.split for record in records} != {split}:
            raise ValueError(f"{split} reference manifest contains another split")
        leaked = {
            "pair_ids": sorted(selected_pairs & {record.pair_id for record in records}),
            "placement_ids": sorted(
                selected_placements & {record.placement_id for record in records}
            ),
            "source_states": sorted(
                selected_sources & {record.source_state_sha256 for record in records}
            ),
        }
        if any(leaked.values()):
            raise ValueError(f"heldout cohort leaks into {split}: {leaked}")
        resolved[split] = path
    return resolved


def load_evaluation_contract(
    *,
    placement_manifest: str | Path,
    trajectory_manifest: str | Path,
    evaluation_cohort: str | Path,
    protocol: str | Path,
    checkpoint_metadata: Mapping[str, Any],
    checkpoint_sha256: str,
    base_checkpoint_revision: str,
    unnorm_key: str,
) -> EvaluationContract:
    """Load and fail-close the complete accepted evaluation contract."""

    placement_path = Path(placement_manifest).resolve()
    trajectory_path = Path(trajectory_manifest).resolve()
    cohort_path = Path(evaluation_cohort).resolve()
    protocol_path = Path(protocol).resolve()
    for path, label in (
        (placement_path, "placement manifest"),
        (trajectory_path, "trajectory manifest"),
        (cohort_path, "evaluation cohort"),
        (protocol_path, "protocol"),
    ):
        if not path.is_file():
            raise ValueError(f"required {label} is missing: {path}")

    _validate_primary_checkpoint_metadata(dict(checkpoint_metadata))
    placements, placement_payload = read_placement_manifest(placement_path)
    records = read_trajectory_manifest(trajectory_path)
    if {record.schema_version for record in records} != {SCHEMA_VERSION}:
        raise ValueError("evaluation accepts only schema-v2 trajectory manifests")
    primary, primary_sha, evaluation, evaluation_sha, _ = _load_protocol(protocol_path)
    cohort = json.loads(cohort_path.read_text())
    if cohort.get("schema_version") != EVALUATION_COHORT_SCHEMA_VERSION:
        raise ValueError(
            f"evaluation cohort must use schema {EVALUATION_COHORT_SCHEMA_VERSION}"
        )
    if cohort.get("kind") != "glass_recovery_accepted_evaluation_cohort":
        raise ValueError("evaluation cohort is not accepted-pair scoped")
    split = str(cohort.get("split", ""))
    if split not in ("train", "validation", "heldout"):
        raise ValueError("evaluation cohort split must be train, validation, or heldout")
    if cohort.get("placement_manifest_sha256") != file_sha256(placement_path):
        raise ValueError("evaluation cohort placement manifest hash mismatch")
    if cohort.get("trajectory_manifest_sha256") != file_sha256(trajectory_path):
        raise ValueError("evaluation cohort trajectory manifest hash mismatch")
    if cohort.get("primary_protocol_sha256") != primary_sha:
        raise ValueError("evaluation cohort primary protocol hash mismatch")
    if cohort.get("evaluation_protocol_sha256") != evaluation_sha:
        raise ValueError("evaluation cohort evaluation protocol hash mismatch")
    frozen_cohort = _require_mapping(
        evaluation.get("cohort_contract"), "evaluation_protocol.cohort_contract"
    )
    if frozen_cohort.get("split") != split:
        raise ValueError("evaluation cohort split differs from the frozen protocol")
    if frozen_cohort.get("placement_manifest_sha256") != file_sha256(placement_path):
        raise ValueError("placement manifest differs from the frozen evaluation protocol")
    if frozen_cohort.get("trajectory_manifest_sha256") != file_sha256(trajectory_path):
        raise ValueError("trajectory manifest differs from the frozen evaluation protocol")

    collection_summary_path = trajectory_path.parent / "collection_summary.json"
    if not collection_summary_path.is_file():
        raise ValueError("trajectory manifest needs sibling collection_summary.json")
    collection_summary = json.loads(collection_summary_path.read_text())
    collection_metadata = _require_mapping(
        collection_summary.get("metadata"), "collection summary metadata"
    )
    if collection_summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("collection summary schema disagrees with evaluation")
    if collection_metadata.get("primary_protocol_sha256") != primary_sha:
        raise ValueError("collection and supplied primary protocol differ")
    if collection_metadata.get("primary_protocol") != primary:
        raise ValueError("collection primary protocol body differs")
    if collection_metadata.get("placement_design_sha256") != canonical_sha256(
        placement_payload
    ):
        raise ValueError("collection placement design identity is invalid")

    metadata = dict(checkpoint_metadata)
    if metadata.get("protocol_sha256") != primary_sha:
        raise ValueError("checkpoint protocol does not match evaluation protocol")
    if metadata.get("trajectory_schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint schema does not match trajectory schema")
    if metadata.get("base_resolved_revision") != base_checkpoint_revision:
        raise ValueError("configured Base revision does not match checkpoint")
    if metadata.get("unnorm_key") != unnorm_key:
        raise ValueError("configured Base unnorm key does not match checkpoint")
    expected_checkpoint_sha = evaluation["checkpoint_sha256_by_training_seed"].get(
        str(metadata.get("seed"))
    )
    if expected_checkpoint_sha != checkpoint_sha256:
        raise ValueError(
            "checkpoint file hash/training seed is outside the frozen evaluation protocol"
        )
    threshold = metadata.get("calibration", {}).get("threshold")
    if (
        not isinstance(threshold, (int, float))
        or not np.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ValueError("checkpoint calibration threshold is missing or invalid")
    checkpoint_identity = _require_mapping(
        collection_metadata.get("checkpoint_identity"), "collection Base identity"
    )
    if checkpoint_identity.get("resolved_revision") != base_checkpoint_revision:
        raise ValueError("collection Base revision does not match checkpoint/configuration")
    policy_protocol = _require_mapping(primary.get("policy"), "primary protocol policy")
    if policy_protocol.get("unnorm_key") != unnorm_key:
        raise ValueError("primary protocol Base unnorm key mismatch")

    horizons = {record.trigger_horizon_actions for record in records}
    checkpoint_h = metadata.get("trigger_horizon_actions")
    if horizons != {checkpoint_h}:
        raise ValueError(f"trajectory manifest H={horizons} disagrees with checkpoint H={checkpoint_h}")
    if primary.get("precrash_horizon_actions") != checkpoint_h:
        raise ValueError("primary protocol H disagrees with checkpoint")
    if split == "validation" and metadata.get("validation_manifest_sha256") != file_sha256(
        trajectory_path
    ):
        raise ValueError("checkpoint validation manifest does not match evaluation manifest")
    if split == "train" and metadata.get("train_manifest_sha256") != file_sha256(
        trajectory_path
    ):
        raise ValueError("checkpoint train manifest does not match evaluation manifest")

    placements_by_id = {placement.placement_id: placement for placement in placements}
    records_by_pair: dict[str, list[PairedTrajectoryRecord]] = {}
    for record in records:
        records_by_pair.setdefault(record.pair_id, []).append(record)
    cohort_rows = cohort.get("pairs")
    if not isinstance(cohort_rows, list) or not cohort_rows:
        raise ValueError("evaluation cohort must list at least one accepted pair")
    cohort_pair_ids = [
        str(_require_mapping(row, "evaluation cohort pair").get("pair_id", ""))
        for row in cohort_rows
    ]
    if set(cohort_pair_ids) != set(frozen_cohort["pair_ids"]):
        raise ValueError("evaluation cohort pair selection differs from the frozen protocol")
    file_hashes = _require_mapping(cohort.get("file_sha256"), "cohort file_sha256")
    roots = (cohort_path.parent, placement_path.parent, trajectory_path.parent)
    accepted_pairs: list[AcceptedEvaluationPair] = []
    seen: set[str] = set()
    for row_value in cohort_rows:
        row = _require_mapping(row_value, "evaluation cohort pair")
        pair_id = str(row.get("pair_id", ""))
        placement_id = str(row.get("placement_id", ""))
        if not pair_id or pair_id in seen:
            raise ValueError(f"empty or duplicate cohort pair_id {pair_id!r}")
        seen.add(pair_id)
        group = records_by_pair.get(pair_id)
        if group is None:
            raise ValueError(f"cohort pair {pair_id} is not accepted by trajectory manifest")
        validation = validate_primary_pair(group)
        if validation["pair_id"] != pair_id:
            raise ValueError(f"accepted pair identity mismatch for {pair_id}")
        by_kind = {record.trajectory_kind: record for record in group}
        if set(by_kind) != {
            "nominal_catastrophe", "oracle_recovery", "off_path_control"
        }:
            raise ValueError(
                f"cohort pair {pair_id} primary evaluation cannot include auxiliary branches"
            )
        oracle = by_kind["oracle_recovery"]
        nominal = by_kind["nominal_catastrophe"]
        if not oracle.oracle_verified:
            raise ValueError(f"cohort pair {pair_id} has no verified oracle")
        if nominal.branch_start_state_sha256 != oracle.branch_start_state_sha256:
            raise ValueError(f"cohort pair {pair_id} nominal/oracle state mismatch")
        placement = placements_by_id.get(placement_id)
        if placement is None or any(record.placement_id != placement_id for record in group):
            raise ValueError(f"cohort pair {pair_id} placement is unaccepted or mismatched")
        family = placement.metadata.get("geometry_family")
        expected_fields = {
            "split": placement.split,
            "task_suite": placement.task_suite,
            "task_id": placement.task_id,
            "family": family,
            "source_state_sha256": placement.source_state_sha256,
        }
        if family in (None, ""):
            raise ValueError(f"placement {placement_id} lacks geometry_family")
        mismatches = {
            key: (row.get(key), expected)
            for key, expected in expected_fields.items()
            if row.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"cohort pair {pair_id} split/task/family mismatch: {mismatches}")
        if placement.split != split or any(record.split != split for record in group):
            raise ValueError(f"cohort pair {pair_id} is outside declared split {split}")
        if any(
            record.task_suite != placement.task_suite
            or record.task_id != placement.task_id
            or record.instruction != placement.instruction
            or record.source_state_sha256 != placement.source_state_sha256
            for record in group
        ):
            raise ValueError(f"cohort pair {pair_id} placement/trajectory task mismatch")
        expected_onpath_scene = canonical_sha256([placement.on_path_glass])
        expected_offpath_scene = canonical_sha256([placement.off_path_glass])
        if (
            nominal.scene_sha256 != expected_onpath_scene
            or oracle.scene_sha256 != expected_onpath_scene
            or by_kind["off_path_control"].scene_sha256 != expected_offpath_scene
        ):
            raise ValueError(f"cohort pair {pair_id} placement/trajectory scene mismatch")

        arrays_paths = {
            kind: (trajectory_path.parent / record.arrays_path).resolve()
            for kind, record in by_kind.items()
        }
        pair_roots = {path.parent for path in arrays_paths.values()}
        if len(pair_roots) != 1:
            raise ValueError(f"cohort pair {pair_id} branch artifacts are not co-located")
        pair_root = next(iter(pair_roots))
        source_path = (placement_path.parent / placement.source_state_path).resolve()
        onpath_state_path = pair_root / "precrash_onpath_state.npy"
        offpath_state_path = pair_root / "offpath_start_state.npy"
        matched_robot_state_path = pair_root / "matched_robot_state.npy"
        controller_state_path = pair_root / "controller_state.npz"
        onpath_model_xml_path = pair_root / "onpath_model.xml"
        offpath_model_xml_path = pair_root / "offpath_model.xml"
        required_paths = [
            source_path,
            *arrays_paths.values(),
            onpath_state_path,
            offpath_state_path,
            matched_robot_state_path,
            controller_state_path,
        ]
        if "model_xml_sha256" in nominal.metadata.get("branch_start_hashes", {}):
            required_paths.append(onpath_model_xml_path)
        if "model_xml_sha256" in by_kind["off_path_control"].metadata.get(
            "branch_start_hashes", {}
        ):
            required_paths.append(offpath_model_xml_path)
        for required_path in required_paths:
            _verify_declared_file(file_hashes, required_path, roots=roots)

        source_state = np.load(source_path, allow_pickle=False)
        onpath_state = np.load(onpath_state_path, allow_pickle=False)
        offpath_state = np.load(offpath_state_path, allow_pickle=False)
        matched_robot_state = np.load(matched_robot_state_path, allow_pickle=False)
        with np.load(controller_state_path, allow_pickle=False) as archive:
            controller_state = {key: archive[key] for key in archive.files}
        if array_sha256(source_state) != placement.source_state_sha256:
            raise ValueError(f"cohort pair {pair_id} source-state content hash mismatch")
        if array_sha256(onpath_state) != nominal.branch_start_state_sha256:
            raise ValueError(f"cohort pair {pair_id} exact on-path state hash mismatch")
        control = by_kind["off_path_control"]
        if array_sha256(offpath_state) != control.branch_start_state_sha256:
            raise ValueError(f"cohort pair {pair_id} exact off-path state hash mismatch")
        if any(
            array_sha256(matched_robot_state) != record.matched_robot_state_sha256
            for record in group
        ):
            raise ValueError(f"cohort pair {pair_id} matched robot-state hash mismatch")
        expected_controller = nominal.metadata.get("controller_state_sha256")
        if _controller_state_sha256(controller_state) != expected_controller:
            raise ValueError(f"cohort pair {pair_id} exact controller hash mismatch")
        for kind, path in arrays_paths.items():
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            validate_episode_arrays(
                arrays, by_kind[kind].n_steps, schema_version=SCHEMA_VERSION
            )

        accepted_pairs.append(AcceptedEvaluationPair(
            placement=placement,
            records=by_kind,
            source_state_path=source_path,
            nominal_arrays_path=arrays_paths["nominal_catastrophe"],
            oracle_arrays_path=arrays_paths["oracle_recovery"],
            control_arrays_path=arrays_paths["off_path_control"],
            onpath_state_path=onpath_state_path,
            offpath_state_path=offpath_state_path,
            matched_robot_state_path=matched_robot_state_path,
            controller_state_path=controller_state_path,
            onpath_model_xml_path=(
                onpath_model_xml_path if onpath_model_xml_path.is_file() else None
            ),
            offpath_model_xml_path=(
                offpath_model_xml_path if offpath_model_xml_path.is_file() else None
            ),
        ))

    selected_records = [record for pair in accepted_pairs for record in pair.records.values()]
    references = _load_reference_manifests(
        cohort,
        cohort_path=cohort_path,
        checkpoint_metadata=metadata,
        selected_records=selected_records,
        selected_split=split,
    )
    return EvaluationContract(
        placement_manifest=placement_path,
        trajectory_manifest=trajectory_path,
        cohort_path=cohort_path,
        protocol_path=protocol_path,
        split=split,
        primary_protocol_sha256=primary_sha,
        evaluation_protocol_sha256=evaluation_sha,
        primary_protocol=primary,
        evaluation_protocol=evaluation,
        checkpoint_metadata=metadata,
        pairs=tuple(accepted_pairs),
        reference_manifests=references,
    )


def _load_controller_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _make_oracle_controller(
    pair: AcceptedEvaluationPair,
    glasses: list[dict],
    obs: Mapping[str, Any],
) -> DetourComplete:
    config = pair.records["oracle_recovery"].metadata.get("oracle_config")
    if not isinstance(config, Mapping):
        raise ValueError(f"accepted pair {pair.placement.placement_id} lacks oracle_config")
    required = {
        "side", "lane_margin", "transit_z", "descend_off",
        "orientation_target", "path_aligned", "grasp_xy_offset",
    }
    if not required.issubset(config):
        raise ValueError(f"accepted oracle config lacks {sorted(required - set(config))}")
    bowl = np.asarray(obs[f"{TARGET}_pos"], dtype=float)
    plate = np.asarray(obs[f"{PLATE}_pos"], dtype=float)
    return DetourComplete(
        _controller_glass(glasses[len(glasses) // 2]),
        bowl,
        plate,
        side=float(config["side"]),
        lane_margin=float(config["lane_margin"]),
        transit_z=float(config["transit_z"]),
        descend_off=float(config["descend_off"]),
        leg_cap=140,
        target_name=TARGET,
        orientation_target=config["orientation_target"],
        path_aligned=bool(config["path_aligned"]),
        grasp_xy_offset=config["grasp_xy_offset"],
        departure_clearance=float(config.get("departure_clearance", 0.06)),
    )


class EvaluationPolicy:
    """One trace-complete evaluation condition sharing the frozen risk head."""

    def __init__(
        self,
        base,
        risk_model: GlassRecoveryPolicy,
        *,
        condition: str,
        pair: AcceptedEvaluationPair,
        glasses: list[dict],
        raw_obs: Mapping[str, Any],
        oracle_trigger_step: int,
        stop_action: np.ndarray,
    ):
        if condition not in ALL_CONDITIONS:
            raise ValueError(f"unknown condition {condition}")
        if condition == "blocked_safe_abort":
            raise ValueError(
                "blocked_safe_abort requires a separately accepted auxiliary manifest"
            )
        self.base = base
        self.risk_model = risk_model
        self.condition = condition
        self.pair = pair
        self.glasses = glasses
        self.oracle_trigger_step = int(oracle_trigger_step)
        self.stop_action = np.asarray(stop_action, dtype=np.float32)
        if self.stop_action.shape != (7,):
            raise ValueError("always-stop action must have shape [7]")
        self.threshold = float(risk_model.checkpoint_metadata["calibration"]["threshold"])
        self.risk_index = risk_model.risk_index
        self.controller = None
        if condition == "risk_gate_retreat_hold":
            self.controller = RetreatHold()
        elif condition in ("risk_gate_oracle_recovery", "oracle_timed_oracle_recovery"):
            self.controller = _make_oracle_controller(pair, glasses, raw_obs)
        self.reset(raw_obs)

    @property
    def resize_size(self) -> int:
        return self.base.resize_size

    @property
    def privileged(self) -> bool:
        return self.condition in PRIVILEGED_CONDITIONS

    def reset(self, raw_obs: Mapping[str, Any]) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset()
        self.step_index = -1
        self.mode = "nominal"
        self.previous_risk: float | None = None
        self.first_threshold_crossing_step: int | None = None
        self.first_intervention_step: int | None = None
        self.ownership_age: int | None = None
        self.decisions: list[dict] = []

    def _instruction(self, instruction: str) -> str:
        if self.condition == "generic_careful":
            return compose_instruction("glass", "generic_careful", instruction)
        if self.condition == "hazard_specific_careful":
            return compose_instruction("glass", "hazard_specific", instruction)
        if self.condition == "legacy_e14_careful":
            return f"{CAREFUL_PROMPT_PREFIX} {instruction}"
        return instruction

    def act(
        self,
        policy_obs: Mapping[str, Any],
        raw_obs: Mapping[str, Any],
        instruction: str,
        *,
        captured_base_tte: int | None,
    ) -> tuple[np.ndarray, dict]:
        self.step_index += 1
        nominal = np.asarray(
            self.base.act(dict(policy_obs), self._instruction(instruction)), dtype=np.float32
        )
        hidden = getattr(self.base, "last_hidden", None)
        if hidden is None:
            raise RuntimeError("Base policy did not expose hidden state for the risk trace")
        prediction = self.risk_model._predict(
            np.asarray(hidden, dtype=np.float32),
            np.asarray(policy_obs["state"], dtype=np.float32),
            nominal,
        )
        risk_vector = np.asarray(prediction["risk"], dtype=float)
        recovery = np.asarray(prediction["recovery_action"], dtype=np.float32)
        selected_risk = float(risk_vector[self.risk_index])
        threshold_crossing = bool(
            selected_risk >= self.threshold
            and (self.previous_risk is None or self.previous_risk < self.threshold)
        )
        if threshold_crossing and self.first_threshold_crossing_step is None:
            self.first_threshold_crossing_step = self.step_index
        learned_crossing = bool(self.mode == "nominal" and threshold_crossing)
        uses_learned_timing = self.condition in (
            "risk_gate_retreat_hold",
            "full_learned_gate_recovery",
            "risk_gate_oracle_recovery",
        )
        uses_oracle_timing = self.condition in (
            "oracle_timed_learned_recovery",
            "oracle_timed_oracle_recovery",
        )
        first_trigger = bool(
            self.mode == "nominal"
            and (
                (uses_learned_timing and learned_crossing)
                or (uses_oracle_timing and self.step_index == self.oracle_trigger_step)
            )
        )
        trigger_type = None
        if first_trigger:
            trigger_type = "learned_risk" if uses_learned_timing else "oracle_timing"
            self.mode = "recovery_latched"
            self.first_intervention_step = self.step_index
            self.ownership_age = 0
            if self.controller is not None:
                self.controller.engage(dict(raw_obs))
        elif self.mode == "recovery_latched":
            assert self.ownership_age is not None
            self.ownership_age += 1

        if self.condition == "always_stop":
            executed = self.stop_action.copy()
            mode = "always_stop"
            intervened = True
            first_trigger = self.step_index == 0
            trigger_type = "sanity_always_stop" if first_trigger else None
            if self.first_intervention_step is None:
                self.first_intervention_step = 0
        elif self.mode == "recovery_latched":
            intervened = True
            if self.controller is not None:
                executed = np.asarray(self.controller.step(dict(raw_obs)), dtype=np.float32)
                mode = (
                    "recovery_latched_retreat_hold"
                    if self.condition == "risk_gate_retreat_hold"
                    else "recovery_latched_oracle"
                )
            else:
                executed = recovery
                mode = "recovery_latched_learned"
        else:
            intervened = False
            executed = nominal
            mode = "nominal"
        executed = np.clip(executed, -1.0, 1.0).astype(np.float32)
        decision = {
            "step": self.step_index,
            "risk_horizons_actions": list(RISK_HORIZONS),
            "risk_vector": [float(value) for value in risk_vector],
            "selected_risk_horizon_actions": self.risk_model.risk_horizon,
            "selected_risk_probability": selected_risk,
            "risk_threshold": self.threshold,
            "nominal_action": nominal.astype(float).tolist(),
            "recovery_action": recovery.astype(float).tolist(),
            "executed_action": executed.astype(float).tolist(),
            "mode": mode,
            "intervened": intervened,
            "first_trigger": first_trigger,
            "trigger_type": trigger_type,
            "first_trigger_step": self.first_intervention_step,
            "threshold_crossing": threshold_crossing,
            "first_threshold_crossing": (
                threshold_crossing
                and self.first_threshold_crossing_step == self.step_index
            ),
            "first_threshold_crossing_step": self.first_threshold_crossing_step,
            "ownership_age": self.ownership_age,
            "captured_base_time_to_catastrophe_actions": captured_base_tte,
        }
        self.decisions.append(decision)
        self.previous_risk = selected_risk
        return executed, decision


def _build_attributed_predicates(glasses: list[dict], sim_view):
    predicates = []
    for index, spec in enumerate(_glass_predicate_specs(glasses)):
        predicate = build_predicate(spec)
        prime_predicate(predicate, sim_view)
        object_name = spec.params.get("object_name")
        suffix = f":{object_name}" if object_name else ""
        predicates.append((f"{index}:{spec.type}{suffix}", predicate))
    return predicates


def _runtime_hashes(env, obs: Mapping[str, Any]) -> dict[str, str]:
    return _branch_start_hashes(env, obs)


def _restore_identity_evidence(
    actual: Mapping[str, str], expected: Mapping[str, str], *, label: str
) -> dict[str, Any]:
    """Require every continuation identity field declared by the cohort."""

    required = ["simulator_state_sha256", "controller_state_sha256"]
    for key in ("continuation_state_sha256", "model_xml_sha256"):
        if key in expected:
            required.append(key)
    # Legacy cohorts did not serialize observable history, so their observation
    # hash remains diagnostic.  New continuation snapshots do serialize it and
    # therefore require exact observation identity.
    if "continuation_state_sha256" in expected and "observation_sha256" in expected:
        required.append("observation_sha256")
    mismatched = [key for key in required if actual.get(key) != expected.get(key)]
    if mismatched:
        raise RuntimeError(f"{label} restore mismatch for {mismatched}")
    expected_observation = expected.get("observation_sha256")
    actual_observation = actual.get("observation_sha256")
    return {
        "simulator_controller_exact": True,
        "continuation_exact": True,
        "observation_exact": expected_observation == actual_observation,
        "expected_observation_sha256": expected_observation,
        "restored_observation_sha256": actual_observation,
    }


def _reset_episode(
    env,
    pair: AcceptedEvaluationPair,
    *,
    mode: str,
    regime: str,
    settle_steps: int,
) -> tuple[dict, list[dict], int, int, dict[str, Any] | None]:
    placement = pair.placement
    glasses = (
        [placement.on_path_glass] if regime == "treatment" else [placement.off_path_glass]
    )
    if mode == "source_to_task":
        obs = env.reset_to(np.load(pair.source_state_path, allow_pickle=False), movable_objects=glasses)
        for _ in range(settle_steps):
            obs, _, _, _ = env.step(env.dummy_action())
        deadline = int(
            pair.records["nominal_catastrophe"].metadata["source_scan_precrash_index"]
        )
        collision_timeline = int(
            pair.records["nominal_catastrophe"].metadata["source_scan_collision_step"]
        )
        return obs, glasses, deadline, collision_timeline, None
    if mode != "exact_anchor":
        raise ValueError(f"unknown evaluation mode {mode}")
    state_path = pair.onpath_state_path if regime == "treatment" else pair.offpath_state_path
    exact_state = np.load(state_path, allow_pickle=False)
    exact_model_xml_path = (
        pair.onpath_model_xml_path if regime == "treatment"
        else pair.offpath_model_xml_path
    )
    if exact_model_xml_path is not None:
        obs = env.reset_to_exact(
            exact_state, model_xml=exact_model_xml_path.read_text()
        )
    else:
        obs = env.reset_to_exact(exact_state, movable_objects=glasses)
    controller_state = _load_controller_state(pair.controller_state_path)
    try:
        restored_obs = env.restore_controller_state(
            controller_state, restore_observables=(regime == "treatment")
        )
    except TypeError as exc:
        if "restore_observables" not in str(exc):
            raise
        # Compatibility for zero-GPU test doubles and legacy external adapters.
        restored_obs = env.restore_controller_state(controller_state)
    if restored_obs is not None:
        obs = restored_obs
    expected_record = pair.records[
        "nominal_catastrophe" if regime == "treatment" else "off_path_control"
    ]
    expected = expected_record.metadata["branch_start_hashes"]
    actual = _runtime_hashes(env, obs)
    restore_evidence = _restore_identity_evidence(
        actual,
        expected,
        label=f"exact-anchor {placement.placement_id}/{regime}",
    )
    model = env._raw_model()
    matched_robot_state = env._strip_movable_state(
        exact_state, int(model.nq), int(model.nv), len(glasses)
    )
    if array_sha256(matched_robot_state) != expected_record.matched_robot_state_sha256:
        raise RuntimeError(
            f"exact-anchor matched robot/task state mismatch for "
            f"{placement.placement_id}/{regime}"
        )
    return obs, glasses, 0, pair.horizon - 1, restore_evidence


def _save_trigger_state(
    root: Path,
    *,
    env,
    obs: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    state = np.asarray(env.flat_state(), dtype=np.float64)
    controller = env.controller_state()
    root.mkdir(parents=True, exist_ok=False)
    state_path = root / "simulator_state.npy"
    controller_path = root / "controller_state.npz"
    np.save(state_path, state)
    np.savez_compressed(controller_path, **controller)
    identity = {
        **_runtime_hashes(env, obs),
        "simulator_state_file": str(state_path),
        "simulator_state_file_sha256": file_sha256(state_path),
        "controller_state_file": str(controller_path),
        "controller_state_file_sha256": file_sha256(controller_path),
    }
    if hasattr(env, "model_xml"):
        model_xml_path = root / "model.xml"
        model_xml_path.write_text(env.model_xml())
        identity.update(
            model_xml_file=str(model_xml_path),
            model_xml_file_sha256=file_sha256(model_xml_path),
        )
    return identity, state, controller


def _posthoc_oracle(
    env,
    pair: AcceptedEvaluationPair,
    glasses: list[dict],
    state: np.ndarray,
    controller_state: Mapping[str, np.ndarray],
    expected_trigger_identity: Mapping[str, Any],
    *,
    max_steps: int,
) -> dict[str, Any]:
    model_xml_file = expected_trigger_identity.get("model_xml_file")
    if model_xml_file:
        obs = env.reset_to_exact(state, model_xml=Path(str(model_xml_file)).read_text())
    else:
        obs = env.reset_to_exact(state, movable_objects=glasses)
    restored_obs = env.restore_controller_state(dict(controller_state))
    if restored_obs is not None:
        obs = restored_obs
    restored_identity = _runtime_hashes(env, obs)
    expected_runtime_identity = {
        key: str(expected_trigger_identity[key])
        for key in (
            "simulator_state_sha256", "controller_state_sha256",
            "continuation_state_sha256", "observation_sha256", "model_xml_sha256",
        ) if key in expected_trigger_identity
    }
    restore_evidence = _restore_identity_evidence(
        restored_identity,
        expected_runtime_identity,
        label="post-hoc oracle trigger-state",
    )
    controller = _make_oracle_controller(pair, glasses, obs)
    controller.engage(obs)
    predicates = _build_attributed_predicates(glasses, env.sim_view)
    peak_glass_force = 0.0
    for step in range(max_steps):
        action = np.asarray(controller.step(obs), dtype=np.float32)
        obs, _, _, _ = env.step(action.tolist())
        peak_glass_force = max(peak_glass_force, _glass_force(env.sim_view, glasses))
        attribution = [name for name, predicate in predicates if predicate(env.sim_view)]
        if attribution:
            return {
                "status": "completed",
                "oracle_verified_recoverable": False,
                "catastrophe": True,
                "task_success": False,
                "timeout": False,
                "steps": step + 1,
                "catastrophe_predicates": attribution,
                "peak_glass_force_n": float(peak_glass_force),
                "restore_identity": restore_evidence,
            }
        if env.sim_view.libero_done:
            return {
                "status": "completed",
                "oracle_verified_recoverable": True,
                "catastrophe": False,
                "task_success": True,
                "timeout": False,
                "steps": step + 1,
                "catastrophe_predicates": [],
                "peak_glass_force_n": float(peak_glass_force),
                "restore_identity": restore_evidence,
            }
    return {
        "status": "completed",
        "oracle_verified_recoverable": False,
        "catastrophe": False,
        "task_success": False,
        "timeout": True,
        "steps": max_steps,
        "catastrophe_predicates": [],
        "peak_glass_force_n": float(peak_glass_force),
        "restore_identity": restore_evidence,
    }


def run_evaluation_episode(
    *,
    env,
    base,
    risk_model: GlassRecoveryPolicy,
    pair: AcceptedEvaluationPair,
    mode: str,
    regime: str,
    condition: str,
    rollout_seed: int,
    training_seed: int,
    settle_steps: int,
    max_steps: int,
    trigger_artifact_root: Path,
    paired_base_collision_step: int | None = None,
    use_paired_base_timeline: bool = False,
) -> dict[str, Any]:
    """Run one fake-env-testable, trace-complete P0-D episode."""

    random.seed(int(rollout_seed))
    np.random.seed(int(rollout_seed))
    try:
        import torch

        torch.manual_seed(int(rollout_seed))
    except ImportError:  # pragma: no cover - production recovery checkpoints require torch
        pass
    if hasattr(env, "seed"):
        env.seed(int(rollout_seed))
    (
        obs,
        glasses,
        recoverability_deadline,
        collision_timeline,
        anchor_restore,
    ) = _reset_episode(env, pair, mode=mode, regime=regime, settle_steps=settle_steps)
    predicates = _build_attributed_predicates(glasses, env.sim_view)
    policy = EvaluationPolicy(
        base,
        risk_model,
        condition=condition,
        pair=pair,
        glasses=glasses,
        raw_obs=obs,
        oracle_trigger_step=recoverability_deadline if mode == "source_to_task" else 0,
        stop_action=np.asarray(env.dummy_action(), dtype=np.float32),
    )
    peak_glass_force = 0.0
    peak_global_force = 0.0
    catastrophe = False
    task_success = False
    attribution: list[str] = []
    trace: list[dict] = []
    trigger_identity = None
    trigger_state = None
    trigger_controller = None
    for step in range(max_steps):
        policy_obs = env.policy_observation(obs, policy.resize_size)
        timing_collision_step = (
            paired_base_collision_step if use_paired_base_timeline else collision_timeline
        )
        captured_tte = (
            None
            if timing_collision_step is None
            else int(timing_collision_step) - step + 1
        )
        action, decision = policy.act(
            policy_obs,
            obs,
            pair.placement.instruction,
            captured_base_tte=(
                captured_tte if captured_tte is not None and captured_tte >= 1 else None
            ),
        )
        if decision["first_trigger"]:
            artifact_dir = trigger_artifact_root / (
                f"{mode}__{condition}__{regime}__{pair.placement.placement_id}__"
                f"seed_{rollout_seed}"
            )
            trigger_identity, trigger_state, trigger_controller = _save_trigger_state(
                artifact_dir, env=env, obs=obs
            )
        obs, _, _, _ = env.step(action.tolist())
        glass_force = _glass_force(env.sim_view, glasses)
        global_force = float(getattr(env.sim_view, "peak_force", 0.0))
        peak_glass_force = max(peak_glass_force, glass_force)
        peak_global_force = max(peak_global_force, global_force)
        attribution = [name for name, predicate in predicates if predicate(env.sim_view)]
        decision = {
            **decision,
            "glass_force_n_after_action": float(glass_force),
            "global_force_n_after_action": float(global_force),
            "catastrophe_predicates_after_action": attribution,
            "task_success_after_action": bool(env.sim_view.libero_done),
        }
        trace.append(decision)
        if attribution:
            catastrophe = True
            break
        if env.sim_view.libero_done:
            task_success = True
            break
    timeout = not catastrophe and not task_success
    safe_noncompletion = timeout
    # Ordinary max-step exhaustion is never promoted to safe_abort.  The only
    # appendix condition with that label needs its own verified blocked cohort.
    safe_abort = False
    intervened = any(bool(row["intervened"]) for row in trace)
    first_intervention = next((row for row in trace if row["first_trigger"]), None)
    first_threshold_crossing = next(
        (row for row in trace if row["first_threshold_crossing"]), None
    )
    first_tte = (
        None if first_intervention is None
        else first_intervention["captured_base_time_to_catastrophe_actions"]
    )
    trigger_by_certified_deadline = bool(
        first_intervention is not None
        and int(first_intervention["step"]) <= recoverability_deadline
    )
    if trigger_state is None:
        posthoc = {"status": "not_applicable_no_intervention"}
    elif (
        mode == "exact_anchor"
        and regime == "treatment"
        and first_intervention is not None
        and int(first_intervention["step"]) == 0
    ):
        posthoc = {
            "status": "accepted_anchor_witness",
            "oracle_verified_recoverable": True,
            "evidence": "accepted exact-state independent oracle recapture",
        }
    else:
        assert trigger_controller is not None
        assert trigger_identity is not None
        posthoc = _posthoc_oracle(
            env,
            pair,
            glasses,
            trigger_state,
            trigger_controller,
            trigger_identity,
            max_steps=max_steps,
        )
    timely_trigger = bool(
        trigger_by_certified_deadline
        and posthoc.get("oracle_verified_recoverable") is True
        and (not use_paired_base_timeline or paired_base_collision_step is not None)
    )
    return {
        "evaluation_mode": mode,
        "estimand_role": "paper_primary" if mode == "source_to_task" else "component_diagnostic",
        "condition": condition,
        "condition_role": condition_role(condition),
        "privileged": condition in PRIVILEGED_CONDITIONS,
        "regime": regime,
        "pair_id": pair.records["nominal_catastrophe"].pair_id,
        "placement_id": pair.placement.placement_id,
        "source_state_sha256": pair.placement.source_state_sha256,
        "cluster_id": pair.placement.cluster_id,
        "split": pair.placement.split,
        "task_suite": pair.placement.task_suite,
        "task_id": pair.placement.task_id,
        "family": pair.family,
        "rollout_seed": int(rollout_seed),
        "training_seed": int(training_seed),
        "task_success": task_success,
        "succeeded": task_success,
        "catastrophe": catastrophe,
        "crashed": catastrophe,
        "timeout": timeout,
        "safe_noncompletion": safe_noncompletion,
        "safe_abort": safe_abort,
        "outcome": (
            "catastrophe" if catastrophe else "safe_task_success" if task_success else "timeout"
        ),
        "intervened": intervened,
        "first_intervention_step": (
            None if first_intervention is None else int(first_intervention["step"])
        ),
        "first_threshold_crossing_step": (
            None
            if first_threshold_crossing is None
            else int(first_threshold_crossing["step"])
        ),
        "first_threshold_crossing_time_to_catastrophe_actions": (
            None
            if first_threshold_crossing is None
            else first_threshold_crossing["captured_base_time_to_catastrophe_actions"]
        ),
        "first_intervention_time_to_catastrophe_actions": first_tte,
        "timing_evidence": (
            "paired_evaluation_base_rollout"
            if use_paired_base_timeline
            else "accepted_capture_timeline"
        ),
        "paired_base_collision_step": (
            paired_base_collision_step if use_paired_base_timeline else None
        ),
        "certified_recoverability_deadline_step": recoverability_deadline,
        "latest_recoverability_deadline_step": recoverability_deadline,
        "anchor_restore_identity": anchor_restore,
        "timely_trigger": timely_trigger,
        "post_hoc_trigger_state_oracle": posthoc,
        "trigger_state": trigger_identity,
        "catastrophe_predicate_attribution": attribution,
        "peak_glass_force_n": float(peak_glass_force),
        "peak_global_force_n": float(peak_global_force),
        "steps": len(trace),
        "trace": trace,
    }


def _validate_loaded_base(base, contract: EvaluationContract, unnorm_key: str) -> None:
    identity = getattr(base, "checkpoint_identity", None)
    if not isinstance(identity, Mapping):
        raise RuntimeError("loaded Base policy does not expose checkpoint identity")
    resolved = identity.get("resolved_revision") or identity.get("model_config_commit_hash")
    if resolved != contract.checkpoint_metadata["base_resolved_revision"]:
        raise RuntimeError("loaded Base revision differs from checkpoint/evaluation contract")
    if getattr(getattr(base, "cfg", None), "unnorm_key", None) != unnorm_key:
        raise RuntimeError("loaded Base unnorm key differs from evaluation contract")


def _attach_observed_source_base_timeline(row: dict[str, Any]) -> int | None:
    """Replace accepted-capture TTE labels with this repeat's observed Base event."""

    collision_step = next((
        int(step["step"])
        for step in row["trace"]
        if step["catastrophe_predicates_after_action"]
    ), None)
    for step in row["trace"]:
        tte = None if collision_step is None else collision_step - int(step["step"]) + 1
        step["captured_base_time_to_catastrophe_actions"] = (
            tte if tte is not None and tte >= 1 else None
        )
    crossing = next(
        (step for step in row["trace"] if step["first_threshold_crossing"]), None
    )
    row["first_threshold_crossing_time_to_catastrophe_actions"] = (
        None
        if crossing is None
        else crossing["captured_base_time_to_catastrophe_actions"]
    )
    row["timing_evidence"] = "self_observed_evaluation_base_rollout"
    row["paired_base_collision_step"] = collision_step
    return collision_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placement-manifest", required=True)
    parser.add_argument("--trajectory-manifest", required=True)
    parser.add_argument("--evaluation-cohort", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-checkpoint-revision", required=True)
    parser.add_argument("--unnorm-key", required=True)
    parser.add_argument("--mode", choices=EVALUATION_MODES, nargs="+", default=list(EVALUATION_MODES))
    parser.add_argument("--conditions", choices=ALL_CONDITIONS, nargs="+", default=list(MAIN_CONDITIONS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    if out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {out}; pass --overwrite")
    trigger_root = out.parent / f"{out.stem}_trigger_states"
    if trigger_root.exists():
        raise SystemExit(f"refusing to overwrite trigger-state artifacts at {trigger_root}")
    if len(set(args.mode)) != len(args.mode) or len(set(args.conditions)) != len(args.conditions):
        raise SystemExit("evaluation modes and conditions cannot contain duplicates")
    if (
        "source_to_task" in args.mode
        and any(condition != "base" for condition in args.conditions)
        and "base" not in args.conditions
    ):
        raise SystemExit("source_to_task comparisons require the paired base condition")

    # Verify the frozen checkpoint file identity before deserializing it.  The
    # recovery checkpoint is then loaded on CPU only so its metadata can enter
    # the remaining contract checks; OpenVLA and LIBERO are still untouched.
    checkpoint_digest = file_sha256(args.checkpoint)
    _, _, evaluation_header, _, _ = _load_protocol(Path(args.protocol).resolve())
    if checkpoint_digest not in evaluation_header["checkpoint_sha256_by_training_seed"].values():
        raise SystemExit("checkpoint file hash is outside the frozen evaluation protocol")
    _, checkpoint_metadata = GlassRecoveryNetwork.load_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    contract = load_evaluation_contract(
        placement_manifest=args.placement_manifest,
        trajectory_manifest=args.trajectory_manifest,
        evaluation_cohort=args.evaluation_cohort,
        protocol=args.protocol,
        checkpoint_metadata=checkpoint_metadata,
        checkpoint_sha256=checkpoint_digest,
        base_checkpoint_revision=args.base_checkpoint_revision,
        unnorm_key=args.unnorm_key,
    )
    declared_modes = set(contract.evaluation_protocol["evaluation_modes"])
    declared_conditions = set(contract.evaluation_protocol["conditions"])
    if not set(args.mode).issubset(declared_modes):
        raise SystemExit("requested evaluation mode is outside the frozen protocol")
    if not set(args.conditions).issubset(declared_conditions):
        raise SystemExit("requested condition is outside the frozen protocol")
    if "blocked_safe_abort" in args.conditions:
        raise SystemExit(
            "blocked_safe_abort requires a separate accepted auxiliary manifest/evaluator"
        )

    base = OpenVLAPolicy(
        pretrained_checkpoint=args.base_checkpoint,
        checkpoint_revision=args.base_checkpoint_revision,
        unnorm_key=args.unnorm_key,
        center_crop=True,
        capture_hidden=True,
    )
    _validate_loaded_base(base, contract, args.unnorm_key)
    risk_model = GlassRecoveryPolicy(
        base,
        args.checkpoint,
        risk_horizon=contract.horizon,
        risk_enter_threshold=float(checkpoint_metadata["calibration"]["threshold"]),
    )
    max_steps = int(contract.evaluation_protocol["max_steps"])
    rollout_seeds = [int(seed) for seed in contract.evaluation_protocol["rollout_seeds"]]
    settle_steps = int(contract.primary_protocol["settle_steps"])
    training_seed = int(checkpoint_metadata["seed"])
    envs: dict[tuple[str, int], Any] = {}
    rows: list[dict] = []
    ordered_conditions = [
        *(["base"] if "base" in args.conditions else []),
        *(condition for condition in args.conditions if condition != "base"),
    ]
    for pair in contract.pairs:
        key = (pair.placement.task_suite, pair.placement.task_id)
        if key not in envs:
            envs[key] = LiberoEnv(*key)
        for mode in args.mode:
            for regime in ("treatment", "control"):
                for rollout_seed in rollout_seeds:
                    paired_base_collision_step: int | None = None
                    for condition in ordered_conditions:
                        row = run_evaluation_episode(
                            env=envs[key],
                            base=base,
                            risk_model=risk_model,
                            pair=pair,
                            mode=mode,
                            regime=regime,
                            condition=condition,
                            rollout_seed=rollout_seed,
                            training_seed=training_seed,
                            settle_steps=settle_steps,
                            max_steps=max_steps,
                            trigger_artifact_root=trigger_root,
                            paired_base_collision_step=paired_base_collision_step,
                            use_paired_base_timeline=(
                                mode == "source_to_task" and condition != "base"
                            ),
                        )
                        if condition == "base":
                            paired_base_collision_step = (
                                _attach_observed_source_base_timeline(row)
                                if mode == "source_to_task"
                                else next((
                                    int(step["step"])
                                    for step in row["trace"]
                                    if step["catastrophe_predicates_after_action"]
                                ), None)
                            )
                        rows.append(row)
                        print(
                            f"{mode:14s} {condition:32s} {regime:9s} "
                            f"{pair.placement.placement_id} seed={rollout_seed} "
                            f"{row['outcome']}",
                            flush=True,
                        )

    payload = {
        "schema_version": 3,
        "kind": "glass_recovery_accepted_cohort_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": os.environ.get("CB_CODE_COMMIT"),
        "inputs": {
            "placement_manifest": str(contract.placement_manifest),
            "placement_manifest_sha256": file_sha256(contract.placement_manifest),
            "trajectory_manifest": str(contract.trajectory_manifest),
            "trajectory_manifest_sha256": file_sha256(contract.trajectory_manifest),
            "evaluation_cohort": str(contract.cohort_path),
            "evaluation_cohort_sha256": file_sha256(contract.cohort_path),
            "protocol": str(contract.protocol_path),
            "protocol_file_sha256": file_sha256(contract.protocol_path),
            "primary_protocol_sha256": contract.primary_protocol_sha256,
            "evaluation_protocol_sha256": contract.evaluation_protocol_sha256,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_metadata": checkpoint_metadata,
            "base_checkpoint_identity": base.checkpoint_identity,
        },
        "evaluation_modes": {
            "source_to_task": "paper_primary: original LIBERO source state through completion",
            "exact_anchor": "component_diagnostic: accepted exact-H state/controller restore",
        },
        "independent_cluster": "source_state_sha256",
        "repeat_policy": "nested within placement; never increases independent n",
        "selected_pairs": [
            pair.records["nominal_catastrophe"].pair_id for pair in contract.pairs
        ],
        "episodes": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
