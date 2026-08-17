"""Auditable schema and validation for glass pre-crash paired data.

One placement is a *matched group*, not one rollout.  Every schema-v2 accepted
group has three primary branches starting from the same robot/task state:

``nominal_catastrophe``
    Base OpenVLA continues into the on-path glass and actually triggers a crash
    predicate.
``oracle_recovery``
    The same on-path state is restored and a verified safe, task-completing
    oracle trajectory is executed.
``off_path_control``
    The robot/task state is unchanged, but the same glass is moved outside the
    swept corridor.  The target action is the unchanged nominal action.
``blocked_safe_abort`` (optional auxiliary)
    The same robot/task state is placed in a separately declared blocked scene;
    the accepted target is a stable safe abort.  The record must carry evidence
    for the controller class under which it was judged unrecoverable.

Large arrays and images live outside Git.  JSON records retain hashes and paths
so data leakage and same-state split overlap can be checked without loading
MuJoCo, OpenVLA, or torch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


LEGACY_PLACEMENT_SCHEMA_VERSION = 1
PLACEMENT_SCHEMA_VERSION = 2
LEGACY_TRAJECTORY_SCHEMA_VERSION = 1
SCHEMA_VERSION = 2
SPLITS = ("train", "validation", "heldout")
PRIMARY_TRAJECTORY_KINDS = (
    "nominal_catastrophe",
    "oracle_recovery",
    "off_path_control",
)
AUXILIARY_TRAJECTORY_KINDS = (
    "blocked_safe_abort",
)
TRAJECTORY_KINDS = PRIMARY_TRAJECTORY_KINDS + AUXILIARY_TRAJECTORY_KINDS
RISK_HORIZONS = (1, 3, 5, 10, 20)
HAZARD_TYPES = ("none", "glass", "wall", "other")


def steps_until_event(event_action_index: int, observation_index: int) -> int:
    """Return action distance from pre-action observation ``s_i`` to event action ``c``.

    Rows store the observation before action ``a_i`` and test the event after
    that action.  Consequently the event caused by ``a_c`` is one action away
    from ``s_c``, not zero actions away.
    """

    return int(event_action_index) - int(observation_index) + 1


def exact_h_anchor_index(event_action_index: int, horizon: int) -> int:
    """Return the observation index whose captured suffix has exactly ``horizon`` actions."""

    if int(horizon) < 1:
        raise ValueError("horizon must be positive")
    anchor = int(event_action_index) - int(horizon) + 1
    if anchor < 0:
        raise ValueError(
            f"event action {event_action_index} has no {horizon}-action pre-event anchor"
        )
    return anchor


def array_sha256(value: Sequence[float] | np.ndarray) -> str:
    """Hash an array including dtype and shape, not only its raw bytes."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any] | Sequence[Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GlassPlacement:
    placement_id: str
    split: str
    cluster_id: str
    task_suite: str
    task_id: int
    instruction: str
    source_state_path: str
    source_state_sha256: str
    on_path_glass: dict[str, Any]
    off_path_glass: dict[str, Any]
    blocked_glasses: list[dict[str, Any]]
    nominal_fraction: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"bad split {self.split!r}; expected one of {SPLITS}")
        if not self.placement_id or not self.cluster_id:
            raise ValueError("placement_id and cluster_id are required")
        if not (0.0 < float(self.nominal_fraction) < 1.0):
            raise ValueError("nominal_fraction must lie strictly inside (0, 1)")
        for name, glass in (
            ("on_path_glass", self.on_path_glass),
            ("off_path_glass", self.off_path_glass),
        ):
            _validate_glass(glass, name)
        for index, glass in enumerate(self.blocked_glasses):
            _validate_glass(glass, f"blocked_glasses[{index}]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GlassPlacement":
        return cls(**dict(value))


def _validate_glass(glass: Mapping[str, Any], label: str) -> None:
    missing = {"name", "pos", "size"} - set(glass)
    if missing:
        raise ValueError(f"{label} missing {sorted(missing)}")
    if len(glass["pos"]) != 3:
        raise ValueError(f"{label}.pos must be xyz")
    if len(glass["size"]) not in (2, 3):
        raise ValueError(f"{label}.size must be a MuJoCo cylinder/box size")
    if "movable" in glass and not isinstance(glass["movable"], bool):
        raise ValueError(f"{label}.movable must be boolean when provided")
    numbers = np.asarray([*glass["pos"], *glass["size"]], dtype=float)
    if not np.isfinite(numbers).all() or np.any(np.asarray(glass["size"], dtype=float) <= 0):
        raise ValueError(f"{label} has invalid geometry")


@dataclass(frozen=True)
class PairedTrajectoryRecord:
    pair_id: str
    placement_id: str
    split: str
    trajectory_kind: str
    source_state_sha256: str
    matched_robot_state_sha256: str
    branch_start_state_sha256: str
    arrays_path: str
    instruction: str
    n_steps: int
    outcome: str
    crashed: bool
    succeeded: bool
    safe_abort: bool
    oracle_verified: bool
    scene_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Direct construction defaults to the historical contract.  New writers
    # must opt in to v2 explicitly, which prevents an incomplete record from
    # being silently promoted merely because the process imports newer code.
    schema_version: int = LEGACY_TRAJECTORY_SCHEMA_VERSION
    task_suite: str | None = None
    task_id: int | None = None
    trigger_horizon_actions: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in (LEGACY_TRAJECTORY_SCHEMA_VERSION, SCHEMA_VERSION):
            raise ValueError(f"unsupported trajectory schema {self.schema_version!r}")
        if self.split not in SPLITS:
            raise ValueError(f"bad split {self.split!r}")
        if self.trajectory_kind not in TRAJECTORY_KINDS:
            raise ValueError(f"bad trajectory kind {self.trajectory_kind!r}")
        if self.n_steps < 1:
            raise ValueError("a trajectory must have at least one step")
        if sum((bool(self.crashed), bool(self.succeeded), bool(self.safe_abort))) > 1:
            raise ValueError("crashed, succeeded, and safe_abort are mutually exclusive")
        if self.trajectory_kind == "nominal_catastrophe" and not self.crashed:
            raise ValueError("nominal_catastrophe must contain a measured catastrophe")
        if self.trajectory_kind == "oracle_recovery":
            if self.crashed or not self.succeeded or not self.oracle_verified:
                raise ValueError("oracle_recovery must be verified safe and task-completing")
        if self.trajectory_kind == "off_path_control" and self.crashed:
            raise ValueError("off_path_control must be catastrophe-free")
        if self.schema_version == SCHEMA_VERSION:
            expected_outcome = {
                "nominal_catastrophe": "crash",
                "oracle_recovery": "recovery_success",
                "off_path_control": "task_success" if self.succeeded else "timeout",
                "blocked_safe_abort": "safe_abort",
            }[self.trajectory_kind]
            if self.outcome != expected_outcome:
                raise ValueError(
                    f"schema-v2 {self.trajectory_kind} outcome must be "
                    f"{expected_outcome!r}, got {self.outcome!r}"
                )
            if self.trajectory_kind == "off_path_control" and self.safe_abort:
                raise ValueError("schema-v2 off_path_control timeout cannot be safe_abort")
        if self.trajectory_kind == "blocked_safe_abort":
            if self.crashed or self.succeeded or not self.safe_abort or not self.oracle_verified:
                raise ValueError("blocked_safe_abort must be a verified stable abort")
            evidence = self.metadata.get("blocked_evidence")
            if not isinstance(evidence, Mapping) or not evidence.get("controller_class"):
                raise ValueError("blocked_safe_abort requires scoped blocked_evidence")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairedTrajectoryRecord":
        payload = dict(value)
        # Historical JSONL rows predate an explicit per-record version.  They
        # remain readable, but central v2 admission rejects them until migrated
        # and revalidated.
        payload.setdefault("schema_version", LEGACY_TRAJECTORY_SCHEMA_VERSION)
        return cls(**payload)


def write_placement_manifest(
    path: str | Path,
    placements: Iterable[GlassPlacement],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    placements = list(placements)
    metadata = dict(metadata or {})
    design = validate_placement_design(
        placements,
        require_all_splits=metadata.get("design_purpose") != "fresh_evaluation",
    )
    payload = {
        "schema_version": PLACEMENT_SCHEMA_VERSION,
        "kind": "glass_precrash_placement_design",
        "risk_horizons": list(RISK_HORIZONS),
        "trajectory_kinds": list(TRAJECTORY_KINDS),
        "primary_trajectory_kinds": list(PRIMARY_TRAJECTORY_KINDS),
        "optional_auxiliary_trajectory_kinds": list(AUXILIARY_TRAJECTORY_KINDS),
        "metadata": metadata,
        "design": design,
        "placements": [placement.to_dict() for placement in placements],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def read_placement_manifest(path: str | Path) -> tuple[list[GlassPlacement], dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") not in (
        LEGACY_PLACEMENT_SCHEMA_VERSION,
        PLACEMENT_SCHEMA_VERSION,
    ):
        raise ValueError(f"unsupported placement schema {payload.get('schema_version')!r}")
    placements = [GlassPlacement.from_dict(row) for row in payload.get("placements", [])]
    validate_placement_design(
        placements,
        require_all_splits=(
            payload.get("metadata", {}).get("design_purpose") != "fresh_evaluation"
        ),
    )
    return placements, payload


def write_trajectory_manifest(path: str | Path, records: Iterable[PairedTrajectoryRecord]) -> None:
    records = list(records)
    if any(record.schema_version != SCHEMA_VERSION for record in records):
        raise ValueError("legacy trajectory schema v1 is read-only")
    validate_paired_records(records)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records))


def read_trajectory_manifest(path: str | Path) -> list[PairedTrajectoryRecord]:
    records = [
        PairedTrajectoryRecord.from_dict(json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    validate_paired_records(records)
    return records


def validate_auxiliary_records(
    records: Iterable[PairedTrajectoryRecord],
) -> dict[str, Any]:
    """Validate a standalone schema-v2 blocked-appendix cohort."""

    records = list(records)
    seen_pairs: set[str] = set()
    state_splits: dict[str, set[str]] = {}
    for record in records:
        if record.schema_version != SCHEMA_VERSION:
            raise ValueError("auxiliary manifests require schema v2")
        if record.trajectory_kind != "blocked_safe_abort":
            raise ValueError("auxiliary manifests may contain only blocked_safe_abort rows")
        if record.pair_id in seen_pairs:
            raise ValueError(f"duplicate auxiliary pair {record.pair_id}")
        seen_pairs.add(record.pair_id)
        state_splits.setdefault(record.source_state_sha256, set()).add(record.split)
        if not record.metadata.get("attempt_key"):
            raise ValueError(f"{record.pair_id} auxiliary record requires an attempt key")
        controller_hash = record.metadata.get("controller_state_sha256")
        hashes = record.metadata.get("branch_start_hashes")
        if not isinstance(hashes, Mapping):
            raise ValueError(f"{record.pair_id} auxiliary record requires branch-start hashes")
        if (
            hashes.get("simulator_state_sha256") != record.branch_start_state_sha256
            or not hashes.get("observation_sha256")
            or hashes.get("controller_state_sha256") != controller_hash
        ):
            raise ValueError(f"{record.pair_id} auxiliary branch-start hashes are inconsistent")
    leaked = {key: sorted(value) for key, value in state_splits.items() if len(value) > 1}
    if leaked:
        raise ValueError(f"source initial state leaks across auxiliary splits: {leaked}")
    return {
        "records": len(records),
        "pairs": len(seen_pairs),
        "records_by_split": {
            split: sum(record.split == split for record in records) for split in SPLITS
        },
    }


def write_auxiliary_trajectory_manifest(
    path: str | Path,
    records: Iterable[PairedTrajectoryRecord],
) -> None:
    records = list(records)
    validate_auxiliary_records(records)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(
        json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records
    ))


def read_auxiliary_trajectory_manifest(path: str | Path) -> list[PairedTrajectoryRecord]:
    records = [
        PairedTrajectoryRecord.from_dict(json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    validate_auxiliary_records(records)
    return records


def validate_placement_design(
    placements: Iterable[GlassPlacement], *, require_all_splits: bool = True
) -> dict[str, Any]:
    placements = list(placements)
    if not placements:
        raise ValueError("placement design is empty")
    ids: set[str] = set()
    state_splits: dict[str, set[str]] = {}
    cluster_splits: dict[str, set[str]] = {}
    family_fingerprint_splits: dict[str, set[str]] = {}
    physical_scene_ids: dict[str, str] = {}
    counts = {split: 0 for split in SPLITS}
    unique_states = {split: set() for split in SPLITS}
    clusters = {split: set() for split in SPLITS}
    for placement in placements:
        if placement.placement_id in ids:
            raise ValueError(f"duplicate placement_id {placement.placement_id}")
        ids.add(placement.placement_id)
        counts[placement.split] += 1
        unique_states[placement.split].add(placement.source_state_sha256)
        clusters[placement.split].add(placement.cluster_id)
        state_splits.setdefault(placement.source_state_sha256, set()).add(placement.split)
        cluster_splits.setdefault(placement.cluster_id, set()).add(placement.split)
        family_fingerprint = placement.metadata.get("geometry_family_fingerprint")
        if family_fingerprint:
            family_fingerprint_splits.setdefault(
                str(family_fingerprint), set()
            ).add(placement.split)
        physical_scene = placement.metadata.get("physical_scene_sha256")
        if physical_scene:
            prior = physical_scene_ids.get(str(physical_scene))
            if prior is not None:
                raise ValueError(
                    f"duplicate physical scene {physical_scene}: {prior}, "
                    f"{placement.placement_id}"
                )
            physical_scene_ids[str(physical_scene)] = placement.placement_id
    leaked_states = {key: sorted(value) for key, value in state_splits.items() if len(value) > 1}
    if leaked_states:
        raise ValueError(f"source initial state leaks across splits: {leaked_states}")
    leaked_clusters = {key: sorted(value) for key, value in cluster_splits.items() if len(value) > 1}
    if leaked_clusters:
        raise ValueError(f"scene cluster leaks across splits: {leaked_clusters}")
    leaked_family_fingerprints = {
        key: sorted(value)
        for key, value in family_fingerprint_splits.items()
        if len(value) > 1
    }
    if leaked_family_fingerprints:
        raise ValueError(
            "physical geometry family leaks across splits: "
            f"{leaked_family_fingerprints}"
        )
    missing = [split for split, count in counts.items() if count == 0]
    if missing and require_all_splits:
        raise ValueError(f"placement design has empty splits: {missing}")
    return {
        "placements_by_split": counts,
        "unique_source_states_by_split": {
            split: len(values) for split, values in unique_states.items()
        },
        "clusters_by_split": {split: sorted(values) for split, values in clusters.items()},
        "geometry_family_fingerprints_by_split": {
            split: sorted(
                fingerprint for fingerprint, splits in family_fingerprint_splits.items()
                if split in splits
            )
            for split in SPLITS
        },
        "physical_scene_count": len(physical_scene_ids),
        "split_unit": "source LIBERO initial-state sha256; never shared across splits",
        "pairing_unit": (
            "placement_id; three primary trajectory branches share one matched "
            "robot/task state; blocked_safe_abort is optional auxiliary data"
        ),
    }


def _legacy_validate_pair(pair_id: str, group: list[PairedTrajectoryRecord]) -> None:
    """Validate historical v1 structure without granting v2 admission."""

    kinds = [record.trajectory_kind for record in group]
    if sorted(kinds) != sorted(TRAJECTORY_KINDS):
        raise ValueError(f"{pair_id} legacy v1 pair must have exactly four branches; got {kinds}")
    if len({record.placement_id for record in group}) != 1:
        raise ValueError(f"{pair_id} spans multiple placements")
    if len({record.matched_robot_state_sha256 for record in group}) != 1:
        raise ValueError(f"{pair_id} branches do not share the same robot/task state")
    controller_hashes = {record.metadata.get("controller_state_sha256") for record in group}
    if controller_hashes != {None} and (None in controller_hashes or len(controller_hashes) != 1):
        raise ValueError(f"{pair_id} branches do not share the same controller state")
    by_kind = {record.trajectory_kind: record for record in group}
    if (
        by_kind["nominal_catastrophe"].branch_start_state_sha256
        != by_kind["oracle_recovery"].branch_start_state_sha256
    ):
        raise ValueError(f"{pair_id} nominal and oracle branches are not exact-state matched")


def validate_primary_pair(records: Iterable[PairedTrajectoryRecord]) -> dict[str, Any]:
    """Validate one schema-v2 accepted pair using the canonical primary contract.

    Careful-prompt measurements and the optional blocked branch are deliberately
    absent from the admission decision.  Callers may attach either after the
    three primary branches have satisfied this validator.
    """

    group = list(records)
    if not group:
        raise ValueError("primary pair is empty")
    if any(record.schema_version != SCHEMA_VERSION for record in group):
        raise ValueError("legacy schema v1 cannot enter a schema-v2 primary cohort")
    pair_ids = {record.pair_id for record in group}
    if len(pair_ids) != 1:
        raise ValueError("records do not agree on pair_id")
    pair_id = next(iter(pair_ids))
    if not pair_id:
        raise ValueError("pair_id is required")
    kinds = [record.trajectory_kind for record in group]
    missing = set(PRIMARY_TRAJECTORY_KINDS) - set(kinds)
    duplicates = {kind for kind in kinds if kinds.count(kind) > 1}
    unexpected = set(kinds) - set(TRAJECTORY_KINDS)
    if missing or duplicates or unexpected:
        raise ValueError(
            f"{pair_id} needs exactly three primary branches and at most one optional "
            f"blocked branch; missing={sorted(missing)}, duplicates={sorted(duplicates)}, "
            f"unexpected={sorted(unexpected)}"
        )

    primary = {
        record.trajectory_kind: record
        for record in group
        if record.trajectory_kind in PRIMARY_TRAJECTORY_KINDS
    }
    agreement_fields = (
        "placement_id",
        "split",
        "task_suite",
        "task_id",
        "instruction",
        "source_state_sha256",
        "matched_robot_state_sha256",
        "trigger_horizon_actions",
    )
    for field_name in agreement_fields:
        values = {getattr(record, field_name) for record in primary.values()}
        if len(values) != 1 or next(iter(values)) in (None, ""):
            raise ValueError(f"{pair_id} primary branches do not agree on {field_name}")

    trigger_horizon = primary["nominal_catastrophe"].trigger_horizon_actions
    if not isinstance(trigger_horizon, int) or trigger_horizon < 1:
        raise ValueError(f"{pair_id} trigger_horizon_actions must be a positive integer")
    controller_hashes = {
        record.metadata.get("controller_state_sha256") for record in primary.values()
    }
    if None in controller_hashes or "" in controller_hashes or len(controller_hashes) != 1:
        raise ValueError(f"{pair_id} primary branches require one shared controller hash")
    attempt_keys = {record.metadata.get("attempt_key") for record in primary.values()}
    if None in attempt_keys or "" in attempt_keys or len(attempt_keys) != 1:
        raise ValueError(f"{pair_id} primary branches require one shared attempt key")
    if any(not record.scene_sha256 for record in primary.values()):
        raise ValueError(f"{pair_id} primary branches require scene hashes")

    branch_hashes: dict[str, Mapping[str, Any]] = {}
    for kind, record in primary.items():
        hashes = record.metadata.get("branch_start_hashes")
        if not isinstance(hashes, Mapping):
            raise ValueError(f"{pair_id} {kind} requires exact branch-start hashes")
        required_hashes = {
            "simulator_state_sha256",
            "controller_state_sha256",
            "observation_sha256",
        }
        if any(not hashes.get(key) for key in required_hashes):
            raise ValueError(f"{pair_id} {kind} has incomplete branch-start hashes")
        if hashes.get("simulator_state_sha256") != record.branch_start_state_sha256:
            raise ValueError(f"{pair_id} {kind} simulator hash disagrees with its start state")
        if hashes.get("controller_state_sha256") != record.metadata.get(
            "controller_state_sha256"
        ):
            raise ValueError(f"{pair_id} {kind} controller hash is inconsistent")
        branch_hashes[kind] = hashes

    nominal = primary["nominal_catastrophe"]
    oracle = primary["oracle_recovery"]
    control = primary["off_path_control"]
    if not nominal.branch_start_state_sha256 or not oracle.branch_start_state_sha256:
        raise ValueError(f"{pair_id} nominal and oracle branches require start-state hashes")
    if nominal.branch_start_state_sha256 != oracle.branch_start_state_sha256:
        raise ValueError(f"{pair_id} nominal and oracle branches are not exact-state matched")
    if nominal.scene_sha256 != oracle.scene_sha256:
        raise ValueError(f"{pair_id} nominal and oracle branches do not share the scene hash")
    if dict(branch_hashes["nominal_catastrophe"]) != dict(
        branch_hashes["oracle_recovery"]
    ):
        raise ValueError(
            f"{pair_id} nominal and oracle simulator/controller/observation hashes differ"
        )
    if nominal.n_steps != trigger_horizon:
        raise ValueError(f"{pair_id} nominal suffix is not exactly trigger_horizon_actions long")
    if nominal.metadata.get("time_to_catastrophe_actions") != trigger_horizon:
        raise ValueError(f"{pair_id} nominal time_to_catastrophe_actions is inconsistent")
    source_scan_index = nominal.metadata.get("source_scan_precrash_index")
    source_scan_collision = nominal.metadata.get("source_scan_collision_step")
    if (
        not isinstance(source_scan_index, int)
        or not isinstance(source_scan_collision, int)
        or source_scan_index < 0
        or steps_until_event(source_scan_collision, source_scan_index) != trigger_horizon
    ):
        raise ValueError(f"{pair_id} source-scan timing does not identify the exact-H anchor")
    if (
        nominal.metadata.get("source_scan_anchor_state_sha256")
        != nominal.branch_start_state_sha256
        or nominal.metadata.get("source_scan_anchor_controller_state_sha256")
        != nominal.metadata.get("controller_state_sha256")
    ):
        raise ValueError(f"{pair_id} source-scan anchor identity is inconsistent")
    replay = nominal.metadata.get("action_replay_evidence")
    if not isinstance(replay, Mapping) or not replay.get("verified"):
        raise ValueError(f"{pair_id} lacks verified captured-action replay evidence")
    expected_event_index = trigger_horizon - 1
    if (
        replay.get("n_actions") != trigger_horizon
        or replay.get("expected_catastrophe_action_index") != expected_event_index
        or replay.get("actual_catastrophe_action_index") != expected_event_index
    ):
        raise ValueError(f"{pair_id} action replay does not prove an exact-H catastrophe")

    label_hashes = {
        nominal.metadata.get("row_zero_label_sha256"),
        oracle.metadata.get("row_zero_label_sha256"),
    }
    if None in label_hashes or "" in label_hashes or len(label_hashes) != 1:
        raise ValueError(f"{pair_id} oracle row-zero labels do not match the nominal anchor")

    if not oracle.oracle_verified or oracle.crashed or not oracle.succeeded:
        raise ValueError(f"{pair_id} oracle branch is not verified safe task completion")
    oracle_evidence = oracle.metadata
    if (
        oracle_evidence.get("time_to_catastrophe_actions") != trigger_horizon
        or oracle_evidence.get("oracle_recoverable_from_this_state") is not True
        or oracle_evidence.get("oracle_verified_mask") is not True
        or oracle_evidence.get("latest_verified_recoverable_state")
        != oracle.branch_start_state_sha256
        or oracle_evidence.get("runtime_trigger_eligible") is not True
    ):
        raise ValueError(f"{pair_id} oracle recoverability/trigger evidence is incomplete")
    oracle_verification = oracle_evidence.get("oracle_verification")
    oracle_config = oracle_evidence.get("oracle_config")
    if (
        not isinstance(oracle_verification, Mapping)
        or not isinstance(oracle_config, Mapping)
        or oracle_verification.get("search_success") is not True
        or oracle_verification.get("independent_recapture") is not True
        or oracle_verification.get("recapture_success") is not True
        or oracle_verification.get("search_successful_config_sha256")
        != canonical_sha256(oracle_config)
    ):
        raise ValueError(
            f"{pair_id} oracle requires successful search and independent recapture"
        )

    if control.crashed or not control.succeeded or control.safe_abort:
        raise ValueError(f"{pair_id} off_path_control must safely complete the original task")
    if control.metadata.get("termination") != "task_success":
        raise ValueError(f"{pair_id} off_path_control lacks task-success termination evidence")
    return {
        "pair_id": pair_id,
        "schema_version": SCHEMA_VERSION,
        "primary_branches": list(PRIMARY_TRAJECTORY_KINDS),
        "blocked_auxiliary_present": "blocked_safe_abort" in kinds,
        "trigger_horizon_actions": trigger_horizon,
    }


def validate_paired_records(records: Iterable[PairedTrajectoryRecord]) -> dict[str, Any]:
    records = list(records)
    if not records:
        raise ValueError("trajectory manifest is empty")
    by_pair: dict[str, list[PairedTrajectoryRecord]] = {}
    state_splits: dict[str, set[str]] = {}
    for record in records:
        by_pair.setdefault(record.pair_id, []).append(record)
        state_splits.setdefault(record.source_state_sha256, set()).add(record.split)
    leaked = {key: sorted(value) for key, value in state_splits.items() if len(value) > 1}
    if leaked:
        raise ValueError(f"source initial state leaks across trajectory splits: {leaked}")
    versions: dict[int, int] = {}
    for pair_id, group in by_pair.items():
        pair_versions = {record.schema_version for record in group}
        if len(pair_versions) != 1:
            raise ValueError(f"{pair_id} mixes trajectory schema versions")
        version = next(iter(pair_versions))
        versions[version] = versions.get(version, 0) + 1
        if version == LEGACY_TRAJECTORY_SCHEMA_VERSION:
            _legacy_validate_pair(pair_id, group)
        else:
            validate_primary_pair(group)
    return {
        "pairs": len(by_pair),
        "records": len(records),
        "pairs_by_schema_version": versions,
        "records_by_split": {
            split: sum(record.split == split for record in records) for split in SPLITS
        },
    }


REQUIRED_ARRAYS = {
    "hidden": 2,
    "robot_state": 2,
    "nominal_action": 2,
    "target_action": 2,
    "executed_action": 2,
    "risk_targets": 2,
    "risk_mask": 2,
    "hazard_type": 1,
    "severity_force": 1,
    "severity_mask": 1,
    "abort_target": 1,
    "recovery_mask": 1,
    "invariance_mask": 1,
    "sensitivity_mask": 1,
}

V2_REQUIRED_ARRAYS = {
    "time_to_catastrophe_actions": 1,
    "time_to_catastrophe_mask": 1,
    "oracle_recoverable_from_this_state": 1,
    "oracle_verified_mask": 1,
    "latest_verified_recoverable_state": 1,
    "runtime_trigger_eligible": 1,
}


def validate_episode_arrays(
    arrays: Mapping[str, np.ndarray],
    n_steps: int | None = None,
    *,
    schema_version: int = SCHEMA_VERSION,
) -> int:
    required = dict(REQUIRED_ARRAYS)
    if schema_version == SCHEMA_VERSION:
        required.update(V2_REQUIRED_ARRAYS)
    elif schema_version != LEGACY_TRAJECTORY_SCHEMA_VERSION:
        raise ValueError(f"unsupported episode-array schema {schema_version!r}")
    missing = set(required) - set(arrays)
    if missing:
        raise ValueError(f"episode arrays missing {sorted(missing)}")
    inferred = int(np.asarray(arrays["hidden"]).shape[0])
    if n_steps is not None and inferred != n_steps:
        raise ValueError(f"hidden rows {inferred} != record n_steps {n_steps}")
    for key, ndim in required.items():
        array = np.asarray(arrays[key])
        if array.ndim != ndim or array.shape[0] != inferred:
            raise ValueError(f"{key} must have ndim={ndim}, first dim={inferred}; got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains non-finite values")
    for key in ("nominal_action", "target_action", "executed_action"):
        if np.asarray(arrays[key]).shape[1] != 7:
            raise ValueError(f"{key} must be [N, 7]")
    if np.asarray(arrays["robot_state"]).shape[1] != 8:
        raise ValueError("robot_state must be [N, 8]")
    if np.asarray(arrays["risk_targets"]).shape[1] != len(RISK_HORIZONS):
        raise ValueError(f"risk_targets must follow horizons {RISK_HORIZONS}")
    if np.asarray(arrays["risk_mask"]).shape != np.asarray(arrays["risk_targets"]).shape:
        raise ValueError("risk_mask shape must match risk_targets")
    for key in ("risk_targets", "risk_mask", "abort_target", "recovery_mask", "invariance_mask", "sensitivity_mask"):
        values = np.asarray(arrays[key])
        if np.any((values != 0) & (values != 1)):
            raise ValueError(f"{key} values must be binary")
    hazards = np.asarray(arrays["hazard_type"], dtype=int)
    if np.any((hazards < 0) | (hazards >= len(HAZARD_TYPES))):
        raise ValueError("hazard_type index is out of range")
    if schema_version == SCHEMA_VERSION:
        for key in (
            "time_to_catastrophe_mask",
            "oracle_recoverable_from_this_state",
            "oracle_verified_mask",
            "latest_verified_recoverable_state",
            "runtime_trigger_eligible",
        ):
            values = np.asarray(arrays[key])
            if np.any((values != 0) & (values != 1)):
                raise ValueError(f"{key} values must be binary")
        times = np.asarray(arrays["time_to_catastrophe_actions"])
        time_mask = np.asarray(arrays["time_to_catastrophe_mask"]).astype(bool)
        if np.any(times != np.floor(times)):
            raise ValueError("time_to_catastrophe_actions values must be integers")
        if np.any(times[time_mask] < 1):
            raise ValueError("verified time_to_catastrophe_actions values must be positive")
        if np.any(times[~time_mask] != -1):
            raise ValueError("censored time_to_catastrophe_actions values must use -1")

        risk_targets = np.asarray(arrays["risk_targets"])
        risk_mask = np.asarray(arrays["risk_mask"])
        if np.any(risk_targets[risk_mask == 0] != 0):
            raise ValueError("masked risk_targets must be zero")
        if np.any(risk_mask[time_mask] != 1):
            raise ValueError("verified time-to-catastrophe rows require complete risk labels")
        if np.any(time_mask):
            expected_risk = np.asarray([
                [float(1 <= int(remaining) <= horizon) for horizon in RISK_HORIZONS]
                for remaining in times[time_mask]
            ])
            if not np.array_equal(risk_targets[time_mask], expected_risk):
                raise ValueError("risk_targets disagree with time_to_catastrophe_actions")

        oracle_recoverable = np.asarray(
            arrays["oracle_recoverable_from_this_state"]
        ).astype(bool)
        oracle_verified = np.asarray(arrays["oracle_verified_mask"]).astype(bool)
        latest_verified = np.asarray(
            arrays["latest_verified_recoverable_state"]
        ).astype(bool)
        trigger_eligible = np.asarray(arrays["runtime_trigger_eligible"]).astype(bool)
        if np.any(oracle_recoverable & ~oracle_verified):
            raise ValueError("oracle recoverability must be verified")
        if np.any(latest_verified & ~oracle_recoverable):
            raise ValueError("latest recoverable state must itself be recoverable")
        if np.any(trigger_eligible & ~(oracle_recoverable & oracle_verified)):
            raise ValueError("runtime trigger eligibility requires verified recoverability")
    return inferred
