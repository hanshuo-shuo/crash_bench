"""Auditable schema and validation for glass pre-crash paired data.

One placement is a *matched group*, not one rollout.  Every accepted group has
four branches starting from the same robot/task state:

``nominal_catastrophe``
    Base OpenVLA continues into the on-path glass and actually triggers a crash
    predicate.
``oracle_recovery``
    The same on-path state is restored and a verified safe, task-completing
    oracle trajectory is executed.
``off_path_control``
    The robot/task state is unchanged, but the same glass is moved outside the
    swept corridor.  The target action is the unchanged nominal action.
``blocked_safe_abort``
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


SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "heldout")
TRAJECTORY_KINDS = (
    "nominal_catastrophe",
    "oracle_recovery",
    "off_path_control",
    "blocked_safe_abort",
)
RISK_HORIZONS = (1, 3, 5, 10, 20)
HAZARD_TYPES = ("none", "glass", "wall", "other")


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
        if not self.blocked_glasses:
            raise ValueError("blocked_glasses must contain at least one glass")
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

    def __post_init__(self) -> None:
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
        return cls(**dict(value))


def write_placement_manifest(
    path: str | Path,
    placements: Iterable[GlassPlacement],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    placements = list(placements)
    design = validate_placement_design(placements)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "glass_precrash_placement_design",
        "risk_horizons": list(RISK_HORIZONS),
        "trajectory_kinds": list(TRAJECTORY_KINDS),
        "metadata": dict(metadata or {}),
        "design": design,
        "placements": [placement.to_dict() for placement in placements],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def read_placement_manifest(path: str | Path) -> tuple[list[GlassPlacement], dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported placement schema {payload.get('schema_version')!r}")
    placements = [GlassPlacement.from_dict(row) for row in payload.get("placements", [])]
    validate_placement_design(placements)
    return placements, payload


def write_trajectory_manifest(path: str | Path, records: Iterable[PairedTrajectoryRecord]) -> None:
    records = list(records)
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


def validate_placement_design(placements: Iterable[GlassPlacement]) -> dict[str, Any]:
    placements = list(placements)
    if not placements:
        raise ValueError("placement design is empty")
    ids: set[str] = set()
    state_splits: dict[str, set[str]] = {}
    cluster_splits: dict[str, set[str]] = {}
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
    leaked_states = {key: sorted(value) for key, value in state_splits.items() if len(value) > 1}
    if leaked_states:
        raise ValueError(f"source initial state leaks across splits: {leaked_states}")
    leaked_clusters = {key: sorted(value) for key, value in cluster_splits.items() if len(value) > 1}
    if leaked_clusters:
        raise ValueError(f"scene cluster leaks across splits: {leaked_clusters}")
    missing = [split for split, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"placement design has empty splits: {missing}")
    return {
        "placements_by_split": counts,
        "unique_source_states_by_split": {
            split: len(values) for split, values in unique_states.items()
        },
        "clusters_by_split": {split: sorted(values) for split, values in clusters.items()},
        "split_unit": "source LIBERO initial-state sha256; never shared across splits",
        "pairing_unit": "placement_id; four trajectory branches share one matched robot/task state",
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
    for pair_id, group in by_pair.items():
        kinds = [record.trajectory_kind for record in group]
        if sorted(kinds) != sorted(TRAJECTORY_KINDS):
            raise ValueError(f"{pair_id} must have exactly four branches; got {kinds}")
        if len({record.placement_id for record in group}) != 1:
            raise ValueError(f"{pair_id} spans multiple placements")
        if len({record.matched_robot_state_sha256 for record in group}) != 1:
            raise ValueError(f"{pair_id} branches do not share the same robot/task state")
        on_path = {
            record.trajectory_kind: record for record in group
            if record.trajectory_kind in {"nominal_catastrophe", "oracle_recovery"}
        }
        if (
            on_path["nominal_catastrophe"].branch_start_state_sha256
            != on_path["oracle_recovery"].branch_start_state_sha256
        ):
            raise ValueError(f"{pair_id} nominal and oracle branches are not exact-state matched")
    return {
        "pairs": len(by_pair),
        "records": len(records),
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


def validate_episode_arrays(arrays: Mapping[str, np.ndarray], n_steps: int | None = None) -> int:
    missing = set(REQUIRED_ARRAYS) - set(arrays)
    if missing:
        raise ValueError(f"episode arrays missing {sorted(missing)}")
    inferred = int(np.asarray(arrays["hidden"]).shape[0])
    if n_steps is not None and inferred != n_steps:
        raise ValueError(f"hidden rows {inferred} != record n_steps {n_steps}")
    for key, ndim in REQUIRED_ARRAYS.items():
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
        if np.any((values < 0) | (values > 1)):
            raise ValueError(f"{key} values must be in [0, 1]")
    hazards = np.asarray(arrays["hazard_type"], dtype=int)
    if np.any((hazards < 0) | (hazards >= len(HAZARD_TYPES))):
        raise ValueError("hazard_type index is out of range")
    return inferred
