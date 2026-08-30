"""Deterministic physical-source splitting and one-time test authorization."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .source_registry import ExposureRegistry, SourceIdentity, SplitRole, canonical_json_sha256


@dataclass(frozen=True)
class SourceRecord:
    physical_source_id: str
    mechanism_source_id: str
    policy_source_id: str
    mechanism_id: str
    task_id: str
    policy_id: str
    source_state_sha256: str | None = None
    reset_seed: int | str | None = None
    source_manifest_sha256: str | None = None
    candidate_index: int | None = None
    scene_fingerprint: str | None = None

    def identity(self) -> SourceIdentity:
        return SourceIdentity(
            source_state_sha256=self.source_state_sha256,
            physical_source_id=self.physical_source_id,
            reset_seed=self.reset_seed,
            source_manifest_sha256=self.source_manifest_sha256,
            candidate_index=self.candidate_index,
            scene_fingerprint=self.scene_fingerprint,
        )


def _allocation_key(protocol_sha256: str, cell: tuple[str, str], physical_id: str) -> str:
    return hashlib.sha256(
        "\0".join((protocol_sha256, cell[0], cell[1], physical_id)).encode()
    ).hexdigest()


def freeze_split_manifest(
    records: Iterable[SourceRecord],
    *,
    counts_by_cell: Mapping[str, int],
    protocol_sha256: str,
    exposure_registry: ExposureRegistry,
) -> dict[str, Any]:
    """Assign exact cell quotas while keeping every policy on its physical parent split."""

    parsed_counts = [(SplitRole(role), int(count)) for role, count in counts_by_cell.items()]
    if any(count < 0 for _, count in parsed_counts) or not parsed_counts:
        raise ValueError("split counts must be a non-empty set of non-negative integers")
    rows = list(records)
    if not rows:
        raise ValueError("cannot freeze an empty source registry")

    by_cell: dict[tuple[str, str], dict[str, list[SourceRecord]]] = {}
    physical_cells: dict[str, set[tuple[str, str]]] = {}
    seen_policy_ids: set[str] = set()
    for row in rows:
        if row.policy_source_id in seen_policy_ids:
            raise ValueError(f"duplicate policy_source_id: {row.policy_source_id}")
        seen_policy_ids.add(row.policy_source_id)
        cell = (row.mechanism_id, row.task_id)
        by_cell.setdefault(cell, {}).setdefault(row.physical_source_id, []).append(row)
        physical_cells.setdefault(row.physical_source_id, set()).add(cell)
    cross_cell = {source: cells for source, cells in physical_cells.items() if len(cells) > 1}
    if cross_cell:
        raise ValueError(
            "v1 split freezer requires one mechanism/task cell per physical source; "
            f"cross-cell parents: {sorted(cross_cell)}"
        )

    assignments: list[dict[str, Any]] = []
    physical_roles: dict[str, SplitRole] = {}
    expected_per_cell = sum(count for _, count in parsed_counts)
    for cell in sorted(by_cell):
        physical_groups = by_cell[cell]
        if len(physical_groups) != expected_per_cell:
            raise ValueError(
                f"cell {cell} has {len(physical_groups)} physical sources, expected {expected_per_cell}"
            )
        ordered = sorted(
            physical_groups,
            key=lambda physical_id: _allocation_key(protocol_sha256, cell, physical_id),
        )
        cursor = 0
        for role, count in parsed_counts:
            for physical_id in ordered[cursor:cursor + count]:
                physical_roles[physical_id] = role
                for record in sorted(
                    physical_groups[physical_id], key=lambda item: item.policy_source_id
                ):
                    exposure_registry.assert_role_allowed(record.identity(), role)
                    assignments.append({**asdict(record), "role": role.value})
            cursor += count

    fingerprint_roles: dict[str, set[str]] = {}
    for row in assignments:
        fingerprint = row.get("scene_fingerprint")
        if fingerprint:
            fingerprint_roles.setdefault(fingerprint, set()).add(row["role"])
    leaks = {
        fingerprint: sorted(roles)
        for fingerprint, roles in fingerprint_roles.items()
        if len(roles) > 1
    }
    if leaks:
        raise ValueError(f"near-duplicate scene fingerprints cross splits: {leaks}")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_split_manifest",
        "protocol_sha256": protocol_sha256,
        "allocation": "sha256(protocol,mechanism,task,physical_source_id)",
        "counts_by_cell": {role.value: count for role, count in parsed_counts},
        "assignments": sorted(assignments, key=lambda row: row["policy_source_id"]),
        "physical_source_count": len(physical_roles),
        "policy_source_count": len(assignments),
        "test_outcomes_read": 0,
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    return payload


@dataclass(frozen=True)
class TestAuthorization:
    run_id: str
    protocol_sha256: str
    test_source_manifest_sha256: str
    model_sha256: str
    comparator_sha256: str
    calibration_sha256: str
    analysis_script_sha256: str

    def payload(self) -> dict[str, Any]:
        base = {
            "schema_version": 1,
            "kind": "crashbench_one_time_test_authorization",
            **asdict(self),
        }
        base["authorization_token"] = canonical_json_sha256(base)
        return base


def create_test_authorization(lock_root: str | Path, authorization: TestAuthorization) -> dict[str, Any]:
    root = Path(lock_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "test_authorization.json"
    payload = authorization.payload()
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    (root / "test_open.lock").touch(exist_ok=False)
    return payload


def validate_test_retry(lock_root: str | Path, authorization: TestAuthorization) -> str:
    root = Path(lock_root)
    if (root / "test_complete.seal").exists():
        raise RuntimeError("test is already complete; scientific reopen is forbidden")
    if not (root / "test_open.lock").is_file():
        raise RuntimeError("test authorization is not open")
    actual = json.loads((root / "test_authorization.json").read_text())
    expected = authorization.payload()
    if actual != expected:
        raise RuntimeError("operational retry identity differs from the authorized test")
    return str(actual["authorization_token"])


def seal_test_complete(lock_root: str | Path, *, completeness_audit_sha256: str) -> Path:
    root = Path(lock_root)
    if not (root / "test_open.lock").is_file():
        raise RuntimeError("cannot complete a test that is not open")
    path = root / "test_complete.seal"
    payload = {
        "kind": "crashbench_test_complete_seal",
        "completeness_audit_sha256": completeness_audit_sha256,
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, (json.dumps(payload, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path
