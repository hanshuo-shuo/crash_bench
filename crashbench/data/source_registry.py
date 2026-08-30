"""Fail-closed source exposure registry and split firewall.

The expansion benchmark treats every historically inspected source, seed, and
upstream candidate pool as exposed.  This module deliberately has no simulator
dependencies so every authoring, collection, and evaluation entry point can
enforce the same rule.
"""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


class SplitRole(str, Enum):
    ENGINEERING_SCREEN = "engineering_screen"
    TRAIN = "train"
    CALIBRATION = "calibration"
    DEVELOPMENT = "development"
    CONFIRMATORY_ID_TEST = "confirmatory_id_test"
    OOD_TASK_TEST = "ood_task_test"
    OOD_SEVERITY_TEST = "ood_severity_test"
    CROSS_POLICY_TEST = "cross_policy_test"
    CROSS_POLICY_CALIBRATION = "cross_policy_calibration"
    FRESH_SEQUENTIAL_TEST = "fresh_sequential_test"
    ADAPTIVE_TRAINING = "adaptive_training"
    LEGACY_PRETRAINING_ABLATION = "legacy_pretraining_ablation"


TEST_ROLES = frozenset(
    {
        SplitRole.CONFIRMATORY_ID_TEST,
        SplitRole.OOD_TASK_TEST,
        SplitRole.OOD_SEVERITY_TEST,
        SplitRole.CROSS_POLICY_TEST,
        SplitRole.FRESH_SEQUENTIAL_TEST,
    }
)


class ExposureViolation(ValueError):
    """Raised when an exposed lineage is assigned to claim-bearing test data."""


@dataclass(frozen=True)
class SourceIdentity:
    source_state_sha256: str | None = None
    physical_source_id: str | None = None
    reset_seed: int | str | None = None
    source_manifest_sha256: str | None = None
    candidate_index: int | None = None
    scene_fingerprint: str | None = None
    upstream_source_key: str | None = None


def _canonical_sha(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.lower()
    if len(lowered) == 64 and all(char in "0123456789abcdef" for char in lowered):
        return lowered
    return None


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class ExposureRegistry:
    """Read and query the immutable-union exposure artifact."""

    def __init__(self, payload: Mapping[str, Any]):
        if payload.get("kind") != "crashbench_expansion_exposure_registry":
            raise ValueError("not a CrashBench expansion exposure registry")
        self.payload = dict(payload)
        self.source_hashes = frozenset(
            source_hash
            for row in payload.get("sources", [])
            if (source_hash := _canonical_sha(row.get("source_state_sha256")))
        )
        self.physical_source_ids = frozenset(
            str(row["physical_source_id"])
            for row in payload.get("sources", [])
            if row.get("physical_source_id") not in (None, "")
        )
        self.reset_seeds = frozenset(
            str(row["reset_seed"])
            for row in payload.get("sources", [])
            if row.get("reset_seed") not in (None, "")
        )
        self.scene_fingerprints = frozenset(
            str(row["scene_fingerprint"])
            for row in payload.get("sources", [])
            if row.get("scene_fingerprint") not in (None, "")
        )
        self.pool_blacklists = tuple(payload.get("pool_blacklists", []))
        identifiers = payload.get("identifiers", [])
        self.physical_source_ids = self.physical_source_ids | frozenset(
            str(row["value"]) for row in identifiers if row.get("identifier_type") == "physical_source_id"
        )
        self.reset_seeds = self.reset_seeds | frozenset(
            str(row["value"])
            for row in identifiers
            if row.get("identifier_type") in {"reset_seed", "generator_seed"}
        )
        self.scene_fingerprints = self.scene_fingerprints | frozenset(
            str(row["value"]) for row in identifiers if row.get("identifier_type") == "scene_fingerprint"
        )
        self.manifest_sha256s = frozenset(
            str(row["value"]).lower()
            for row in identifiers
            if row.get("identifier_type") == "source_manifest_sha256"
        )
        self.upstream_source_keys = frozenset(
            str(row["value"])
            for row in identifiers
            if row.get("identifier_type") == "upstream_source_key"
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExposureRegistry":
        return cls(json.loads(Path(path).read_text()))

    def exposure_reasons(self, identity: SourceIdentity) -> list[str]:
        reasons: list[str] = []
        source_hash = _canonical_sha(identity.source_state_sha256)
        if source_hash and source_hash in self.source_hashes:
            reasons.append(f"source_state_sha256:{source_hash}")
        if identity.physical_source_id and identity.physical_source_id in self.physical_source_ids:
            reasons.append(f"physical_source_id:{identity.physical_source_id}")
        if identity.reset_seed is not None and str(identity.reset_seed) in self.reset_seeds:
            reasons.append(f"reset_seed:{identity.reset_seed}")
        if identity.scene_fingerprint and identity.scene_fingerprint in self.scene_fingerprints:
            reasons.append(f"scene_fingerprint:{identity.scene_fingerprint}")
        if identity.upstream_source_key and identity.upstream_source_key in self.upstream_source_keys:
            reasons.append(f"upstream_source_key:{identity.upstream_source_key}")
        manifest = _canonical_sha(identity.source_manifest_sha256)
        if manifest:
            if manifest in self.manifest_sha256s:
                reasons.append(f"source_manifest_sha256:{manifest}")
            for pool in self.pool_blacklists:
                if _canonical_sha(pool.get("manifest_sha256")) != manifest:
                    continue
                start = pool.get("candidate_index_start")
                end = pool.get("candidate_index_end")
                if identity.candidate_index is None or start is None or end is None:
                    reasons.append(f"blacklisted_pool:{manifest}")
                elif int(start) <= identity.candidate_index <= int(end):
                    reasons.append(f"blacklisted_pool:{manifest}:{identity.candidate_index}")
        return reasons

    def assert_role_allowed(self, identity: SourceIdentity, role: SplitRole | str) -> None:
        try:
            parsed_role = SplitRole(role)
        except ValueError as exc:
            raise ValueError(f"unregistered split role: {role!r}") from exc
        reasons = self.exposure_reasons(identity)
        if parsed_role in TEST_ROLES and reasons:
            raise ExposureViolation(
                f"exposed source cannot enter {parsed_role.value}: {', '.join(reasons)}"
            )

    def assert_candidates_fresh(
        self, candidates: Iterable[SourceIdentity], role: SplitRole | str
    ) -> None:
        for candidate in candidates:
            self.assert_role_allowed(candidate, role)


EXPOSURE_ATTEMPT_KEYS = frozenset(
    {
        "source_state_sha256",
        "physical_source_id",
        "reset_seed",
        "generator_seed",
        "source_manifest_sha256",
        "candidate_index",
        "scene_fingerprint",
        "upstream_source_key",
    }
)


def append_exposure_attempt(path: str | Path, record: Mapping[str, Any]) -> bool:
    """Append one pre-outcome source attempt, idempotently and under a file lock.

    Returns ``True`` for a new row and ``False`` when the exact attempt already
    exists. Reusing an attempt ID with different bytes is a hard error.
    """

    required = {"attempt_id", "artifact_role", "protocol_sha256"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"exposure attempt lacks required fields: {sorted(missing)}")
    if record.get("test_eligible", False) is not False:
        raise ValueError("exposure attempts can never be test eligible")
    if not any(record.get(key) not in (None, "") for key in EXPOSURE_ATTEMPT_KEYS):
        raise ValueError("exposure attempt has no lineage identifier")
    payload = {
        "schema_version": 1,
        "kind": "crashbench_expansion_exposure_attempt",
        **dict(record),
        "test_eligible": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(ledger, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        with os.fdopen(descriptor, "r+", closefd=False) as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            for line in handle:
                if not line.strip():
                    continue
                existing = json.loads(line)
                if existing.get("attempt_id") != payload["attempt_id"]:
                    continue
                if json.dumps(existing, sort_keys=True, separators=(",", ":")) == encoded:
                    return False
                raise ValueError(f"conflicting exposure attempt_id: {payload['attempt_id']}")
            os.write(handle.fileno(), (encoded + "\n").encode())
            os.fsync(handle.fileno())
            return True
    finally:
        os.close(descriptor)
