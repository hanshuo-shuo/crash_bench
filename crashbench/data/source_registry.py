"""Fail-closed source exposure registry and split firewall.

The expansion benchmark treats every historically inspected source, seed, and
upstream candidate pool as exposed.  This module deliberately has no simulator
dependencies so every authoring, collection, and evaluation entry point can
enforce the same rule.
"""

from __future__ import annotations

import hashlib
import json
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
        manifest = _canonical_sha(identity.source_manifest_sha256)
        if manifest:
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
