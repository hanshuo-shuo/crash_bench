from __future__ import annotations

import json

import pytest

from crashbench.data.source_registry import (
    ExposureRegistry,
    ExposureViolation,
    SourceIdentity,
    SplitRole,
    append_exposure_attempt,
)


EXPOSED = "a" * 64
MANIFEST = "b" * 64


def registry() -> ExposureRegistry:
    return ExposureRegistry(
        {
            "kind": "crashbench_expansion_exposure_registry",
            "sources": [
                {
                    "source_state_sha256": EXPOSED,
                    "physical_source_id": "physical-1",
                    "reset_seed": 17,
                    "scene_fingerprint": "scene-1",
                }
            ],
            "pool_blacklists": [
                {
                    "manifest_sha256": MANIFEST,
                    "candidate_index_start": 0,
                    "candidate_index_end": 29,
                }
            ],
        }
    )


@pytest.mark.parametrize("role", sorted(role.value for role in {
    SplitRole.CONFIRMATORY_ID_TEST,
    SplitRole.OOD_TASK_TEST,
    SplitRole.OOD_SEVERITY_TEST,
    SplitRole.CROSS_POLICY_TEST,
    SplitRole.FRESH_SEQUENTIAL_TEST,
}))
def test_every_claim_bearing_test_role_rejects_exposed_hash(role):
    with pytest.raises(ExposureViolation):
        registry().assert_role_allowed(SourceIdentity(source_state_sha256=EXPOSED), role)


def test_non_test_roles_may_use_exposed_fixture():
    registry().assert_role_allowed(
        SourceIdentity(source_state_sha256=EXPOSED), SplitRole.ENGINEERING_SCREEN
    )


def test_seed_physical_scene_and_missing_pool_each_block_test():
    identities = [
        SourceIdentity(reset_seed=17),
        SourceIdentity(physical_source_id="physical-1"),
        SourceIdentity(scene_fingerprint="scene-1"),
        SourceIdentity(source_manifest_sha256=MANIFEST, candidate_index=12),
    ]
    for identity in identities:
        with pytest.raises(ExposureViolation):
            registry().assert_role_allowed(identity, SplitRole.CONFIRMATORY_ID_TEST)


def test_unregistered_role_cannot_bypass_firewall():
    with pytest.raises(ValueError, match="unregistered split role"):
        registry().assert_role_allowed(SourceIdentity(), "fresh_test")


def test_registry_load(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry().payload))
    loaded = ExposureRegistry.load(path)
    assert EXPOSED in loaded.source_hashes


def test_identifier_union_blocks_seed_and_manifest_without_source_hash():
    payload = {
        "kind": "crashbench_expansion_exposure_registry",
        "sources": [],
        "identifiers": [
            {"identifier_type": "reset_seed", "value": "42"},
            {"identifier_type": "source_manifest_sha256", "value": MANIFEST},
        ],
        "pool_blacklists": [],
    }
    frozen = ExposureRegistry(payload)
    for identity in (
        SourceIdentity(reset_seed=42),
        SourceIdentity(source_manifest_sha256=MANIFEST),
    ):
        with pytest.raises(ExposureViolation):
            frozen.assert_role_allowed(identity, SplitRole.CONFIRMATORY_ID_TEST)


def test_append_exposure_attempt_is_idempotent_and_conflicts_fail(tmp_path):
    ledger = tmp_path / "attempts.jsonl"
    row = {
        "attempt_id": "screen-1",
        "artifact_role": "ENGINEERING_SCREEN",
        "protocol_sha256": "c" * 64,
        "reset_seed": 17,
    }
    assert append_exposure_attempt(ledger, row)
    assert not append_exposure_attempt(ledger, row)
    with pytest.raises(ValueError, match="conflicting"):
        append_exposure_attempt(ledger, {**row, "reset_seed": 18})
    stored = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(stored) == 1
    assert stored[0]["test_eligible"] is False
