from __future__ import annotations

import json

import pytest

from crashbench.data.source_registry import (
    ExposureRegistry,
    ExposureViolation,
    SourceIdentity,
    SplitRole,
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
    registry().assert_role_allowed(SourceIdentity(source_state_sha256=EXPOSED), SplitRole.ENGINEERING)


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
