from __future__ import annotations

import json

import pytest

from crashbench.data.source_registry import ExposureRegistry, ExposureViolation
from crashbench.data.splits import (
    SourceRecord,
    TestAuthorization as Authorization,
    create_test_authorization,
    freeze_split_manifest,
    seal_test_complete,
    validate_test_retry,
)


def empty_registry(exposed: str | None = None):
    sources = [] if exposed is None else [{"source_state_sha256": exposed}]
    return ExposureRegistry(
        {"kind": "crashbench_expansion_exposure_registry", "sources": sources, "pool_blacklists": []}
    )


def records(count=4, *, fingerprint_collision=False):
    rows = []
    for index in range(count):
        fingerprint = "same" if fingerprint_collision else f"scene-{index}"
        for policy in ("openvla", "pi0"):
            rows.append(
                SourceRecord(
                    physical_source_id=f"physical-{index}",
                    mechanism_source_id=f"mechanism-source-{index}",
                    policy_source_id=f"policy-source-{index}-{policy}",
                    mechanism_id="fragile_v2",
                    task_id="libero_spatial:0",
                    policy_id=policy,
                    source_state_sha256=f"{index + 1:064x}",
                    scene_fingerprint=fingerprint,
                )
            )
    return rows


def test_deterministic_split_is_policy_inherited_and_exact():
    kwargs = {
        "counts_by_cell": {"train": 2, "confirmatory_id_test": 2},
        "protocol_sha256": "a" * 64,
        "exposure_registry": empty_registry(),
    }
    first = freeze_split_manifest(records(), **kwargs)
    second = freeze_split_manifest(reversed(records()), **kwargs)
    assert first == second
    roles = {}
    for row in first["assignments"]:
        roles.setdefault(row["physical_source_id"], set()).add(row["role"])
    assert all(len(value) == 1 for value in roles.values())
    assert first["physical_source_count"] == 4
    assert first["policy_source_count"] == 8
    assert first["test_outcomes_read"] == 0


def test_exposed_source_is_rejected_from_any_test_allocation():
    candidate_rows = records(1)
    exposed = candidate_rows[0].source_state_sha256
    with pytest.raises(ExposureViolation):
        freeze_split_manifest(
            candidate_rows,
            counts_by_cell={"confirmatory_id_test": 1},
            protocol_sha256="a" * 64,
            exposure_registry=empty_registry(exposed),
        )


def test_near_duplicate_fingerprint_cannot_cross_split():
    with pytest.raises(ValueError, match="near-duplicate"):
        freeze_split_manifest(
            records(fingerprint_collision=True),
            counts_by_cell={"train": 2, "confirmatory_id_test": 2},
            protocol_sha256="a" * 64,
            exposure_registry=empty_registry(),
        )


def authorization(run_id="run-1"):
    return Authorization(
        run_id=run_id,
        protocol_sha256="a" * 64,
        test_source_manifest_sha256="b" * 64,
        model_sha256="c" * 64,
        comparator_sha256="d" * 64,
        calibration_sha256="e" * 64,
        analysis_script_sha256="f" * 64,
    )


def test_one_time_test_lock_allows_identical_retry_then_seals(tmp_path):
    payload = create_test_authorization(tmp_path, authorization())
    assert validate_test_retry(tmp_path, authorization()) == payload["authorization_token"]
    with pytest.raises(RuntimeError, match="differs"):
        validate_test_retry(tmp_path, authorization("different"))
    seal_test_complete(tmp_path, completeness_audit_sha256="1" * 64)
    with pytest.raises(RuntimeError, match="already complete"):
        validate_test_retry(tmp_path, authorization())
    with pytest.raises(FileExistsError):
        seal_test_complete(tmp_path, completeness_audit_sha256="1" * 64)


def test_authorization_cannot_be_created_twice(tmp_path):
    create_test_authorization(tmp_path, authorization())
    with pytest.raises(FileExistsError):
        create_test_authorization(tmp_path, authorization())
