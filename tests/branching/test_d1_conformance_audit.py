from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/audit_d1_conformance.py"
SPEC = importlib.util.spec_from_file_location("audit_d1_conformance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def result(backend, task_id, passed=True):
    criteria = {
        "branch_start_bundle_identical": passed,
        "branch_start_components_identical": passed,
        "next_action_trace_identical": passed,
        "repeat_count_exact": passed,
        "state_trace_identical": passed,
        "terminal_signature_identical": passed,
    }
    return {
        "backend": backend,
        "source": {"task_id": task_id, "role": "EXPOSED_ENGINEERING_ONLY"},
        "execution": {"slurm_job_id": "job", "git_commit": "commit"},
        "gate": {
            "status": "PASS" if passed else "FAIL",
            "classification": "DETERMINISTIC_EXACT" if passed else "UNCLASSIFIED_FAIL_CLOSED",
            "criteria": criteria,
        },
        "bundle": {"capture_seconds": 0.1},
        "repeats": [{"restore_and_trace_seconds": 1.0}] * 3,
        "claim_boundary": "Engineering conformance only; no scientific option outcome was opened.",
    }


def test_primary_complete_transfer_pending_allows_primary_only_progress():
    decision = MODULE.audit_cells([result("openvla", 0), result("openvla", 2)])
    assert decision["status"] == "PRIMARY_GO_TRANSFER_PENDING"
    assert "PRIMARY_ONLY" in decision["next_action"]


def test_transfer_failure_drops_scope_without_overriding_primary_pass():
    rows = [
        result("openvla", 0),
        result("openvla", 2),
        result("pi0", 0),
        result("pi0", 2, passed=False),
    ]
    decision = MODULE.audit_cells(rows)
    assert decision["status"] == "PRIMARY_GO_SINGLE_POLICY_SCOPE"
    assert decision["transfer_failure_changes_scope_not_primary_validity"]


def test_primary_failure_is_hard_no_go():
    rows = [result("openvla", 0), result("openvla", 2, passed=False)]
    decision = MODULE.audit_cells(rows)
    assert decision["status"] == "PRIMARY_OPENVLA_NO_GO"


def test_all_four_pass_is_full_d1_go_and_deterministic():
    rows = [result(backend, task) for backend, task in MODULE.EXPECTED_CELLS]
    first = MODULE.audit_cells(rows)
    second = MODULE.audit_cells(reversed(rows))
    assert first == second
    assert first["status"] == "PRIMARY_AND_TRANSFER_GO"


def test_duplicate_cell_is_invalid():
    decision = MODULE.audit_cells([result("openvla", 0), result("openvla", 0)])
    assert decision["status"] == "INVALID_DUPLICATE_CELL"
