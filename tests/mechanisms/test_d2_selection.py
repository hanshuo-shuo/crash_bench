from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/analyze_d2_selection.py"
SPEC = importlib.util.spec_from_file_location("analyze_d2_selection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def result(mechanism, status):
    return {
        "mechanism_id": mechanism,
        "analysis_sha256": mechanism,
        "gate": {
            "status": status,
            "hard_failures": [] if status in {"GO", "SCOPED_CONTINUE"} else ["hard"],
            "missing_evidence": [],
            "claim_scope_failures": [] if status == "GO" else ["scope"],
        },
    }


AUTHORIZED = [
    "fragile_path_collision_v2",
    "observation_staleness_v1",
    "action_drift_v1",
    "narrow_clearance_v1",
]


def test_three_go_is_strict_broad_pass():
    rows = [result(mechanism, "GO" if index < 3 else "SCOPED_CONTINUE") for index, mechanism in enumerate(AUTHORIZED)]
    decision = MODULE.resolve_selection(rows)
    assert decision["status"] == "BROAD_GATE_A0_GO"
    assert decision["strict_broad_gate_pass"]


def test_two_go_is_scoped_not_fake_broad_pass():
    rows = [result(mechanism, "GO" if index < 2 else "SCOPED_CONTINUE") for index, mechanism in enumerate(AUTHORIZED)]
    decision = MODULE.resolve_selection(rows)
    assert decision["status"] == "SCOPED_TWO_MECHANISM_CONTINUE"
    assert not decision["strict_broad_gate_pass"]


def test_one_go_with_three_hard_valid_is_labelled_pilot():
    rows = [result(mechanism, "GO" if index == 0 else "SCOPED_CONTINUE") for index, mechanism in enumerate(AUTHORIZED)]
    decision = MODULE.resolve_selection(rows)
    assert decision["status"] == "SINGLE_FORMAL_PLUS_EXPLORATORY_METHOD_PILOT"


def test_missing_authorized_mechanism_is_incomplete():
    rows = [result(mechanism, "GO") for mechanism in AUTHORIZED[:-1]]
    assert MODULE.resolve_selection(rows)["status"] == "INCOMPLETE"
