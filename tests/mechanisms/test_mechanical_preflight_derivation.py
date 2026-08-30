from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/derive_mechanical_preflight.py"
SPEC = importlib.util.spec_from_file_location("derive_mechanical_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(mechanism):
    return {
        "attempt_id": "attempt",
        "mechanism_id": mechanism,
        "task_id": 0,
        "candidate_index": 0,
        "reset_seed": 1,
        "source_state_sha256": "a" * 64,
        "status": "NOMINAL_COMPLETE",
        "eef_path_xyz": [[0, 0, 0.9], [0.2, 0, 0.9], [0.4, 0, 0.9]],
    }


def test_fragile_derivation_requires_support_and_builds_three_controls():
    result = MODULE.derive_geometry_for_row(
        row("fragile_path_collision_v2"), support_top_z=0.75
    )
    assert result["status"] == "MECHANICAL_VALID"
    assert set(result["severities"]) == {"radius_020", "radius_025", "radius_030"}
    first = result["severities"]["radius_020"]["conditions"]
    assert set(first) == {"on_path", "off_path_control", "no_object_control"}


def test_narrow_derivation_uses_live_robot_width():
    result = MODULE.derive_geometry_for_row(
        row("narrow_clearance_v1"), support_top_z=0.75, robot_envelope_width_m=0.12
    )
    assert result["status"] == "MECHANICAL_VALID"
    assert set(result["severities"]) == {"clearance_005", "clearance_010", "clearance_015"}
    assert result["severities"]["clearance_005"]["conditions"]["narrow"]["gap_m"] == 0.125


def test_staleness_and_drift_need_no_live_geometry():
    stale = MODULE.derive_geometry_for_row(row("observation_staleness_v1"))
    drift = MODULE.derive_geometry_for_row(row("action_drift_v1"))
    assert stale["status"] == drift["status"] == "MECHANICAL_VALID"
    assert len(stale["severities"]) == len(drift["severities"]) == 3


def test_missing_nominal_path_is_retained_as_invalid():
    bad = row("action_drift_v1")
    bad["status"] = "NOMINAL_OPERATIONAL_FAILURE"
    bad["eef_path_xyz"] = []
    result = MODULE.derive_geometry_for_row(bad)
    assert result["status"] == "MECHANICAL_INVALID"
    assert result["option_outcomes_opened"] == 0
