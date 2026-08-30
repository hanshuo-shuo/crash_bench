from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/collect_staleness_statewise.py"
SPEC = importlib.util.spec_from_file_location("collect_staleness_statewise", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_active_assignments_excludes_every_test_role():
    rows = []
    for task in (0, 2):
        for role, count in {
            "train": 12,
            "calibration": 6,
            "development": 6,
            "confirmatory_id_test": 16,
            "fresh_sequential_test": 10,
        }.items():
            rows.extend(
                {
                    "task_id": f"libero_spatial:{task}",
                    "role": role,
                    "physical_source_id": f"{task}:{role}:{index}",
                }
                for index in range(count)
            )
    active = MODULE.active_assignments({"assignments": rows})
    assert len(active) == 48
    assert {row["role"] for row in active} == {"train", "calibration", "development"}
    assert not any("test" in row["role"] for row in active)


def test_active_assignment_count_fails_closed():
    with pytest.raises(ValueError, match="48"):
        MODULE.active_assignments({"assignments": []})


def test_test_assignments_are_exact_and_separate_from_active_roles():
    rows = [
        {
            "task_id": f"libero_spatial:{task}",
            "role": "confirmatory_id_test",
            "physical_source_id": f"{task}:test:{index}",
        }
        for task in (0, 2)
        for index in range(16)
    ]
    selected = MODULE.test_assignments({"assignments": rows})
    assert len(selected) == 32
    assert {row["role"] for row in selected} == {"confirmatory_id_test"}


def test_collector_source_requires_full_test_authorization_contract():
    text = SCRIPT.read_text()
    for token in (
        "validate_test_retry", "benchmark_freeze", "test_source_manifest_sha256",
        "collector run_id differs", "test_rows_read = 1",
    ):
        assert token in text


def test_formal_outcome_contract_declares_all_u0_continuous_fields():
    text = SCRIPT.read_text()
    for field in (
        "option_duration_steps",
        "path_length_m",
        "force_exposure_ns",
        "latency_ms",
        "inference_latency_ms",
        "actuation_latency_ms",
        "max_force_n",
    ):
        assert field in text
