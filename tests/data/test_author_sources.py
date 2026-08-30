from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/author_sources.py"
SPEC = importlib.util.spec_from_file_location("author_sources", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def config():
    return {
        "tasks": [
            {"suite": "libero_spatial", "task_id": 0},
            {"suite": "libero_spatial", "task_id": 2},
        ],
        "screen": {
            "ordered_mechanisms": ["a", "b", "c", "d", "e"],
            "sources_per_mechanism_task": 8,
            "base_reset_seed": 1000,
        },
        "formal_nominal": {
            "candidates_per_task": 188,
            "base_reset_seed": 2000,
            "option_outcomes_allowed_during_enumeration": False,
            "router_scores_allowed": False,
        },
    }


def test_screen_plan_has_all_5_by_2_by_8_attempts_and_unique_seeds():
    rows = MODULE.build_source_plan(config(), mode="screen", protocol_sha256="a" * 64)
    assert len(rows) == 5 * 2 * 8 == 80
    assert len({row["reset_seed"] for row in rows}) == 80
    assert len({row["attempt_id"] for row in rows}) == 80


def test_formal_nominal_plan_is_376_label_free_attempts():
    rows = MODULE.build_source_plan(
        config(), mode="formal_nominal", protocol_sha256="a" * 64
    )
    assert len(rows) == 2 * 188 == 376
    assert all(row["mechanism_id"] is None for row in rows)


def test_plan_is_deterministic_and_protocol_bound():
    first = MODULE.build_source_plan(config(), mode="screen", protocol_sha256="a" * 64)
    second = MODULE.build_source_plan(config(), mode="screen", protocol_sha256="a" * 64)
    changed = MODULE.build_source_plan(config(), mode="screen", protocol_sha256="b" * 64)
    assert first == second
    assert [row["attempt_id"] for row in first] != [row["attempt_id"] for row in changed]


def test_formal_authoring_refuses_score_or_outcome_access():
    bad = config()
    bad["formal_nominal"]["router_scores_allowed"] = True
    try:
        MODULE.build_source_plan(bad, mode="formal_nominal", protocol_sha256="a" * 64)
    except ValueError as exc:
        assert "router scores" in str(exc)
    else:
        raise AssertionError("score-guided source authoring was accepted")


def test_cli_appends_exposure_before_writing_plan(tmp_path):
    config_path = tmp_path / "sampling.json"
    config_path.write_text(json.dumps(config()))
    ledger = tmp_path / "exposure.jsonl"
    output = tmp_path / "plan.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config_path),
            "--mode",
            "screen",
            "--protocol-sha256",
            "a" * 64,
            "--exposure-ledger",
            str(ledger),
            "--output",
            str(output),
        ],
        check=True,
        cwd=SCRIPT.parents[2],
        capture_output=True,
        text=True,
    )
    plan = json.loads(output.read_text())
    attempts = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(attempts) == plan["source_attempt_count"] == 80
    assert all(row["test_eligible"] is False for row in attempts)
    assert plan["outcomes_opened"] == plan["router_scores_read"] == 0
