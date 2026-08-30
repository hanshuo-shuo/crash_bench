from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/analyze_formal_nominal.py"
SPEC = importlib.util.spec_from_file_location("analyze_formal_nominal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def shards(successes_per_task=55):
    rows = []
    for shard_index, (task, start) in enumerate(
        [(0, 0), (0, 21), (0, 42), (2, 0), (2, 21), (2, 42)]
    ):
        shard_rows = []
        for index in range(start, start + 21):
            success = index < successes_per_task
            shard_rows.append(
                {
                    "attempt_id": f"{task}:{index}",
                    "task_id": task,
                    "candidate_index": index,
                    "reset_seed": 100000 + task * 100 + index,
                    "status": "NOMINAL_COMPLETE",
                    "task_success": success,
                    "source_state_sha256": f"{task * 1000 + index + 1:064x}",
                    "scene_fingerprint": f"scene:{task}:{index}",
                    "source_state_dtype": "float64",
                    "source_state": [float(index)],
                    "option_outcomes_opened": 0,
                }
            )
        rows.append({"shard_index": shard_index, "rows": shard_rows})
    return rows


def test_selects_exactly_50_per_task_when_support_sufficient():
    result = MODULE.analyze_formal_shards(shards())
    assert result["status"] == "GO"
    assert result["selected_physical_sources"] == 100
    assert result["task_summary"]["0"]["selected"] == 50
    assert result["task_summary"]["2"]["selected"] == 50


def test_insufficient_success_support_is_no_go():
    result = MODULE.analyze_formal_shards(shards(successes_per_task=49))
    assert result["status"] == "NO_GO"
    assert "task0_unique_successful_support" in result["hard_failures"]
    assert "task2_unique_successful_support" in result["hard_failures"]
