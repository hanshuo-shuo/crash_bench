from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path

from crashbench.branching.artifacts import ContentAddressedStore
from crashbench.data.utility import PhysicalBudgets


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/merge_statewise_dataset.py"
SPEC = importlib.util.spec_from_file_location("merge_statewise_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_merge_validates_blob_exactness_and_recomputes_u0(tmp_path):
    store = ContentAddressedStore(tmp_path / "store")
    ref = store.put_bytes(b"feature")
    outcomes = []
    for option, terminal in (
        ("base_continue", (1, 0, 0, 0)),
        ("observation_refresh", (1, 0, 0, 1)),
        ("safe_stop", (0, 0, 1, 1)),
    ):
        success, catastrophe, noncompletion, intervention = terminal
        outcomes.append(
            {
                "option_id": option,
                "branch_start_bundle_id": "bundle",
                "branch_start_component_hashes": {"sim": "hash"},
                "mechanism_queue_sha256": "queue",
                "terminal_signature": option,
                "outcome": {
                    "task_success": success,
                    "catastrophe": catastrophe,
                    "safe_noncompletion": noncompletion,
                    "intervention_invoked": intervention,
                    "human_help": 0,
                    "option_duration_steps": 10,
                    "path_length_m": 0.1,
                    "force_exposure_ns": 0,
                    "latency_ms": 10,
                },
            }
        )
    shard = {
        "assignment_index": 0,
        "split_role": "train",
        "test_rows_read": 0,
        "planned_blocks": 27,
        "all_blocks_accounted": True,
        "physical_source_id": "physical",
        "mechanism_source_id": "mechanism",
        "policy_source_id": "policy",
        "task_id": "libero_spatial:0",
        "blocks": [
            {
                "block_id": "block",
                "anchor_steps": 5,
                "severity_id": "delay_1",
                "condition": "stale",
                "status": "COMPLETE_REALIZED_OPTIONS",
                "bundle_id": "bundle",
                "mechanism_queue_sha256": "queue",
                "anchor_feature_blob": asdict(ref),
                "option_outcomes": outcomes,
            }
        ]
        + [
            {
                "block_id": f"invalid-{index}",
                "anchor_steps": 5,
                "severity_id": "delay_1",
                "condition": "stale",
                "status": "PRE_ANCHOR_MECHANICAL_INVALID",
                "reason": "fixture",
            }
            for index in range(26)
        ],
    }
    manifest, anchors, branches, invalid = MODULE.merge_shards(
        [shard],
        store=store,
        budgets=PhysicalBudgets(100, 2, 75, 1000, "physical limits"),
        expected_indices=[0],
    )
    assert manifest["status"] == "GO"
    assert len(anchors) == 1 and len(branches) == 3 and len(invalid) == 26
    assert next(row for row in branches if row["option_id"] == "base_continue")["u0"] > 0


def test_merge_rejects_test_role_and_missing_source(tmp_path):
    store = ContentAddressedStore(tmp_path / "store")
    shard = {
        "assignment_index": 0,
        "split_role": "confirmatory_id_test",
        "test_rows_read": 1,
        "planned_blocks": 27,
        "all_blocks_accounted": True,
        "physical_source_id": "p",
        "mechanism_source_id": "m",
        "policy_source_id": "q",
        "task_id": "t",
        "blocks": [],
    }
    manifest, *_ = MODULE.merge_shards(
        [shard],
        store=store,
        budgets=PhysicalBudgets(100, 2, 75, 1000, "physical limits"),
        expected_indices=[0, 1],
    )
    assert manifest["status"] == "NO_GO"
    assert "missing_assignment_indices" in manifest["errors"]
    assert any(error.startswith("forbidden_split_role") for error in manifest["errors"])
