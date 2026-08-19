from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from crashbench.direct_recovery_router import (
    DIRECT_OPTIONS,
    DirectFirstCrossingRouter,
    DirectRecoveryWindowHead,
    router_output_feature,
)
from scripts.analyze_direct_recovery_head_dynamic_closeout import (
    analyze,
    paired_source_differences,
    summarize_method,
)
from scripts.calibrate_direct_recovery_head_sequential import SEQUENTIAL_ALPHA


def test_router_output_feature_uses_frozen_ten_value_order():
    probabilities = np.arange(9, dtype=np.float64).reshape(3, 3) / 10.0
    feature = router_output_feature({
        "option_outcome_probabilities": probabilities,
        "base_catastrophe_probability": 0.91,
    })

    assert feature.shape == (10,)
    assert np.array_equal(feature[:9], probabilities.reshape(-1))
    assert feature[9] == 0.91


def _head() -> DirectRecoveryWindowHead:
    weight = np.zeros((10, 3), dtype=np.float64)
    weight[0] = np.asarray([0.0, 2.0, -2.0])
    return DirectRecoveryWindowHead(
        feature_mean=np.zeros(10),
        feature_scale=np.ones(10),
        head_weight=weight,
        head_bias=np.asarray([0.0, 0.0, 2.0]),
    )


def test_direct_first_crossing_is_strict_and_latches_nonbase_argmax():
    gate = DirectFirstCrossingRouter(head=_head(), sequential_boundary=2.0)

    equal = gate.observe(np.zeros(10), action_index=4)
    assert equal["margin"] == 2.0
    assert equal["first_crossing"] is False
    assert gate.selected_option_index is None

    crossing_feature = np.zeros(10)
    crossing_feature[0] = 2.0
    crossing = gate.observe(crossing_feature, action_index=5)
    assert crossing["candidate_option_index"] == DIRECT_OPTIONS.index("Detour")
    assert crossing["first_crossing"] is True
    assert gate.selected_option_index == DIRECT_OPTIONS.index("Detour")
    assert gate.trigger_action_index == 5

    later_hold = gate.observe(np.zeros(10), action_index=6)
    assert later_hold["candidate_option_index"] == DIRECT_OPTIONS.index("FailSafeHold")
    assert later_hold["first_crossing"] is False
    assert gate.selected_option_index == DIRECT_OPTIONS.index("Detour")


def test_tracked_p3_head_loads_as_exact_ten_dimensional_linear_model():
    path = Path(__file__).resolve().parents[1] / "results" / (
        "p3_1_direct_recovery_head_model.npz"
    )
    head = DirectRecoveryWindowHead.load(path)
    probabilities = head.probabilities(np.zeros(10))

    assert head.classes == DIRECT_OPTIONS
    assert head.head_weight.shape == (10, 3)
    assert probabilities.shape == (3,)
    assert np.isclose(probabilities.sum(), 1.0)
    assert SEQUENTIAL_ALPHA == 0.1


def _row(source: str, condition: str, method: str, outcome: str, option: str) -> dict:
    known = condition == "glass"
    intervened = option != "Base"
    return {
        "episode_id": f"{source}:{condition}:{method}",
        "placement_id": f"placement-{source}",
        "source_state_sha256": source,
        "condition": condition,
        "method": method,
        "reference_base_outcome": (
            "catastrophe" if condition == "glass" else "task_success"
        ),
        "reference_collision_action_index": 30 if condition == "glass" else None,
        "outcome": outcome,
        "task_success": outcome == "task_success",
        "catastrophe": outcome == "catastrophe",
        "safe_noncompletion": outcome == "safe_noncompletion",
        "intervened": intervened,
        "selected_option": option,
        "trigger_action_index": 5 if intervened else None,
        "intervention_duration_actions": 20 if intervened else 0,
        "contact_force_p95_n": 0.0,
        "contact_force_max_n": 0.0,
        "known_recovery_t20": known,
        "known_recovery_anchor_action_index": 11 if known else None,
        "missed_known_recovery_opportunity": known and outcome != "task_success",
        "late_or_absent_on_known_recovery": known and not intervened,
        "steps": 40,
        "termination": "synthetic",
    }


def _synthetic_rows() -> list[dict]:
    rows = []
    for source_index in range(8):
        source = f"source-{source_index}"
        for condition in ("glass", "offpath", "noglass"):
            base_outcome = "catastrophe" if condition == "glass" else "task_success"
            rows.append(_row(source, condition, "Base", base_outcome, "Base"))
            rows.append(_row(
                source,
                condition,
                "P2SequentialRouter",
                base_outcome,
                "Base",
            ))
            direct_outcome = (
                "task_success"
                if condition == "glass" and source_index < 4
                else base_outcome
            )
            direct_option = (
                "Detour" if condition == "glass" and source_index < 4 else "Base"
            )
            rows.append(_row(
                source,
                condition,
                "DirectRecoveryRouter",
                direct_outcome,
                direct_option,
            ))
    return rows


def test_method_summary_and_source_pairing_use_matched_source_units():
    rows = _synthetic_rows()
    direct = summarize_method([
        row for row in rows if row["method"] == "DirectRecoveryRouter"
    ])
    _, paired = paired_source_differences(rows)

    assert direct["outcomes"]["task_success"]["count"] == 20
    assert direct["glass"]["recovered_base_catastrophe_count"] == 4
    assert direct["base_success_controls"]["base_decision_retention_rate"] == 1.0
    assert direct["known_recovery_t20"]["missed_count"] == 4
    assert paired["Direct_minus_P2"]["metrics"]["task_success"][
        "mean_source_paired_rate_difference"
    ] == 1.0 / 6.0
    assert paired["Direct_minus_P2"]["metrics"]["catastrophe"][
        "mean_source_paired_rate_difference"
    ] == -1.0 / 6.0


def test_closeout_analysis_writes_required_stopped_artifacts(tmp_path: Path):
    rows = _synthetic_rows()
    (tmp_path / "method_episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    diagnostics = [
        {
            "source_state_sha256": f"source-{index}",
            "placement_id": f"placement-source-{index}",
            "condition": "glass",
            "outcome": "task_success",
        }
        for index in range(8)
    ]
    (tmp_path / "known_recovery_t20_diagnostics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in diagnostics), encoding="utf-8"
    )
    (tmp_path / "router_trace.jsonl").write_text(
        json.dumps({"synthetic": True}) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "single frozen dynamic closeout complete; no retuning permitted",
        "direct_head_model_sha256": "a" * 64,
        "direct_sequential_boundary": {"margin": 1.25},
        "old_sequential_boundary": {"effective_margin": 1.5},
    }
    (tmp_path / "capture_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    result = analyze(tmp_path)

    assert result["counts"]["sources"] == 8
    assert result["counts"]["source_condition_units"] == 24
    assert result["counts"]["method_episode_rows"] == 72
    assert all(value is False for value in result["no_retuning_audit"].values())
    for name in (
        "analysis.json", "REPORT.md", "paired_source_differences.csv"
    ):
        assert (tmp_path / name).exists()
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "experiment stopped" in report
    assert "No head refit" in report
