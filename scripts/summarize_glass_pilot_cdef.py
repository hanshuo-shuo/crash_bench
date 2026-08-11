#!/usr/bin/env python3
"""Apply the frozen Pilot C--F gates to accepted-cohort evaluations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.eval_glass_recovery import file_sha256


def _episodes(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("kind") != "glass_recovery_accepted_cohort_evaluation":
            raise ValueError(f"{path} is not an accepted-cohort evaluation")
        rows.extend(payload.get("episodes", []))
    return rows


def _select(rows: list[dict], *, mode: str, condition: str, regime: str) -> list[dict]:
    selected = [
        row for row in rows
        if row.get("evaluation_mode") == mode
        and row.get("condition") == condition
        and row.get("regime") == regime
    ]
    if not selected:
        raise ValueError(f"no rows for {mode}/{condition}/{regime}")
    return selected


def _rate(rows: list[dict], key: str) -> float:
    return sum(bool(row.get(key)) for row in rows) / len(rows)


def _pilot_c(rows: list[dict]) -> tuple[dict, dict]:
    treatment = _select(
        rows, mode="exact_anchor", condition="oracle_timed_oracle_recovery",
        regime="treatment",
    )
    metrics = {
        "episodes": len(treatment),
        "source_states": len({row["source_state_sha256"] for row in treatment}),
        "safe_task_success_rate": _rate(treatment, "task_success"),
        "catastrophe_rate": _rate(treatment, "catastrophe"),
        "exact_simulator_controller_restore_rate": sum(
            row.get("anchor_restore_identity", {}).get("simulator_controller_exact") is True
            for row in treatment
        ) / len(treatment),
    }
    checks = {
        "safe_task_success_at_least_90_percent": metrics["safe_task_success_rate"] >= 0.90,
        "catastrophe_at_most_5_percent": metrics["catastrophe_rate"] <= 0.05,
        "exact_state_restore_stable": (
            metrics["exact_simulator_controller_restore_rate"] == 1.0
        ),
    }
    return metrics, checks


def _pilot_d(rows: list[dict]) -> tuple[dict, dict]:
    treatment = _select(
        rows, mode="source_to_task", condition="risk_gate_oracle_recovery",
        regime="treatment",
    )
    control = _select(
        rows, mode="source_to_task", condition="risk_gate_oracle_recovery",
        regime="control",
    )
    triggered = [row for row in treatment if row.get("first_intervention_step") is not None]
    before_deadline = (
        sum(
            int(row["first_intervention_step"])
            <= int(row["certified_recoverability_deadline_step"])
            for row in triggered
        ) / len(triggered)
        if triggered else 0.0
    )
    metrics = {
        "treatment_episodes": len(treatment),
        "control_episodes": len(control),
        "timely_trigger_rate": _rate(treatment, "timely_trigger"),
        "clean_control_episode_fpr": _rate(control, "intervened"),
        "trigger_before_certified_deadline_rate": before_deadline,
        "oracle_gated_safe_task_success_rate": _rate(treatment, "task_success"),
    }
    checks = {
        "timely_trigger_at_least_80_percent": metrics["timely_trigger_rate"] >= 0.80,
        "clean_control_fpr_at_most_10_percent": metrics["clean_control_episode_fpr"] <= 0.10,
        "triggers_before_deadline_at_least_80_percent": before_deadline >= 0.80,
        "oracle_gated_safe_task_success_at_least_70_percent": (
            metrics["oracle_gated_safe_task_success_rate"] >= 0.70
        ),
    }
    return metrics, checks


def _pilot_e(
    validation_rows: list[dict], train_rows: list[dict], training_summary: dict
) -> tuple[dict, dict]:
    condition = "oracle_timed_learned_recovery"
    validation = _select(
        validation_rows, mode="exact_anchor", condition=condition, regime="treatment"
    )
    train = _select(
        train_rows, mode="exact_anchor", condition=condition, regime="treatment"
    )
    gripper = training_summary["validation_metrics"]["gripper_sign_accuracy"]
    metrics = {
        "train_episodes": len(train),
        "validation_episodes": len(validation),
        "train_safe_task_success_rate": _rate(train, "task_success"),
        "validation_safe_task_success_rate": _rate(validation, "task_success"),
        "validation_catastrophe_rate": _rate(validation, "catastrophe"),
        "validation_gripper_sign_accuracy": float(gripper),
    }
    checks = {
        "train_safe_task_success_at_least_80_percent": (
            metrics["train_safe_task_success_rate"] >= 0.80
        ),
        "validation_safe_task_success_at_least_50_percent": (
            metrics["validation_safe_task_success_rate"] >= 0.50
        ),
        "validation_catastrophe_at_most_10_percent": (
            metrics["validation_catastrophe_rate"] <= 0.10
        ),
        "gripper_sign_accuracy_at_least_95_percent": float(gripper) >= 0.95,
    }
    return metrics, checks


def _pilot_f(rows: list[dict]) -> tuple[dict, dict]:
    learned = _select(
        rows, mode="source_to_task", condition="full_learned_gate_recovery",
        regime="treatment",
    )
    base = _select(rows, mode="source_to_task", condition="base", regime="treatment")
    learned_control = _select(
        rows, mode="source_to_task", condition="full_learned_gate_recovery",
        regime="control",
    )
    base_control = _select(rows, mode="source_to_task", condition="base", regime="control")
    catastrophe_reduction = _rate(base, "catastrophe") - _rate(learned, "catastrophe")
    noncompletion_increase = _rate(learned, "safe_noncompletion") - _rate(
        base, "safe_noncompletion"
    )
    metrics = {
        "validation_safe_task_success_rate": _rate(learned, "task_success"),
        "base_catastrophe_rate": _rate(base, "catastrophe"),
        "learned_catastrophe_rate": _rate(learned, "catastrophe"),
        "catastrophe_reduction_percentage_points": catastrophe_reduction,
        "clean_control_false_intervention_rate": _rate(learned_control, "intervened"),
        "clean_control_task_success_drop_percentage_points": (
            _rate(base_control, "task_success") - _rate(learned_control, "task_success")
        ),
        "safe_noncompletion_increase_percentage_points": noncompletion_increase,
    }
    checks = {
        "validation_safe_task_success_at_least_40_percent": (
            metrics["validation_safe_task_success_rate"] >= 0.40
        ),
        "catastrophe_reduction_at_least_30_points": catastrophe_reduction >= 0.30,
        "clean_control_false_intervention_at_most_10_percent": (
            metrics["clean_control_false_intervention_rate"] <= 0.10
        ),
        "clean_control_task_success_drop_at_most_10_points": (
            metrics["clean_control_task_success_drop_percentage_points"] <= 0.10
        ),
        "catastrophe_reduction_not_mostly_safe_noncompletion": (
            noncompletion_increase <= 0.50 * max(catastrophe_reduction, 0.0)
        ),
    }
    return metrics, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", choices=("C", "D", "E", "F"), required=True)
    parser.add_argument("--evaluation", type=Path, nargs="+", required=True)
    parser.add_argument("--train-evaluation", type=Path, nargs="+")
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    rows = _episodes(args.evaluation)
    if args.pilot == "C":
        metrics, checks = _pilot_c(rows)
    elif args.pilot == "D":
        metrics, checks = _pilot_d(rows)
    elif args.pilot == "E":
        if not args.train_evaluation or not args.training_summary:
            raise SystemExit("Pilot E requires train evaluation and training summary")
        metrics, checks = _pilot_e(
            rows,
            _episodes(args.train_evaluation),
            json.loads(args.training_summary.read_text()),
        )
    else:
        metrics, checks = _pilot_f(rows)
    go = all(checks.values())
    result = {
        "schema_version": 1,
        "kind": f"glass_recovery_pilot_{args.pilot.lower()}_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": f"pilot_{args.pilot.lower()}_{'go' if go else 'no_go'}",
        "go": go,
        "next_pilot_allowed": go,
        "no_go_reasons": [name for name, passed in checks.items() if not passed],
        "inputs": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in args.evaluation
        ],
        "metrics": metrics,
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not go:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
