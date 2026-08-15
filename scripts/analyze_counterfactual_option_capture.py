#!/usr/bin/env python3
"""Audit exact-state option outcomes before fitting a router."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


OPTIONS = ("base_continue", "detour_complete", "retreat_hold")
OUTCOMES = ("task_success", "catastrophe", "safe_noncompletion")
UTILITY = {"task_success": 1.0, "safe_noncompletion": 0.0, "catastrophe": -5.0}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_decisions(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        grouped[str(row["decision_id"])].append(row)
    decisions = []
    for decision_id, group in grouped.items():
        by_option = {str(row["option"]): row for row in group}
        if len(group) != len(OPTIONS) or set(by_option) != set(OPTIONS):
            raise ValueError(f"decision {decision_id} does not have exactly three options")
        first = group[0]
        invariant = (
            "source_state_sha256", "placement_key", "split", "condition",
            "horizon_actions", "feature_index",
        )
        for key in invariant:
            if len({json.dumps(row[key], sort_keys=True) for row in group}) != 1:
                raise ValueError(f"decision {decision_id} disagrees on {key}")
        outcomes = {option: str(by_option[option]["outcome"]) for option in OPTIONS}
        if any(outcome not in OUTCOMES for outcome in outcomes.values()):
            raise ValueError(f"decision {decision_id} has an invalid outcome")
        decisions.append({
            "decision_id": decision_id,
            **{key: first[key] for key in invariant},
            "outcomes": outcomes,
        })
    decisions.sort(key=lambda row: (
        str(row["split"]), str(row["source_state_sha256"]),
        str(row["condition"]), -int(row["horizon_actions"]),
    ))
    return decisions


def _fixed_method(decisions: list[dict[str, Any]], option: str) -> dict[str, Any]:
    outcomes = Counter(row["outcomes"][option] for row in decisions)
    count = len(decisions)
    utility = sum(UTILITY[row["outcomes"][option]] for row in decisions)
    return {
        "decisions": count,
        "outcomes": {name: outcomes.get(name, 0) for name in OUTCOMES},
        "rates": {
            name: outcomes.get(name, 0) / count if count else 0.0 for name in OUTCOMES
        },
        "mean_utility_lambda_5": utility / count if count else 0.0,
    }


def summarize_decisions(decisions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    decisions = [dict(row) for row in decisions]
    count = len(decisions)
    patterns = Counter(tuple(row["outcomes"][option] for option in OPTIONS) for row in decisions)
    oracle_choices = Counter()
    oracle_outcomes = Counter()
    oracle_utility = 0.0
    base_utility = 0.0
    positive_value = 0
    for row in decisions:
        utilities = {option: UTILITY[row["outcomes"][option]] for option in OPTIONS}
        best = max(utilities.values())
        # Prefer Base on a utility tie to minimize unnecessary intervention.
        choice = next(option for option in OPTIONS if utilities[option] == best)
        oracle_choices[choice] += 1
        oracle_outcomes[row["outcomes"][choice]] += 1
        oracle_utility += best
        base_utility += utilities["base_continue"]
        positive_value += best > utilities["base_continue"]
    return {
        "decisions": count,
        "source_states": len({str(row["source_state_sha256"]) for row in decisions}),
        "all_options_identical": sum(
            len(set(row["outcomes"].values())) == 1 for row in decisions
        ),
        "counterfactual_outcome_diversity": sum(
            len(set(row["outcomes"].values())) > 1 for row in decisions
        ),
        "positive_oracle_value_over_base": positive_value,
        "base_catastrophe_detour_success": sum(
            row["outcomes"]["base_continue"] == "catastrophe"
            and row["outcomes"]["detour_complete"] == "task_success"
            for row in decisions
        ),
        "base_catastrophe_retreat_safe": sum(
            row["outcomes"]["base_continue"] == "catastrophe"
            and row["outcomes"]["retreat_hold"] == "safe_noncompletion"
            for row in decisions
        ),
        "base_success_detour_worse": sum(
            row["outcomes"]["base_continue"] == "task_success"
            and row["outcomes"]["detour_complete"] != "task_success"
            for row in decisions
        ),
        "base_success_retreat_worse": sum(
            row["outcomes"]["base_continue"] == "task_success"
            and row["outcomes"]["retreat_hold"] != "task_success"
            for row in decisions
        ),
        "fixed_methods": {
            option: _fixed_method(decisions, option) for option in OPTIONS
        },
        "counterfactual_oracle": {
            "choice_counts_tie_break_base": {
                option: oracle_choices.get(option, 0) for option in OPTIONS
            },
            "outcomes": {name: oracle_outcomes.get(name, 0) for name in OUTCOMES},
            "rates": {
                name: oracle_outcomes.get(name, 0) / count if count else 0.0
                for name in OUTCOMES
            },
            "mean_utility_lambda_5": oracle_utility / count if count else 0.0,
            "mean_utility_gain_over_base": (
                (oracle_utility - base_utility) / count if count else 0.0
            ),
            "intervention_rate_tie_break_base": (
                (count - oracle_choices.get("base_continue", 0)) / count if count else 0.0
            ),
        },
        "outcome_patterns": [
            {"outcomes": dict(zip(OPTIONS, pattern)), "count": pattern_count}
            for pattern, pattern_count in sorted(
                patterns.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }


def analyze_capture(root: Path) -> dict[str, Any]:
    input_paths = {
        name: root / name
        for name in (
            "capture_manifest.json", "decision_metadata.json", "option_rollouts.jsonl"
        )
    }
    manifest = json.loads(input_paths["capture_manifest.json"].read_text())
    rows = [
        json.loads(line)
        for line in input_paths["option_rollouts.jsonl"].read_text().splitlines()
        if line.strip()
    ]
    decisions = group_decisions(rows)
    validation = manifest["validation"]
    if len(decisions) != int(validation["decision_states"]):
        raise ValueError("decision count does not match capture manifest")
    if len(rows) != int(validation["option_rollouts"]):
        raise ValueError("option-row count does not match capture manifest")
    by_split = {
        split: summarize_decisions([row for row in decisions if row["split"] == split])
        for split in sorted({str(row["split"]) for row in decisions})
    }
    by_condition = {
        condition: summarize_decisions([
            row for row in decisions if row["condition"] == condition
        ])
        for condition in sorted({str(row["condition"]) for row in decisions})
    }
    glass = [row for row in decisions if row["condition"] == "glass"]
    glass_by_horizon = {
        str(horizon): summarize_decisions([
            row for row in glass if int(row["horizon_actions"]) == horizon
        ])
        for horizon in sorted({int(row["horizon_actions"]) for row in glass}, reverse=True)
    }
    return {
        "schema_version": 1,
        "kind": "counterfactual_option_capture_audit",
        "capture_root": str(root),
        "capture_commit": manifest["repository"]["git_commit"],
        "source_artifact_sha256": {
            name: _sha256(path) for name, path in input_paths.items()
        },
        "utility": UTILITY,
        "overall": summarize_decisions(decisions),
        "by_split": by_split,
        "by_condition": by_condition,
        "glass_by_horizon": glass_by_horizon,
        "exclusion_counts": manifest["exclusion_counts"],
        "attempted_placements": manifest["attempted_placements"],
        "valid_placements": manifest["valid_placements"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument(
        "--capture-label",
        help="Stable display path to record instead of the local analysis copy.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze_capture(args.capture.resolve())
    if args.capture_label is not None:
        result["capture_root"] = args.capture_label
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
