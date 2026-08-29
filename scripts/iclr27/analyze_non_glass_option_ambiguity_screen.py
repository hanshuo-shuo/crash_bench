#!/usr/bin/env python3
"""Deterministic source-aware analysis for the bounded non-glass screen."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.unstable_placement import (
    FAMILY_ID,
    OPTIONS,
    SCHEMA_VERSION,
    UTILITY,
    branch_hashes_identical,
    canonical_sha256,
    strict_best_action,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")


def _canonical_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        decision_id = str(row.get("canonical_decision_id", ""))
        if not decision_id:
            continue
        prior = output.get(decision_id)
        if prior is not None and canonical_sha256(prior) != canonical_sha256(row):
            raise ValueError(f"conflicting canonical rows for {decision_id}")
        output[decision_id] = row
    return output


def analyze_screen(result_root: str | Path) -> dict[str, Any]:
    root = Path(result_root).resolve()
    manifest = _read_json(root / "screen_manifest.json")
    config = _read_json(root / "resolved_config.json")
    sources = _read_json(root / "source_registry.json")
    attempts = _read_jsonl(root / "attempted_configurations.jsonl")
    validity_rows = _read_jsonl(root / "mechanical_validity.jsonl")
    audit_rows = _read_jsonl(root / "restoration_audit.jsonl")
    metadata_rows = _read_jsonl(root / "decision_metadata.jsonl")
    outcome_rows = _read_jsonl(root / "option_outcomes.jsonl")

    if manifest.get("family_id") != FAMILY_ID:
        raise ValueError("screen manifest uses the wrong mechanical family")
    if manifest.get("protocol_sha256") != canonical_sha256(manifest["frozen_protocol"]):
        raise ValueError("screen manifest protocol fingerprint is invalid")
    if len(attempts) > 108:
        raise ValueError("attempted decision cap exceeded")
    if len(outcome_rows) > 324:
        raise ValueError("ordinary option-outcome row cap exceeded")

    attempted_sources = {str(row["source_state_sha256"]) for row in attempts}
    selected_sources = {
        str(row["source_state_sha256"]) for row in sources.get("selected_sources", [])
    }
    if attempted_sources - selected_sources:
        raise ValueError("attempted grid contains an unregistered screen source")

    attempted_option_keys = {
        (
            str(row["source_state_sha256"]), str(row["parameter_id"]),
            int(row["horizon_actions"]), str(row["condition"]), option,
        )
        for row in attempts for option in OPTIONS
    }
    actual_option_keys = [
        (
            str(row.get("source_state_sha256")), str(row.get("parameter_id")),
            int(row.get("horizon_actions", -1)), str(row.get("condition")),
            str(row.get("option")),
        )
        for row in outcome_rows
    ]
    option_coverage_complete = (
        len(actual_option_keys) == len(set(actual_option_keys))
        and set(actual_option_keys) == attempted_option_keys
    )

    validity_by_decision = _canonical_rows(validity_rows)
    audits_by_decision = _canonical_rows(audit_rows)
    metadata_by_decision = _canonical_rows(
        row for row in metadata_rows if bool(row.get("canonical_evidence", False))
    )
    outcomes_by_decision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcome_rows:
        if bool(row.get("canonical_evidence", False)):
            outcomes_by_decision[str(row["canonical_decision_id"])].append(row)
    attempted_canonical_ids = {
        str(row["canonical_decision_id"]) for row in attempts
    }
    validity_coverage_complete = set(validity_by_decision) == attempted_canonical_ids

    invalidity_counts = Counter()
    for row in validity_by_decision.values():
        if not bool(row.get("eligible", False)):
            invalidity_counts[str(row.get("reason") or "unspecified_invalidity")] += 1

    decisions: list[dict[str, Any]] = []
    incomplete_counts = Counter()
    for decision_id, rows in sorted(outcomes_by_decision.items()):
        validity = validity_by_decision.get(decision_id, {})
        audit = audits_by_decision.get(decision_id, {})
        metadata = metadata_by_decision.get(decision_id, {})
        by_option = {str(row.get("option")): row for row in rows}
        if set(by_option) != set(OPTIONS):
            incomplete_counts["missing_option_outcome"] += 1
            continue
        if not branch_hashes_identical(rows):
            incomplete_counts["branch_start_hash_mismatch"] += 1
            continue
        if not bool(validity.get("eligible", False)):
            incomplete_counts["mechanically_ineligible"] += 1
            continue
        if not bool(audit.get("passed", False)):
            incomplete_counts["restoration_failed"] += 1
            continue
        outcomes = {option: str(by_option[option].get("outcome")) for option in OPTIONS}
        if any(value not in UTILITY for value in outcomes.values()):
            incomplete_counts["null_or_invalid_option_outcome"] += 1
            continue
        strict = strict_best_action(outcomes)
        base_r = int(outcomes["base_continue"] == "catastrophe")
        first = by_option[OPTIONS[0]]
        decisions.append({
            "canonical_decision_id": decision_id,
            "source_id": str(first["source_id"]),
            "source_state_sha256": str(first["source_state_sha256"]),
            "parameter_id": str(first["parameter_id"]),
            "condition": str(first["condition"]),
            "horizon_actions": int(first["horizon_actions"]),
            "R": base_r,
            "outcomes": outcomes,
            "utilities": {option: UTILITY[outcomes[option]] for option in OPTIONS},
            "strict_best_action": strict,
            "branch_start_sha256": str(first["branch_start_sha256"]),
            "anchor_index": metadata.get("anchor_index"),
        })

    strict_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    decision_by_source: Counter[str] = Counter()
    for row in decisions:
        source = row["source_state_sha256"]
        decision_by_source[source] += 1
        if row["strict_best_action"] is not None:
            strict_by_source[source][row["strict_best_action"]] += 1

    support_sources = {
        option: sorted(source for source, counts in strict_by_source.items() if counts[option] > 0)
        for option in OPTIONS
    }
    source_csv_rows = []
    selected_by_hash = {
        str(row["source_state_sha256"]): row for row in sources.get("selected_sources", [])
    }
    for source in sorted(selected_sources):
        counts = strict_by_source[source]
        source_csv_rows.append({
            "source_id": selected_by_hash[source]["source_id"],
            "source_state_sha256": source,
            "eligible_decisions": int(decision_by_source[source]),
            "strict_base_continue_decisions": int(counts["base_continue"]),
            "strict_stable_offset_place_decisions": int(counts["stable_offset_place"]),
            "strict_safe_setdown_decisions": int(counts["safe_setdown"]),
            "supports_base_continue": int(counts["base_continue"] > 0),
            "supports_stable_offset_place": int(counts["stable_offset_place"] > 0),
            "supports_safe_setdown": int(counts["safe_setdown"] > 0),
        })
    csv_path = root / "strict_support_by_source.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        fields = list(source_csv_rows[0]) if source_csv_rows else [
            "source_id", "source_state_sha256", "eligible_decisions",
            "strict_base_continue_decisions", "strict_stable_offset_place_decisions",
            "strict_safe_setdown_decisions", "supports_base_continue",
            "supports_stable_offset_place", "supports_safe_setdown",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(source_csv_rows)

    # Tight prospective match: source, condition, release horizon, and binary R
    # are exact; severity is the mechanically varied field.
    strata: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        if row["strict_best_action"] is not None:
            strata[(
                row["source_state_sha256"], row["condition"],
                row["horizon_actions"], row["R"],
            )].append(row)
    witnesses = []
    for key, rows in sorted(strata.items()):
        ordered = sorted(rows, key=lambda row: (row["parameter_id"], row["canonical_decision_id"]))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1:]:
                if left["strict_best_action"] == right["strict_best_action"]:
                    continue
                witnesses.append({
                    "schema_version": SCHEMA_VERSION,
                    "source_state_sha256": key[0],
                    "condition": key[1],
                    "horizon_actions": key[2],
                    "R": key[3],
                    "left_decision_id": left["canonical_decision_id"],
                    "left_parameter_id": left["parameter_id"],
                    "left_strict_action": left["strict_best_action"],
                    "left_outcomes": left["outcomes"],
                    "right_decision_id": right["canonical_decision_id"],
                    "right_parameter_id": right["parameter_id"],
                    "right_strict_action": right["strict_best_action"],
                    "right_outcomes": right["outcomes"],
                    "strict_intervention_flip": {
                        left["strict_best_action"], right["strict_best_action"]
                    } == {"stable_offset_place", "safe_setdown"},
                })
    _write_jsonl_exclusive(root / "same_r_option_witnesses.jsonl", witnesses)
    ambiguity_sources = sorted({row["source_state_sha256"] for row in witnesses})
    flip_sources = sorted({
        row["source_state_sha256"] for row in witnesses if row["strict_intervention_flip"]
    })

    audit_total = len(audits_by_decision)
    audit_passed = sum(bool(row.get("passed", False)) for row in audits_by_decision.values())
    restoration_rate = audit_passed / audit_total if audit_total else 0.0
    gate_cfg = config["gate"]
    gate_values = {
        "exact_restoration_all_retained": bool(audit_total and audit_passed == audit_total),
        "exact_restoration_passed": audit_passed,
        "exact_restoration_attempted": audit_total,
        "exact_restoration_pass_rate": restoration_rate,
        "strict_base_source_count": len(support_sources["base_continue"]),
        "strict_stable_offset_place_source_count": len(support_sources["stable_offset_place"]),
        "strict_safe_setdown_source_count": len(support_sources["safe_setdown"]),
        "same_r_different_strict_action_strata": len(witnesses),
        "same_r_different_strict_action_source_count": len(ambiguity_sources),
        "stable_offset_safe_setdown_flip_strata": sum(row["strict_intervention_flip"] for row in witnesses),
        "stable_offset_safe_setdown_flip_source_count": len(flip_sources),
        "learned_scores_or_posthoc_thresholds_used": False,
        "all_failed_and_null_configurations_retained": bool(
            len(attempts) == int(manifest.get("expected_linked_decisions", -1))
            and option_coverage_complete and validity_coverage_complete
        ),
        "option_outcome_coverage_complete": option_coverage_complete,
        "mechanical_validity_coverage_complete": validity_coverage_complete,
    }
    gate_pass = {
        "restoration": gate_values["exact_restoration_all_retained"],
        "strict_base": gate_values["strict_base_source_count"] >= int(gate_cfg["minimum_strict_base_sources"]),
        "strict_stable_offset_place": gate_values["strict_stable_offset_place_source_count"] >= int(gate_cfg["minimum_strict_stable_offset_place_sources"]),
        "strict_safe_setdown": gate_values["strict_safe_setdown_source_count"] >= int(gate_cfg["minimum_strict_safe_setdown_sources"]),
        "same_r_ambiguity": gate_values["same_r_different_strict_action_source_count"] >= int(gate_cfg["minimum_same_r_different_action_sources"]),
        "intervention_flip": gate_values["stable_offset_safe_setdown_flip_source_count"] >= int(gate_cfg["minimum_stable_offset_safe_setdown_flip_sources"]),
        "no_learned_or_posthoc_selection": not gate_values["learned_scores_or_posthoc_thresholds_used"],
        "all_attempts_retained": gate_values["all_failed_and_null_configurations_retained"],
    }

    engineering_only = bool(manifest.get("engineering_only", False))
    freeze_blocker = manifest.get("freeze_blocker")
    eligible_sources = sorted({row["source_state_sha256"] for row in decisions})
    if engineering_only or freeze_blocker:
        decision = "SCREEN_BLOCKED_ENVIRONMENT"
    elif not validity_by_decision or not decisions:
        decision = "SCREEN_FAIL_MECHANICAL_VALIDITY"
    elif not gate_pass["restoration"]:
        decision = "SCREEN_FAIL_RESTORATION"
    elif not (
        gate_pass["strict_base"] and gate_pass["strict_stable_offset_place"]
        and gate_pass["strict_safe_setdown"]
    ):
        decision = "SCREEN_FAIL_STRICT_SUPPORT"
    elif not gate_pass["same_r_ambiguity"]:
        decision = "SCREEN_FAIL_SAME_R_AMBIGUITY"
    elif not gate_pass["intervention_flip"]:
        decision = "SCREEN_FAIL_INTERVENTION_FLIP"
    elif all(gate_pass.values()):
        decision = "SCREEN_PASS"
    else:
        decision = "SCREEN_FAIL_MECHANICAL_VALIDITY"

    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "non_glass_unstable_placement_screen_decision",
        "family_id": FAMILY_ID,
        "protocol_sha256": manifest["protocol_sha256"],
        "decision": decision,
        "engineering_only": engineering_only,
        "freeze_blocker": freeze_blocker,
        "source_unit": "source_state_sha256",
        "counts": {
            "attempted_source_count": len(selected_sources),
            "eligible_source_count": len(eligible_sources),
            "attempted_linked_decision_count": len(attempts),
            "canonical_mechanical_validity_count": len(validity_by_decision),
            "eligible_canonical_decision_count": len(decisions),
            "ordinary_option_outcome_rows": len(outcome_rows),
            "invalidity_reason_counts": dict(sorted(invalidity_counts.items())),
            "incomplete_reason_counts": dict(sorted(incomplete_counts.items())),
        },
        "strict_support_sources": support_sources,
        "same_r_ambiguity_sources": ambiguity_sources,
        "intervention_flip_sources": flip_sources,
        "gate_values": gate_values,
        "gate_pass": gate_pass,
        "scope": "disposable development-only authoring screen; no benchmark or deployable-method authorization",
    }
    _write_json_exclusive(root / "screen_decision.json", result)

    report_lines = [
        "# Non-glass unstable final-placement screen",
        "",
        f"Decision: **{decision}**",
        "",
        "This is one disposable, model-free authoring screen. It does not authorize benchmark collection, router training, or a deployable-method claim.",
        "",
        "## Frozen mechanism",
        "",
        "A low-profile, low-friction rotated static support patch changes bowl stability only at the final plate region. The matched offpath condition moves the identical patch outside both the nominal swept volume and final support footprint; no_hazard removes it.",
        "",
        "## Accounting",
        "",
        f"- Attempted sources: {len(selected_sources)}",
        f"- Eligible sources: {len(eligible_sources)}",
        f"- Attempted linked decisions: {len(attempts)}",
        f"- Eligible canonical decisions: {len(decisions)}",
        f"- Ordinary option-outcome rows: {len(outcome_rows)} / 324",
        f"- Invalidity reasons: `{json.dumps(dict(sorted(invalidity_counts.items())), sort_keys=True)}`",
        f"- Incomplete/null reasons: `{json.dumps(dict(sorted(incomplete_counts.items())), sort_keys=True)}`",
        "",
        "## Frozen gate values",
        "",
        f"- Exact restoration: {audit_passed}/{audit_total} = {restoration_rate:.6f}",
        f"- Strict Base sources: {len(support_sources['base_continue'])}",
        f"- Strict stable_offset_place sources: {len(support_sources['stable_offset_place'])}",
        f"- Strict safe_setdown sources: {len(support_sources['safe_setdown'])}",
        f"- Same-R/different-strict-action strata: {len(witnesses)} across {len(ambiguity_sources)} sources",
        f"- stable_offset_place <-> safe_setdown flips: {sum(row['strict_intervention_flip'] for row in witnesses)} strata across {len(flip_sources)} sources",
        f"- Learned scores or post-hoc thresholds used: {gate_values['learned_scores_or_posthoc_thresholds_used']}",
        f"- All attempted configurations retained: {gate_values['all_failed_and_null_configurations_retained']}",
        "",
        "## Gate evaluation",
        "",
    ]
    report_lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in gate_pass.items())
    report_lines.extend([
        "",
        "All attempted, failed, tied, null, and de-duplicated no_hazard links remain in the machine-readable artifacts.",
    ])
    with (root / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="immutable screen result root")
    args = parser.parse_args()
    decision = analyze_screen(args.input)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
