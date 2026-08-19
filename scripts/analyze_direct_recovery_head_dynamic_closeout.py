#!/usr/bin/env python3
"""Analyze the immutable P3.2 three-method dynamic closeout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


METHODS = ("Base", "P2SequentialRouter", "DirectRecoveryRouter")
CONDITIONS = ("glass", "offpath", "noglass")
OUTCOMES = ("task_success", "catastrophe", "safe_noncompletion")
OPTIONS = ("Base", "Detour", "FailSafeHold")
PAIRED_METRICS = (*OUTCOMES, "intervention")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_method(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(rows)
    count = len(records)
    outcome_counts = Counter(str(row["outcome"]) for row in records)
    selected_counts = Counter(str(row["selected_option"]) for row in records)
    controls = [
        row for row in records
        if row["condition"] in {"offpath", "noglass"}
        and row["reference_base_outcome"] == "task_success"
    ]
    glass = [row for row in records if row["condition"] == "glass"]
    glass_base_catastrophe = [
        row for row in glass if row["reference_base_outcome"] == "catastrophe"
    ]
    known = [row for row in glass if bool(row["known_recovery_t20"])]
    return {
        "episodes": count,
        "outcomes": {
            outcome: {
                "count": int(outcome_counts[outcome]),
                "rate": float(outcome_counts[outcome] / count),
            }
            for outcome in OUTCOMES
        },
        "intervention": {
            "count": int(sum(bool(row["intervened"]) for row in records)),
            "rate": float(np.mean([bool(row["intervened"]) for row in records])),
        },
        "selected_option_counts": {
            option: int(selected_counts[option]) for option in OPTIONS
        },
        "base_success_controls": {
            "support": len(controls),
            "base_decision_retention_count": int(sum(
                row["selected_option"] == "Base" for row in controls
            )),
            "base_decision_retention_rate": float(np.mean([
                row["selected_option"] == "Base" for row in controls
            ])),
            "task_success_retention_count": int(sum(
                row["outcome"] == "task_success" for row in controls
            )),
            "task_success_retention_rate": float(np.mean([
                row["outcome"] == "task_success" for row in controls
            ])),
            "intervention_count": int(sum(bool(row["intervened"]) for row in controls)),
        },
        "glass": {
            "support": len(glass),
            "task_success_count": int(sum(
                row["outcome"] == "task_success" for row in glass
            )),
            "task_success_rate": float(np.mean([
                row["outcome"] == "task_success" for row in glass
            ])),
            "base_catastrophe_support": len(glass_base_catastrophe),
            "recovered_base_catastrophe_count": int(sum(
                row["outcome"] == "task_success" for row in glass_base_catastrophe
            )),
            "recovered_base_catastrophe_rate": float(np.mean([
                row["outcome"] == "task_success" for row in glass_base_catastrophe
            ])),
        },
        "known_recovery_t20": {
            "support": len(known),
            "missed_count": int(sum(
                bool(row["missed_known_recovery_opportunity"]) for row in known
            )),
            "missed_rate": (
                None
                if not known
                else float(np.mean([
                    bool(row["missed_known_recovery_opportunity"]) for row in known
                ]))
            ),
            "late_or_absent_count": int(sum(
                bool(row["late_or_absent_on_known_recovery"]) for row in known
            )),
        },
    }


def summarize_by_condition(
    rows: Sequence[Mapping[str, Any]], method: str, condition: str
) -> dict[str, Any]:
    subset = [
        row for row in rows
        if row["method"] == method and row["condition"] == condition
    ]
    outcomes = Counter(str(row["outcome"]) for row in subset)
    return {
        "episodes": len(subset),
        "task_success": int(outcomes["task_success"]),
        "catastrophe": int(outcomes["catastrophe"]),
        "safe_noncompletion": int(outcomes["safe_noncompletion"]),
        "intervention": int(sum(bool(row["intervened"]) for row in subset)),
        "selected_options": dict(sorted(Counter(
            str(row["selected_option"]) for row in subset
        ).items())),
    }


def _indicator(row: Mapping[str, Any], metric: str) -> float:
    if metric == "intervention":
        return float(bool(row["intervened"]))
    return float(str(row["outcome"]) == metric)


def paired_source_differences(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {
        (str(row["source_state_sha256"]), str(row["condition"]), str(row["method"])): row
        for row in rows
    }
    sources = sorted({str(row["source_state_sha256"]) for row in rows})
    comparisons = (
        ("Direct_minus_P2", "DirectRecoveryRouter", "P2SequentialRouter"),
        ("Direct_minus_Base", "DirectRecoveryRouter", "Base"),
        ("P2_minus_Base", "P2SequentialRouter", "Base"),
    )
    records = []
    aggregate: dict[str, Any] = {}
    for comparison, method_a, method_b in comparisons:
        comparison_rows = []
        for source in sources:
            row: dict[str, Any] = {
                "comparison": comparison,
                "method_a": method_a,
                "method_b": method_b,
                "source_state_sha256": source,
            }
            for metric in PAIRED_METRICS:
                differences = [
                    _indicator(by_key[(source, condition, method_a)], metric)
                    - _indicator(by_key[(source, condition, method_b)], metric)
                    for condition in CONDITIONS
                ]
                row[f"{metric}_count_difference"] = float(sum(differences))
                row[f"{metric}_rate_difference"] = float(np.mean(differences))
            comparison_rows.append(row)
            records.append(row)
        aggregate[comparison] = {
            "method_a": method_a,
            "method_b": method_b,
            "difference_definition": "method_a minus method_b",
            "metrics": {
                metric: {
                    "mean_source_paired_rate_difference": float(np.mean([
                        row[f"{metric}_rate_difference"] for row in comparison_rows
                    ])),
                    "sources_positive": int(sum(
                        row[f"{metric}_count_difference"] > 0
                        for row in comparison_rows
                    )),
                    "sources_tied": int(sum(
                        row[f"{metric}_count_difference"] == 0
                        for row in comparison_rows
                    )),
                    "sources_negative": int(sum(
                        row[f"{metric}_count_difference"] < 0
                        for row in comparison_rows
                    )),
                }
                for metric in PAIRED_METRICS
            },
        }
    return records, aggregate


def _write_paired_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "comparison", "method_a", "method_b", "source_state_sha256",
        *[
            f"{metric}_{suffix}"
            for metric in PAIRED_METRICS
            for suffix in ("count_difference", "rate_difference")
        ],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _fmt_diff(value: float) -> str:
    return f"{100.0 * float(value):+.1f} pp"


def _write_report(result: Mapping[str, Any], path: Path) -> None:
    summaries = result["methods"]
    manifest = result["capture_manifest"]
    lines = [
        "# P3.2 Frozen Dynamic Closeout",
        "",
        "Status: **single frozen closeout complete; experiment stopped without head or "
        "boundary retuning**.",
        "",
        f"Cohort: **{result['counts']['sources']} fresh source states × 3 conditions = "
        f"{result['counts']['source_condition_units']} paired source-condition units**. "
        "The three reported methods reuse each unit's identical Base scan and exact "
        "serialized prefix.",
        "",
        "## Frozen protocol",
        "",
        f"- Direct input/head: 10-D prediction-output feature, fixed standardization and "
        f"linear weights (`{manifest['direct_head_model_sha256'][:12]}...`).",
        f"- Direct sequential boundary: `{manifest['direct_sequential_boundary']['margin']:.10f}` "
        "from the original 10 source-disjoint calibration sources, fixed `alpha=0.1`, "
        "strict `>` crossing.",
        f"- Old P2 boundary: `{manifest['old_sequential_boundary']['effective_margin']:.10f}` "
        "at the already-frozen `alpha=0.1` point.",
        "- Runtime: Base before crossing; first crossing selects the larger Detour/Hold "
        "logit and latches. No probability threshold, class margin, or post-run parameter "
        "exists.",
        "- Eligibility was decided before either Router ran: Base catastrophizes under "
        "glass, succeeds under off-path/no-glass, and exposes an exact T-20 glass branch.",
        "",
        "## Main results",
        "",
        "| Method | Task success | Catastrophe | Safe noncompletion | Intervention |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        summary = summaries[method]
        lines.append(
            f"| {method} | {_fmt_rate(summary['outcomes']['task_success']['rate'])} "
            f"({summary['outcomes']['task_success']['count']}/24) | "
            f"{_fmt_rate(summary['outcomes']['catastrophe']['rate'])} "
            f"({summary['outcomes']['catastrophe']['count']}/24) | "
            f"{_fmt_rate(summary['outcomes']['safe_noncompletion']['rate'])} "
            f"({summary['outcomes']['safe_noncompletion']['count']}/24) | "
            f"{_fmt_rate(summary['intervention']['rate'])} "
            f"({summary['intervention']['count']}/24) |"
        )
    lines.extend([
        "",
        "## Controls, glass recovery, and selected options",
        "",
        "| Method | Base-success control decision retention | Control task-success retention | Glass recovery | Missed known T-20 recovery | Selected Base / Detour / Hold |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        summary = summaries[method]
        control = summary["base_success_controls"]
        glass = summary["glass"]
        known = summary["known_recovery_t20"]
        selected = summary["selected_option_counts"]
        lines.append(
            f"| {method} | {_fmt_rate(control['base_decision_retention_rate'])} "
            f"({control['base_decision_retention_count']}/{control['support']}) | "
            f"{_fmt_rate(control['task_success_retention_rate'])} "
            f"({control['task_success_retention_count']}/{control['support']}) | "
            f"{_fmt_rate(glass['recovered_base_catastrophe_rate'])} "
            f"({glass['recovered_base_catastrophe_count']}/{glass['base_catastrophe_support']}) | "
            f"{known['missed_count']}/{known['support']} | "
            f"{selected['Base']} / {selected['Detour']} / "
            f"{selected['FailSafeHold']} |"
        )

    paired = result["source_level_paired_differences"]
    lines.extend([
        "",
        "## Source-level paired differences",
        "",
        "Differences below are method A minus method B, averaged after computing each "
        "source's rate across its three matched conditions.",
        "",
        "| Comparison | Task success | Catastrophe | Safe noncompletion | Intervention |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in ("Direct_minus_P2", "Direct_minus_Base", "P2_minus_Base"):
        item = paired[name]
        lines.append(
            f"| {name} | "
            f"{_fmt_diff(item['metrics']['task_success']['mean_source_paired_rate_difference'])} | "
            f"{_fmt_diff(item['metrics']['catastrophe']['mean_source_paired_rate_difference'])} | "
            f"{_fmt_diff(item['metrics']['safe_noncompletion']['mean_source_paired_rate_difference'])} | "
            f"{_fmt_diff(item['metrics']['intervention']['mean_source_paired_rate_difference'])} |"
        )
    direct_p2_rows = [
        row for row in result["paired_source_rows"]
        if row["comparison"] == "Direct_minus_P2"
    ]
    lines.extend([
        "",
        "### Direct minus P2 by held-out source",
        "",
        "| Source | Delta success count | Delta catastrophe count | Delta safe-noncompletion count | Delta intervention count |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in direct_p2_rows:
        lines.append(
            f"| `{row['source_state_sha256'][:12]}...` | "
            f"{row['task_success_count_difference']:+.0f} | "
            f"{row['catastrophe_count_difference']:+.0f} | "
            f"{row['safe_noncompletion_count_difference']:+.0f} | "
            f"{row['intervention_count_difference']:+.0f} |"
        )

    lines.extend([
        "",
        "## Condition-level counts",
        "",
        "| Method | Condition | Success | Catastrophe | Safe noncompletion | Intervention |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        for condition in CONDITIONS:
            row = result["by_condition"][method][condition]
            lines.append(
                f"| {method} | {condition} | {row['task_success']}/8 | "
                f"{row['catastrophe']}/8 | {row['safe_noncompletion']}/8 | "
                f"{row['intervention']}/8 |"
            )

    lines.extend([
        "",
        "## Stop rule",
        "",
        "This is the only P3.2 fresh cohort. Its outcomes were analyzed under the "
        "pre-frozen head, option mapping, `alpha=0.1` boundary, and latch. **No head "
        "refit, boundary recalibration, alpha sweep, threshold change, or follow-up "
        "cohort is authorized from these results.**",
        "",
        "Known-recovery opportunity means a successful structured Detour from the exact "
        "matched T-20 glass state. It is a diagnostic label, not a fourth compared method.",
        "",
        "Raw paired rows and complete provenance are retained in `paired_source_differences.csv` "
        "and `analysis.json`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(capture: Path) -> dict[str, Any]:
    root = capture.resolve()
    manifest_path = root / "capture_manifest.json"
    episodes_path = root / "method_episodes.jsonl"
    diagnostics_path = root / "known_recovery_t20_diagnostics.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(episodes_path)
    diagnostics = _read_jsonl(diagnostics_path)

    sources = sorted({str(row["source_state_sha256"]) for row in rows})
    keys = Counter(
        (str(row["source_state_sha256"]), str(row["condition"]), str(row["method"]))
        for row in rows
    )
    expected = {
        (source, condition, method)
        for source in sources for condition in CONDITIONS for method in METHODS
    }
    if set(keys) != expected or any(count != 1 for count in keys.values()):
        raise ValueError("closeout rows are not a complete 8 x 3 x 3 matched table")
    if len(sources) != 8 or len(rows) != 72 or len(diagnostics) != 8:
        raise ValueError("P3.2 closeout cohort size differs from the frozen contract")
    if manifest["status"] != (
        "single frozen dynamic closeout complete; no retuning permitted"
    ):
        raise ValueError("capture is not marked as the stopped frozen closeout")

    method_summaries = {
        method: summarize_method([row for row in rows if row["method"] == method])
        for method in METHODS
    }
    by_condition = {
        method: {
            condition: summarize_by_condition(rows, method, condition)
            for condition in CONDITIONS
        }
        for method in METHODS
    }
    paired_rows, paired = paired_source_differences(rows)
    paired_path = root / "paired_source_differences.csv"
    _write_paired_csv(paired_path, paired_rows)

    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "p3_2_frozen_dynamic_closeout_analysis",
        "status": "complete and stopped; no post-closeout tuning permitted",
        "capture": str(root),
        "capture_manifest": manifest,
        "counts": {
            "sources": len(sources),
            "conditions": len(CONDITIONS),
            "source_condition_units": len(sources) * len(CONDITIONS),
            "methods": len(METHODS),
            "method_episode_rows": len(rows),
            "known_recovery_diagnostics": len(diagnostics),
        },
        "methods": method_summaries,
        "by_condition": by_condition,
        "source_level_paired_differences": paired,
        "paired_source_rows": paired_rows,
        "no_retuning_audit": {
            "head_refit_after_closeout": False,
            "boundary_recalibrated_after_closeout": False,
            "alpha_sweep": False,
            "threshold_sweep": False,
            "additional_cohort": False,
        },
        "input_artifacts": {
            "capture_manifest": _artifact(manifest_path),
            "method_episodes": _artifact(episodes_path),
            "known_recovery_diagnostics": _artifact(diagnostics_path),
            "router_trace": _artifact(root / "router_trace.jsonl"),
        },
        "outputs": {
            "paired_source_differences": _artifact(paired_path),
            "report": {"path": str(root / "REPORT.md")},
            "analysis": {"path": str(root / "analysis.json")},
        },
    }
    report_path = root / "REPORT.md"
    _write_report(result, report_path)
    result["outputs"]["report"] = _artifact(report_path)
    analysis_path = root / "analysis.json"
    analysis_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "capture": str(root),
        "sources": len(sources),
        "methods": method_summaries,
        "direct_minus_p2": paired["Direct_minus_P2"],
        "status": result["status"],
    }, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.capture)


if __name__ == "__main__":
    main()
