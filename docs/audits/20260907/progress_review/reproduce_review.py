#!/usr/bin/env python3
"""Read-only progress-review calculations; no fitting, simulation or D8 outcomes.

Run with the existing research Python environment and redirect stdout to a NEW
file. Frozen inputs are never modified. The historical configuration overlap is
retrospective description, not an amended retest panel or confirmatory analysis.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.expansion.analyze_selection_retest import evaluate
from scripts.expansion.selection_retest import freeze, validate_records

RETEST = ROOT / "docs/audits/20260907/selection_retest"
D5 = ROOT / "results/expansion/d5_statewise/e2a5264a802a_fe660c168e72_20260830T105726Z"
INPUTS: dict[str, str] = {}


def read(path: Path):
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    refs = ["main", "origin/codex/iclr27-exact-state-intervention-routing",
            "origin/codex/odur-negative", "HEAD"]
    branches = [{"ref": ref, "commit": git("rev-parse", ref),
                 "last_commit": git("log", "-1", "--date=iso-strict",
                                    "--format=%ad %s", ref)} for ref in refs]
    ancestry = [{"from": a, "to": b,
                 "is_ancestor": subprocess.run(
                     ["git", "merge-base", "--is-ancestor", a, b], cwd=ROOT).returncode == 0,
                 "additional_commits": int(git("rev-list", "--count", f"{a}..{b}"))}
                for a, b in zip(refs, refs[1:])]
    evidence = RETEST / "evidence"
    manifest = read(evidence / "local_evidence_sha256.json")
    for rel, expected in manifest.items():
        assert hashlib.sha256((RETEST / rel).read_bytes()).hexdigest() == expected, rel
    anchors = read(evidence / "anchors.json")
    records_a = read(evidence / "A.json")
    records_b = read(evidence / "B.json")
    selection = read(evidence / "freeze.json")
    for phase, records in [("A", records_a), ("B", records_b)]:
        validate_records(anchors, records, phase)
    assert (evidence / "freeze.sha256").read_text().strip() == hashlib.sha256(
        (evidence / "freeze.json").read_bytes()).hexdigest()
    assert selection["A_sha256"] == INPUTS[str((evidence / "A.json").relative_to(ROOT))]
    assert selection["anchors_sha256"] == INPUTS[str((evidence / "anchors.json").relative_to(ROOT))]
    assert all(freeze(anchors, records_a)[k] == v for k, v in selection.items()
               if k not in {"A_sha256", "anchors_sha256"})
    result = evaluate(anchors, records_b, selection)
    result["cross_job_pair_diagnostic"] = evaluate(
        anchors, records_b, selection, excluded_pairs=[("b17", 5)])
    recorded_metrics = read(RETEST / "metrics.json")
    # JSON normalization also makes tuple/list representation immaterial.
    assert json.loads(json.dumps(result)) == recorded_metrics
    anchor_by_id = {a["panel_id"]: a for a in anchors}
    groups = []
    all_records = records_a + records_b
    for phase in ("A", "B"):
        for task in sorted({a["task_id"] for a in anchors}):
            for condition in ("stale", "matched_buffer_control"):
                for option in (0, 1):
                    rows = [r for r in all_records if r["phase"] == phase and r["option"] == option
                            and anchor_by_id[r["panel_id"]]["task_id"] == task
                            and anchor_by_id[r["panel_id"]]["condition"] == condition]
                    groups.append({"phase": phase, "task": task, "condition": condition,
                                   "option": option, "n": len(rows),
                                   "successes": sum(r["task_success"] for r in rows),
                                   "accidents": sum(r["catastrophe"] for r in rows),
                                   "mean_steps": sum(r["steps"] for r in rows) / len(rows)})
    variation = []
    for anchor in anchors:
        for option in (0, 1):
            rows = [r for r in all_records if r["panel_id"] == anchor["panel_id"] and r["option"] == option]
            counts = Counter((r["task_success"], r["catastrophe"]) for r in rows)
            if len(counts) > 1:
                variation.append({"panel_id": anchor["panel_id"], "option": option,
                                  "terminal_counts": [{"success": k[0], "accident": k[1], "n": v}
                                                      for k, v in sorted(counts.items())]})
    config_keys = {(a["physical_source_id"], a["anchor_steps"], f"delay_{a['delay_steps']}",
                    a["condition"]): a["panel_id"] for a in anchors}
    selected_sources = {a["physical_source_id"] for a in anchors}
    counts = Counter()
    matches, historical_rescues = [], []
    condition_counts = Counter()
    for path in sorted((D5 / "shards").glob("source_*.json")):
        shard = read(path)
        assert shard["test_rows_read"] == 0
        if shard["split_role"] != "train":
            continue
        for block in shard["blocks"]:
            if not block.get("option_outcomes"):
                continue
            outcomes = {r["option_id"]: r["outcome"] for r in block["option_outcomes"]}
            base, refresh = outcomes["base_continue"], outcomes["observation_refresh"]
            rescue = int(not base["task_success"] and bool(refresh["task_success"]))
            key = (shard["physical_source_id"], block["anchor_steps"], block["severity_id"], block["condition"])
            counts["all_train_decisions"] += 1
            counts["all_train_success_conversions"] += rescue
            condition_counts[block["condition"]] += rescue
            if shard["physical_source_id"] in selected_sources:
                counts["selected_source_historical_decisions"] += 1
                counts["selected_source_success_conversions"] += rescue
                if rescue:
                    historical_rescues.append({"source": shard["physical_source_id"],
                                               "anchor_steps": block["anchor_steps"],
                                               "severity": block["severity_id"],
                                               "condition": block["condition"],
                                               "matched_panel_id": config_keys.get(key)})
            if key in config_keys:
                counts["configuration_matches"] += 1
                counts["configuration_match_success_conversions"] += rescue
                matches.append({"panel_id": config_keys[key], "old_bundle_id": block["bundle_id"],
                                "old_base": {k: base[k] for k in ("task_success", "catastrophe")},
                                "old_refresh": {k: refresh[k] for k in ("task_success", "catastrophe")}})
    assert counts["configuration_matches"] == len(anchors) == len({r["panel_id"] for r in matches})
    output = {"kind": "retrospective_project_progress_review", "date": "2026-09-07",
              "branches": branches, "ancestry": ancestry, "reachable_commits": int(git("rev-list", "--all", "--count")),
              "tracked_results_diff_since_odur_negative": git("diff", "--name-only", refs[2], "HEAD", "--", "results/"),
              "retest_evidence_hashes_verified": len(manifest),
              "retest_A_selection_recomputed_equal": True, "retest_metrics_recomputed_equal": True,
              "retest_group_descriptives": groups, "retest_terminal_variation": variation,
              "historical_train_overlap": {"counts": dict(counts),
                  "all_train_success_conversions_by_condition": dict(condition_counts),
                  "selected_source_historical_success_conversions": historical_rescues,
                  "matched_configurations": sorted(matches, key=lambda r: r["panel_id"])},
              "limits": ["Local Git refs only; no remote fetch or Quest SSH in this review.",
                         "No fitting, simulation, new confirmatory test or raw D8 outcomes.",
                         "Historical configuration matches are not exact bundle replays.",
                         "Overlap analysis was designed after seeing results; descriptive only.",
                         "Recorded conversions are not independently certified stable causal rescues.",
                         "Repeated runs do not increase the number of independent sources."],
              "input_sha256": INPUTS}
    print(json.dumps(output, indent=2, ensure_ascii=False) + "\n", end="")


if __name__ == "__main__":
    main()
