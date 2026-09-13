#!/usr/bin/env python3
"""Descriptive follow-up on frozen candidate outcomes; no fitting or new rollouts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "docs/audits/20260908/candidate_refresh/verified_evidence"


def completion_cost(row):
    # Failure/accident is not a fast completion. This is a newly introduced
    # descriptive 200-step capped completion measure, not a frozen primary metric.
    return row["first_success_step"] if row["task_success"] else 200


def summarize(rows, anchors):
    groups = defaultdict(list)
    for row in rows:
        groups[anchors[row["panel_id"]]["physical_source_id"]].append(row)
    return {
        "n_executions": len(rows), "n_sources": len(groups),
        "successes_200": sum(r["task_success"] for r in rows),
        "accidents": sum(r["catastrophe"] for r in rows),
        "configuration_weighted_capped_completion_200": mean(completion_cost(r) for r in rows),
        "source_macro_capped_completion_200": mean(mean(completion_cost(r) for r in group) for group in groups.values()),
        "configuration_weighted_inference_calls": mean(r["inference_calls"] for r in rows),
        "source_macro_inference_calls": mean(mean(r["inference_calls"] for r in group) for group in groups.values()),
        "instrumented_wall_seconds": mean(r["elapsed_seconds"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)
    outputs = [out / name for name in ("evidence.json", "deadline_curve.png", "deadline_curve.svg")]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Use a new output directory to preserve reviewed evidence")
    hashes = {}
    def load(name):
        raw = (SOURCE / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    anchors = {a["panel_id"]: a for a in load("anchors.json")}
    records = {p: load(p + ".json") for p in ("A", "B")}
    freeze = load("freeze.json")
    assert freeze["A_sha256"] == hashes["A.json"]
    assert freeze["anchors_sha256"] == hashes["anchors.json"]
    assert (SOURCE / "freeze.sha256").read_text().strip() == hashes["freeze.json"]
    for phase, rows in records.items():
        repeats = range(4) if phase == "A" else range(4, 8)
        expected = {(a, r, o) for a in anchors for r in repeats for o in (0, 1)}
        keys = [(r["panel_id"], r["repeat"], r["option"]) for r in rows]
        assert len(keys) == len(set(keys)) and set(keys) == expected
        assert all(r["phase"] == phase for r in rows)
    summaries, per_anchor, curves = {}, [], {}
    for phase, rows in records.items():
        summaries[phase] = {}
        for condition in ("stale", "matched_buffer_control"):
            summaries[phase][condition] = {}
            for option, label in ((0, "Base"), (1, "AlwaysRefresh")):
                group = [r for r in rows if r["option"] == option and anchors[r["panel_id"]]["condition"] == condition]
                summaries[phase][condition][label] = summarize(group, anchors)
                if phase == "B" and condition == "stale":
                    curves[label] = [sum(bool(r["task_success"]) and r["first_success_step"] <= h for r in group)
                                     for h in range(1, 201)]
    for panel, anchor in anchors.items():
        if anchor["condition"] != "stale":
            continue
        item = {k: anchor[k] for k in ("panel_id", "physical_source_id", "task_id", "age_steps", "anchor_steps")}
        item["phases"] = {}
        for phase, rows in records.items():
            item["phases"][phase] = {str(o): summarize([r for r in rows if r["panel_id"] == panel and r["option"] == o], anchors)
                                      for o in (0, 1)}
        per_anchor.append(item)
    output = {
        "kind": "posthoc_descriptive_candidate_refresh_assessment", "input_sha256": hashes,
        "summaries": summaries, "stale_anchors": per_anchor,
        "B_stale_deadline_success_counts": curves,
        "age_source_support": {str(age): sorted({a["physical_source_id"] for a in anchors.values()
                                                 if a["condition"] == "stale" and a["age_steps"] == age})
                               for age in sorted({a["age_steps"] for a in anchors.values() if a["condition"] == "stale"})},
        "limits": ["Historical-success-selected training mechanism panel; 8 independent sources.",
                   "Capped completion metric and full deadline curve are retrospective diagnostics.",
                   "No model fitting, threshold selection, confidence claim or new experiment.",
                   "Instrumented elapsed_seconds includes simulation and trace writing; not pure inference latency.",
                   "Fixed-continuation repeats do not establish cross-source or new-anchor generalization.",
                   "Raw remote trace hashes were not rechecked in this local follow-up."],
    }
    assert curves["Base"][99] == 23 and curves["AlwaysRefresh"][99] == 26
    assert curves["Base"][199] == 33 and curves["AlwaysRefresh"][199] == 31
    outputs[0].write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), layout="constrained", sharey=True)
    for ax in axes:
        for label, color in (("Base", "#245970"), ("AlwaysRefresh", "#d56a32")):
            ax.step(range(1, 201), [n / 36 * 100 for n in curves[label]], where="post", label=label, color=color, linewidth=2)
        ax.axvline(100, color="#858585", linestyle="--", linewidth=1)
        ax.set_xlabel("Post-anchor action deadline")
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e3e6e8", linewidth=.7)
    axes[0].set_xlim(1, 200)
    axes[0].set_ylabel("Task success by deadline (%)")
    axes[0].set_title("Full observed horizon")
    axes[0].legend(loc="lower right", frameon=False)
    axes[1].set_xlim(90, 110)
    axes[1].set_xticks([90, 95, 100, 102, 105, 110])
    axes[1].set_title("Near the original 100-step cutoff")
    fig.suptitle("Refresh advantage changes with the evaluation deadline", fontsize=14)
    fig.supxlabel("B phase; 9 stale configurations / 8 sources; 36 executions per option. Descriptive only.", fontsize=9)
    fig.savefig(outputs[1], dpi=170)
    fig.savefig(outputs[2])
    plt.close(fig)
    print(json.dumps(summaries["B"]["stale"], indent=2))


if __name__ == "__main__":
    main()
