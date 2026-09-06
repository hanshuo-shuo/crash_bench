#!/usr/bin/env python3
"""Bounded development experiment for the 2026-09-06 audit repair.

Same pooled features, optimizer, 100 epochs and five seeds per architecture.
Never refit on development or read D8. No automatic test authorization.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.models.selective import PairwiseSourceConformalSelector
from scripts.expansion.calibrate_selector import MECHANISM_ID, sha256_file
from scripts.expansion.hash_tree_manifest import resolve_git_head
from scripts.expansion.train_option_value import OPTION_IDS, read_jsonl

ARCHITECTURES = ("additive", "interaction", "per_option")
GAIN_THRESHOLD = 0.05  # Declared before this run; not selected on development.


def prediction_choices(row_ids, options, utility_seeds, catastrophe_seeds, calibration):
    utility_seeds = np.asarray(utility_seeds)
    catastrophe_seeds = np.asarray(catastrophe_seeds)
    if (utility_seeds.ndim != 2 or utility_seeds.shape[0] < 2
            or utility_seeds.shape != catastrophe_seeds.shape
            or utility_seeds.shape[1] != len(row_ids)
            or len(row_ids) != len(options)):
        raise ValueError("unaligned ensemble prediction arrays")
    if not np.all(np.isfinite(utility_seeds)) or not np.all(
            np.isfinite(catastrophe_seeds) & (catastrophe_seeds >= 0) & (catastrophe_seeds <= 1)):
        raise ValueError("invalid ensemble predictions")
    selector = PairwiseSourceConformalSelector(
        OPTION_IDS, utility_quantiles=calibration["utility_pairwise_quantiles"],
        catastrophe_difference_quantiles=calibration["catastrophe_difference_quantiles"],
        catastrophe_absolute_quantiles=calibration["catastrophe_absolute_quantiles"],
    )
    mean_u, mean_cat = utility_seeds.mean(0), catastrophe_seeds.mean(0)
    groups = defaultdict(dict)
    for index, (row_id, option) in enumerate(zip(row_ids, options)):
        block = str(row_id).rsplit(":", 1)[0]
        if option in groups[block]:
            raise ValueError("duplicate block/option prediction")
        groups[block][str(option)] = index
    choices = {mode: {} for mode in ("point", "benefit_gate", "conformal")}
    for block, indices in groups.items():
        if set(indices) != set(OPTION_IDS):
            raise ValueError("incomplete option predictions")
        u = {option: float(mean_u[indices[option]]) for option in OPTION_IDS}
        cat = {option: float(mean_cat[indices[option]]) for option in OPTION_IDS}
        # Catalog order breaks ties toward Base, then Refresh, then Stop.
        winner = max(OPTION_IDS, key=u.get)
        choices["point"][block] = winner
        choices["benefit_gate"][block] = (
            winner if u[winner] - u["base_continue"] > GAIN_THRESHOLD else "base_continue")
        scales = {(a, b): max(float(np.std(
            utility_seeds[:, indices[a]] - utility_seeds[:, indices[b]], ddof=1)), 0.05)
            for a in OPTION_IDS for b in OPTION_IDS if a != b}
        choices["conformal"][block] = selector.select(
            mechanism_id=MECHANISM_ID, predicted_utility=u,
            predicted_catastrophe=cat, pairwise_scale=scales).option_id
    return choices, selector.certificate_diagnostic(MECHANISM_ID)


def policy_metrics(branches, choices):
    groups = defaultdict(dict)
    for row in branches:
        block, option = str(row["block_id"]), str(row["option_id"])
        if option in groups[block]:
            raise ValueError("duplicate branch")
        groups[block][option] = row
    if set(choices) != set(groups) or not choices:
        raise ValueError("choices must cover every eligible decision exactly")
    per_source = defaultdict(lambda: defaultdict(list))
    b0 = b1 = beneficial = needless = 0
    for block, options in groups.items():
        if set(options) != set(OPTION_IDS):
            raise ValueError("incomplete branch block")
        selected = options[choices[block]]
        base_u = float(options["base_continue"]["u0"])
        benefit = max(float(row["u0"]) for row in options.values()) > base_u
        b1 += benefit
        b0 += not benefit
        beneficial += benefit and float(selected["u0"]) > base_u
        needless += not benefit and choices[block] != "base_continue"
        source = str(selected["physical_source_id"])
        per_source[source]["u0"].append(float(selected["u0"]))
        for key in ("task_success", "catastrophe"):
            per_source[source][key].append(float(selected["outcome"][key]))
    source_means = {source: {key: float(np.mean(values)) for key, values in metrics.items()}
                    for source, metrics in per_source.items()}
    return {
        "source_count": len(per_source), "decision_count": len(choices),
        **{key: float(np.mean([row[key] for row in source_means.values()]))
           for key in ("u0", "task_success", "catastrophe")},
        "intervention_rate": sum(o != "base_continue" for o in choices.values()) / len(choices),
        "B1_beneficial_intervention_recall": beneficial / b1 if b1 else None,
        "B0_unnecessary_intervention_rate": needless / b0 if b0 else None,
        "B1_decisions": b1, "B0_decisions": b0,
        "option_counts": dict(Counter(choices.values())), "per_source": source_means,
    }


def paired_interval(a, b, key="u0"):
    if set(a["per_source"]) != set(b["per_source"]):
        raise ValueError("paired comparison source identity mismatch")
    sources = sorted(a["per_source"])
    delta = np.array([a["per_source"][s][key] - b["per_source"][s][key] for s in sources])
    rng = np.random.default_rng(20260906)
    means = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(1)
    return {"delta": float(delta.mean()), "source_paired_bootstrap_95pct": np.quantile(means, [0.025, 0.975]).tolist(),
            "source_count": len(delta), "selection_adjusted": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, default=ROOT / "configs/expansion/utility_v1.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite repair experiment: {args.output_dir}")
    anchors = read_jsonl(args.merged_dir / "anchors.jsonl")
    branches = read_jsonl(args.merged_dir / "branches.jsonl")
    if {r["split_role"] for r in anchors + branches} - {"train", "development", "calibration"}:
        raise ValueError("repair experiment accepts D5 roles only")
    dev = [row for row in branches if row["split_role"] == "development"]
    args.output_dir.mkdir(parents=True)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
    def run(script, *extra):
        subprocess.run([sys.executable, str(ROOT / "scripts/expansion" / script), *map(str, extra)], cwd=ROOT, env=env, check=True)
    common = ["--merged-dir", args.merged_dir, "--artifact-store", args.artifact_store,
              "--feature-cache", args.feature_cache, "--utility-config", args.utility_config]
    baseline_dir = args.output_dir / "baselines"
    run("train_baselines.py", *common, "--output-dir", baseline_dir)
    baseline = json.loads((baseline_dir / "baseline_selection.json").read_text())
    report = {
        "schema_version": 1, "kind": "audit_repair_round1_development",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": resolve_git_head(ROOT), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "fit_on": "train", "evaluation_role": "development_held_out_from_fit",
        "feature_dim": 104, "epochs": args.epochs, "seeds": list(range(5)),
        "benefit_threshold": GAIN_THRESHOLD, "test_rows_read": 0,
        "confirmatory": False, "test_authorized": False,
        "input_sha256": {name: sha256_file(args.merged_dir / name)
                         for name in ("anchors.jsonl", "branches.jsonl", "dataset_manifest.json")},
        "feature_cache_sha256": sha256_file(args.feature_cache),
        "utility_config_sha256": sha256_file(args.utility_config),
        "baselines": {key: policy_metrics(dev, value) for key, value in baseline["development_choices"].items()},
        "architectures": {},
    }
    for architecture in ARCHITECTURES:
        out = args.output_dir / architecture
        seed_dirs = []
        for seed in range(5):
            directory = out / f"seed_{seed}"
            run("train_option_value.py", *common, "--seed", seed, "--epochs", args.epochs,
                "--architecture", architecture, "--fit-on", "train", "--output-dir", directory)
            seed_dirs.extend(["--seed-dir", directory])
        selection = out / "development_selection.json"
        run("analyze_development_models.py", "--merged-dir", args.merged_dir,
            "--baseline-dir", baseline_dir, *seed_dirs, "--output", selection)
        cal_dir = out / "calibration"
        run("calibrate_selector.py", *common, *seed_dirs, "--fit-on", "train",
            "--development-selection", selection, "--baseline-selection", baseline_dir / "baseline_selection.json",
            "--output-dir", cal_dir)
        freeze = cal_dir / "statewise_calibration_freeze.json"
        calibration = json.loads(freeze.read_text())
        rows = [[r for r in read_jsonl(out / f"seed_{s}" / "predictions.jsonl") if r["role"] == "development"] for s in range(5)]
        identities = [(r["row_id"], r["option_id"], r["physical_source_id"]) for r in rows[0]]
        if any([(r["row_id"], r["option_id"], r["physical_source_id"]) for r in rs] != identities for rs in rows):
            raise ValueError("seed alignment drift")
        choices, diagnostic = prediction_choices(
            [r["row_id"] for r in rows[0]], [r["option_id"] for r in rows[0]],
            [[r["predicted_u0"] for r in rs] for rs in rows],
            [[r["outcome_probabilities"][1] for r in rs] for rs in rows], calibration)
        metrics = {mode: policy_metrics(dev, choice) for mode, choice in choices.items()}
        report["architectures"][architecture] = {"modes": metrics, "certificate_diagnostic": diagnostic,
            "per_seed": json.loads(selection.read_text())["per_seed"], "choices": choices,
            "paired_vs_DirectQ": {mode: {key: paired_interval(value, report["baselines"]["DirectQ"], key)
                for key in ("u0", "task_success", "catastrophe")} for mode, value in metrics.items()}}
        run("analyze_scoped_gate_b.py", *common, *seed_dirs, "--calibration-freeze", freeze,
            "--development-selection", selection, "--baseline-selection", baseline_dir / "baseline_selection.json",
            "--output", out / "development_gate_diagnostic.json")
    additive = report["architectures"]["additive"]["modes"]["point"]
    for value in report["architectures"].values():
        value["point_paired_vs_additive"] = paired_interval(value["modes"]["point"], additive)
    report["status"] = "COMPLETE_DEVELOPMENT_DIAGNOSTIC"
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output_dir), "status": report["status"]}))


if __name__ == "__main__":
    main()
