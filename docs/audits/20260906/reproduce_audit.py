#!/usr/bin/env python3
"""Read-only audit of three frozen CrashBench refs; never rewrites experiments.

Usage: python reproduce_audit.py --odur-root PATH [--output PATH]
Requires numpy. The ODUR tree must be c3378366d53cf514914ae837292126dbd806c4b2.
All alternative counts are retrospective diagnostics, not a replacement test.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import math
import pathlib
import sys

import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_support(anchors, branches):
    by_block = collections.defaultdict(dict)
    for row in branches:
        by_block[row["block_id"]][row["option_id"]] = row
    by_source = collections.defaultdict(list)
    by_condition = collections.defaultdict(list)
    winners = collections.Counter()
    decisions = []
    for anchor in anchors:
        options = by_block[anchor["block_id"]]
        assert len(options) == 3
        utility = {option: float(row["u0"]) for option, row in options.items()}
        benefit = max(v for k, v in utility.items() if k != "base_continue") > utility["base_continue"]
        winner = max(utility, key=utility.get)
        strict = utility[winner] - sorted(utility.values(), reverse=True)[1] >= 0.1
        row = dict(block_id=anchor["block_id"], benefit=benefit, winner=winner,
                   strict_winner=winner if strict else None, utility=utility)
        by_source[anchor["physical_source_id"]].append(row)
        by_condition[anchor["condition"]].append(row)
        winners[winner if strict else "tie_or_gap_below_0_10"] += 1
        decisions.append(row)
    source_counts = {
        "total": len(by_source),
        "contains_B0": sum(any(not row["benefit"] for row in rs) for rs in by_source.values()),
        "contains_B1": sum(any(row["benefit"] for row in rs) for rs in by_source.values()),
        "contains_both": sum(any(row["benefit"] for row in rs) and any(not row["benefit"] for row in rs) for rs in by_source.values()),
        "entirely_B0_original_gate": sum(all(not row["benefit"] for row in rs) for rs in by_source.values()),
        "contains_strict_Base": sum(any(row["strict_winner"] == "base_continue" for row in rs) for rs in by_source.values()),
    }
    n = len(decisions)
    assert by_source and all(len(rs) == 27 for rs in by_source.values()), "audit expects the frozen 27-anchor source grid"
    result = {
        "source_counts": source_counts,
        "decisions": n,
        "B0_decisions": sum(not row["benefit"] for row in decisions),
        "B1_decisions": sum(row["benefit"] for row in decisions),
        "strict_winners": dict(winners),
        "oracle_beneficial_intervention_rate": sum(row["benefit"] for row in decisions) / n,
        "maximum_coverage_given_strict_Base_recall_0_8": 1 - math.ceil(0.8 * winners["base_continue"] - 1e-10) / n,
        "conditions": {condition: {"n": len(rs), "B0": sum(not row["benefit"] for row in rs),
                                   "B1": sum(row["benefit"] for row in rs)} for condition, rs in by_condition.items()},
    }
    advantages = sorted((max(v for k, v in row["utility"].items() if k != "base_continue")
                         - row["utility"]["base_continue"] for row in decisions), reverse=True)
    required_interventions = math.ceil(0.40 * n)
    beneficial = sum(value > 0 for value in advantages)
    base_value = sum(row["utility"]["base_continue"] for row in decisions) / n
    oracle_value = base_value + sum(max(0, value) for value in advantages) / n
    constrained_value = base_value + sum(advantages[:max(beneficial, required_interventions)]) / n
    result["coverage_gate_diagnostic"] = {
        "minimum_required_interventions": required_interventions,
        "minimum_required_nonbeneficial_interventions": max(0, required_interventions - beneficial),
        "oracle_passes_minimum_coverage": beneficial / n >= 0.40,
        "unconstrained_oracle_u0": oracle_value,
        "best_hindsight_u0_under_minimum_coverage": constrained_value,
        "utility_lost_to_minimum_coverage": oracle_value - constrained_value,
        "independence_note": "All audited sources have 27 anchors; decision mean equals source macro here.",
    }
    return result, decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--odur-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    root = args.odur_root.resolve()
    pinned_code = {
        "crashbench/models/selective.py": "87f2977b2a882983d550d0f9ace158ff6a422ad933801d59794169966d49b453",
        "crashbench/models/option_outcome.py": "98ef52fae16b7239d2cf7db292cd577db11104c5d14c9e8a1d8cd26e0e9d5fd2",
        "scripts/expansion/analyze_benchmark_test.py": "395277cf8b158f86e689be216c4b5b71f919a1e038800edd76c4b76df7bf4395",
    }
    for relative, digest in pinned_code.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"This reproduction targets c337836; code differs: {relative}")
    sys.path.insert(0, str(root))
    from crashbench.models.selective import PairwiseSourceConformalSelector
    from scripts.expansion.analyze_benchmark_test import support_counts

    d5 = next((root / "results/expansion/d5_statewise").glob("*/postprocess_a4b1714/merged"))
    d8 = next((root / "results/expansion/d8_benchmark").glob("*/postprocess_pinned_9c869dc/merged"))
    d6 = next((root / "results/expansion/d6_scoped").iterdir())
    report = {"snapshot": "c3378366d53cf514914ae837292126dbd806c4b2", "scope": "retrospective_read_only_audit", "support": {}}
    report["D2_support"] = {}
    for family, module_name, block_function in [
        ("fragile", "analyze_fragile_screen", "_complete_block_summary"),
        ("staleness", "analyze_staleness_screen", "_block"),
        ("action_drift", "analyze_action_drift_screen", "_block"),
        ("narrow", "analyze_narrow_clearance_screen", "_block"),
    ]:
        directory = next((root / "results/expansion" / f"d2_{family}_screen").iterdir())
        original = json.loads(next(directory.glob("*screen_analysis.json")).read_text())
        summarize_block = getattr(importlib.import_module(f"scripts.expansion.{module_name}"), block_function)
        sources = []
        for path in sorted(directory.glob("task*_source*.json")):
            shard = json.loads(path.read_text())
            blocks = [summarize_block(block) for block in shard["blocks"]
                      if block.get("status") == "COMPLETE_REALIZED_OPTIONS"]
            complete = [block for block in blocks if block["complete"]]
            if complete:
                sources.append(complete)
        report["D2_support"][family] = {
            "eligible_sources": len(sources),
            "production_gate_B0": original["summary"]["benefit_zero_sources"],
            "contains_B0": sum(any(not block["benefit"] for block in source) for source in sources),
            "contains_B1": sum(any(block["benefit"] for block in source) for source in sources),
            "original_status": original["gate"]["status"],
            "original_hard_failures": original["gate"].get("hard_failures"),
            "original_claim_failures": original["gate"].get("claim_scope_failures"),
            "source_shards_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in sorted(directory.glob("task*_source*.json"))},
        }
    for label, directory, roles in [("D5_train_development", d5, {"train", "development"}),
                                    ("D5_development", d5, {"development"}),
                                    ("D8_sealed_test", d8, None)]:
        aa, bb = rows(directory / "anchors.jsonl"), rows(directory / "branches.jsonl")
        if roles:
            aa = [row for row in aa if row["split_role"] in roles]
            bb = [row for row in bb if row["split_role"] in roles]
        summary, _ = summarize_support(aa, bb)
        original = support_counts(aa, bb)
        assert original["benefit_zero_sources"] == summary["source_counts"]["entirely_B0_original_gate"]
        assert original["benefit_one_sources"] == summary["source_counts"]["contains_B1"]
        summary["production_gate_B0"] = original["benefit_zero_sources"]
        report["support"][label] = summary

    calibration = json.loads((d6 / "calibration/statewise_calibration_freeze.json").read_text())
    option_ids = ("base_continue", "observation_refresh", "safe_stop")
    selector = PairwiseSourceConformalSelector(
        option_ids, utility_quantiles=calibration["utility_pairwise_quantiles"],
        catastrophe_difference_quantiles=calibration["catastrophe_difference_quantiles"],
        catastrophe_absolute_quantiles=calibration["catastrophe_absolute_quantiles"],
    )
    rng = np.random.default_rng(20260906)
    selections = collections.Counter()
    for _ in range(10000):
        probabilities = rng.uniform(0, 1, size=3)
        utility = rng.uniform(-2.12, 1.0, size=3)
        selected = selector.select(
            mechanism_id="observation_staleness_v1", predicted_utility=dict(zip(option_ids, utility)),
            predicted_catastrophe=dict(zip(option_ids, probabilities)),
            pairwise_scale={(a, b): 0.05 for a in option_ids for b in option_ids if a != b},
        )
        selections[selected.option_id] += 1
    perfect_recovery = selector.select(
        mechanism_id="observation_staleness_v1",
        predicted_utility={"base_continue": -2.0, "observation_refresh": 0.95, "safe_stop": -0.30},
        predicted_catastrophe={"base_continue": 1.0, "observation_refresh": 0.0, "safe_stop": 0.0},
        pairwise_scale={(a, b): 0.05 for a in option_ids for b in option_ids if a != b},
    )
    report["selector"] = {
        "absolute_quantile": calibration["catastrophe_absolute_quantiles"]["__global__"],
        "absolute_cap": selector.catastrophe_absolute,
        "utility_quantile": calibration["utility_pairwise_quantiles"]["__global__"],
        "minimum_pairwise_penalty": 0.05 * calibration["utility_pairwise_quantiles"]["__global__"],
        "all_probabilities_blocked_by_absolute_cap": calibration["catastrophe_absolute_quantiles"]["__global__"] > selector.catastrophe_absolute,
        "synthetic_prediction_selections": dict(selections),
        "perfect_recovery_prediction_result": perfect_recovery.__dict__,
        "conformal_rank": math.ceil((calibration["calibration_source_count"] + 1) * (1 - calibration["alpha"])),
        "calibration_sources": calibration["calibration_source_count"],
        "minimum_alpha_for_absolute_quantile_le_0_1_on_saved_scores": 1 - sum(
            row["catastrophe_absolute"] <= 0.1 for row in calibration["source_max_scores"].values()
        ) / (calibration["calibration_source_count"] + 1),
    }

    predictions = rows(d6 / "selection/seed_0/predictions.jsonl")
    by_block = collections.defaultdict(dict)
    for row in predictions:
        by_block[row["row_id"].rsplit(":", 1)[0]][row["option_id"]] = row
    log_odds_shifts = []
    for group in by_block.values():
        base = group["base_continue"]["outcome_probabilities"]
        refresh = group["observation_refresh"]["outcome_probabilities"]
        log_odds_shifts.append(math.log(refresh[0] / refresh[1]) - math.log(base[0] / base[1]))
    report["option_conditioning"] = {
        "seed": 0, "anchors": len(log_odds_shifts),
        "refresh_minus_Base_success_catastrophe_log_odds_shift_min": min(log_odds_shifts),
        "refresh_minus_Base_success_catastrophe_log_odds_shift_max": max(log_odds_shifts),
        "spread": max(log_odds_shifts) - min(log_odds_shifts),
        "structural_reason": "outcome_logits(state, option) = W_state * trunk(state) + W_option * embedding(option) + bias; no state-option interaction after concatenation",
    }

    development = rows(d5 / "branches.jsonl")
    development = [row for row in development if row["split_role"] == "development"]
    baseline = json.loads((d6 / "baselines/baseline_selection.json").read_text())
    ensemble = collections.defaultdict(list)
    for seed in range(5):
        for row in rows(d6 / "selection" / f"seed_{seed}/predictions.jsonl"):
            if row["role"] == "development":
                ensemble[row["row_id"]].append(row["predicted_u0"])
    candidate_utilities = collections.defaultdict(dict)
    for row_id, values in ensemble.items():
        block, option = row_id.rsplit(":", 1)
        candidate_utilities[block][option] = float(np.mean(values))
    choices = {block: max(values, key=values.get) for block, values in candidate_utilities.items()}
    direct = baseline["development_choices"]["DirectQ"]
    source_values = collections.defaultdict(lambda: {"ODUR": [], "DirectQ": []})
    for row in development:
        for method, selected in [("ODUR", choices), ("DirectQ", direct)]:
            if selected[row["block_id"]] == row["option_id"]:
                source_values[row["physical_source_id"]][method].append(row["u0"])
    differences = np.asarray([np.mean(values["ODUR"]) - np.mean(values["DirectQ"])
                              for values in source_values.values()])
    bootstrap_rng = np.random.default_rng(20260906)
    bootstrap = np.mean(bootstrap_rng.choice(differences, (10000, len(differences))), axis=1)
    report["ODUR_vs_DirectQ"] = {
        "role": "posthoc_development_diagnostic_fixed_trained_ensemble",
        "physical_sources": len(differences), "delta_u0": float(np.mean(differences)),
        "paired_source_bootstrap_95pct": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "bootstrap_replicates": len(bootstrap), "seed": 20260906,
        "per_source_differences": dict(zip(source_values, map(float, differences))),
        "does_not_account_for_model_selection": True,
    }

    aa, bb = rows(d5 / "anchors.jsonl"), rows(d5 / "branches.jsonl")
    all_predictions = []
    for stage in ["selection", "refit"]:
        stage_result = []
        for seed in range(5):
            manifest = json.loads((d6 / stage / f"seed_{seed}/manifest.json").read_text())
            stage_result.append({key: manifest.get(key) for key in ["seed", "fit_on", "fit_rows", "train_rows", "development_rows"]})
        all_predictions.append({"stage": stage, "manifests": stage_result})
    report["training_roles"] = all_predictions

    feature_path = d5.parent / "feature_cache/features.npz"
    with np.load(feature_path, allow_pickle=False) as cache:
        feature_map = dict(zip(map(str, cache["sha256"]), cache["features"]))
    feature_groups = collections.defaultdict(list)
    for anchor in aa:
        feat = np.asarray(feature_map[anchor["anchor_feature_blob"]["sha256"]], dtype=np.float32)
        feature_groups[hashlib.sha256(feat.tobytes()).hexdigest()].append(anchor)
    _, decisions = summarize_support(aa, bb)
    decision_by_block = {row["block_id"]: row for row in decisions}
    ambiguous = []
    feature_restricted_value = 0.0
    exact_oracle_value = sum(max(row["utility"].values()) for row in decisions)
    for group in feature_groups.values():
        dd = [decision_by_block[row["block_id"]] for row in group]
        strict_labels = {row["strict_winner"] for row in dd if row["strict_winner"]}
        if len(strict_labels) > 1:
            ambiguous.append({"blocks": len(group), "sources": len({row["physical_source_id"] for row in group}), "strict_labels": sorted(strict_labels)})
        feature_restricted_value += max(sum(row["utility"][option] for row in dd) for option in option_ids)
    report["feature_aliasing"] = {
        "scope": "D5 all roles; hindsight diagnostic, not a generalization estimate",
        "anchors": len(aa), "feature_width": len(next(iter(feature_map.values()))),
        "exact_distinct_feature_vectors": len(feature_groups),
        "duplicate_vector_groups": sum(len(group) > 1 for group in feature_groups.values()),
        "groups_with_conflicting_strict_winners": len(ambiguous),
        "conflicting_groups": ambiguous,
        "exact_oracle_u0": exact_oracle_value / len(aa),
        "best_deterministic_exact_feature_lookup_u0": feature_restricted_value / len(aa),
    }
    report["input_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [d5 / "anchors.jsonl", d5 / "branches.jsonl", d8 / "anchors.jsonl", d8 / "branches.jsonl",
                     d6 / "calibration/statewise_calibration_freeze.json", d6 / "selection/seed_0/predictions.jsonl",
                     root / "crashbench/models/selective.py", root / "crashbench/models/option_outcome.py",
                     root / "scripts/expansion/analyze_benchmark_test.py"]
    }
    def finite_json(value):
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite_json(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    output = json.dumps(finite_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
