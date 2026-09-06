#!/usr/bin/env python3
"""Bounded train-fitting diagnosis and direct paired-gain development study."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from crashbench.data.utility import PhysicalBudgets
from crashbench.models.option_outcome import OptionOutcomeModel, source_balanced_weights
from crashbench.models.paired_gain import (
    OPTION_IDS, GAIN_OPTIONS, PairedGainModel, pair_decisions,
    train_standardization, choose_from_gains, tiny_train_sources,
)
from crashbench.models.advantage import RidgeOptionValue
from scripts.expansion.train_option_value import build_training_arrays, read_jsonl, sha256_file
from scripts.expansion.run_round1_repair import policy_metrics, paired_interval
from scripts.expansion.hash_tree_manifest import resolve_git_head

RECIPES = ("outcome_interaction", "outcome_per_option", "gain_layernorm", "gain_positive5", "gain_standardized")
TERMINAL_WEIGHTS = np.array([1., -2., -.25], dtype=np.float32)
COST_WEIGHTS = np.array([-.02, -.02, -.02, -.01], dtype=np.float32)


def gain_diagnostics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError("unaligned or invalid gain predictions")
    positive = actual > 0
    p, n = predicted[positive], predicted[~positive]
    auc = None if not len(p) or not len(n) else float(((p[:, None] > n) + .5 * (p[:, None] == n)).mean())
    selected = predicted > 0
    return {"positive_count": int(positive.sum()), "negative_count": int((~positive).sum()),
            "predicted_positive": int(selected.sum()), "mse": float(np.mean((predicted - actual) ** 2)),
            "gain_sign_recall": float(selected[positive].mean()) if positive.any() else None,
            "gain_sign_precision": float(positive[selected].mean()) if selected.any() else None,
            "auc": auc,
            "true_positive_mean_prediction": float(p.mean()) if len(p) else None,
            "true_negative_mean_prediction": float(n.mean()) if len(n) else None}


def recovery_metrics(paired, mask, choices):
    blocks = paired["blocks"][mask]
    outcomes = paired["outcomes"][mask]
    chosen = np.array([OPTION_IDS.index(choices[str(block)]) for block in blocks])
    selected = outcomes[np.arange(len(blocks)), chosen]
    rescue = (outcomes[:, 0] != 0) & (outcomes[:, 1] == 0)
    rescued = rescue & (chosen == 1)
    return {"refresh_rescue_opportunities": int(rescue.sum()),
            "refresh_rescues_recovered": int(rescued.sum()),
            "refresh_rescue_recall": float(rescued.sum() / rescue.sum()) if rescue.any() else None,
            "base_successes_lost": int(((outcomes[:, 0] == 0) & (selected != 0)).sum()),
            "new_catastrophes_vs_base": int(((outcomes[:, 0] != 1) & (selected == 1)).sum()),
            "refresh_new_catastrophes": int(((outcomes[:, 0] != 1) & (selected == 1) & (chosen == 1)).sum()),
            "refresh_count": int((chosen == 1).sum()), "stop_count": int((chosen == 2).sum())}


def evaluate(paired, branches, gains, role, *, subset_sources=None, decomposition=None):
    mask = paired["roles"] == role
    if subset_sources is not None:
        mask &= np.isin(paired["sources"], subset_sources)
    blocks = set(map(str, paired["blocks"][mask]))
    rows = [row for row in branches if str(row["block_id"]) in blocks]
    result = {"decision_count": len(blocks), "source_count": len(set(paired["sources"][mask])),
              "gain_diagnostics": {o: gain_diagnostics(paired["gains"][mask, i], gains[mask, i])
                                   for i, o in enumerate(GAIN_OPTIONS)}, "policies": {}}
    for mode, allow_stop in (("all_options", True), ("refresh_only", False)):
        choices = choose_from_gains(gains[mask], paired["blocks"][mask], allow_stop=allow_stop)
        result["policies"][mode] = {**policy_metrics(rows, choices), **recovery_metrics(paired, mask, choices)}
    if decomposition is not None:
        parts = {}
        for label, positive in (("refresh_beneficial", True), ("refresh_not_beneficial", False)):
            group = mask & ((paired["gains"][:, 0] > 0) == positive)
            parts[label] = {"count": int(group.sum()), **{
                name: float(value[group, 0].mean()) if group.any() else None for name, value in decomposition.items()}}
        result["refresh_gain_decomposition"] = parts
    return result


def train_seed(data, paired, *, recipe, seed, fit_sources, checkpoints, output):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pair_fit = (paired["roles"] == "train") & np.isin(paired["sources"], fit_sources)
    row_fit = (data["roles"] == "train") & np.isin(data["sources"], fit_sources)
    if not pair_fit.any() or set(data["sources"][row_fit]) != set(fit_sources):
        raise ValueError("invalid train source selection")
    is_outcome = recipe.startswith("outcome_")
    if is_outcome:
        model = OptionOutcomeModel(input_dim=data["features"].shape[1], option_ids=OPTION_IDS,
                                   architecture=recipe.removeprefix("outcome_"))
        x = torch.from_numpy(data["features"][row_fit])
        idx = model.option_indices(data["options"][row_fit])
        y = torch.from_numpy(data["outcomes"][row_fit])
        cost = torch.from_numpy(data["costs"][row_fit])
        weights = torch.from_numpy(source_balanced_weights(data["sources"][row_fit]))
    else:
        normalization = "train_standardized" if recipe == "gain_standardized" else "layernorm"
        mean = scale = None
        if normalization == "train_standardized":
            mean, scale = train_standardization(paired["features"][pair_fit], paired["roles"][pair_fit])
        model = PairedGainModel(data["features"].shape[1], normalization=normalization, mean=mean, scale=scale)
        x = torch.from_numpy(paired["features"][pair_fit])
        y = torch.from_numpy(paired["gains"][pair_fit])
        weights = torch.from_numpy(source_balanced_weights(paired["sources"][pair_fit]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    history, predictions = [], {}
    output.mkdir(parents=True)
    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        optimizer.zero_grad()
        if is_outcome:
            losses = model.loss(model(x, idx), outcome_targets=y, cost_targets=cost, row_weights=weights)
            loss = losses["loss"]
        else:
            loss = model.loss(model(x), y, weights, positive_weight=5. if recipe == "gain_positive5" else 1.)
            losses = {"loss": loss}
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss {recipe} seed {seed} epoch {epoch}")
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch in checkpoints:
            history.append({"epoch": epoch, **{k: float(v.detach()) for k, v in losses.items()}})
        if epoch in checkpoints:
            model.eval()
            with torch.no_grad():
                if is_outcome:
                    out = model(torch.from_numpy(data["features"]), model.option_indices(data["options"]))
                    prob = torch.softmax(out["outcome_logits"], -1).numpy()[paired["indices"]]
                    costs = out["cost_prediction"].numpy()[paired["indices"]]
                    terminal = prob @ TERMINAL_WEIGHTS
                    continuous = costs @ COST_WEIGHTS
                    terminal_gain, cost_gain = terminal[:, 1:] - terminal[:, :1], continuous[:, 1:] - continuous[:, :1]
                    gain = terminal_gain + cost_gain - .05
                    record = {"gains": gain, "terminal_gain": terminal_gain, "cost_gain": cost_gain,
                              "probabilities": prob, "predicted_costs": costs}
                else:
                    record = {"gains": model(torch.from_numpy(paired["features"])).numpy()}
            predictions[epoch] = record
            np.savez_compressed(output / f"epoch_{epoch}_predictions.npz", **record)
    model_metadata = {"state_dict": model.state_dict(), "recipe": recipe, "seed": seed,
                      "input_dim": data["features"].shape[1], "option_ids": OPTION_IDS}
    torch.save(model_metadata, output / "model.pt")
    manifest = {"recipe": recipe, "seed": seed, "fit_on": "train", "fit_source_ids": list(fit_sources),
                "fit_decisions": int(pair_fit.sum()), "feature_dim": data["features"].shape[1],
                "checkpoints": list(checkpoints), "saved_model_epoch": max(checkpoints),
                "train_history_loss_timing": "pre_optimizer_step", "parameter_count": sum(p.numel() for p in model.parameters()),
                "history": history, "test_rows_read": 0, "calibration_outcomes_used": False,
                "git_commit": resolve_git_head(ROOT), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "model_sha256": sha256_file(output / "model.pt"),
                "prediction_sha256": {str(ep): sha256_file(output / f"epoch_{ep}_predictions.npz") for ep in checkpoints}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"recipe": recipe, "seed": seed, "fit_decisions": int(pair_fit.sum()), "last_loss": history[-1]["loss"], "output": str(output)}), flush=True)
    return predictions, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, default=ROOT / "configs/expansion/utility_v1.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="one seed and two epochs; not scientific evidence")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite round-two output: {args.output_dir}")
    import torch
    torch.set_num_threads(2)
    anchors = read_jsonl(args.merged_dir / "anchors.jsonl")
    branches = read_jsonl(args.merged_dir / "branches.jsonl")
    if {r["split_role"] for r in anchors + branches} - {"train", "development", "calibration"}:
        raise ValueError("round two accepts D5 inputs only")
    # Filter labels before feature loading and all learning/diagnosis.
    branches = [r for r in branches if r["split_role"] in {"train", "development"}]
    norm = json.loads(args.utility_config.read_text())["normalization"]
    budgets = PhysicalBudgets(norm["option_duration_steps"], norm["path_length_m"], norm["force_exposure_ns"], norm["latency_ms"], norm["source"])
    data = build_training_arrays(anchors, branches, artifact_store=args.artifact_store, budgets=budgets, feature_cache=args.feature_cache)
    paired = pair_decisions(data)
    reconstructed = TERMINAL_WEIGHTS[paired["outcomes"]] + paired["costs"] @ COST_WEIGHTS - np.array([0., .05, .05])
    if not np.allclose(reconstructed, paired["utility"], atol=2e-6, rtol=0):
        raise ValueError("stored U0 disagrees with terminal/cost/invocation decomposition")
    fit_sources = sorted(set(map(str, paired["sources"][paired["roles"] == "train"])))
    dev_sources = sorted(set(map(str, paired["sources"][paired["roles"] == "development"])))
    if set(fit_sources) & set(dev_sources) or not dev_sources:
        raise ValueError("development sources must be disjoint and nonempty")
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "decision_index.npz", blocks=paired["blocks"], sources=paired["sources"],
                        roles=paired["roles"], actual_gains=paired["gains"], outcomes=paired["outcomes"])
    checkpoints, tiny_checkpoints = ((1, 2), (1, 2)) if args.smoke else ((100, 500), (100, 500, 2000))
    seeds = (0,) if args.smoke else tuple(range(5))
    manifest = {"schema_version": 1, "kind": "refresh_gain_round2", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": resolve_git_head(ROOT), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "fit_sources": fit_sources, "development_sources": dev_sources, "feature_dim": data["features"].shape[1],
                "test_rows_read": 0, "calibration_outcomes_used": False, "confirmatory": False,
                "smoke": args.smoke, "seeds": list(seeds), "checkpoints": checkpoints,
                "input_sha256": {name: sha256_file(args.merged_dir / name) for name in ("anchors.jsonl", "branches.jsonl", "dataset_manifest.json")},
                "feature_cache_sha256": sha256_file(args.feature_cache), "utility_config_sha256": sha256_file(args.utility_config),
                "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "hostname": platform.node()}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    train_rows = data["roles"] == "train"
    directq = RidgeOptionValue(OPTION_IDS, ridge=1.).fit(data["features"][train_rows], data["options"][train_rows],
        data["actual_u0"][train_rows], source_balanced_weights(data["sources"][train_rows]))
    q = directq.predict(data["features"], data["options"])[paired["indices"]]
    direct_gains = q[:, 1:] - q[:, :1]
    np.savez_compressed(args.output_dir / "directq.npz", gains=direct_gains,
                        coefficients=np.array([directq.coefficients_[o] for o in OPTION_IDS]))
    report = {"manifest": manifest, "baselines": {}, "tiny_fit": {}, "recipes": {},
              "feature_metadata_audit": {"anchor_field_names": sorted(anchors[0]),
                  "model_inputs": "existing pooled images and proprio only",
                  "age_history_fields_in_anchor_schema": [k for k in anchors[0] if k in {"observation_age", "history", "chunk_progress"}],
                  "privileged_metadata_used": False}}
    for name, gain in (("Base", np.zeros_like(direct_gains)), ("DirectQ", direct_gains), ("OracleDiagnostic", paired["gains"])):
        report["baselines"][name] = {role: evaluate(paired, branches, gain, role) for role in ("train", "development")}
    tiny_sources = tiny_train_sources(paired)
    report["tiny_source_selection"] = {"rule": "first four lexicographic train sources containing rescue and nonbenefit; retain all blocks", "source_ids": tiny_sources,
                                       "outcome_selected_engineering_only": True}
    if not tiny_sources:
        report["tiny_fit_status"] = "NO_TRAIN_RESCUE_SOURCE_AVAILABLE"
    else:
        for recipe in ("outcome_per_option", "gain_layernorm", "gain_standardized"):
            snapshots, info = train_seed(data, paired, recipe=recipe, seed=0, fit_sources=tiny_sources,
                checkpoints=tiny_checkpoints, output=args.output_dir / "tiny" / recipe)
            report["tiny_fit"][recipe] = {"fit_decisions": info["fit_decisions"], "history": info["history"], "checkpoints": {
                str(ep): evaluate(paired, branches, value["gains"], "train", subset_sources=tiny_sources)
                for ep, value in snapshots.items()}}
    for recipe in RECIPES:
        snapshot_seeds, seed_evaluations = [], []
        for seed in seeds:
            snapshots, info = train_seed(data, paired, recipe=recipe, seed=seed, fit_sources=fit_sources,
                checkpoints=checkpoints, output=args.output_dir / recipe / f"seed_{seed}")
            snapshot_seeds.append(snapshots)
            seed_evaluations.append({"seed": seed, "history": info["history"], "checkpoints": {
                str(ep): {role: evaluate(paired, branches, value["gains"], role) for role in ("train", "development")}
                for ep, value in snapshots.items()}})
        combined = {}
        for ep in checkpoints:
            records = [s[ep] for s in snapshot_seeds]
            gains = np.mean([value["gains"] for value in records], axis=0)
            np.savez_compressed(args.output_dir / recipe / f"ensemble_epoch_{ep}.npz", gains=gains)
            parts = None
            if "terminal_gain" in records[0]:
                parts = {key: np.mean([r[key] for r in records], axis=0) for key in ("terminal_gain", "cost_gain")}
                parts["net_gain"] = gains
            combined[str(ep)] = {role: evaluate(paired, branches, gains, role, decomposition=parts) for role in ("train", "development")}
            combined[str(ep)]["development_paired_vs_DirectQ"] = {
                mode: {key: paired_interval(combined[str(ep)]["development"]["policies"][mode],
                        report["baselines"]["DirectQ"]["development"]["policies"][mode], key)
                       for key in ("u0", "task_success", "catastrophe")}
                for mode in ("all_options", "refresh_only")}
        report["recipes"][recipe] = {"ensemble_checkpoints": combined, "per_seed": seed_evaluations}
    report["status"] = "COMPLETE_DEVELOPMENT_DIAGNOSTIC"
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    ledger = {str(p.relative_to(args.output_dir)): sha256_file(p) for p in sorted(args.output_dir.rglob('*')) if p.is_file()}
    (args.output_dir / "output_sha256.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output_dir), "status": report["status"]}), flush=True)

if __name__ == "__main__":
    main()
