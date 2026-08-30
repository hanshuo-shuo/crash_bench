#!/usr/bin/env python3
"""Train one ODUR seed on D5 train rows and score frozen development rows."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.branching.artifacts import unpack_numeric_mapping
from crashbench.data.utility import OutcomeVector, PhysicalBudgets, normalized_costs
from crashbench.models.option_outcome import (
    COST_TARGETS,
    OptionOutcomeModel,
    pooled_anchor_features,
    source_balanced_weights,
)
from scripts.expansion.hash_tree_manifest import resolve_git_head
from scripts.expansion.hash_tree_manifest import resolve_git_head


OPTION_IDS = ("base_continue", "observation_refresh", "safe_stop")
ALLOWED_ROLES = {"train", "development"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_training_arrays(
    anchors: list[dict[str, Any]],
    branches: list[dict[str, Any]],
    *,
    artifact_store: Path,
    budgets: PhysicalBudgets,
) -> dict[str, Any]:
    anchor_by_block = {row["block_id"]: row for row in anchors}
    feature_cache = {}
    features, options, outcomes, costs, sources, roles, row_ids, actual_u0 = (
        [], [], [], [], [], [], [], [],
    )
    for branch in branches:
        anchor = anchor_by_block.get(branch["block_id"])
        if anchor is None:
            raise ValueError(f"branch lacks anchor: {branch['block_id']}")
        role = str(anchor["split_role"])
        if role not in ALLOWED_ROLES:
            continue
        ref = anchor["anchor_feature_blob"]
        key = ref["sha256"]
        if key not in feature_cache:
            payload = (artifact_store / ref["relative_path"]).read_bytes()
            feature_cache[key] = pooled_anchor_features(unpack_numeric_mapping(payload))
        outcome = branch["outcome"]
        terminal = [
            int(outcome["task_success"]),
            int(outcome["catastrophe"]),
            int(outcome["safe_noncompletion"]),
        ]
        if sum(terminal) != 1:
            raise ValueError(f"branch terminal indicators are invalid: {branch['block_id']}")
        vector = OutcomeVector(
            task_success=terminal[0],
            catastrophe=terminal[1],
            safe_noncompletion=terminal[2],
            intervention_invoked=int(outcome["intervention_invoked"]),
            human_help=int(outcome["human_help"]),
            option_duration_steps=float(outcome["option_duration_steps"]),
            path_length_m=float(outcome["path_length_m"]),
            force_exposure_ns=float(outcome["force_exposure_ns"]),
            latency_ms=float(outcome["latency_ms"]),
        )
        normalized = normalized_costs(vector, budgets)
        features.append(feature_cache[key])
        options.append(branch["option_id"])
        outcomes.append(int(np.argmax(terminal)))
        costs.append([normalized[name] for name in COST_TARGETS])
        sources.append(anchor["physical_source_id"])
        roles.append(role)
        row_ids.append(f"{branch['block_id']}:{branch['option_id']}")
        actual_u0.append(float(branch["u0"]))
    if not features:
        raise ValueError("training dataset contains no train/development rows")
    widths = {len(row) for row in features}
    if len(widths) != 1:
        raise ValueError("anchor feature width drift")
    return {
        "features": np.asarray(features, dtype=np.float32),
        "options": np.asarray(options),
        "outcomes": np.asarray(outcomes, dtype=np.int64),
        "costs": np.asarray(costs, dtype=np.float32),
        "sources": np.asarray(sources),
        "roles": np.asarray(roles),
        "row_ids": np.asarray(row_ids),
        "actual_u0": np.asarray(actual_u0, dtype=np.float32),
    }


def expected_u0(probabilities: np.ndarray, costs: np.ndarray, options: np.ndarray) -> np.ndarray:
    base = probabilities @ np.asarray([1.0, -2.0, -0.25], dtype=np.float32)
    intervention = (options != "base_continue").astype(np.float32)
    continuous = costs @ np.asarray([-0.02, -0.02, -0.02, -0.01], dtype=np.float32)
    return base - 0.05 * intervention + continuous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite ODUR seed output: {args.output_dir}")
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    utility = json.loads(args.utility_config.read_text())
    norm = utility["normalization"]
    budgets = PhysicalBudgets(
        norm["option_duration_steps"], norm["path_length_m"],
        norm["force_exposure_ns"], norm["latency_ms"], norm["source"]
    )
    data = build_training_arrays(
        read_jsonl(args.merged_dir / "anchors.jsonl"),
        read_jsonl(args.merged_dir / "branches.jsonl"),
        artifact_store=args.artifact_store,
        budgets=budgets,
    )
    train = data["roles"] == "train"
    development = data["roles"] == "development"
    if not np.any(train) or not np.any(development):
        raise ValueError("ODUR requires both train and development rows")
    model = OptionOutcomeModel(
        input_dim=data["features"].shape[1], option_ids=OPTION_IDS
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.from_numpy(data["features"][train])
    option_index = model.option_indices(data["options"][train])
    y = torch.from_numpy(data["outcomes"][train])
    cost = torch.from_numpy(data["costs"][train])
    weights = torch.from_numpy(source_balanced_weights(data["sources"][train]))
    model.train()
    history = []
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        outputs = model(x, option_index)
        losses = model.loss(
            outputs,
            outcome_targets=y,
            cost_targets=cost,
            row_weights=weights,
        )
        losses["loss"].backward()
        optimizer.step()
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 10 == 0:
            history.append({"epoch": epoch + 1, **{key: float(value.detach()) for key, value in losses.items()}})
    model.eval()
    with torch.no_grad():
        all_outputs = model(
            torch.from_numpy(data["features"]),
            model.option_indices(data["options"]),
        )
        probabilities = torch.softmax(all_outputs["outcome_logits"], dim=-1).cpu().numpy()
        predicted_costs = all_outputs["cost_prediction"].cpu().numpy()
    predicted_utility = expected_u0(probabilities, predicted_costs, data["options"])
    predictions = [
        {
            "row_id": str(data["row_ids"][index]),
            "physical_source_id": str(data["sources"][index]),
            "role": str(data["roles"][index]),
            "option_id": str(data["options"][index]),
            "outcome_probabilities": probabilities[index].tolist(),
            "predicted_costs": predicted_costs[index].tolist(),
            "predicted_u0": float(predicted_utility[index]),
            "actual_u0": float(data["actual_u0"][index]),
        }
        for index in range(len(predicted_utility))
    ]
    args.output_dir.mkdir(parents=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": data["features"].shape[1],
            "option_ids": OPTION_IDS,
            "seed": args.seed,
        },
        args.output_dir / "model.pt",
    )
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
    )
    manifest = {
        "schema_version": 1,
        "kind": "crashbench_expansion_odur_seed",
        "seed": args.seed,
        "git_commit": resolve_git_head(Path.cwd()),
        "train_rows": int(np.sum(train)),
        "development_rows": int(np.sum(development)),
        "calibration_rows_read": 0,
        "test_rows_read": 0,
        "history": history,
        "prediction_rows": len(predictions),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(args.output_dir), "seed": args.seed, "train": int(np.sum(train)), "development": int(np.sum(development))}, sort_keys=True))


if __name__ == "__main__":
    main()
