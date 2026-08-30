#!/usr/bin/env python3
"""Calibrate the frozen five-seed ODUR ensemble without reading test outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.data.utility import PhysicalBudgets
from crashbench.models.option_outcome import OptionOutcomeModel
from crashbench.models.selective import (
    PairwiseSourceConformalSelector,
    source_conformal_quantile,
)
from scripts.expansion.train_option_value import (
    OPTION_IDS,
    build_training_arrays,
    expected_u0,
    read_jsonl,
)


MECHANISM_ID = "observation_staleness_v1"
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
SIGMA_FLOOR = 0.05
ALPHA = 0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict_seed(model_path: Path, data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if tuple(checkpoint["option_ids"]) != OPTION_IDS:
        raise ValueError("refit model option catalog identity drift")
    model = OptionOutcomeModel(
        input_dim=int(checkpoint["input_dim"]), option_ids=checkpoint["option_ids"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        outputs = model(
            torch.from_numpy(data["features"]),
            model.option_indices(data["options"]),
        )
        probabilities = torch.softmax(outputs["outcome_logits"], dim=-1).cpu().numpy()
        costs = outputs["cost_prediction"].cpu().numpy()
    return expected_u0(probabilities, costs, data["options"]), probabilities[:, 1]


def calibration_source_scores(
    *,
    sources: np.ndarray,
    row_ids: np.ndarray,
    options: np.ndarray,
    actual_utility: np.ndarray,
    actual_catastrophe: np.ndarray,
    predicted_utility_seeds: np.ndarray,
    predicted_catastrophe_seeds: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return per-source max utility, cat-difference, and absolute-cat scores."""

    if predicted_utility_seeds.shape != predicted_catastrophe_seeds.shape:
        raise ValueError("utility/catastrophe ensemble shape mismatch")
    if predicted_utility_seeds.shape[1] != len(row_ids):
        raise ValueError("ensemble prediction row mismatch")
    mean_u = np.mean(predicted_utility_seeds, axis=0)
    mean_cat = np.mean(predicted_catastrophe_seeds, axis=0)
    blocks = np.asarray([str(row_id).rsplit(":", 1)[0] for row_id in row_ids])
    utility_scores: dict[str, float] = {}
    difference_scores: dict[str, float] = {}
    absolute_scores: dict[str, float] = {}
    for source in sorted(set(map(str, sources))):
        source_mask = sources == source
        utility_residuals = []
        difference_residuals = []
        absolute_residuals = []
        for block in sorted(set(blocks[source_mask])):
            indices = np.flatnonzero(source_mask & (blocks == block))
            by_option = {str(options[index]): int(index) for index in indices}
            if set(by_option) != set(OPTION_IDS):
                raise ValueError(f"calibration block option incompleteness: {block}")
            base_index = by_option["base_continue"]
            absolute_residuals.extend(
                max(0.0, float(actual_catastrophe[index] - mean_cat[index]))
                for index in indices
            )
            for option in OPTION_IDS:
                option_index = by_option[option]
                observed_cat_gap = actual_catastrophe[option_index] - actual_catastrophe[base_index]
                predicted_cat_gap = mean_cat[option_index] - mean_cat[base_index]
                difference_residuals.append(max(0.0, float(observed_cat_gap - predicted_cat_gap)))
                for other in OPTION_IDS:
                    if other == option:
                        continue
                    other_index = by_option[other]
                    predicted_gaps = (
                        predicted_utility_seeds[:, option_index]
                        - predicted_utility_seeds[:, other_index]
                    )
                    scale = max(float(np.std(predicted_gaps, ddof=1)), SIGMA_FLOOR)
                    observed_gap = actual_utility[option_index] - actual_utility[other_index]
                    predicted_gap = mean_u[option_index] - mean_u[other_index]
                    utility_residuals.append(abs(float(predicted_gap - observed_gap)) / scale)
        utility_scores[source] = max(utility_residuals)
        difference_scores[source] = max(difference_residuals)
        absolute_scores[source] = max(absolute_residuals)
    return utility_scores, difference_scores, absolute_scores


def quantile_payload(scores: Mapping[str, float]) -> dict[str, float]:
    value = source_conformal_quantile(list(scores.values()), alpha=ALPHA)
    # The scoped dataset has one mechanism, so its valid Mondrian stratum is
    # identical to global.  Keeping both keys enforces the runtime API.
    return {PairwiseSourceConformalSelector.GLOBAL: value, MECHANISM_ID: value}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, action="append", required=True)
    parser.add_argument("--development-selection", type=Path, required=True)
    parser.add_argument("--baseline-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite calibration output: {args.output_dir}")
    if len(args.seed_dir) != 5:
        raise ValueError("calibration requires exactly five frozen refit seeds")
    development = json.loads(args.development_selection.read_text())
    baseline_sha = sha256_file(args.baseline_selection)
    if development["baseline_selection_sha256"] != baseline_sha:
        raise ValueError("frozen comparator identity hash drift")
    utility = json.loads(args.utility_config.read_text())
    norm = utility["normalization"]
    budgets = PhysicalBudgets(
        norm["option_duration_steps"], norm["path_length_m"],
        norm["force_exposure_ns"], norm["latency_ms"], norm["source"]
    )
    anchors = read_jsonl(args.merged_dir / "anchors.jsonl")
    branches = read_jsonl(args.merged_dir / "branches.jsonl")
    data = build_training_arrays(
        anchors,
        branches,
        artifact_store=args.artifact_store,
        budgets=budgets,
        allowed_roles={"calibration"},
    )
    if set(data["roles"]) != {"calibration"}:
        raise ValueError("calibration loader accessed a non-calibration role")
    utility_predictions, catastrophe_predictions, model_artifacts = [], [], []
    for expected_seed, directory in zip(EXPECTED_SEEDS, args.seed_dir):
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("seed") != expected_seed or manifest.get("fit_on") != "train_development":
            raise ValueError("calibration requires ordered train+development refit seeds")
        if manifest.get("calibration_rows_read") != 0 or manifest.get("test_rows_read") != 0:
            raise ValueError("refit artifact reports forbidden role access")
        model_path = directory / "model.pt"
        predicted_u, predicted_cat = predict_seed(model_path, data)
        utility_predictions.append(predicted_u)
        catastrophe_predictions.append(predicted_cat)
        model_artifacts.append(
            {"seed": expected_seed, "model_sha256": sha256_file(model_path), "manifest_sha256": sha256_file(manifest_path)}
        )
    predicted_u = np.asarray(utility_predictions)
    predicted_cat = np.asarray(catastrophe_predictions)
    actual_cat = (data["outcomes"] == 1).astype(float)
    utility_scores, difference_scores, absolute_scores = calibration_source_scores(
        sources=data["sources"], row_ids=data["row_ids"], options=data["options"],
        actual_utility=data["actual_u0"], actual_catastrophe=actual_cat,
        predicted_utility_seeds=predicted_u,
        predicted_catastrophe_seeds=predicted_cat,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_statewise_calibration_freeze",
        "scope": "single_primary_policy_single_staleness_mechanism_pilot",
        "alpha": ALPHA,
        "sigma_floor": SIGMA_FLOOR,
        "utility_pairwise_quantiles": quantile_payload(utility_scores),
        "catastrophe_difference_quantiles": quantile_payload(difference_scores),
        "catastrophe_absolute_quantiles": quantile_payload(absolute_scores),
        "calibration_source_count": len(utility_scores),
        "calibration_row_count": len(data["row_ids"]),
        "source_max_scores": {
            source: {
                "utility_pairwise": utility_scores[source],
                "catastrophe_difference": difference_scores[source],
                "catastrophe_absolute": absolute_scores[source],
            }
            for source in sorted(utility_scores)
        },
        "model_artifacts": model_artifacts,
        "development_selection_sha256": sha256_file(args.development_selection),
        "baseline_selection_sha256": baseline_sha,
        "calibration_rows_read": len(data["row_ids"]),
        "test_rows_read": 0,
        "frozen": True,
    }
    payload["calibration_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "statewise_calibration_freeze.json"
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "sources": len(utility_scores), "test_rows_read": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
