#!/usr/bin/env python3
"""Freeze D6 development-only model and comparator identities for the scoped pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.models.advantage import source_macro_policy_value


ALLOWED_ROLES = {"train", "development"}
EXPECTED_SEEDS = (0, 1, 2, 3, 4)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choices_from_prediction_rows(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[tuple[float, str]]] = {}
    for row in rows:
        block = str(row["row_id"]).rsplit(":", 1)[0]
        grouped.setdefault(block, []).append(
            (float(row["predicted_u0"]), str(row["option_id"]))
        )
    return {
        block: max(sorted(candidates), key=lambda value: value[0])[1]
        for block, candidates in grouped.items()
    }


def _oracle_choices(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[tuple[float, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["block_id"]), []).append(
            (float(row["u0"]), str(row["option_id"]))
        )
    return {
        block: max(sorted(candidates), key=lambda value: value[0])[1]
        for block, candidates in grouped.items()
    }


def analyze(
    *,
    branches: list[dict[str, Any]],
    seed_dirs: list[Path],
    baseline_dir: Path,
) -> dict[str, Any]:
    if len(seed_dirs) != len(EXPECTED_SEEDS):
        raise ValueError("D6 requires exactly five seed directories")
    development_rows = [row for row in branches if row["split_role"] == "development"]
    if not development_rows:
        raise ValueError("merged dataset has no development rows")
    prediction_sets: list[list[dict[str, Any]]] = []
    seed_artifacts = []
    reference_ids: list[str] | None = None
    for expected_seed, directory in zip(EXPECTED_SEEDS, seed_dirs):
        manifest_path = directory / "manifest.json"
        prediction_path = directory / "predictions.jsonl"
        model_path = directory / "model.pt"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("seed") != expected_seed:
            raise ValueError(f"seed order mismatch: expected {expected_seed}")
        if manifest.get("fit_on") not in (None, "train"):
            raise ValueError("development selection may use train-only fit artifacts")
        if manifest.get("calibration_rows_read") != 0 or manifest.get("test_rows_read") != 0:
            raise ValueError("D6 artifact reports calibration/test access")
        predictions = read_jsonl(prediction_path)
        if set(row["role"] for row in predictions) - ALLOWED_ROLES:
            raise ValueError("D6 predictions contain forbidden split roles")
        current_ids = [str(row["row_id"]) for row in predictions]
        if reference_ids is None:
            reference_ids = current_ids
        elif current_ids != reference_ids:
            raise ValueError("seed prediction row identity/order drift")
        if not all(np.isfinite(float(row["predicted_u0"])) for row in predictions):
            raise ValueError("D6 predictions contain NaN/Inf")
        prediction_sets.append(predictions)
        seed_artifacts.append(
            {
                "seed": expected_seed,
                "model_sha256": sha256_file(model_path),
                "predictions_sha256": sha256_file(prediction_path),
                "manifest_sha256": sha256_file(manifest_path),
                "development_rows": manifest["development_rows"],
                "test_rows_read": 0,
            }
        )
    baseline_path = baseline_dir / "baseline_selection.json"
    baseline = json.loads(baseline_path.read_text())
    if baseline.get("calibration_rows_read") != 0 or baseline.get("test_rows_read") != 0:
        raise ValueError("baseline selection reports calibration/test access")
    comparator = str(baseline["frozen_strongest_deployable_comparator"])
    comparator_value = float(baseline["development_source_macro_u0"][comparator])
    per_seed = []
    dev_prediction_sets = [
        [row for row in predictions if row["role"] == "development"]
        for predictions in prediction_sets
    ]
    for seed, rows in zip(EXPECTED_SEEDS, dev_prediction_sets):
        value = source_macro_policy_value(development_rows, choices_from_prediction_rows(rows))
        per_seed.append(
            {"seed": seed, "source_macro_u0": value, "delta_vs_comparator": value - comparator_value}
        )
    ensemble_rows = []
    for aligned in zip(*dev_prediction_sets):
        row_ids = {row["row_id"] for row in aligned}
        if len(row_ids) != 1:
            raise ValueError("seed development prediction alignment failed")
        template = dict(aligned[0])
        template["predicted_u0"] = float(np.mean([row["predicted_u0"] for row in aligned]))
        ensemble_rows.append(template)
    ensemble_choices = choices_from_prediction_rows(ensemble_rows)
    ensemble_value = source_macro_policy_value(development_rows, ensemble_choices)
    oracle_value = source_macro_policy_value(development_rows, _oracle_choices(development_rows))
    base_value = source_macro_policy_value(
        development_rows,
        {str(row["block_id"]): "base_continue" for row in development_rows},
    )
    oracle_gap = oracle_value - comparator_value
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_development_model_selection",
        "scope": "single_primary_policy_single_staleness_mechanism_pilot",
        "selected_model_identity": "ODUR-v1-fixed-pooled-dual-camera-proprio",
        "fit_stage": "train_only_development_readout",
        "seed_artifacts": seed_artifacts,
        "baseline_selection_sha256": sha256_file(baseline_path),
        "frozen_strongest_deployable_comparator": comparator,
        "comparator_development_source_macro_u0": comparator_value,
        "per_seed": per_seed,
        "seeds_beating_comparator": sum(row["delta_vs_comparator"] > 0 for row in per_seed),
        "median_delta_u0": float(np.median([row["delta_vs_comparator"] for row in per_seed])),
        "ensemble_development_source_macro_u0": ensemble_value,
        "ensemble_delta_u0": ensemble_value - comparator_value,
        "base_development_source_macro_u0": base_value,
        "deployable_oracle_development_source_macro_u0": oracle_value,
        "oracle_gap_recovered": (
            (ensemble_value - comparator_value) / oracle_gap if oracle_gap > 0 else None
        ),
        "calibration_rows_read": 0,
        "test_rows_read": 0,
        "next_action": "REFIT_SELECTED_RECIPE_ON_TRAIN_DEVELOPMENT_THEN_CALIBRATE",
    }
    payload["selection_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, action="append", required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite model selection: {args.output}")
    payload = analyze(
        branches=read_jsonl(args.merged_dir / "branches.jsonl"),
        seed_dirs=args.seed_dir,
        baseline_dir=args.baseline_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "next_action": payload["next_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
