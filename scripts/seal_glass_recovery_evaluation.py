#!/usr/bin/env python3
"""Seal one accepted split and trained checkpoints for E15 evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from crashbench.glass_recovery_data import (
    canonical_sha256,
    read_placement_manifest,
    read_trajectory_manifest,
    validate_primary_pair,
)
from crashbench.glass_recovery_model import GlassRecoveryNetwork
from scripts.eval_glass_recovery import (
    EVALUATION_MODES,
    MAIN_CONDITIONS,
    file_sha256,
)


def _relative(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def _artifact_key(path: Path, roots: tuple[Path, ...]) -> str:
    for root in roots:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return path.resolve().as_posix()


def seal(args: argparse.Namespace) -> dict:
    placement_path = Path(args.placements).resolve()
    dataset_root = Path(args.dataset).resolve()
    trajectory_path = dataset_root / f"{args.split}.jsonl"
    collection_path = dataset_root / "collection_summary.json"
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)

    placements, _ = read_placement_manifest(placement_path)
    placements_by_id = {row.placement_id: row for row in placements}
    records = read_trajectory_manifest(trajectory_path)
    groups: dict[str, list] = {}
    for record in records:
        groups.setdefault(record.pair_id, []).append(record)
    selected_ids = sorted(groups) if args.pair_ids is None else list(args.pair_ids)
    if len(selected_ids) != len(set(selected_ids)) or not selected_ids:
        raise SystemExit("evaluation pair IDs must be unique and nonempty")

    collection = json.loads(collection_path.read_text())
    primary = collection["metadata"]["primary_protocol"]
    primary_sha = canonical_sha256(primary)
    if collection["metadata"]["primary_protocol_sha256"] != primary_sha:
        raise SystemExit("collection primary protocol hash is inconsistent")

    pair_rows = []
    file_hashes: dict[str, str] = {}
    roots = (placement_path.parent, dataset_root)
    for pair_id in selected_ids:
        group = groups.get(pair_id)
        if group is None:
            raise SystemExit(f"pair {pair_id} is absent from {trajectory_path}")
        validate_primary_pair(group)
        placement = placements_by_id[group[0].placement_id]
        if placement.split != args.split:
            raise SystemExit(f"pair {pair_id} is outside split {args.split}")
        pair_rows.append({
            "pair_id": pair_id,
            "placement_id": placement.placement_id,
            "split": placement.split,
            "task_suite": placement.task_suite,
            "task_id": placement.task_id,
            "family": placement.metadata["geometry_family"],
            "source_state_sha256": placement.source_state_sha256,
        })
        arrays = [(dataset_root / record.arrays_path).resolve() for record in group]
        pair_root = arrays[0].parent
        required = [
            (placement_path.parent / placement.source_state_path).resolve(),
            *arrays,
            pair_root / "precrash_onpath_state.npy",
            pair_root / "offpath_start_state.npy",
            pair_root / "matched_robot_state.npy",
            pair_root / "controller_state.npz",
        ]
        for path in required:
            if not path.is_file():
                raise SystemExit(f"missing accepted artifact {path}")
            file_hashes[_artifact_key(path, roots)] = file_sha256(path)

    checkpoint_hashes: dict[str, str] = {}
    for value in args.checkpoint:
        checkpoint = Path(value).resolve()
        _, metadata = GlassRecoveryNetwork.load_checkpoint(checkpoint, map_location="cpu")
        if metadata.get("protocol_sha256") != primary_sha:
            raise SystemExit(f"checkpoint protocol differs: {checkpoint}")
        seed = str(metadata["seed"])
        if seed in checkpoint_hashes:
            raise SystemExit(f"duplicate checkpoint training seed {seed}")
        checkpoint_hashes[seed] = file_sha256(checkpoint)

    evaluation = {
        "name": "glass_recovery_eval_p0d_v1",
        "evaluation_modes": list(EVALUATION_MODES),
        "conditions": list(MAIN_CONDITIONS),
        "rollout_seeds": list(args.rollout_seeds),
        "max_steps": args.max_steps,
        "checkpoint_sha256_by_training_seed": checkpoint_hashes,
        "cohort_contract": {
            "split": args.split,
            "pair_ids": selected_ids,
            "placement_manifest_sha256": file_sha256(placement_path),
            "trajectory_manifest_sha256": file_sha256(trajectory_path),
        },
    }
    evaluation_sha = canonical_sha256(evaluation)
    protocol = {
        "schema_version": 1,
        "kind": "glass_recovery_evaluation_protocol",
        "primary_protocol": primary,
        "primary_protocol_sha256": primary_sha,
        "evaluation_protocol": evaluation,
        "evaluation_protocol_sha256": evaluation_sha,
    }
    cohort = {
        "schema_version": 1,
        "kind": "glass_recovery_accepted_evaluation_cohort",
        "split": args.split,
        "placement_manifest_sha256": file_sha256(placement_path),
        "trajectory_manifest_sha256": file_sha256(trajectory_path),
        "primary_protocol_sha256": primary_sha,
        "evaluation_protocol_sha256": evaluation_sha,
        "pairs": pair_rows,
        "file_sha256": file_hashes,
    }
    if args.split == "heldout":
        cohort["reference_manifests"] = {
            split: {
                "path": _relative(dataset_root / f"{split}.jsonl", output),
                "sha256": file_sha256(dataset_root / f"{split}.jsonl"),
            }
            for split in ("train", "validation")
        }

    protocol_path = output / "evaluation_protocol.json"
    cohort_path = output / f"{args.split}_cohort.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    cohort_path.write_text(json.dumps(cohort, indent=2, sort_keys=True) + "\n")
    result = {
        "split": args.split,
        "pair_ids": selected_ids,
        "source_states": len({row["source_state_sha256"] for row in pair_rows}),
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "cohort": str(cohort_path),
        "cohort_sha256": file_sha256(cohort_path),
        "primary_protocol_sha256": primary_sha,
        "evaluation_protocol_sha256": evaluation_sha,
        "checkpoint_sha256_by_training_seed": checkpoint_hashes,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("train", "validation", "heldout"), required=True)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--pair-ids", nargs="+")
    parser.add_argument("--rollout-seeds", type=int, nargs="+", default=[101])
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.rollout_seeds) != len(set(args.rollout_seeds)) or args.max_steps < 1:
        raise SystemExit("rollout seeds must be unique and max steps positive")
    seal(args)


if __name__ == "__main__":
    main()
