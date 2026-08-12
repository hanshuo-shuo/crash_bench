#!/usr/bin/env python3
"""Capture full-feature D0 detector episodes from frozen placement manifests.

The exposed r5/r7 placement design supplies source-disjoint train/calibration
data.  The later fresh r2 design remains development-only.  Every selected
placement is rolled out under on-path, off-path, and no-glass conditions with
the same frozen policy and source state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import (
    GlassPlacement,
    array_sha256,
    read_placement_manifest,
)
from crashbench.predicates import build_any
from crashbench.provenance import repository_provenance, require_checkpoint_revision
from scripts.collect_glass_recovery_pairs import (
    _glass_force,
    _glass_predicate_specs,
    _prime_glass_predicates,
)


HORIZON_ACTIONS = 20
CONDITIONS = ("glass", "offpath", "noglass")
EXPOSED_SPLIT_MAP = {
    "train": "train",
    "validation": "calibration",
    "heldout": "calibration",
}


@dataclass(frozen=True)
class PlannedPlacement:
    cohort: str
    detector_split: str
    placement: GlassPlacement
    manifest_path: Path

    @property
    def key(self) -> str:
        return f"{self.cohort}:{self.placement.placement_id}"


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_capture_plan(
    exposed_manifest: str | Path,
    development_manifest: str | Path,
    *,
    placement_keys: set[str] | None = None,
    max_placements: int | None = None,
) -> tuple[list[PlannedPlacement], dict[str, str]]:
    """Freeze the 8/10/9 exposed/calibration/development source assignment."""

    exposed_path = Path(exposed_manifest).resolve()
    development_path = Path(development_manifest).resolve()
    exposed, _ = read_placement_manifest(exposed_path)
    development, _ = read_placement_manifest(development_path)
    plan = [
        PlannedPlacement(
            cohort="exposed",
            detector_split=EXPOSED_SPLIT_MAP[placement.split],
            placement=placement,
            manifest_path=exposed_path,
        )
        for placement in exposed
    ] + [
        PlannedPlacement(
            cohort="fresh",
            detector_split="development",
            placement=placement,
            manifest_path=development_path,
        )
        for placement in development
    ]
    plan.sort(key=lambda item: (item.cohort, item.placement.placement_id))
    all_keys = {item.key for item in plan}
    if placement_keys is not None:
        unknown = sorted(placement_keys - all_keys)
        if unknown:
            raise ValueError(f"unknown placement keys {unknown}")
        plan = [item for item in plan if item.key in placement_keys]
    if max_placements is not None:
        if int(max_placements) < 1:
            raise ValueError("max_placements must be positive")
        plan = plan[:int(max_placements)]
    if not plan:
        raise ValueError("capture plan is empty")

    source_splits: dict[str, str] = {}
    for item in plan:
        source = item.placement.source_state_sha256
        previous = source_splits.setdefault(source, item.detector_split)
        if previous != item.detector_split:
            raise ValueError(
                f"source {source} leaks across detector splits {previous}/{item.detector_split}"
            )
    return plan, source_splits


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _condition_glasses(placement: GlassPlacement, condition: str) -> list[dict]:
    if condition == "glass":
        return [placement.on_path_glass]
    if condition == "offpath":
        return [placement.off_path_glass]
    if condition == "noglass":
        return []
    raise ValueError(f"unsupported condition {condition!r}")


def _rollout(
    env: LiberoEnv,
    policy,
    source_state: np.ndarray,
    placement: GlassPlacement,
    condition: str,
    *,
    settle_steps: int,
    max_steps: int,
) -> dict:
    glasses = _condition_glasses(placement, condition)
    obs = env.reset_to(source_state, movable_objects=glasses or None)
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(env.dummy_action())
    crash = None
    if glasses:
        crash = build_any(_glass_predicate_specs(glasses))
        _prime_glass_predicates(crash, env.sim_view)
    rows = []
    collision_step = None
    succeeded = False
    peak_force = 0.0
    for step in range(max_steps):
        policy_obs = env.policy_observation(obs, policy.resize_size)
        nominal_action = np.asarray(
            policy.act(policy_obs, placement.instruction), dtype=np.float32
        )
        hidden = policy.last_hidden
        if hidden is None:
            raise RuntimeError("OpenVLA hidden hook returned None")
        row = {
            "t": step,
            "hidden": np.asarray(hidden, dtype=np.float16),
            "robot_state": np.asarray(policy_obs["state"], dtype=np.float32),
            "nominal_action": nominal_action,
            "eef_xyz": np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        }
        obs, _, done, _ = env.step(nominal_action.tolist())
        force = 0.0 if not glasses else _glass_force(env.sim_view, glasses)
        peak_force = max(peak_force, force)
        row["glass_force_after"] = float(force)
        rows.append(row)
        if crash is not None and crash(env.sim_view):
            collision_step = step
            break
        if done:
            succeeded = True
            break
    return {
        "rows": rows,
        "crashed": collision_step is not None,
        "collision_step": collision_step,
        "succeeded": succeeded,
        "peak_glass_force_n": float(peak_force),
    }


def _episode_exclusion(condition: str, result: Mapping) -> str | None:
    if condition == "offpath" and bool(result["crashed"]):
        return "offpath_not_a_clean_control"
    if condition == "glass" and bool(result["crashed"]):
        if int(result["collision_step"]) + 1 < HORIZON_ACTIONS:
            return "onpath_collision_before_T20_anchor"
    return None


def capture(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite non-empty {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    revision = require_checkpoint_revision(args.checkpoint_revision)
    repo = repository_provenance(Path(__file__).resolve().parents[1], require_clean=True)
    declared = os.environ.get("CB_CODE_COMMIT")
    if declared is not None and declared != repo["git_commit"]:
        raise SystemExit("CB_CODE_COMMIT does not match checked-out source")
    placement_keys = None if not args.placement_key else set(args.placement_key)
    plan, source_splits = build_capture_plan(
        args.exposed_placements,
        args.development_placements,
        placement_keys=placement_keys,
        max_placements=args.max_placements,
    )

    from crashbench.policies import OpenVLAPolicy

    policy = OpenVLAPolicy(
        pretrained_checkpoint=args.checkpoint,
        checkpoint_revision=revision,
        unnorm_key=args.unnorm_key,
        center_crop=True,
        capture_hidden=True,
    )
    envs: dict[tuple[str, int], LiberoEnv] = {}
    hidden_rows, robot_rows, action_rows, metadata_rows = [], [], [], []
    episode_summaries = []
    excluded_episodes = []
    for item in plan:
        placement = item.placement
        env_key = (str(placement.task_suite), int(placement.task_id))
        if env_key not in envs:
            envs[env_key] = LiberoEnv(*env_key)
        env = envs[env_key]
        source_path = item.manifest_path.parent / placement.source_state_path
        source_state = np.load(source_path, allow_pickle=False)
        if array_sha256(source_state) != placement.source_state_sha256:
            raise ValueError(f"{item.key}: source-state hash mismatch")
        for rollout_seed in args.rollout_seeds:
            for condition in CONDITIONS:
                _seed_everything(rollout_seed)
                env.seed(rollout_seed)
                policy.reset()
                result = _rollout(
                    env,
                    policy,
                    source_state,
                    placement,
                    condition,
                    settle_steps=args.settle_steps,
                    max_steps=args.max_steps,
                )
                episode_id = f"{item.key}:{condition}:seed{rollout_seed}"
                exclusion = _episode_exclusion(condition, result)
                episode = {
                    "episode_id": episode_id,
                    "placement_key": item.key,
                    "placement_id": placement.placement_id,
                    "cohort": item.cohort,
                    "detector_split": item.detector_split,
                    "source_state_sha256": placement.source_state_sha256,
                    "condition": condition,
                    "rollout_seed": int(rollout_seed),
                    "steps": len(result["rows"]),
                    "crashed": bool(result["crashed"]),
                    "collision_step": result["collision_step"],
                    "succeeded": bool(result["succeeded"]),
                    "peak_glass_force_n": result["peak_glass_force_n"],
                    "detector_usable": exclusion is None,
                    "exclusion_reason": exclusion,
                }
                episode_summaries.append(episode)
                if exclusion is not None:
                    excluded_episodes.append(episode)
                    print(json.dumps(episode, sort_keys=True), flush=True)
                    continue
                for row in result["rows"]:
                    hidden_rows.append(row["hidden"])
                    robot_rows.append(row["robot_state"])
                    action_rows.append(row["nominal_action"])
                    metadata_rows.append({
                        "schema_version": 2,
                        "episode_id": episode_id,
                        "placement_key": item.key,
                        "placement_id": placement.placement_id,
                        "cohort": item.cohort,
                        "detector_split": item.detector_split,
                        "source_state_sha256": placement.source_state_sha256,
                        "condition": condition,
                        "cond": condition,
                        "rollout_seed": int(rollout_seed),
                        "t": int(row["t"]),
                        "crashed_episode": bool(result["crashed"]),
                        "crash_step": (
                            -1 if result["collision_step"] is None
                            else int(result["collision_step"])
                        ),
                        "time_to_catastrophe_actions": (
                            None if result["collision_step"] is None
                            else int(result["collision_step"] - row["t"] + 1)
                        ),
                        "task_succeeded": bool(result["succeeded"]),
                        "glass_force_after": row["glass_force_after"],
                        "eef_x": float(row["eef_xyz"][0]),
                        "eef_y": float(row["eef_xyz"][1]),
                        "eef_z": float(row["eef_xyz"][2]),
                    })
                print(json.dumps(episode, sort_keys=True), flush=True)

    if not metadata_rows:
        raise RuntimeError("capture produced no detector-usable frames")
    capture_path = output / "hidden.npz"
    metadata_path = output / "meta.json"
    np.savez_compressed(
        capture_path,
        hidden=np.asarray(hidden_rows, dtype=np.float16),
        robot_state=np.asarray(robot_rows, dtype=np.float32),
        nominal_action=np.asarray(action_rows, dtype=np.float32),
    )
    metadata_path.write_text(json.dumps(metadata_rows) + "\n")
    source_split_payload = {
        "schema_version": 1,
        "kind": "glass_detector_source_split",
        "horizon_actions": HORIZON_ACTIONS,
        "assignment": {
            "method": "frozen_E15_provenance_groups",
            "exposed_original_train": "train",
            "exposed_original_validation_and_heldout": "calibration",
            "fresh_r2_all_splits": "development",
            "uses_current_capture_outcomes_or_scores": False,
        },
        "source_splits": dict(sorted(source_splits.items())),
    }
    split_path = output / "source_split.json"
    split_path.write_text(json.dumps(source_split_payload, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "kind": "glass_detector_placement_capture",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repo,
        "checkpoint_identity": policy.checkpoint_identity,
        "checkpoint_revision": revision,
        "unnorm_key": args.unnorm_key,
        "horizon_actions": HORIZON_ACTIONS,
        "settle_steps": args.settle_steps,
        "max_steps": args.max_steps,
        "rollout_seeds": args.rollout_seeds,
        "inputs": {
            "exposed_placements": str(Path(args.exposed_placements).resolve()),
            "exposed_placements_sha256": _file_sha256(args.exposed_placements),
            "development_placements": str(Path(args.development_placements).resolve()),
            "development_placements_sha256": _file_sha256(args.development_placements),
        },
        "placements": len(plan),
        "source_states": len(source_splits),
        "source_states_by_split": dict(Counter(source_splits.values())),
        "episodes": len(episode_summaries),
        "usable_episodes": sum(row["detector_usable"] for row in episode_summaries),
        "excluded_episodes": excluded_episodes,
        "episode_outcomes": episode_summaries,
        "frames": len(metadata_rows),
        "artifacts": {
            "capture": capture_path.name,
            "metadata": metadata_path.name,
            "source_split": split_path.name,
        },
    }
    manifest_path = output / "capture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposed-placements", required=True)
    parser.add_argument("--development-placements", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial"
    )
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--unnorm-key", default="libero_spatial")
    parser.add_argument("--rollout-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--placement-key", action="append")
    parser.add_argument("--max-placements", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.settle_steps < 0 or args.max_steps < HORIZON_ACTIONS:
        raise SystemExit(f"max steps must be at least {HORIZON_ACTIONS}")
    capture(args)


if __name__ == "__main__":
    main()
