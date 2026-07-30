#!/usr/bin/env python
"""Replay the recorded oracle-stop episodes into an auditable OpenVLA dataset.

The source JSON already contains the proposed and oracle-filtered actions.  This
script replays only those recorded actions in the simulator to recover the exact
pre-action camera observations; it never queries OpenVLA and never labels held-out
scenarios for training.  Each manifest row stores both LIBERO execution convention
and the normalized OpenVLA action tokens used by the fine-tuning script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from crashbench.envs import LiberoEnv
from crashbench.oracle_recovery import (
    DEFAULT_HELDOUT_SCENARIOS,
    DEFAULT_TRAIN_SCENARIOS,
    env_action_to_openvla_raw,
    normalize_openvla_action,
    validate_scenario_split,
)
from crashbench.scenario import Scenario, scenario_fingerprint


SETTLE_STEPS = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _action_stats(checkpoint: str, unnorm_key: str) -> dict:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    stats = config.norm_stats
    key = unnorm_key if unnorm_key in stats else f"{unnorm_key}_no_noops"
    if key not in stats:
        raise KeyError(f"{unnorm_key!r} not present in checkpoint norm_stats: {sorted(stats)}")
    return {"key": key, **stats[key]["action"]}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def build(args: argparse.Namespace) -> None:
    source_path = Path(args.source)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite non-empty {out}; pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    source = json.loads(source_path.read_text())
    episodes = [row for row in source["episodes"] if row["condition"] == "oracle_stop"]
    if not episodes:
        raise SystemExit("source JSON has no oracle_stop episodes")
    scenario_root = Path(args.scenarios)
    scenarios = {
        path.parent.name: Scenario.load(path.parent)
        for path in sorted(scenario_root.glob("*/scenario.json"))
    }
    train_ids, heldout_ids = validate_scenario_split(
        scenarios, args.train_scenarios, args.heldout_scenarios
    )
    selected = set(train_ids) | set(heldout_ids)
    episodes = [row for row in episodes if row["scenario_id"] in selected]
    expected = {(scenario_id, rep) for scenario_id in selected for rep in range(args.repeats)}
    actual = {(row["scenario_id"], int(row["rep"])) for row in episodes if row["rep"] < args.repeats}
    if actual != expected:
        raise ValueError(f"oracle coverage mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")

    stats = _action_stats(args.checkpoint, args.unnorm_key)
    action_stats = {key: stats[key] for key in ("q01", "q99", "mask")}
    env: LiberoEnv | None = None
    rows_by_split: dict[str, list[dict]] = {"train": [], "heldout": []}
    replay = []

    for episode in sorted(episodes, key=lambda x: (x["scenario_id"], int(x["rep"]))):
        if int(episode["rep"]) >= args.repeats:
            continue
        sc = scenarios[episode["scenario_id"]]
        split = "train" if sc.id in train_ids else "heldout"
        if env is None or (env.task_suite, env.task_id) != (sc.task_suite, sc.task_id):
            env = LiberoEnv(sc.task_suite, sc.task_id)
        obs = env.reset_to(sc.init_state, obstacles=sc.obstacles)
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = env.step(env.dummy_action())

        max_wall_force = 0.0
        for step in episode["steps"]:
            observation = env.policy_observation(obs, args.resize_size)
            image = np.asarray(observation["full_image"], dtype=np.uint8)
            image_rel = Path("images") / split / f"{sc.id}__rep{episode['rep']}__t{step['t']:03d}.png"
            image_path = out / image_rel
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(image_path, optimize=True)

            env_action = np.asarray(step["executed_action"], dtype=np.float32)
            raw_action = env_action_to_openvla_raw(env_action)
            normalized = normalize_openvla_action(raw_action, action_stats)
            row = {
                "image": image_rel.as_posix(),
                "instruction": sc.instruction.lower(),
                "env_action": env_action.round(7).tolist(),
                "openvla_raw_action": raw_action.round(7).tolist(),
                "normalized_action": normalized.round(7).tolist(),
                "intervened": bool(step["intervened"]),
                "scenario_id": sc.id,
                "scenario_fingerprint": scenario_fingerprint(scenario_root / sc.id),
                "rep": int(episode["rep"]),
                "t": int(step["t"]),
                "oracle": step.get("oracle", {}),
            }
            rows_by_split[split].append(row)
            obs, _, _, _ = env.step(env_action.tolist())
            max_wall_force = max(
                max_wall_force,
                float(env.sim_view.max_contact_force(env.sim_view._robot_bodies(), against=["crash_wall"])),
            )
        replay.append({
            "scenario_id": sc.id,
            "rep": int(episode["rep"]),
            "split": split,
            "n_steps": len(episode["steps"]),
            "max_wall_force_n": round(max_wall_force, 4),
        })

    for split, rows in rows_by_split.items():
        if not rows:
            raise ValueError(f"empty {split} dataset")
        _write_jsonl(out / f"{split}.jsonl", rows)
    train_stop = sum(row["intervened"] for row in rows_by_split["train"])
    train_go = len(rows_by_split["train"]) - train_stop
    if not train_stop or not train_go:
        raise ValueError("training split must contain both oracle stops and non-stop reference actions")
    if any(row["max_wall_force_n"] >= args.crash_threshold_n for row in replay):
        raise ValueError("recorded oracle replay crossed the declared crash-force threshold")

    metadata = {
        "schema_version": 1,
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "source_checkpoint": source["config"].get("checkpoint"),
        "training_base_checkpoint": args.checkpoint,
        "unnorm_key": stats.pop("key"),
        "action_stats": stats,
        "conversion": "env gripper -> raw=(1-env)/2; q01/q99 BOUNDS_Q99 normalization on mask=true dims",
        "train_scenarios": list(train_ids),
        "heldout_scenarios": list(heldout_ids),
        "counts": {
            split: {
                "samples": len(rows),
                "oracle_stop": sum(row["intervened"] for row in rows),
                "reference_action": sum(not row["intervened"] for row in rows),
            }
            for split, rows in rows_by_split.items()
        },
        "replays": replay,
        "heldout_policy": "heldout samples are exported for audit only and are never read by the trainer",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"wrote oracle dataset to {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/safety_baselines_nonvlm.json")
    ap.add_argument("--scenarios", default="scenarios")
    ap.add_argument("--out", default="results/oracle_recovery/dataset_v1")
    ap.add_argument("--checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial")
    ap.add_argument("--unnorm-key", default="libero_spatial")
    ap.add_argument("--train-scenarios", nargs="+", default=list(DEFAULT_TRAIN_SCENARIOS))
    ap.add_argument("--heldout-scenarios", nargs="+", default=list(DEFAULT_HELDOUT_SCENARIOS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--resize-size", type=int, default=224)
    ap.add_argument("--crash-threshold-n", type=float, default=75.0)
    ap.add_argument("--overwrite", action="store_true")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
