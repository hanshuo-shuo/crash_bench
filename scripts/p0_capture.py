#!/usr/bin/env python3
"""Capture the provenance-complete P0 wall/probe dataset.

The script deliberately refuses underpowered or overlapping train/calibration/held-out
designs.  Every rollout gets an independent seed and its own lossless float32 activation
file, full robot/action state arrays, scoped force trace, and episode JSON.  Each scenario
also gets a no-wall nominal full-arm swept-volume measurement fixed before treatment data
are analysed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.corridor import ClearanceBins, DEFAULT_CORRIDOR_BODIES, swept_volume_signed_distance
from crashbench.envs import LiberoEnv
from crashbench.envs.libero_adapter import ROBOT_CONTACT_BODIES
from crashbench.p0 import expand_runs, load_config, validate_design
from crashbench.policies import OpenVLAPolicy
from crashbench.predicates import build_predicate
from crashbench.provenance import repository_provenance, runtime_provenance, write_json_exclusive
from crashbench.scenario import Scenario, scenario_fingerprint


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int, deterministic_torch: bool) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch_info = {"available": False, "deterministic_algorithms": False}
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic_torch, warn_only=True)
        torch_info = {
            "available": True,
            "deterministic_algorithms": bool(deterministic_torch),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        }
    except ImportError:
        pass
    return {"python_seed": seed, "numpy_seed": seed, "torch_seed": seed, **torch_info}


def select_wall(sc: Scenario) -> dict:
    walls = [o for o in (sc.obstacles or []) if o.get("type", "box") == "box"]
    for wall in walls:
        if wall.get("name") == "crash_wall":
            return wall
    if len(walls) != 1:
        raise ValueError(f"{sc.id}: expected one box obstacle or one named crash_wall")
    return walls[0]


def nominal_swept_volume(env, policy, sc: Scenario, bodies: list[str], seed: int,
                         settle_steps: int, *, require_success: bool = False) -> list[dict]:
    seed_everything(seed, deterministic_torch=False)
    env.seed(seed)
    obs = env.reset_to(sc.init_state, obstacles=None, movable_objects=sc.movable_objects)
    if hasattr(policy, "reset"):
        policy.reset()
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(env.dummy_action())
    sampled = []
    previous = {}
    succeeded = False
    for step in range(sc.max_steps):
        rows = env.sim_view.robot_geom_aabbs(bodies)
        for row in rows:
            row["step"] = step
            row["sample_kind"] = "configuration"
            key = (row["body"], row["geom"])
            if key in previous:
                prior = previous[key]
                sampled.append({
                    "body": row["body"], "geom": row["geom"], "step": step,
                    "sample_kind": "between_step_swept_bound",
                    "lo": np.minimum(prior["lo"], row["lo"]).tolist(),
                    "hi": np.maximum(prior["hi"], row["hi"]).tolist(),
                })
            previous[key] = row
            sampled.append(row)
        observation = env.policy_observation(obs, policy.resize_size)
        action = policy.act(observation, sc.instruction)
        obs, _, done, _ = env.step(np.asarray(action).tolist())
        if done:
            succeeded = True
            break
    if not sampled:
        raise RuntimeError(f"{sc.id}: no distal robot collision geoms found")
    if require_success and not succeeded:
        raise RuntimeError(f"{sc.id}: nominal geometry rollout did not complete the task")
    return sampled


def task_phase(sim, target: str) -> int:
    """Generic pick/place phase: 0 approach, 1 carrying, 2 task complete."""
    if sim.libero_done:
        return 2
    return 1 if sim.is_grasped(target) else 0


def run_episode(env, policy, run, episode_seed: int, cfg: dict, corridor_row: dict,
                episode_dir: Path) -> dict:
    sc = run.scenario
    start = utc_now()
    rng = seed_everything(episode_seed, bool(cfg.get("deterministic_torch", False)))
    env.seed(episode_seed)
    obstacles = None if run.condition == "nowall" else sc.obstacles
    obs = env.reset_to(sc.init_state, obstacles=obstacles, movable_objects=sc.movable_objects)
    sim = env.sim_view
    if hasattr(policy, "reset"):
        policy.reset()
    crash_preds = [(spec.type, build_predicate(spec)) for spec in sc.crash_predicates]
    success_pred = build_predicate(sc.success_predicate)
    settle_steps = int(cfg.get("settle_steps", 10))
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(env.dummy_action())

    wall = select_wall(sc)
    target = cfg["task_targets"][f"{sc.task_suite}:{sc.task_id}"]
    if f"{target}_to_robot0_eef_pos" not in obs:
        raise RuntimeError(
            f"{sc.id}: target {target!r} has no relative-pose observation; "
            "task-phase baseline would be invalid"
        )
    hidden, action_rows, eef_rows, joint_qpos, joint_qvel = [], [], [], [], []
    phase_rows, scoped_force, global_force = [], [], []
    joint_names = None
    fired: list[str] = []
    outcome = "timeout"
    event_step = sc.max_steps
    for step in range(sc.max_steps):
        observation = env.policy_observation(obs, policy.resize_size)
        joint = sim.joint_state()
        if joint_names is None:
            joint_names = joint["names"]
        elif joint_names != joint["names"]:
            raise RuntimeError(f"{sc.id}: robot joint ordering changed within episode")
        phase_rows.append(task_phase(sim, target))
        eef_rows.append(np.concatenate([
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
        ]))
        action = np.asarray(policy.act(observation, sc.instruction), dtype=np.float32)
        if policy.last_hidden is None:
            raise RuntimeError(f"{sc.id}: capture_hidden produced no activation at step {step}")
        hidden.append(np.asarray(policy.last_hidden, dtype=np.float32))
        action_rows.append(action)
        joint_qpos.append(np.asarray(joint["qpos"], dtype=np.float32))
        joint_qvel.append(np.asarray(joint["qvel"], dtype=np.float32))

        obs, _, done, _ = env.step(action.tolist())
        scope = 0.0 if run.condition == "nowall" else sim.max_contact_force(
            list(ROBOT_CONTACT_BODIES), against=[wall["name"]])
        scoped_force.append(scope)
        global_force.append(float(sim.peak_force))
        fired = [name for name, pred in crash_preds if pred(sim)]
        if fired:
            outcome, event_step = "crash", step
            break
        if bool(done) or success_pred(sim):
            outcome, event_step = "task_success", step
            break
    if outcome == "timeout":
        stable_threshold = float(cfg.get("stable_force_threshold_N", 1.0))
        if sim.max_contact_force(list(ROBOT_CONTACT_BODIES)) < stable_threshold:
            outcome = "safe_abort"

    n = len(hidden)
    steps_to_impact = np.full(n, -1, dtype=np.int32)
    if outcome == "crash":
        steps_to_impact = event_step - np.arange(n, dtype=np.int32)
    arrays_path = episode_dir / "arrays.npz"
    np.savez(
        arrays_path,
        # float32 is intentional: these are raw hidden states, not lossy float16 summaries.
        hidden=np.asarray(hidden, dtype=np.float32),
        action=np.asarray(action_rows, dtype=np.float32),
        eef_pose=np.asarray(eef_rows, dtype=np.float32),
        joint_qpos=np.asarray(joint_qpos, dtype=np.float32),
        joint_qvel=np.asarray(joint_qvel, dtype=np.float32),
        task_phase=np.asarray(phase_rows, dtype=np.int8),
        scoped_force_N=np.asarray(scoped_force, dtype=np.float32),
        global_peak_force_to_date_N=np.asarray(global_force, dtype=np.float32),
        steps_to_impact=steps_to_impact,
    )
    metadata = {
        "schema_version": 1,
        "started_at_utc": start,
        "finished_at_utc": utc_now(),
        "scenario_id": sc.id,
        "scenario_fingerprint_sha256": scenario_fingerprint(run.path.parent),
        "scenario_json": str(run.path),
        "scenario_metadata": sc.metadata,
        "split": run.split,
        "condition": run.condition,
        "task_suite": sc.task_suite,
        "task_id": sc.task_id,
        "instruction": sc.instruction,
        "target_object": target,
        "episode_seed": episode_seed,
        "rng": rng,
        "repeat": int(episode_dir.name.rsplit("_", 1)[-1]),
        "settle_steps": settle_steps,
        "max_steps": sc.max_steps,
        "n_policy_steps": n,
        "outcome": outcome,
        "crashed": outcome == "crash",
        "task_succeeded": outcome == "task_success",
        "safe_abort": outcome == "safe_abort",
        "event_step": event_step,
        "crash_predicates_fired": fired,
        "scoped_peak_force_N": float(max(scoped_force, default=0.0)),
        "global_peak_force_N": float(sim.peak_force),
        "joint_names": joint_names or [],
        "corridor": corridor_row,
        "arrays": str(arrays_path.relative_to(episode_dir.parents[1])),
    }
    write_json_exclusive(episode_dir / "episode.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs = expand_runs(cfg, ROOT)
    design = validate_design(cfg, runs)
    repo = repository_provenance(ROOT, require_clean=not args.preflight_only)
    report = {"design": design, "repository": repo}
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return

    out = (ROOT / cfg["output_dir"]).resolve()
    try:
        out.relative_to(ROOT / "results" / "p0_runs")
    except ValueError as exc:
        raise ValueError("output_dir must be under ignored results/p0_runs/") from exc
    out.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(out / "run_provenance.json", {
        "schema_version": 1,
        "experiment_id": cfg.get("experiment_id"),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg["_config_path"],
        "repository": repo,
        "runtime": runtime_provenance(),
        "design": design,
        "status": "running",
    })

    seed_everything(int(cfg["base_seed"]), bool(cfg.get("deterministic_torch", False)))
    policy = OpenVLAPolicy(
        pretrained_checkpoint=cfg["checkpoint"],
        checkpoint_revision=cfg["checkpoint_revision"],
        unnorm_key=cfg.get("unnorm_key", "libero_spatial"),
        center_crop=bool(cfg.get("center_crop", True)),
        capture_hidden=True,
    )
    write_json_exclusive(out / "checkpoint.json", policy.checkpoint_identity)
    envs = {
        task: LiberoEnv(*task, model_family="openvla", seed=int(cfg["base_seed"]))
        for task in {(r.scenario.task_suite, r.scenario.task_id) for r in runs}
    }

    bins = ClearanceBins(**cfg.get("corridor_bins_m", {}))
    bodies = list(cfg.get("corridor_bodies", DEFAULT_CORRIDOR_BODIES))
    unique = {}
    for run in runs:
        unique.setdefault(str(run.path), run.scenario)
    corridor_rows = {}
    for index, (path_text, sc) in enumerate(sorted(unique.items())):
        env = envs[(sc.task_suite, sc.task_id)]
        geometry_seed = int(sc.metadata.get("nominal_geometry_seed", int(cfg["geometry_seed"]) + index))
        sampled = nominal_swept_volume(
            env, policy, sc, bodies, geometry_seed,
            int(cfg.get("settle_steps", 10)), require_success=True,
        )
        distance, closest = swept_volume_signed_distance(select_wall(sc), sampled)
        row = {
            "signed_distance_m": distance,
            "predeclared_bin": bins.classify(distance),
            "bin_definition": bins.as_dict(),
            "nominal_geometry_seed": geometry_seed,
            "bodies": bodies,
            **closest,
        }
        corridor_rows[path_text] = row
        print(f"corridor {sc.id}: d={distance:+.4f} m bin={row['predeclared_bin']}", flush=True)
    write_json_exclusive(out / "corridor.json", corridor_rows)
    path_split = {str(run.path): run.split for run in runs}
    bin_counts = {split: defaultdict(int) for split in ("train", "calibration", "heldout")}
    for path_text, row in corridor_rows.items():
        bin_counts[path_split[path_text]][row["predeclared_bin"]] += 1
    missing_bins = {
        split: sorted({"intrusion", "boundary", "clear"} - set(counts))
        for split, counts in bin_counts.items() if set(counts) != {"intrusion", "boundary", "clear"}
    }
    if missing_bins:
        raise RuntimeError(
            "each split must cover all three predeclared corridor intervals before treatment "
            f"rollouts begin; missing={missing_bins}. See {out / 'corridor.json'}"
        )

    episodes_root = out / "episodes"
    episodes_root.mkdir()
    records = []
    episode_counter = 0
    by_key = defaultdict(int)
    for run in runs:
        for repeat in range(run.repeats):
            seed = int(cfg["base_seed"]) + episode_counter
            stem = f"{run.split}_{run.condition}_{run.scenario.id}_{repeat}"
            episode_dir = episodes_root / stem
            episode_dir.mkdir()
            record = run_episode(
                envs[(run.scenario.task_suite, run.scenario.task_id)], policy, run, seed, cfg,
                corridor_rows[str(run.path)], episode_dir,
            )
            records.append(record)
            by_key[(run.split, run.condition, record["outcome"])] += 1
            episode_counter += 1
            print(
                f"{run.split:11s} {run.condition:7s} {run.scenario.id} rep={repeat} "
                f"seed={seed} outcome={record['outcome']} force={record['scoped_peak_force_N']:.1f}N",
                flush=True,
            )
    write_json_exclusive(out / "episodes.json", records)
    write_json_exclusive(out / "complete.json", {
        "status": "complete", "finished_at_utc": utc_now(), "n_episodes": len(records),
        "counts": {"|".join(key): value for key, value in sorted(by_key.items())},
    })
    print(f"complete: {len(records)} episodes in {out}")


if __name__ == "__main__":
    main()
