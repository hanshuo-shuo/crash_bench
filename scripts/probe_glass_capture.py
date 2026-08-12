#!/usr/bin/env python
"""Self-report probe — capture OpenVLA hidden states on Category 2 (glass), three arms.

R4 fit the "I will crash" probe on the static WALL (env_collision). This extends the same
rigorous self-report test to the fragile-object hazard (object_collision, §6d) so the probe
claim is not wall-specific. Data collection only; `scripts/probe_glass_analysis.py` trains the
within-glass probe AND tests cross-hazard transfer of the frozen wall probe.

For the saved glass scenarios (scenarios_glass/) we run the FROZEN policy closed-loop under
three matched conditions — same init state, same task, only the glass changes:

    glass   : glass ON the reach path (treatment)  -> struck, crash by contact_force>=25N
    offpath : glass pushed OFF the path (control)   -> glass VISIBLE but not hit  [confound]
    noglass : glass dropped (movable_objects=None)  -> the normal reach           [negative]

Per step we log the LM's last hidden state h_t + full nominal action + 8-D robot state +
robot-vs-glass contact force + eef position + whether/when the episode ultimately crashes.
Output (mirrors results/selfreport/ schema while adding deployable D0 features):
    results/selfreport_glass/hidden.npz   (hidden, robot_state, nominal_action)
    results/selfreport_glass/meta.json    (per-row metadata, aligned to H)

Run on a GPU node:  cd ~/crash_bench/setup && sbatch probe_glass.sbatch
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from crashbench.scenario import Scenario
from crashbench.envs import LiberoEnv
from crashbench.envs.libero_adapter import ROBOT_CONTACT_BODIES
from crashbench.glass_recovery_data import array_sha256
from crashbench.policies import OpenVLAPolicy
from crashbench.predicates import build_any
from crashbench.provenance import repository_provenance, require_checkpoint_revision

SETTLE = 10
GLASS_BODY = "glass_1"


def glass_force(sim):
    return sim.max_contact_force(list(ROBOT_CONTACT_BODIES), against=[GLASS_BODY])


def rollout(env, policy, sc, drop_glass: bool):
    """One closed-loop episode with per-step hidden capture. drop_glass=True runs the nominal
    reach (no injected cup). Returns list of step-dicts + the crash step (or -1)."""
    movable = None if drop_glass else (sc.movable_objects or None)
    gx = (sc.movable_objects[0]["pos"][0] if (sc.movable_objects and not drop_glass) else float("nan"))
    obs = env.reset_to(sc.init_state, movable_objects=movable)
    sim = env.sim_view
    # the glass crash predicates reference glass_1; with no glass there is nothing to hit
    crash_pred = None if drop_glass else build_any(sc.crash_predicates)

    for _ in range(SETTLE):                                   # settle (mirror run_episode)
        obs, _, _, _ = env.step(env.dummy_action())

    steps, crash_step = [], -1
    for t in range(sc.max_steps):
        observation = env.policy_observation(obs, policy.resize_size)
        action = policy.act(observation, sc.instruction)
        h = policy.last_hidden
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        gf = 0.0 if drop_glass else glass_force(sim)
        steps.append({
            "t": t,
            "hidden": None if h is None else h.astype(np.float16),
            "robot_state": np.asarray(observation["state"], dtype=np.float32),
            "nominal_action": np.asarray(action, dtype=np.float32),
            "act_xyz_norm": float(np.linalg.norm(np.asarray(action[:3], dtype=np.float32))),
            "gripper": float(action[6]),
            "glass_force": float(gf),
            "eef_x": float(eef[0]), "eef_y": float(eef[1]), "eef_z": float(eef[2]),
            "glass_x": float(gx),
        })
        obs, _, _, _ = env.step(action.tolist())
        if crash_pred is not None and crash_pred(sim):
            crash_step = t
            break
    return steps, crash_step


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/selfreport_glass")
    parser.add_argument("--scenarios", default="scenarios_glass")
    parser.add_argument(
        "--checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial"
    )
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--unnorm-key", default="libero_spatial")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--rollout-seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    checkpoint_revision = require_checkpoint_revision(args.checkpoint_revision)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite non-empty {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    repo = repository_provenance(Path(__file__).resolve().parents[1], require_clean=True)

    # (cond, regime-filter, drop_glass): which saved scenarios, and whether to keep the cup
    conditions = [
        ("glass",   "treatment", False),   # cup on the path
        ("offpath", "control",   False),   # cup off the path (confound)
        ("noglass", "treatment", True),    # cup dropped -> nominal reach (negative)
    ]
    all_scn = [Scenario.load(os.path.dirname(p)) for p in sorted(
        glob.glob(str(Path(args.scenarios) / "*" / "scenario.json"))
    )]
    if not all_scn:
        raise SystemExit(f"no scenarios found under {args.scenarios}")
    by_regime = {"treatment": [], "control": []}
    for sc in all_scn:
        by_regime.get(sc.metadata.get("regime", "?"), []).append(sc)

    policy = OpenVLAPolicy(
        pretrained_checkpoint=args.checkpoint,
        checkpoint_revision=checkpoint_revision,
        unnorm_key=args.unnorm_key,
        center_crop=True,
        capture_hidden=True,
    )
    env = LiberoEnv(args.suite, args.task_id)

    H, robot_states, nominal_actions, meta = [], [], [], []
    for cond, regime, drop in conditions:
        scns = by_regime.get(regime, [])
        print(
            f"\n=== condition '{cond}' : {len(scns)} scenarios x "
            f"{len(args.rollout_seeds)} seeds (drop_glass={drop}) ==="
        )
        for sc in scns:
            source_state_sha256 = array_sha256(np.asarray(sc.init_state))
            for rep, rollout_seed in enumerate(args.rollout_seeds):
                env.seed(rollout_seed)
                policy.reset()
                steps, cstep = rollout(env, policy, sc, drop_glass=drop)
                crashed = cstep >= 0
                episode_id = f"{sc.id}::{cond}::seed{rollout_seed}"
                for s in steps:
                    if s["hidden"] is None:
                        continue
                    H.append(s["hidden"])
                    robot_states.append(s["robot_state"])
                    nominal_actions.append(s["nominal_action"])
                    meta.append({
                        "schema_version": 2,
                        "episode_id": episode_id,
                        "source_state_sha256": source_state_sha256,
                        "condition": cond,
                        "cond": cond, "scenario_id": sc.id, "rep": rep,
                        "rollout_seed": rollout_seed, "t": s["t"],
                        "regime": sc.metadata.get("regime", "?"),
                        "tag": sc.metadata.get("tag", "?"),
                        "clearance": sc.metadata.get("clearance_to_path"),
                        "crashed_episode": bool(crashed),
                        "crash_step": int(cstep),
                        "steps_to_crash": int(cstep - s["t"]) if crashed else -1,
                        "time_to_catastrophe_actions": (
                            int(cstep - s["t"] + 1) if crashed else None
                        ),
                        "act_xyz_norm": s["act_xyz_norm"], "gripper": s["gripper"],
                        "glass_force": s["glass_force"],
                        "eef_x": s["eef_x"], "eef_y": s["eef_y"], "eef_z": s["eef_z"],
                        "glass_x": s["glass_x"],
                    })
                print(f"  {sc.id:48s} rep{rep} crash={'Y' if crashed else 'n'} "
                      f"crash_step={cstep:3d} steps={len(steps):3d}")

    Harr = np.asarray(H, dtype=np.float16)
    robot_arr = np.asarray(robot_states, dtype=np.float32)
    action_arr = np.asarray(nominal_actions, dtype=np.float32)
    capture_path = output / "hidden.npz"
    metadata_path = output / "meta.json"
    np.savez_compressed(
        capture_path,
        hidden=Harr,
        robot_state=robot_arr,
        nominal_action=action_arr,
    )
    metadata_path.write_text(json.dumps(meta))
    manifest = {
        "schema_version": 1,
        "kind": "glass_detector_d0_capture",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repo,
        "checkpoint_identity": policy.checkpoint_identity,
        "checkpoint_revision": checkpoint_revision,
        "unnorm_key": args.unnorm_key,
        "suite": args.suite,
        "task_id": args.task_id,
        "rollout_seeds": args.rollout_seeds,
        "settle_steps": SETTLE,
        "scenarios_root": str(Path(args.scenarios).resolve()),
        "frames": len(meta),
        "source_states": len({row["source_state_sha256"] for row in meta}),
        "episodes": len({row["episode_id"] for row in meta}),
        "capture_file": capture_path.name,
        "metadata_file": metadata_path.name,
    }
    (output / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"\nwrote {capture_path} H={Harr.shape} and {metadata_path} rows={len(meta)}"
    )
    tally = {}
    for m in meta:
        tally[m["cond"]] = tally.get(m["cond"], 0) + 1
    print("frames by condition:", tally)


if __name__ == "__main__":
    main()
