#!/usr/bin/env python
"""Phase 2-④ grasp_instability (cat-5) reconnaissance.

A grasp-instability pre-crash state = the target object is ALREADY in the gripper and lifted,
but held precariously (tilted / near the finger edge), so continuing the nominal policy drops
it. To author that precisely we first HARVEST a real, secure grasped-lift state from OpenVLA
itself (so the hold is natural, not hand-faked), then later perturb the held object's pose.

This recon, on high-competence libero-spatial task 0 (OpenVLA reliably picks+places):
  1. roll out OpenVLA nominally; detect the first step where the target bowl is LIFTED
     (bowl z risen above its resting z) -> snapshot the full flat sim state at that step
  2. record the bowl's free-joint qpos index (xyz+quat) + its pose, the gripper qpos, rest_z,
     and the lift step, so the author script can perturb the held bowl by a known amount
  3. render the grasped-lift frame

Outputs -> results/grasp/{snapshot_state.npy, recon.json, grasped_lift.png}
Run on a GPU node (MUJOCO_GL=egl). Uses the libero-spatial checkpoint.
"""

from __future__ import annotations

import os, json

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import imageio
import mujoco

from crashbench.envs import LiberoEnv

SUITE, TASK = "libero_spatial", 0
TARGET = "akita_black_bowl_1"
SETTLE = 10
LIFT_MARGIN = 0.04          # m above resting z to call it "lifted"
OUT = "results/grasp"
os.makedirs(OUT, exist_ok=True)


def live(env):
    sim = env.env.sim
    return getattr(sim.model, "_model", sim.model), getattr(sim.data, "_data", sim.data)


def flat_state(env):
    """Full flattened sim state [time, qpos, qvel] — same layout as LIBERO init states."""
    _, data = live(env)
    return np.concatenate([[data.time], data.qpos.copy(), data.qvel.copy()])


def joints_named(env, kw):
    model, data = live(env)
    out = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if kw in name.lower():
            out.append((name, int(model.jnt_qposadr[j]), int(model.jnt_type[j])))
    return out


def main():
    from crashbench.policies import OpenVLAPolicy
    env = LiberoEnv(SUITE, TASK)
    base = np.asarray(env.default_init_states()[0])
    sv = env.sim_view
    policy = OpenVLAPolicy(unnorm_key="libero_spatial")
    print(f"task: {env.task_description!r}  init_len={len(base)}")

    obs = env.reset_to(base)
    for _ in range(SETTLE):
        obs, _, _, _ = env.step(env.dummy_action())
    rest_z = float(np.asarray(obs[f"{TARGET}_pos"])[2])
    nq = live(env)[0].nq
    bowl_joints = joints_named(env, "akita_black_bowl_1")
    grip_joints = joints_named(env, "gripper")
    print(f"rest_z(bowl)={rest_z:.3f}  nq={nq}  bowl_joints={bowl_joints}  grip_joints={grip_joints}")

    snap, lift_step, lifted = None, None, False
    success = False
    for t in range(220):
        o = env.policy_observation(obs, policy.resize_size)
        a = policy.act(o, env.task_description)
        obs, _, done, _ = env.step(a.tolist() if hasattr(a, "tolist") else a)
        bz = float(np.asarray(obs[f"{TARGET}_pos"])[2])
        grasped = sv.is_grasped(TARGET)
        if (not lifted) and grasped and bz > rest_z + LIFT_MARGIN:
            lifted = True
            lift_step = t
            snap = flat_state(env)
            bpos = np.asarray(obs[f"{TARGET}_pos"]).copy()
            imageio.imwrite(f"{OUT}/grasped_lift.png", env.render(obs, 256))
            print(f"  LIFT @step {t}: bowl_z={bz:.3f} grasped={grasped}")
        if done:
            success = True
            break
    print(f"nominal success={success} steps={t} lift_step={lift_step}")

    if snap is None:
        raise SystemExit("never detected a grasped-lift; inspect is_grasped / LIFT_MARGIN")

    # bowl free-joint -> indices into the flat state (layout [time, qpos, qvel])
    bname, badr, btype = bowl_joints[0]
    i0 = 1 + badr
    rec = {
        "suite": SUITE, "task_id": TASK, "target": TARGET,
        "instruction": env.task_description, "nominal_success": success,
        "lift_step": lift_step, "rest_z": round(rest_z, 4),
        "snapshot_state_len": int(len(snap)),
        "bowl_joint": bname, "bowl_qposadr": badr, "bowl_init_idx": i0,
        "bowl_xyz_at_snapshot": [round(float(x), 4) for x in snap[i0:i0 + 3]],
        "bowl_quat_at_snapshot": [round(float(x), 4) for x in snap[i0 + 3:i0 + 7]],
        "bowl_z_at_snapshot": round(float(np.asarray(bpos)[2]), 4),
        "gripper_joints": grip_joints,
        "gripper_qpos_at_snapshot": [
            round(float(snap[1 + adr]), 4) for (_, adr, _) in grip_joints],
    }
    np.save(f"{OUT}/snapshot_state.npy", snap)
    json.dump(rec, open(f"{OUT}/recon.json", "w"), indent=2)
    print(f"\nwrote {OUT}/snapshot_state.npy + recon.json")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
