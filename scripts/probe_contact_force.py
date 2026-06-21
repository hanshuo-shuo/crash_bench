#!/usr/bin/env python
"""Probe: verify we can read real MuJoCo contact forces in the LIBERO env.

Step 1 of the env-collision pivot (README §4.3 cat-1). Before authoring any
"gripper plunges into table / wall" scenario, prove the `contact_force` predicate
can actually SEE a rising force. No new geometry needed: drive the gripper straight
DOWN into the table and watch the force on the gripper bodies climb.

Two readout methods are compared so we bake the robust one into the adapter:
  (A) cfrc_ext  : data.cfrc_ext[body] net external 6-force (world frame), ||force||.
  (B) contacts  : iterate data.contact, mj_contactForce per contact, sum on bodies.

Run on a GPU node (MUJOCO_GL=egl) via setup/probe_contact_force.sbatch.
"""

from __future__ import annotations

import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
from crashbench.envs import LiberoEnv

SUITE, TASK = "libero_spatial", 0


def raw_model(model):
    """Unwrap robosuite MjModel wrapper -> mujoco.MjModel."""
    return getattr(model, "_model", model)


def raw_data(data):
    """Unwrap robosuite MjData wrapper -> mujoco.MjData (it also has ._model, so be explicit)."""
    return getattr(data, "_data", data)


def list_bodies(model, needles):
    out = {}
    for i in range(model.nbody):
        name = model.body(i).name
        if any(n in name.lower() for n in needles):
            out[name] = i
    return out


def all_contacts(rmodel, rdata):
    """Every contact this step as (bodyA_id, bodyB_id, |force|). Force from mj_contactForce."""
    out = []
    res = np.zeros(6, dtype=np.float64)
    for i in range(rdata.ncon):
        c = rdata.contact[i]
        mujoco.mj_contactForce(rmodel, rdata, i, res)
        out.append((int(rmodel.geom_bodyid[c.geom1]),
                    int(rmodel.geom_bodyid[c.geom2]),
                    float(np.linalg.norm(res[:3]))))
    return out


def max_force_on(contacts, body_ids):
    """Max |force| over contacts involving any of body_ids."""
    body_ids = set(body_ids)
    f = 0.0
    for b1, b2, mag in contacts:
        if b1 in body_ids or b2 in body_ids:
            f = max(f, mag)
    return f


def main():
    env = LiberoEnv(SUITE, TASK)
    print(f"task: {env.task_description!r}")

    rs = env.env                      # robosuite/LIBERO env

    # CRITICAL: robosuite REBUILDS the sim (new MjModel/MjData) on every env.reset().
    # So we must re-fetch the raw model/data AFTER each reset/settle, never cache across one.
    def live():
        sim = rs.sim
        return raw_model(sim.model), raw_data(sim.data), sim.model

    rmodel, rdata, model = live()
    print(f"rdata={type(rdata).__name__} has ncon={hasattr(rdata, 'ncon')} "
          f"nbody={model.nbody} ngeom={model.ngeom}")

    grip = list_bodies(model, ["gripper", "finger", "hand"])
    print("gripper/finger/hand bodies:", grip)
    grip_ids = list(grip.values())

    def bname(bid):
        return model.body(bid).name

    ft_id = grip["gripper0_finger_joint1_tip"]   # fingertip body, to see how low fingers reach

    def grip_contacts(rmodel, rdata, cons):
        """contacts involving a gripper body, as 'bodyA|bodyB=F' strings, force-sorted."""
        g = set(grip_ids)
        hits = [(b1, b2, m) for (b1, b2, m) in cons if b1 in g or b2 in g]
        hits.sort(key=lambda c: -c[2])
        return [f"{bname(b1)}|{bname(b2)}={m:.1f}" for b1, b2, m in hits[:3]]

    def settle():
        base = np.asarray(env.default_init_states()[0])
        obs = env.reset_to(base)
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        return obs

    # (1) plunge straight down; re-fetch live raw data AFTER the reset inside settle()
    obs = settle()
    rmodel, rdata, _ = live()
    print(f"\n=== DOWN plunge: action=[0,0,-1,..]  settled eef={np.round(obs['robot0_eef_pos'],3)} ===")
    print(f"{'step':>4} {'eef_z':>7} {'ftip_z':>7} {'maxF_grip':>10}  grip_contacts(top3)")
    for t in range(60):
        obs, _, done, _ = env.step([0, 0, -1, 0, 0, 0, -1])
        cons = all_contacts(rmodel, rdata)
        fg = max_force_on(cons, grip_ids)
        if t % 5 == 0 or fg > 0.5:
            print(f"{t:>4} {obs['robot0_eef_pos'][2]:>7.3f} {float(rdata.xpos[ft_id][2]):>7.3f} "
                  f"{fg:>10.2f}  {grip_contacts(rmodel, rdata, cons)}")

    # (2) home over the target bowl, then press DOWN onto it (rigid -> real normal force)
    obs = settle()
    rmodel, rdata, _ = live()
    print(f"\n=== PRESS bowl: home over akita_black_bowl_1 then push down ===")
    print(f"  bowl pos = {np.round(obs['akita_black_bowl_1_pos'],3)}")
    print(f"{'step':>4} {'eef_z':>7} {'ftip_z':>7} {'maxF_grip':>10}  grip_contacts(top3)")
    for t in range(80):
        d = np.asarray(obs["akita_black_bowl_1_pos"]) - np.asarray(obs["robot0_eef_pos"])
        ax = float(np.clip(d[0] * 12, -1, 1)); ay = float(np.clip(d[1] * 12, -1, 1))
        obs, _, done, _ = env.step([ax, ay, -1, 0, 0, 0, -1])
        cons = all_contacts(rmodel, rdata)
        fg = max_force_on(cons, grip_ids)
        if t % 5 == 0 or fg > 0.5:
            print(f"{t:>4} {obs['robot0_eef_pos'][2]:>7.3f} {float(rdata.xpos[ft_id][2]):>7.3f} "
                  f"{fg:>10.2f}  {grip_contacts(rmodel, rdata, cons)}")

    print("\nOK if maxF_grip rises > 0 when the gripper presses the bowl/table.")


if __name__ == "__main__":
    main()
