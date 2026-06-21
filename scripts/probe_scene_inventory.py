#!/usr/bin/env python
"""Inventory the LIBERO-spatial task-0 scene so we can author env-collision scenarios
from real geometry (not guesses): what objects exist, where they sit, how big they are,
where the eef homes, and where the target bowl / place plate are.

Goal: pick an existing rigid obstacle (or a spot to reposition one) that lies ON
OpenVLA's nominal grasp path, so a normal reach drives the gripper into it.
"""

from __future__ import annotations

import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
from crashbench.envs import LiberoEnv

SUITE, TASK = "libero_spatial", 0


def main():
    env = LiberoEnv(SUITE, TASK)
    obs = env.reset_to(np.asarray(env.default_init_states()[0]))
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    sim = env.env.sim
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    print(f"task: {env.task_description!r}")
    print(f"eef home pos = {np.round(obs['robot0_eef_pos'], 3)}")

    # object positions straight from obs (the convention predicates use)
    print("\n=== obs object *_pos (world xyz) ===")
    for k in sorted(obs):
        if k.endswith("_pos") and not k.startswith("robot0"):
            print(f"  {k:34s} {np.round(np.asarray(obs[k]), 3)}")

    # body-level world positions for every non-robot/non-gripper body
    print("\n=== scene bodies (world xpos) ===")
    for b in range(model.nbody):
        name = model.body(b).name
        if name.startswith(("robot0", "gripper0")) or name in ("world", "table"):
            continue
        print(f"  b{b:<3d} {name:30s} xpos={np.round(np.asarray(data.xpos[b]), 3)}")

    # geom sizes for the obstacle-ish bodies (stove etc.) so we know how tall they stand
    print("\n=== geoms of candidate obstacles (name, body, type, size, world pos) ===")
    for g in range(model.ngeom):
        bid = int(model.geom_bodyid[g])
        bname = model.body(bid).name
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if any(s in bname.lower() for s in ("stove", "plate", "ramekin", "wine", "cabinet", "shelf")):
            print(f"  g{g:<4d} {gname:28s} body={bname:22s} type={int(model.geom_type[g])} "
                  f"size={np.round(np.asarray(model.geom_size[g]), 3)} "
                  f"wpos={np.round(np.asarray(data.geom_xpos[g]), 3)}")

    # table top z for reference
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    tz = [float(data.geom_xpos[g][2] + model.geom_size[g][2])
          for g in range(model.ngeom) if int(model.geom_bodyid[g]) == tid]
    print(f"\ntable top z ~= {max(tz) if tz else '?'}")


if __name__ == "__main__":
    main()
