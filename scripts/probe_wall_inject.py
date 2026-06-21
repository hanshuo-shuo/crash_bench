#!/usr/bin/env python
"""Probe: inject a STATIC wall into the LIBERO scene and confirm (a) the obs/action
bridge survives, (b) the pre-crash state still loads, (c) the gripper hitting the wall
registers a contact force via the adapter.

Why a static (jointless) body: it adds geoms/bodies but NO qpos/qvel DOF, so the LIBERO
state vector layout is unchanged and set_init_state() still applies our pre-crash states.

Reset flow that KEEPS the wall (verified reasoning, env_wrapper.py):
  env.reset_from_xml_string(xml_with_wall)   # rebuild sim WITH wall (one deterministic reset)
  env.set_init_state(state)                  # set qpos/qvel only -- does NOT call env.reset()
Calling LiberoEnv.reset_to() would call env.reset() (non-deterministic) and WIPE the wall.
"""

from __future__ import annotations

import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
from crashbench.envs import LiberoEnv
from crashbench.envs.libero_adapter import ROBOT_CONTACT_BODIES

SUITE, TASK = "libero_spatial", 0

# wall on the home(-0.211,-0.011) -> target-bowl(-0.063,0.202) grasp path, standing tall
WALL_NAME = "crash_wall"
# thin slab across the +x approach, standing tall (z 0.86->1.30) so the gripper can't
# clear it by transiting high; centered on the path point at x=-0.13 (y~0.10)
WALL_POS = (-0.13, 0.10, 1.08)
WALL_SIZE = (0.025, 0.08, 0.22)


def inject_wall(xml: str) -> str:
    body = (f'<body name="{WALL_NAME}" pos="{WALL_POS[0]} {WALL_POS[1]} {WALL_POS[2]}">'
            f'<geom name="{WALL_NAME}_g" type="box" '
            f'size="{WALL_SIZE[0]} {WALL_SIZE[1]} {WALL_SIZE[2]}" '
            f'rgba="0.85 0.2 0.2 1" group="0" contype="1" conaffinity="1"/></body>')
    assert "</worldbody>" in xml, "no </worldbody> in model xml"
    return xml.replace("</worldbody>", body + "</worldbody>", 1)


def main():
    env = LiberoEnv(SUITE, TASK)
    base = np.asarray(env.default_init_states()[0])
    obs = env.reset_to(base)
    baseline_keys = set(obs.keys())
    print(f"task: {env.task_description!r}")
    print(f"baseline obs keys: {len(baseline_keys)}; eef={np.round(obs['robot0_eef_pos'],3)}")

    nq_before = int(getattr(env.env.sim.model, "_model", env.env.sim.model).nq)
    base_bowl = np.round(base[10:13], 3)        # bowl free-joint xyz in the state vector

    # inject wall + rebuild sim from the modified xml
    xml = env.env.sim.model.get_xml()
    env.env.reset_from_xml_string(inject_wall(xml))
    nq_after = int(getattr(env.env.sim.model, "_model", env.env.sim.model).nq)
    obs = env.env.set_init_state(base)          # NOTE: not env.reset() -> wall survives
    bowl_preselect = np.round(obs["akita_black_bowl_1_pos"], 3)
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    print(f"\nnq before/after inject = {nq_before}/{nq_after}  (must match -> state layout intact)")
    print(f"base bowl xyz[10:13] = {base_bowl}")
    print(f"bowl pos right after set_init_state = {bowl_preselect}")

    # (a) bridge intact?
    new_keys = set(obs.keys())
    print(f"\nobs keys after inject: {len(new_keys)}  (added={new_keys-baseline_keys}, "
          f"dropped={baseline_keys-new_keys})")

    # (b) wall body present?
    model = getattr(env.env.sim.model, "_model", env.env.sim.model)
    wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WALL_NAME)
    print(f"wall body id = {wid} (>=0 means present); bowl pos={np.round(obs['akita_black_bowl_1_pos'],3)}")

    # (c) drive from home toward the bowl; the wall sits in the way -> contact force
    sv = env.sim_view
    print(f"\n{'step':>4} {'eef_x':>7} {'eef_y':>7} {'eef_z':>7} {'F_wall':>8} {'F_grip':>8}  top_grip_contact")
    for t in range(50):
        d = np.asarray(obs["akita_black_bowl_1_pos"]) - np.asarray(obs["robot0_eef_pos"])
        ax = float(np.clip(d[0] * 12, -1, 1)); ay = float(np.clip(d[1] * 12, -1, 1))
        obs, _, _, _ = env.step([ax, ay, -0.3, 0, 0, 0, -1])
        f_wall = sv.max_contact_force(list(ROBOT_CONTACT_BODIES), against=[WALL_NAME])
        f_grip = sv.max_contact_force(list(ROBOT_CONTACT_BODIES))
        if t % 4 == 0 or f_wall > 1.0:
            e = obs["robot0_eef_pos"]
            print(f"{t:>4} {e[0]:>7.3f} {e[1]:>7.3f} {e[2]:>7.3f} {f_wall:>8.1f} {f_grip:>8.1f}")

    print(f"\nfinal peak_force={sv.peak_force:.1f}")
    print("PASS" if (wid >= 0 and new_keys == baseline_keys and sv.peak_force > 20)
          else "CHECK: see above (need wall present + keys unchanged + a contact force)")


if __name__ == "__main__":
    main()
