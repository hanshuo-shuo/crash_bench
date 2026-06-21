#!/usr/bin/env python
"""Integration check: the REAL adapter (LiberoSimView.max_contact_force) reads force.

The raw mechanism is proven in probe_contact_force.py; this confirms the wired adapter
path (name->id resolution, live re-fetch, peak_force tracking) works end-to-end. Drive
the gripper straight down into the scene and watch the adapter's force + peak climb.
"""

from __future__ import annotations

import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from crashbench.envs import LiberoEnv
from crashbench.envs.libero_adapter import ROBOT_CONTACT_BODIES

SUITE, TASK = "libero_spatial", 0


def main():
    env = LiberoEnv(SUITE, TASK)
    sv = env.sim_view
    base = np.asarray(env.default_init_states()[0])
    obs = env.reset_to(base)
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    print(f"task: {env.task_description!r}")
    print(f"{'step':>4} {'eef_z':>7} {'adapter_F':>10} {'peak_force':>10} "
          f"{'F_vs_stove':>11}")
    for t in range(40):
        obs, _, _, _ = env.step([0, 0, -1, 0, 0, 0, -1])
        # update() is called inside env.step -> sim_view; query the adapter API directly
        f_all = sv.max_contact_force(list(ROBOT_CONTACT_BODIES))
        f_stove = sv.max_contact_force(list(ROBOT_CONTACT_BODIES), against=["flat_stove_1_burner"])
        if t % 4 == 0 or f_all > 1.0:
            print(f"{t:>4} {obs['robot0_eef_pos'][2]:>7.3f} {f_all:>10.2f} "
                  f"{sv.peak_force:>10.2f} {f_stove:>11.2f}")

    print(f"\nfinal peak_force = {sv.peak_force:.1f} N")
    print("PASS" if sv.peak_force > 50 else "FAIL: adapter never saw a contact force")


if __name__ == "__main__":
    main()
