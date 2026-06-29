#!/usr/bin/env python
"""Re-render the scripted DETOUR (go-around) recovery for the one missing wall (d70).

The other 4 treatment walls already have a plain `<id>.mp4` detour clip in
results/phase2_witness/; d70 was never saved. The detour is pure scripted P-control
physics (no OpenVLA), but MuJoCo rendering needs a GPU (egl) — run on a gengpu node.

    sbatch setup/render_d70_detour.sbatch
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make sibling scripts importable

import numpy as np
import imageio

from crashbench.scenario import Scenario
from crashbench.envs import LiberoEnv
import phase2_witness as W

SC_ID = "env_collision__T5__libero_spatial_t0_wall_d70"
OUT = f"results/phase2_witness/{SC_ID}.mp4"


def main():
    sc = Scenario.load(f"scenarios/{SC_ID}")
    env = LiberoEnv("libero_spatial", 0)
    wall = sc.obstacles[0]
    best = None
    for side in (-1.0, 1.0):
        r = W.run_detour(env, sc, wall, side, save_frames=True)
        key = (r["success"], r["avoided"], -r["max_wall_force"])
        if best is None or key > best["key"]:
            best = {"key": key, **r}
    print(f"d70 detour: success={best['success']} avoided={best['avoided']} "
          f"fmax={best['max_wall_force']:.0f} N frames={len(best['frames'])}")
    imageio.mimwrite(OUT, [np.asarray(f).astype(np.uint8) for f in best["frames"]], fps=20)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
