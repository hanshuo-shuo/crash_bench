#!/usr/bin/env python
"""Record OpenVLA's NOMINAL eef trajectory (no wall) so the figure scripts can draw the
top-down path map and the distance-to-path-vs-crash scatter (the two key explanatory
figures for the OOD-but-not-crash control). Saves:
    results/nominal_traj.npy   # (N,2) eef xy over the episode(s)
    results/nominal_traj.json  # episode lengths/success + object xy (bowl, plate) + home
Run on a GPU node (MUJOCO_GL=egl)."""

from __future__ import annotations

import os, json
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from crashbench.envs import LiberoEnv

SUITE, TASK, TARGET, SETTLE, N_EPISODES = "libero_spatial", 0, "akita_black_bowl_1", 10, 2


def main():
    from crashbench.policies import OpenVLAPolicy
    env = LiberoEnv(SUITE, TASK)
    base = np.asarray(env.default_init_states()[0])
    policy = OpenVLAPolicy(unnorm_key="libero_spatial")

    all_xy, episodes, objs = [], [], {}
    for ep in range(N_EPISODES):
        obs = env.reset_to(base)
        for _ in range(SETTLE):
            obs, _, _, _ = env.step(env.dummy_action())
        xy, success, home = [], False, np.asarray(obs["robot0_eef_pos"])[:2]
        for _ in range(220):
            o = env.policy_observation(obs, policy.resize_size)
            a = policy.act(o, env.task_description)
            obs, _, done, _ = env.step(a.tolist() if hasattr(a, "tolist") else a)
            xy.append(np.asarray(obs["robot0_eef_pos"])[:2])
            if done:
                success = True; break
        all_xy.extend(xy)
        episodes.append({"episode": ep, "steps": len(xy), "success": bool(success)})
        objs = {k[:-4]: [round(float(v), 4) for v in np.asarray(obs[k])[:2]]
                for k in obs if k.endswith("_pos") and np.asarray(obs[k]).shape == (3,)}
        print(f"  ep{ep}: success={success} steps={len(xy)}")

    traj = np.asarray(all_xy)
    np.save("results/nominal_traj.npy", traj)
    meta = {"episodes": episodes, "home_xy": [round(float(x), 4) for x in home],
            "objects": {k: v for k, v in objs.items() if k == TARGET or "plate" in k},
            "all_object_xy": objs, "n_points": len(traj)}
    json.dump(meta, open("results/nominal_traj.json", "w"), indent=2)
    print("saved results/nominal_traj.npy + .json", meta["objects"])


if __name__ == "__main__":
    main()
