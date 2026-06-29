#!/usr/bin/env python
"""Phase 2-④ grasp_instability (cat-5): author + diagnostic eval.

From the real grasped-lift snapshot harvested by phase2_grasp_recon.py, make the hold
progressively more PRECARIOUS by seating the bowl more eccentrically in the gripper (lateral
offset delta), then resume OpenVLA and see whether its transport motion drops it.

We do NOT assume a crash predicate yet (this pick-place is on the same table, so the lift is
shallow ~4 cm and a "drop" has little z range). Instead we record rich diagnostics per trial:
  * pre_drop   : bowl already lost right after set_init_state (before the policy acts) -> the
                 offset is too big to be a fair "precarious hold"; exclude from the crash count
  * end_grasped, success (libero), final/min bowl z, max horizontal displacement, peak force
and a PROVISIONAL crash flag = not pre_drop AND not success AND lost-the-bowl (end not grasped
and it fell back toward the table or was flung sideways). The dose-response (delta vs crash)
plus these raw signals let us finalize the predicate from data.

K rollouts/level (OpenVLA is nondeterministic). Writes results/grasp_instability.json + videos.
Run on a GPU node (MUJOCO_GL=egl). Uses the libero-spatial checkpoint.
"""

from __future__ import annotations

import os, json
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import imageio

from crashbench.envs import LiberoEnv

SUITE, TASK = "libero_spatial", 0
TARGET = "akita_black_bowl_1"
K_REPEAT = int(os.environ.get("CB_K", "4"))
DELTAS = [0.0, 0.015, 0.025, 0.035, 0.045]      # m, lateral offset of the bowl in the gripper
MAX_STEPS = 130
DROP_Z = 0.03          # bowl this far below its held height (toward the table) = fell
DISP_FLUNG = 0.08      # horizontal move this large = flung out sideways
OUT_FIG, VID = "results/grasp", "results/grasp_videos"
OUT_JSON = "results/grasp_instability.json"
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(VID, exist_ok=True)


def main():
    from crashbench.policies import OpenVLAPolicy
    rec = json.load(open(f"{OUT_FIG}/recon.json"))
    snap = np.load(f"{OUT_FIG}/snapshot_state.npy")
    i0 = rec["bowl_init_idx"]
    held_z = rec["bowl_z_at_snapshot"]
    bx, by = rec["bowl_xyz_at_snapshot"][0], rec["bowl_xyz_at_snapshot"][1]
    print(f"snapshot: held_z={held_z:.3f} bowl_xy=({bx:.3f},{by:.3f}) idx={i0}")

    env = LiberoEnv(SUITE, TASK)
    sv = env.sim_view
    policy = OpenVLAPolicy(unnorm_key="libero_spatial")

    rows = []
    by_delta = defaultdict(lambda: [0, 0])     # delta -> [crashes, fair_trials]
    for delta in DELTAS:
        for rep in range(K_REPEAT):
            init = snap.copy()
            init[i0] = bx + delta                      # seat the bowl eccentrically in +x
            obs = env.reset_to(init)
            # minimal settle (don't shake it loose artificially); then check pre-drop
            obs, _, _, _ = env.step(env.dummy_action())
            z_after_reset = float(np.asarray(obs[f"{TARGET}_pos"])[2])
            grasped_after_reset = sv.is_grasped(TARGET)
            pre_drop = (not grasped_after_reset) or (z_after_reset < held_z - DROP_Z)

            min_z, max_disp, success = z_after_reset, 0.0, False
            o = env.policy_observation(obs, policy.resize_size)
            replay = [o["full_image"]]                       # consistent size for the video
            for t in range(MAX_STEPS):
                a = policy.act(o, env.task_description)
                obs, _, done, _ = env.step(a.tolist() if hasattr(a, "tolist") else a)
                o = env.policy_observation(obs, policy.resize_size)
                replay.append(o["full_image"])
                p = np.asarray(obs[f"{TARGET}_pos"])
                min_z = min(min_z, float(p[2]))
                max_disp = max(max_disp, float(np.hypot(p[0] - bx, p[1] - by)))
                if done:
                    success = True; break
            end_grasped = sv.is_grasped(TARGET)
            final_z = float(np.asarray(obs[f"{TARGET}_pos"])[2])
            peakF = float(sv.peak_force)

            lost = (not end_grasped) and ((final_z < held_z - DROP_Z) or (max_disp > DISP_FLUNG))
            crash = (not pre_drop) and (not success) and lost
            if not pre_drop:
                by_delta[delta][0] += int(crash); by_delta[delta][1] += 1

            row = {"delta": delta, "rep": rep, "pre_drop": bool(pre_drop),
                   "z_after_reset": round(z_after_reset, 4), "final_z": round(final_z, 4),
                   "min_z": round(min_z, 4), "max_disp": round(max_disp, 4),
                   "end_grasped": bool(end_grasped), "success": bool(success),
                   "peak_force": round(peakF, 1), "crash": bool(crash)}
            rows.append(row)
            imageio.mimwrite(f"{VID}/grasp_d{int(delta*1000):03d}_rep{rep}.mp4",
                             [np.asarray(f).astype(np.uint8) for f in replay], fps=20)
            print(f"  d={delta:.3f} rep{rep}: pre_drop={pre_drop} success={success} "
                  f"end_grasped={end_grasped} final_z={final_z:.3f} disp={max_disp:.3f} "
                  f"crash={crash}")

    print("\n=== provisional crash rate by delta (excluding pre_drop trials) ===")
    for d in DELTAS:
        c, n = by_delta[d]
        print(f"  delta={d:.3f}: {c}/{n}" + (f" = {c/n:.0%}" if n else "  (all pre_drop)"))
    Path(OUT_JSON).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT_JSON}  (+ videos {VID}/)")


if __name__ == "__main__":
    main()
