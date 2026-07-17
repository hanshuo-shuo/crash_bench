#!/usr/bin/env python
"""Historical low-wall d62 detour existence demo (not a general recovery runner).

Demonstrates the full chain only on the separately identified lowered d62 wall: bare OpenVLA
crashes into that low-wall geometry; the same
policy wrapped in GuardedPolicy with recovery=DetourComplete detects imminent crash via the probe,
hands off to the witnessed full-arm collision-free detour, and COMPLETES the pick-and-place ->
eval.run_episode returns RECOVERY_SUCCESS (not CRASH, not just SAFE_ABORT).

The witness is ~379 steps, longer than the scenario's crash-horizon max_steps (220), so we raise
max_steps for the recovery run. DetourComplete needs the bowl/plate world positions — read once from
the episode-start obs (static pre-grasp) and injected at construction.

Runs on a GPU node (openvla env + GPU).
  python scripts/phase3_detour_handoff.py
"""

from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from crashbench.scenario import Scenario
from crashbench.envs import LiberoEnv
from crashbench.policies import OpenVLAPolicy, GuardedPolicy
from crashbench.probe import Probe
from crashbench.recovery import DetourComplete, WitnessReplay
from crashbench.eval import run_episode, Outcome

SCEN = "scenarios_detour_lowwall/env_collision__T5__libero_spatial_t0_wall_d62__lowwall_detour_v1"
TARGET, PLATE = "akita_black_bowl_1", "plate_1"
RECOVERY_MAX_STEPS = 500          # witness detour is ~379 steps; crash-horizon (220) is too short
OUT = "results/phase3_detour"


def main():
    os.makedirs(OUT, exist_ok=True)
    sc = Scenario.load(SCEN)
    wall = sc.obstacles[0]
    print(f"scenario {sc.id}\n  wall pos={wall['pos']} size={wall['size']} (top z={wall['pos'][2]+wall['size'][2]:.2f})")

    probe = Probe.load(); print(f"  probe thr={probe.thr:.3f}")
    base = OpenVLAPolicy(pretrained_checkpoint="openvla/openvla-7b-finetuned-libero-spatial",
                         unnorm_key="libero_spatial", center_crop=True, capture_hidden=True)
    env = LiberoEnv(sc.task_suite, sc.task_id)

    # read static bowl/plate world positions once (what the witness used); inject into DetourComplete.
    # IMPORTANT: settle first (10 gripper-open steps, exactly like the witness) — objects fall from the
    # raw init pose (~z0.97) to rest (~z0.912); reading pre-settle gives wrong coords -> failed grasp.
    obs = env.reset_to(sc.init_state, obstacles=sc.obstacles)
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1.0])
    target_pos = list(map(float, obs[f"{TARGET}_pos"]))
    plate_pos = list(map(float, obs[f"{PLATE}_pos"]))
    print(f"  bowl={target_pos}  plate={plate_pos} (after settle)")

    # (1) bare OpenVLA baseline -> expect CRASH
    print("\n=== (1) bare OpenVLA ===")
    r0 = run_episode(sc, env, base)
    print(f"  outcome={r0.outcome.value}  steps={r0.steps_to_event}  peakF={r0.peak_contact_force:.0f}N")

    # (2) GuardedPolicy(recovery=WitnessReplay) -> expect RECOVERY_SUCCESS.
    # The probe fires at episode start (settled pre-crash state = the witness's start), so replaying
    # the proven witness verbatim reproduces the recovery deterministically.
    print("\n=== (2) GuardedPolicy + WitnessReplay ===")
    witness = sc.witness
    print(f"  witness: {None if witness is None else len(witness)} steps")
    recovery = WitnessReplay(witness)
    guard = GuardedPolicy(base, probe, recovery=recovery)
    sc.max_steps = RECOVERY_MAX_STEPS        # give the detour room to finish (crash predicate unchanged)
    guard.reset()
    r1 = run_episode(sc, env, guard, save_video_path=f"{OUT}/{sc.id}_guarded_detour.mp4")
    bx, by = env.sim_view.object_xy(TARGET); px2, py2 = env.sim_view.object_xy(PLATE)
    import numpy as _np
    gdist = _np.hypot(bx - px2, by - py2)
    print(f"  outcome={r1.outcome.value}  trigger_step={guard.trigger_step}  "
          f"steps={r1.steps_to_event}  peakF={r1.peak_contact_force:.0f}N  "
          f"final bowl-plate dist={gdist:.3f}  libero_done={env.sim_view.libero_done}")

    # (3) DIRECT open-loop replay (no OpenVLA/guard) — isolates whether replay reproduces the witness
    print("\n=== (3) direct witness replay (deterministic, no policy) ===")
    obs = env.reset_to(sc.init_state, obstacles=sc.obstacles)
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1.0])
    done = False
    for a in witness:
        obs, _, done, _ = env.step(a.tolist() if hasattr(a, "tolist") else list(a))
    import numpy as _np
    b = _np.asarray(obs[f"{TARGET}_pos"]); p = _np.asarray(obs[f"{PLATE}_pos"])
    print(f"  final bowl-plate xy dist={_np.linalg.norm(b[:2]-p[:2]):.3f} "
          f"bowl_z={b[2]:.3f} libero_done={bool(done) or bool(env.sim_view.libero_done)}")

    print("\n=== VERDICT ===")
    ok = (r0.outcome == Outcome.CRASH) and (r1.outcome == Outcome.RECOVERY_SUCCESS)
    print(f"  bare=CRASH: {r0.outcome==Outcome.CRASH}   guarded=RECOVERY_SUCCESS: "
          f"{r1.outcome==Outcome.RECOVERY_SUCCESS}   => {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
