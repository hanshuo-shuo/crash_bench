#!/usr/bin/env python
"""Author a pre-crash scenario by perturbing a LIBERO initial state (PLAN.md §4 Phase 1 step 1).

Pipeline (README §4.4):
  1. take a LIBERO task's default initial state
  2. perturb qpos so the gripper/object is in a pre-crash configuration
     (drifting toward the table / object at a table edge / held object tilted)
  3. snapshot the sim state -> Scenario.init_state
  4. attach crash + success predicates
  5. (Phase 2) verify a recovery witness exists, else drop

IMPORTANT — this is a SCAFFOLD. The state-vector layout that env.set_init_state()
expects is robosuite's flattened MjSim state ([time, qpos, qvel, act?]). The exact
qpos indices for the eef / a given object must be confirmed interactively on a real
env (print state, nudge, re-render). The `perturb_*` functions below are stubs that
show *where* that logic goes; fill them in once you've inspected the layout.

Run on a GPU node:
    MUJOCO_GL=egl python scripts/author_scenario.py --suite libero_spatial --task_id 0 \
        --state_idx 0 --category env_collision --horizon T-5 --out scenarios
"""

from __future__ import annotations

import argparse

import numpy as np

from crashbench.envs import LiberoEnv
from crashbench.scenario import Scenario, PredicateSpec


# ---- perturbation stubs (FILL IN after inspecting the state layout) ---------
def perturb_toward_table(state: np.ndarray, dz: float = -0.08) -> np.ndarray:
    """Lower the end-effector toward the table to create an env-collision pre-crash state.

    TODO(verify): locate the eef-height entry in the flattened state and shift it by dz.
    The placeholder returns the state unchanged so the scaffold runs end-to-end.
    """
    s = state.copy()
    # s[EEF_Z_INDEX] += dz    # <-- determine EEF_Z_INDEX from a real env
    return s


PERTURBATIONS = {
    "toward_table": perturb_toward_table,
    # add: "object_at_edge", "held_object_tilted", ...
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--state_idx", type=int, default=0, help="which default init state to perturb")
    ap.add_argument("--perturb", default="toward_table", choices=list(PERTURBATIONS))
    ap.add_argument("--category", required=True)
    ap.add_argument("--horizon", required=True, choices=["T-1", "T-5", "T-20"])
    ap.add_argument("--id", default=None, help="scenario id (auto if omitted)")
    ap.add_argument("--out", default="scenarios")
    args = ap.parse_args()

    env = LiberoEnv(args.suite, args.task_id)
    base = env.default_init_states()[args.state_idx]
    init_state = PERTURBATIONS[args.perturb](np.asarray(base))

    sid = args.id or f"{args.category}__{args.horizon.replace('-', '')}__{args.suite}_t{args.task_id}_{args.state_idx:03d}"

    # Default predicate set for the pilot (PLAN.md §3). Tune thresholds/body names per scenario.
    crash_predicates = [
        PredicateSpec("contact_force", {
            "bodies": [f"robot0_link{i}" for i in range(5, 8)],  # TODO(verify) body names
            "threshold": 20.0,
        }),
    ]
    success = PredicateSpec("libero_task_success", {})

    sc = Scenario(
        id=sid,
        category=args.category,
        horizon=args.horizon,
        task_suite=args.suite,
        task_id=args.task_id,
        instruction=env.task_description,
        init_state=init_state,
        crash_predicates=crash_predicates,
        success_predicate=success,
        max_steps=220,
        metadata={"perturb": args.perturb, "base_state_idx": args.state_idx},
    )
    d = sc.save(args.out)
    print(f"saved scenario -> {d}")
    print(f"  instruction: {sc.instruction}")
    print("  NOTE: perturbation is a stub — fill in perturb_* once you've inspected the state layout,")
    print("  then re-render to confirm the state is actually pre-crash before trusting it.")


if __name__ == "__main__":
    main()
