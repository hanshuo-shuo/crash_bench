# Phase 2 — witnesses / recoverability (README §4.4)

> 2026-06-27 · CrashBench Phase 2 · feasibility filter for the env_collision scenarios.

README §4.4 step 4 (feasibility filter): **every scenario must have a witness — at least one
recovery trajectory** found by a planner/teleop, or the scenario is dropped. The witness proves
the pre-crash state is recoverable (the crash was avoidable, the scenario is fair) and becomes
recovery-demo data for the fine-tuning baseline (README §7).

## Result (5 env_collision treatment walls)

| witness | result | meaning |
|---|---|---|
| **safety-recoverability (safe-abort)** | **5/5**, wall force **0 N** | a retreat→hold continuation avoids the wall for the full horizon → **the crash is avoidable, not forced**; all 5 pre-crash states are recoverable |
| **task-completion (scripted)** | **0/5** | a scripted end-effector detour that grasps the bowl and places it → fails (wall force 165–625 N) |

Generator: [`scripts/phase2_witness.py`](../scripts/phase2_witness.py) →
[`results/witness.json`](witness.json); the safe-abort witness trajectory is saved into each
scenario (`scenarios/*/witness.npy` + `metadata.witness`, replayable / usable as a demo); videos
in `results/phase2_witness/`.

## Why the scripted task-completion witness fails (and what it needs)

The bowl is reachable in principle — it always sits past the wall in x (`bowl_clear_x` > 0 for all
5). But the wall is a **tall slab (z 0.86–1.30 m)**, and the failure is **arm-body collision, not
gripper collision**: a scripted end-effector waypoint plan can route the *gripper* around the
wall's y-edge, but the robot's **forearm/elbow links** (which span from the fixed base behind the
wall out to the gripper past it) still clip the slab. Routing the end-effector around ≠ routing the
**arm** around — that is a configuration-space problem.

This is exactly the tooling README §4.4 anticipates: *"a sample-based planner (or human
teleoperator) must find at least one recovery trajectory."* End-effector scripting is insufficient
for tall obstacles; a **joint-space RRT\* with full-body collision checking, or teleop**, is
required for the task-completion witness.

## Status / next

- **Recoverability (feasibility) established:** 5/5 have a safe-abort witness → none are forced-
  crash, all pass the minimal feasibility filter. The OpenVLA crashes are genuine avoidable
  failures, not impossible scenarios. (This is the load-bearing claim for the paper's fairness.)
- **Recovery-demo dataset (task-completion witnesses):** the concrete Phase-2 tooling need —
  implement a joint-space RRT\* (MuJoCo full-body collision) or record teleop. These trajectories
  then double as the recovery-finetuning data (README §7 baseline).
- Authoring note for the benchmark: tall walls make task-completion witnesses hard; consider
  shorter slabs (clear the arm transit) or accept that contact-rich scenarios need teleop
  witnesses (README open q #3).
