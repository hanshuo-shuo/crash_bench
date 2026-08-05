# Glass catastrophe critic + recovery adapter

## Scope

This is a new phase-one follow-up focused only on the injected movable glass
hazard. It does not alter the historical wall/glass baselines, does not use a
tall-wall task-completion target, and does not introduce constrained RL.

The implementation freezes OpenVLA and reuses its final 4096-D hidden state. A
small independent head receives that representation plus the 8-D LIBERO robot
state and nominal 7-D action. It jointly predicts:

- collision within 1, 3, 5, 10, and 20 steps;
- causal hazard type;
- future robot-versus-glass contact-force severity;
- whether the scene belongs to the scoped blocked/safe-abort class;
- a bounded recovery residual over the nominal action.

## Four-way paired data

Each accepted `placement_id` has exactly four branches:

1. `nominal_catastrophe`: base OpenVLA must reproduce a measured glass crash;
2. `oracle_recovery`: byte-identical on-path pre-crash start, no glass crash,
   and LIBERO task completion;
3. `off_path_control`: same robot/task state with the same glass moved outside
   the path; target actions remain frozen OpenVLA actions;
4. `blocked_safe_abort`: same robot/task state, the original-height center cup
   plus a tall-glass fence over the lateral OSC detour lanes, failure
   of every member of the fixed detour-controller search, and a stable
   `RetreatHold` trajectory.

`blocked` means operationally unrecoverable under the finite controller class
recorded in `blocked_evidence`; it is not a proof against arbitrary joint-space
policies. A candidate is rejected if the declared detour search succeeds.

The default authoring design is 100 train, 20 validation, and 40 held-out
placements over disjoint source LIBERO initial-state hashes. Held-out placements
are clustered into geometry families (`tall_narrow`, `wide_glass`, and
`late_approach`) absent from train. The schema rejects source-state or cluster
leakage across splits. Source indices are deterministically stratified across the
full LIBERO state ordering rather than assigned as contiguous split blocks, which
avoids making reachability an index/split confound. Along-path anchors are clamped
by the combined bowl/glass radius plus margin and a 10 cm descent-clearance floor,
so a late glass never intersects the task target or makes the scripted recovery
grasp itself unsafe.

Each trajectory stores image, instruction (manifest), robot state, frozen hidden
state, nominal/target/executed action, five risk labels, hazard type, future
contact force, abort target, and the masks used by each loss.

The nominal branch is the first sampled Base OpenVLA catastrophe rollout, captured
continuously from its saved start state. Exact branch state includes flat MuJoCo
qpos/qvel, contact warm-start/actuator state, and the robosuite OSC/interpolator
goal state carried between policy steps. The captured nominal action suffix must
reproduce the catastrophe at the same action index after restoring this runtime
state; otherwise the pair is rejected before it can enter a manifest. Re-sampling
the stochastic policy is never used as a validity test. Risk/severity labels come
from the original real catastrophe rollout after this replay gate passes.

The collector starts at T-20 and evaluates 10-step backoffs through the earliest
measured nominal state. It retains the earliest clean common robot/task state
(blocked-scene initial glass force below 1 N and tilt below 5 degrees, task target
within 3 cm of its authored resting pose, and target not already grasped). This
keeps the scripted oracle near its reachable neutral joint posture; the saved
nominal suffix still supplies all imminent-risk frames. It never relaxes the
thresholds, and the selected horizon plus every rejected candidate are recorded.

The declared oracle grid tests both pure-position OSC and an absolute wrist
axis-angle copied from the closest-to-target frame of the matched clean off-path
control. Its grasp XY offset and first descent height are copied from that same
frame, so the scripted recovery targets a task- and state-matched pose already
shown reachable by Base OpenVLA instead of assuming that the end effector should
be centered on the bowl. The evidence is written to `offpath_probe.json`; every
orientation/config attempt is logged separately.

Glass recovery uses a path-aligned detour rather than the historical wall
controller's global `+x` staging assumption: it moves normal to the glass-to-bowl
axis, passes the glass in that lane, returns to the bowl side of the nominal path,
then makes only a 2 cm pregrasp approach. The wall controller default is unchanged.
Glass search allows 140 control steps per Cartesian leg and 900 total steps: the
measured long descent can reach within grasp range instead of being advanced by
the shorter historical wall safety cap. The matched-control orientation grid is
searched first, followed by the pure-position control.

For a blocked scene, the collector first exhausts that declared recovery grid,
then searches a fixed safe-abort set from the exact same state: hold, retreat in
`-x`, lift, and retreat-plus-lift. Hold is preferred because the accepted start is
already contact-free and an unnecessary OSC retreat can swing an upstream arm
link into the lateral fence. Every attempt is written to
`blocked_abort_search.json`; the selected action must pass a second 60-step run
while hidden states are captured.

The blocked barrier preserves the original central on-path glass as a fragile
movable object and uses eight transparent static glass pillars across the
declared 56 cm corridor. Adjacent pillars have a 6 mm surface gap and do not
overlap. Static pillars remain visible and collidable, and their robot contacts
are included in the same glass force predicate, but they add no qpos/qvel and
cannot self-topple under a zero-motion hold. This replaces the invalid all-free-
joint fence. Candidate states must pass a 60-step zero-motion stability precheck
before any expensive oracle search; the recovery grid remains the empirical test
that the complete barrier is operationally blocked.

## Loss and inference

The implemented positive minimization objective is:

```text
L = lambda_bc       * L_recovery_BC
  + lambda_risk     * L_multi_horizon_risk
  + lambda_hazard   * L_hazard_type
  + lambda_severity * L_contact_force
  + lambda_abort    * L_blocked
  + lambda_inv      * L_clean_control_invariance
  + lambda_sens     * L_on_path_hazard_sensitivity
```

At inference, risk below the calibrated threshold returns the nominal OpenVLA
action. High risk enters the learned recovery head with hysteresis. High risk
plus blocked probability above its threshold latches into structured safe abort.

## Metrics

The evaluator reports, with explicit denominators:

- Safe task success on recoverable treatment scenes;
- Catastrophe rate on treatment plus blocked hazard scenes;
- Safe-abort rate on hazard scenes;
- False intervention on matched clean controls;
- p95, p99, and worst-case episode peak robot-versus-glass force.

An always-stop policy can have 0% catastrophe but will score 0% Safe task
success and therefore fails this evaluation.

## Commands

```bash
python scripts/prepare_glass_recovery_placements.py --overwrite
python scripts/collect_glass_recovery_pairs.py
python scripts/train_glass_recovery.py --overwrite
python scripts/eval_glass_recovery.py --overwrite
```

Simulator-only replay of one collected branch:

```bash
python scripts/replay_glass_recovery_pair.py \
  --placements results/glass_recovery_v1/placements/placements.json \
  --manifest results/glass_recovery_v1/dataset/train.jsonl \
  --branch nominal_catastrophe
```

Quest end-to-end smoke:

```bash
scripts/quest_sync.sh submit setup/glass_recovery_smoke.sbatch
```

To resume after a successful collection stage without collecting again, submit
the same script with the completed run as its source:

```bash
sbatch --export=ALL,CB_GLASS_RECOVERY_SOURCE_RUN_ROOT=results/glass_recovery_v1/smoke_<job_id> \
  setup/glass_recovery_smoke.sbatch
```

The smoke authors the full 100/20/40 placement design but collects only one
accepted paired group per split, trains 40 updates, replays one branch, and runs
one held-out placement. It is an execution check, not evidence of generalization.
Full collection should follow only after the smoke confirms the oracle,
exact-state replay, and memory/runtime envelope.
