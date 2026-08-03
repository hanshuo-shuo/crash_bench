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
leakage across splits. Along-path anchors are clamped by the combined bowl/glass
radius plus margin and a 10 cm descent-clearance floor, so a late glass never
intersects the task target or makes the scripted recovery grasp itself unsafe.

Each trajectory stores image, instruction (manifest), robot state, frozen hidden
state, nominal/target/executed action, five risk labels, hazard type, future
contact force, abort target, and the masks used by each loss.

The collector starts at T-20 and, when necessary, walks backward from the crash
in 10-step increments along the measured nominal history until every branch has a clean
common robot/task state (blocked-scene initial glass force below 1 N and tilt
below 5 degrees, task target within 3 cm of its authored resting pose, and target
not already grasped). It never makes an invalid late state pass by relaxing those
thresholds; the selected horizon and every rejected candidate are recorded.

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
  --pair-dir results/glass_recovery_v1/dataset/train/<placement_id> \
  --branch nominal_catastrophe
```

Quest end-to-end smoke:

```bash
scripts/quest_sync.sh submit setup/glass_recovery_smoke.sbatch
```

The smoke authors the full 100/20/40 placement design but collects only one
accepted paired group per split, trains 40 updates, replays one branch, and runs
one held-out placement. It is an execution check, not evidence of generalization.
Full collection should follow only after the smoke confirms the oracle,
exact-state replay, and memory/runtime envelope.
