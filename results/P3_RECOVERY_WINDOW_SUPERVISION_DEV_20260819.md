# P3.0 Dense Recovery-Window Supervision

Status: **development supervision authored; no model training or Router retuning performed**.

Quest job: `9844164+9847692`.  Collection commit(s): `5f072d9d6ef75beea48a8a282797ab234e062055, 974219c82736dcfaf530e8f0977351904ee21053`.

Each authored trajectory uses one accepted Base scan under the glass condition. Every dense anchor branches from that scan's corresponding exact serialized simulator/controller state into BaseContinue, DetourComplete, and the zero-delta FailSafeHold. Directional RetreatHold was not used.

## Dense trajectory map

| Trajectory | Recovery-open anchors | Intervals (earliest→latest H) | Latest recoverable | Preferred-option transitions | Notes |
|---|---|---|---:|---|---|
| heldout_0003 | none | none | n/a | FailSafeHold → Base → FailSafeHold → Base → FailSafeHold → Base → FailSafeHold | library-unsolved |
| heldout_0004 | [30, 28, 24, 22, 20, 18, 16, 14, 12, 10, 8] | 30→28 (2 anchors, span 2); 24→8 (9 anchors, span 16) | 8 | Detour → Base → Detour → FailSafeHold → Base → FailSafeHold | non-monotonic |
| heldout_0009 | [30, 28, 22, 16, 14, 12, 10, 4] | 30→28 (2 anchors, span 2); 22→22 (1 anchors, span 0); 16→10 (4 anchors, span 6); 4→4 (1 anchors, span 0) | 4 | Detour → FailSafeHold → Detour → FailSafeHold → Detour → FailSafeHold → Detour → FailSafeHold | non-monotonic |
| heldout_0010 | [28, 26, 24, 22, 18, 16, 14, 12, 8] | 28→22 (4 anchors, span 6); 18→12 (4 anchors, span 6); 8→8 (1 anchors, span 0) | 8 | FailSafeHold → Detour → Base → Detour → Base → Detour → Base → FailSafeHold | non-monotonic |
| heldout_0019 | none | none | n/a | FailSafeHold | library-unsolved |

## Required questions

- `heldout_0010` latest recoverable anchor: **8 actions before collision**. Prior P2 T-20 recoverability did not reproduce on this accepted dense Base scan.
- `heldout_0019` latest recoverable anchor: **not observed**. Prior P2 T-20 recoverability did not reproduce on this accepted dense Base scan.
- Recoverable → loss-control temporal conversion: **yes** on heldout_0004, heldout_0009, heldout_0010.
- Recovery continuity: 2/5 trajectories have at most one interval; non-monotonic recovery is present on heldout_0004, heldout_0009, heldout_0010.
- FailSafeHold contract violations: **0**. All violating anchors, if any, are retained.

## Cross-run stability

The old P2 T-20 outcome is a prior run, not a label copied into P3. Both missed episodes changed their T-20 signature on the newly accepted dense Base scan:

| Trajectory | Prior P2 T-20 Base / Detour | Dense T-20 Base / Detour | Prior collision action | Dense collision action |
|---|---|---|---:|---:|
| heldout_0010 | catastrophe / task_success | task_success / task_success | 35 | 40 |
| heldout_0019 | catastrophe / task_success | catastrophe / safe_noncompletion | 30 | 31 |

Excluded scan attempts: **1**.
- `fresh:glass_recovery_heldout_0010`: `onpath_no_catastrophe` in `/gpfs/home/shv7753/crash_bench/results/counterfactual_router/p3_dense_5f072d9_20260819T064818Z`.

`heldout_0010` was subsequently accepted as a one-trajectory supplement; its failed no-catastrophe scan remains in provenance. No outcome was imputed from P2.

## Hard controls and feature overlap

The authoring selected 4 control regions (62 Base-success states) from the union of the highest-peak and longest pointwise-margin exceedances in the existing P2.5 Base traces. No new control option rollout was run.

Overlap is measured in the 10-D standardized frozen-Router prediction-output space. State-level leave-one-out 1-NN balanced accuracy is `1.000`; 0.000 of recovery-open points have a hard-control neighbor no farther than their nearest other recovery-open point. Median recovery→hard distance is `3.275` versus median recovery→recovery distance `0.511`. This state-level result is descriptive and optimistic because adjacent states from the same trajectory are not source-disjoint.

On the old Router's actual scalar decision axis, overlap is substantial: 0.750 of recovery-open states lie inside the hard-negative score range, and 0.984 of hard negatives lie inside the recovery-open range. Recovery-open-v-hard-negative score AUC is `0.351` (higher is presumed more recoverable). Thus the current operational score does not separate these labels. These diagnostics are not a newly tuned decision rule.

## Direct-supervision targets

The next model should learn three related but distinct targets from causal history: (1) whether intervention is needed relative to Base, (2) whether Detour task completion is still available and the actions remaining to the local window closure, and (3) the preferred option among Base, Detour, and FailSafeHold. Loss-control and Hold-contract violation should remain explicit auxiliary labels rather than being folded into one binary hazard score. Hard Base-success regions must be included as intervention-negative examples because the old Router assigns them sustained high advantage.

## Outputs and provenance

- Dense exact-state anchors: 80 (5/5 complete grids).
- Recovery-open anchors: 28; loss-control anchors: 35; Base-preferred anchors: 15.
- Capture root(s): `/gpfs/home/shv7753/crash_bench/results/counterfactual_router/p3_dense_5f072d9_20260819T064818Z, /gpfs/home/shv7753/crash_bench/results/counterfactual_router/p3_dense_0010_supplement_974219c_20260819T074000Z_r2`.
- Hard-control source: `/gpfs/home/shv7753/crash_bench/results/counterfactual_router/p2_dynamic_exact_seq_0f92a1b420dd_20260818T162140Z`.
- Excluded scan attempts retained in provenance: 1.
- Collection command: `python scripts/collect_counterfactual_option_rollouts.py --conditions glass --history-length 8 --horizons 30 28 ... 4 2 1 --retreat-mode hold ...`
- Authoring command: `python scripts/author_recovery_window_supervision.py --capture <P3 capture> --router-model <frozen Router> --hard-control-capture <P2 exact-prefix capture> ...`

Completion here is data authoring only. No temporal model, new Router, dynamic evaluation, alpha, margin, or intervention threshold was trained or adjusted.
