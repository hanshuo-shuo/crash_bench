# Pilot B repair and re-entry plan

## Status

**Engineering continuation repair: pass. Targeted oracle repair: pass. Fresh
H=20 frontier: no-go.** Quest
diagnostic job `9071778` first showed exact H=20 continuation in both reset
paths but only 1/2 oracle recovery. The previously failing placement was then
isolated and repaired in Quest job `9094715`: its exact Base suffix and safe
task-completing oracle now pass in both the in-memory and serialized-reset
paths. This clears the known engineering blockers without retroactively
turning the exposed r7 outcomes into a Pilot B pass.

The r7 placements were exposed to repeated diagnostic outcomes and may be used
only for debugging. Any renewed frontier must use newly authored,
outcome-blind placements.

The fresh frontier has now been run as Quest job `9095054`. It used 15 new
physical scenes with no r7 source-state overlap, but achieved only 9/12 strict
exact replay and 3/12 safe task-completing recovery conditional on live Base
catastrophe. See `docs/PILOT_B_FRESH_H20_20260812.md`. Pilot C and training
remain blocked for this placement/controller design.

## Targeted oracle repair result

Quest job `9094715` (detached diagnostic commit
`e192b46817b7a60d40cfac223006be3f5afd28c8`) reran only the previously failing
`glass_recovery_train_0001` at H=20 with a 360-action oracle ceiling. It
completed with exit `0:0` in 3 minutes 27 seconds.

| Measurement | In-memory rewind | Serialized rebuild |
|---|---:|---:|
| Exact branch starts | 2/2 | 2/2 |
| Exact H=20 Base suffix replay | 1/1 | 1/1 |
| Collision on expected suffix index 19 | 1/1 | 1/1 |
| Oracle safe task success | 1/1 | 1/1 |
| Oracle glass-collision attempts | 0 | 0 |

Both reset paths selected the same recovery configuration on search attempt 2
and completed the original task in 147 actions. The matched off-path control
also completed in 70 actions. Together with the already successful
`glass_recovery_train_0000` result from job `9071778`, both exposed H=20
development catastrophes now have an exact replay and a safe task-completing
recovery witness.

The practical controller change is deliberately small: reuse a reachable
grasp pose observed in the matched collision-free control rollout, and retain
the stable post-close pose from the successful no-glass source trace as a
fallback. The detour still includes the safe radial departure waypoint and
held-bowl placement compensation. No learned model was trained or evaluated.

The result file is
`$HOME/crash_bench/results/glass_recovery_v2/pilot_b_repair_diag_20260812_v8_stable_grasp/continuation_oracle_diagnostic_h20.json`.
It is marked `diagnostic_only=true` and `cohort_admission_allowed=false`.

**Re-entry decision:** the known snapshot and oracle-controller blockers are
now sufficiently repaired to run one small, fresh, outcome-blind H=20
frontier. Do not spend more time hardening the exposed diagnostic. If the fresh
frontier has adequate Base crash yield, exact replay, and broad oracle success,
freeze H=20 and continue to Pilot C; otherwise revise only the scenario
geometry/controller eligibility rather than reopening state serialization.

## Final diagnostic result

The authoritative stopped result is:

- Quest job: `9071778`
- detached diagnostic commit: `67f66392b15806c0c47475b01db70b5d75055004`
- result: `$HOME/crash_bench/results/glass_recovery_v2/`
  `pilot_b_repair_diag_20260811_v6_h20_oracle360/`
  `continuation_oracle_diagnostic_h20.json`
- candidates: the first two exposed r7 placements, diagnostic-only
- takeover horizon: H=20 actions
- Base scan budget: 220 actions
- oracle budget: 360 actions

| Measurement | In-memory rewind | Serialized rebuild |
|---|---:|---:|
| Exact branch starts | 4/4 | 4/4 |
| Exact H=20 Base suffix replays | 2/2 | 2/2 |
| Collision on expected suffix index 19 | 2/2 | 2/2 |
| Oracle safe task successes | 1/2 | 1/2 |
| Start/restore errors | 0 | 0 |

The five identity layers matched at every checked branch start: simulator
state, projected controller state, complete continuation state, policy-boundary
observation, and captured model XML. The in-memory and serialized paths also
gave the same per-candidate oracle conclusion. This is the intended evidence
that the old exact-state continuation defect has been fixed rather than hidden
by a weaker hash.

### Candidate-level oracle outcome

`glass_recovery_train_0000` succeeded in both reset paths using the same first
configuration and 176 actions. It incurred no glass collision and reached
controller stage 10 before LIBERO task success.

`glass_recovery_train_0001` failed in both reset paths. All 16 searched
configurations were collision-free, but none satisfied the original-task
success predicate within 360 actions. The controller reached stage 12, i.e.
through its final release/settle program, so this is not adequately explained
as the old 220-action right-censoring problem. The matched off-path control
probe also failed to finish the task in 220 actions, so this placement does not
currently supply a broadly task-completing controller condition.

Observed oracle recoverability is therefore 50% in this two-catastrophe
diagnostic: it meets only the old permissive frontier threshold numerically and
is far below the near-90% condition needed for the later paper story. With
`n=2`, it is not an estimate of population performance and cannot qualify a
horizon.

## What was actually fixed

The diagnostic sequence isolated and repaired four concrete continuation
defects:

1. The snapshot omitted MuJoCo integration inputs, full OSC state, robot
   temporal buffers, episode counters, observable timers/cache, and the exact
   observation presented to the policy.
2. LIBERO's `ControlEnv` forwards a `robots` property, causing the adapter to
   stop unwrapping one layer too early; the real robosuite `done`, `timestep`,
   `_observables`, and `_obs_cache` were therefore not captured.
3. A fresh robosuite reset represents `cur_time` as integer zero. Scalar restore
   followed that temporary target type and truncated captured floating-point
   time; restore now follows the snapshot dtype.
4. Repeated `get_xml()` output was treated as a canonical model identity even
   though it is a serialization of the compiled model. The adapter now retains
   the exact XML source bytes actually associated with the continuation model.

The final local regression suite passes: `123 passed`. The main local and Quest
branches were not committed or pushed; Quest validation used detached commits
in `$HOME/crashbench-pilotb-repair-20260811`.

## Remaining problems

The current blocker is no longer state restoration. It is the scientific
existence of a sufficiently broad, task-completing oracle/controller class at
one common H:

- the repaired oracle is successful on one exposed geometry and fails after
  completing its motion program on the other;
- increasing the rollout ceiling from 220 to 360 did not rescue the failure;
- the failed placement's off-path Base continuation also did not complete in
  220 actions, so task controllability and placement eligibility are entangled;
- repeated fresh OpenVLA scans can still move the live collision time by a few
  actions. Exact continuation is now stable *after capture*, but independent
  fresh-episode determinism is a separate question;
- the diagnostic uses exposed r7 train placements and cannot be promoted into
  a cohort or used to choose favorable geometries.

Consequently no fixed H is frozen, no 10/5/5 dataset exists, and Pilots C--F
remain unrun.

## Stopping decision and legitimate re-entry options

Work stops at job `9071778`; no new frontier, collection, or training job is
authorized by this diagnostic.

Before resuming, choose and predeclare one of two routes:

1. **Improve the oracle/controller class.** Diagnose the failed placement's
   final bowl/plate pose and LIBERO predicate after stage 12, repair grasp/place
   geometry without using new frontier outcomes, freeze the controller and its
   rollout budget, then test on fresh outcome-blind placements.
2. **Tighten scenario eligibility outcome-blindly.** Define a cheap,
   pre-OpenVLA controller/task feasibility screen that does not use Base crash
   or oracle recovery outcomes, then author a new placement manifest and run a
   fresh frontier once. The screen and exclusions must be frozen before seeing
   frontier outcomes.

Neither route may reuse the successful r7 cells as training data or select
scenes based on this diagnostic. Only a fresh frontier with broad exact replay
and oracle success at a single H can reopen Pilot C.

## Repaired continuation contract

The old branch snapshot contained `[time, qpos, qvel]`, three MuJoCo arrays,
and a partial OSC/interpolator snapshot.  The repaired snapshot additionally
contains:

- all available MuJoCo integration inputs (`qacc_warmstart`, actuator/control,
  applied forces, mocap, equality, userdata, and plugin state);
- OSC goals, gains, initial joint reference, torques, and interpolators;
- robosuite time/episode counters and robot temporal buffers;
- observable values, timers, sampled flags, and the observation cache;
- the exact injected model XML used at capture time.

Same-scene branches must now match simulator, controller, full continuation,
observation, and model hashes against the values captured in the live Base
rollout.  A reset-derived hash is no longer allowed to define its own expected
value.  Cross-scene off-path controls restore dynamics/controller state but
intentionally regenerate observations after moving the glass.

## Oracle repair

The path-aligned detour now first moves radially away from the glass at the
current height and lifts from that safe point before entering the lateral
lane.  This removes the old diagonal lift-and-sidestep segment that could cut
through the glass at late trigger states.  Carry and placement stage indices
are now derived from the waypoint list instead of being hard-coded.

## Required diagnostic

Run the exposed r7 placement manifest through the diagnostic-only 2x2 at one
candidate H (start with H=20):

```bash
export CB_PILOT_B_STAGE=diagnose
export CB_PILOT_B_ROOT="$HOME/results/glass_recovery_v2/pilot_b_repair_diag_20260811"
export CB_PILOT_B_DIAGNOSTIC_PLACEMENTS=/absolute/path/to/r7/frontier_placements/placements.json
export CB_PILOT_B_DIAGNOSTIC_H=20
export CB_PILOT_B_DIAGNOSTIC_CANDIDATES=6
setup/submit_glass_recovery_pilot_b.sh diagnose
```

The output has reset-method rows (`in_memory`, `serialized`) and branch columns
(`base_suffix`, `oracle_recovery`).  Interpret it as follows:

| Result | Interpretation |
|---|---|
| in-memory Base exact, serialized Base not exact | model serialization/rebuild is still defective |
| neither Base cell exact | dynamic continuation snapshot or event determinism is still defective |
| both Base cells exact, oracle fails in both | remaining bottleneck is controller/geometry recoverability |
| in-memory oracle succeeds, serialized oracle fails | rebuild still changes recovery conditions |
| all four cells pass broadly | repair is eligible for a fresh outcome-blind frontier |

Do not rerun the full six-H frontier until every observed live Base catastrophe
in the diagnostic has exact replay in both reset rows.  Oracle success should
be read separately from replay success; the diagnostic reports both instead of
using accepted three-branch pairs as a proxy for oracle recoverability.

## Re-entry rule

After the diagnostic passes, capture or reserve fresh source states, author a
new placement manifest without observing Base/oracle outcomes, and run the
frontier from scratch.  Do not promote any r7 placement or the diagnostic's
successful cells.  Pilot C remains blocked until that new frontier freezes one
common H and the fixed-H collection reaches its declared split counts.
