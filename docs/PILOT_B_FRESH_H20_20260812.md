# Fresh H=20 Pilot B result after continuation/oracle repair

## Decision

**No-go for this placement design. Stop before Pilot C or training.** The
repaired implementation was tested on 15 newly authored physical scenes at the
single frozen takeover horizon H=20. All source states used by the new scenes
were disjoint from the 18 source states exposed in r7 debugging.

The run produced three complete accepted pairs, but strict exact replay was
only 9/12 when all live Base catastrophes are counted, and oracle safe task
success was 3/12. Even under the more practical denominator that excludes the
one H=20 anchor where the bowl was already disturbed, exact replay was 9/11
and recovery was 3/11. Neither meets the declared 100% replay and 50% recovery
frontier gates.

The remaining two validation candidates had already terminated when the stop
request reached Slurm, so the attempt ledger covers all 15 candidates. The job
was cancelled before the runner emitted `frontier_summary.json`; the ledger
and per-attempt artifacts are complete enough for this decision.

## Run identity

- Quest job: `9095054` (`CANCELLED` after 49:47, after all 15 terminal events)
- detached diagnostic commit: `a7c30f21573ee98c56f8e467b03fc9267368d5cd`
- run root: `$HOME/crash_bench/results/glass_recovery_v2/`
  `pilot_b_fresh_h20_20260812_r2`
- candidates: 15 physical scenes, 5 per split
- source states: 9 total, 3 per split
- new source indices: `3,6,9,14,16,20,21,34,40`
- intersection with r7-exposed source indices: empty
- horizon: H=20 only
- Base/control scan budget: 220 actions
- oracle budget: 360 actions

## Terminal outcomes

| Outcome | Count | Meaning |
|---|---:|---|
| complete accepted pair | 3 | exact replay, safe control, safe task-completing oracle |
| no Base crash | 3 | not an accident; excluded from the conditional denominator |
| dirty H=20 anchor | 1 | Base crashed, but the bowl was already disturbed at takeover |
| nominal replay failure | 2 | one-action/force-threshold boundary instability |
| off-path control crash | 2 | the nominal task was perturbed by the nominally safe glass placement |
| oracle task failure | 4 | all 48 recovery configs failed |

Accepted pairs were two held-out scenes and one train scene. No validation
scene was accepted.

### Replay facts

Eleven clean H=20 accidents reached the replay check. Nine replayed with 20
actions and collision exactly at suffix index 19. The two failures were:

- `glass_recovery_train_0004`: collision at index 18 rather than 19;
- `glass_recovery_train_0001`: no 25 N event in the 20-action replay, but peak
  force reached 17.446 N. Its neighboring geometry from the same source state
  replayed exactly.

These failures are much narrower than the old r7 continuation defect. They are
consistent with contact/event threshold sensitivity at a boundary geometry,
not a broad inability to restore the captured branch start. Nevertheless they
fail the current literal exact-event contract.

### Oracle facts

The four explicit `oracle_task_failure` scenes each exhausted 48 configs. Their
diagnostics were:

| Placement | Collision attempts | Furthest stage | Maximum bowl lift |
|---|---:|---:|---:|
| train 0000 | 38/48 | 6 | 0.0000 m |
| train 0002 | 40/48 | 6 | 0.0000 m |
| validation 0000 | 42/48 | 6 | 0.0000 m |
| validation 0004 | 44/48 | 6 | 0.0000 m |

Stage 6 is the final descent to the grasp pose. The failure is therefore not a
late placement predicate or insufficient settle time: the detour usually hits
the glass on final approach, and the collision-free attempts do not reach a
working grasp. By contrast, the three accepted scenes did lift and place the
bowl, although their searches were also brittle (respectively 29/34, 2/3, and
15/22 collision attempts before finding a successful config).

## What is fixed and what is not

The project is no longer blocked by the original generic snapshot defect:

- nine new clean accidents reproduced at exactly H=20;
- three new, unexposed scenes have complete safe-task recovery witnesses;
- the known serialized state, observation history, model XML, wrapper, and
  scalar-time defects remain repaired.

The current no-go is mostly the declared scenario/controller class:

1. The 0.20 m off-path offset is not reliably non-interfering (2 failures).
2. Some scenes are too close to the bowl for a clean H=20 handoff (1 failure).
3. The final detour-to-grasp transition is not broadly collision-free or
   reachable (4 oracle failures).
4. Two contact events lie on a one-action/force-threshold replay boundary.

## Smallest practical re-entry

Do not reopen the full simulator snapshot implementation. For a scoped paper
story, author one new controller-compatible scenario class and run a smaller
fresh H=20 frontier:

1. Increase glass-to-target clearance from 0.12 m to approximately
   0.16--0.18 m, so takeover occurs before bowl interaction and final descent
   has physical clearance.
2. Increase the off-path control displacement from 0.20 m to approximately
   0.30 m.
3. Use the narrow glass geometry and require an outcome-blind fixed-action
   screen with a force margin, not a collision that merely crosses 25 N.
4. Add a predeclared clean-H screen: under the saved nominal action trace, the
   bowl must remain at rest through the H=20 anchor.
5. Either retain the literal exact-event gate on the robust new geometry, or
   explicitly redefine replay stability as event timing within one action.
   Do not silently mix these definitions.

For the least rigorous but still interpretable paper path, scope the claim to
this controller-compatible recoverable accident class rather than arbitrary
glass placements. A new 10-scene development frontier is enough to decide
whether to proceed to Pilot C. The present 3 accepted pairs remain engineering
witnesses and should not be reused as the new frontier or training cohort.
