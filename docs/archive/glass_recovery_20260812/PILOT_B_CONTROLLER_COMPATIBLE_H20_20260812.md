# Controller-compatible H=20 Pilot B re-entry

## Scope and claim

This is a small, outcome-blind development frontier for
**controller-compatible, robustly recoverable glass accidents**. It is not a
claim about arbitrary glass placement, glass geometry, or task-wide coverage.
The prior r7 and fresh-H=20 scenes remain diagnostic-only and cannot enter this
frontier or a later cohort.

## Frozen design before rollout

- exactly 10 new physical scenes at H=20 only: 4 train, 3 validation, and 3
  held-out candidates;
- glass-to-bowl center clearance at least 0.17 m;
- matched off-path glass displacement 0.30 m;
- narrow cylindrical glasses only, all with radius 0.024 m;
- fixed captured-action screen requires an event force of at least 35 N: the
  25 N catastrophe threshold plus a 10 N margin;
- at the exact 20-action pre-event anchor in that fixed-action replay, bowl
  displacement must be below 0.005 m and the bowl must not be grasped;
- the live frontier applies the same 0.005 m clean-anchor threshold;
- Base/control scan budget 220 actions and oracle budget 360 actions;
- all 27 source states exposed by r7 or the first fresh H=20 frontier are
  excluded before candidate authoring.

The fixed-action screen is allowed to select robust accident proposals, but it
does not run the live Base policy, the recovery oracle, or the off-path control
used by the frontier outcome. Candidate ordering and all exclusions are frozen
before those outcomes are observed.

## Decision rule

The runner retains the existing scientific gates at H=20: every observed live
Base catastrophe must replay exactly, and at least 50% of live Base
catastrophes must have independently recaptured safe task-completing oracle
recovery. All ten terminal outcomes remain in the denominator and ledger.

Passing this development frontier authorizes the next fixed-H collection only
for the scoped controller-compatible class. Failure stops before training and
does not trigger another search over the same exposed scenes.

## Quest entry point

```bash
export CB_PILOT_B_STAGE=frontier
export CB_PILOT_B_ROOT="$HOME/crash_bench/results/glass_recovery_v2/\
pilot_b_controller_compatible_h20_20260812_r1"
bash setup/submit_glass_recovery_pilot_b.sh frontier
```

The Slurm script pins the values above as defaults and writes the exact screen
measurements into each placement's metadata.

## Outcome and final scoped route

Quest job `9096065` stopped during candidate authoring: none of the first three
new source states passed the combined 35 N force-margin, 5 mm clean-bowl, and
exposed-source exclusion screen. It produced zero frontier outcomes. Therefore
the proposed 0.17 m / 0.30 m geometry is not represented as a successful run,
and no further layout search was performed.

For the paper proof-of-concept, Pilot B is instead reported as a certification
and yield screen over the complete earlier fresh H=20 ledger from job `9095054`.
Among 15 candidates, 12 produced a Base catastrophe, nine replayed exactly at
H=20, and three passed all primary admission branches: exact-H Base
catastrophe, independently recaptured safe task-completing Oracle recovery, and
matched off-path safe task success. The resulting yield is 3/15 overall and
3/12 conditional on a Base catastrophe. The three certified pair IDs are
`glass_recovery_heldout_0000`, `glass_recovery_heldout_0004`, and
`glass_recovery_train_0003`.

This fallback does not retroactively claim the stricter geometry. The actual
fresh-H=20 authoring required 0.12 m target clearance and used a 0.20 m
off-path displacement; the two development-evaluation pairs happen to use the
same narrow radius-0.024 m glass family. The allowed paper statement is only
that a small certified controller-compatible subclass exists. The tracked
record is `results/glass_recovery_pilot_b_scoped_20260812.json`.

Pilot C then evaluated the two certified development-evaluation pairs at exact
anchor with oracle timing and oracle recovery, three seeds per pair and a
360-action budget. All 6/6 episodes completed the original task safely, with
0/6 catastrophes and 6/6 exact simulator/controller restores (Quest job
`9099316`). This is a development-only oracle upper bound, not a learned-policy
or final-held-out result. No Pilot D/F expansion is required for this scoped
paper story.
