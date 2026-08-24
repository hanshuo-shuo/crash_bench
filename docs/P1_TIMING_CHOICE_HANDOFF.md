# P1 timing-choice handoff

## Current state

P1 now uses a trajectory group (`source_state_sha256 + placement_id +
condition`) with matched T-30/T-20/T-10/T-5 states.  The schema, authoring
script, Quest smoke entrypoint, small promoted reports, and unit tests are in
this branch.

Completed Quest jobs:

| Job | Sources | Decisions | Option rollouts | Result |
|---|---:|---:|---:|---|
| `9761440` | 1 | 12 | 36 | pipeline proof; recoverable at T-30/T-10 |
| `9762786` | 2 | 24 | 72 | one library-unsolved trajectory; directional Retreat fails at 0004 T-5 |

Both jobs ran at clean commit `7ac05defa388`.  Their raw outputs exist on Quest
and in the local ignored result tree:

```text
results/counterfactual_router/p1_timing_choice_smoke_7ac05de_20260818_r1
results/counterfactual_router/p1_timing_choice_smoke_7ac05de_20260818_r2
```

The promoted compact evidence is
`results/P1_TIMING_CHOICE_SMOKE_20260818.md`.  Do not add the ignored raw NPZ
files to Git.

## Why the next run changed the third option

For placement `glass_recovery_heldout_0004`:

```text
T-20: Base catastrophe, Detour success, Retreat safe noncompletion
T-10: Base catastrophe, Detour success, Retreat safe noncompletion
T-5 : Base catastrophe, Detour safe noncompletion, Retreat catastrophe
```

The historical `RetreatHold` always moves in world `-x`; at T-5 that motion can
hit the glass.  P1 therefore adds `FailSafeHold`, a zero-Cartesian-delta option,
and exposes it as `--retreat-mode hold`.  Historical runs remain unchanged
because the collector default is still `--retreat-mode retreat`.

## Immediate next run

After pulling the pushed branch on Quest, validate only placement 0004, only
the glass condition, and all four timing anchors.  The maintained sbatch file
already passes `--conditions glass` and `--retreat-mode hold`.

```bash
ssh -S /tmp/quest.sock quest.northwestern.edu \
  'cd /gpfs/home/shv7753/crash_bench && git pull --ff-only origin codex/glass-recovery-d0-fprime-rescue'

ssh -S /tmp/quest.sock quest.northwestern.edu \
  'cd /gpfs/home/shv7753/crash_bench && sbatch --parsable \
    --export=ALL,CB_P1_OUTPUT=results/counterfactual_router/p1_timing_choice_hold_20260818_r3,CB_P1_PLACEMENT_IDS=glass_recovery_heldout_0004,CB_P1_TARGET_VALID=1 \
    setup/timing_choice_pilot.sbatch'
```

Acceptance for this smoke:

- one valid placement;
- four glass decisions and twelve option rollouts;
- `decision_features.npz` retained;
- `FailSafeHold` is `safe_noncompletion` at T-5;
- a timing pair exists if an earlier anchor is recoverable and a later anchor
  is loss-control under the hold fail-safe.

Then author the result:

```bash
PYTHONPATH=. envs/openvla/bin/python scripts/author_timing_choice_benchmark.py \
  --capture results/counterfactual_router/p1_timing_choice_hold_20260818_r3 \
  --output results/counterfactual_router/p1_timing_choice_hold_20260818_r3/p1_authoring
```

If Hold is safe but no timing pair appears, do not scan more random sources.
Add later predeclared anchors (T-3/T-1) or construct a hazard-relative retreat.
If Hold itself is unsafe, the branch state is already dynamically unrecoverable
and the fail-safe contract needs simulator/controller braking rather than a new
router.

## After the hold contract works

1. Run a small source-disjoint timing cohort with glass only.
2. Add a predeclared near-path clearance band to create strict unnecessary
   states where Base succeeds and Detour is worse.  Do not use no-glass states
   as the primary unnecessary class.
3. Freeze the four-class balanced manifest and macro-average classes.
4. Only then add from-reset dynamic first-crossing evaluation.

Geometry Gate remains a diagnostic P0 result and is not the organizing variable
of P1.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests -q
PYTHONDONTWRITEBYTECODE=1 python scripts/audit_repo.py
git diff --check
```

At handoff these checks report 163 tests passed and a clean repository audit.
