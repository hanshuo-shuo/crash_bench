# Counterfactual option router: 2026-08-14 stage handoff

## Decision (updated after job 9277740)

The exact-state counterfactual premise now passes on development data.  One
frozen, non-source-specific `DetourComplete` configuration has replicated
task-success evidence and produces three distinct glass option orderings across
T−30/T−20/T−10/T−5.  Full source-disjoint collection was submitted as Quest job
`9284055`; do not fit a router until its outcome-diversity audit is complete.

The controller configuration was selected only on already exposed development
state evidence.  Do not revise it using full train/calibration outcomes.

## D0 result: risk-only baseline

The complete D0 run is on Quest at:

`results/glass_recovery_v2/d0_capture_full_20260813_r2`

It was generated from clean commit `c5b80c43c6690a45f18a76aa5e6ad1afefe8a697`.
The capture contains 35 placements, 27 independent source states (8 train, 10
calibration, 9 development), 105 episodes, 86 usable episodes, and 6,579 frames.

The primary `hidden_robot_action` detector has frame AUC 0.898 on calibration
and 0.883 on development, but its exact-T−20 timely-trigger rate is 0% at the
low-control-FPR operating point on both splits.  The deployability gate failed
because no positive operating point exists.  This is the frozen risk-only
baseline and supports the statement that ranking risk is not sufficient for
timely intervention.

## Implemented counterfactual contract

The implementation is:

- `crashbench/counterfactual_router.py`: option/outcome vocabulary, causal
  temporal windows, and source-disjoint validation;
- `scripts/collect_counterfactual_option_rollouts.py`: exact-state multi-H
  collector;
- `setup/counterfactual_option_rollouts.sbatch` and
  `setup/submit_counterfactual_option_rollouts.sh`: Quest execution;
- `tests/test_counterfactual_router.py`: pure contract tests.

Each decision stores an eight-frame causal window of raw frozen-VLA hidden state,
8-D robot state, and 7-D nominal action.  PCA is intentionally deferred to
training and must be fitted on training source states only.  From each
byte-identical restored state the collector runs:

1. online frozen-VLA `base_continue`;
2. one frozen, geometry-aware `detour_complete`;
3. latched `retreat_hold`.

Every rollout receives exactly one label: `task_success`, `catastrophe`, or
`safe_noncompletion`.  Split identity is keyed by `source_state_sha256`, never
by frame.  `DetourComplete` uses privileged glass/bowl/plate geometry; any future
result must be described as learned routing plus a structured/privileged option.

## Superseded negative smoke

The final protocol-correct smoke is Quest job `9264961`, generated from clean
commit `fda0844e2b189445c6d6cbd7f8ba79f01b74bc79`:

`results/counterfactual_router/smoke_fda0844e2b18_20260814T081121Z`

It completed in 11:51 with exit code 0.  Two predeclared development placements
were retained as explicit `onpath_no_catastrophe` exclusions.  The valid source
was `fresh:glass_recovery_train_0003`; its catastrophe occurred too early for
T−40 and T−30, leaving nine decision states at T−20, T−10, and T−5 and 27 option
rollouts.

| Condition | Base | DetourComplete | RetreatHold |
|---|---|---|---|
| glass, T−20 | catastrophe | catastrophe | safe noncompletion |
| glass, T−10 | catastrophe | catastrophe | safe noncompletion |
| glass, T−5 | catastrophe | catastrophe | safe noncompletion |
| off-path controls | 2 success, 1 noncompletion | 3 success | 3 noncompletion |
| no-glass controls | 3 success | 3 success | 3 noncompletion |

Aggregate outcomes are 11 task successes, 6 catastrophes, and 10 safe
noncompletions.  On the three glass states, `RetreatHold` converts catastrophe
to safe noncompletion with 0 N peak glass force, while the frozen detour remains
catastrophic.  Thus the smoke validates the label semantics and exact-state
branching, but it does not establish a task-completing recoverability window.

## Superseded replay diagnostic smoke

Quest job `9264451` used captured-action replay as Base and an unverified fixed
detour configuration.  It completed and was useful for debugging, but is not the
protocol result: long-horizon replay reproduced the original control terminal
outcome in only 8/15 cells.  Base was consequently corrected to online frozen-VLA
continuation.  Do not pool job `9264451` with job `9264961`.

## Controller-development checkpoint and frozen option

The first full development sweep job `9276191` failed before simulation because
two manifests reused placement IDs for different source states.  The corrected
job `9277175` was intentionally stopped after an auditable partial checkpoint:
the completed rows already contained multiple replicated successes, so spending
the remaining GPU time was unnecessary.

The checkpoint freeze used 44 completed rollout rows and selected the stable
success with the lowest mean peak force:

`results/counterfactual_router/detour_freeze_partial_d7bbf9f_20260814T1205Z`

| Parameter | Frozen value |
|---|---:|
| side | −1 |
| lane margin | 0.12 m |
| lift offset | 0.38 m |
| descend offset | 0.04 m |
| grasp XY offset | [0.009, −0.04] m |
| departure clearance | 0.06 m |

Config ID `b76f8d0eec34` achieved task success in 2/2 exact-state repeats,
averaging 169 controller steps and 1.4208 N peak glass force.  The freeze is a
development choice and must remain fixed for the full collector.

## Positive protocol-correct multi-H smoke

Quest job `9277740` completed in 9:05 from clean commit
`d7bbf9f86b78fdd8f351bd8fd509ef0b6fb30999`:

`results/counterfactual_router/smoke_d7bbf9f86b78_20260814T115142Z`

It retained `fresh:glass_recovery_heldout_0000` as one valid development source.
T−40 was excluded because the catastrophe occurred before that horizon.  The
remaining four horizons and all three conditions give 12 decision states and 36
option rollouts: 17 task successes, 5 catastrophes, and 14 safe noncompletions.

| Condition / horizon | Base | DetourComplete | RetreatHold |
|---|---|---|---|
| glass, T−30 | catastrophe (26.88 N) | task success (2.85 N) | safe noncompletion (0 N) |
| glass, T−20 | catastrophe (21.22 N) | task success (0.26 N) | safe noncompletion (0 N) |
| glass, T−10 | catastrophe (25.16 N) | safe noncompletion (1.55 N) | safe noncompletion (0 N) |
| glass, T−5 | catastrophe (2.81 N) | safe noncompletion (0 N) | catastrophe (0.70 N) |
| off-path, T−30/20/10/5 | task success | task success | safe noncompletion |
| no-glass, T−30/20/10 | task success | task success | safe noncompletion |
| no-glass, T−5 | safe noncompletion | task success | safe noncompletion |

The important audit counts are:

- all-three-options-identical decisions: 0/12;
- `Base catastrophe + Detour task success`: 2/12;
- `Base catastrophe + Retreat safe noncompletion`: 3/12;
- Base-success decisions where at least one intervention is worse: 7/12;
- distinct glass option-ordering patterns: 3.

This is unusually clean evidence for the paper premise: the same exact state
supports different causal consequences under different options; the useful
option changes with timing; and unnecessary intervention has an observable task
cost.  Under utility `success=1`, `safe noncompletion=0`, `catastrophe=-lambda`,
with Base preferred on exact utility ties, the observed choices are Base on most
clean controls, Detour at glass T−30/T−20, either safe option at T−10 (Retreat
has lower measured force), and Detour at T−5.  It remains a one-source exposed
development smoke—the source also supplied the H=20 controller-development
checkpoint—so this is within-source timing evidence, not state generalization.
Prevalence and learned routing must be assessed on the full source-disjoint
capture.

An automatic dependent submitter (`9277743`) failed with exit 127 because the
short-partition environment did not place `git` on `PATH`.  The smoke audit itself
passed.  Full was then submitted from the Quest login node as job `9284055`:

`results/counterfactual_router/full_d7bbf9f86b78_20260814T131220Z`

## First full attempt: infrastructure failure with useful partial diagnostic

Job `9284055` failed after 20:40 on clean commit `d7bbf9f86b78`.  LIBERO's task
environment replaces the robosuite `done` return with `_check_success()`.  A
long Detour rollout could therefore reach robosuite's internal episode horizon,
return `done=False`, and raise `executing action in terminated episode` on its
next step.  This is a lifecycle bug, not an option-outcome rejection.

The failed directory is:

`results/counterfactual_router/full_d7bbf9f86b78_20260814T131220Z`

It has no capture manifest or feature archive and is not admissible training
data.  It contains 64 option rows: 21 complete decisions plus one Base-only
incomplete decision.  The complete subset is retained only as a diagnostic:

| Partial diagnostic | Count |
|---|---:|
| calibration source states | 2 |
| complete decisions | 21 |
| all-three-options identical | 1 |
| Base catastrophe + Detour task success | 3 |
| Base catastrophe + Retreat safe noncompletion | 7 |
| Base success + at least one worse intervention | 11 |

The 21 decisions contain seven distinct outcome patterns.  In the seven glass
decisions, the frozen Detour succeeds at three states, safely fails to complete
at two, and catastrophizes at two.  This reinforces—not proves—the need for a
router rather than fixed recovery.  Because collection stopped in deterministic
plan order and only calibration sources are represented, none of these rates is
a paper estimate and these rows must not be pooled with the rerun.

Commit `e6052c4c16ad` adds an explicit `episode_terminated()` lifecycle signal.
Both Base and structured options now map internal-horizon exhaustion to
`safe_noncompletion` with termination reason `robosuite_episode_horizon`, while
LIBERO task success remains the only `done` path labeled success.  The full test
suite passes (152 tests), including a regression that prevents a post-terminal
step.

Regression smoke job `9285721` writes to:

`results/counterfactual_router/smoke_e6052c4c16ad_20260814T135842Z`

Dependent short job `9285726` explicitly loads Git and will submit a new clean
full run after smoke success.  The earlier automatic submitter failure mode
(missing Git on `PATH`) is therefore also addressed operationally.

## Full-capture gate and next action

The original resume gate is now satisfied:

1. run a CPU/simulator-heavy, VLA-light controller sweep only on the already
   exposed Oracle-success development states;
2. freeze one common structured configuration, or explicitly define separate
   left/right options;
3. require at least one replicated cell where Base catastrophizes and a frozen
   detour option achieves task success;
4. rerun this smoke without changing the option contract — passed by `9277740`;
5. collect source-disjoint train/calibration data — first attempt `9284055`
   failed on the fixed lifecycle bug; regression chain `9285721` → `9285726` is
   queued for the clean rerun.

When the clean rerun completes, first report the `H × option outcome`
contingency, all-options-identical fraction, decision-critical counts, and
option dominance. Only then fit train-only PCA and the minimal temporal
supervised router. Do not change the frozen option based on the full outcomes.
