# Counterfactual option router: completed full-capture handoff

## Current decision

The exact-state counterfactual premise passes the full audit. The official Quest
capture is job `9288681`, completed with exit code 0 in 3:22:56 from clean commit
`d4751330395eec11691f4830761b76f5f1ac36b7`:

`results/counterfactual_router/full_d4751330395e_20260814T152231Z`

It contains 20 source states, 273 matched decisions, and 819 option rollouts.
Only 21/273 decisions (7.69%) have identical outcomes under all three options;
252/273 (92.31%) therefore carry counterfactual outcome information. The first
minimal source-disjoint supervised router is now complete; see
[COUNTERFACTUAL_ROUTER_MINIMAL_RESULT.md](COUNTERFACTUAL_ROUTER_MINIMAL_RESULT.md).
Do not modify the frozen options after seeing these outcomes.

The machine-readable aggregate audit is
`results/counterfactual_router_full_audit_20260815.json`; regenerate it with
`scripts/analyze_counterfactual_option_capture.py`.

## Frozen scientific contract

Each decision uses an eight-frame causal history of the frozen VLA's 4096-D
hidden state, 8-D robot state, and 7-D nominal action. From a byte-identical
restored state, the collector evaluates:

1. online frozen-VLA `base_continue`;
2. frozen geometry-aware `detour_complete`;
3. latched `retreat_hold`.

Every rollout has exactly one outcome: `task_success`, `catastrophe`, or
`safe_noncompletion`. Splits are keyed by source-state hash, never by frames or
anchors. PCA and all learned preprocessing must be fitted on training sources
only. `DetourComplete` uses privileged geometry, so the supported claim is
learned routing over structured/privileged options, not end-to-end learned
recovery.

The frozen detour is config `b76f8d0eec34`: side −1, lane margin 0.12 m, lift
offset 0.38 m, descend offset 0.04 m, grasp XY offset `[0.009, -0.04]` m, and
departure clearance 0.06 m. It was selected on exposed development evidence
before the full capture.

## Full-capture integrity

| Item | Value |
|---|---:|
| attempted / valid placements | 35 / 23 |
| source states | 20 |
| train / calibration / development sources | 5 / 7 / 8 |
| matched decision states | 273 |
| option rollouts | 819 |
| success / catastrophe / safe noncompletion rows | 285 / 158 / 376 |

The final manifest sealed the feature archive and decision metadata. Declared
exclusions were 14 `catastrophe_before_horizon`, 30
`condition_terminal_before_matched_anchor`, 7 `onpath_no_catastrophe`, and 5
`scan_invalid_initial_state`. These reduce available anchors; they are not
silently relabeled rollouts.

## Outcome-diversity audit

Utility is frozen for this first analysis as success `+1`, safe noncompletion
`0`, catastrophe `−5`, with Base preferred on exact utility ties.

| Method | Success | Catastrophe | Safe noncompletion | Intervention | Mean utility |
|---|---:|---:|---:|---:|---:|
| Base only | 57.51% | 33.70% | 8.79% | 0% | −1.110 |
| Always Detour | 46.89% | 14.65% | 38.46% | 100% | −0.264 |
| Always Retreat | 0% | 9.52% | 90.48% | 100% | −0.476 |
| Counterfactual Oracle | 68.86% | 3.66% | 27.47% | 33.33% | 0.505 |

The Oracle chooses Base / Detour / Retreat on 182 / 68 / 23 decisions and gains
1.615 mean utility over Base. It is a retrospective ceiling, not a deployable
result. Its remaining ten catastrophes are states where no offered option avoids
catastrophe.

Decision-critical counts directly reject a fixed-recovery interpretation:

- `Base catastrophe + Detour success`: 22;
- `Base catastrophe + Retreat safe`: 73;
- `Base success + Detour worse`: 60;
- `Base success + Retreat worse`: 157;
- positive Oracle value over Base: 91/273 (33.33%).

In glass alone (101 decisions), Base catastrophizes on 91 (90.10%), while
Detour succeeds on 23 but also catastrophizes on 35. The Oracle chooses Base /
Detour / Retreat on 19 / 59 / 23, reaching 23.76% success, 8.91% catastrophe,
and 67.33% safe noncompletion. Thus Detour is useful but clearly not universally
best. On no-glass and off-path controls, Base succeeds on 91.86% and 88.37%; the
Oracle intervenes on only 5.81% and 4.65%, respectively.

Glass timing also changes option value:

| Horizon | N | Base catastrophe | Detour success | Detour catastrophe | Oracle catastrophe | Oracle choice B/D/R |
|---:|---:|---:|---:|---:|---:|---:|
| T−40 | 11 | 81.82% | 9.09% | 18.18% | 9.09% | 3 / 7 / 1 |
| T−30 | 21 | 80.95% | 19.05% | 14.29% | 4.76% | 5 / 15 / 1 |
| T−20 | 23 | 91.30% | 30.43% | 26.09% | 0% | 2 / 16 / 5 |
| T−10 | 23 | 95.65% | 30.43% | 43.48% | 4.35% | 2 / 12 / 9 |
| T−5 | 23 | 95.65% | 17.39% | 60.87% | 26.09% | 7 / 9 / 7 |

The late collapse of Detour and increased use of Retreat provide the desired
timing-dependent boundary. T−40 has only 11 valid decisions, so the table is
descriptive rather than a monotonicity claim.

Held-out split audits remain diverse: train 69/71, calibration 84/96, and
development 99/106 decisions have non-identical option outcomes. Evaluation
must nevertheless aggregate uncertainty by source state because multiple
horizons and conditions from one source are correlated.

| Split | Sources / decisions | Base success / catastrophe | Oracle success / catastrophe | Oracle utility gain |
|---|---:|---:|---:|---:|
| train | 5 / 71 | 54.93% / 33.80% | 76.06% / 1.41% | +1.831 |
| calibration | 7 / 96 | 60.42% / 29.17% | 67.71% / 4.17% | +1.323 |
| development | 8 / 106 | 56.60% / 37.74% | 65.09% / 4.72% | +1.736 |

The counterfactual ceiling improves catastrophe and mean utility in every
source-disjoint split. This supports a stable decision-value premise; it does
not establish that a learned router can recover that value.

## Minimal router result and next action

The train-only PCA-16 plus source-balanced linear utility router recovers 34.78%
of Oracle decision value on development, versus 14.67% for binary-risk
supervision. It lowers catastrophe from 29.25% to 23.58% and raises success from
39.62% to 46.23% relative to that risk baseline at a nearby intervention rate.
It does not beat Always Detour under the frozen λ=5 utility, so the result is
positive method evidence rather than a completed deployment claim.

The predeclared single-frame hidden+robot+action ablation is descriptively
stronger than the 8-frame primary, reaching 54.72% success, 22.64% catastrophe,
39.62% intervention, and 42.39% Oracle value recovery. That architecture was
subsequently frozen and tested in fresh online rollouts; the 8-frame result
remains the original development primary rather than being rewritten after
inspection.

Include the completed E13 hazard-specific glass prompt as a second whole-episode
baseline in that fresh evaluation. Existing E13 evidence is already informative
(2/15 treatment catastrophes but 0/15 task successes and 13/15 safe aborts), but
it comes from a different five-scenario cohort and cannot be inserted into the
106-decision exact-state table. Run the exact prompt from reset on the same new
episodes as Base and the router.

## Fresh online frontier: completed

The frozen single-frame outcome router has now been evaluated online. The
default-state run (Quest `9677761`, commit `2eab4a4a53dc`) retained five eligible
sources and is an operating-point pilot. The independent random-reset chain
(prepare `9679136`, full `9679137`, collection commit `9ed5ad656a08`) sampled 30
new task-0 resets, retained 25 nominal successes, and collected eight eligible
sources / 24 matched decisions. The router JSON and NPZ hashes are identical
across cohorts; source overlap is zero.

The independent cohort passes the requested frontier audit. At the predeclared
`lambda=1,target=0.6` point, Router / rate-matched risk / Always Detour have:

| Method | Success | Catastrophe | Safe noncompletion | Intervention |
|---|---:|---:|---:|---:|
| Router | 87.50% | 8.33% | 4.17% | 58.33% |
| Binary risk → Retreat | 41.67% | 8.33% | 50.00% | 50.00% |
| Always Detour | 70.83% | 4.17% | 25.00% | 100% |
| Base | 66.67% | 33.33% | 0% | 0% |
| Counterfactual Oracle | 91.67% | 0% | 8.33% | 33.33% |

Router beats risk by 45.83 success points at identical catastrophe point
estimate and beats Always Detour by 16.67 success points with 41.67 fewer
intervention points. The paired source-bootstrap success differences exclude
zero; the Router-minus-Detour catastrophe interval includes equality. Router
also retains 15/16 off-path/no-glass task successes, versus 9/16 for the hazard
prompt.

Four predeclared `(lambda,target)` points satisfy all four criteria in the
eight-source cohort: `(1,0.6)`, `(3,0.7)`, `(5,0.6)`, and `(8,0.6)`. Across all
13 eligible sources, the frontier satisfies every criterion but no one point
satisfies all four. Keep both statements: this is a positive frontier result
with cohort heterogeneity, not a universal fixed-policy dominance claim. Exact
point estimates, intervals, job IDs, hashes, and artifact paths are in
[`counterfactual_router_fresh_online_20260817.json`](../results/counterfactual_router_fresh_online_20260817.json).

## Provenance not to pool

| Job | Status | Use |
|---|---|---|
| `9277740` | successful positive smoke | exposed development timing evidence only |
| `9284055` | failed after 21 complete decisions | lifecycle diagnostic only |
| `9285934` | failed after 45 complete decisions | branch-start diagnostic only |
| `9288443` | successful final smoke | regression evidence only |
| `9288681` | successful full | sole official full capture |

The two failed directories have no sealed final manifest/features and must never
be pooled with the official capture. Their failures motivated the internal
episode-horizon handling and pre-catastrophic-anchor exclusions now covered by
tests.

## D0 motivation retained

The frozen D0 run at `results/glass_recovery_v2/d0_capture_full_20260813_r2`
has calibration/development frame AUC 0.898/0.883 but 0% exact-T−20 timely
trigger rate at the low-control-FPR operating point. The paper-facing bridge is:

> Knowing that failure is likely is not the same as knowing whether
> intervention is beneficial.

Risk ranking is therefore a baseline; exact-state option consequences supply
the supervision for counterfactual action selection.
