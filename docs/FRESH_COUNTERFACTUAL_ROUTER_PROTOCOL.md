# Fresh online counterfactual router protocol

The experiment is complete. This file remains the frozen design/provenance
contract; final tables, figures, and interpretation are in
[COUNTERFACTUAL_ROUTER_MAIN_RESULT.md](COUNTERFACTUAL_ROUTER_MAIN_RESULT.md).

## Frozen method

The deployed model is intentionally linear and single-frame:

`train-only PCA-16 hidden + robot state + nominal action -> three option-specific softmax heads`

For each option it predicts `task_success`, `catastrophe`, and
`safe_noncompletion`.  At catastrophe cost `lambda`, the score is

`U(o) = p(task_success | h,o) - lambda * p(catastrophe | h,o)`.

Base remains active unless the best Detour/Retreat score exceeds the Base score
by the calibration-frozen margin `delta`.  The method freezes
`lambda in {1,2,3,5,8}` and calibration intervention targets from 10% through
90%; the 40% calibration target is the confirmation operating point.  A binary
Base-catastrophe logistic head, trained on the identical features and source
split, routes only to Retreat at rate-matched calibration thresholds.

PCA and both heads use the five training source states from the sealed full
capture.  Margins and risk thresholds use only seven calibration source states.
The eight development states are diagnostic only.  No fresh source outcome is
used to fit or select any parameter.

## Fresh matched cohort

The source-trace inventory has 50 LIBERO initial states, 47 of which complete
the no-glass task.  The 27 indices exposed by either previous placement manifest
are excluded before authoring.  All remaining 20 successful states receive one
placement from a frozen evaluation geometry.  Catastrophe/T-20 eligibility is
deferred to the actual online Base scan instead of duplicated in an authoring
replay.  The 0.12 m target-clearance boundary is inherited from the accepted
development placements rather than selected on fresh outcomes.  Five positions
previously known to miss the T-20 hazard screen remain
outside evaluation, the next three positions are source-exclusive mechanical
smoke, and the official full job uses the final 12 never-screened candidates.
Any live
eligibility failure is excluded as a whole source placement, never as an
individual method or condition.

If the finite 12-state default-initialization pool yields fewer than eight live
eligible sources, the confirmation cohort is completed from 30 task-0 random
reset initializations generated with consecutive seeds beginning at 2026081700.
This supplement is triggered only by the eligibility count, before reading any
router comparison, and keeps the already-frozen router, geometry, rollout seed,
T-20 rule, options, and analysis unchanged.  Its collector stops after eight
eligible sources, so it independently meets the planned 8--12 source range.
The paper analysis pools every eligible source from the default and random-reset
cohorts and clusters the bootstrap by source; neither cohort is selected by its
router result.

The default-initialization run retained only five eligible sources and is
therefore an operating-point pilot, not the confirmation sample.  Before any
random-reset matched outcomes were collected, its frontier fixed
`lambda=5, target=0.2` as the conservative secondary confirmation point: it was
the lowest nonzero-intervention router point and was not dominated by another
router point.  The originally declared `lambda=5, target=0.4` point remains the
primary audit row and is not rewritten; the independent eight-source supplement
reports both points and the complete predeclared frontier.

Each accepted source contributes an on-path glass, matched off-path glass, and
no-glass control.  Frozen Base first establishes the actual on-path collision.
At exactly T-20 actions, the simulator and controller state is restored and
Base, frozen DetourComplete, and RetreatHold are executed online from the exact
same state.  Model predictions select among those observed branches.  The exact
E13 hazard-specific prompt is additionally run from the same episode reset.

The comparison contains Base, hazard-specific prompt from reset, binary
risk-to-Retreat, Always Detour, Always Retreat, the frozen outcome-router
frontier, and the realized Counterfactual Oracle upper bound.

## Analysis and acceptance

The independent statistical unit is `source_state_sha256`.  All intervals and
paired method differences use a shared 5,000-replicate source-state cluster
bootstrap.  Reports include task success, catastrophe, safe noncompletion,
intervention, unnecessary intervention, missed beneficial intervention, Oracle
regret, and Oracle value recovered, overall and by glass/off-path/no-glass
condition.

The machine-readable acceptance audit checks whether the 40%-target router:

1. improves on the rate-matched binary-risk router in success and catastrophe;
2. supplies the intended success/safety/intervention tradeoff against fixed
   Detour;
3. retains controls better than the from-reset hazard prompt; and
4. contributes a non-dominated success/catastrophe/intervention point.

That fixed-point audit is retained for provenance, along with the post-pilot
20%-target audit.  The paper's primary acceptance is additionally evaluated at
the **frontier level**, as requested by the study design: it reports every
predeclared `(lambda, target intervention rate)` point satisfying each of the
four criteria and passes only when each criterion has at least one such point.
It also reports whether any single point satisfies all four.  The deployable
Pareto calculation excludes Counterfactual Oracle because Oracle is an upper
bound, not a competing method; a second Pareto list including Oracle is retained
in the same artifact.

Quest entry point: `setup/submit_fresh_counterfactual_router.sh`.  It submits a
freeze/authoring job, a source-exclusive GPU smoke, and the dependent full GPU
evaluation plus analysis.  The eligibility-only supplement is submitted with
`setup/submit_fresh_counterfactual_router_supplement.sh` when needed.

## Sealed outcome

The default-state full job `9677761` retained 5/12 sources. The random-reset
prepare/full jobs `9679136`/`9679137` sampled 30 new resets, found 25 nominal
successes, and retained the first 8 eligible sources from 12 attempts. Both jobs
completed with exit code 0. Router JSON/NPZ hashes are identical across runs and
the eligible source sets are disjoint.

The independent eight-source frontier passes the frontier-level audit. Four
predeclared points satisfy all four criteria; the most compact paper point is
`lambda=1,target=0.6` with 87.5% success, 8.33% catastrophe, and 58.33%
intervention. The corresponding risk router has 41.67% success, 8.33%
catastrophe, and 50.0% intervention; Always Detour has 70.83%, 4.17%, and 100%.
The combined 13-source frontier also passes at the portfolio level, but no
single combined point satisfies all four criteria. Fixed-point audits at the
original 40% and post-pilot 20% `lambda=5` points remain negative. The exact
paper summary is
[`counterfactual_router_fresh_online_20260817.json`](../results/counterfactual_router_fresh_online_20260817.json).
Promoted full analyses, frontier CSVs, and figures are indexed in
[COUNTERFACTUAL_ROUTER_MAIN_RESULT.md](COUNTERFACTUAL_ROUTER_MAIN_RESULT.md).
