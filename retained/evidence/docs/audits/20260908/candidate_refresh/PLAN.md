# Historical-candidate Refresh diagnosis — 2026-09-08

The user explicitly authorized this new bounded training-source mechanism experiment
on 2026-09-08, superseding the completed retest's no-automatic-follow-up notice only
for this scope. No benchmark expansion, D8 access, new confirmatory test, architecture
search, or additional outcome-led sampling. Continue on codex/odur-repair.

## Frozen panel and execution

`panel.json` contains all nine historical D5 train stale configurations with Base
non-success and Refresh success, across eight physical sources, each paired with
its matched-buffer control configuration (18 anchors). Historical shard hashes are
included. Eight Base outcomes were 100-step noncompletion, one was an accident;
four Refresh completions were at 96, 96, 99, 99 steps. Selection is historical-outcome
conditioned, so this panel cannot estimate population success or generalization.

Reconstruct a NEW anchor from the same formal source, reset seed, anchor steps and
delay. Old complete bundles are unavailable: never describe this as historical exact
replay. Use the existing retest generation path, restore the same initial pi0
continuation before each condition, ten warmup actions, five-action chunks, unchanged
observation queue / Base / one-time Refresh semantics and 75 N accident proxy.
Matched-buffer controls have their own condition-specific generated physical anchors;
they are not identical physical states to the stale-condition anchor.
Record pre-anchor termination/accidents without replacement. Save exact state,
controller, policy/RNG continuation, mechanism queue, frame timestamps, generation
trace, fresh/delivered observations and selector context before any scored branch.

For each valid anchor: Base and Refresh each eight executions, A repeats 0–3 and
B repeats 4–7, alternating option order balanced within each phase. Every execution
restores the same new bundle and queue. At most 288 branches; no extra seeds or
replacement branches. Repeats measure residual execution variability conditional on
one bundle, not eight independent anchor reconstructions.

Each branch runs once to at most 200 post-anchor actions, stopping on success,
accident or environment termination. Extract the 100-step result from that SAME
trajectory; terminal outcomes before 100 carry forward to 200. An accident takes
precedence over simultaneous success. Environment horizon must support warmup +
anchor + 200, otherwise fail before rollouts. Do not reset or re-infer at step 100.
No assertion of perpetual failure is possible from 200-step noncompletion.

## Frozen selection and analysis

Freeze two independent A-based reference choices, one at 100 and one at 200 steps,
before ALL B execution. Selection ranks success, fewer accidents, then Base on ties.
Also retain fixed Base/AlwaysRefresh and existing age1/3/5 and period10 comparators.
No new learner is fitted. B is primary; A and pooled results are descriptive.

Report success and accident rates separately by horizon and condition, with paired,
task-stratified physical-source bootstrap intervals (10,000, seed 20260907). Nine
configurations are only eight sources. Same-bundle A/B is not held-out-source testing.
Report every anchor's counts, same-option terminal disagreement, successful completion
steps, total executed actions, actual inference calls and wall time in raw records.
Do not interpret mean stopping steps alone as faster completion: early accident can
shorten a trajectory. Record paired 100-step rescue followed by Base catch-up at 200,
persistent 200-step difference, accident-to-success and success-to-failure harms.
Pair transitions are diagnostic and repeat-pairing dependent; marginal differences
are primary. Separate neutral-control differences before interpreting stale effects.
No binary GO/NO-GO replaces the data, and no automatic method-training job follows.

## Resources and deliverables

One A100, 8 CPUs, 48 GB, at most 4 hours, account p33100 / partition gengpu.
Entry `setup/candidate_refresh.sbatch`, clean tested published commit only.
New output `/projects/p33100/siosio/crashbench_candidate_refresh/<commit>_<job>`;
692 GB available at preflight. Never write full traces to home or overwrite old runs.
Checkpoint tree manifest, source and panel hashes, software versions, Slurm ID,
commit, bundles and per-branch trace hashes accompany the dataset.
Deliver `A.json`, `B.json`, dual-horizon frozen selection, per-anchor outcomes,
complete replay bundles/traces, machine-readable metrics and a Chinese report.
Unexpected runtime failure preserves all completed branches and is reported; no
blind rerun or extra budget. Inspect final results before choosing method work.
