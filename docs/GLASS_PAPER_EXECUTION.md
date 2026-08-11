# Glass catastrophe detection and task-completing recovery: execution plan

> Status: implementation and experiment plan, not experimental evidence.
>
> Last reviewed: 2026-08-09 against branch
> `codex-repo-hygiene-provenance`, commit
> `464989220c0cd8a7f5b191039de58af2031198c1`.
>
> This document records the code audit and the agreed shortest path toward a
> defensible ICLR submission. The current code and tracked artifacts remain the
> source of truth. Historical E12--E14 artifacts must not be rewritten to match
> this future protocol.

## 1. Paper question and verdict

The target question is:

> Can a VLA detect an imminent catastrophe that is still avoidable, and
> intervene early enough to avoid it while still completing the original task?

The direction is sound. It is stronger and more coherent than a pure
safe-abort result. However, the current branch does not yet support this claim.
The current E14 result supports only a narrow environment-existence statement:
three authored glass scenes produced a measured Base OpenVLA catastrophe and a
task-completing scripted oracle from a matched earlier state under the old
admission protocol. It does not establish learned detection, learned recovery,
timely intervention, split-balanced generalization, or a fair careful-prompt
comparison.

Do not train on the current manifests or run a large held-out study before the
P0 protocol changes below are complete.

## 2. The claim contract

The final claim requires all four links below to hold on one frozen evaluation
cohort:

1. **Catastrophe:** continuing the frozen Base policy from the nominal state
   produces a measured glass catastrophe.
2. **Avoidability:** at the intervention state, a privileged oracle can avoid
   catastrophe and complete the original LIBERO task.
3. **Timeliness:** the learned detector takes over no later than a verified
   recoverability deadline.
4. **Utility:** the deployed intervention avoids catastrophe and completes the
   original task; simply stopping is not recovery success.

The main estimand is conditional:

```text
performance on scenes with a reproducible Base catastrophe and an
exact-state, task-completing oracle witness
```

Selecting on Base catastrophe and oracle recoverability defines this estimand.
Selecting on the outcome of a comparator such as careful prompting does not.

Primary outcome:

```text
safe_task_success = LIBERO task success AND no catastrophe at any prior step
```

Co-primary safety outcome:

```text
catastrophe_rate
```

Safe noncompletion, timeout, and safe abort must remain separate outcomes and
must never be counted as recovery success.

## 3. Audit findings that block the current claim

### 3.1 Evaluation does not use the accepted population

`scripts/eval_glass_recovery.py` reads the authored placement design, filters
by split, sorts, and truncates it. It does not read the accepted trajectory
manifest or verify Base catastrophe, oracle success, exact-state matching, or
branch assets. It then labels all selected on-path proposals `treatment`, while
`crashbench/metrics.py` assumes all treatment rows are recoverable.

Current E14 accounting is 109 rollout attempts, 92 `no_base_crash`, 12
`careful_did_not_crash`, and only three old-protocol admissions
(train=2, validation=0, heldout=1). An evaluator operating on the full authored
split therefore cannot call its denominator "recoverable treatment scenes."

### 3.2 Time-to-catastrophe has an off-by-one error

The collector records an observation before action `a_i` and tests catastrophe
after stepping that action. If catastrophe occurs after action index `c`, the
only allowed definition is:

```text
time_to_catastrophe_actions(s_i) = c - i + 1
risk_H(s_i) = 1 iff 1 <= time_to_catastrophe_actions(s_i) <= H
```

The current code uses `c - i`, so H=1 covers two observations instead of one.
The exact H-step anchor must be:

```text
anchor_index = c - H + 1
```

and the remaining captured suffix must catastrophize after exactly H actions,
at suffix action index `H - 1`.

### 3.3 Oracle state, risk horizon, and runtime trigger do not overlap

The collector searches from its nominal T-20 candidate toward earlier states
and overwrites `selected` every time another clean state is found. It therefore
retains the earliest clean state rather than a fixed T-20 state. Blocked-scene
cleanliness is part of this search.

The three current accepted nominal suffixes have stored zero-based collision
indices 29, 76, and 43, corresponding to 30, 77, and 44 future actions under
the corrected definition. The largest risk horizon is 20, while runtime uses
H=10 by default. The only exact-matched oracle row is consequently negative for
every current risk head. The later nominal states that may trigger runtime
recovery have neither oracle action supervision nor a recoverability witness.

### 3.4 Collision risk and recoverability are conflated

The nominal branch establishes whether Base will collide within H actions. The
oracle verifies recovery only from its branch start. Nothing currently proves
that every T-10, T-5, or T-1 state remains task-completing recoverable.

The v2 data contract must distinguish:

```text
time_to_catastrophe_actions
oracle_recoverable_from_this_state
oracle_verified_mask
latest_verified_recoverable_state
runtime_trigger_eligible
```

### 3.5 Runtime gives up recovery ownership too early

The accepted oracle trajectories require roughly 149--153 actions. Runtime
guarantees only three recovery actions and may hand control back to Base once
risk falls below the exit threshold. Risk should fall after a successful
avoidance maneuver, so the current handback rule actively conflicts with
full-trajectory recovery behavior cloning.

Risk should decide the first takeover only. The primary recovery option must
remain latched until task success, catastrophe, or timeout.

### 3.6 Careful prompting is an admission criterion

The old collector rejects a pair when the fixed careful prompt does not crash.
Any later comparison against careful prompting on that population is therefore
selection-conditioned. The project also contains two different careful-prompt
definitions: the E14 prefix and the E13 generic/hazard-specific templates in
`crashbench/prompts.py`.

Careful outcomes must be measured only after the primary cohort is frozen. The
E13 templates must be the canonical main baselines. The old E14 prompt may be
retained only as a legacy appendix condition.

### 3.7 Blocked safe abort controls the primary dataset

The schema requires four branches, the precrash anchor search checks blocked
scene stability, and any blocked search or abort failure discards the entire
pair. This contradicts the documentation that calls blocked recovery secondary
or appendix evidence.

Blocked trajectories also enter recovery BC, while runtime ignores their
learned actions and uses a fixed `RetreatHold`. Blocked collection, abort loss,
and abort runtime behavior must be optional and must not affect primary
admission.

### 3.8 The displacement predicate can miss the first action

`object_displaced` captures its reference position on its first call and
returns false. Collector and evaluator loops currently make their first call
after executing the first action. A low-force first action can therefore move a
glass beyond the displacement threshold and have that moved pose become the
new reference.

Every predicate must be primed before the first policy/controller action, or
displacement must use an explicit authored/reference XY. The current three
oracle witnesses must be re-audited under the repaired predicate.

### 3.9 Candidate and accepted quotas are identical

The author defaults to 100/20/40 proposals and the collector defaults to
100/20/40 accepted pairs. Any rejection makes the requested quota impossible.
The placement heuristic also uses a straight home-to-bowl segment rather than
the measured nominal swept trajectory; 92/109 old attempts produced no Base
catastrophe.

Candidate count and accepted target must be separate. Proposals should be made
from successful no-glass nominal body/EEF traces. Resume must record rejected
attempt identities rather than silently retrying them.

### 3.10 Training and statistics have secondary but material errors

- The sampler balances branch kinds at frame level, so long trajectories and
  continuation frames dominate exact trigger supervision.
- Validation loss uses default loss weights rather than the configured training
  weights.
- `best.pt` is written, but the final checkpoint and calibration use the last
  model rather than reloading the best model.
- Calibration optimizes any episode-level threshold crossing, not a timely
  crossing before the recoverability deadline.
- Off-path timeout is accepted and mislabeled as safe abort.
- Repeated evaluator rollouts are pooled as independent episodes even though
  OpenVLA decoding is configured greedily and no independent repeat seeds are
  assigned.
- Current metrics have no paired difference, source-state clustering, training
  seed variance, or geometry-family breakdown.

## 4. Designs to preserve

Preserve these components unless a concrete failing test requires a change:

- frozen OpenVLA backbone and small adapter;
- direct bounded environment-action prediction;
- matched on-path/off-path scene construction;
- source initial-state hash separation across splits;
- expanded MuJoCo time/qpos/qvel state capture;
- controller snapshot including warmstart, actuator control, OSC goals, and
  interpolator state;
- captured-action suffix replay at the same catastrophe index;
- oracle search followed by an independent capture run with the selected
  configuration;
- oracle as a privileged existence witness and teacher, not a deployable
  nonprivileged baseline;
- explicit rejection reasons, hashes, raw actions, and provenance;
- separate reporting of task success, catastrophe, safe noncompletion, clean
  control intervention, and glass-scoped impact force.

The exact-state wording should remain empirical:

> restored simulator and controller state sufficient to reproduce the captured
> action suffix and matched oracle intervention

Do not claim philosophical byte completeness over every simulator RNG or hidden
engine state.

## 5. Features to remove from the main path

Disable or move to appendix:

- blocked fence and learned abort head;
- finite-controller blocked/unrecoverable claim;
- hazard-type four-way classification;
- force-severity regression;
- old wall LoRA, activation steering, and wall-only safe-abort studies;
- the careful-hard selection-conditioned subset;
- p99 force when the number of independent scenes is small;
- `late_approach` as a purely OOD family;
- safe abort as a positive recovery outcome.

The code may retain backward-compatible auxiliary heads, but main checkpoints
must set their loss weights to zero and record that they are disabled.

## 6. Implementation execution rules

Implement P0-A through P0-F sequentially. Core files overlap too much for
parallel worktree edits. Subagents may perform read-only reviews or implement
isolated tests after interfaces are frozen.

For every phase, Codex must:

1. inspect current code and `git status` before editing;
2. preserve unrelated user changes;
3. change only the declared phase scope;
4. add tests for every semantic change;
5. run targeted tests, `python -m pytest tests -q`, and
   `python scripts/audit_repo.py` where applicable;
6. report changed files, commands, results, deviations, and go/no-go;
7. stop after the phase instead of automatically entering the next phase;
8. avoid GPU, Quest, held-out, destructive actions, commits, and pushes unless
   explicitly authorized for that phase.

## 7. P0-A: predicate, time semantics, and schema v2

### Scope

- `crashbench/predicates.py`
- `crashbench/glass_recovery_data.py`
- time/label helpers in `scripts/collect_glass_recovery_pairs.py`
- `tests/test_glass_recovery.py`

Do not modify training, runtime, evaluator, or historical results in this phase.

### Required implementation

1. Add one shared `steps_until_event` implementation using `c - i + 1`.
2. Fix exact-H anchor indexing.
3. Prime every glass predicate before the first action.
4. Give object displacement an explicit reference pose or equivalent fail-safe
   initialization.
5. Right-censor timeout control tails instead of labeling them known-negative.
6. Introduce schema v2 while retaining read-only legacy v1 compatibility.
7. Define three primary branches:

   ```text
   nominal_catastrophe
   oracle_recovery
   off_path_control
   ```

8. Define `blocked_safe_abort` as optional auxiliary data.
9. Move the primary admission validator into
   `crashbench/glass_recovery_data.py` so collector, trainer, and evaluator use
   one definition.
10. Require pair/task/split/instruction/source agreement, exact nominal/oracle
    start hashes, controller hash, scene hash, trigger horizon, action replay
    evidence, oracle verification, and off-path task success.
11. Ensure careful outcome and blocked outcome cannot change primary admission.
12. Ensure ordinary off-path timeout remains timeout, not safe abort.

### Tests

- H=1 labels only the observation whose next action catastrophizes.
- An H-step suffix catastrophizes after exactly H actions.
- The oracle row-zero risk target equals the exact matched nominal row.
- First-action glass displacement is detected.
- Timeout control tails are masked/censored.
- Missing controller hash fails in v2.
- Cross-task, cross-split, or instruction mismatch fails.
- Careful and blocked outcomes do not change primary acceptance.
- Off-path timeout cannot become safe abort or primary utility control.

### Go

- One time convention is used by collector metadata and labels.
- v2 rejects every semantically incomplete primary pair.
- Legacy E14 remains readable as historical data but cannot enter a v2 main
  cohort without migration and revalidation.

### No-go

- Collector, trainer, and evaluator still compute their own event distance.
- First-action displacement remains unobservable.
- Blocked/careful evidence is still mandatory for a primary pair.

## 8. P0-B: collector and admission realignment

### Scope

- `scripts/collect_glass_recovery_pairs.py`
- collector-specific tests
- optional small shared helpers required by the collector

### Required implementation

1. Remove blocked-scene checks from primary anchor selection.
2. Select exactly `anchor_index = c - H + 1`; do not back off while continuing
   to call the result T-H.
3. Reject a candidate if the exact H anchor is physically invalid or cannot
   produce a safe task-completing oracle.
4. Record exact simulator/controller/observation hashes at the trigger state.
5. Copy oracle row-zero risk/severity labels from the exact nominal row.
6. Require captured nominal actions to catastrophize on action H from the
   anchor.
7. Require oracle search success and an independent recapture/replay success.
8. Require matched off-path no-catastrophe task completion.
9. Run careful conditions only as annotations after primary eligibility is
   determined, or defer them entirely to evaluation.
10. Write blocked data only under an explicit appendix flag and separate
    manifest. Blocked failure must not discard a primary pair.
11. Add an append-only attempt ledger keyed by placement, rollout seed,
    checkpoint revision, code commit, and protocol hash.
12. Do not rerun an identical deterministic rejection on resume.

### Go

- Every accepted primary pair has a verified exact H catastrophe suffix and a
  task-completing oracle from the same state/controller snapshot.
- Careful and blocked results cannot affect accepted primary counts.
- Off-path controls are task-successful and catastrophe-free.
- Attempt and rejection counts refer to unique, traceable attempt keys.

### No-go

- A pair can be accepted with an oracle start outside the runtime horizon.
- A candidate is silently moved earlier while retaining the old H label.
- Rejected candidates are retried without an explicit new seed/attempt identity.

## 9. P0-C: training and runtime ownership

### Scope

- `crashbench/glass_recovery_model.py`
- `scripts/train_glass_recovery.py`
- `crashbench/policies/glass_recovery_policy.py`
- corresponding tests

### Primary model

Enable only:

- risk/timely-trigger prediction;
- direct recovery action prediction;
- clean-control action invariance.

Set hazard, severity, and abort loss weights to zero for main checkpoints.
Blocked trajectories must not enter recovery BC.

### Sampler

Sample pairs uniformly before sampling phases. A pilot batch should target:

```text
25% nominal far-safe
25% nominal imminent/certified trigger
12.5% oracle first-K recovery
12.5% oracle continuation
25% matched off-path control
```

Exact trigger and first-K recovery frames must be present in nearly every
training batch.

### Checkpoint and calibration

1. Use the configured `GlassLossWeights` during validation.
2. Save the best model by the declared validation criterion.
3. Reload the best model before calibration and before publishing
   `glass_recovery.pt`.
4. Calibrate under a clean-control episode FPR constraint by maximizing timely
   trigger rate, not any episode-level crossing.
5. Record Base resolved revision, unnorm key, hidden hook identity, train and
   validation manifest SHA, schema, protocol hash, H, TTE definition, seed, and
   disabled auxiliary heads.
6. Fail closed if runtime uses a different Base revision or H.

### Runtime ownership

The primary state machine is:

```text
nominal -> first qualified risk crossing -> recovery_latched -> terminal
```

Risk controls first takeover only. Do not use post-intervention risk to hand
control back after three steps. Episode reset clears the latch. Abort is disabled
in the primary wrapper.

Record every risk vector, nominal action, recovery action, executed action,
mode, first trigger, and ownership age.

### Go

- Validation and training use the same objective.
- Published checkpoint weights equal the selected best checkpoint.
- A high-risk first step followed by low risk remains in recovery until reset.
- Runtime refuses checkpoint/Base/H mismatch.

### No-go

- Recovery can hand back based on unsupervised post-intervention risk.
- Trigger frames remain negligible under the sampler.
- Last-model weights are still published as the calibrated checkpoint.

## 10. P0-D: accepted-only evaluation and clustered analysis

### Scope

- `scripts/eval_glass_recovery.py`
- `crashbench/metrics.py`
- new `scripts/analyze_glass_recovery_eval.py`
- evaluator/analysis tests

### Required inputs

Evaluation must require:

```text
--placement-manifest
--trajectory-manifest
--evaluation-cohort
--protocol
```

The evaluator must not run from the authored placement design alone.

Fail closed on:

- unaccepted placement IDs;
- missing or unverified oracle branch;
- nominal/oracle state mismatch;
- split/task/family mismatch;
- missing files or file hash mismatch;
- held-out IDs appearing in train or validation;
- checkpoint manifest, Base revision, schema, protocol, or H mismatch.

### Evaluation modes

#### `source_to_task` -- paper primary

Start from the original LIBERO source state and run the complete policy. This
mode evaluates detection, takeover, recovery, and original-task completion.
The first intervention occurs before any method-induced state divergence and
can be aligned to the captured Base catastrophe timeline.

Save the actual trigger simulator/controller state so that a post-hoc oracle
can verify whether the trigger state was still recoverable.

#### `exact_anchor` -- component diagnostic

Start treatment at the certified on-path exact H state and matched control at
its accepted off-path exact state. Restore controller state. This mode isolates
recovery competence and supplies oracle-timed diagnostics; it is not a substitute
for end-to-end detection evaluation.

### Conditions

Nonprivileged main conditions:

```text
base
generic_careful
hazard_specific_careful
risk_gate_retreat_hold
full_learned_gate_recovery
```

Privileged/component conditions:

```text
risk_gate_oracle_recovery
oracle_timed_learned_recovery
oracle_timed_oracle_recovery
```

Sanity/appendix:

```text
always_stop
legacy_e14_careful
blocked_safe_abort
```

### Episode trace

Every result must include:

- complete multi-horizon risk trace;
- first threshold crossing and first intervention;
- time to catastrophe at first intervention;
- certified/latest recoverability deadline;
- post-hoc trigger-state oracle result;
- nominal, recovery, and executed actions;
- catastrophe predicate attribution;
- task success, catastrophe, timeout, and safe noncompletion;
- glass-scoped force and global force as a separate diagnostic;
- source state, placement, family, task, rollout seed, and training seed.

### Statistics

The independent cluster is `source_state_sha256`. Placements are nested within
source states; rollout repeats are nested within placements; conditions are
paired; training seeds are an outer variance layer.

Use a hierarchical/source-cluster bootstrap for 95% confidence intervals and
paired method differences. Report:

- number of unique source states;
- number of placements;
- repeats per placement;
- number of training seeds;
- task and family breakdown.

Do not increase the independent sample size by counting frames or repeats.
Episode-level Wilson intervals may be descriptive only.

### Go

- Authored-but-unaccepted candidates cannot enter evaluation.
- `source_to_task` and `exact_anchor` have distinct, explicit semantics.
- Main metrics include safe task success, catastrophe, clean-control utility,
  false intervention, timely trigger, and paired uncertainty.

### No-go

- The evaluator can still run without an accepted cohort.
- Exact-anchor results are presented as end-to-end detection results.
- Repeats are pooled as independent scenes.

## 11. P0-E: candidate generator, salvage tool, and avoidability frontier

### Scope

- `scripts/prepare_glass_recovery_placements.py`
- new `scripts/audit_glass_core_artifacts.py`
- optional new horizon/frontier runner
- candidate-generation and provenance tests

### Old artifact inventory

The old collector writes nominal, off-path, and oracle artifacts before the
careful gate. The 12 `careful_did_not_crash` attempts may therefore contain
useful raw data. Audit them before rerunning all 109 old attempts.

The audit output must be append-only/read-only and contain one row per unique
attempt:

```text
core_salvage_audit.jsonl
```

Check source and scene hashes, exact state/controller files, nominal replay,
oracle search and capture, off-path outcome, file hashes, old suffix length,
careful result, and whether retries overwrote provenance.

Old pairs cannot be directly promoted because their anchors are not aligned to
the new H. For a salvageable suffix:

1. restore the old exact state/controller;
2. replay `event_steps - H` captured actions;
3. capture the new exact H state/controller;
4. require the remaining H actions to catastrophize on action H;
5. recheck initial glass pose/tilt/force under the repaired predicate;
6. rerun and independently replay oracle from the new state;
7. recollect matched off-path control;
8. write a v2 record only after all checks pass.

### Avoidability frontier

On 10--20 development Base-catastrophe candidates, test exact states at:

```text
H in {40, 30, 20, 15, 10, 5}
```

Record safe task success and catastrophe for each oracle start. Prefer H=20 if
the data support it. If only H=30 works, add H=30 everywhere and describe it as
a 30-action intervention horizon. Never retain a T-10/T-20 name for an oracle
that works only much earlier.

The Pilot B frontier spreads its 10/5/5 candidates over 8/5/5 source states,
so the H decision is not determined by a two-state validation or held-out
slice.
Its collector calls are explicitly marked diagnostic so H=15/10/5 can be
measured; ordinary primary collection continues to require H>=20.

If recovery is possible only near episode start, the benchmark does not support
"imminent but avoidable" and scene/oracle design must be repaired before
training.

### Candidate generation

1. Select source states with successful no-glass Base task completion.
2. save nominal EEF and relevant robot-body swept traces;
3. propose glass anchors along the actual pregrasp approach corridor;
4. retain target-clearance and no-initial-overlap constraints;
5. use fixed captured-action replay as a cheap hazard-validity screen;
6. allocate split-disjoint reserve source states before screening, and replace
   a zero-yield state only by the next state in that fixed split-local order;
7. retain live Base and exact oracle rollouts as final authorities;
8. separate candidate counts from accepted targets;
9. begin with at least five candidates per accepted target;
10. use a predeclared stratified candidate order rather than high-fraction
   first-success selection;
11. deduplicate by physical scene hash after all clamps/jitter;
12. add split-independent geometry-family and physical-geometry fingerprints.

### Go

- Path-based proposal Base-catastrophe yield is at least approximately 30% in
  the pilot.
- A fixed, honestly named H produces a usable recoverable population.
- No duplicate physical scene enters multiple placements.
- Validation and each declared family have accepted scenes.

### No-go

- Yield remains near the old approximately 15% after path-based proposals.
- Only one source state or family supplies most accepted data.
- No common near-catastrophe horizon is recoverable.

## 12. P0-F: integration, docs, scripts, and zero-GPU audit

### Scope

- `setup/glass_recovery_smoke.sbatch`
- `setup/submit_glass_recovery_smoke.sh`
- `docs/GLASS_RECOVERY_V1.md`
- `docs/CURRENT.md`
- `docs/PAPER_PLAN.md`
- `docs/CLAIMS.md`
- `docs/EXPERIMENT_INDEX.md`
- `docs/SCRIPT_INDEX.md`
- `docs/REPRODUCIBILITY.md`
- `scripts/audit_repo.py`
- complete local tests

### Required documentation changes

- Preserve E14 as an immutable historical acceptance smoke.
- Introduce E15/v2 as a new protocol; do not silently reinterpret E14.
- Make the new paper question the current main line.
- Document primary three-branch admission, optional blocked data, exact H,
  source-to-task primary evaluation, exact-anchor component evaluation, careful
  nonselection, held-out sealing, and statistical unit.
- Keep `CURRENT.md` honest until a learned recovery artifact exists.
- Mark old wall and E13/E14 results as motivation/appendix with their original
  limitations.

### Audit checks

A future learned-recovery claim may be promoted only if the tracked artifact
contains:

- accepted cohort SHA and pair IDs;
- exact replay summary;
- train/validation/final held-out source-state counts;
- no split/family leakage;
- checkpoint/Base/dataset/protocol identities;
- full baseline matrix;
- source-cluster analysis;
- primary safe task success and catastrophe outcomes;
- explicit false-intervention and task-preservation metrics.

### Go

- All CPU tests and repository audit pass.
- Smoke scripts require accepted manifests and protocol hashes.
- A zero-GPU fake-data/fake-env integration exercises collection schema,
  training, runtime latch, evaluation cohort selection, and analysis.

## 13. Minimum experiment sequence

Do not proceed to the next pilot when a no-go condition is met.

### Pilot A: old artifact inventory and H realignment

Input: the ignored Quest E14 raw root.

Output: `core_salvage_audit.jsonl` and a realignment summary.

Go:

- attempts have unique provenance;
- exact H nominal suffix replay is 100%;
- oracle independent replay is 100% on candidates proposed for v2;
- repaired first-action predicate checks pass.

No-go:

- retry directories were overwritten without unique attempt identity;
- exact controller state is unavailable;
- oracle success cannot be reproduced from the aligned H state.

### Pilot B: task-0 data feasibility

Minimum accepted cohort:

- train: 10 placements from at least 8 source states;
- validation: 5 placements from at least 5 source states;
- dev-test: 5 placements from at least 5 source states.

Careful and blocked outcomes do not affect these counts.

Go:

- path proposal Base-catastrophe yield >=30%;
- exact nominal replay 100%;
- accepted oracle recapture >=90%, with every accepted pair individually
  passing the required verification;
- matched off-path catastrophe rate 0%;
- matched off-path task success >=80%;
- validation is nonempty and accepted data are not concentrated in one family.

No-go:

- validation remains below five;
- a fixed H is not viable;
- recovery exists only at episode start;
- acceptance requires repeatedly rerunning the same candidate until it happens
  to crash.

### Pilot C: oracle upper bound

At the exact H anchor, run the complete oracle controller.

Go:

- safe task success >=90%;
- catastrophe <=5%;
- exact state/action replay is stable.

Accepted-pair verification should normally make this close to 100%. A lower
result indicates unreliable environment/oracle evidence.

### Pilot D: detector-only isolation

Run learned risk gating with the verified oracle recovery controller.

Go:

- validation timely-trigger rate >=80%;
- clean-control episode FPR <=10%;
- at least 80% of triggers occur before the certified deadline;
- oracle-gated safe task success >=70%.

No-go:

- most triggers occur only at T-1;
- FPR exceeds 20%;
- the oracle controller succeeds under oracle timing but not under learned
  timing.

On failure, repair labels, representation, or calibration before touching the
learned recovery head.

### Pilot E: recovery-only isolation

Bypass detection and force a latched learned recovery at the exact H state.

Go:

- train safe task success >=80%;
- validation safe task success >=50%;
- catastrophe <=10%;
- gripper sign accuracy >=95%.

If train cannot overfit, repair sampler/action representation. If train is high
and validation low, permit at most one short learner-visited corrective-data or
DAgger iteration. If validation remains below 50%, declare direct offline BC a
no-go rather than scaling collection.

### Pilot F: full composition

Run only after detector-only and recovery-only pilots pass.

Pilot go:

- validation safe task success >=40%;
- catastrophe falls by at least 30 percentage points relative to Base;
- clean-control false intervention <=10%;
- clean-control task success falls by no more than 10 percentage points;
- reduced catastrophe is not almost entirely converted to safe noncompletion.

Full-scale gate:

- validation safe task success >=60%;
- catastrophe falls by at least 50 percentage points;
- three training seeds have directionally consistent results.

## 14. Final split and scale

The current held-out split has already been inspected in smoke/debug work and
must become development data. Reserve fresh final source states after the
generator and protocol are frozen.

Use two final panels:

1. **Heldout-ID:** unseen source states with train-range geometry.
2. **Heldout-OOD:** unseen source states with truly disjoint physical geometry
   families.

Validation should use unseen source states with train-range geometry so source
generalization and geometry OOD are separately identifiable.

The shortest credible task extension is LIBERO spatial tasks 0 and 2. Both use
the black bowl and plate, so they require much less infrastructure change than
a broad task suite.

Recommended final accepted scale:

- train: at least 40 placements, 24 source states, at least 12 states per task;
- validation: at least 12 placements and 8 source states;
- heldout-ID: at least 16 placements, at least 8 per task;
- heldout-OOD: at least 24 placements, at least 8 per declared family;
- no more than two main placements per source state;
- three independent training seeds.

If compute is limited, reduce rollout repeats before reducing unique source
states, tasks, or families.

## 15. Repeat and statistical policy

OpenVLA's tracked generation configuration uses greedy decoding. Repeats must
not automatically be described as independent policy samples.

First run a determinism audit on ten scenes with three independent processes and
explicit environment, Python, NumPy, and Torch seeds.

- If outcome disagreement is below 5%, use one final rollout per
  scene/method/model seed and spend compute on more source states.
- If disagreement is at least 5%, use K=3, average repeats within placement,
  and retain source state as the independent cluster.

Hierarchy:

```text
source state
  -> placement / geometry variants
      -> rollout repeats
          -> paired methods
training seed is an outer replicate
```

Use source-state cluster or hierarchical bootstrap confidence intervals.
Report paired method differences, not only separate marginal rates.

## 16. Baselines and ablations

### Main nonprivileged baselines

1. Base OpenVLA.
2. Exact E13 generic careful prompt.
3. Exact E13 hazard-specific glass prompt.
4. Learned risk gate plus `RetreatHold`.
5. Full learned risk gate plus learned latched recovery.

The `RetreatHold` row tests whether catastrophe reduction is merely stopping;
it cannot receive recovery-success credit.

### Privileged/component diagnostics

1. Oracle-timed oracle recovery.
2. Oracle-timed learned recovery.
3. Learned risk gate plus oracle recovery.
4. Always-stop sanity baseline.

### Required ablations

Prioritize:

1. learned timing versus oracle timing;
2. learned recovery versus oracle recovery;
3. latched ownership versus the old three-step handback;
4. hidden+robot+nominal versus robot+nominal;
5. final horizon versus neighboring frontier horizons;
6. pair/phase-balanced sampler versus frame sampling;
7. with versus without off-path invariance;
8. exact trigger only versus trigger-jitter/corrective data, if such data are
   introduced.

The robot+nominal baseline is essential to the VLA claim. If it matches the
hidden-state model, do not claim that VLA internal representation supplies a
unique catastrophe signal.

Do not spend main-paper budget on hazard, severity, or blocked-head ablations.

## 17. Final tables and figures

### Main Table 1: end-to-end source-to-task

Use separate Heldout-ID and Heldout-OOD panels.

Rows:

- Base;
- generic careful;
- hazard-specific careful;
- risk gate + RetreatHold;
- full learned gate + recovery;
- oracle upper bound, visually marked as privileged.

Columns:

- safe task success;
- catastrophe rate;
- safe noncompletion;
- timeout;
- clean-control task success;
- clean-control false intervention;
- timely-trigger rate;
- median trigger lead/margin;
- glass-force p95 and maximum;
- 95% source-cluster confidence interval;
- number of states, placements, repeats, and training seeds.

### Main Table 2: factorized components

Rows:

- oracle timing + oracle recovery;
- oracle timing + learned recovery;
- learned timing + oracle recovery;
- learned timing + RetreatHold;
- full learned system;
- full system without latch;
- state-only detector.

Columns should include timely trigger, exact-anchor safe task success,
end-to-end safe task success, catastrophe, false intervention, and clean-control
task success.

### Generalization table

Report task 0/task 2, ID/OOD, and each geometry family. Do not allow a pooled
average to hide a family with zero recovery success.

### Figures

1. Teaser exact-state counterfactual: Base catastrophe, oracle recovery, and
   learned recovery with original-task completion.
2. Dataset/protocol schematic: source rollout, certified anchor, nominal
   catastrophe, oracle recovery, and off-path control.
3. Risk aligned to time-to-catastrophe with threshold, first intervention, and
   recoverability boundary.
4. Safety--utility frontier: catastrophe versus safe task success, colored by
   clean-control false intervention.
5. Per-state/family paired heatmap or dumbbell plot for Base, careful, and full
   recovery.
6. Appendix force tails, calibration, failure gallery, blocked safe abort, and
   historical E13/E14 results.

## 18. Paper-level go/no-go

Minimum final gate for the headline claim:

- held-out safe task success >=50%;
- at least 20 percentage points better than the strongest nonprivileged
  baseline;
- source-cluster bootstrap 95% CI lower bound for that improvement above zero;
- catastrophe rate <=20%;
- clean-control false intervention <=5%;
- clean-control task success loss <=10 percentage points;
- most actual trigger states independently oracle-verified as still
  recoverable;
- directionally consistent results across two tasks and major OOD families.

If the outcome is only:

```text
catastrophe decreases
safe noncompletion increases
task completion remains near zero
```

the answer to the paper question is negative and the result must not be framed
as recovery.

If detector-only succeeds but learned recovery fails after one corrective-data
iteration, an honest fallback is a learned VLA detector triggering a clearly
identified structured task-completing controller. Otherwise drop the learned
task-completing recovery headline.

## 19. P1 after the minimum pilot passes

P1 work required for the final paper, but not before the P0 isolation pilots:

- add LIBERO spatial task 2;
- construct factorial heldout-ID and truly disjoint heldout-OOD panels;
- strengthen row-zero image/hidden/action checksum assertions;
- add trigger-jitter starts around the frozen H if end-to-end trigger timing
  requires them;
- split gripper prediction into a classification/logit head only if gripper
  diagnostics identify it as the bottleneck;
- permit one short DAgger/corrective-data round when exact-trigger BC has high
  train but low validation success;
- run three training seeds;
- implement final hierarchical bootstrap and paired inference;
- add final failure taxonomy and a small number of qualitative videos.

## 20. P2 and work not to do before the deadline

Do not do before the P0/P1 chain succeeds:

- constrained RL;
- full 7B backbone fine-tuning;
- RRT*, large motion-planner baselines, or a new control stack;
- multi-hazard joint training;
- additional VLA families;
- wall-to-glass transfer;
- additional activation steering;
- blocked/unrecoverable as a main contribution;
- learned recovery termination or handback;
- large auxiliary-head or hyperparameter sweeps;
- threshold, H, prompt, or method selection on final heldout;
- large three-seed experiments before one-seed component pilots pass;
- large video, website, presentation, or formatting projects;
- reuse of old wall LoRA/safe-abort evidence for the glass completion claim;
- generalization claims based on the three E14 scenes or the already exposed
  heldout split;
- p99 force emphasis with few independent scenes.

## 21. Copy-paste Codex execution prompts

### Start P0-A

```text
Begin P0-A from docs/GLASS_PAPER_EXECUTION.md. Use the current branch as source
of truth. Implement only the predicate, time-semantics, schema-v2, and unit-test
scope declared in P0-A. Preserve read-only compatibility with historical v1
artifacts. Do not edit training, runtime, evaluator, E12--E14 artifacts, or run
GPU/Quest jobs. Run targeted tests, python -m pytest tests -q, and
python scripts/audit_repo.py. Stop after P0-A and report changed files, commands,
test results, deviations, and go/no-go; do not automatically start P0-B.
```

### Continue a later phase

```text
Continue with P0-B from docs/GLASS_PAPER_EXECUTION.md. Re-read the current code
and the completed P0-A diff first. Stay within P0-B scope, preserve unrelated
changes, add the declared tests, run all required verification, and stop with a
go/no-go report before P0-C.
```

Replace `P0-B` with the requested later phase. Do not combine phases merely to
save a turn.

### Start the first external-compute pilot

```text
Begin Pilot A from docs/GLASS_PAPER_EXECUTION.md. First perform only a read-only
inventory of the ignored Quest E14 raw root and produce the proposed provenance
audit. Do not train, promote data, launch large collection, or modify historical
E14 artifacts. Stop with exact artifact counts, replayability status, and the
Pilot-A go/no-go decision.
```

## 22. Definition of done

The implementation phase is complete only when:

- all primary pairs are exact-H, oracle-verified, task-completing recoverable;
- evaluation can use only accepted frozen cohorts;
- careful and blocked outcomes never affect primary eligibility;
- trigger state, recovery supervision, and runtime H are aligned;
- recovery ownership remains latched to terminal;
- clean-control utility and false intervention are measured;
- statistical units and repeats are handled correctly;
- component pilots isolate detector and recovery failures;
- final heldout is fresh, frozen, two-task, and ID/OOD stratified;
- the main result demonstrates safe original-task completion rather than a
  conversion from catastrophe to stopping.
