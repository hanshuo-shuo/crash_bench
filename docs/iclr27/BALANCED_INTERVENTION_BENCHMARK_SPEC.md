# Balanced Intervention Benchmark specification

> **Post-v1 status (2026-08-30):** This remains the prospective specification
> for an optional broad benchmark, not the completion gate for the active
> glass-scoped paper. The one v1 non-glass action was consumed by job `5165648`
> but produced zero option outcomes because geometry preflight was invalid.
> Current work is governed by
> [`PUBLICATION_FIRST_RESOLUTION.md`](PUBLICATION_FIRST_RESOLUTION.md); the v1
> failure chain is recorded in
> [`NON_GLASS_UNSTABLE_PLACEMENT_V1_AUDIT.md`](NON_GLASS_UNSTABLE_PLACEMENT_V1_AUDIT.md).

**Document role:** prospective specification only; nothing here has been
executed  
**Current story status:** `GLASS_SCOPED_BENCHMARK_PIVOT`  
**Historical pre-v1 authorization:** only one small, disposable non-glass
option-ambiguity authoring screen; consumed by job `5165648`
**Not authorized:** the frozen benchmark collection, backbone expansion,
router training, or confirmatory claims

## 1. Objective

The benchmark is designed to create and measure **source-diverse intervention
ambiguity**, not to maximize a learned router's accuracy. It must make these
four questions separately auditable:

```text
Does Base catastrophize?
Would any intervention improve utility over Base?
Which intervention is best?
Can a statewise opportunity be realized sequentially?
```

The current 20-source corpus cannot satisfy the broad claim. Its `glass`,
`offpath`, and `noglass` conditions belong to one `glass_recovery` mechanical
design; all 23 strict Retreat states and 51/60 strict Detour/Retreat states are
glass. This specification defines the smallest expansion that could support a
claim across backbones and mechanical families. Passing the specification
would not establish deployment success.

## 2. Minimum benchmark matrix

The smallest admissible frozen design has:

- **two genuinely distinct VLA backbone families**, not two checkpoints of the
  same architecture; one OpenVLA-family and one pi0-family policy is an
  admissible minimal pairing, with exact model and preprocessing revisions
  pinned;
- **three mechanically distinct hazard families**; the existing glass/contact
  design can count as at most one;
- **three matched conditions per family and source**: on-hazard, offpath, and
  no-hazard;
- **three options from every eligible state**: Base and two semantically
  distinct interventions;
- **at least 12 independent source blocks per backbone-family cell**, for a
  minimum of `2 x 3 x 12 = 72` source blocks before exclusions.

The 12 sources in each backbone-family cell are assigned before outcomes as
six training, three calibration, and three test sources. This is the minimum
for a pooled, stratified diagnostic claim; it does not authorize strong
per-cell method-superiority claims. A power analysis may require a larger
fixed count before collection, but never a smaller one and never an adaptive
top-up after labels are observed.

Backbone, family, condition, option, horizon, and repeated branch are strata
nested within a source; they are not independent samples. A physical reset
seed evaluated under two backbones creates two policy-specific source blocks
because the nominal histories and saved decision states differ. The shared
physical seed may be retained as a pairing key.

## 3. Mechanical-family contract

Families are mechanically distinct only if they differ in the physical
failure mechanism and in the state variables that determine the safe
alternative. Texture, transparency, object color, or moving the same obstacle
does not create a new family.

The minimal proposed roster is:

1. **Swept-path obstruction/contact:** a fixed or movable obstacle intersects
   the nominal end-effector/object corridor; the current glass design belongs
   here.
2. **Object-stability disturbance:** contact, acceleration, or placement can
   topple, spill, dislodge, or otherwise invalidate a task object without the
   same path-blocking mechanism.
3. **Kinematic-clearance or reachability trap:** the nominal motion enters a
   narrow-clearance, joint-limit, pinch, or dead-end configuration in which
   backing out and rerouting have different task consequences.

Before any frozen collection, each family must have a machine-readable hazard
predicate, catastrophe predicate, task-success predicate, safe-noncompletion
predicate, validity predicate, and list of randomized physical parameters.
The predicates must be fixed without inspecting learned router scores.

## 4. Matched control contract

Every on-hazard decision has two controls derived from the same task source,
initial robot/object state, nominal instruction, and randomization block:

- **offpath:** retain the hazard object's material, visibility, and other
  nuisance properties, but move the mechanical hazard outside the
  predeclared swept/task-relevant region;
- **no-hazard:** remove or disable the mechanical hazard while preserving all
  non-hazard scene elements that can be held fixed.

Offpath and no-hazard are controls, not additional hazard families. Their
placement transforms and tolerance checks are specified before rollouts. A
control that changes task identity, camera calibration, instruction, or
unrelated object layout is invalid rather than post-hoc "matched."

## 5. Option contract

Each state branches to:

- **Base:** the frozen VLA/controller continuation;
- **progress-preserving intervention:** a bypass, replan, regrasp, or other
  controller intended to continue the task while changing the hazardous
  motion;
- **conservative intervention:** retreat, hold, release, or abort behavior
  intended to reduce exposure and allowed to end in safe noncompletion.

The latter two must differ semantically, not merely by gain, duration, or
threshold. Family-specific implementations are allowed, but their intent,
privileged inputs, controller revision, termination conditions, and safety
contract must be frozen. The paper may use Detour and Retreat as common aliases
only if captions explain the family-specific controllers. Structured or
privileged options are proposal controllers, not learned end-to-end recovery.

## 6. Exact-state restoration contract

All three branches for a decision must begin from one immutable capture that
includes, as applicable:

- simulator generalized position/velocity, object poses and velocities,
  contacts, constraints, task variables, and simulation clock;
- low-level controller targets, integrator/filter state, gripper state, action
  queue, and termination state;
- VLA observation and action history, prompt/instruction state, recurrent or
  cache state, preprocessing configuration, and nominal action;
- Python, NumPy, framework CPU/GPU, policy, simulator, and controller RNG
  states;
- source, condition, family, horizon, backbone, and branch-start hashes.

A preflight must restore Base twice and verify the frozen tolerances for the
initial observation, nominal action, simulator/controller state, and outcome
trace. Failure invalidates the decision under a predeclared mechanical rule;
it does not permit a hand-repaired branch. The capture manifest hashes every
serialized state and records software, simulator, controller, checkpoint,
hardware, seed, and commit provenance.

Option eligibility is decided from Base history and mechanical validity before
non-Base outcomes are opened. All eligible decisions are retained. Outcome-
dependent recensoring, hand-selected frames, and branch-specific state repair
are forbidden.

## 7. Labels, utility, and strict support

Use the current frozen utility, epsilon, and tie contract:

```text
task success       -> +1
safe noncompletion ->  0
catastrophe        -> -1

R = 1[Base catastrophes]
B = 1[max(U_I1, U_I2) > U_Base + epsilon]
O = frozen-tie-order argmax(U_Base, U_I1, U_I2)
```

Ties remain ties and never count as strict support. The collection grid must be
balanced by mechanics, sources, and prespecified state templates—not by
discarding outcomes until label counts look equal.

A frozen benchmark is support-valid only if, in every backbone-family cell:

- strict Base, intervention-1, and intervention-2 each occur in at least four
  independent sources overall;
- each strict action occurs in at least two calibration and two test sources;
- both `B=0` and `B=1` occur in at least four sources;
- at least three sources contain exact same-canonical-`R`, different-`B` or
  different-strict-`O` matched strata;
- at least two sources contain a strict flip between the two interventions.

These are minimum effective-source gates, not target decision counts. Each
source receives equal primary weight, and published strict-label summaries cap
each source's contribution at the same predeclared number of decisions. If any
cell misses a gate, the frozen result is reported as scope failure; there is no
outcome-driven top-up.

## 8. Similar-risk intervention ambiguity

The authoring target is a paired mechanical design in which Base risk is held
similar while benefit or the best option changes. The primary match is exact on
backbone, source, family, horizon/phase, and canonical binary `R`. Within that
stratum, at least one of the following must change:

- `B=0` versus `B=1`;
- strict Base versus a strict intervention;
- strict intervention-1 versus strict intervention-2.

This definition never uses a learned router score. If a continuous Base-risk
estimate is reported, it must come from a separately frozen, source-held-out
Base-only protocol with a prespecified caliper; it cannot determine which
states are authored, retained, or collected. Binary same-`R` witnesses remain
the primary model-free evidence.

## 9. Source-level train/calibration/test separation

Before outcomes are collected:

1. enumerate source IDs and the complete backbone/family/condition/template
   grid;
2. assign source blocks by a deterministic hash to train/calibration/test,
   stratified by backbone and family;
3. freeze option controllers, utility, epsilon, tie rules, eligibility,
   restoration tolerances, horizons, metrics, baselines, seed list, and stop
   rule;
4. publish the protocol commit and manifest fingerprint.

No physical reset seed, source state, near-duplicate initial scene, or authored
screen source may cross splits. Training fits representations and predictors;
calibration selects thresholds or intervention budgets; test is opened once
for the frozen analysis. Test outcomes cannot select a model, option mapping,
threshold, utility, family subset, witness rule, or figure operating point.

## 10. Authoring without router scores

Authoring may use only task mechanics, geometry, controller feasibility, and
the realized branches of a separate disposable screen. It may not rank or
select states using risk scores, advantage scores, learned embeddings, router
errors, uncertainty, margins, or predictions from any current or future
selector.

The authoring log must record all candidate parameterizations, including
failures. Mechanical ranges are frozen as ranges or grids, not as a list of
visually attractive successful witnesses. The final collection samples the
frozen grid with fixed seeds and retains every eligible state.

## 11. The one disposable authoring screen

The only action authorized when this prospective section was frozen was one
bounded non-glass screen:

- one candidate non-glass mechanical family;
- at most four disposable source states;
- at most three predeclared physical parameterizations and three decision
  horizons per source;
- all Base/intervention-1/intervention-2 branches plus matched offpath and
  no-hazard controls;
- no learned model fitting, score computation, threshold selection, or
  sequential-policy experiment.

Its feasibility readout is purely mechanical: valid exact restoration; all
three strict actions represented across at least two sources each; and at least
two source-supported same-`R`, different-strict-action strata, including one
intervention-1/intervention-2 flip. All attempted configurations and failures
are retained in the screen report.

Screen sources, states, outcomes, parameter IDs, and RNG seeds are permanently
excluded from a claim-bearing benchmark. Passing the screen does **not** itself
authorize collection; it only supplies evidence for a separate reviewed
collection decision. Failing it closes this candidate family without a larger
or score-guided screen.

## 12. One frozen collection, if separately authorized

Only after a separate authorization may the full source list, mechanical
grids, split hashes, branch count, compute budget, and analysis contract be
sealed. The collection is then executed once. It has:

- no adaptive source or label top-up;
- no replacement of difficult test sources except predeclared restoration or
  simulator-validity failures;
- no authoring-screen data reuse;
- no router-score selection;
- no method-driven stopping rule;
- one immutable result root with commit, configuration, source, seed, and job
  provenance.

If strict support, ambiguity, restoration, or source-diversity gates fail, the
collection remains frozen and the broad claim fails. A failure is not
permission to tune mechanics on test or collect a preferred method's errors.

## 13. Required benchmark analyses

The primary release is model-free and source-aware:

1. raw and source-macro `R x B`, with conditional-denominator rules;
2. benefit and strict-option support by backbone, mechanical family, source,
   condition, and horizon;
3. an exhaustive deterministic same-risk witness census;
4. fixed/risk-only policy regret and Oracle opportunity;
5. OracleGate/OracleChoice attribution for frozen learned baselines;
6. harmful, missed, and unnecessary intervention rates;
7. effective independent-source counts for every headline.

Learned selectors are secondary benchmark baselines: calibrated scalar risk,
Risk -> Best Fixed, direct choice, direct value/Q, outcome decomposition,
simple deployable geometry/state, and a small fixed nonlinear capacity
diagnostic. They use source-disjoint train/calibration/test splits and a common
intervention-rate/cost contract. No method is required to win for the
benchmark to be useful.

A sequential first-crossing track is required only for a sequential or
deployment claim. If run, it must be frozen independently of statewise test
outcomes and report known opportunities, misses, false interventions, Base
retention/collapse, and from-reset task outcomes. Statewise accuracy alone
never authorizes sequential wording.

## 14. Claim-ready release gate

A broad diagnostic claim is permitted only if the frozen test set shows, with
source counts visible:

- risk-benefit disagreement in all three mechanical families and both
  backbones;
- same-risk benefit or strict-action changes supported by at least three test
  sources per family and by both backbones;
- strict Base and both intervention labels satisfying the cell-level support
  gates;
- no single family contributing more than half of all strict intervention-1/
  intervention-2 states or effective supporting sources;
- exact restoration passing the predeclared audit;
- all negative and null cells retained.

This gate authorizes only the corresponding scoped benchmark/diagnostic
wording. Method-success still requires a separately frozen method gate, and
sequential-success still requires a separately frozen first-crossing test.

## 15. Current decision

At the time this specification was written, evidence authorized only the small
non-glass option-ambiguity authoring screen in Section 11. It did not authorize
the 72-source minimum collection, additional VLA backbone execution, model
training, or a confirmatory test. That historical screen action was consumed by
job `5165648`; Section 16 records the current resolution.

## 16. Post-v1 publication-first resolution

Section 15 records the prospective decision before v1 execution. It is no
longer the active next action. The v1 screen stopped before physical-block
freeze, restoration audit, or any option outcome, so it neither passes nor
scientifically fails the candidate family.

The current scoped paper proceeds without this broad benchmark. If the user
later authorizes an additive v2 after a complete manuscript snapshot, its
engineering corrections and source-aware evidence classifications are defined
in `PUBLICATION_FIRST_RESOLUTION.md`. A v2 result cannot overwrite v1 and does
not automatically authorize the 72-source collection.
