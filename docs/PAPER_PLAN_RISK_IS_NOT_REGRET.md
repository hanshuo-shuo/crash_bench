# Candidate paper plan: Risk Is Not Regret

> This is a new candidate plan. It does not modify or supersede the historical
> record in [`PAPER_PLAN.md`](PAPER_PLAN.md). Frozen gate interpretations remain
> in [`MASTER_PLAN.md`](iclr27/MASTER_PLAN.md),
> [`SUPPORT_CROSSFIT_RESULT.md`](iclr27/SUPPORT_CROSSFIT_RESULT.md), and
> [`ADVANTAGE_ROUTER_RESOLUTION.md`](iclr27/ADVANTAGE_ROUTER_RESOLUTION.md).

**Paper type:** benchmark/diagnostic paper, not a successful-router paper  
**Current story status:** `GLASS_SCOPED_BENCHMARK_PIVOT`  
**Evidence status:** exposed-development synthesis; no confirmatory claim  
**Proposed title:** *Risk Is Not Regret: Exact-State Intervention Diagnostics
for Vision-Language-Action Policies*

## 1. Paper identity and scope

The paper's scientific object is the intervention decision, not a new router
architecture. It studies four targets that are often collapsed:

```text
Base catastrophe risk
    != intervention benefit
    != best intervention
    != sequentially realizable intervention
```

CrashBench's defensible asset is exact-state branching. A finite set of Base,
Detour, and Retreat controllers is restored from the same serialized
simulator/controller/RNG state, making all option outcomes observable for that
decision. This supports a benchmark and diagnostic protocol for measuring what
a scalar risk label fails to identify.

The current corpus is a scoped pilot, not yet the broad benchmark. It contains
273 eligible decisions from 20 sources, but `glass`, `offpath`, and `noglass`
are treatment and controls in one `glass_recovery` mechanical design. All 23
strict Retreat states and 51/60 strict Detour/Retreat states are glass. The
paper must display that concentration wherever it presents source diversity.

## 2. Claim hierarchy

### Primary current claim

> Within the frozen glass-recovery exact-state design, Base catastrophe status
> does not uniquely determine intervention benefit or the strict best action;
> exact-state branching makes these distinctions measurable, while current
> option support is too glass-concentrated for a broad benchmark claim.

The evidence for this exact wording is:

- 19/273 risk-benefit disagreements across 12 independent sources;
- 10 risk-positive/no-benefit and nine risk-negative/positive-benefit states,
  each direction occurring in six disjoint sources;
- exactly three tight source/condition/horizon/binary-risk matched,
  different-strict-action pairs across two sources;
- 19/20 sources with both benefit labels, but zero non-glass strict Retreat;
- all 23 strict Retreat states and 51/60 strict D/R states in glass, with the
  latter supported by 16 sources versus 17 sources overall.

### Attribution claim

> In the frozen OOF tiny-candidate decomposition, OracleGate plus learned
> choice loses `0.0725` source-macro utility from Oracle, whereas LearnedGate
> plus OracleChoice loses `0.2928`; benefit gating is the dominant empirical
> loss in that composition.

This is an oracle diagnostic, not a deployment result. The tiny nonlinear head
appears only as a capacity diagnostic: it passes the frozen conditional strict
D/R point-estimate gate, while the primary split-linear OracleGate hybrid has
slightly higher utility (`0.4975` versus `0.4917`). No architecture-success
claim follows.

### Boundary claim

> Strong offline statewise separability need not yield reliable fresh
> sequential first crossing.

P3.1 reaches recovery-open-v-hard-control AUC `1.000` and intervention-needed-
v-hard-control AUC `0.995`; the frozen P3.2 direct policy then chooses Base on
24/24 fresh episodes, recovers 0/8 glass episodes, and misses both known
recovery opportunities.

### Status claims that must remain unchanged

- Phase 2.5A is marginal `PASS-SUPPORT`.
- Phase 2.5B is `INCONCLUSIVE` and fail-closed.
- Phase 2.5B-R is `CHOICE_CAPACITY_BOTTLENECK` for machine provenance.
- No deployable ADR passes `method_pass`.
- No method-success, deployment-success, or confirmatory claim is authorized.

## 3. Five-minute narrative

1. **Detection answers the wrong decision question.** A Base catastrophe label
   does not say whether an intervention improves utility or which option to
   choose.
2. **Exact-state branching exposes the missing labels.** From one saved state,
   realize Base, Detour, and Retreat under one utility and restoration
   contract; define risk, benefit, option, and regret without fitting a model.
3. **The distinction exists but is rare and structured.** Risk and benefit
   disagree on 19/273 decisions across 12 sources, and same-binary-risk action
   flips exist, but the tight witness set contains only three pairs across two
   sources.
4. **Gate and choice are empirically separable.** Oracle hybrids preserve much
   more value when only choice is learned than when only the benefit gate is
   learned; a complete router still fails.
5. **Mechanical scope determines the story.** Retreat and D/R support are
   overwhelmingly glass. Offpath and noglass are controls, not independent
   hazards. The correct decision is `GLASS_SCOPED_BENCHMARK_PIVOT`.
6. **Statewise opportunity is not sequential realization.** The frozen
   P2/P2.5/P3.1/P3.2 chain retains the first-crossing null as a deployment
   boundary.

## 4. Main-text organization

### 4.1 Problem: from failure alarms to intervention regret

Open with a pair of same-risk decisions requiring different actions, not a
learned architecture diagram. Define the finite-option regret of an action as
`U_Oracle(x)-U_action(x)`. Show why `R`, `B`, and `O` answer different
questions even when utilities are discrete.

### 4.2 Exact-state realized-outcome protocol

Specify the source, eligibility, restore, branch, utility, strict-label, and
tie contracts. Emphasize that this is observed full information for three
executed branches at saved states, not general causal identification. Make the
source state the independent unit and group every placement, condition,
horizon, and branch beneath it.

### 4.3 Model-free evidence: risk is not regret

Report raw and source-macro `R x B`, conditional probabilities, disagreement
directions, source counts, family/horizon breakdowns, and the deterministic
witness rule. The source-macro conditional denominator must be explicit; do
not use a ratio of macro joint cells as a substitute.

### 4.4 Policy-regret and oracle-hybrid diagnostics

Compare Always Base/Detour/Retreat, Risk -> Detour/Retreat/Best Fixed, Oracle
Benefit Gate -> Best Fixed, oracle hybrids, full learned ADR, and Oracle on the
same 20 OOF source blocks. Report success, catastrophe, safe noncompletion,
intervention, utility, regret, and fraction of Oracle opportunity recovered.

The central waterfall is anchored at Risk -> Best Fixed `0.2544` and Oracle
`0.5642`, an opportunity of `0.3097`. For the tiny candidate, the recovered
fractions are `0.7659` with OracleGate plus learned choice, `0.0547` with
LearnedGate plus OracleChoice, and `-0.0753` for the full composition.

### 4.5 Source-support and mechanical-scope audit

Show source counts beside decision counts. In particular, distinguish the
five pooled strict D/R-flip sources from the two sources with a within-glass
strict D/R flip, and from the single source supporting the tight exact D/R
witness. State that non-glass has nine strict Detour states across six sources
and no strict Retreat state.

### 4.6 Statewise-to-sequential realization gap

Use one canonical table from matched-state OOF through P2, P2.5, P3.1, and
P3.2. This is a compact boundary section. It must not introduce a temporal
method, new threshold, trajectory aggregation, or recovery-rescue proposal.

### 4.7 Balanced benchmark specification and story decision

End the results with the explicit `GLASS_SCOPED_BENCHMARK_PIVOT` decision. The
design in
[`BALANCED_INTERVENTION_BENCHMARK_SPEC.md`](iclr27/BALANCED_INTERVENTION_BENCHMARK_SPEC.md)
describes what would be required for a broad paper, but its frozen collection
is not currently authorized.

## 5. Method placement

There is no proposed learned method section. Learned components are organized
as diagnostic baselines:

| Component | Paper role | Forbidden interpretation |
|---|---|---|
| Risk -> Best Fixed | strong scalar-risk reference | a universal risk baseline or globally fixed Detour policy |
| OutcomeRouter / direct value methods | frozen source-OOF baselines | successful routing or outcome-decomposition superiority |
| ADR linear heads | choice-capacity and composed baselines | evidence that linearity is the only blocker |
| Tiny nonlinear choice head | conditional capacity diagnostic only | paper method, architecture selection, or license to scale an MLP |
| OracleGate / OracleChoice | error attribution upper bounds | deployable policies |
| Oracle | realized finite-option upper bound | a learned controller or reachable deployment point |

The primary result is the benchmark diagnosis produced by option outcomes,
not a learned selector's score. Model details belong after the protocol and
model-free evidence; the tiny head's architecture belongs in the appendix.

## 6. Main figures and tables

1. **Exact-state risk-benefit flow.** One branch schematic plus the raw `R x B`
   table and source counts for both disagreement directions.
2. **Oracle-gap waterfall.** Risk -> Best Fixed; Tiny LearnedGate +
   OracleChoice; OracleGate + primary split-linear choice; OracleGate + tiny
   nonlinear choice; full tiny ADR; OracleGate + OracleChoice.
3. **Source/family support.** Condition-level independent-source support for
   disagreement and each strict label, with glass/offpath/noglass separated and
   the one-mechanical-design caveat in the caption. Keep the full per-source
   flags in the released table.
4. **Risk-only policy-regret table.** All fixed, risk-gated, hybrid, full, and
   Oracle policies under the same source-macro contract.
5. **Deterministic witness table.** All three exact tight pairs with state
   hashes, deployable feature summaries, outcomes, utilities, and strict
   actions.
6. **Sequential realization table.** Matched-state opportunity/learning,
   offline recovery separability, fresh first crossing, known misses, false
   interventions, and Base collapse.

Do not lead with the old E16 frontier or an ADR architecture figure. Historical
matched-state results can appear as context or appendix diagnostics, but no
selected router operating point is the new headline.

## 7. Statistical and provenance contract

- Treat source as the independent unit; give raw decision counts only as
  repeated-observation descriptives.
- Report raw and equal-source summaries together. For a source-macro
  conditional probability, average only denominator-bearing source
  conditionals and state the number of such sources.
- Use source-paired resampling or exact source-block inference; never infer
  from pooled frames, branches, conditions, or horizons.
- Preserve all 273 eligible decisions and the frozen utility, epsilon, source,
  eligibility, and tie contracts. Do not recensor after seeing labels.
- Distinguish historical split labels from the 20-source exposed-development
  use in PIVOT-0.
- Identify every oracle combination as diagnostic-only.
- Cite the new immutable PIVOT-0 package under `results/iclr27/` without
  rewriting frozen Phase 2.5 artifacts.
- Report effective source counts for every headline, including the two-source
  exact witness set and one-source exact D/R witness.

## 8. Nearest-work positioning

The related-work paragraph should say:

> SAFE detects VLA failure from hidden state; CheckVLA verifies and repairs
> action chunks using predicted futures; CoWAM selects among candidate actions
> with coordination contracts; SafeVLA aligns policies during training; and
> Counterfactual VLA uses generated reasoning traces to revise meta-actions.
> CrashBench instead diagnoses finite-option intervention ambiguity from all
> realized branches of identical serialized states. Its current evidence is a
> glass-scoped exact-state pilot, not the first detector, verifier, selective
> intervention layer, safety benchmark, or use of counterfactual reasoning.

CoWAM is the nearest selective-intervention threat. Any later benchmark must
include strong fixed, risk, direct-choice, direct-value, and intervention-
error comparisons rather than rely on Risk -> Retreat. CheckVLA and SAFE have
stronger sequential breadth; the current first-crossing null must remain
visible. SafeVLA blocks generic benchmark novelty, and Counterfactual VLA
blocks novelty based on terminology alone.

## 9. Abstract and contribution policy

Use the title, abstract, and four-item contribution list in
[`STORY_PIVOT_RISK_IS_NOT_REGRET.md`](iclr27/STORY_PIVOT_RISK_IS_NOT_REGRET.md).
The abstract must retain all four limiting facts: 19/273 disagreements, three
tight pairs across two sources, the `0.0725` versus `0.2928` decomposition, and
the glass concentration. It must also say that no deployable ADR passes.

Avoid contribution language around a tiny MLP, a deployment policy, universal
risk/value decoupling, or broad hazard coverage. The benchmark specification
is a design contribution only until a separately authorized frozen collection
exists.

## 10. Unsupported headlines and stopping rules

Do not claim:

- "we solve knowing when/how to intervene";
- a successful advantage router or nonlinear option selector;
- broad risk-benefit disagreement across three hazard families;
- reliable Retreat choice outside glass;
- reliable sequential intervention, end-to-end recovery, or a state-of-the-art
  runtime monitor;
- hidden-state necessity or learned-method superiority;
- a confirmatory benchmark result from the exposed 20-source corpus.

Do not create Phase 2.5B-R2, tune another gate, train another router, or reopen
P3 sequential rescue. If later balanced authoring fails to produce independent
non-glass option ambiguity, retain the scoped negative diagnosis rather than
relabeling offpath/noglass controls as hazard diversity.

## 11. Path to a claim-ready benchmark paper

The target paper becomes broadly claim-ready only if a later, separately
authorized frozen benchmark satisfies all of the following without adaptive
top-up: at least two VLA backbones; at least three mechanically distinct hazard
families; matched offpath and no-hazard controls; Base plus two semantically
distinct interventions; source-balanced strict Base/Detour/Retreat support;
same-risk benefit or option flips; exact restoration; and source-disjoint
train/calibration/test evaluation.

That future dataset's objective is intervention ambiguity and selector
diagnosis, not maximizing a learned router's accuracy. A failure of the frozen
support gate is itself reportable benchmark evidence; it is not permission to
collect until a preferred method wins.

## 12. Exactly one next action

Run one small, disposable non-glass option-ambiguity authoring screen, as
bounded in the companion benchmark specification. PIVOT-0 does not authorize
the subsequent frozen collection, multi-backbone expansion, or any learned
model training.
