# Publication-first resolution for Risk Is Not Regret

**Resolution date:** 2026-08-30
**Canonical current status:** `PUBLICATION_FIRST_GLASS_SCOPED_DIAGNOSTIC`
**Active next action:** write and package the scoped diagnostic paper
**New outcome-bearing experiments required before writing:** none
**Learned-router line:** closed
**Non-glass unstable-placement v1:** invalid preflight with zero option outcomes
**Broad benchmark expansion:** optional, inactive, and not a prerequisite for the scoped paper

This document is the current execution plan. It changes the active action and
submission scope without rewriting any frozen Phase 2.5A, Phase 2.5B,
Phase 2.5B-R, PIVOT-0, sequential-closeout, or non-glass-v1 artifact. When an
older plan says that the non-glass screen is the "exactly one next action," that
instruction is historical and has been consumed by Quest job `5165648`. The
machine result and its implementation audit are preserved separately in
[`NON_GLASS_UNSTABLE_PLACEMENT_V1_AUDIT.md`](NON_GLASS_UNSTABLE_PLACEMENT_V1_AUDIT.md).

## 1. Executive decision

CrashBench will first complete a paper from the evidence that already exists.
The paper is an exact-state diagnostic and negative-result paper, not a
successful-router paper and not a broad VLA-safety benchmark.

Three research questions now have separate terminal statuses:

| Question | Status | Consequence |
|---|---|---|
| Can exact-state realized option outcomes separate Base risk, intervention benefit, best option, and sequential realization? | **Supported within the frozen glass-recovery design** | This is the active paper. |
| Has CrashBench learned a deployable or superior router? | **No** | Stop router, gate, architecture, and sequential-rescue experiments. Report the null honestly. |
| Does unstable final placement replicate option ambiguity outside glass? | **Unknown** | v1 produced zero option outcomes. It is neither positive nor negative scientific evidence. |

The broad `2 backbones x 3 families x 12 sources` benchmark remains a possible
future project. It is not the completion criterion for the current paper.

## 2. Minimum publishable claim

The paper's primary claim is:

> Within one frozen OpenVLA--LIBERO glass-recovery exact-state design, realized
> outcomes from Base, a privileged task-preserving Detour, and a conservative
> Retreat show that binary Base catastrophe status does not uniquely determine
> intervention benefit or the utility-maximizing finite option. The observed
> statewise oracle opportunity is not reliably realized by the evaluated
> learned selectors or by fresh sequential first crossing.

This is an empirical case-study claim. It is not a universal theorem about
VLAs, hazards, tasks, options, or backbones.

The following quantities must appear with their independent-source limits:

- 273 eligible exact-state decisions from 20 exposed-development sources;
- 19/273 `R != B` decisions across 12 sources;
- ten `R=1,B=0` and nine `R=0,B=1` decisions, each direction across six
  disjoint sources;
- three tight source/condition/horizon/binary-`R` matched,
  different-strict-action pairs across two sources;
- only one independent source in the tight Detour--Retreat witness;
- Risk -> Best Fixed source-macro utility `0.2544`, Oracle `0.5642`, and an
  observed finite-option opportunity of `0.3097`;
- OutcomeRouter gain of only `+0.0178` over Risk -> Best Fixed, with no method
  gate pass;
- full tiny ADR utility `0.2311`, below the risk reference;
- fresh direct sequential selection choosing Base on 24/24 episodes,
  recovering 0/8 glass episodes, and missing 2/2 known opportunities;
- all 23 strict Retreat states and 51/60 strict Detour-or-Retreat states are
  glass.

## 3. Paper contributions

The paper has exactly four contributions:

1. **Exact-state realized-option protocol.** Restore one complete
   simulator/controller/policy-continuation/RNG state and execute every member
   of a finite option set, observing outcomes rather than predicting them with
   a learned world model.
2. **Source-aware risk--regret diagnosis.** Separately measure Base risk,
   intervention benefit, strict option support, same-risk action ambiguity,
   and risk-only policy regret without treating frames, conditions, horizons,
   or branches as independent samples.
3. **Fair-baseline null and oracle attribution.** Show that the current learned
   selectors do not beat the strong risk/direct-value references by the frozen
   method gate, and localize the larger empirical loss to benefit gating in the
   reported oracle-hybrid composition.
4. **Statewise-to-sequential boundary.** Preserve the fresh first-crossing null
   as evidence that statewise option opportunity does not imply reliable
   sequential realization.

The following are not contributions and must not be implied:

- a successful, superior, or deployable learned router;
- end-to-end learned recovery;
- general risk--benefit decoupling across tasks, hazards, or backbones;
- a broad VLA-safety benchmark;
- reliable non-glass Retreat or stable-setdown support;
- reliable sequential intervention;
- novelty based only on the words "counterfactual" or "risk is not regret."

## 4. Why the current paper is complete enough to write

The active paper does not require the non-glass screen. The screen was intended
to decide whether a later broad benchmark was mechanically plausible; even a
screen pass would only have motivated another, separately reviewed collection.
It was never evidence required for the narrow glass-scoped claim.

The exact-state protocol, source-aware evidence, strong-baseline audit,
oracle-gap attribution, scope concentration, and sequential null already form
a closed narrative:

```text
failure detection asks an incomplete decision question
    -> exact-state branches expose all finite-option outcomes
    -> risk, benefit, and best option differ in the observed glass design
    -> learned routing does not reliably recover the oracle opportunity
    -> strong offline statewise structure collapses at fresh first crossing
    -> the result is a scoped diagnostic, not a deployment method
```

## 5. Active publication work plan

No stage below authorizes a new simulator rollout, model fit, threshold choice,
or source collection.

### W0. Claim lock — complete

- Use the minimum claim in Section 2.
- Preserve all frozen machine decisions and numerical limitations.
- Treat `GLASS_SCOPED` as the final current scope, not a temporary failure to
  be hidden.
- Treat non-glass v1 as an engineering preflight, not a headline result.

### W1. Full manuscript draft

Create a complete paper draft under `docs/iclr27/manuscript/` with:

1. title and abstract;
2. introduction led by one same-risk/different-action witness;
3. related work and novelty boundary;
4. exact-state realized-option protocol;
5. model-free risk/benefit/option evidence;
6. fair baselines and oracle attribution;
7. source/family support audit;
8. statewise-to-sequential boundary;
9. limitations, ethics/safety scope, and conclusion.

The default scoped title is:

> **Risk Does Not Specify Intervention: Exact-State Diagnostics for an
> OpenVLA--LIBERO Safety Case Study**

The broader historical title may appear as an internal alias, but the submitted
title must not imply multiple VLA families or a broad benchmark unless new
evidence is separately authorized and collected.

### W2. Tables and figures

Build the manuscript around existing reviewed artifacts:

- exact-state protocol plus raw/source-aware `R x B` flow;
- complete three-pair witness table;
- Risk -> Best Fixed / learned / hybrid / Oracle regret table;
- oracle-gap waterfall;
- source/family concentration figure;
- statewise-to-sequential evidence table.

Every caption must display effective source counts and distinguish the single
`glass_recovery` family from its `glass`, `offpath`, and `noglass` conditions.

### W3. Reproducibility and claim audit

- Link every headline number to a tracked artifact and commit where available.
- State explicitly which raw historical artifacts are unavailable.
- Document checkpoint, source, split, utility, option, exact-restoration, and
  privileged-controller contracts.
- Run the repository audit and a manuscript claim scan that rejects every
  prohibited headline in Section 3.
- Keep the non-glass v1 result immutable and outside the paper's scientific
  denominator.

### W4. Submission package

- Produce one complete manuscript, appendix, figure/table inventory,
  reproducibility statement, limitations section, and venue-neutral cover
  summary.
- The first submission target should accept scoped diagnostic, negative, or
  empirical case-study work. Main-conference formatting is allowed, but the
  claim must not be inflated to compensate for venue ambition.

## 6. Definition of done

The publication-first task reaches `PAPER_PACKAGE_READY_GLASS_SCOPED` when all
of the following are true:

- a full manuscript exists; no section is a placeholder;
- the abstract contains the 19/273, three-pair/two-source, oracle-gap, method
  null, glass-concentration, and sequential-boundary facts;
- all main figures and tables are generated or linked to reviewed artifacts;
- every headline uses source counts rather than pooled branch/frame counts;
- privileged Detour and exposed-development status are visible in the abstract
  or main method/limitations text as appropriate;
- no sentence claims a successful router, broad benchmark, cross-mechanism
  generality, or reliable sequential intervention;
- the repository audit, manuscript claim audit, tests for any new builders,
  and `git diff --check` pass;
- the manuscript and supporting files are committed on a clean branch.

Completion does not depend on non-glass v2, another router, another backbone,
or the 72-source broad matrix.

## 7. Optional main-conference enhancement: non-glass v2

This section is inactive unless the user explicitly chooses a higher-risk
main-conference expansion after the scoped manuscript is complete. It is not
the next action for the publication-first plan.

If authorized, v2 is an additive engineering correction, not a rerun that
overwrites or reinterprets v1. It must use a new config, result root, protocol
fingerprint, commit, and audit. Because v1 opened zero option outcomes, fixing
the preflight does not tune mechanics to intervention labels, but every change
must still be prospective and documented.

The eight v1 source hashes/seeds remain permanently excluded from
claim-bearing evidence. They may be reused only for engineering tests that
verify invariant geometry, controller mechanics, accounting, and restoration.
Any scientific v2 extension must freeze a fresh source list before outcomes and
must not select replacements using the engineering-source results.

### Required v2 engineering corrections

- derive invariant asset dimensions in model-local coordinates;
- use the bowl's bottom contact footprint/support polygon, not its whole-body
  world AABB, for COM support;
- separate invariant shape identity from source-specific pose;
- assign the reference geometry only after derivation succeeds, so one failed
  candidate cannot poison later candidates;
- validate the held-to-release event independently of one brittle grasp
  heuristic;
- fail freeze-only unless it produces four selected sources, 12 resolved
  source/parameter physical blocks, 108 linked decisions, and the declared
  ordinary-row count before any intervention branch opens;
- report eight attempted source candidates separately from selected sources.

### Revised v2 scientific readout

The mandatory gate is interpretability of the experiment, not a preferred
positive label balance:

- the eligible-decision and restoration denominators are nonzero;
- exact restoration passes for every retained decision;
- on-hazard failure, when present, is final-support instability rather than a
  swept-path robot collision;
- matched offpath and no-hazard controls are mechanically valid;
- learned scores, adaptive top-ups, outcome recensoring, and post-hoc threshold
  changes remain prohibited;
- every failed, null, tied, and mechanically invalid configuration remains.

Once that engineering/scientific-validity gate passes, classify rather than
rescue the result:

- `V2_VALID_REPLICATED_INTERVENTION_FLIP`: at least two independent sources
  each contain an exact source/condition/horizon/binary-`R` matched strict
  `stable_offset_place <-> safe_setdown` flip across frozen severity;
- `V2_VALID_PARTIAL_SUPPORT`: the same phenomenon occurs in only one source,
  or source-supported ambiguity exists without a replicated intervention flip;
- `V2_VALID_NULL`: execution is mechanically valid but contains no supported
  non-glass ambiguity;
- `V2_BLOCKED_ENGINEERING`: physical blocks, controls, eligible decisions, or
  restoration do not support scientific interpretation.

Strict offset, setdown, Base, benefit, tie, and flip source counts are all
reported. They are not separate redundant gates once the replicated matched
flip criterion is evaluated.

Strict Base support remains a reported diagnostic, not a hard non-glass
feasibility gate. Under the frozen `+1/0/-1` utility, a correct Base completion
and a correct task-completing offset placement both receive `+1`; requiring
strict Base would often reward failure of the progress-preserving controller.

A replicated v2 signal would support only a scoped two-mechanism development
diagnosis. Partial support or a valid null is still a terminal, reportable
result and does not authorize a v3 rescue. No v2 classification by itself
authorizes a deployable method, a broad benchmark, or a claim-bearing
confirmatory collection.

## 8. Hard stop rules

- No Phase 2.5B-R2.
- No new router architecture, benefit gate, calibration sweep, temporal model,
  threshold rescue, or sequential cohort.
- No relabeling `offpath` and `noglass` as independent hazard families.
- No use of non-glass v1 as evidence that unstable placement succeeds or fails.
- No 72-source benchmark, second backbone, or second new family without an
  explicit user decision made after the scoped manuscript is complete.
- No postponing the scoped manuscript because a stronger optional claim might
  someday become available.

## 9. Next-agent execution contract

The next agent should:

1. read this resolution, the candidate paper plan, PIVOT-0 story, claim ledger,
   and non-glass-v1 audit;
2. treat manuscript construction as the active task;
3. reuse existing reviewed tables, figures, and result artifacts;
4. make no remote Quest call and run no outcome-bearing experiment unless the
   user explicitly replaces this publication-first action;
5. preserve all null results and limitations;
6. stop when the Definition of Done in Section 6 is satisfied.

The next agent should not infer that "develop this further" means reopening the
router or broad benchmark. Expansion begins only after the current scoped paper
is complete and the user explicitly selects the optional path.
