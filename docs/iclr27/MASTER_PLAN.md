# CrashBench ICLR 2027 master plan

> **Forward-plan resolution (2026-08-30):** Frozen machine decisions below are
> unchanged. The active publication action is now controlled by
> [`PUBLICATION_FIRST_RESOLUTION.md`](PUBLICATION_FIRST_RESOLUTION.md): write the
> glass-scoped exact-state diagnostic from existing evidence. The former
> non-glass next action was consumed by job `5165648` and produced zero option
> outcomes; see
> [`NON_GLASS_UNSTABLE_PLACEMENT_V1_AUDIT.md`](NON_GLASS_UNSTABLE_PLACEMENT_V1_AUDIT.md).
> No router rescue, new outcome collection, or broad benchmark is currently
> authorized.

**Truth-source status:** frozen method gates, PIVOT-0, non-glass-v1 audit, and publication-first resolution recorded through 2026-08-30
**Current scoped title:** *Risk Does Not Specify Intervention: Exact-State Diagnostics for an OpenVLA--LIBERO Safety Case Study*
**Historical working title:** *CrashBench: Exact-State Potential Outcomes for Selective VLA Intervention*
**Stage-1 decision:** **CONDITIONAL GO** pending the strongest-baseline gate.
**Current submission decision:** **NO-GO for a deployable advantage-router
claim**; Phase 2.5B remains frozen as **INCONCLUSIVE**, and the exhaustive
Phase 2.5B-R resolver records **CHOICE_CAPACITY_BOTTLENECK**. It authorizes no
method-success or outcome-bearing method rescue. Later PIVOT-0 separately
authorized one non-glass authoring screen; that historical action is now
consumed and does not alter the frozen resolver decision.

**Current publication decision:** **GO-WRITE for a glass-scoped diagnostic and
negative-result paper; NO-GO for a deployable method claim.** Manuscript
completion, not another experimental gate, is the active terminal objective.

This directory implements stages 1 and 2 of the
[revised ICLR 2027 execution plan](../../CrashBench_ICLR2027_Revised_Plan_After_Baseline_NoGo.md)
and is the paper-decision truth source for the ICLR 2027 effort.
Existing `docs/CURRENT.md`, `docs/PAPER_PLAN.md`, and `docs/CLAIMS.md` remain the
record of the frozen pre-audit paper story. When the wording or submission
decision differs, this file, `BASELINE_AUDIT_RESULT.md`, `CLAIM_LEDGER.md`, and
`REVIEWER_RISK_AUDIT.md` control the ICLR 2027 claim.

## One-page viability judgment

### Verdict

The frozen strongest-baseline audit is complete at commit `eed1fee`, Quest Job
`5128781`. Its formal branch is **INCONCLUSIVE**: GO-A, GO-B, PIVOT-C, and
STOP-D each fail at least one frozen tolerance. Operationally this is a no-go
for the current method headline because no positive gate cleared. Outcome
Router does not beat Risk -> Best Fixed in source-macro utility or success and
uses 12.92 points more intervention; its lower catastrophe rate is a tradeoff,
not dominance. Full results and the next authorized action are in
[`BASELINE_AUDIT_RESULT.md`](BASELINE_AUDIT_RESULT.md).

The subsequent 20-source Phase 2.5A audit passes `PASS-SUPPORT` at commit
`7876232`: strict Base/Detour/Retreat support comes from 16/13/9 sources, 13
sources meet the predeclared within-source flip rule, and an idealized
Base-versus-fixed-Detour gate recovers 90/113 = 79.65% of decision-level Oracle
value. The equal-source sensitivity is higher at 82.90%, and all strict Retreat
states remain glass-only, so this is a narrow authorization for source-LOSO
learning rather than evidence that option ambiguity has already been solved.
See [`OPTION_SUPPORT_AUDIT.md`](OPTION_SUPPORT_AUDIT.md).

The authorized Phase 2.5B source-cross-fitted rescue is now complete at commit
`df3168c`, Quest Job `5137872`. Its formal result is **INCONCLUSIVE**: no
outcome or direct-value method clears the strict Base/Detour/Retreat recall and
utility gates, neither condition-only nor horizon-only clears the diagnostic
gate, and neither STOP criterion fires. The current Outcome Router has the
best learned OOF utility (`0.2722`) but gains only `+0.0178` over Risk -> Best
Fixed and recalls strict Base/Detour/Retreat at only `0.5926/0.3269/0.3704`.
The result is fail-closed: no candidate is frozen and Screen A remains blocked.
See [`SUPPORT_CROSSFIT_RESULT.md`](SUPPORT_CROSSFIT_RESULT.md).

The existing-data-only Phase 2.5B-R post-hoc resolver is complete at commit
`bf032cb`, Quest Job `5148751`. The fixed 32-unit nonlinear choice head passes
the strict conditional D/R point-estimate gate (`0.7253` balanced accuracy,
`0.8559/0.5947` Detour/Retreat recall), while both linear choice heads fail.
Every complete deployable ADR loses utility to Risk -> Best Fixed and fails the
unchanged method gate. The exhaustive decision is
**CHOICE_CAPACITY_BOTTLENECK**, with benefit-gate failure retained as a
secondary flag. See
[`ADVANTAGE_ROUTER_RESOLUTION.md`](ADVANTAGE_ROUTER_RESOLUTION.md).

The stage-1 rationale below remains the historical reason the project entered
the gate. It no longer authorizes a confirmatory run by itself.

The project is worth continuing, but only behind a hard baseline gate. The
scientifically defensible asset is not generic VLA failure detection. It is the
exact-state branching protocol: from one serialized simulator/controller/RNG
state, execute Base, privileged Detour, and Retreat and observe all option
outcomes. This gives full-information interventional supervision for a
selective decision problem and cleanly separates three questions:

1. Is the nominal behavior risky?
2. Would intervention improve the outcome?
3. Which available intervention has the best task--safety tradeoff?

The frozen E16 result is promising. On eight independent sources and 24
matched decisions, one of four all-criteria frontier points reaches 87.5% task
success, 8.33% catastrophe, and 58.33% intervention, versus 41.67%, 8.33%, and
50.0% for Binary Risk -> Retreat. The positive interpretation is nevertheless
conditional: that displayed point was identified in a 45-point predeclared
test frontier, the combined 13-source cohort has no joint all-criteria point,
and the current binary-risk baseline is structurally restricted to Retreat.

The paper therefore does **not** yet establish that outcome decomposition is
the reason for the gain. Risk -> Detour, Risk -> best fixed option, two-stage
risk-plus-choice, Direct Choice, Direct-Q, Pairwise Advantage, and simple
observable-geometry baselines are missing. CoWAM already studies conservative
selective intervention with action-conditioned futures; CheckVLA studies
sequential action-conditioned verification and repair; SAFE and newer runtime
monitors study hidden-state failure detection. CrashBench can survive this
literature only by being precise about what is different: exact-state observed
potential outcomes, option-level consequence supervision, source-level paired
evaluation, and an explicit demonstration that risk alone is insufficient for
choosing among interventions.

The sequential evidence is a useful honesty boundary, not a second positive
method. P2 gives a small development-only improvement; P2.5 and P3.2 show that
the matched-state score does not reliably transfer to from-reset first
crossing. That negative closeout should remain in the paper because it prevents
the matched-state result from being oversold.

### Continue/stop rule

- **Continue as a method paper** only if the frozen outcome router beats
  Risk -> Best Fixed on source-held-out development and is not comprehensively
  covered by Direct Choice, Direct-Q, and Pairwise Advantage at matched
  intervention rate/cost.
- **Reframe as multi-option intervention value learning** if direct value
  methods tie outcome decomposition but clearly beat scalar risk.
- **Build a new option-ambiguity cohort** if Risk -> Detour or Risk -> Best
  Fixed explains the headline gap. The current +45.83-point comparison then
  diagnoses a weak Retreat mapping rather than a general routing advantage.
- **Stop the current ICLR method claim** if robot state or simple deployable
  geometry covers the learned methods. A benchmark/protocol paper would then
  require broader task, VLA, and hazard coverage.

Until this gate is complete, do not spend GPU budget on a new task family,
deeper temporal models, a larger test cohort, or figure-polishing around the
current headline point. Phase 2 should first reuse only development/calibration
artifacts under a frozen baseline protocol. The fresh test outcomes must not be
used for model, threshold, utility, or baseline selection.

## Frozen claim core

The current safe wording is:

> At fresh matched decision states, option-conditioned counterfactual outcome
> prediction exposes a useful safety--success--intervention frontier relative
> to the evaluated baselines.

It is not yet safe to say that outcome decomposition is superior to direct
decision/value learning, that hidden representations are necessary, that one
deployment point was confirmatory, or that the method performs reliable
sequential intervention.

## Evidence hierarchy

| Level | Meaning | Current examples |
|---|---|---|
| `protocol-frozen` | Cohort, source IDs, tasks, method family, thresholds, and primary hypotheses are hashed before outcomes. | Required for the phase-2 baseline audit; not yet created. |
| `development-only` | May select models and protocols; cannot confirm the final claim. | Five train, seven calibration, eight diagnostic sources; P2/P3.0/P3.1. |
| `confirmatory-test` | One locked execution; no selection from its outcomes. | The n=8 cohort is fresh, but the reported display point is frontier-selected, so only the full frontier is confirmatory. |
| `exploratory-test` | Test-cohort descriptive analysis, explicitly non-confirmatory. | The displayed `lambda=1,target=0.6` point and pooled n=13 slices. |
| `negative-closeout` | Frozen failure retained without another test-cohort tuning loop. | P2.5/P3.2 sequential closeout. |

## Non-negotiable statistical contract

- The independent unit is source state. Conditions, horizons, anchors, frames,
  and option branches are repeated observations.
- Test eligibility is determined by Base and mechanical validity only.
- Raw numerators/denominators and per-source paired outcomes accompany every
  paper number.
- Source-cluster bootstrap intervals are descriptive at n=8/n=13 and must be
  paired with exact source-block inference in the next audit.
- The 45-point frontier is one family. It is not 45 independent discoveries;
  any selected point is labeled exploratory unless separately frozen.
- Oracle is an unattainable upper bound. Privileged Detour is a structured
  geometry controller, not learned end-to-end recovery.

## Stage map

| Stage | Gate | Status |
|---|---|---|
| 1. Paper truth source | Every current paper number maps to immutable evidence; unsupported wording and reviewer risks are explicit. | **PASS** — local 186/186 tests and Quest Job 5124071 (`short`, exit `0:0`) passed at commit `cbc965b`. |
| 2. Strongest-baseline audit | Outcome decomposition/multi-option value survives fair direct, risk, and geometry baselines. | **INCONCLUSIVE / current method NO-GO** — Quest Job 5128781 (`short`, exit `0:0`), commit `eed1fee`. |
| 2.5A. Existing-corpus option support | Pooled strict support is source-diverse and a fixed non-Base option leaves enough canonical Oracle value. | **PASS-SUPPORT** — local CPU-only audit, commit `7876232`; only Phase 2.5B is authorized. |
| 2.5B. Source-cross-fitted support rescue | Strict 20-fold source-OOF test of current, support-balanced, ranking, risk, and value selectors under one constrained calibration rule. | **INCONCLUSIVE / FAIL-CLOSED** — Quest Job 5137872, commit `df3168c`; Screen A and every new outcome-bearing rollout remain blocked. |
| 2.5B-R. Advantage-decomposed resolver | Exhaustively distinguish method readiness, choice capacity, gate/calibration, margin, and insufficient-signal explanations on the same frozen corpus. | **CHOICE_CAPACITY_BOTTLENECK** — Quest Job 5148751, commit `bf032cb`; nonlinear conditional choice passes, but no deployable ADR passes. |
| PIVOT-0. Risk-is-not-regret synthesis | Determine the strongest paper story supported by the complete frozen corpus. | **GLASS_SCOPED_BENCHMARK_PIVOT** — diagnostic claim supported; method and broad claims remain closed. |
| Non-glass v1. Unstable-placement authoring preflight | Test whether the optional broadening mechanism can freeze a valid exact-state grid. | **SCREEN_BLOCKED_ENVIRONMENT / INVALID PREFLIGHT** — Quest Job 5165648, commit `53f62a6`; zero option rows, no scientific family result. |
| Publication. Scoped diagnostic writeup | Convert existing exact-state, baseline, support, attribution, scope, and sequential-boundary evidence into a complete paper. | **ACTIVE / GO-WRITE** — no new experiment required; Definition of Done is in `PUBLICATION_FIRST_RESOLUTION.md`. |
| Optional future expansion | Broaden mechanism/backbone coverage only after a complete scoped manuscript and explicit user authorization. | **INACTIVE** — not a current paper gate. |

## Stage-1 artifacts

- `CLAIM_LEDGER.md`: paper wording, evidence, hashes, statistical unit, limits,
  and prohibited stronger wording for every current claim.
- `RELATED_WORK_MATRIX.md`: nearest-work comparison and positioning.
- `REVIEWER_RISK_AUDIT.md`: mandatory red-team list with closure criteria.
- `results/iclr27/manifest.json`: content hashes for every frozen artifact used
  by the ledger plus machine-readable key-value checks.
- `scripts/audit_repo.py`: fails closed if the truth-source schema, hashes,
  required risks, or key frozen values drift.
- `results/iclr27/stage1_verify_cbc965bc1355_20260829T091913Z_job5124071/verification.json`:
  pulled Slurm provenance for the successful Quest verification.

## Stage-2 artifacts

- `BASELINE_AUDIT_RESULT.md`: paper decision, scoped interpretation, and next
  authorized experiment.
- `results/iclr27/baseline_audit_eed1feecb3bd_20260829T101548Z_job5128781/`:
  sealed predictions, choices, metrics, source inference, calibration, figures,
  and Slurm provenance.
- `configs/iclr27/baseline_suite.yaml`: frozen development-only comparison and
  GO/NO-GO contract.

## Phase-2.5A artifacts

- `OPTION_SUPPORT_AUDIT.md`: label contract, result, limitations, and the exact
  downstream authorization.
- `configs/iclr27/option_support_audit.yaml`: frozen support, diagnostic, and
  gate definitions.
- `scripts/iclr27/audit_option_support.py`: CPU-only, tie-preserving audit.
- `results/iclr27/option_support_audit_787623226de0_20260829T112149Z/`:
  clean-commit manifest, tables, matched pairs, and machine gate.

## Phase-2.5B protocol artifacts

- `configs/iclr27/support_crossfit.yaml`: frozen 20-fold source-OOF,
  support-weighting, constrained-calibration, metrics, and fail-closed gate.
- `scripts/iclr27/train_support_crossfit_suite.py`: outer/inner cross-fit and
  OOF prediction producer; fresh/test outcomes are rejected before loading.
- `scripts/iclr27/analyze_support_crossfit_suite.py` and
  `plot_support_crossfit_suite.py`: source-paired analysis, shared bootstrap,
  exact sign-flip tests, gate, and compact figures.
- `setup/iclr27_support_crossfit.sbatch`: CPU-only Quest entry point.
- `SUPPORT_CROSSFIT_RESULT.md`: protocol, reviewed result, gate interpretation,
  and the exact downstream prohibition.
- `results/iclr27/support_crossfit_df3168c75843_20260829T115455Z_job5137872/`:
  sealed OOF predictions, metrics, inference, figures, and Slurm provenance;
  the large fold-model archive remains on Quest under its manifest hash.

## Phase-2.5B-R resolver artifacts

- `configs/iclr27/advantage_router_resolver.yaml`: frozen post-hoc amendment,
  fixed three-model suite, metrics, and exhaustive five-way decision rule.
- `scripts/iclr27/run_advantage_router_resolver.py`: leakage-safe OOF training,
  oracle-hybrid decomposition, source bootstrap, and machine decision.
- `ADVANTAGE_ROUTER_RESOLUTION.md`: reviewed result, limitations, primary
  decision, and the single authorized next action.
- `results/iclr27/advantage_router_resolver_d4751330395e_20260829T135723Z_job5148751/`:
  frozen OOF predictions, conditional-choice/gate/composed metrics,
  oracle-hybrid attribution, source/family/margin diagnostics, bootstrap
  intervals, calibration records, configuration, and Slurm provenance.

## Change control

Do not edit a frozen result to repair a claim. Add a new uniquely named result
directory, record commit/UTC/Slurm/configuration/seed/source provenance, then
change the ledger status in a separate reviewed commit. A claim cannot move
from `conditional` or `unsupported` to `supported` on narrative judgment alone.
