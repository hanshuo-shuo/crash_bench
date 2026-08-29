# CrashBench ICLR 2027 master plan

**Truth-source status:** stage 1 frozen; stage 2 recorded on 2026-08-29
**Working title:** *CrashBench: Exact-State Potential Outcomes for Selective VLA Intervention*
**Stage-1 decision:** **CONDITIONAL GO** pending the strongest-baseline gate.
**Current submission decision:** **NO-GO for the current outcome-decomposition
headline**; only a scoped option-support / option-ambiguity pivot is authorized.

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
| 3. Problem and method rewrite | Exact-state full-information supervision and general utility are formalized without observational-causal overclaim. | Positive method rewrite blocked; only option-support/ambiguity pivot design is authorized. |
| 4+. New evidence and paper | Only evidence justified by the gate is collected and written. | No confirmatory cohort authorized. |

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

## Change control

Do not edit a frozen result to repair a claim. Add a new uniquely named result
directory, record commit/UTC/Slurm/configuration/seed/source provenance, then
change the ledger status in a separate reviewed commit. A claim cannot move
from `conditional` or `unsupported` to `supported` on narrative judgment alone.
