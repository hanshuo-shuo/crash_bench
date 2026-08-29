# ICLR 2027 reviewer-risk audit

**Audit date:** 2026-08-29
**Current decision:** CONDITIONAL GO; none of the high-severity method risks is
closed by the existing n=8/n=13 test outcomes.

Severity is about threat to the ICLR claim, not whether the underlying frozen
experiment was executed correctly. “Closure evidence” must come from
train/calibration/development or a newly frozen protocol; the existing fresh
test outcomes cannot be mined again to close a method-selection risk.

| ID | Sev. | Reviewer objection / current evidence | Required closure and allowed wording now |
|---|---:|---|---|
| R01 | Critical | **Binary Risk only dispatches Retreat, so it is structurally weak.** At the display point this baseline has 41.67% success, 8.33% catastrophe, and 50.0% intervention. The result may measure the choice of a poor fallback rather than the value of outcome decomposition. | Add fair Risk -> Detour and Risk -> Best Fixed variants with identical features/splits/rate calibration. For now say “versus Binary Risk -> Retreat,” never “versus risk-based routing generally.” |
| R02 | Critical | **Risk -> Detour, Risk -> best fixed option, and a risk-gate + option classifier are absent.** Always Detour already reaches 70.83% success and 4.17% catastrophe, exposing option-choice confounding. | Phase-2 B4--B6 under the same source split and candidate set. If they match the router, pivot to an option-ambiguity cohort. |
| R03 | Critical | **Direct Choice, Direct Utility/Q, and Pairwise Advantage baselines are absent.** The current data do not isolate outcome decomposition from generic multi-class/value learning. | Phase-2 B7--B9 with capacity and latency reported. Until then, “outcome decomposition is a useful inductive bias” is unsupported. |
| R04 | Critical | **n=8 confirmation is small; pooling to n=13 produces no single all-criteria point.** Source-level intervals are wide and cohort heterogeneity is material. | Report raw source-paired outcomes, exact tests, discordant counts, and both cohorts separately. Do not use “robust,” “general,” or “consistently dominant.” |
| R05 | Critical | **`lambda=1,target=0.6` was identified from the full test frontier.** It was one of four all-criteria points, not a pre-frozen single primary setting. | Treat the 45-point frontier as the confirmatory object and the displayed point as exploratory/descriptive. A future single operating point must be frozen without test outcomes. |
| R06 | Critical | **The 45 frontier points create post-selection and multiple-comparison risk.** Four passing points cannot be read as four independent confirmations. | Family-level claims only; disclose all points; add simultaneous or selection-aware analysis if making point-level inference. No “statistically significant best point” claim. |
| R07 | High | **Fresh T-20 Base outcomes are nearly determined by condition:** on-path glass catastrophes while off-path/no-glass controls succeed. A classifier may reduce to hazard/control recognition. | Add per-condition/source analyses and an option-ambiguity cohort where the beneficial option varies within hazard condition. Do not claim general intervention reasoning from the current cohort. |
| R08 | Critical | **Detour uses privileged geometry.** The router learns selection over an engineered candidate, not geometry-free recovery generation. | Label Detour `privileged structured option` in abstract, method, tables, and captions. Never call it end-to-end learned recovery or fully deployable without the geometry source. |
| R09 | High | **Option budgets may be unfair.** Base, Detour, and Retreat can differ in horizon, path length, control effort, and wall-clock duration. Current utility does not charge these asymmetries. | Audit common termination/budget rules; report path length, steps, latency, and control cost; introduce `kappa`/cost sensitivity. Current claim is outcome routing under the frozen option contracts only. |
| R10 | High | **Hazard Prompt starts from reset rather than the exact anchor.** It is not a same-state selector baseline and its 100% intervention semantics differ. | Keep it as a prompt-scope comparator, not evidence that routing beats all prompting. Add an exact-anchor instruction/prompt comparator only under a frozen protocol if technically meaningful. |
| R11 | High | **`delta` is a calibration margin/switching penalty, not a lower confidence bound.** Calling it confidence-aware or uncertainty-certified would be false. | Use “calibration-frozen advantage margin.” A statistical lower bound requires an explicitly constructed and evaluated uncertainty method. |
| R12 | High | **The model predicts safe noncompletion, but primary utility gives it zero weight:** `U_lambda=p(S)-lambda p(C)`. This weakens the claim that decomposition enables preference reweighting. | Use and sensitivity-test `V_{lambda,eta}=p(S)-eta p(N)-lambda p(C)` and switching cost `kappa` on non-test data before claiming general preference reweighting. |
| R13 | Critical | **Only source-cluster bootstrap is reported with 8 or 13 clusters.** Exact paired inference and discordant-source counts are missing. | Add source-block exact sign-flip/randomization tests, raw numerators/denominators, and source-level paired tables. Bootstrap CIs remain descriptive. |
| R14 | Critical | **Coverage is one task, one main VLA, and one main hazard family.** Supporting wall/OFT/pi0 evidence does not replicate E16 routing. | Either add a source-disjoint second task-family replication after the baseline gate or narrow the title/abstract/claims to the evaluated OpenVLA--LIBERO glass setting. |
| R15 | Critical | **Contemporary work narrows the novelty window.** SAFE covers hidden-state failure detection, CheckVLA covers sequential action-conditioned verification/repair, and CoWAM covers selective intervention with action-conditioned futures. | Lead with observed exact-state potential outcomes and fair multi-option decision baselines. Cite and compare directly; do not claim first selective VLA intervention, first sequential verification, or first hidden-state failure detector. |
| R16 | High | **Test eligibility can leak method comparison if exclusions depend on Router/baseline outcomes.** Current protocol says Base/mechanical eligibility, but this must stay machine-auditable. | Preserve every attempt and exclusion; eligibility must depend only on Base and mechanical contracts and be resolved before Router evaluation. |
| R17 | High | **Several historical supporting claims lack code commit/checkpoint/raw-log provenance.** Frozen summaries establish scoped facts but not complete rerunnability. | Keep those claims supporting/conditional, distinguish tracked summary from raw reproducibility, and never fill missing provenance by inference. |

## Red-team bottom line

The paper should not be written around the +45.83-point Router-vs-Retreat gap
until R01--R03 are resolved. The correct current headline is an observed
frontier under the evaluated baseline set. Phase 2 determines whether the main
contribution is outcome decomposition, generic multi-option value learning, an
exact-state benchmark/protocol, or a no-go for the present ICLR framing.
