# PIVOT-0 story decision: risk is not regret

**Story status:** `GLASS_SCOPED_BENCHMARK_PIVOT`  
**Evidence role:** model-free synthesis of the frozen, exposed-development exact-state corpus  
**Current authorization:** one small, disposable non-glass option-ambiguity authoring screen; no benchmark collection

This document is an additive paper-story decision. It does not rewrite the
frozen Phase 2.5A, Phase 2.5B, or Phase 2.5B-R records. Its quantitative source
is the immutable PIVOT-0 evidence package under `results/iclr27/`, together
with the frozen artifacts cited below.

## 1. Why the deployable-router story is closed

The historical gates retain their exact meanings:

| Stage | Frozen status | Paper consequence |
|---|---|---|
| Phase 2.5A | `PASS-SUPPORT`, marginal | The 20-source corpus contains strict option support, but the pass is narrow and all strict Retreat support is glass. |
| Phase 2.5B | `INCONCLUSIVE`, fail-closed | No source-cross-fitted outcome or value method clears the utility and strict-recall gates. No candidate, Screen A, or fresh outcome collection was authorized. |
| Phase 2.5B-R | `CHOICE_CAPACITY_BOTTLENECK` | A fixed tiny nonlinear head passes the conditional strict D/R point-estimate gate while both linear heads fail; no complete deployable ADR passes `method_pass`. |

The Phase 2.5B-R machine label remains `CHOICE_CAPACITY_BOTTLENECK` for
provenance. The paper-facing magnitude decomposition adds an important
qualification. For the tiny candidate, OracleGate plus learned choice is only
`0.0725` source-macro utility below Oracle, whereas LearnedGate plus
OracleChoice is `0.2928` below Oracle. The learned benefit gate is therefore
the dominant empirical loss in the composed system. This does not rename the
ordered machine decision; it explains why that machine decision cannot be
promoted into a nonlinear-router method story.

The resolver's historical authorized-action sentence remains frozen in its
own record. PIVOT-0 closes that candidate method storyline after the broader
evidence synthesis; the current paper-facing action is the non-glass screen
stated at the end of this document.

No deployable ADR passes the unchanged method gate. The strongest Phase 2.5B
learned method gains only `+0.0178` source-macro utility over source-cross-
fitted Risk -> Best Fixed and misses the strict Base/Detour/Retreat recall
requirements. The tiny nonlinear choice head is consequently a conditional
capacity diagnostic only. It is neither the paper method nor authorization to
train a larger classifier, router, temporal model, or world model. There is no
Phase 2.5B-R2.

## 2. New problem statement

The new paper asks what a scalar catastrophe-risk label leaves unidentified.
For a saved decision state `x`, let the frozen branch utilities be `U_B`,
`U_D`, and `U_R` for Base, Detour, and Retreat. Under the existing utility,
epsilon, and tie-breaking contract:

```text
R(x)     = 1[Base catastrophes]
B(x)     = 1[max(U_D, U_R) > U_B + epsilon]
O(x)     = argmax over {Base, Detour, Retreat}
A_int(x) = max(U_D, U_R) - U_B
A_DR(x)  = U_D - U_R
```

The central distinction is:

```text
Base catastrophe risk
    != intervention benefit
    != best intervention
    != sequentially realizable intervention
```

`R` asks whether Base catastrophizes. `B` asks whether any available
intervention improves utility over Base. `O` asks which option is best. A
sequential policy must additionally decide from a repeatedly observed process
when a statewise opportunity is available without collapsing Base behavior or
firing on controls. Confusing these targets turns a detector error, a benefit-
gate error, an option-choice error, and a realization error into one opaque
"router accuracy" number.

"Risk is not regret" is therefore a decision-diagnostic statement: Base risk
alone does not determine the regret of choosing Base or the value of a
particular intervention. It is not a claim that the current corpus establishes
this broadly across hazards, tasks, or VLA backbones.

## 3. Exact-state interventional outcome protocol

The current synthesis uses all 273 eligible decisions from 20 source states in
the frozen `d4751330395e` capture. At each decision, Base, privileged Detour,
and Retreat start from the same serialized simulator, controller, and RNG
state. The observed branch outcome is mapped to the frozen utility:

| Outcome | Utility |
|---|---:|
| task success | `+1` |
| safe noncompletion | `0` |
| catastrophe | `-1` |

The canonical benefit test is the existing strict `A_int > 0` test; no new
threshold is introduced. The Oracle tie order is Base, then Detour, then
Retreat. Strict support excludes ties under the pre-existing `gamma=0.5`
support-audit rule, which must not be confused with the benefit epsilon.

The protocol observes all three outcomes for a finite option set at saved
states. It is full-information for those branches, not an observational causal
estimator and not evidence about unexecuted controllers. Source state is the
independent unit. Conditions, placements, horizons, branches, and frames are
repeated observations; pooled frame-level inference is prohibited.

The capture names three conditions—`glass`, `offpath`, and `noglass`—but they
are treatment and controls within one `glass_recovery` mechanical design. They
must not be reported as three independent hazard families. Historical
train/calibration/development labels are retained only for provenance; all 20
sources are exposed development for this synthesis.

## 4. Risk-benefit-option-sequential decomposition

| Layer | Observable target | Diagnostic question | What failure means |
|---|---|---|---|
| Risk | `R` | Does Base catastrophize? | The monitor misses or falsely predicts Base catastrophe. |
| Benefit gate | `B` or `A_int` | Is the best intervention better than Base? | A risky state may not benefit, or a non-catastrophic Base outcome may still be improvable. |
| Option choice | `O` or `A_DR` conditional on benefit | Which intervention has higher utility? | A fixed fallback can destroy task value or miss a safer alternative. |
| Sequential realization | first-crossing policy over time | Can the statewise decision be reached at the right time from reset? | Offline separation may collapse to Base, miss known opportunities, or fire on controls. |

For the oracle-hybrid attribution, define

```text
oracle_opportunity = U_oracle - U_risk_best_fixed

choice_recovered_fraction =
    (U_oracle_gate_learned_choice - U_risk_best_fixed)
    / oracle_opportunity

gate_recovered_fraction =
    (U_learned_gate_oracle_choice - U_risk_best_fixed)
    / oracle_opportunity

full_recovered_fraction =
    (U_full - U_risk_best_fixed)
    / oracle_opportunity
```

These are attribution diagnostics on frozen source-OOF predictions. OracleGate
and OracleChoice are unavailable at deployment and cannot count as methods.

## 5. Current supporting evidence

### 5.1 Risk and benefit disagree, but only in 19 decisions

The raw decision table is:

| Base risk `R` | `B=0` | `B=1` | Total |
|---|---:|---:|---:|
| `R=0` | 172 | 9 | 181 |
| `R=1` | 10 | 82 | 92 |
| **Total** | **182** | **91** | **273** |

The equal-source mean joint proportions are:

| Source-macro joint proportion | `B=0` | `B=1` |
|---|---:|---:|
| `R=0` | 0.5831 | 0.0278 |
| `R=1` | 0.0742 | 0.3150 |

Thus `P(B=1 | R=1)=82/92=0.8913`, `P(B=1 | R=0)=9/181=0.0497`, and
`P(R=1 | B=1)=82/91=0.9011`. Under the repository convention of averaging a
conditional rate equally across denominator-bearing sources, the corresponding
source-macro estimates are `0.8700` over 20 sources, `0.0415` over 19 sources,
and `0.9328` over 19 sources. Ratios of source-macro joint cells are different
estimands and are not substituted for these values.

The 19 disagreements occur in 12 independent sources, but the two directions
are mechanically segregated:

| Disagreement | Decisions | Sources | Condition support | Horizon counts |
|---|---:|---:|---|---|
| risk-positive, no benefit (`R=1,B=0`) | 10 | 6 | glass 9; offpath 1 | H5/H10/H30/H40 = 6/1/1/2 |
| risk-negative, positive benefit (`R=0,B=1`) | 9 | 6 | noglass 5; offpath 4 | H5/H10/H20/H30 = 5/2/1/1 |

The source sets are disjoint. There are no glass `R=0,B=1` states and no
noglass `R=1,B=0` states. Source diversity therefore supports the existence of
risk-benefit disagreement inside this design, not a broad cross-family law.

### 5.2 Same binary risk can require different strict actions

Witnesses are selected by a deterministic exhaustive rule: form strata
matched exactly on source, condition, horizon, and binary `R`; retain every
stratum containing at least two different strict optimal labels. The tightest
match yields exactly three pairs across two sources, so no visually preferred
subset is chosen.

| Source / condition / horizon / `R` | State and realized `(Base, Detour, Retreat)` | Strict action | Why `R` is insufficient |
|---|---|---|---|
| `69fa7e93…` / glass / H5 / 1 | `43347d99…`: (catastrophe, success, safe noncompletion); `f83d0199…`: (catastrophe, catastrophe, safe noncompletion) | Detour; Retreat | Both Base branches catastrophize, but only one Detour completes safely. |
| `69fa7e93…` / offpath / H5 / 0 | `db05c179…`: (success, safe noncompletion, safe noncompletion); `09da0904…`: (safe noncompletion, success, safe noncompletion) | Base; Detour | Neither Base branch catastrophizes, yet one has positive Detour value. |
| `b830bede…` / noglass / H5 / 0 | `7cc66a27…`: (success, safe noncompletion, safe noncompletion); `2ccd971e…`: (safe noncompletion, success, safe noncompletion) | Base; Detour | The same binary risk status hides whether Base or Detour completes. |

These are same-`R` witnesses, not claims of identical learned risk probability.
The compact witness artifact reports observation hashes and deployable hidden,
proprioceptive, and nominal-action features; raw image pixels are not retained.
The tight exact D/R witness has effective independent support of one source.

### 5.3 Option support is source-diverse but glass-dominated

Nineteen of 20 sources contain both `B=0` and `B=1`. Nine sources contain a
strict Base/Detour flip, nine a strict Base/Retreat flip, and five a strict
Detour/Retreat flip when conditions are pooled. Those pooled counts conceal the
mechanical concentration:

| Condition | Decisions / sources | Strict Base | Strict Detour | Strict Retreat | D/R ties |
|---|---:|---:|---:|---:|---:|
| glass | 101 / 20 | 2 states / 2 sources | 28 / 9 | 23 / 9 | 31 |
| offpath | 86 / 17 | 24 / 12 | 4 / 4 | 0 | 0 |
| noglass | 86 / 17 | 35 / 12 | 5 / 4 | 0 | 0 |

All 23 strict Retreat states are glass. Glass supplies 51/60 strict D/R
states across 16 sources, versus 17 sources for all 60; non-glass supplies
only nine strict Detour states across six sources and no strict Retreat state. Only two sources contain a
within-glass strict D/R flip. The overall five-source D/R-flip count partly
combines a Detour label in one condition with a Retreat label in glass.

### 5.4 Risk-only regret and oracle-hybrid attribution

All values below are source-macro estimates on the same 20 source-OOF blocks.
`Risk -> Best Fixed` selects its option and threshold within each outer fold;
it is not a globally fixed Risk -> Detour rule.

| Policy | Utility | Success | Catastrophe | Safe noncompletion | Intervention | Regret to Oracle |
|---|---:|---:|---:|---:|---:|---:|
| Always Base | 0.1306 | 0.5197 | 0.3892 | 0.0911 | 0.0000 | 0.4336 |
| Always Detour | 0.2547 | 0.4347 | 0.1800 | 0.3853 | 1.0000 | 0.3094 |
| Always Retreat | -0.1858 | 0.0000 | 0.1858 | 0.8142 | 1.0000 | 0.7500 |
| Risk -> Detour | 0.2478 | 0.4989 | 0.2511 | 0.2500 | 0.4981 | 0.3164 |
| Risk -> Retreat | 0.1297 | 0.4394 | 0.3097 | 0.2508 | 0.3506 | 0.4344 |
| Risk -> Best Fixed | 0.2544 | 0.5022 | 0.2478 | 0.2500 | 0.4947 | 0.3097 |
| Oracle Benefit Gate -> best fixed Detour | 0.4900 | 0.6383 | 0.1483 | 0.2133 | 0.3428 | 0.0742 |
| OracleGate + tiny learned choice | 0.4917 | 0.6158 | 0.1242 | 0.2600 | 0.3428 | 0.0725 |
| Tiny LearnedGate + OracleChoice | 0.2714 | 0.4878 | 0.2164 | 0.2958 | 0.4567 | 0.2928 |
| Full tiny ADR | 0.2311 | 0.4733 | 0.2422 | 0.2844 | 0.4567 | 0.3331 |
| Oracle | 0.5642 | 0.6383 | 0.0742 | 0.2875 | 0.3428 | 0.0000 |

The oracle opportunity over Risk -> Best Fixed is `0.3097`. For the tiny
candidate, OracleGate plus learned choice recovers `0.7659` of that
opportunity, LearnedGate plus OracleChoice recovers `0.0547`, and the full
composition recovers `-0.0753`. The primary split-linear choice hybrid reaches
`0.4975`, slightly above the tiny hybrid's `0.4917`; the tiny head's diagnostic
result is its strict D/R balanced-accuracy/recall pass, not universal hybrid
utility superiority.

### 5.5 Statewise evidence does not establish sequential realization

The frozen sequential line provides a boundary rather than a rescue:

| Stage | Frozen evidence | Interpretation |
|---|---|---|
| Matched-state OOF | Oracle `0.5642`; Risk -> Best Fixed `0.2544`; strongest Phase 2.5B learned OutcomeRouter `0.2722` | A large statewise oracle opportunity exists, but the learned gain is only `+0.0178` and no method passes. |
| P2 | Four stable development sources x three conditions; success/catastrophe changes from 58.3%/33.3% to 66.7%/25.0% at 16.7% intervention | Small development-only benefit; 2/2 known T-20 recovery opportunities are missed. |
| P2.5 | Raw missed-recovery-v-control AUC `0.357`; simple moving-average variants `0.286` | Smoothing, accumulation, stability, run length, and trend do not repair ordering. |
| P3.1 | Strict nine-source LOSO: recovery-open-v-hard-control AUC `1.000`; intervention-needed-v-hard-control AUC `0.995`; source-macro accuracy `0.632` | Offline statewise separability exists in the frozen representation. |
| P3.2 | Eight fresh sources x three conditions: Direct chooses Base 24/24, recovers 0/8 glass episodes, and misses 2/2 known recoveries | The offline separation does not transfer to fresh first crossing; the direct policy collapses operationally to Base. |

The older P2 policy chooses Detour on 5/24 P3.2 episodes, recovers 1/8 glass
episodes, and retains task success on 16/16 controls. That contrast still does
not authorize reliable sequential intervention.

## 6. Unsupported claims

The current evidence does not support any of the following:

- a successful, deployable, or confirmatory router;
- broad risk-benefit or option-choice generalization across mechanically
  distinct hazard families, tasks, embodiments, or VLA backbones;
- treating glass, offpath, and noglass as three independent hazard families;
- reliable non-glass Retreat support or broad Detour-versus-Retreat ambiguity;
- presenting the tiny nonlinear choice head as the method, as architecture
  selection, or as evidence that a larger MLP would solve benefit gating;
- superiority of outcome decomposition, hidden representations, or ADR over
  direct value, risk, geometry, or fixed-option baselines;
- reliable sequential first crossing, recovery-window detection, or learned
  end-to-end recovery;
- causal effects beyond the finite options and exact serialized states that
  were actually branched;
- pooled frame-, branch-, condition-, or decision-level inference as if those
  observations were independent;
- novelty as the first VLA failure detector, action-conditioned verifier,
  selective intervention layer, VLA safety benchmark, or use of
  "counterfactual" reasoning.

No method-success, deployment-success, or broad benchmark claim is authorized.

## 7. Required benchmark expansion

A claim-ready expansion must introduce diversity that the current capture does
not contain. At minimum it requires two VLA backbones; three mechanically
distinct hazard families; matched on-hazard, offpath, and no-hazard controls;
Base plus two semantically distinct interventions; source-balanced strict
Base/Detour/Retreat support; and mechanically authored states in which Base
risk is held similar while benefit or the best option changes.

Every branch must restore the exact simulator, controller, policy-history, and
RNG state. Sources—not frames or branches—must be disjoint across training,
calibration, and test. Authoring may use mechanics and disposable branch
outcomes, never router scores. A small disposable screen must precede one
predeclared frozen collection; screen sources and outcomes cannot enter that
collection.

The complete design and release gates are specified in
[`BALANCED_INTERVENTION_BENCHMARK_SPEC.md`](BALANCED_INTERVENTION_BENCHMARK_SPEC.md).
PIVOT-0 authorizes only the small non-glass option-ambiguity authoring screen.
It does not authorize the frozen collection, a second backbone run, or any
router training.

## 8. Nearest-work differentiation

The distinction is operational and narrow, not a priority claim:

| Nearest work | What it establishes | PIVOT-0 distinction and boundary |
|---|---|---|
| SAFE | Hidden-state VLA failure scores, temporal models, and calibrated alarms across multiple VLA families and tasks | PIVOT-0 branches a finite option set from identical saved states to observe intervention-specific outcomes. It does not compete on broad failure detection or sequential calibration. |
| CheckVLA | Action-conditioned world-model verification, conformal control of intervention, and deployable suffix repair | PIVOT-0 uses simulator-realized branch outcomes rather than a learned rollout world model and does not rewrite action suffixes. CheckVLA has stronger sequential breadth. |
| CoWAM | Conservative preserve/override/abstain selection using predicted futures, candidate pools, and coordination contracts | This is the nearest selective-intervention threat. PIVOT-0's defensible asset is exact serialized-state branching with all option outcomes and source-paired diagnosis; it cannot claim the first selective intervention layer. |
| SafeVLA | Training-time safety alignment and broad physical-hazard evaluation | PIVOT-0 leaves the VLA frozen and diagnoses post-hoc option value. The current single mechanical design is not a broad VLA safety benchmark. |
| Counterfactual VLA | Generated counterfactual reasoning traces used to revise driving meta-actions | PIVOT-0's "counterfactuals" are realized simulator branches from identical full states, not language reasoning traces. Novelty cannot rest on terminology. |

## 9. Proposed title, abstract, and four contributions

### Proposed title

**Risk Is Not Regret: Exact-State Intervention Diagnostics for
Vision-Language-Action Policies**

### Abstract

Runtime VLA safety is often formulated as catastrophe detection, although a
failure probability does not determine whether intervention improves utility,
which intervention is best, or whether a statewise decision can be realized
sequentially. We study these distinctions with exact-state branching: Base,
Detour, and Retreat are restored from identical simulator, controller, and RNG
states and evaluated under a common task-success, catastrophe, and safe-
noncompletion utility. In the current 20-source, 273-decision development
corpus, risk and intervention benefit disagree on 19 decisions spanning 12
sources. Exhaustive tight matching finds three same-binary-risk,
different-strict-action pairs across two sources. Oracle-hybrid attribution
shows that a tiny learned choice head with an oracle benefit gate loses only
0.0725 source-macro utility from Oracle, while the learned benefit gate with
oracle choice loses 0.2928; nevertheless no deployable advantage router passes
the frozen method gate. The evidence is mechanically narrow: all 23 strict
Retreat states and 51/60 strict Detour/Retreat states are glass, while offpath
and no-glass are controls within the same glass-recovery design. Frozen
sequential experiments further show that strong offline statewise separation
can collapse to Base at fresh first crossing. We therefore identify a
glass-scoped benchmark pivot, not a successful router, and specify the balanced
multi-backbone, multi-mechanism evidence required for a broader claim.

### Four contributions

1. An exact-state realized-outcome protocol that operationally separates Base
   catastrophe risk, intervention benefit, best-option choice, and sequential
   realization for a fixed option set.
2. A source-aware, model-free audit that quantifies risk-benefit disagreement
   and exhaustively selects same-risk, different-strict-action witnesses
   without pooled-frame inference or score-based cherry-picking.
3. An oracle-hybrid and sequential-gap diagnosis showing where the frozen
   selector loses value while retaining the null result that no deployable ADR
   passes.
4. A glass-scoped story decision and a predeclared specification for the
   source-balanced, mechanically diverse benchmark needed before broader
   claims are considered.

## 10. Main figure/table plan

The main text should lead with diagnostics rather than a router architecture.

| Item | Content | Claim role |
|---|---|---|
| Figure 1: risk-benefit flow | Exact-state Base/Detour/Retreat branching, the raw `R x B` flow, and the two disagreement directions with source counts | Defines why scalar risk is not intervention regret. |
| Figure 2: oracle-gap waterfall | Risk -> Best Fixed (`0.2544`); Tiny LearnedGate + OracleChoice (`0.2714`); OracleGate + primary split-linear choice (`0.4975`); OracleGate + tiny nonlinear choice (`0.4917`); full tiny ADR (`0.2311`); OracleGate + OracleChoice (`0.5642`) | Central attribution figure. It must note that oracle hybrids are diagnostic and that the tiny head is not the method. |
| Figure 3: source/family support | Condition-level independent-source support bars for disagreement and strict Base/Detour/Retreat, split into glass, offpath, and noglass; annotate all 23 Retreat and 51/60 D/R states as glass. The companion CSV retains every per-source flag. | Makes the `GLASS_SCOPED_BENCHMARK_PIVOT` decision visually unavoidable. |
| Table 1: risk-only regret | Utility, catastrophe, success, safe noncompletion, intervention, regret, and Oracle-improvement recovery for fixed, risk-gated, hybrid, full, and Oracle policies | Prevents a weak fixed-Retreat comparison from carrying the story. |
| Table 2: tight witnesses | All three deterministic source/condition/horizon/`R`-matched pairs, branch outcomes/utilities, hashes, and deployable feature summary | Demonstrates same-risk action ambiguity with its two-source limit visible. |
| Table 3: source support | Both-benefit-label, strict flip, disagreement, family, and effective-source counts | Separates decision volume from independent support. |
| Table 4: sequential realization gap | Matched-state Oracle opportunity and learned result; P2/P2.5 offline timing diagnostics; P3.1 separability; P3.2 fresh first-crossing null and missed opportunities | Keeps sequential failure as a deployment boundary, not a second method arc. |

Exactly one next action follows from this story decision: run a small,
disposable non-glass option-ambiguity authoring screen under the separate
benchmark specification. Nothing in this document authorizes the subsequent
frozen collection.
