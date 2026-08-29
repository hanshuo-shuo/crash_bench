# Phase 2.5B-R counterfactual advantage router resolution

## 1. Frozen context

Phase 2.5A is a marginal `PASS-SUPPORT`, not a strong GO. Phase 2.5B is frozen
as `INCONCLUSIVE`. Its strongest deployable learned method, OutcomeRouter,
improves source-macro utility over Risk -> Best Fixed by only `+0.0178`, below
the frozen `+0.03` gate. Hidden-only OutcomeRouter retains more Detour/Retreat
states but loses Base retention; full OutcomeRouter retains Base more often but
substantially loses Detour/Retreat recall. Risk+TwoStage predicts Base
catastrophe risk, not counterfactual benefit relative to Base.

All Phase 2.5A and Phase 2.5B artifacts remain frozen. This resolver uses only
the existing exact-state capture with fingerprint `d4751330395e`. It performs
no simulator rollout and reads no fresh or test outcomes.

## 2. Why the original 2.5B gate was non-exhaustive

The original predeclared branches left the observed grey region unmatched:
learned choice signal was neither strong enough for a method GO nor weak enough
for a STOP branch. This resolver is therefore explicitly a **post-hoc protocol
amendment**. Its five-way decision rule, fixed model set, calibration rule, and
margin subsets are frozen before resolver outcomes are examined.

## 3. Counterfactual advantage formulation

For each frozen exact state, the resolver constructs

```text
A_int = max(U_D, U_R) - U_B
A_DR  = U_D - U_R
```

The deployable rule selects Base when predicted `A_int <= tau`; otherwise it
selects Detour when predicted `A_DR >= 0` and Retreat when predicted `A_DR < 0`.

## 4. Experimental contract

The experiment reuses the Phase 2.5B capture hashes, source definitions,
utility, decision grid, source-equalized trajectory weighting, LOSO outer
evaluation, deterministic 14-fit/5-calibration split, `rho <= 0.60`, seed
`2027`, and 5,000-replicate shared source bootstrap. PCA, scaling, Ridge, and
MLP preprocessing are fitted only on the 14 fitting sources. `tau` is selected
only on the five inner source-held-out calibration sources. Condition and
horizon are diagnostic metadata, never deployable features.

The only new models are ADR-linear-full, primary ADR-linear-split, and the
fixed 32-unit one-hidden-layer ADR-tiny-nonlinear-choice capacity diagnostic.
OracleGate and OracleChoice combinations are diagnostic-only and cannot count
as deployable methods.

## 5. Main results

The completed reviewed run is:

- implementation/execution commit: `bf032cb1d6d482423b229e304997ea29134bb4bb`;
- Quest Slurm job: `5148751`, account `p33100`, partition `short`,
  `COMPLETED`, elapsed `00:05:48`, exit `0:0`;
- result root:
  [`advantage_router_resolver_d4751330395e_20260829T135723Z_job5148751`](../../results/iclr27/advantage_router_resolver_d4751330395e_20260829T135723Z_job5148751/);
- machine decision:
  [`decision.json`](../../results/iclr27/advantage_router_resolver_d4751330395e_20260829T135723Z_job5148751/decision.json).

The first execution attempt, job `5148521`, stopped before completing the
first outer fold when L-BFGS reached the fixed 1,000-iteration limit. It
produced no resolver result. Commit `bf032cb` kept the architecture, loss,
initialization, seed, and iteration cap unchanged, accepted a finite endpoint
at that cap, and recorded it as non-converged. In the completed run, 17/20 MLP
folds converged and three stopped at the fixed cap.

| Method | Utility | Gain vs Risk -> Best Fixed | Catastrophe | Intervention | Base recall | Detour recall | Retreat recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Risk -> Best Fixed | 0.2544 | 0.0000 | 0.2478 | 0.4947 | 0.5595 | 0.5705 | 0.0556 |
| Frozen OutcomeRouter | 0.2722 | +0.0178 | 0.2286 | 0.4831 | 0.5926 | 0.3269 | 0.3704 |
| ADR-linear-full | 0.2097 | -0.0447 | 0.2539 | 0.4608 | 0.5874 | 0.3910 | 0.2593 |
| ADR-linear-split | 0.2222 | -0.0322 | 0.2306 | 0.5225 | 0.4999 | 0.4487 | 0.3704 |
| ADR-tiny-nonlinear-choice | 0.2311 | -0.0233 | 0.2422 | 0.4567 | 0.5749 | 0.4551 | 0.4444 |
| Oracle upper bound, diagnostic-only | 0.5642 | +0.3097 | 0.0742 | 0.3428 | 1.0000 | 1.0000 | 1.0000 |

The paired source-bootstrap utility-gain intervals are `[-0.1836, 0.0739]`,
`[-0.1725, 0.0819]`, and `[-0.1667, 0.0925]` for ADR-linear-full,
ADR-linear-split, and ADR-tiny-nonlinear-choice respectively. All 60 calibrated
fold-method records obey `rho <= 0.60`; the maximum is `0.5967`. No deployable
ADR satisfies `method_pass`.

On the 60 strict D/R states, the fixed conditional-choice audit is:

| Choice head | Balanced accuracy | Source-bootstrap 95% CI | Macro-F1 | Detour recall | Retreat recall | `choice_pass` |
|---|---:|---:|---:|---:|---:|---:|
| ADR-linear-full | 0.5579 | [0.4193, 0.6979] | 0.5539 | 0.7421 | 0.3738 | no |
| ADR-linear-split | 0.6369 | [0.5261, 0.7538] | 0.6356 | 0.8354 | 0.4385 | no |
| ADR-tiny-nonlinear-choice | **0.7253** | **[0.6465, 0.8167]** | **0.7297** | **0.8559** | **0.5947** | **yes** |
| Fixed-Detour diagnostic | 0.5000 | -- | 0.3665 | 1.0000 | 0.0000 | no |
| Condition-only diagnostic | 0.3744 | -- | 0.3461 | 0.6441 | 0.1047 | no |
| Horizon-only diagnostic | 0.5000 | -- | 0.3665 | 1.0000 | 0.0000 | no |

Thus the tiny nonlinear head improves conditional balanced accuracy by `+0.2253`
over fixed Detour and horizon-only, and by `+0.3509` over condition-only. Its
source-bootstrap recall intervals are `[0.7675, 0.9490]` for Detour and
`[0.4755, 0.7583]` for Retreat; the point estimates, not the confidence bounds,
define the frozen `choice_pass`.

## 6. Gate-vs-choice error decomposition

All three candidates use the same OOF full-feature Ridge `A_int` scores, so
their AUROC is `0.6062` and AUPRC is `0.4783`. Calibration changes their
operating points:

| Method | Benefit-Base recall | Benefit-intervention recall | Balanced accuracy | AUROC | AUPRC | Intervention |
|---|---:|---:|---:|---:|---:|---:|
| ADR-linear-full | 0.6036 | 0.5555 | 0.5796 | 0.6062 | 0.4783 | 0.4608 |
| ADR-linear-split | 0.5475 | 0.6310 | 0.5892 | 0.6062 | 0.4783 | 0.5225 |
| ADR-tiny-nonlinear-choice | 0.6092 | 0.5555 | 0.5823 | 0.6062 | 0.4783 | 0.4567 |

The tiny-head gate balanced-accuracy interval is `[0.5291, 0.6302]`, and its
AUROC interval is `[0.4968, 0.7245]`. It fails the predeclared secondary benefit-
gate diagnostic.

The attribution table reports source-macro utility; every oracle combination
is diagnostic-only.

| Choice head | OracleGate + OracleChoice | OracleGate + LearnedChoice | LearnedGate + OracleChoice | LearnedGate + LearnedChoice |
|---|---:|---:|---:|---:|
| ADR-linear-full | 0.5642 | 0.4708 | 0.2797 | 0.2097 |
| ADR-linear-split | 0.5642 | 0.4975 | 0.2831 | 0.2222 |
| ADR-tiny-nonlinear-choice | 0.5642 | 0.4917 | 0.2714 | 0.2311 |

For the tiny head, the OracleGate learned-choice loss is `0.0725`, the learned-
gate OracleChoice loss is `0.2928`, the full composed loss is `0.3331`, and the
residual interaction is `-0.0322`. The split head with OracleChoice reaches
`+0.0286` utility over Risk -> Best Fixed, still below `+0.03`, while retaining
only `0.4999` of strict Base states. The learned benefit gate/calibration is
therefore also a secondary bottleneck, but it does not override the ordered
primary rule triggered by the tiny-vs-linear conditional-choice result.

## 7. Margin and source robustness

The strict D/R `|U_D-U_R|` quartile boundaries are all `1.0`: 57 states are at
margin 1, only three are above it, and the middle two numerical bins are empty.
Under the frozen inclusive rule `margin >= median`, the upper-half subset is
therefore all 60 strict D/R states. The upper-half result is identical to the
full result and supplies no separate label-margin rescue. No margin threshold
was selected after seeing outcomes.

The tiny nonlinear choice result is strongest in glass, which contains 51/60
strict D/R states: balanced accuracy is `0.7219`, Detour recall `0.8463`, and
Retreat recall `0.5975`. The five no-glass and four off-path strict states are
too sparse for two-class family claims. By horizon, tiny-head Retreat recall is
`0.75` at horizon 10, `0.50` at 20, `0.00` at 30, `1.00` at 40 (three total
strict states), and `0.4576` at 5. These small-cell fluctuations and the wide
source-bootstrap Retreat interval remain limitations.

Complete per-source, per-family, per-horizon, and family-by-horizon results are
in [`source_metrics.csv`](../../results/iclr27/advantage_router_resolver_d4751330395e_20260829T135723Z_job5148751/source_metrics.csv)
and [`family_metrics.csv`](../../results/iclr27/advantage_router_resolver_d4751330395e_20260829T135723Z_job5148751/family_metrics.csv).

## 8. Primary decision

**`CHOICE_CAPACITY_BOTTLENECK`**

ADR-tiny-nonlinear-choice passes the full strict D/R `choice_pass`, while both
linear choice heads fail it. No complete deployable advantage router passes the
unchanged Phase 2.5B method gate. The machine-readable secondary flags are
`tiny_nonlinear_choice_only` and
`ADR-tiny-nonlinear-choice:benefit_gate_failure`. The resolver does not return
`INCONCLUSIVE`, does not select a deployable method, and does not authorize a
method-success claim.

## 9. Exactly one authorized next action

Freeze the nonlinear conditional-choice diagnosis and make nonlinear option
selection the sole method story without expanding the benchmark or model set.

## 10. Claim ledger update

The exposed existing corpus supports one narrow diagnostic claim: with an
oracle counterfactual-benefit gate, the fixed tiny nonlinear hidden-only choice
head clears the predeclared conditional D/R point-estimate thresholds, whereas
both linear heads do not. It does **not** support a deployable routing win, a
foundation-model or representation claim, or progression to Phase 2.5C,
Phase 3, Phase 4, fresh outcomes, or confirmatory rollout. Phase 2.5A and 2.5B
results retain their original frozen interpretations.
