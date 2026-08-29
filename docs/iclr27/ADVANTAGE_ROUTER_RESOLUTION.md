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

Protocol frozen; Quest execution pending.

## 6. Gate-vs-choice error decomposition

Protocol frozen; Quest execution pending.

## 7. Margin and source robustness

Protocol frozen; Quest execution pending. Margin quartiles and the upper-half
subset are explanatory diagnostics only and cannot select a threshold or
change the primary decision rule.

## 8. Primary decision

Protocol frozen; the exhaustive resolver has not yet been executed.

## 9. Exactly one authorized next action

Protocol frozen; the action will be populated directly from the predeclared
decision mapping after the single Quest run.

## 10. Claim ledger update

No new method claim is authorized before the resolver run. Phase 2.5C, Phase 3,
Phase 4, fresh outcomes, and confirmatory rollouts remain blocked.
