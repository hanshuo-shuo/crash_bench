# ICLR 2027 Phase 2.5B source-cross-fitted support rescue

- **Evidence role:** exposed-development-only
- **Corpus:** the sealed existing 20-source exact-state full capture
- **New rollout:** none
- **Fresh test outcomes:** prohibited
- **Primary preference:** `lambda=1`, `eta=0`
- **Maximum source-macro intervention rate:** `rho=0.60`
- **Reviewed result:** pending a clean, tested, published protocol run

## Leakage contract

Each source is held out once. For an outer fold, the other 19 sources are
ordered by the frozen SHA-256 rule and divided into 14 fitting sources and five
calibration sources. The held source contributes to none of PCA, feature
standardization, model fitting, ranking-`beta` selection, or threshold
calibration. It receives one prediction per method, after which all 20 OOF
source blocks are pooled for development-only analysis.

The ranking auxiliary selects `beta` from `{0.25, 1.0}` using five inner source
folds contained wholly inside the 14 outer fitting sources. Inner validation
sources are excluded from their corresponding PCA and model fit. Outer
calibration sources select only the operating threshold, never `beta`.

Historical `train`, `calibration`, and `development` labels are preserved for
provenance but have no modeling role in Phase 2.5B. The trainer preflights the
metadata before opening option outcomes, rejects every test/fresh-test split,
and verifies the capture hashes against the reviewed Phase 2.5A manifest.

## Support balancing and ranking

`SB-OutcomeRouter`, `SB-DirectQ`, and the outcome component of `SB-COR` use
decision weights constrained to give every fitting source equal total weight
and to give `strict_base`, `strict_detour`, and `strict_retreat` equal total
weight. The tie/no-good rows form a separate auxiliary stratum and are not
forced to equal a strict class. Individual weight is capped at 12 times the
mean; a fold fails rather than relaxing the source or strict-stratum margins.

`SB-COR` adds pairwise logistic ranking only where two realized option
utilities differ. Exact utility ties never receive an order. Pair mass inherits
the source/strict-stratum-balanced decision mass and is split equally across
the unequal pairs of that decision.

The beneficial-only `SB-Risk+TwoStage` subset can make exact source and
Detour/Retreat margins structurally incompatible (for example, a fold may
contain several Detour-only sources but few Retreat-bearing sources). Its
predeclared baseline weight is therefore the product of inverse source
frequency and inverse strict-intervention-stratum frequency; auxiliary ties
retain source-frequency weight. It uses the same 12× cap and does not claim the
exact two-margin property reserved for M1/M2.

## Fair calibration

Every learned or risk-gated method uses the same outer-calibration rule:

```text
maximize source-macro calibration utility
subject to source-macro intervention rate <= 0.60
```

Ties prefer lower catastrophe, then lower intervention, then the higher strict
`score > threshold` threshold. `Risk->BestFixed` jointly selects Detour versus
Retreat and its threshold on the same five calibration sources. No method is
calibrated by closeness to a target rate.

Condition-only and horizon-only models are OOF diagnostics. These metadata are
explicitly forbidden as deployable features and cannot support `GO-SIGNAL` or
`GO-VALUE-ONLY`.

## Gate interpretation

The machine gate implements the revised plan's `GO-SIGNAL`,
`GO-VALUE-ONLY`, `BENCHMARK-PIVOT`, and `STOP-RESCUE` branches. Any gap between
those rules is `INCONCLUSIVE`, which is fail-closed. Screen A is authorized for
a method/value GO, or for benchmark-only authoring after `BENCHMARK-PIVOT`;
confirmatory rollout remains unauthorized in every Phase 2.5B branch.

## Reproduction

After publishing a clean protocol commit, submit the CPU-only job from the
repository root:

```bash
scripts/quest_sync.sh submit setup/iclr27_support_crossfit.sbatch
```

The non-overwriting result root records the exact commit, Slurm job, frozen
config, capture/support-audit hashes, all fold assignments and class support,
inner `beta` selection, calibration records, OOF predictions, source-paired
inference, plots, and the machine gate.
