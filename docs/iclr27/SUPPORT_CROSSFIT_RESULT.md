# ICLR 2027 Phase 2.5B source-cross-fitted support rescue

- **Evidence role:** exposed-development-only
- **Corpus:** the sealed existing 20-source exact-state full capture
- **New rollout:** none
- **Fresh test outcomes:** prohibited
- **Primary preference:** `lambda=1`, `eta=0`
- **Maximum source-macro intervention rate:** `rho=0.60`
- **Reviewed result:** complete; formal gate **INCONCLUSIVE** (fail-closed)

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

## Reviewed result

- Protocol and execution commit:
  `df3168c75843f33484209b9908ac939814de31e9`
- Quest Slurm job: `5137872`, account `p33100`, partition `short`,
  `COMPLETED`, elapsed `00:02:03`, exit `0:0`
- Result root:
  [`support_crossfit_df3168c75843_20260829T115455Z_job5137872`](../../results/iclr27/support_crossfit_df3168c75843_20260829T115455Z_job5137872/)
- Machine decision:
  [`gate_decision.json`](../../results/iclr27/support_crossfit_df3168c75843_20260829T115455Z_job5137872/gate_decision.json)
- New rollout: none; fresh test outcomes loaded: false

All 20 outer folds were support-valid. Across the 14-source fitting sets, the
minimum Base/Detour/Retreat strict source supports were respectively `10/8/5`.
All 100 inner folds used for `beta` selection were support-valid. Thirteen
outer folds selected `beta=1.0` and seven selected `beta=0.25`. Every one of
the 320 method-fold calibration records obeyed the frozen source-macro
intervention cap; the largest floating-point value was `0.6000000000000002`.

Primary OOF source-macro results are:

| Method | Utility | Catastrophe | Intervention | Base recall | Detour recall | Retreat recall | Strict macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Risk -> Best Fixed | 0.2544 | 0.2478 | 0.4947 | 0.5595 | 0.5705 | 0.0556 | 0.3805 |
| Direct-Q | 0.1969 | 0.2717 | 0.4131 | 0.6260 | 0.3141 | 0.3704 | 0.3836 |
| Support-balanced Direct-Q | 0.1928 | 0.2708 | 0.3847 | 0.6031 | 0.2244 | **0.5185** | 0.4088 |
| Current Outcome Router | **0.2722** | **0.2286** | 0.4831 | 0.5926 | 0.3269 | 0.3704 | 0.4255 |
| Support-balanced Outcome Router | 0.2672 | 0.2394 | 0.4514 | **0.6534** | 0.3013 | 0.4815 | 0.4711 |
| SB-COR | 0.2428 | 0.2447 | **0.4236** | 0.6061 | 0.2500 | 0.4815 | 0.4452 |
| Hidden-only Outcome Router | 0.2644 | 0.2189 | 0.5222 | 0.4881 | **0.4359** | **0.5185** | **0.4780** |

The current Outcome Router is the strongest deployable learned method in OOF
utility, but it gains only `+0.0178` over Risk -> Best Fixed, below the frozen
`+0.03` requirement. Its paired source bootstrap interval for that difference
is `[-0.0747, 0.1000]`, with sign-flip `p=0.7131`; 13/20 source differences
are nonnegative. It also misses all three strict-recall requirements. Support
balancing improves Retreat recall but does not recover Detour or Base recall.
The ranking auxiliary does not rescue the selector and reduces utility below
the risk reference.

No branch in the frozen gate fires:

- **GO-SIGNAL fails:** every outcome candidate misses the Base, Detour, and
  Retreat recall floors and the required utility gain.
- **GO-VALUE-ONLY fails:** Direct-Q, support-balanced Direct-Q, and Pairwise
  Advantage all lose utility to Risk -> Best Fixed and miss strict recalls.
- **BENCHMARK-PIVOT fails:** no learned selector reaches both intervention
  recalls at `0.55`, but neither condition-only nor horizon-only reaches both
  diagnostic recalls at `0.55` either.
- **STOP-RESCUE does not fire:** cross-fitted Risk -> Best Fixed recovers only
  `0.2857` of source-macro Oracle value, below `0.85`, and the best learned
  Retreat recall is `0.5185`, above the frozen one-of-three random level.

The resulting **INCONCLUSIVE** is not promoted to a positive branch. It does
not freeze a candidate architecture and does not authorize Screen A, fresh
outcomes, or confirmatory rollout. Continuing requires an explicit revision of
the scientific plan rather than post-hoc threshold, utility, or model changes.

## Reproduction

After publishing a clean protocol commit, submit the CPU-only job from the
repository root:

```bash
scripts/quest_sync.sh submit setup/iclr27_support_crossfit.sbatch
```

The non-overwriting Quest result root records the exact commit, Slurm job, frozen
config, capture/support-audit hashes, all fold assignments and class support,
inner `beta` selection, calibration records, OOF predictions, source-paired
inference, plots, fold-model archive, and the machine gate. The reviewed Git
copy promotes the manifest, predictions, analysis tables, inference, plots,
and provenance; the 11 MB `model_parameters.npz` remains sealed on Quest under
the manifest-recorded SHA-256 rather than being duplicated into Git history.
