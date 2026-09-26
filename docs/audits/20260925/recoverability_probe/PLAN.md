# Fixed-Detour completion probe: frozen first step

The user requested this existing-data probe and its interpretation on Quest on
2026-09-25. This specifically authorizes small probe fits on the historical
273-decision capture. No rollout, new source, controller change, benchmark
expansion, D8 outcome access, or confirmatory test is part of this work.

## Question and labels

Can the frozen VLA's current hidden state predict whether **the fixed Detour
completes within the recorded collector budget**, on an unseen physical source?
Compare it to Base catastrophe prediction on exactly the same records, features,
folds and fitting rule. Base task success is a separate interpretive target:
Detour failure does not imply Base failure, intervention necessity, failure of
other recovery skills, or physical impossibility. One realization is a noisy
binary observation, not a repeat-estimated recoverability probability.

The sealed job9288681 capture contains 273 decisions from20 sources:
101 glass,86 offpath and86 noglass, at T−40/30/20/10/5 where available. Its
4096-dimensional hidden state is stored as eight causal frames. The primary
probe uses the last valid frame, not future frames or concatenated horizons.
No row is relabeled or excluded based on the new predictions.

## Fixed analysis, before fitting

Configuration: `configs/recoverability/hidden_probe_v1.json`. Input hashes are
locked to the existing archive. Main regime:20-fold leave-one-physical-source-out
(LOSO) on all273 records. All conditions, placements and times sharing a source
remain together. PCA16 and standardization are fitted on the training fold only.
PCA is deterministic and uses training current frames; logistic loss gives each
training source equal total weight. L2=0.01, unpenalized intercept, no class
rebalancing, no hyperparameter or threshold search.

Five fixed inputs: hidden-only (primary); robot8+nominal-action7 (simple available
baseline); hidden+robot+action (sensitivity); condition+horizon one-hot (privileged
diagnostic, never deployable); training source-weighted prevalence (prior).
The three targets use identical preprocessing and recipes.

Two prespecified sensitivities: glass-only LOSO, which asks directly about
recoverability within hazardous scenes; and the historical5-source training to
8-source development split (calibration sources are unused because there is no
tuning). Historical labels stay preserved. All20 sources have already been used
in prior exploratory analyses; none is called a fresh test.

Report source-balanced and decision-pooled AUROC, average precision, Brier score,
balanced accuracy at fixed0.5, positive and negative source support, within-source
AUROC where identifiable, fixed five-bin calibration, and per-horizon/per-source
results. Main subsets are all, glass, controls, Base failures, and glass Base
failures. Outcome-conditioned subsets are retrospective diagnostics, not online
input selection. Also score inverse Base-risk against Detour success as a direct
risk-score baseline; AUROC comparisons between different targets are descriptive.

For primary LOSO scores, paired2000 source-block bootstrap resamples quantify
ranking and Brier uncertainty and hidden-minus-simple-baseline differences. These
intervals condition on already-fitted OOF scores; they do not refit models or
account for the complete model-selection history. Single-class samples have no
AUROC and are counted explicitly, not assigned0.5.

At prespecified Detour-success cutoffs0.1/0.2/0.5, report how many flagged states
actually succeed with Detour, Base, or either. This audits the danger of calling
low scores 'irrecoverable'; it is not a fitted stopping policy or a safety claim.

## Execution and interpretation

Run numerical data analysis and fitting only on a Quest `short` CPU Slurm job,
account p33100,4 CPUs,8GB,30-minute cap. Local tests use synthetic data only.
Record clean published commit, config/input/output hashes, folds, fitted model
parameters, convergence, seeds, software and Slurm provenance in a fresh result
directory. Preserve existing working edits and all historical artifacts.

Interpret the result using glass-only ranking, positive-source support,
source variability, simple baselines and calibration together. A high pooled
AUROC alone cannot establish recoverability detection. A failed linear probe
does not prove that representations contain no useful information. This first
step cannot estimate repeated success probability at a state, validate safe
abstention, or justify a new benchmark. The follow-up recommendation must be
based on this evidence and the existing repeat studies; no automatic new test.
