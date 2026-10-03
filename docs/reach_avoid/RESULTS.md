# Reach-avoid initial-state results, 2026-10-03

The bounded study completed real rollouts and trained readouts. It did **not**
establish a useful incremental feasibility judgment, information destruction,
or a general reasoning deficit. The frozen validation-selected readout failed
to transfer reliably from slit to cage constructions. Strong simple-cue rankings
show that shortcut explanations remain available.

Read the [Chinese executive summary](RESULTS_ZH.md),
[protocol](PROTOCOL.md), [dated reporting amendment](REPORTING_AMENDMENT.md), and
[failure ledger](FAILURE_LEDGER.md). No additional run is queued by this result.

## Outcomes and labels

All 168 initial states were valid: 72 full safe witnesses, 48 conditional
RA-SEP-1 certificates, and 48 UNKNOWN. At execution horizon T=300:

| Independent initial label | Safe completion | Safe timeout | Collision | Invalid | Observation only |
|---|---:|---:|---:|---:|---:|
| Feasible | 13 | 44 | 15 | 0 | 0 |
| Conditional infeasible | 0 | 31 | 16 | 1 | 0 |
| UNKNOWN | 0 | 0 | 0 | 0 | 48 |

All 31 collision endpoints contain protected contacts; none is only a declared
crossing-guard violation. The illegal command at L04_cage_.020 remains undefined
as a full-horizon endpoint and lies outside primary fitting/evaluation. The
28-180 mm sampled unresolved bracket is 152 mm wide, not a true-boundary estimate.

## Frozen comparison

Six train and two validation layout groups use slit scenes; four test groups use
cage scenes. All paired variants remain grouped. Primary counts are 30/10/20
known states; the test failure subset is 18 = 8 infeasible + 10 feasible.

Validation selected `layers_02` with `mlp32_alpha0.001`, log loss 0.00046562.
Test log loss is 6.752394. All 20 test scores are below 0.001578, so the fixed
0.5 threshold classifies every state as feasible, including all eight negatives.

| Representation | Known AUROC | Failure-subset AUROC | Known balanced accuracy |
|---|---:|---:|---:|
| Selected layer 2 / MLP | .500 | .525 | .500 |
| Own vision tower | .802 | .825 | .500 |
| Final native features | .708 | .650 | .646 |
| DINOv2 patch mean | .833 | .800 | .708 |
| RGB pooled/red cues | .979 | .975 | .625 |
| Privileged geometry | 1.000 | 1.000 | 1.000 |

Main descriptive evidence is the four paired layouts, not pooled significance:

| Layout | Selected known AUROC | Selected failure AUROC | Failure score only | Failure + feasibility |
|---|---:|---:|---:|---:|
| L00 | .500 | .500 | .250 | .250 |
| L03 | .167 | .167 | .667 | .667 |
| L06 | .333 | .500 | .000 | .250 |
| L08 | 1.000 | 1.000 | .667 | 1.000 |

The SAFE-style initial-state failure probe has only two safe completions and
28 failures in training; all six OOF folds retain both classes. It predicts the
full-T failure endpoint, not short-horizon collision, and is not original SAFE
temporal detection. The infeasibility stacker AUROC changes .4750 to .5375,
while Brier worsens .247579 to .379668 and log loss .688300 to 1.168427.
Both fixed-threshold balanced accuracies remain .5. No rescue or useful
calibrated incremental benefit is demonstrated.

The predeclared linear-head diagnostics (own vision .979, selected layer .719)
do not replace the primary chosen model. They reinforce head/selection
sensitivity; they do not establish that information was destroyed.

## Separately dated nuisance supplement

Width-only AUROC is .833/.850 on all-known/failure-subset states, balanced
accuracy .750. On the 12 known numeric-gap test states, width alone achieves
AUROC and balanced accuracy 1.0. All 12 are policy failures.

Continuous target pixel fractions rank known labels perfectly, including on
held-out cages, but their all-variant fixed-threshold balanced accuracy is .5.
The earlier binary visible/not-visible check therefore did not remove continuous
visibility shortcuts. Predictive cues do not prove the VLA uses those cues.

The eight UNKNOWN test-cage states remain unknown. Width-only heads give very
high scores while visibility/combined heads give low scores; none supplies a
certificate or a witness. UNKNOWN coverage is 28.6% in the all-variant test scope
and 40% in the numeric-gap scope. No UNKNOWN accuracy is computed.

## Evidence and scope

The two constructors share one separator proof family and an explicitly stated
digital crossing contract. This is neither independent-mechanism generalization
nor unrestricted physical impossibility. Every readout is at t=0; no initial
label is extrapolated to an AEGIS deadlock state. Four test groups do not support
reliable nominal bootstrap coverage. Original 2,000-draw calculations remain in
an exploratory appendix; the primary AUROC range is [.28125,.8125] and the paired
incremental delta range [0,.25]. No significance claim is made.

Independent readback checked 168 states and 1,464 saved hashes. A separate
read-only auditor recomputed all 47 representations' metrics, group scores,
source/output hashes, nested selections and principal bootstrap outputs with
no discrepancy. Primary numerical outputs are unchanged by the supplement.

CPU matrix source: `87c5b06ef2d5e7b0f21f38350044e9f495097a59`.
GPU matrix/frozen analysis: `465f112ea18bf12204c2c87f1df99c75dd67bd4a`.
Pinned public upstream: `2457feed5968ae803926e178c8ce8243b9ecdcf9`.
CPU array 8419904 and GPU array 8422120 are fully COMPLETED/0:0.
Matrix GPU allocation:153m23s of the approved180min; CPU labels:136m44s core time.
All study GPU allocations including development failures:174m31s. Zero paid API.

Local artifacts: `results/reach_avoid/session_20261003/`; frozen analysis under
`analysis_465f112_full`, separate supplement under
`supplemental_nuisance_20261003T205611`. Raw Quest roots remain unchanged in
`/projects/p33100/siosio/crashbench_safelibero/reach_avoid_matrix_labels/20261003T182027Z_cpu_87c5b06ef2d5`
and `reach_avoid_matrix_gpu/20261003T185040Z_gpu_465f112ea18b`.
