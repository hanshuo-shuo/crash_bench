# Stage 1 closeout: engineering fit, not scientific verification

The eight-sample Stage 1 smokes remain valid: A and B each produced 8/8 exact
outputs, and both generative loss components backpropagated. The authorized fixed
full 120 experiment is now complete: **A fits 120/120 training verdicts; B fits
73/120**, while B reproduces the three observation/limitation rationale templates
correctly on 120/120. B has not achieved the full-dataset binary fit. These are
engineering results on one original source scene, with no generalization claim.

The current recommendation is **no-go for scientific training with this pool**.
Use the established training infrastructure for further engineering work, but
collect source-diverse, observable, evidence-verified data and resolve B's verdict
underfitting before a scientific A/B comparison. No new simulator data, additional
optimization run, hyperparameter search or paid API call was made in this task.

## Implementation and fixed configuration

The architecture is unchanged: Qwen/Qwen2.5-VL-3B-Instruct, revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`, with 29,933,568 language-LoRA parameters
across 252 language attention/MLP projections. Vision encoder, visual merger /
projector, embeddings, LM head and all other base weights remain frozen. Both
completed runs passed the frozen-vision checks and bitwise projector comparison.

The original [A](configs/proposed_A.json) and [B](configs/proposed_B.json) proposals
were executed without changing their optimization settings: rank 16, alpha 32,
dropout 0.05, LR 1e-4, constant AdamW, weight decay 0, BF16/SDPA, microbatch 1,
accumulation 4, seed 713, ten complete epochs and 300 updates. Both used the exact
same saved epoch orders, starting from fresh base-model adapters. No early
stopping, best-checkpoint selection or extension followed the observed scores.
Execution copies record the user's new full-training authorization; the earlier
proposal files retain their historical not-launched flag.

All 20 preflight unit tests passed on Quest, followed by 240 real Qwen encoding
checks. A/B have identical full inference inputs: 372 prompt tokens, attention
masks, image grids and image tensors. Student inputs are only two original
224x224 RGB views, task and constraints. Source ancestry, filenames, scene IDs,
geometry, visibility masks and evidence provenance remain metadata.

[Dataset schema](dataset.schema.json), [exporter](export.py), [trainer](train.py),
[loss masking](data.py), [loss/gradient checks](loss.py),
[observability audit](observability.py), [diagnostics](diagnostics.py) and
[execution protocol](ENGINEERING_PROTOCOL.md) are isolated in this directory.
Existing scientific pipelines were not edited.

All prompt/image/padding positions are ignored in the causal next-token loss.
Every assistant rationale, verdict, format and genuine EOS token is supervised;
trailing template text after EOS is ignored. The reported rationale span includes
its format text up to the verdict value; the verdict span includes the value and
EOS. A has no rationale target. Component metrics are means over each example's
component tokens, then over examples. Different target lengths make A/B total
token losses unsuitable as direct capability comparisons.

The full-run B autograd probe measured rationale loss 2.834320 / gradient norm
2.793778 and verdict loss 3.524432 / gradient norm 23.068357. A's verdict probe
measured loss 4.936475 / gradient norm 39.597372. Both B objectives reach LoRA
parameters; its weak verdict result is not an empty or detached verdict loss.

## Full-dataset fit

| Measurement | A: direct | B: generative |
|---|---:|---:|
| Fixed optimizer updates | 300 | 300 |
| Final generated training verdicts | 120/120 (100%) | 73/120 (60.83%) |
| Exact complete generated outputs | 120/120 | 73/120 |
| Valid output formats | 120/120 | 120/120 |
| Total CE, before → after | 4.406381 → 0.00118437 | 2.897894 → 0.00561148 |
| Rationale CE, before → after | Not applicable | 2.883727 → 0.0000451502 |
| Verdict CE, before → after | 4.259067 → 0.00229032 | 3.307893 → 0.114464 |
| Free rationale visibility-template agreement | Not applicable | 120/120 |
| Peak allocated GPU memory | 8.36 GiB | 8.46 GiB |

At update 150, both models predicted feasible for every sample and scored 72/120.
B's midpoint rationales matched visibility evidence on 106/120 and contradicted
it on 14; only 58 complete outputs matched exactly. By update 300, B corrected all
14 observation errors but still predicted 119 feasible and one infeasible. It
classifies 72/72 positives and only 1/48 negatives correctly. Its sole negative
prediction is `ra_095`. Hidden-milk verdict accuracy is 32/32 for A and 9/32 for B;
the other 88 rows score 88/88 and 64/88 respectively.

The reported B rationale agreement is narrow: three segmentation-supported
observation templates with shared limitations. It does not establish correct
causal feasibility explanations. Low average next-token loss can coexist with
poor greedy class selection, especially when easy common text and EOS contribute
many tokens. This one fixed run does not isolate the cause of B's underfitting or
establish comparative architecture capability. There was no corrective retuning.

In raw `RESULT.json`, `passed: true` for a full run means the fixed execution
completed with integrity checks. B separately records `exact_fit_criterion:
false`; it must not be described as passing the full-dataset fit.

## Observability and source audit

The [all-row audit](../results/verifier_training/20261011T001506Z_engineering_27e7fe7cf4e3/audit/samples.csv)
contains all 120 samples. The [audit analysis](OBSERVABILITY_ANALYSIS.md) documents
methods, per-camera evidence, nearest opposite-label pairs and limitations.

All 72 positives retain verified complete safe witnesses and all 48 negatives
retain conditional RA-SEP-1 certificates. None comes from an uncertainty score,
failed rollout or native-positive / synthetic-negative merge. All 120 descend
from `object_I_t2_e0`; slit and cage share one separator proof family.

The milk is hidden in both views in 32 samples: 24 negative and 8 positive. Barrier
pixels are cropped in all 120, and 2–48 red geometries per sample are not fully
inside either frame. Eight samples have positive entity fragments under 16 pixels,
including two with only 10 milk pixels. Segmentation-supported presence is weaker
than recognizable object identity or sufficient physical evidence.

All 120 decoded image pairs are distinct. There are no exact conflicting-label
pairs or collisions in the predeclared coarse-image screen. The strict near-pair
screen flags zero rows, which does not certify perceptual distinguishability or
physical identifiability. The closest opposite-label pair differs by just 0.04747
mean RGB channel levels on 0..255, with about 0.7803% of pixels changed. Both hide
the milk; a small visible enclosure cue does not reveal all certificate premises.

Task/constraint text is identical across all 120 and thus does not identify each
label. However, source-construction cues are confounded with outcomes: a
hidden-versus-present milk majority rule fits 88/120 (73.33%), and all 24 cases
without wrist-view barrier pixels are feasible irrelevant controls. Privileged
variant identity separates labels perfectly, although it is excluded from inputs.
Visual correlates of those variants remain possible shortcuts.

All 120 are retained unchanged **only for engineering fitting**, with explicit
observability strata. None is certified as scientifically adequate visual binary
supervision. This is an isolation decision, not a relabeling of valid evidence.

## Shortcut diagnostics

These are perturbations of the training pool, not held-out evaluation. Images are
shuffled as coherent external/wrist pairs using a fixed label-independent
derangement. Text-only removes the image blocks and explicitly says no images are
provided. The two horizon edits change only 300 to 600 or 0 commands.

| Condition | A correct | B correct | Interpretation |
|---|---:|---:|---|
| Original images | 120/120 | 73/120 | Same training samples |
| Shuffled images, recipient labels | 66/120 | 73/120 | Sensitivity to replacing the scene |
| Shuffled images, donor labels | 120/120 | 73/120 | Score against the supplied image state |
| Text-only | 72/120 | 72/120 | Identical prompts; majority ceiling 72/120 |
| 600-command horizon | 120/120 | 72/120 | Verified labels preserved |
| Zero-command horizon | 48/120 | 0/120 | All 120 diagnostic labels are infeasible |

A follows the shuffled donor images exactly. It predicts feasible throughout the
text-only condition and leaves all 120 predictions unchanged under both horizon
edits. In particular, none of its 72 original positive verdicts flips at zero
commands. This demonstrates image dependence within the collection and failure
of the tested constraint-change rule, not learned physical constraint reasoning.

Because every task/constraint prompt is identical, a shuffled input is exactly
the donor's original training input. Donor-label performance is therefore a replay
control, not a novel physical test. B's visibility descriptions follow the donor
evidence on 120/120, while its verdict score remains 73/120. These probes cannot
distinguish relevant geometry use from construction-specific visual cues.

B produces one shared no-image response for all 120 inputs:

```text
Rationale: The verdict is conditional on the stated reach-avoid rules. These rules do not establish exact clearance or a complete safe route. The verdict is based solely on the stated restrictions.
Verdict: feasible
```

The finite-grammar checker leaves this novel wording for manual review. Inspection
of the one shared input/output confirms that it asserts no object visibility or
hidden geometry. Its limitation statements are supported; its feasible conclusion
has no case-specific evidence. The review is bound to the exact input and output
hashes. This is not counted as a visual hallucination or a physical feasibility
justification. Text-only collapses all 120 cases to one input with mixed labels,
so 72/120 is the deterministic majority ceiling for both models.

B predicts feasible for all 120 cases under both horizon edits. Its one original
negative prediction flips to feasible even when the budget becomes zero. Neither
model flips any of the 72 verified positive examples to infeasible at zero
commands. These outcomes fail the supported constraint check. B's image-conditioned
visibility descriptions still match the unchanged evidence on 120/120 in each
horizon condition, illustrating that factual observation text does not establish
a valid verdict. All diagnostic formats are valid; no hidden-geometry claims are
generated in these completed outputs.

The 600-command labels are supported by the existing <=300-step safe witnesses
and horizon-independent conditional separator certificates. The zero-command
labels follow from every persisted step 0 having `contract_success=false` and no
action: with zero commands there are no transitions. This is a trivial logic
probe, not a second scientific physical mechanism. Removing a digital guard
would leave unsupported counterfactual labels and was not scored.

An actual final B output for certified-infeasible `ra_005` illustrates the
distinction between grounded observations and a correct verdict:

```text
Rationale: The milk is not visible in either view. The basket is visible in both views. Red barriers are visible in both views. These views alone do not establish exact clearance or a complete safe route. The verdict is conditional on the stated reach-avoid rules.
Verdict: feasible
```

Its observation statements agree with the available visibility evidence, but its
verdict is wrong. No hidden pose, exact clearance or safe route is invented by
this output; the missing causal justification remains a limitation. All 47 such
original B verdict errors are retained for case-level review, alongside format,
visibility and constraint violations in the diagnostic outputs.

## Stage 2 and go/no-go

The [collection specification](STAGE2_COLLECTION.md) and
[machine-readable plan](stage2_collection_spec.json) target at least 12 independent
original source scenes and 96 verified examples: 48 feasible and 48 certified
infeasible, including matched irrelevant-change controls for both labels. Each
source has both labels under two distinct mechanisms. Slit/cage count as one;
the proposed second mechanism is nontrivial finite-positive-horizon reachability
exclusion, whose sound verifier must be developed and independently validated
before collection. Zero-action cases do not satisfy that requirement.

Keep all descendants of a source together in the future 6/3/3 source split for
training/development/locked testing. Three test sources support only a small
exploratory check. Require student-visible label-critical premises, sentence-level
grounding, matched same-image constraint changes, nuisance-balanced controls and
a frozen ancestry manifest. The current source is not one of the 12 new sources.

**Go:** reuse the tested masking, frozen-front-end LoRA implementation and
evidence/provenance infrastructure for engineering work. **No-go:** scientific
training or generalization claims using the current 120, or treating B's low
rationale loss as successful feasibility learning. Resolve the data requirements
and B's verdict-underfitting issue on training/development material before a
scientific comparison. No new collection or scientific training is launched here.

## Provenance and delivered records

Training/audit/diagnostic source:
`27e7fe7cf4e3facb39b698d5d9141d25f3a2ce99`. Dataset SHA-256:
`1ab19294758acd2c59cda91077b15fd28938417685a9470d5e72c43a7d41c7b0`.

Main run: `20261011T001506Z_engineering_27e7fe7cf4e3`, under
`/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs/` on Quest,
with lightweight results mirrored under local `results/verifier_training/`.
Adapters remain in that Quest run's `A/adapter` and `B/adapter` directories.

| Quest job | Work | Result |
|---|---|---|
| 9440314 | Audit, 20 unit tests, 240 encodings | Completed / 0:0; 2m20s |
| 9440568 | A, 300 updates | Completed / 0:0; 17m11s |
| 9440569 | B, 300 updates | Completed / 0:0; 33m16s |
| 9441382 | A shortcut probes | Completed / 0:0; 5m04s |
| 9441383 | B shortcut probes | Completed / 0:0; 32m01s |
| 9441384 | Curves and aggregate report | Completed / 0:0; 16s |

All GPU allocations use p33100/gengpu; CPU work uses p33100/short. Pending wall
requests were reduced within the proposed resource caps and recorded separately;
optimization settings stayed fixed. A used an A100-SXM4-80GB and B an
A100-PCIE-40GB, so wall times are not a controlled throughput comparison. Jobs use
immutable source archives, offline staged assets and unique output roots. Original
scientific pipelines, historical results and the earlier smokes remain intact.

The four GPU allocations total 87m32s. The downloaded campaign files passed their
SHA-256 delivery manifest (111 files), and both plots were visually checked.

Delivered [loss curves](../results/verifier_training/20261011T001506Z_engineering_27e7fe7cf4e3/training_curves.png),
[training accuracy curve](../results/verifier_training/20261011T001506Z_engineering_27e7fe7cf4e3/training_accuracy.png),
[all generated outputs](../results/verifier_training/20261011T001506Z_engineering_27e7fe7cf4e3/outputs.csv),
[aggregate metrics](../results/verifier_training/20261011T001506Z_engineering_27e7fe7cf4e3/SUMMARY.json),
and [execution configurations](../results/verifier_training/20261011T001506Z_engineering_27e7fe7cf4e3/execution_B.json)
retain the full measurements. A separate CPU case-review run adds a complete
verdict-error inventory and the [hash-bound no-image adjudication](manual_reviews/full120_no_image.json)
without changing the original automatic grades. Its four dedicated tests protect
the distinction between visibility agreement, wrong verdicts and unparsed prose.
The original `INVALID_OR_INCONSISTENT.json` is a grammar-review queue, not a
complete or manually adjudicated error count; use the supplementary case review
for those decisions.
