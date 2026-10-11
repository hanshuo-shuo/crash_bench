# All 120 observability and label audit

Quest CPU job 9440314 completed successfully in 2m20s on frozen source
`27e7fe7cf4e3facb39b698d5d9141d25f3a2ce99`. All 20 unit tests and 240 native
Qwen A/B encoding checks passed. The checksummed dataset is unchanged:
`1ab19294758acd2c59cda91077b15fd28938417685a9470d5e72c43a7d41c7b0`.

The row-level deliverable is
`results/verifier_training/20261011T001506Z_engineering_27e7fe7cf4e3/audit/samples.csv`
(also `samples.json`). It includes every sample's label basis, visibility counts,
cropping, missing physical evidence, nearest opposite-label input, rationale
support and disposition. `AUDIT.json` records methods, counts and associations;
12 `near_pair_*.png` sheets show native-resolution pairs and amplified differences.

## Labels versus student-identifiable verdicts

All 72 feasible labels still have verified complete safe witnesses. All 48 negative
labels still have verified conditional RA-SEP-1 separator certificates. Original
evidence hashes, persisted certificate/witness checks, observation evidence,
rationale provenance and initial-state goal status were rechecked. The existing
invalid later policy endpoint never supplies a negative label. No native
SafeLIBERO positive pool, failed rollout or uncertainty score is used.

This establishes the label under its recorded contract. It does **not** establish
that the binary verdict follows from the student's two 224x224 RGB images and
common text. None of the 120 is certified as fully image-identifiable scientific
supervision. Images omit metric material/clearance information, complete enclosure
surfaces or interiors, and dynamics. Feasible witnesses are privileged evidence,
not visible complete routes. Barrier pixels touch an image boundary in all 120;
between 2 and 48 individual red geometries per sample are not fully inside either
camera frame. Corner projection coverage itself does not certify absence of
occlusion.

| Observability group | Feasible | Infeasible | Total |
|---|---:|---:|---:|
| Milk has no segmented pixels in either view | 8 | 24 | 32 |
| Milk has some segmented pixels | 64 | 24 | 88 |
| All samples | 72 | 48 | 120 |

All 32 hidden cases are slit variants: all 24 narrow-gap negatives and eight .180
positives from L00–L07. Their images cannot directly locate the milk or verify its
relationship to hidden faces. The other 88 still do not supply a complete physical
proof. Two feasible .180 cases, ra_087/ra_097, expose only 10 milk pixels, so even
the word "visible" is weaker than recognizable identity.

Disposition: retain the original 120 labels and targets for the explicitly
authorized engineering fit, tag all as scientifically unresolved, and report the
hidden 32 separately. Do not fix missing observability by inserting certificate
measurements, hidden poses or source ancestry into student prompts. Do not
silently relabel valid certificates or drop the difficult cases from the fit.

## Identical and visually similar inputs

All 120 decoded native RGB pairs are distinct, and no exact pair has conflicting
labels. No contradictory group was found by the fixed 28x28 BOX/round-channel-by-16
coarse hash either. The predeclared strict near-conflict screen (MAE<=1/255 and
at most 0.1% pixels with any channel difference>8/255) flags zero rows. These
negative screen results are not proof that all relevant physical distinctions
are recognizable.

The closest opposite-label pair is ra_006 (`L00_slit_.028`, infeasible) versus
ra_007 (`L00_slit_.180`, feasible); the analogous L01 pair has the same distance.
Across both images, mean absolute channel difference is only **0.04747 on 0..255**.
About 0.7803% of pixels change at all, and 0.6328% change by more than 8 in at least
one channel. The difference sheet locates the change in a small part of the red
enclosure's upper surface; the wrist view is unchanged. Both images hide the
milk. There is a detectable render cue, but the images do not establish the
target's hidden position or a complete separator/route argument. Accordingly,
do not call the pair exactly identical or proven perceptually indistinguishable;
do not infer scientific observability from its differing pixels.

All six original contact sheets covering 120 pairs were reviewed, together with
the closest pair's native-resolution difference sheet. This is a visual audit
by the coding assistant, not an independent blinded human annotation study.

## Text, nuisance cues and rationale targets

The task and full constraints form **one identical text group** containing both
labels. There is no per-row task/constraint template that reveals the verdict.
Sample IDs, source identity, constructor, gap values, filepaths, masks, proof
records and provenance stay out of student inputs. Both A/B inference prompts,
image tensors, attention masks and image grids are identical for each row.

Nevertheless, construction and observability are correlated with labels:

* A constant feasible verdict fits 72/120 (60%). That is the best possible
  deterministic label accuracy for this identical text-only pool.
* A rule using only hidden-versus-present milk, with the majority label in each
  group, fits 88/120 (73.33%). This is an in-sample descriptive association.
* All 24 examples with no wrist-view red-barrier pixels are feasible irrelevant
  controls. That cue is not balanced across labels.
* Privileged variant identity perfectly separates the labels: .020/.028 are
  negative; .180/open/irrelevant are positive. Variant identity is not supplied,
  but associated visual morphology can act as a shortcut. This audit does not
  prove that a trained model uses any particular one of those cues.

There are only **three rationale templates** across 120 examples. Each contains
three segmentation-supported entity-presence statements and the same two
sentences acknowledging missing exact clearance/route evidence and conditional
reach-avoid rules. No target asserts hidden target coordinates, gap measurements,
a safe route, complete face coverage or a physical impossibility proof.

All 120 target rationales match their persisted sentence-level grounding. However,
eight rows contain positive entity fragments under 16 pixels: six have 11–12 red
barrier pixels in the wrist view, and the two noted above have 10 milk pixels.
The presence claims are supported by aligned segmentation, while recognizability
from RGB remains weaker and requires independent review for scientific use.
Thus template agreement measures grounded observation/limitation text generation;
it does not measure generation of a case-specific causal feasibility explanation.

## Supported constraint probes

The later shortcut evaluation uses original images, label-independent shuffled
coherent pairs, and no-image inputs. For image swaps, both recipient-label and
donor-label scores are necessary because the task/constraints are identical.

Two matched text edits have supported diagnostic labels. Raising 300 commands
to 600 preserves every positive witness and every horizon-independent separator
certificate. Setting the budget to 0 yields infeasible for all 120: every recorded
initial state has no action and contract_success=false, and no commands permit
no state transitions. The latter is a trivial logical constraint check, not a
second physical mechanism or a scientific test set. Guard removal has no supported
feasible counterfactual labels and is not scored. No simulation is performed.
