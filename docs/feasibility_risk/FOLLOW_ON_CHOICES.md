# Minimal follow-on choices after FR-1 / FR-1B

These are proposals, not an execution queue. No experiment below is authorized
by this document. The completed FR-1B caps remain the current execution limit.

## 1. Observation ablation in the existing task (smallest useful next check)

Keep the eight completed physical states and their independently established
labels: six FR-1 states in initial states 10/11 and the two FR-1B decoys in 10.
Add exactly one render-only condition: set alpha to 0.25 for **all** red enclosure
panels, with their original RGB unchanged. Freeze this setting before rendering;
do not tune alpha or choose a camera after seeing features. Opaque inputs are the
existing comparison. Preserve cameras, lights, prompt, state, collision masks,
geometries, action contract, and goals. Do not advance simulation or roll out a
policy. This yields eight additional observations, not eight new physical states.

Before extraction, check every non-render model array and complete physics state
against its opaque counterpart; verify no simulation step occurred. Confirm that
the actual preprocessed RGB exposes the milk in sealed states. A native integer
segmentation mask alone is inadequate for alpha blending: measure target pixel
influence using a diagnostic render with only the target's appearance changed,
without feeding that diagnostic image to the model. Require nonzero influence
inside the projected target region, keep both raw renders, and report the influence
map. A pass does not establish full boundary observability or human interpretability.

Extract the same three frozen feature summaries and RGB/red-pixel baselines.
Extract opaque and transparent observations together in one frozen process;
repeat one opaque anchor at the end. Cross-job discrepancies in FR-1B make mixing
old and new feature values inappropriate for the ablation. Use the unchanged
extraction audit: 17 inputs total, 34 feature forward passes and 102 local
unexecuted action inferences, zero environment actions, zero paid API calls. Suggested cap: one CPU job, 1 core / 8 GiB / 5 min;
one A100 job, 4 cores / 32 GiB / 10 min. Expected GPU use roughly 4–8 min from the
completed extraction timings; cap failure ends the attempt without scientific tuning.

Predetermine only descriptive contrasts: within-state opaque/transparent distances;
relevant and decoy closure displacement; image versus prefix displacement; visible
target influence. Do not select a probe layer, pooling scheme, threshold, or model
from the eight states. Report the two original groups separately; no population
accuracy interval, bootstrap over images, or held-out AUROC.

**Can test:** whether the same fixed representation changes when one label-irrelevant
rendering choice restores target evidence, and whether a closure contrast behaves
differently when the enclosure is irrelevant. It can remove zero-target-pixels as
a deterministic feasibility shortcut *within the transparent condition* if all
targets become observable.

**Cannot test:** ordinary-RGB native feasibility reasoning, causal use by control,
general feasibility, or unseen-layout decoding. Transparent surfaces are out of
distribution and may expose easier lid/closure cues. This is not a full visibility
by feasibility factorial: it supplies visible infeasible states, but does not
automatically supply hidden feasible states. A failure would be a limitation of
this fixed ablation, not proof of representation incapability or unidentifiability.

## 2. Stop the current representation experiment here

Retain the actual safe/unsafe-policy distinction, conditional exclusion labels,
parked-lid and irrelevant-closed-box controls. Report that held-out decoding was
not tested. This is justified by only two closely related physical groups,
opaque target-visibility confounding, camera clipping, and one policy-risk class.
It is not justified to claim that no feature direction exists. Cost: zero compute.

## What would permit a later small supervised readout

A transparent ablation alone does not justify fitting a classifier on these eight
states. A later design must preselect spatially distinct layouts of the same milk
task, keep all arm/opacity variants of each layout in one split, and acquire labels
without treating failed witnesses as infeasibility. The old provisional 24 official
jitter states are not an adequate substitute for real layout variation. Freeze
layout generation, splits, readout, scaling, regularization and metric definitions
before collecting a full matrix; retain construction failures/unknowns in coverage.
At least one closure-irrelevance control must occur in each split, and target
visibility cannot perfectly identify the label in the evaluated condition.

The readout would be an added estimator of an **external contract label**. Its
success would not imply native introspection. The shared natural-language prompt
mentions no-contact safety, but does not communicate RE-1's integration-segment rule
or fixed-goal conjunction; supervised external labels do not repair that mismatch
into a native semantic-understanding claim. Collision prediction is a separate
endpoint: its full-horizon-safe class is absent here, so no collision AUROC can be
estimated. Future feasibility-only decoding need not wait for a balanced risk
dataset if the two goals and claims are explicitly separated in a new freeze.
