# Stage 2: minimal source-diverse collection specification

This is a design only. No new simulator state, rollout, benchmark task, controller,
or training job is created by this specification. Keep Qwen2.5-VL-3B-Instruct and
the established language LoRA architecture, frozen vision encoder/projector, and
identical A/B student inputs.

## Collection unit and minimum matrix

Collect at least **12 independent original source scenes**. A source is an
independently chosen scene composition and initial physical state with its own
target, goal, clutter arrangement and task instance, recorded before intervention.
Translations, rotations, camera changes, obstacle variants, alternate seeds applied
to one stored scene, and reconstructed copies are descendants, not new sources.
The existing `object_I_t2_e0` family is one engineering source and supplies none of
the 12 new-source requirement.

Select the sources from at least three existing task/scene families, with four
independent sources per family where available. Record asset sharing and any
common initialization ancestry; sources with a common parent must be merged into
one split group. If the available native scenes cannot provide 12 independent
groups, the minimum has not been met. Do not manufacture independence by renaming
cases or selecting new views of the same physical scene.

The minimum complete matrix is **96 labeled examples**:

| Per original source | Mechanism 1 | Mechanism 2 | Total |
|---|---:|---:|---:|
| Verified feasible base member | 1 | 1 | 2 |
| Matched certified infeasible member | 1 | 1 | 2 |
| Irrelevant-change control for feasible member | 1 | 1 | 2 |
| Irrelevant-change control for infeasible member | 1 | 1 | 2 |
| All descendants per source | 4 | 4 | 8 |

Across 12 sources: 48 feasible and 48 certified infeasible; 48 controls. A missing
witness or certificate leaves a member UNKNOWN and does not count toward 96.
Collect replacements only under a separately authorized collection run. More
sources should take priority over many additional descendants of these 12.

## Two distinct constraint mechanisms

**Mechanism 1: static spatial separation under an explicit swept-motion guard.**
Use the already established conditional separator contract, with immutable
obstacles, verified material containment, complete face coverage, initial target
inside and fixed goal outside. A feasible matched member must admit a fully
verified safe trajectory, for example after changing a relevant opening or a
specified keep-out margin. Recompute every certificate from actual collision
geometry and restore state. Any altered guard margin requires a versioned,
validated contract; the existing checker is not assumed to certify arbitrary
new margins. Slit and cage are variants of this one mechanism.
Do not describe conditional digital infeasibility as unrestricted physical
impossibility.

**Mechanism 2: nontrivial finite-horizon reachability exclusion.**
Use a positive action/time budget and an independently validated conservative
outer bound on all legal reachable target states. An infeasible certificate must
prove that this reachable set is disjoint from the fixed goal for every allowed
action and contact mode up to that budget. A larger matched budget must have an
independently verified safe witness, or the feasible member remains UNKNOWN.
The label-changing pair keeps images and scene fixed and changes only the stated
budget. This tests a temporal/actuation constraint, distinct from spatial
enclosure. Do not use a policy's failure, a long found trajectory, action clipping
alone, or empirical maximum speeds as the proof. Zero actions is only the present
engineering sanity check and does not qualify as this scientific mechanism.

This second certificate verifier does **not yet exist in this pilot**. Before
collecting its matrix, specify the dynamics, integration semantics, bounded
initial-state uncertainty, control limits and all allowed contact modes; implement
and independently review a sound reachable-set outer bound. If those assumptions
cannot be certified for the simulator, Stage 2 is blocked on mechanism design.
Do not silently replace it with another enclosure constructor or fabricate
negative labels. Building that verifier or changing task contracts requires the
subsequent collection authorization; this document does not execute those tasks.

Where physically supported, include matched constraint changes with the same
images: a keep-out margin pair for mechanism 1 or horizon pair for mechanism 2.
Both members need valid evidence under their own contract. Counterbalance budget
and margin values across sources so no fixed number or phrase is always one label.
The same constraint text should appear with both outcomes across different scenes.

## Observability and rationale acceptance

Use exactly two RGB views plus task and constraints. Choose cameras and framing
without using the label; apply the same rule across both members of a matched
pair. Confirm at the actual processor resolution that the target, goal and every
label-critical spatial relation can be assessed. Avoid the current pattern in
which a hidden milk carton and hidden opening distinguish contradictory labels.
Segmentation-derived positive pixel counts alone do not certify recognizable
object identity, metric clearance, or a complete obstruction.

For each example record an evidence-to-input map: each certificate premise or
witness claim needed by the intended explanation; the visible RGB region or
explicit student constraint supporting it; its resolution/occlusion limitation;
and the independent label evidence. Physical constants needed to interpret a
metric rule must be stated in the common task contract or observable from a
calibrated view. Privileged ground-truth measurements and proof outcomes remain
metadata. If label-critical information cannot be supplied through the permitted
input contract, exclude the example from binary scientific supervision and keep
it in a separate partial-observability diagnostic set.

Two reviewers independently inspect the student-only inputs, then compare their
claims with the persisted evidence. They must distinguish a useful visual cue
from a sound physical conclusion. Record disagreements and adjudication. At
least one reviewer should not have built the scene or its certificate. Reject
claims of complete visual observability based only on matching the final label.

Rationales should connect a relevant observed relation and stated constraint to
the verdict, while distinguishing support from proof. Every factual clause needs
a provenance entry. No hidden target pose, unseen opening width, unobserved safe
route, exact dynamic bound, or unrestricted impossibility assertion is allowed.
Store the evidence map, witnesses/certificates, source IDs, paths, hashes and
reviewer notes in metadata, never in the student prompt. Use the same style and
evidence availability for both labels; avoid label-specific boilerplate that
substitutes for case-specific explanation.

## Irrelevant controls and nuisance balancing

For each labeled member, produce one matched irrelevant-change descendant only
after authorization. Use a texture/lighting or camera nuisance variation within
an observability-preserving envelope, or a remote object change independently
verified not to alter any collision, goal, dynamics or certificate premise.
Recheck restore state, image alignment and label evidence. A visually irrelevant
change is not assumed physically irrelevant. Controls must preserve the label
and causal explanation, while allowing the observation description to change.

Counterbalance obstacle color/shape, target visibility, camera viewpoint, scene
family and wording across labels. Audit exact and approximate duplicate inputs,
including processor tensors; check every near-duplicate with conflicting labels.
Quantitative image-distance screens locate cases for review but do not establish
perceptual equivalence. Explicitly inspect source/constructor/filename leakage,
task-text and constraint-value associations, and rationale-template associations.

## Provenance, splits and evaluation

Each record keeps the established `student`, `target`, `metadata` schema, adding
`source_id`, `parent_source_id`, `intervention_id`, `matched_pair_id`,
`irrelevant_control_of`, `mechanism_id`, `contract_version`, proof/witness verifier
version, observability review and per-sentence grounding. Preserve native code,
upstream commit, model/rendering configuration, initial state, action trace,
all safety substeps, proof assumptions, hashes, seed and Slurm allocation.

Before scientific optimization, freeze an ancestry graph and group split. For
this minimum pilot, assign **6 sources to training, 3 to development and 3 to
locked testing**, stratifying task/scene family where possible. All descendants,
views, renderings, mechanisms, matched pairs and controls of a source remain in
its one split. Hash the split manifest and dataset. Keep test scenes sealed until
configuration, rationale rubric and decision rules are frozen. Do not transfer
the present one-source memorizing adapters into the new comparison; start both
arms from the same pinned base model and matched fixed optimization budget.

Measure source-macro verdict accuracy/balanced accuracy, per-mechanism performance,
matched constraint flip consistency, irrelevant-change invariance, free-generated
rationale factuality, unsupported claims, omissions and rationale/verdict
consistency. Invalid outputs count as errors. Keep no-image, coherent image-swap
and controlled constraint probes as diagnostic analyses. Review blinded A/B
rationales against evidence with the same rubric, not a verdict-aware evaluator
that can reward agreement alone. Aggregate uncertainty by original source group;
descendant counts do not create independent observations.

The current exporter and execution gate deliberately pin the original 120-row
hash and single source. A later collection implementation will need a versioned,
source-independent exporter, typed witness/certificate validators for both
mechanisms, the source-ancestry split manifest, and richer sentence-level targets.
Those are data/protocol changes; retain the existing model, language LoRA,
assistant-token masking and frozen vision/projector. Validate the new targets on
the training/development sources before opening the locked test sources. No such
new exporter, certificate mechanism or scientific run is launched here.

Three locked test sources provide a small exploratory check, not strong evidence
of broad robustness or mechanism-independent transfer. If this pilot is promising,
expand independent source count and design a separately powered confirmatory
evaluation. Claims must match the observed sources, tasks and certified contracts.

## Decision gates

Proceed to scientific training only after 12 independent sources, both separately
validated mechanisms, the complete balanced evidence matrix, student-visible
premises, adjudicated grounded targets, and a frozen ancestry-safe split pass.
The current 120 cannot satisfy these gates. Infrastructure success is permission
to use the training code, not evidence that the data are scientifically adequate.
The fixed 120-sample run also leaves B's verdict optimization unresolved. Diagnose
that issue on training/development material under a predeclared protocol before
a scientific comparison; low rationale loss is not evidence of feasibility learning.
No optimization search is conducted in the current task.

The next collection step requires authorization of this collection/verification design;
no collection is launched by the current task.
