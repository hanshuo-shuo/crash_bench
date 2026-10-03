# FR-1C prospective fixed-alpha observation ablation

Authorized continuation on 2026-10-03. This supersedes the proposal-only status of
follow-on choice 1, for this check only. No broader matrix or redesign is queued.
The original FR-1/FR-1B report and Library versions remain intact.

The complete freeze is `experiments/feasibility_risk/opacity_protocol.json`.
Eight inherited physical states, alpha1.0 and alpha0.25, seventeen total initial
inputs including an end anchor. No new label, environment action, API call, policy
rollout, fitting, layer selection or normalization. Same three frozen feature
pools and initial action checks/proposals. One CPU job (1core/8GiB/5min), one A100
job (4cores/32GiB/10min), one attempt per stage. Both use immutable published code.

The older constructor's 20 settling commands are bypassed. The already validated
complete snapshots are restored, and environment/integration step methods are
blocked before construction. The CPU gate verifies inherited label provenance,
runtime panel geometry, full snapshot equality, and alpha-only model-array changes.
The GPU stage additionally requires every opaque policy input to reproduce its
saved input exactly, all read-only extraction checks, and an exact end anchor.

For each condition and both cameras, save native and preprocessed RGB plus fixed
cyan and magenta target-appearance diagnostic renders. Override target material
selection only for those diagnostics, preserving target alpha and physical
attributes; restore model arrays and repeat the normal image exactly. Count >=1
uint8-level differences inside the target's projected bounds plus a fixed2pixel
margin. This detects target influence through alpha blending; segmentation alone
would not. Diagnostic marked images are never policy inputs. This establishes
appearance influence, not human/object-recognition sufficiency or full boundary
observability. Shadows/reflections outside the projected region are not counted.

All transparent states must have nonzero influence in at least one original
camera for the visibility gate to pass. Zero counts remain in the result. A failed
gate does not trigger alpha tuning, replacement states, or another submission.
The planned feature set is still collected unless an integrity failure stops it.
Contrasts are fixed before execution: within-state opacity, relevant closure and
parked lid, decoy closure, closure-displacement alignment, and image/prefix/RGB
baseline changes. Groups10 and11 remain separate descriptive units; no image-level
confidence interval, held-out decoding metric, or population claim is appropriate.

If the gate passes, zero visible target pixels no longer identifies infeasibility
within the transparent condition. The inputs remain out of distribution and may
make closure cues easier. There are still only two closely related groups and no
safe-horizon risk class. Recommend a separately frozen spatial-diversity study
before any readout accuracy claim; stop this experimental round after FR-1C.
