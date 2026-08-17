# Paper plan

## Working title and one-sentence contribution

**Decoded but Not Routed: Causal Diagnosis and Closed-Loop Repair of VLA
Collision Failures**

VLA representations can make an imminent manipulation collision linearly
readable while the action continues into impact; a small explicit
**risk-readout → controller** interface repairs this representation–action
misalignment in closed loop.

The paper uses the C0–C13 vocabulary in [CLAIMS.md](CLAIMS.md). It does not use
E14/E15 execution order as its narrative structure.

## Four-act main text

### Act I — Causal localization in the swept corridor

Start with matched geometry, not a benchmark catalog.

- Same visible wall, on the executed path: 15/15 crash.
- Same wall family, clear of the path: 0/33 crash.
- Transition clearances: graded response.
- Same matched contrast across OpenVLA, OFT, and pi0: 5/5 versus 0/10 per
  action-head family.
- Movable glass: 30/50 on-path versus 0/50 matched off-path, with an f30→f70
  dose response.

Main claim: collision follows membership in the policy's executed swept
corridor, not obstacle novelty alone.

### Act II — The risk is decoded but not routed

Join the representation and behavior evidence in the same section.

- OpenVLA T-5 AUC 0.998; OFT 0.903.
- Glass-specific T-5 AUC 0.944 as a hazard-specific extension.
- Hidden-only macro AUPRC 0.716 versus 0.442 for the strongest observable
  baseline under strict held-out-scenario analysis.
- No final-window EEF retreat in 25/25 wall episodes; toward-wall command
  projection increases in 22/25.

Main claim: the signal is available to a simple readout but is not expressed as
sustained braking. Use “representation–action gap,” not a generic “features
predict failures” headline.

### Act III — Close the loop with an explicit interface

Introduce the interface as the causal repair:

```text
frozen VLA state → risk readout → latched handoff → structured controller
```

- `Probe → RetreatHold`: 15/15 → 0/15 crash, 321.7 N → 0 N, 0/22 benign fires.
- Final-readout steering: 100% crash for all tested alphas.

The contrast is mechanistic: a detector direction need not be a controller
direction. Explicit routing succeeds where treating the probe direction as a
native action knob fails.

### Act IV — Safety is not task completion

End with a frontier, not a new main pipeline.

- Hazard-specific language reduces collision but produces 0/15 treatment task
  successes for both wall and glass.
- `RetreatHold` is a safe-abort controller.
- Exact-state glass counterfactuals show that a task-completing continuation is
  physically available: Oracle timing/actions achieve 6/6 on two development
  source states.

State the boundary explicitly: the current closed-loop learned component fixes
collision behavior, while task-completing learned routing remains open.

## Main figures and tables

1. **Geometry causal panel:** top-down swept corridor, on-path/off-path matched
   scenes, wall clearance curve, and compact glass dose response.
2. **Decoded/not-routed panel:** probe performance beside final-window
   wall-directed behavior.
3. **Routing intervention panel:** Base versus probe-gated `RetreatHold`, including
   force and benign-fire outcomes.
4. **Safety–utility panel:** one whole-episode subpanel for Base versus the exact
   hazard-specific prompt, and one matched exact-state/online subpanel for Base,
   binary risk, fixed options, learned router, and Oracle. Do not mix the old
   prompt cohort's 15 episodes with the router cohort's 106 decision anchors.
5. **One glass filmstrip:** exact same state, Base catastrophe versus Oracle safe
   task completion.

Keep the steering null as one compact ablation. Do not give every historical
experiment its own main-text figure.

## Highest-value new experiments

### 1. Five-fold held-out online wall guard

For each of five wall scenarios, fit and threshold on the other four, then deploy
online only on the unseen wall. Evaluate the held-out on-path wall together with
matched clear off-path and no-wall rollouts. Report folds/scenarios as the
independent units and keep threshold selection inside each training fold.

This upgrades C7 from “operational on the fitted geometry set” to a held-out
risk-routing interface test.

### 2. Glass learned detector + structured task-completing controller

Recover or recapture glass hidden states, serialize a deployable glass-specific
probe, and use it to trigger `DetourComplete`. First run the two development pairs;
only after a positive development signal freeze a small fresh certified cohort.
The wall probe is not an acceptable substitute because wall→glass AUC is 0.359.

Name the method by what is learned: **learned detector + structured controller**.
Do not call the structured actions “learned recovery.”

### 3. OFT-specific online guard replication

Use OFT's own representation and probe to trigger the same `RetreatHold`
interface. This tests whether the routing abstraction, rather than one action
head, carries across model families.

## Before running anything

- Recover the ignored wall/OFT `hidden.npz` and `meta.json` files, or recapture
  them; the committed probe checkpoints alone are insufficient for five-fold
  refitting.
- Recover/recapture glass activations and add probe serialization/runtime loading;
  the repository currently has only a glass probe summary.
- Freeze folds, episode-level threshold calibration, treatment/control sets,
  repeat counts, and source-state independence before observing new outcomes.
- Register new result paths, scenario fingerprints, checkpoint revisions, and
  commits in `results/manifest.json`.

## Appendix

- Complete prompt matrix and instructions.
- E12/P0 negative/indeterminate generalization study.
- π0 probe taps and full probe diagnostics.
- Threshold sweep and complete steering diagnostic.
- Broad glass Pilot B no-go, certification funnel, exact restore/hash checks,
  strict authoring details, Slurm provenance, and Oracle upper-bound traces.
- Low-wall task-completion existence demo and other negative hazards.

See [appendix/README.md](appendix/README.md).

## Cut from the main narrative

- E14/E15 A/B/C/D/E/F chronology.
- Broad B's 109-attempt authoring and repair timeline.
- Full Slurm, hash, restore, and schema implementation details.
- Seven-category benchmark language and selected-band glass averages.
- E12 as positive generalization evidence.
- Large prompt matrices, E12 internals, full π0 probe details, and repeated
  same-seed tables.
- Claims of universal safety, general recovery, or learned task completion.

Pilots D/F are not run targets at the frozen checkpoint. If action learning is
revisited at all, the minimum diagnostic is a tiny Oracle-timing E smoke: one
train pair plus two development pairs, one seed each. A failed train pair ends
the action-head route; train-only success diagnoses data diversity; only a
development signal justifies rebuilding detector calibration.

## Evidence discipline

- 321.7 N is the intervention baseline mean, not the mean of every wall study.
- Threshold 1.0 is not an online-validated wall operating point.
- The glass 6/6 result has two independent development source states and Oracle
  timing/actions.
- The glass checkpoint's learned gate has timely-trigger rate 0.0 at threshold
  1.0; its validation gripper-sign accuracy is 0.846, below the 0.95 gate.
- Prompt-induced collision reduction without task success is conservative
  stopping, not avoidance or recovery.
- Tall-wall and low-wall d62 scenarios remain distinct in all claims and figures.
