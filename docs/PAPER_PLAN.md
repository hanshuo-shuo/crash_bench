# Paper plan

## Current line

The contribution is a controlled causal diagnosis of a collision mode, not a
claim that all VLAs or all hazards are unsafe. Hold obstacle appearance fixed and
vary whether the obstacle occupies the robot's swept corridor:

1. **Diagnose:** visible on-path walls produce collisions.
2. **Localize:** the on-path / clear-off-path and transition controls implicate
   corridor intrusion, rather than obstacle novelty alone.
3. **Explain:** collision imminence is **decoded but not used** — it is
   represented but not read out into safe action.
4. **Exploit:** a wall-trained probe can trigger structured retreat recovery in
   the bounded OpenVLA-base/on-path-wall setting.

Use “representation–behavior dissociation”, “decoded but not used”, and
“represented but not read out into safe action” in formal text. “Knows but does
not act” is at most an informal introduction hook.

The plan uses the same claim vocabulary, C0–C13, as `CLAIMS.md`.

## MUST

- Treat the completed Hazard Validity and Environment Generalization Experiment
  (E12; internal P0) as negative/indeterminate: it had no
  calibration or held-out positive T-5 frames and no held-out vanilla crashes.
  Do not tune its held-out data, horizon, seeds, repeats, or thresholds.
- Keep the completed online probe-guard/trivial-trigger comparison separate from
  frozen-capture threshold analysis; E12 cannot establish crash reduction because
  vanilla was already 0/50 crashes.
- Add probe-confound controls and close the pi0 probe conclusion as partial,
  positive, or negative according to data.
- Report trial and scenario counts, repeats, uncertainty, scenario fingerprints,
  force-threshold sensitivity where raw traces permit it, and exact result
  provenance.
- Preserve the tall-wall treatment; put the lowered d62 detour in a distinct
  scenario root with parent/fingerprint metadata.
- Release a manifest, claims ledger, audit command, and zero-GPU tests.

## SHOULD

- Add an OFT frozen-capture guard sweep and a held-out-wall online guard test.
- Quantify nondeterminism and use task/scenario-level analysis where appropriate.
- Add a direct input/motor confound comparison and within-scene outcomes where
  the frozen capture supports it.

## CUT / do not present as current work

- A seven-category, 50-scenario, 150-trial benchmark headline.
- An old selected-band cross-category headline from glass f50–f70.
- Universal collision avoidance, general task-completing recovery, or all-model
  “knowledge” claims.
- Threshold=1.0 as an online-validated guard setting; it is only an offline
  frozen-capture sweep midpoint in the committed evidence.

## Evidence discipline

- The pilot/corridor wall mean is about 250 N; 321.7 N is specifically the
  online intervention baseline mean.
- The online intervention is 15/15 -> 0/15, 321.7 N -> 0 N, with 0/22 benign
  guard fires at threshold about -0.422.
- The `[-0.7, 2.7]` operating window is offline. Report the complete glass
  f30–f70 dose response separately from the complete wall sweep.
- The steering diagnostic is a readout-norm ratio: under 10% of the full
  probe-direction readout norm remains on the action-token slice. It is not an
  angle measurement.
- E12 is not positive external-validity evidence for C7 or C13. Its tracked
  summary records `dissociation_supported=false`; all guard crash rates are zero,
  while vanilla retains the highest task success (88%).
