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

The plan uses the same claim vocabulary, C0–C12, as `CLAIMS.md`.

## MUST

- External validity: run held-out wall and held-out task checks before expanding
  hazard breadth.
- Compare the probe guard with trivial trigger baselines; separate online closed
  loop results from frozen-capture threshold sweeps.
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
