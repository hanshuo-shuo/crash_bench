# Paper plan

## Current line

The proposed contribution is a controlled causal diagnosis followed by a
fail-closed learned-recovery test, not a claim that all VLAs or hazards are
unsafe. E15 asks whether decoded imminent risk can be converted into a latched,
task-completing response on certified recoverable glass scenes:

1. **Diagnose:** visible on-path walls produce collisions.
2. **Localize:** the on-path / clear-off-path and transition controls implicate
   corridor intrusion, rather than obstacle novelty alone.
3. **Explain:** collision imminence is **decoded but not used** — it is
   represented but not read out into safe action.
4. **Exploit:** train a glass-specific risk/recovery head and test safe original-
   task completion from sealed, accepted source-state cohorts.

E15/v2 primary admission is exactly three branches: exact-H Base catastrophe,
independently recaptured matched-state Oracle safe task success, and matched
off-path safe task success. Careful prompts do not select the cohort. Blocked
safe-abort data use a separate appendix manifest. Source-to-task is the primary
evaluation; exact-anchor is a component diagnostic. The independent unit is the
source state, not a placement, repeat, or frame.

E14 remains historical motivation: its older Base + fixed-careful + Oracle gate
found three placements and no validation placement. It is not migrated into v2
and is not recovery-training or generalization evidence. The old wall line and
E13 are likewise motivation/appendix with their original scope limits.

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
- Respect the broad E15 Pilot B no-go: no tested H supported a source-diverse
  recoverable population. The later scoped route may be reported only as a
  development certification/yield screen plus an exact-anchor Oracle upper
  bound on two certified source states. Do not describe its heldout-labelled
  pairs as final held-out data or its Oracle condition as learned recovery.
- Require checkpoint/Base/dataset/protocol identities, the complete baseline
  matrix, primary task/catastrophe outcomes, false intervention, task
  preservation, and source-cluster analysis before any learned claim.

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
- E13 shows that visually grounded hazard language changes behavior, especially
  for glass, but both hazard-specific treatment cells have 0/15 task successes.
  Describe this as conservative stopping, not task-completing avoidance, and use
  five geometry scenarios per regime as the meaningful cluster count.
- E14 acceptance smoke is an existence result: 3 accepted admissions from 109
  rollout attempts (train=2, validation=0, heldout=1). Its dominant rejection
  was `no_base_crash` (92 attempts), so it is evidence for a clean gate and a
  remaining placement-yield problem, not a denominator for a safety headline.
- E15 Pilot B separates placement yield from recoverable-population viability.
  The 20-scene frontier achieved 50%--75% live Base-catastrophe yield across H,
  but exact replay reached at most 58.3% and oracle safe task success at most
  14.3%. Report this as a frozen feasibility no-go, not as failed model training
  and not as evidence against recovery methods that were never trained.
- The final scoped H=20 ledger has three certified pairs from 15 candidates
  (3/12 conditional on a Base accident). Its two development-evaluation source
  states achieve 6/6 safe task successes with Oracle timing/actions and 0/6
  catastrophes. Report this as a controller-compatible Oracle upper bound and
  stop; it does not support learned-policy or arbitrary-layout language.
