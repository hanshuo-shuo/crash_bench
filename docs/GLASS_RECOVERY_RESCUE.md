# Glass recovery rescue: D0 → counterfactual routing

> **Completed-path provenance:** this document preserves the execution logic
> that led from the D0 no-go to counterfactual routing. Full option collection,
> router training, and fresh online evaluation are now complete. The current
> result is [COUNTERFACTUAL_ROUTER_MAIN_RESULT.md](COUNTERFACTUAL_ROUTER_MAIN_RESULT.md);
> technical development history is
> [COUNTERFACTUAL_ROUTER_HANDOFF.md](COUNTERFACTUAL_ROUTER_HANDOFF.md), and the
> from-reset timing boundary is closed in
> [P2_DYNAMIC_FIRST_CROSSING.md](P2_DYNAMIC_FIRST_CROSSING.md). The steps below
> are not a current experiment queue.

This branch preserves the old D/E/F experiment labels as history. New work uses
new artifacts and names the learned component precisely:

> glass-specific learned detector + frozen structured task-completing controller

The legacy 7-D recovery action head is not a prerequisite for F′.

## D0: deployable H=20 glass detector

The current repository tracks only the old aggregate glass-probe summary. It
does not contain the ignored `hidden.npz` and `meta.json`, so the first cluster
action is to recover those captures or recapture them. A D0 capture now records:

- frozen OpenVLA hidden state;
- 8-D robot state;
- unchanged 7-D nominal action;
- immutable source-state hash, episode ID, and rollout seed;
- action-exact time to catastrophe, where the impact action is one action away;
- pinned checkpoint and clean Git provenance.

Example capture:

```bash
python scripts/probe_glass_capture.py \
  --checkpoint-revision 962318cec55ac10993ff0f5f43eda9a270b4c873 \
  --output results/glass_recovery_v2/d0_capture \
  --rollout-seeds 101 202 303
```

The output directory must not exist unless `--overwrite` is explicit. The old
ten-scenario capture is useful for pipeline validation but is too small for a
deployable FPR claim; expand the source cohort before treating D0 as paper
evidence.

Quest retains two suitable placement designs. The exposed r5 design has 18
independent source states: its original 8 train sources remain D0 train and its
5 validation + 5 heldout sources become D0 calibration because all were exposed
during engineering. The disjoint fresh r2 design has 9 sources and is
development-only. The placement capture freezes that 8/10/9 assignment and
collects all three matched conditions with complete features:

```bash
setup/submit_glass_detector_d0.sh smoke
# Inspect the one-placement result before spending on the full capture.
setup/submit_glass_detector_d0.sh full
```

The full job fits D0 automatically only after capture succeeds. Off-path
episodes that actually collide and on-path episodes with no T−20 anchor are
retained in `capture_manifest.json` as explicit exclusions; they are not
silently converted into negative controls.

### Freeze source splits before fitting

Generate a deterministic, condition-stratified manifest. Assignment consumes
only source hashes and designed conditions, never outcomes, activations, or
scores:

```bash
python scripts/prepare_glass_detector_split.py \
  --metadata results/glass_recovery_v2/d0_capture/meta.json \
  --output results/glass_recovery_v2/d0_capture/source_split.json \
  --seed 20260812
```

Commit or hash-pin `source_split.json` before fitting. Every source belongs to
exactly one of `train`, `calibration`, or `development`; frame-level and repeated
episode splits are rejected.

### Fit, calibrate, and compare baselines

```bash
python scripts/fit_glass_detector.py \
  --capture results/glass_recovery_v2/d0_capture/hidden.npz \
  --metadata results/glass_recovery_v2/d0_capture/meta.json \
  --split-manifest results/glass_recovery_v2/d0_capture/source_split.json \
  --output results/glass_recovery_v2/d0_detector
```

The primary model is `hidden_robot_action`. Two fixed baselines are always fit
with the same splits and labels: `hidden_only` and `robot_action_only`.

Calibration is episode-level and deliberately conservative:

- treatment score: detector score at the exact T−20 anchor;
- control score: maximum over the full off-path or no-glass episode;
- objective: maximize exact-T−20 trigger rate subject to control episode FPR;
- threshold: a real logit-space value, never the old `1.0` sentinel.

`d0_summary.json` allows D1 only when all checks pass: timely trigger rate at
least 80%, control episode FPR at most 10%, enough control episodes to resolve a
10% step, both control types present, at least five independent calibration
source states, and a nonzero feasible operating point.

## D1: development-only online composition

D1 is closed as a no-go: the completed D0 operating gate failed.  It must not be
resubmitted merely to complete an experiment label.

## Counterfactual option routing smoke

The replacement direction learns option outcomes rather than a binary risk
threshold.  Its collector runs online Base, frozen `DetourComplete`, and
`RetreatHold` from byte-identical multi-H states and records task success,
catastrophe, or safe noncompletion.  The implementation and final smoke are
complete, but full collection is gated on a task-completing frozen option.

The final smoke validates exact-state branching, eight-frame temporal features,
source-state split enforcement, online Base continuation, and three-way outcome
labels.  It does not validate the current detour: Base and Detour both
catastrophize at T−20, T−10, and T−5 on the valid development source.  Do not fit
a router to this smoke because it contains no positive task-completing advantage
cell.

## E smoke: strict side lane

The old action head receives only Oracle timing: one train pair and two
development pairs, one seed each.

- zero train successes: terminate the old action-head route;
- train signal but zero development signal: do not run F;
- development signal: redesign the head before further work, including a binary
  gripper output and explicit recovery phase/ownership age.

## F′ and fresh cohort

F′ uses the D0 detector and one frozen non-pair-specific structured controller.
After the interface is frozen, collect at least 8–12 independent fresh source
states. Repeated seeds are robustness repetitions, not additional independent
states. The final result must report learned detector + structured controller,
not rename `risk_gate_oracle_recovery` as learned recovery.

## Final handoff

The route ultimately became E16 counterfactual outcome routing over Base,
structured Detour, and Retreat rather than detector-gated F′. E16 is positive
at matched T-20 decision states. P2 then evaluates causal from-reset first
crossing: source-level calibration yields a selective development Pareto point
but misses both known recoverable windows. P2.5 shows that simple temporal
aggregation does not fix the score ordering, so threshold and accumulator
tuning are closed. The minimum future method change is recovery-window
supervision; this document remains provenance, not an active submission plan.
