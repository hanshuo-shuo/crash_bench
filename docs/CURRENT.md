# Current state

## Paper thesis

The current paper is **Decoded but Not Routed: Causal Diagnosis and Closed-Loop
Repair of VLA Collision Failures**.

Its central claim is narrower and stronger than the former glass-recovery plan:
a VLA can encode an imminent path collision without routing that information into
its action, and an explicit **risk-readout → controller** interface can repair the
resulting behavior in closed loop.

The paper uses the C0–C13 claim vocabulary defined in [CLAIMS.md](CLAIMS.md).
The primary chain is C1/C2 → C3/C5/C13 → C7, with C11 as the mechanism ablation
and C9/C10 as the glass extension.

## Evidence already closed

### 1. Swept-corridor causal localization

- Base OpenVLA crashes on 15/15 on-path walls and 0/33 clear off-path walls;
  the boundary region changes gradually with clearance.
- OpenVLA base, OpenVLA-OFT, and pi0 each reproduce the matched behavioral
  contrast: 5/5 on-path crashes and 0/10 clear off-path crashes.
- A movable-glass study provides a second contact mechanism: 30/50 on-path
  crashes versus 0/50 matched off-path, with an f30→f70 dose response.

These controls support a geometric statement about the executed swept corridor,
not the broad claim that any novel obstacle causes failure.

### 2. Representation–action gap

- Frozen hidden-state probes decode collision imminence at T-5 with AUC 0.998
  for OpenVLA and 0.903 for OFT. A glass-specific probe reaches 0.944.
- In 25 paired wall episodes, there is no final-window EEF retreat; the
  near-impact toward-wall command projection increases in 22/25.
- Under strict leave-one-scenario-out analysis, hidden state has macro AUPRC
  0.716 versus 0.442 for the strongest measured observable baseline.

The supported language is “decoded but not routed,” “represented but not read
out into safe action,” or “representation–behavior dissociation.” The repository
does not use an unqualified mental-state claim such as “the model knows.”

### 3. Closed-loop routing repair

- A wall probe triggers a latched structured `RetreatHold` controller.
- The intervention changes 15/15 crashes to 0/15, mean peak force from 321.7 N
  to 0 N, and produces 0/22 benign guard fires.
- Direct final-readout activation steering leaves the crash rate at 100% for all
  tested alphas. The diagnostic shows that the direction useful for detection is
  not a reliable controller direction.

This is a completed behavioral closed loop in one OpenVLA-base wall mode. It is
not yet held-out-wall online generalization, an OFT online replication, or a
general collision-avoidance system.

## Glass: supporting evidence, not the title

Glass closes the paper with the safety–utility distinction:

- Hazard-specific prompting reduces treatment collision, especially for glass,
  but both hazard-specific treatment cells have 0/15 task successes.
- `RetreatHold` demonstrates safe abort, not task completion.
- A scoped H=20 certification ledger finds three controller-compatible accidents
  among 15 candidates (3/12 conditional on a Base catastrophe).
- On two development source states, exact-anchor, Oracle-timed Oracle recovery
  achieves 6/6 safe task successes and 0/6 catastrophes. The two states were also
  used for checkpoint validation, so the independent sample size is two and this
  is an Oracle upper bound only.

The paper-facing interpretation and technical audit map are in
[GLASS_SAFETY_UTILITY.md](appendix/GLASS_SAFETY_UTILITY.md). That appendix links
the dated execution records when full restore, hash, authoring, or Slurm details
are needed; those records are provenance sources, not the current run queue.

## Why Pilots D/F are not active

The frozen checkpoint does not support the old learned-recovery sequence:

| Pilot | Timing | Action/controller | Current status |
|---|---|---|---|
| C | Oracle | Oracle | complete; 2 development states, 6/6 |
| D | learned | Oracle | no-go at the frozen calibration |
| E | Oracle | learned | prerequisite smoke not run |
| F | learned | learned | blocked by D and E; do not run |

The calibration threshold is 1.0 with timely-trigger rate 0.0; the six saved C
episodes never reach that threshold. The checkpoint's validation gripper-sign
accuracy is 0.846, below the frozen 0.95 gate. Therefore D would not hand off and
F would collapse to Base. The defensible detector summary is:

> Frame-level ranking signal exists, but a deployable operating point does not.

The hash-pinned derived values are recorded in the
[checkpoint readiness audit](../results/glass_recovery_checkpoint_readiness_audit_20260812.json).

The path-stable E15 code remains in the repository for provenance and tests, but
its submission wrappers are legacy-gated. Completing experiment labels is not a
reason to spend GPU time.

### Counterfactual-router exact-state option evidence

A source-disjoint D0 recapture now closes the old missing-artifact question: the
primary detector reaches calibration/development frame AUC 0.898/0.883, but has
0% exact-T−20 timely trigger rate at the low-control-FPR operating point.  Risk
ranking therefore remains a baseline, not a deployable router.

A development-only exact-state sweep froze one non-source-specific structured
`DetourComplete` configuration after two identical safe task successes on an
exposed H=20 state.  With that configuration, the protocol-correct multi-H smoke
(Quest job `9277740`, clean commit `d7bbf9f86b78`) produced the desired timing
boundary on one exposed development source:

| Glass horizon | Base | DetourComplete | RetreatHold |
|---|---|---|---|
| T−30 | catastrophe | task success | safe noncompletion |
| T−20 | catastrophe | task success | safe noncompletion |
| T−10 | catastrophe | safe noncompletion | safe noncompletion |
| T−5 | catastrophe | safe noncompletion | catastrophe |

Across the 12 matched glass/off-path/no-glass decisions, zero have identical
outcomes under all three options.  Seven clean-control decisions show that Base
can succeed while Retreat sacrifices completion; two glass decisions show a
task-completing Detour advantage.  Thus Detour is not an always-best fixed
recovery, and option value changes with both hazard condition and intervention
time.  This is development evidence from one source, not a generalization
claim; the source also supplied the H=20 controller-development checkpoint.
Full source-disjoint collection is now active as Quest job `9284055` at
`results/counterfactual_router/full_d7bbf9f86b78_20260814T131220Z`.

Exact provenance, the prior negative smoke, frozen configuration, and audit are
in [COUNTERFACTUAL_ROUTER_HANDOFF.md](COUNTERFACTUAL_ROUTER_HANDOFF.md).

## Next experiment queue

1. **Five-fold held-out online wall guard.** Fit/calibrate on four wall scenarios,
   deploy only on the fifth, and test matched off-path/no-wall controls. This is
   the highest-value generalization test.
2. **Complete the source-disjoint counterfactual option collection.** Full job
   `9284055` is queued.  On completion, audit outcome diversity before fitting
   any model, then train only the minimal temporal supervised router.  Call the
   result learned routing plus a structured/privileged controller, not learned
   recovery.
3. **OFT-specific online guard replication.** Use OFT's own probe to trigger the
   same `RetreatHold` interface.

The executable D0 contract, commands, operating gates, and E stop rules are in
[GLASS_RECOVERY_RESCUE.md](GLASS_RECOVERY_RESCUE.md).

There is an artifact prerequisite before these jobs: the working tree contains
the wall/OFT probe checkpoints but not the raw `hidden.npz`/`meta.json` needed for
five-fold refitting. The glass directory contains only the probe summary, not a
deployable checkpoint. Recover the Quest captures or recapture them before
submitting GPU evaluations. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Navigation

- Paper structure and cut list: [PAPER_PLAN.md](PAPER_PLAN.md)
- Exact claims and caveats: [CLAIMS.md](CLAIMS.md)
- Current versus appendix versus legacy experiments:
  [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md)
- Script status: [SCRIPT_INDEX.md](SCRIPT_INDEX.md)
- Frozen protocols and negative evidence: [appendix/](appendix/README.md)
- Superseded execution timelines: [archive/](archive/README.md)
