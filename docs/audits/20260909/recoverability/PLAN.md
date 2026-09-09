# Recoverability mechanism investigation — 2026-09-09

User explicitly authorizes paper-oriented exploration from e18/e21 and one very small experiment. Preserve all prior outcomes; no model expansion, D8 access or automatic confirmation.

First: inspect all twelve fitting-side glass anchors and their existing R traces. Distinguish fixed-controller success, accident, and deadline noncompletion; failure of one controller is not proof of physical impossibility. Reconstruct existing actions from saved simulator states using forward kinematics, checking reconstruction errors before interpreting controller stages. No simulator time integration or new outcomes in this inspection.

Candidate mechanism: physically reachable approach states may enter or leave the fixed recovery controller's success region. A small follow-up may advance Base by bounded steps from exact parent bundles, then branch Base vs unchanged Detour. Select two failed controls using a declared physical-distance rule, with e18/e21 retained as positive references. Freeze exact cells, counts, budgets and stop rules after the read-only inspection and before new trajectories. Do not tune to new outcomes.

The next deliverable is a causal hypothesis and a bounded executable probe, followed by a Chinese paper-oriented report. Do not call a region a certified viability kernel or confuse collision avoidance with task completion. Existing selected parents remain development evidence.

## Frozen very-small experiment

`configs/recoverability/probe_v1.json` fixes e18/e21 and nearest distinct failed fitting glass controls e27/e06. The metric uses relative EEF/glass/plate positions to bowl plus glass radius/height, all in metres; greedy matching e18 then e21, no metric sweep. All four are exposed development parents, not population or confirmation samples.

Generate child states by executing the existing Base proposal/continuation for offsets0,3,10. At most four short prefix continuations and40 added simulator actions, no teleportation, obstacle edits, or recovery-controller tuning. Stop generation at first success/accident; preserve unavailable requested children rather than generating post-accident states. Save new full child bundle and pending proposal before scoring. All child bundles are generated before any new scored outcome is evaluated.

Each available child has Base/R ×2 repeats with identical restored RNG and balanced order. Maximum48 scored branches. Fixed OpenVLA checkpoint and Detour class. New logging only observes controller stages, target errors, cap exits, EEF/bowl/plate positions and gripper values; action parity is unit-tested. No model fitting.

Compare both the historical absolute episode220/440 readouts and uniformly allocated suffix220/440 action budgets from each decision. They come from the same continuous branch, run to decision_index+440; no horizon reset. The uniform suffix budget is a new exploratory measurement, not a replacement or rewrite of the earlier primary protocol. It distinguishes old right-censoring from physical-state effects. Extension actions are separately recorded, and absolute-clock outcomes remain visible.

Primary contrasts: R success at offset3/10 vs offset0 under equal suffix440; Base accident/success under the same budget; appearance of task rescue on failed parents. Record deterioration/lost opportunities in positive parents. Two repeats cannot certify a state effect. If only extra suffix budget changes the label, classify budget censoring. If new R success but Base also succeeds, distinguish controller success from intervention necessity. If no negative parent becomes recoverable, do not expand offsets or controllers automatically.

Read-only job5808426 failed its strict fresh-FK/cached-observation equality check before producing stage conclusions (no rollout). Preserve this as a measurement limitation. The updated inspector reports reconstruction error and flags invalid stage reconstruction; live phase logging in the tiny probe is the authoritative phase evidence.

## Paper framing, primary references

Measure fixed-controller, finite-horizon task recoverability, not universal physical impossibility: R failing does not establish that no other controller can recover. Keep task completion distinct from safe-region recovery as in [Recovery RL](https://arxiv.org/abs/2010.15920). A useful concept is an empirically measured success region and its physically reachable boundary; unlike [Funnel Libraries](https://arxiv.org/abs/1601.04037) and [SOS control funnels](https://arxiv.org/abs/1210.0888), this experiment supplies no invariant-set certificate. Merely renaming a controller success set is not novelty. A paper direction requires a repeatable physical mechanism and prospective generation of recovery-capable states, with necessity and budget controls.
