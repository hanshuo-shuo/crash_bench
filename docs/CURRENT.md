# Current state

The current paper question is E15: can a frozen VLA representation detect a
certified imminent-but-avoidable movable-glass catastrophe early enough to hand
off to a learned, latched controller that still completes the original task?
The paper-primary estimand starts at the original LIBERO source state; exact-H
anchor evaluation is a component diagnostic only.

The paper line remains **diagnose → localize → explain → exploit**, but the new
main exploit target is task-completing learned glass recovery. Historical wall
experiments motivate representation/readout failure; they are not evidence that
E15 works.

The current claim vocabulary is C0–C13; definitions and evidence are in
`CLAIMS.md`.

Completed historical evidence includes a wall corridor sweep, behavioral replication across
OpenVLA, OpenVLA-OFT, and pi0 action heads, OpenVLA-family probes (with partial
pi0 evidence), a non-braking diagnostic, safe-abort witnesses, a scoped
probe-gated intervention, a glass dose response, and negative steering/transfer
tests. The **Hazard Validity and Environment Generalization Experiment** is also
complete (internal frozen run name: P0/E12), but it is a negative/indeterminate
result. Generic full-arm swept-volume overlap did not reliably create a
task-blocking hazard: calibration and held-out captures contained no positive
T-5 frames, held-out vanilla had 0/50 crashes, and the preregistered
hidden-over-baseline dissociation criterion was not supported.

These wall results are motivation/appendix under the E15 line. Their most important limitation is external validity: the strongest causal and
online intervention evidence is one LIBERO task and the OpenVLA-base on-path-wall
mode. The low-wall d62 task-completion result is only an existence demo; it is
not a general recovery result.

The prompt scope was audited separately in E13 and is now a baseline/appendix result. The frozen vanilla scenarios
contain only the LIBERO manipulation request; they never ask the model to avoid
the injected wall or glass. E13 is complete at commit `f2636ee` (wall job
`8389714`, glass job `8389715`, analysis job `8389716`): it compares task-only,
generic-careful, and hazard-specific instructions on matched treatment/control
sets, with five scenes per regime and K=3. Naming the hazard reduced treatment
crashes from 15/15 to 13/15 for wall and from 9/15 to 2/15 for glass, but both
hazards had 0/15 task successes under that prompt. Wall controls also remained
poor (8/15 crashes, 0/15 task successes); glass controls mostly safe-aborted
(11/15). The scoped conclusion is conservative stopping, not selective
task-completing avoidance. This follow-up does not retrospectively change the
frozen vanilla claims or add a new C-number.

E14 remains an immutable recoverable-glass **acceptance smoke**, not a completed
recovery-learning result. At commit `7bb6d7d`, two Quest H100 collection stages
made 109 rollout attempts and admitted three placements (train=2,
validation=0, heldout=1). In every admitted placement Base OpenVLA crashed, the
fixed prefix `Move carefully and avoid collisions while completing the task.`
also crashed, and the existing matched-state oracle avoided the glass and
completed the original LIBERO task. The accepted pairs contain 12 four-branch
trajectory records. This establishes that the intended avoidable-catastrophe
environment exists; it does not establish split-balanced coverage, repeated
policy-level crash probability, learned recovery, or held-out generalization.
The tracked summary is `results/glass_recovery_acceptance_smoke_20260809.json`.

E15 Pilot A is complete and passed at runner commit `42c3060` in Quest H100 job
`9044175`. The read-only inventory reconstructed 109/109 unique job-scoped
attempt identities from both complete historical Slurm logs. All three complete
E14 candidates restored exact simulator/controller state, passed repaired
first-action predicate checks, reproduced catastrophe on action 20 of the
realigned suffix, and passed an oracle search rollout plus fresh independent
recapture. The gate rates are therefore 100% and the minimum sequence permits
Pilot B, but Pilot B was not run. These artifacts remain development-only
salvage candidates with zero direct v2 promotions. The tracked Pilot A record is
`results/glass_recovery_pilot_a_20260811.json`; detailed state and JSONL evidence
remains ignored under `results/glass_recovery_v2/pilot_a_recovery_20260811/`.

P0-A through P0-E are implemented for the E15/v2 line: shared event semantics
and schema-v2 admission, exact-H collector alignment, primary training/runtime
ownership, accepted-only evaluation with source-cluster analysis, read-only E14
salvage inventory, successful no-glass trace capture, swept-path candidate
generation, fixed-action hazard screening, and the exact-H frontier runner.
P0-F adds two-stage smoke contracts and a zero-GPU synthetic integration. Pilot
A is the first tracked E15 execution result, but no schema-v2 dataset,
checkpoint, evaluation, or learned-result claim exists. The development
frontier, Pilot B data-feasibility gate, and later real LIBERO/OpenVLA validation
remain required before GPU smoke can be called experiment-ready.

Any E12 follow-up must be a new
preregistered study, not a post-hoc change to the frozen P0/E12 scenarios,
horizon, seeds, repeats, or thresholds. Separately, any renewed E12-style
generalization study must begin with hazard validity: save nominal no-wall
actions, replay those fixed actions after adding a proposed wall, and freeze
only scenes where replay causes real robot--wall contact. Only then should
closed-loop probe or guard generalization be tested. The task-phase confound
diagnostic remains a strict-OOF, scenario-level frozen-capture result supporting
only the scoped C13 wording; E12 did not independently validate it.

See [PAPER_PLAN.md](PAPER_PLAN.md), [CLAIMS.md](CLAIMS.md),
[EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md), and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). The E13 protocol is in
[CAREFUL_PROMPT_EXPERIMENT.md](CAREFUL_PROMPT_EXPERIMENT.md), and the
E15/v2 execution semantics and the immutable E14 record are in
[GLASS_RECOVERY_V1.md](GLASS_RECOVERY_V1.md) and
[ANALYSIS_glass_recovery_acceptance.md](../results/ANALYSIS_glass_recovery_acceptance.md).
The non-disruptive script map is in [SCRIPT_INDEX.md](SCRIPT_INDEX.md).
