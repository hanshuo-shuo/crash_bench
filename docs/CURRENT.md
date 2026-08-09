# Current state

CrashBench asks whether a VLA that sees a visible obstacle entering its actual
action-swept corridor represents imminent collision yet fails to read that
information out into a safe action.

The paper line is **diagnose → localize → explain → exploit**: controlled
on-path/off-path wall placement diagnoses and localizes the collision effect;
frozen probes show collision imminence is **decoded but not used**; a narrowly
scoped probe-gated retreat controller exploits that signal to avoid crashes.

The current claim vocabulary is C0–C13; definitions and evidence are in
`CLAIMS.md`.

Completed core evidence: a wall corridor sweep, behavioral replication across
OpenVLA, OpenVLA-OFT, and pi0 action heads, OpenVLA-family probes (with partial
pi0 evidence), a non-braking diagnostic, safe-abort witnesses, a scoped
probe-gated intervention, a glass dose response, and negative steering/transfer
tests. The **Hazard Validity and Environment Generalization Experiment** is also
complete (internal frozen run name: P0/E12), but it is a negative/indeterminate
result. Generic full-arm swept-volume overlap did not reliably create a
task-blocking hazard: calibration and held-out captures contained no positive
T-5 frames, held-out vanilla had 0/50 crashes, and the preregistered
hidden-over-baseline dissociation criterion was not supported.

The most important limitation is external validity: the strongest causal and
online intervention evidence is one LIBERO task and the OpenVLA-base on-path-wall
mode. The low-wall d62 task-completion result is only an existence demo; it is
not a general recovery result.

The prompt scope has now been audited separately. The frozen vanilla scenarios
contain only the LIBERO manipulation request; they never ask the model to avoid
the injected wall or glass. The earlier generic prefix (`move slowly, avoid
collisions`) still produced 15/15 wall crashes, but that single wording does not
settle whether a visually grounded, hazard-specific request can elicit safer
behavior. E13 therefore compares task-only, generic-careful, and hazard-specific
instructions on matched wall and glass treatment/control sets. It is a new
follow-up, not a reinterpretation of the frozen results, and it does not create a
new paper claim until its tracked outputs are analyzed. The fixed run was
submitted from commit `f2636ee` as wall job `8389714`, glass job `8389715`, and
dependency-gated analysis job `8389716`.

E14 now has a verified recoverable-glass **acceptance smoke**, not a completed
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

The geometry/provenance split, held-out generalization run, trivial-trigger
controls, and zero-GPU checks are complete. For E14, the immediate next step is
to improve path intersection without weakening the crash/oracle predicates,
obtain validation-split accepted placements, and add repeated Base rollouts
before training the recovery head. Any E12 follow-up must be a new
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
E14 protocol and promoted result are in
[GLASS_RECOVERY_V1.md](GLASS_RECOVERY_V1.md) and
[ANALYSIS_glass_recovery_acceptance.md](../results/ANALYSIS_glass_recovery_acceptance.md).
The non-disruptive script map is in [SCRIPT_INDEX.md](SCRIPT_INDEX.md).
