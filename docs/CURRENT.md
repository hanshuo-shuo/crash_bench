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

The geometry/provenance split, held-out generalization run, trivial-trigger
controls, and zero-GPU checks are complete. Any follow-up must be a new
preregistered study, not a post-hoc change to the frozen P0/E12 scenarios,
horizon, seeds, repeats, or thresholds. The next priority is hazard validity:
save nominal no-wall actions, replay those fixed actions after adding a proposed
wall, and freeze only scenes where replay causes real robot--wall contact. Only
then should closed-loop probe or guard generalization be tested. The task-phase
confound diagnostic remains a strict-OOF, scenario-level frozen-capture result
supporting only the scoped C13 wording; E12 did not independently validate it.

See [PAPER_PLAN.md](PAPER_PLAN.md), [CLAIMS.md](CLAIMS.md),
[EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md), and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). The non-disruptive script map is in
[SCRIPT_INDEX.md](SCRIPT_INDEX.md).
