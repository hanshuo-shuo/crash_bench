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
tests. The provenance-complete two-task P0 extension is also complete, but it is
a negative/indeterminate result: calibration and held-out captures contained no
positive T-5 frames, held-out vanilla had 0/50 crashes, and the preregistered
hidden-over-baseline dissociation criterion was not supported.

The most important limitation is external validity: the strongest causal and
online intervention evidence is one LIBERO task and the OpenVLA-base on-path-wall
mode. The low-wall d62 task-completion result is only an existence demo; it is
not a general recovery result.

The geometry/provenance split, held-out P0 run, trivial-trigger controls, and
zero-GPU checks are complete. Any further external-validity experiment must be a
new preregistered study, not a post-hoc change to P0 scenarios, horizon, seeds,
repeats, or thresholds. The task-phase confound diagnostic remains a strict-OOF,
scenario-level frozen-capture result supporting only the scoped C13 wording;
P0 did not independently validate it. Prioritize a careful pi0 conclusion and a
new design only if it can create an identifiable held-out safety comparison.

See [PAPER_PLAN.md](PAPER_PLAN.md), [CLAIMS.md](CLAIMS.md),
[EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md), and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). The non-disruptive script map is in
[SCRIPT_INDEX.md](SCRIPT_INDEX.md).
