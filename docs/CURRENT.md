# Current state

CrashBench asks whether a VLA that sees a visible obstacle entering its actual
action-swept corridor represents imminent collision yet fails to read that
information out into a safe action.

The paper line is **diagnose → localize → explain → exploit**: controlled
on-path/off-path wall placement diagnoses and localizes the collision effect;
frozen probes show collision imminence is **decoded but not used**; a narrowly
scoped probe-gated retreat controller exploits that signal to avoid crashes.

The current claim vocabulary is C0–C12; definitions and evidence are in
`CLAIMS.md`.

Completed core evidence: a wall corridor sweep, behavioral replication across
OpenVLA, OpenVLA-OFT, and pi0 action heads, OpenVLA-family probes (with partial
pi0 evidence), a non-braking diagnostic, safe-abort witnesses, a scoped
probe-gated intervention, a glass dose response, and negative steering/transfer
tests.

The most important limitation is external validity: the strongest causal and
online intervention evidence is one LIBERO task and the OpenVLA-base on-path-wall
mode. The low-wall d62 task-completion result is only an existence demo; it is
not a general recovery result.

MUST next: complete the geometry/provenance split and run the zero-GPU checks;
then prioritize held-out task/wall external validity, a shield trivial-trigger
baseline, probe-confound controls, and a careful pi0 conclusion. Do not add a
broad hazard benchmark without resolving those items.

See [PAPER_PLAN.md](PAPER_PLAN.md), [CLAIMS.md](CLAIMS.md),
[EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md), and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). The non-disruptive script map is in
[SCRIPT_INDEX.md](SCRIPT_INDEX.md).
