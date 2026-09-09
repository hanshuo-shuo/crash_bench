# Execution provenance

- Read-only old-trace reconstruction job5808426 at a122145 failed after14s because cached EEF observation did not match fresh forward kinematics at e00. No new trajectory was executed. No inferred stage was accepted as ground truth.
- Tiny reachable-state probe job5809503 at `cebbe3445a9151930bf7b54e04689bb695f5ccb9`, one A100, p33100/gengpu, 8CPU/96GB, 1h cap.
- Output: `/projects/p33100/siosio/crashbench_detour_benefit/recoverability_job5809503`, repository alias `results/detour_benefit/recoverability_job5809503`.
- Four fitting-side parents e18/e21/e27/e06, offsets0/3/10, two options × two identical-restored-RNG repeats, max48 branches. At most4 prefix extensions/40 added actions.
- All child bundles generated before scoring. Candidate-unavailable terminals retained; no replacement.
- All parent hashes/configuration and weight bytes verified by runtime; no model trained, controller actions unchanged. Instrumentation action parity and budget semantics covered by15 passing combined tests.
- Final: COMPLETED, elapsed00:23:10, exit0:0. All12 children,48 branches,4 prefixes/40 added actions. No unavailable children or expansion.
- Analysis: no newly successful negative-parent states; detailed results and limitations in RESULTS_ZH.md. Sixteen combined tests passed.
- Local analysis and static figure rendering completed; both figures visually reviewed. Evidence bundle hashes preserved in evidence/manifest.json.
