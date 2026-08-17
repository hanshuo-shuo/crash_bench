# Experiment index

This index is organized by paper role, not execution date. Stable E identifiers
remain unchanged so manifests and historical logs stay interpretable.

## Main paper

| ID | Scientific role | Scenario/model | Canonical implementation | Tracked evidence | Status / limitation |
|---|---|---|---|---|---|
| E16 | **headline fresh intervention-value routing frontier** | matched glass/off-path/no-glass; frozen single-frame router | `train_counterfactual_outcome_router.py`, `collect_fresh_counterfactual_router.py`, `analyze_fresh_counterfactual_router.py` | promoted summary, seven-method table, n8/n13 analyses, frontier CSVs, and four figures; Quest jobs `9677761`, `9679137` | frozen positive frontier; 8-source confirmation has four joint points; combined 13-source heterogeneity; structured privileged options |
| E1 | swept-corridor wall causal sweep | `scenarios/`, `scenarios_control/`; OpenVLA | `phase1_ood_control_v5.py`; `run_ood_control_v5.sbatch` | `pilot_final.json`, `pilot_control_final.json`, `ood_control_final.json`; `ANALYSIS_ood_control.md` | frozen; C1; one task/wall family |
| E2 | cross-action-head behavior | same matched wall/control geometry; Base, OFT, pi0 | `run_pilot_*`; `path3_oft_compare.py` | `*_walls.json`, `*_controls.json`, `path3_oft_summary.json` | frozen; C2; same embodiment/task |
| E3 | representation risk readout | wall/nowall/off-path captures; Base and OFT | `probe_selfreport.py`, `probe_selfreport_analysis.py` | `selfreport*/probe_summary.json`, `probe_T5.npz` | frozen; C3; raw captures ignored |
| E4 | geometry-aligned final-window behavior | paired wall/no-wall runs; Base/OFT/pi0 diagnostics | `wall_directed_braking.py` and probe analyses | probe summaries; `ANALYSIS_selfreport.md` | frozen/partial; C5; EEF measure |
| E5 | explicit risk routing | Base vs probe-gated `RetreatHold`; benign controls | `phase3_intervention.py`; `phase3_intervention.sbatch` | `intervention/{episodes,summary}.json`; `ANALYSIS_intervention.md` | frozen; C7; fitted wall family |
| E7 | detector-direction steering ablation | wall/nowall; OpenVLA | `phase3_steering*.py`; steering sbatch | `steering/*.json`; `ANALYSIS_steering.md` | frozen negative; C11 |
| E11 | representation/observable confound | strict held-out-scenario frozen-capture analysis | `task_phase_confound_analysis.py` | `task_phase_confound/*.json`; analysis figure | frozen; C13 |
| E6 | glass causal extension and transfer limit | f30–f70 matched glass; OpenVLA | `phase2_glass_prototype.py`, `probe_glass_*` | `glass_prototype.json`, glass probe summary; `ANALYSIS_glass.md` | frozen; C9/C10; closing evidence |

## Next experiment queue — not yet evidence

| Priority | Experiment | Gate before submission | Intended upgrade |
|---|---|---|---|
| N1 | frozen-router replication on a new task family | preserve model/options/grid; freeze a source-disjoint cohort before outcomes | test whether the intervention-value frontier transfers beyond task 0 |
| N2 | five-fold held-out online wall guard | recover/recapture wall hidden/meta; freeze fold-local calibration | mechanism generalization of the older risk-routing interface |
| N3 | OFT-specific online `RetreatHold` guard | recover/recapture OFT hidden/meta; fit OFT probe only | cross-action-head mechanism replication |

## Appendix experiments

| ID | Role | Evidence | Status / caveat |
|---|---|---|---|
| E0 | nominal bridge gate | historical 400/500; expected `m1_nominal_gate.json` missing | provenance gap; C0 |
| E8 | structured safe-abort witness | `witness.json`, `WITNESS.md` | frozen; C6; not task completion |
| E9 | low-wall detour existence | `phase2_task_witness/summary.json` | frozen; C12; separate geometry |
| E10 | failed/blocked hazard mechanisms | `nowall.json`, `obj_collision.json`, `grasp*` | retained negative evidence; [NEGATIVE_RESULTS.md](appendix/NEGATIVE_RESULTS.md) |
| E12 | hazard-validity/generalization study (internal P0) | `p0_core_20260726_retry1/summary.json` | complete negative/indeterminate; no held-out vanilla crashes or positive T-5 frames |
| E13 | prompt-scope baseline | `careful_prompt/*`; `ANALYSIS_careful_prompt.md` | complete; lower collision with 0/15 treatment task success |

The complete E12 and E13 protocols are under [appendix/](appendix/README.md).

## Path-stable legacy experiments

| ID | Historical purpose | Frozen outcome | Why it is not current |
|---|---|---|---|
| E14 | recoverable-glass acceptance smoke | 3 accepted admissions from 109 attempts | environment-validity existence only; old v1 cohort |
| E15 | broad exact-H glass recovery frontier | all 120 cells complete; no H qualified | feasibility no-go; no learned result |
| E15-SCOPED | controller-compatible H=20 development route | 3/15 certified; Oracle 6/6 on two development states; checkpoint readiness audit | Oracle timing/actions; learned gate/action no-go; validation reuse; no final-held-out claim |

The E14/E15 Python, setup, test, and result paths remain in place because they
are mutually imported and bound by `results/manifest.json`. Submission wrappers
are legacy-gated to prevent accidental GPU use. Their dated execution records are
under [archive/glass_recovery_20260812/](archive/glass_recovery_20260812/README.md).

For result-level provenance, use `results/manifest.json` and
`results/claims_ledger.json`. For script status, use
[SCRIPT_INDEX.md](SCRIPT_INDEX.md).
