# Experiment index

Stable identifiers below replace Phase/Path/Week labels in current documents.
Status is based on tracked artifacts only.

| ID | Question; treatment / control | Scenario set; model | Command / sbatch | Raw result; analysis / figure | Status; claim; caveat |
|---|---|---|---|---|---|
| E0 Nominal bridge gate | Does the evaluation bridge preserve nominal competence? | LIBERO nominal; OpenVLA | `nominal_task_gate.py`; `m1_nominal_gate.sbatch` | expected output `results/m1_nominal_gate.json` is not tracked | historical sanity only; C0; provenance gap |
| E1 Wall corridor causal sweep | on-path wall / fixed visible off-path clearance sweep | `scenarios`, `scenarios_control`; OpenVLA base | `phase1_ood_control_v5.py`; `run_ood_control_v5.sbatch` | `pilot_final.json`, `pilot_control_final.json`, `ood_control_final.json`; `ANALYSIS_ood_control.md` | frozen; C1; transition zone and one task |
| E2 Cross-policy behavior | same walls and controls / same geometry bins | base, OFT, pi0 | `run_pilot_*`; `path3_oft_compare.py` | `*_walls.json`, `*_controls.json`, `path3_oft_summary.json` | frozen; C2; action-head replication, not independent task diversity |
| E3 Representation probe | imminent on-path collision / nowall and off-path frames | wall captures; base and OFT | `probe_selfreport.py`, `probe_selfreport_analysis.py` | `selfreport*/probe_summary.json`, `probe_T5.npz` | frozen; C3; capture dumps are ignored |
| E4 Geometry-aligned braking | action near impact / nowall action magnitude | wall capture; base/OFT/pi0 | probe analyses and `wall_directed_braking.py` | probe summaries; optional braking output not tracked | frozen/partial; C5; not a full action-space causal test |
| E5 Probe-gated intervention | guarded / bare on-path wall; guarded benign controls | tall on-path walls plus clear controls; OpenVLA base | `phase3_intervention.py`; `phase3_intervention.sbatch` | `intervention/{episodes,summary}.json`; `ANALYSIS_intervention.md` | frozen; C7; scoped controller and one mode |
| E6 Glass object-collision | glass on path / matched off-path at f30–f70 | `scenarios_glass`; OpenVLA base | `phase2_glass_prototype.py`; `probe_glass_*` | `glass_prototype.json`, glass probe summary; `ANALYSIS_glass.md` | frozen; C9/C10; report full dose response |
| E7 Activation steering | final-readout steering alpha sweep / alpha=0 | wall and nowall; OpenVLA base | `phase3_steering.py`, `phase3_steering_diag.py` | `steering/*.json`; `ANALYSIS_steering.md` | frozen negative; C11 |
| E8 Recoverability witnesses | structured retreat or replay / bare collision path | wall scenarios | `phase2_witness.py`, `phase2_task_witness.py` | `witness.json`, `phase2_task_witness/summary.json`; `WITNESS.md` | frozen; C6; task witness is low-wall d62 only |
| E9 Detour existence demo | low-wall task-completing detour / tall-wall failures | d62 low-wall variant | `phase2_task_witness.py`, `phase3_detour_handoff.py` | `phase2_task_witness/summary.json` | frozen existence demo; C12; not general recovery |
| E10 Failed/blocked hazards | alternate hazards and mechanisms / nominal conditions | edge bowl, fixtures, cookies, grasp, border | Phase-2 helper scripts | `nowall.json`, `obj_collision.json`, `grasp*`, notes | retained negative evidence; see `NEGATIVE_RESULTS.md` |
| E12 Hazard Validity and Environment Generalization Experiment (internal P0) | two tasks; 3/3/5 train/calibration/held-out wall placements; wall/nowall; probe/baseline analysis and 11-method online guard | `scenarios_p0`; exact OpenVLA revision | `p0_capture.py`, `p0_probe_analysis.py`, `p0_guard.py`; internal P0 sbatch chain | `p0_core_20260726_retry1/summary.json` | complete negative/indeterminate result; generic swept-volume overlap did not ensure a path-blocking hazard; no calibration/held-out positive frames; held-out vanilla 0/50 crashes |

Current canonical commands must be read together with
[REPRODUCIBILITY.md](REPRODUCIBILITY.md); old sbatch paths remain available for
historical reproducibility.
