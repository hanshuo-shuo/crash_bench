# Script index

“Current” means a preferred implementation for the paper line. “Appendix” means
frozen supporting or negative evidence. “Path-stable legacy” means retained at
its historical location for imports, tests, manifest provenance, or exact
reproduction; it is not a recommended new experiment.

## Current paper line

| Function | Python | Setup / submission |
|---|---|---|
| **P2 dynamic first-crossing routing** | `collect_dynamic_first_crossing_router.py`, `analyze_dynamic_first_crossing_router.py`; latch and calibrated LCB contract in `crashbench/counterfactual_router.py` | `dynamic_first_crossing_router.sbatch`, `submit_dynamic_first_crossing_router.sh`; protocol in `P2_DYNAMIC_FIRST_CROSSING.md` |
| **headline counterfactual option routing** | `train_counterfactual_outcome_router.py`, `collect_fresh_counterfactual_router.py`, `analyze_fresh_counterfactual_router.py`; contract in `crashbench/counterfactual_router.py` | frozen Quest chains in `submit_fresh_counterfactual_router.sh` and `submit_fresh_counterfactual_router_supplement.sh`; result and interpretation in `COUNTERFACTUAL_ROUTER_MAIN_RESULT.md` |
| repository integrity | `audit_repo.py`, `update_scenario_manifest.py`, `prepare_m0_geometry.py` | none; zero-GPU |
| wall causal sweep | `phase1_ood_control_v5.py`, `phase1_ood_control_analysis.py`, `probe_nominal_traj.py`, `phase1_make_figures.py` | `run_ood_control_v5.sbatch`, `probe_nominal_traj.sbatch` |
| cross-policy behavior | `path3_oft_compare.py`, shared `run_pilot.py` runner | `run_pilot_base_matched.sbatch`, `run_pilot_oft.sbatch`, `run_pilot_openpi.sbatch` |
| representation readout | `probe_selfreport.py`, `probe_selfreport_capture.py`, `probe_selfreport_analysis.py`, `probe_dump_T5.py`, `probe_horizon_relabel.py` | Base/OFT probe sbatch files |
| final-window behavior | `wall_directed_braking.py` | `wall_directed_braking.sbatch`, `submit_wall_directed_braking.sh` |
| risk routing repair | `phase3_intervention.py`, `phase3_make_figures.py`, `probe_shield_sweep.py` | `phase3_intervention.sbatch` |
| steering ablation | `phase3_steering.py`, `phase3_steering_diag.py`, `phase3_steering_figures.py` | steering sbatch files |
| observable confound | `task_phase_confound_analysis.py` | none; reads frozen captures |
| glass causal extension | `phase2_glass_prototype.py`, `probe_glass_capture.py`, `probe_glass_analysis.py`, `probe_glass_inject.py`, `probe_joint_transfer.py` | `phase2_glass_prototype.sbatch`, `probe_glass.sbatch` |
| glass detector rescue D0 | `capture_glass_detector_placements.py`, `prepare_glass_detector_split.py`, `fit_glass_detector.py`; runtime artifact in `crashbench/glass_detector.py` | `glass_detector_d0_capture.sbatch`, `submit_glass_detector_d0.sh`; see `GLASS_RECOVERY_RESCUE.md` |

`run_pilot.py` remains shared runtime code even though its earliest experiment
configuration is historical. Status applies to use, not merely filename age.

## Appendix / frozen support

| E ID | Scripts | Status |
|---|---|---|
| E0 | `nominal_task_gate.py` | gate artifact missing; do not headline |
| E8/E9 | `phase2_witness.py`, `phase2_task_witness.py`, `phase3_detour_handoff.py`, `render_d70_detour.py`, `phase2_lowwall_validity.py` | safe-abort and low-wall existence only |
| E10 | `phase2_build_nowall.py`, `phase2_run_nowall.py`, `phase2_obj_collision.py`, `phase2_recon_indist.py`, `phase2_grasp*.py` | retained negative/diagnostic paths |
| E12 | `p0_author_scenarios.py`, `p0_capture.py`, `p0_probe_analysis.py`, `p0_guard.py` | frozen negative/indeterminate study |
| E13 | `careful_prompt_eval.py`, `analyze_careful_prompt.py` | complete prompt-scope baseline |
| old safety baselines | `safety_baseline_suite.py`, `analyze_safety_baselines.py`, report builders, `qwen_vlm_server.py` | preliminary appendix/legacy material; E13 supersedes the prompt interpretation |
| old oracle-stop fine-tune | `build_oracle_stop_dataset.py`, `finetune_openvla_oracle_stop.py`, `analyze_recovery_finetune.py` | learned stop proxy, not recovery |

## Path-stable glass-recovery legacy

The following package is kept intact because its scripts import one another, two
large test modules execute the paths directly, and the result manifest records
the original generating paths:

- data/model/runtime: `crashbench/glass_recovery_data.py`,
  `crashbench/glass_recovery_model.py`,
  `crashbench/policies/glass_recovery_policy.py`;
- collection/frontier: `capture_glass_nominal_source_traces.py`,
  `prepare_glass_recovery_placements.py`, `audit_glass_core_artifacts.py`,
  `realign_glass_core_artifacts.py`, `diagnose_glass_continuation_oracle.py`,
  `run_glass_avoidability_frontier.py`, `collect_glass_recovery_pairs.py`,
  `summarize_glass_pilot_b.py`, `replay_glass_recovery_pair.py`;
- training/evaluation: `train_glass_recovery.py`,
  `seal_glass_recovery_evaluation.py`, `eval_glass_recovery.py`,
  `analyze_glass_recovery_eval.py`, `summarize_glass_pilot_cdef.py`;
- setup: `glass_core_realign*`, `glass_recovery_pilot_b*`, and
  `glass_recovery_smoke*`.

These files preserve the E14/E15 audit trail. They are not the active paper
roadmap, and their submit wrappers require the explicit environment opt-in
`CB_ENABLE_LEGACY_GLASS_RECOVERY=1`.

## Superseded implementations

Early wall builders (`phase1_build_env_collision.py`,
`phase1_build_ood_control.py`), early repeated runners, and old `phase2_*`
authoring paths remain available when referenced by frozen results. Prefer the
canonical row in [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) for any new run.

The obsolete multi-job paper-round submitter and automatic P0 commit/push helper
were moved to [legacy/setup](../legacy/README.md) and require an explicit legacy
opt-in.
