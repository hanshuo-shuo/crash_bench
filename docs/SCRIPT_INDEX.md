# Script index

This index is a deprecation layer, not a disruptive move: historical sbatch files
and script paths remain valid. “Canonical” means the preferred entry for that
function; “helper” means it supports a canonical experiment; “deprecated” means
retain for provenance only and do not use for a new paper result.

| E ID | Status | Python scripts | Paired / relevant sbatch |
|---|---|---|---|
| repository hygiene | canonical | `audit_repo.py`, `update_scenario_manifest.py`, `prepare_m0_geometry.py` | none; all zero-GPU |
| E0 | canonical | `nominal_task_gate.py` | `m1_nominal_gate.sbatch`, `run_libero_sanity.sbatch`, `run_libero_full.sbatch` |
| E1 | canonical | `phase1_ood_control_v5.py`, `phase1_ood_control_analysis.py`, `probe_nominal_traj.py`, `phase1_make_figures.py` | `run_ood_control_v5.sbatch`, `probe_nominal_traj.sbatch` |
| E1 | deprecated historical versions | `phase1_build_env_collision.py`, `phase1_build_ood_control.py`, `run_pilot.py`, `run_repeated_pilot.py` | `phase1_build_env_collision.sbatch`, `phase1_build_ood_control.sbatch`, `run_pilot.sbatch`, `run_ood_control.sbatch` |
| E2 | canonical | `path3_oft_compare.py` | `run_pilot_base_matched.sbatch`, `run_pilot_oft.sbatch`, `run_pilot_openpi.sbatch` |
| E3/C3/C4 | canonical | `probe_selfreport.py`, `probe_selfreport_analysis.py`, `probe_selfreport_capture.py`, `probe_dump_T5.py`, `probe_horizon_relabel.py`, `wall_directed_braking.py` | `probe_selfreport.sbatch`, `probe_selfreport_oft.sbatch`, `probe_selfreport_pi0.sbatch`, `probe_selfreport_pi0_ae.sbatch`, `wall_directed_braking.sbatch` |
| E5/C7/C8 | canonical | `phase3_intervention.py`, `phase3_make_figures.py`, `probe_shield_sweep.py` | `phase3_intervention.sbatch`, `phase3_intervention_expanded.sbatch` |
| E6/C9/C10 | canonical | `phase2_glass_prototype.py`, `probe_glass_capture.py`, `probe_glass_analysis.py`, `probe_glass_inject.py`, `probe_joint_transfer.py` | `phase2_glass_prototype.sbatch`, `probe_glass.sbatch` |
| E7/C11 | canonical | `phase3_steering.py`, `phase3_steering_diag.py`, `phase3_steering_figures.py` | `phase3_steering.sbatch`, `phase3_steering_diag.sbatch` |
| E8 | canonical | `phase2_witness.py` | `phase2_witness.sbatch` |
| E9 | historical / existence demo only | `phase2_task_witness.py`, `phase3_detour_handoff.py`, `render_d70_detour.py`, `phase2_lowwall_validity.py` | `phase2_task_witness_d62.sbatch`, `phase2_task_witness_diag.sbatch`, `phase2_task_witness_prod.sbatch`, `phase3_detour_handoff.sbatch`, `render_d70_detour.sbatch`, `phase2_lowwall_validity.sbatch` |
| E10 | retained negative / one-off diagnostic | `phase2_build_nowall.py`, `phase2_run_nowall.py`, `phase2_obj_collision.py`, `phase2_recon_indist.py`, `phase2_grasp.py`, `phase2_grasp_build.py`, `phase2_grasp_recon.py`, `phase2_grasp_run.py` | matching `phase2_*` sbatch files |
| E11/C13 | canonical zero-GPU analysis | `task_phase_confound_analysis.py` | none; reads existing hidden/meta only |
| E13 | complete / frozen prompt-scope result | `careful_prompt_eval.py`, `analyze_careful_prompt.py`; fixed templates in `crashbench/prompts.py` | `run_careful_prompt_wall.sbatch`, `run_careful_prompt_glass.sbatch`, `analyze_careful_prompt.sbatch`, `submit_careful_prompt.sh` |
| E14 | immutable historical acceptance smoke; use source at commit `7bb6d7d`, never current v2 wrappers | historical v1 behavior of `prepare_glass_recovery_placements.py`, `collect_glass_recovery_pairs.py`, `replay_glass_recovery_pair.py` | historical `glass_recovery_smoke.sbatch` at `7bb6d7d` |
| E15 | current v2 learned-recovery line; code complete, real pilots pending, no learned result | `capture_glass_nominal_source_traces.py`, `prepare_glass_recovery_placements.py`, `audit_glass_core_artifacts.py`, `run_glass_avoidability_frontier.py`, `collect_glass_recovery_pairs.py`, `replay_glass_recovery_pair.py`, `train_glass_recovery.py`, `eval_glass_recovery.py`, `analyze_glass_recovery_eval.py`; schema/model/runtime in `crashbench/glass_recovery_*` | `glass_recovery_smoke.sbatch`, `submit_glass_recovery_smoke.sh train\|evaluate`; accepted manifests and protocol SHAs required |
| utility | helper | `mp4_to_gif.py` | none |
| deprecated selected headline | deprecated | `headline_suite.py` | none; reproduces only the historical artifact |

Environment/verification helpers — `install_openvla_env.sh`, `install_openpi_env.sh`,
`install_openvla_oft_env.sh`, `install_openvla.sbatch`, `verify_openvla.sbatch`,
`verify_openvla_oft.sbatch`, `verify_openpi.sbatch`, `submit_next_round.sh`,
`submit_wall_directed_braking.sh`, and `run_prompted_careful.sbatch` — are operational
helpers, not paper experiment identifiers. Consult `setup/README.md` and preserve
their historical paths until a separately reviewed migration.
