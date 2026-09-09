# Quest execution log

- Main implementation commit: `a89f755da200c853795e429b5ae34944a08f915f`.
- Job: `5791132`, account `p33100`, partition `gengpu`, one A100, 8 CPU, 96 GB, 8-hour cap.
- Output: `/projects/p33100/siosio/crashbench_detour_benefit/a89f755da200_job5791132`.
- Repository alias: `results/detour_benefit/a89f755da200_job5791132`.
- Entry: `setup/detour_benefit.sbatch` through `scripts/quest_sync.sh submit`.
- Clean local submission clone: `/private/tmp/crashbench-detour-submit-20260909`, same branch and published commit. The main workspace's pre-existing untracked artifacts were preserved.
- Quest was verified clean and fast-forwarded from `f67e284` to `a89f755` before submission.
- Validation: 166 relevant branching/recovery/router tests passed; after adding full no-trigger analysis integration coverage the study-specific suite contains 10 passing tests. Repository audit and Bash syntax checks passed.
- Initial state: PENDING (Resources). No result or scientific conclusion at submission.
- No conditional new-source validation queued.

- Runtime started on `qgpu2009`. First candidate completed all four A branches with full-bundle restore audits passing.
- Compute-node SHA-256 of all four safetensors shards matches the content-addressed cache identities; all file sizes match the audited checkpoint inventory.
- Accidents retain the historical glass-specific tilt/displacement/contact-force predicate, not a global robot-contact safety claim.

- Runtime result directory initially appeared as untracked; added only `/results/detour_benefit/` to Quest `.git/info/exclude`, leaving the executing source commit unchanged. Main checkout ignore rule will be published after execution.
- Project storage check showed 654 GB available; raw outputs reside outside home quota.

## Completed execution, result review pending

Slurm job5791132 is COMPLETED, elapsed01:57:42, exit0:0; batch MaxRSS18913908K. Main analysis files exist on Quest (overall, paired comparisons, per-source, repeat diagnostics, full success curve and Chinese report). The SSH master socket disappeared immediately after this status check; result transfer and the additional existing-trace audit await restored connectivity. No new rollout is needed.
