# Quest run — 2026-09-08

- Slurm job: **5753957**, `cb_candidate_refresh`.
- Published execution commit: `f67e28457a58c557ecccf98cd7c189d0a12150f9`.
- Submitted through `scripts/quest_sync.sh submit setup/candidate_refresh.sbatch`.
- Account p33100, gengpu, one A100, 8 CPU, 48 GB, four-hour limit.
- Initial scheduler state: PENDING; no results available at submission.
- Exact raw output root:
  `/projects/p33100/siosio/crashbench_candidate_refresh/f67e28457a58_5753957`.
- Quest log: `$HOME/crash_bench/cb_candidate_refresh_5753957.log`.
- Expected report: `<raw root>/analysis/RESULTS_ZH.md`, metrics beside it.
- Keep this run root even if interrupted. Do not resubmit the entry or overwrite
  completed branches. Retrieve small JSON evidence and report after completion;
  full bundles remain on project storage with verified hashes.

Follow-up must first check `scripts/quest_sync.sh check`, then inspect only this
job and output root. Confirm 288 branches (or documented exclusions), validate
freeze and trace hashes, inspect control harms and horizon catch-up, and bring
back a Chinese interpretation. No automatic extra experiment or method fitting.

Completed: Slurm COMPLETED, exit 0:0, elapsed 00:54:40; all 288 branches and 18
anchors, no exclusions. Small evidence retrieved to verified_evidence; complete
metrics recomputed locally. Interpretation: RESULTS_ZH.md.
