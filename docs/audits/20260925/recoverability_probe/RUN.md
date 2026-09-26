# Quest execution record

- User authorized `codex/recoverability-probe` to resolve the manuscript/probe
  history divergence while retaining the original local edits.
- Published experiment commit: `7801087aa253cb173f507903ea5e86a5228ceefc`.
- Manuscript base: `c4497cfc5d5009605e2e1d5aaf4e82482ae86aa1`.
- Submission: `scripts/quest_sync.sh submit setup/recoverability_hidden_probe.sbatch`
  from the clean managed worktree at
  `/Users/hanshuo/.codex/worktrees/recoverability-probe/crash_bench`.
- Quest checkout: `/gpfs/home/shv7753/crash_bench`, repository identity verified;
  fast-forwarded from `3a2e51c` to the published execution commit. The sync tool
  advanced the existing Quest branch `codex/odur-repair`; the published source
  branch for this task is `codex/recoverability-probe`. No force operation used.
- Job: `7454767`, `p33100`, `short`, 4CPU, 8GB, 30-minute cap.
- State: `COMPLETED`, elapsed `00:00:29`, exit `0:0`; batch `MaxRSS=172868K`.
- Numerical start: `2026-09-26T04:40:25.255407+00:00` (September25 in Chicago).
- Log: Quest `cb_recoverability_probe_7454767.log`.
- Output: `results/recoverability_hidden_probe/7801087aa253_7454767/`.
- 7 synthetic tests passed before execution; all41 evaluation folds and615
  fixed target/input fits completed, followed by2000 source bootstrap draws
  per prespecified comparison. No new model-family/threshold selection.
- Frozen inputs and outputs were hash-verified after retrieval. Historical
  capture and prior experiment directories were not changed. D8 not loaded.

The original desktop checkout remains on `codex/odur-repair` with its own
two unpushed probe-preparation commits and the user's pre-existing manuscript
and presentation edits. Those two probe commits were cherry-picked onto the
new experiment branch; do not force-push the original branch or overwrite its
user edits when resuming work.
