# Storage interruption and bounded completion — 2026-09-07

Job 5680180 (commit `1dfc6046ae9cc70155ed55f0e39d48ad06eac3b9`)
failed after 55:43 with `OSError: [Errno 122] Disk quota exceeded`.
Raw traces were mistakenly written beneath the Quest home checkout rather than
project storage. Filesystem free space checked before submission was not the
user's home quota; that preflight was insufficient. The run occupies about 43 GB.

The user asked to continue the plan. This is an explicit, bounded engineering
completion of the same panel, replacing the initial plan's no-retry default for
this storage interruption only. No new source, option, seed, threshold, or model
is introduced. No completed branch is repeated; no A decision is recomputed.
There is no automatic retry loop.

- A: 192 complete branches, selection frozen before B.
- B: 139 complete branches; missing execution suffix contains 53 cells.
- First missing cell: `b17/B_r5_o1`; its truncated raw trace is retained.
- Frozen A SHA256:
  `d4bfbeb761b870b56737566cc0d8bfc13ec1422a97b97281a4653fa12ff345bf`.
- New outputs go to `/projects/p33100/siosio/crashbench_selection_retest/`;
  the checkout contains only stable `results/selection_retest/` symlinks.
- Preserve old raw outputs. Verify all 331 completed trace hashes, all 24 bundle
  hashes, A input/freeze hashes and original checkpoint file hashes on a compute
  node before any additional rollout. Refuse any unexpected missing-cell pattern.
- One A100, 8 CPU, 48 GB, 1-hour cap for exactly 53 valid B completions. Final
  panel still has 384 scored branches, plus one retained, unscored partial attempt.
- Code: `scripts/expansion/resume_selection_retest.py`; submit entry:
  `setup/selection_retest_resume.sbatch`. The original rollout implementation is
  unchanged. Analysis uses original A and combined B with explicit trace locations.

The interruption creates a runtime/process boundary inside B. The Base and
Refresh executions for `b17`, repeat 5, span that boundary. Original pairing is
retained and disclosed; do not describe the combined run as one uninterrupted
execution block or claim this rules out runtime confounding. Report a diagnostic
excluding that one paired repeat if conclusions depend on it. Do not select
whether to retain it based on the direction of the result.

Local test: resume accepts only an exact execution prefix, rejects holes and
repeated outcomes, and returns only the missing suffix. Existing analysis and
logger tests also pass (6 tests in this file).
