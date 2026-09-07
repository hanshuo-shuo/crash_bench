# Pre-submit validation — 2026-09-07

- Full local suite: 543 passed (16.01 s).
- After adding per-step RNG recording: targeted retest/neutral tests, 9 passed.
- Repository audit passed; sbatch Bash syntax and `git diff --check` passed.
- Actual planned panel: 24 anchors, 12 train sources; every formal attempt and
  source-state content hash matches the split manifest. No D8 outcomes read.
- Fake closed-loop test: matched control at an empty action queue delivers equal
  actions for Base/Refresh; exactly two inference calls for eight execution steps.
- Synthetic evaluation: A winner reverses in B; frozen reference correctly loses
  25 percentage points, while same-result posthoc selection shows no loss.
  Missing/duplicate/wrong-phase records rejected; paired Base CI exactly zero.
- Plot rendered from synthetic data in pytest temporary output (not experiment evidence).
- Quest cached checkpoint path exists; numpy/matplotlib available. Historical
  24-branch neutral job completed in 5:05; new job capped at 3 h with full logging.

Real simulator execution remains unverified until the new Slurm run starts.
