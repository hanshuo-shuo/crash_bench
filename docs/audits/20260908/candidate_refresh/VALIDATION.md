# Pre-submit validation — 2026-09-08

- Full suite: 554 passed in 11.51 s using the existing CrashBench repair venv.
- Initial base-Python check passed 15 tests but could not run the legacy plot test
  because matplotlib was absent. The complete suite in the existing venv passed.
- Repository audit, sbatch syntax and whitespace checks passed.
- Raw historical census verified: 9 stale configurations / 8 sources, 8 Base
  100-step noncompletions, 1 Base accident; late Refresh steps 96/96/99/99.
- New synthetic tests exercise success at 99/100/101/200, no success at 200,
  accident precedence over simultaneous success, environment early termination,
  dual-horizon A-only selection and Base catch-up. No simulated new task data
  was used for selecting this plan.
- Quest identity and clean checkout checked at 0b3eb75a5e09b98dc5289cae9b6b230fd8569e9b.
  Project storage had 692 GB free; new full traces target project storage.
- Real simulator execution is verified only after the Slurm job starts.
- The pre-existing untracked 20260907/progress_review package is preserved without
  content changes in a separate commit before the experiment commit.
