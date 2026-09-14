# Fresh-source Quest execution

- Frozen implementation: `2c5e6163a6bc2644072bc7328279d49445814807`.
- Job: `6243668`, account `p33100`, partition `gengpu`, A100 × 1, 8 CPU,
  64 GB, four-hour wall cap.
- Raw root: `/projects/p33100/siosio/crashbench_repeat_value/2c5e6163a6bc_fresh_6243668`.
- Repository result link: `results/repeat_value/2c5e6163a6bc_fresh_6243668`.
- Entry: `setup/fresh_value.sbatch`.
- Checks before submission: 16 targeted synthetic unit tests, Python syntax,
  shell syntax and whitespace checks passed. Local/upstream/Quest commits were
  synchronized and clean before submission.
- Status at preparation: running. Source authoring and all 36 prefixes are
  complete. A and B each completed 128 branches from 32 candidates; the choice
  and A/B forecast locks are written. Four no-candidate episodes retain their
  shared prefix outcomes. C is running. A/B/C use separate processes with intervening choice and forecast
  locks. Only progress logs are inspected before final analysis.

Do not synchronize a new Quest checkout during this job: its three runtime
blocks must retain the same published commit. Preserve all raw source attempts,
prefixes, bundles and branch records. Completion and final accounting will be
added after the job finishes.

Prepared follow-ups (do not sync the active Quest checkout before job6243668 ends):

1. Pull this run's `analysis/`, source/anchor identities, phase completion files,
   choice and forecast locks, provenance and overall completion record. The large
   raw A/B/C event and physics arrays stay on Quest.
2. Commit and publish the prepared scripts and completed evidence, then sync Quest.
3. `setup/submit_spr_preflight.sh` prepares an isolated external SPR environment on
   CPU and submits a dependent A100 interface check (two queries, zero simulator
   actions). Public fixed-version weights and precompiled dependency wheels have
   already been downloaded under `/projects/p33100/siosio/crashbench_external_recovery/sprvla`.
4. `setup/iclr_value_figures.sbatch` renders conference-size versions of all four
   figures and extracts compact terminal/provenance fields from fresh A/B/C.
5. `setup/value_examples.sbatch` reconstructs nine terminal-state illustrations
   from the *existing* Detour C traces for e15/e18/e21; no new rollout or query.
6. Integrate actual fresh results, external interface status and final figures
   into the manuscript and Chinese report; generate and visually review the
   official-format PDF with `scripts/paper/build_repeat_value_iclr.py`.

The PDF skill operation marker for this edit has already been executed once.
The current local LaTeX build is in `tmp/pdfs/iclr/`; the deliverable PDF has not
yet been replaced during this continuation. The local workspace contains the
prepared follow-up code and manuscript edits; Quest stays clean at `2c5e616`.
