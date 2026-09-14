# Fresh-source Quest execution — complete

- Execution commit: `2c5e6163a6bc2644072bc7328279d49445814807`.
- Job:6243668, account p33100, partition gengpu, one A100,8 CPU,64GB,4-hour cap.
- Main job and batch: COMPLETED, exit0:0, elapsed02:40:00, MaxRSS19367292K.
- Raw root: `/projects/p33100/siosio/crashbench_repeat_value/2c5e6163a6bc_fresh_6243668`.
- Result link: `results/repeat_value/2c5e6163a6bc_fresh_6243668`.
- Actual:12 authoring rollouts,36 prefixes,32 candidates,128 A +128 B +128 C
  branches. Four setup-terminal conditions are retained as shared outcomes.
- Pre-submission validation:16 targeted synthetic unit tests, Python syntax,
  shell syntax and repository whitespace checks. Local/upstream/Quest were clean
  and synchronized at the execution commit before submission.

A/B/C ran on qgpu2007 with PIDs4030528/4045105/4059533 and the same A100-SXM4-80GB,
checkpoint and software. Choice/forecast locks preceded the next process; final
analysis verified all hashes and ordering. Only progress logs were inspected until
all C executions were finished. Quest remained at2c5e616 through main completion.

Final collection appeared slow, so a read-only process-inspection step was added
within the existing allocation. That diagnostic step6243668.0 exited127 because
rg was absent on the compute node. It executed no model or simulator action and
changed no research artifact. The main job and all scientific stages completed
normally. A later inspection attempt found the job had already expired; it did
not start a new scientific execution. No rollout was rerun or replaced.

The reviewed metadata and original analysis are in `evidence/`. Raw A/B/C event
and physics arrays, bundles and per-step traces remain at the original Quest root.
The small review archive was created only after completion. The source exposure
ledger was subsequently appended locally with twelve completed-development
identities, preserving all previous bytes; see `EXPOSURE_RECORD.json`. The frozen
probe contract retains the pre-execution ledger hash and original commit.

## Presentation and external interface follow-ups

After main completion, Quest advanced cleanly to published
`6b91cee1d337e0bc28e2f792b8ccac44fe44c868`.

- CPU figure job6246664: COMPLETED,23seconds, exit0; conference-size plots,
  all-source support summary and compact original terminal fields.
- GPU illustration job6246667: COMPLETED,38seconds, exit0; nine reconstructions
  from existing e15/e18/e21 C terminal states, exact physical-state equality,
  zero new policy queries or environment actions.
- External SPR prepare6246660 and dependent inference6246661: tracked separately
  under `../external_recovery/`; this engineering check is not a recovery study.

The official-format PDF is generated locally from completed Quest results, with
no local scientific computation. Original figure/result folders remain unchanged.
