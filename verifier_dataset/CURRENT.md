# FD1000 collection status — 2026-10-14

The user authorized 1,000 reliable records, broader original scenes, witnessed
positives, certified negatives, grounded reasoning targets and shortcut audits,
then verifier training only after data acceptance. New API spending is capped at
$20. The workflow is implemented in this directory; this is not a claim that
1,000 scientific training records already exist.

## Completed evidence

- Inventory job **9892280 COMPLETED/0:0**: 1,200 instrumented rollouts, 391
  proxy-safe-success tapes, 60 original initial identities, 33 with proxy success.
  This pool has only three task compositions and two suite families. Every
  inventory row remains UNKNOWN until independent evidence is checked.
- Broader-source inventory **9898692 COMPLETED/0:0** checked the full official
  3,200 outcomes and selected 20 additional initial states across four Goal tasks,
  five per task. Selection uses historical proxy success for collection efficiency;
  it supplies neither new executions nor contact-safe labels.
- Repaired 12-source replay **9902948 COMPLETED/0:0**, 4m55s: all original
  per-action physical hashes and success flags matched; canonical commands are
  within the native action bounds. Four distinct Spatial sources have zero
  prohibited contacts, supplying **four witnesses / 40 checkpoint candidates**.
  The other eight cases are retained and not positive-labeled under the strict
  contact rule. Job success alone does not mean scientific dataset acceptance.
- **Zero new paid calls.** Public endpoint metadata was fetched without a key or
  simulator payload and passed the pinned GLM-4.5V / Z.AI pricing guard. The
  proposed four-case GPU collection reserves at most $0.80 for eight attempts
  within the shared $20 ceiling; no paid worker or GPU collection was launched.

Local receipts and checksum-verified copies are under
`results/verifier_dataset/`. The corresponding Quest roots are under
`/projects/p33100/siosio/crashbench_safelibero/verifier_dataset/`:

| Stage | Root suffix | Frozen source |
|---|---|---|
| Inventory | `20261014T053334Z_inventory_0238a3ac1606` | `0238a3ac1606` |
| Broader sources | `20261014T055536Z_source-plan_28670c444e49` | `28670c444e49` |
| Passing 12-case replay | `20261014T060755Z_replay-smoke_bc55dec076a6` | `bc55dec076a6` |
| Spatial CPU batch | `20261014T061523Z_replay-spatial-batch_454f29b3c8fd` | `454f29b3c8fd` |

## Active unaffected work

CPU array **9906479** validates the existing 124 Spatial proxy-success tapes in
eight shards, at most four running concurrently. These are 124 distinct action
tapes, **ten original initial states and one task composition**, including 63
AEGIS and 61 nominal executions. Read its actual `shards/*/PROGRESS.json`,
`REPLAY_COMPLETE.json` and Slurm terminal accounting; do not infer completion
from this paragraph. Keep the live Quest source frozen until the array ends.

The tested `export-candidates` stage is published but not yet run. It requires
every source shard to finish normally, rechecks witness hashes, command legality,
contact records and exact replay, strips future actions/diagnostics from student
inputs, verifies PNG pixel equality and removes duplicate full inputs. It reports
image-pair and original-source counts separately. Exports remain positive
**candidates** with `training_eligible=false`, no invented rationale, and explicit
pending observability/split/negative/shortcut gates.

## Pending decisions and closed gates

1. The replay initially used the strict rule prohibiting protected-object contact
   with **any other body with a motion DOF**, including movable support pedestals.
   Some Object initial states have native `box_small_base_1_g2` support contacts;
   some also have actual target contacts. Earlier SC-CONTACT/RA protocols instead
   monitor robot/target versus protected objects and permit native support.
   The user has been asked which definition FD1000 should use. Preserve v1 and
   its results; any change requires an explicit versioned contract, new checks
   and identical treatment across classes. Existing strict contact-free Spatial
   work proceeds independently of this decision.
2. Automatic approval review rejected the proposed paid GPU/API launch, stating
   that explicit approval of the external GLM-4.5V / Z.AI image/prompt egress and
   notification of key use were needed. The user has been shown the exact model,
   payload scope, existing private-key usage and $0.80 first-batch / $20 campaign
   limits. That approval question is pending. **The rejected operation executed
   no paid request and created no collect-smoke job/root.** Do not retry it until
   the missing approval arrives. The key is never copied into source or logs.

Accepted complete scientific dataset rows: **0**. New matched certified negatives:
**0**. Final grounded reasoning targets: **0**. Verifier training: **not launched**.
The source-coverage, paired-negative, observability, explanation and shortcut
requirements are still open. Do not rename witness/checkpoint counts as accepted
training rows or use policy failures as infeasible labels.

## Preserved engineering failures

Jobs 9892943 and 9895099 aborted during native shutdown (134); 9895099 wrote a
nominal JSON pass after 12 scientific replays but its failed Slurm exit correctly
prevented acceptance. Isolated job 9898144 repeated the error and was canceled
after three failed child exits. Single-case jobs 9898693 and 9898912 also failed.
All roots remain intact. Restoring the simulator method descriptor and isolating
workers did not by themselves solve shutdown. Pinning the established headless
GLX import configuration produced **9899323 COMPLETED/0:0**, followed by the
clean 12-case result above, with unchanged per-action physics. No failed job was
turned into a label and no forced-success process exit was used.

The failed local managed-worktree sync was rejected by approval review; it did
not execute. That unused worktree was archived. The unrelated literature owner
moved its own untracked files out of the shared repository, after which work
continued through the original approved local checkout and `quest_sync.sh`.
