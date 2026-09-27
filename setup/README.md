# Quest reproduction entry points

Read ../QUEST_WORKFLOW.md first. The user establishes /tmp/quest.sock; run
`scripts/quest_sync.sh check` before any SSH command. Work only in
$HOME/crash_bench, origin hanshuo-shuo/crash_bench. Stop on an unexplained mismatch;
never force-pull, hard-reset or copy a source tree over it.

## 从本机直接提交到 Quest

Publish a clean tested commit, then use `scripts/quest_sync.sh push` and
`scripts/quest_sync.sh submit setup/<job>.sbatch`. A wrapper pipeline must be
published first and invoked through `scripts/quest_sync.sh exec`.
All GPU jobs use account p33100 / gengpu; CPU jobs use p33100 / short.
New outputs go below /projects/p33100/siosio/crashbench_safelibero/runs/ with
commit and Slurm job IDs. Do not overwrite a completed run.

Preparation order:

1. `scripts/quest_sync.sh push`
2. `scripts/quest_sync.sh exec 'bash setup/stage_assets.sh'` — downloads only
3. `scripts/quest_sync.sh submit setup/build_aegis_container.sbatch`
4. `scripts/quest_sync.sh submit setup/build_aegis_env.sbatch`
5. `scripts/quest_sync.sh submit setup/nominal_smoke.sbatch`
6. `scripts/quest_sync.sh submit setup/prepare_perception.sbatch`
7. Configure OPENROUTER_API_KEY only in the networked calling environment; fill
   the exported request using scripts/openrouter_perception.py.
8. `scripts/quest_sync.sh submit setup/aegis_smoke.sbatch`

These stages are checked individually; do not submit downstream work with missing
artifacts. Baseline score collection is not authorized to hide installation failures.

Local checks: `python3 -m unittest discover -s tests -v` and `git diff --check`.

## Private OpenRouter configuration (when requested)

In your own Quest terminal, run:

```bash
/projects/p33100/siosio/envs/openpi/bin/python ~/crash_bench/scripts/configure_openrouter_key.py
```

This prompts invisibly and writes only ~/.config/crashbench/openrouter.key with
mode 600. Do not paste the key into chat. The bounded login-node API worker can
then serve the exact image requests exported by the offline AEGIS GPU process.
It is capped at two API requests during initial reproduction.

## Authorized full two-arm matrix

After smoke acceptance and budget-guard tests:

```bash
scripts/quest_sync.sh push
scripts/quest_sync.sh exec 'bash setup/submit_safelibero_full.sh'
```

The wrapper records both Slurm array IDs, starts a bounded network worker in the
project's crashbench tmux session, verifies readiness and releases the held arrays.
First run: Spatial/I/task0, nominal and AEGIS, 50 episodes each. Remaining cells
are dependent on both first cells completing. At most two GPUs run concurrently.
Do not launch a second batch or sync source while this one is active. Review
`results/safelibero_batches/<batch>/status.json`, `budget.json`, and any STOP.json.
The full matrix replaces the initial worker's two-call limit with a durable $5
ledger and at most 1600 API requests. See ../docs/reproduction/BATCH_RUN.md.

## Recover a stopped batch

After verifying that its arrays have terminated and publishing/syncing the
transport-only repair, run:

```bash
scripts/quest_sync.sh exec 'bash setup/submit_safelibero_full.sh --recover-from 20260926T233314Z_af6ac874d168'
```

This creates a NEW batch root and reuses only hash-verified complete cells.
Partial cells start again from episode 0; their old records are preserved and
excluded from the new aggregate. All old charges/reservations carry into the same
$5 cap. A recovery must never be implemented by deleting STOP.json or the ledger.
