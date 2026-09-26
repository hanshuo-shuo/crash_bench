# CrashBench / SafeLIBERO agent instructions

Read docs/CURRENT.md and docs/REPRODUCTION.md first.

On 2026-09-26 the user authorized keeping the eight results in retained/,
archiving other CrashBench work, and starting reproduction of nominal pi0.5
and full AEGIS on the unchanged public SafeLIBERO benchmark. The user has an
OpenRouter API key and wants to be notified when it is needed. Never print,
commit or place that key in a job log. Do not silently substitute another VLM.

The old experiment plans live on codex/archive-crashbench-20260926 and are
historical, not an active run queue. No new learned intervention, controller,
benchmark task, confirmatory study or threshold search is authorized.

Before any Quest SSH use read QUEST_WORKFLOW.md and setup/README.md, then run
scripts/quest_sync.sh check. Only $HOME/crash_bench / hanshuo-shuo/crash_bench is
the valid project checkout. Do not touch another project's files or jobs.
Submit only a clean, tested, published commit through scripts/quest_sync.sh.
Use account p33100, GPU partition gengpu, CPU partition short. No heavy work on
login nodes; only downloads, Git and lightweight checks there. Compute nodes
are offline: stage assets first. Never force pull, hard reset or rsync source.
Keep historical raw results recoverable. Every new run needs a unique output
root, code/upstream commit, configuration, seed and Slurm provenance.

Budget update (2026-09-26): the user has $100/week for OpenRouter alone. Plan
baseline reproduction within $5/week and preserve $95 for their method. This
smoke batch has used two calls ($0.00279368); no more paid calls in this batch.
Before any bulk API worker, inspect usage metadata, estimate the batch and add
a spend guard. No provider-side key cap has been configured by this task.

Full-run authorization (2026-09-26): the user now authorized starting the full
nominal pi0.5 / full AEGIS official matrix, subject to $5 OpenRouter spending for
this reproduction (including earlier smoke calls). Use setup/submit_safelibero_full.sh
only after publishing and syncing. See docs/reproduction/BATCH_RUN.md. The batch
uses at most two GPUs and stops its own arrays on budget or infrastructure failure.
While its arrays are active, do not change or sync the Quest source checkout:
queued jobs require the exact frozen commit. Read the batch receipt/status under
results/safelibero_batches/; do not infer progress from this static document.
