# CrashBench agent instructions

For any task that reads, changes, runs, or submits work on Northwestern Quest:

1. Read and follow `QUEST_WORKFLOW.md` and the “从本机直接提交到 Quest” section of
   `setup/README.md` before using SSH.
2. The user establishes `/tmp/quest.sock`; then run `scripts/quest_sync.sh check` before any
   other remote command.
3. Treat `$HOME/crash_bench` with GitHub repository `hanshuo-shuo/crash_bench` as the only
   valid Quest checkout. Never infer project identity from the socket name, and never touch
   another project's directory or tmux session.
4. For one sbatch file, use `scripts/quest_sync.sh submit setup/<job>.sbatch`. For a maintained
   `setup/submit_*.sh` pipeline, first use `scripts/quest_sync.sh push`, then run the wrapper
   through `scripts/quest_sync.sh exec`.
5. Submit only a clean, tested, published Git commit and keep the Quest checkout clean. Stop
   and report any mismatch or divergence; never repair it with force pull, hard reset, or a
   source-tree rsync.
6. Use Slurm account `p33100`; use `gengpu` for GPU work and `short` only for CPU jobs. Never
   perform compute-heavy work on the login node.
7. Do not overwrite frozen results. New runs need a new output path and recorded commit,
   configuration, seed/repeat, and Slurm provenance.
