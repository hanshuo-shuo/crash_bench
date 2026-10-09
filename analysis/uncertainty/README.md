# SafeLIBERO action sampling uncertainty

2026-10-09 continuation: the user approved the full series and two A100 workers
for at most24h each, and removed the API dollar cap. The earlier smoke receipts
and their $3 ledger remain historical evidence. Full collection still has600
logical VLM requests and bounded retries; no unrelated API/provider is introduced.
`states.json` freezes the60 identities and `split.json`/`STATS_PLAN.md` freeze
analysis choices before fitting. CPU `preflight.py` verifies actual upstream/asset
bytes and reconstructs the saved static geometry anomaly, with zero policy/API
calls. Do not submit expensive collection until its distance result is understood.
All full inference tensors are retained for independent reduction. The pinned
pi05 policy remains image+prompt conditioned; saved continuous state is not input
to its pi05 suffix. Source work remains confined to this directory.

CPU9257292 reproduced a geometric measurement defect in MuJoCo3.2.3:
`gripper0_finger2_pad_collision` and `red_coffee_mug_obstacle_1_g6` are disjoint
OBBs with a positive35.15mm separating-axis gap, yet the native query returns
−100.48mm at pre-action101; there are no engine contacts. `geometry.py` therefore
uses exact vertex/face and edge/edge Euclidean distance for separated box pairs,
and signed minimum-translation SAT depth for overlapping boxes. Other convex
geometry uses the native collision representation. This only changes the
diagnostic measurement; official displacement crash, policy/QP/controller actions
and original dynamics stay fixed. The old smoke's `min_dist` is retained as
unvalidated measurement evidence and is excluded from Task2 fitting.

Initial smoke boundary: one timing rollout, then two scenes × two seeds × nominal/full AEGIS smoke; one A100, 8 CPUs, 64 GB, 30-minute Slurm hard cap. M=8 and the fixed `1e-6` numeric tolerances stay fixed. The2026-10-09 full-series continuation is described above and in `STATS_PLAN.md`.

`config.yaml` is valid YAML written in its JSON subset. Every scientific parameter and spend/resource limit is frozen there. Run roots are immutable and contain a complete published source archive, source file hashes, upstream and code commits, configuration, seeds and Slurm receipt. The existing Quest source checkout is never merged or overwritten.

`serve.py` keeps ordinary native `Policy.infer` actions and its one RNG split per request. `sampling.py` performs one prefix prefill for the same observation and maps an eight-sample batch over the shared KV cache with native B=1 suffix kernels, integrating the native continuous flow with independently hashed Gaussian noise seeds. This is a compiled shared-prefix batch interface, with sequential suffix kernels rather than eight-wide GEMMs. It does not promise suffix parallel speedup. All native input/output transformations run once per sample; only the valid seven output action dimensions enter metrics. No token logits are used. Fixed-noise native serial versus shared-prefix batch equality and bracketed native actions must pass before diagnostic inference continues. `SAMPLING_VALIDATION.json` preserves failed measurements too.

The initial B=8 suffix implementation in commit `5e20522`, job `9085604`, failed the frozen tolerance: maximum physical-output difference `0.002929103`, while production actions and RNG were unchanged. That root is retained with a complete 157.49-second/300-action nominal timing trace and zero API spend. The engineering repair keeps M=8, native BF16 precision, noise seeds and tolerance unchanged, hoists the single prefix cache out of `jax.lax.map`, and uses native B=1 matrix shapes. Job `9086749` passed the retained-input model probe with exactly zero serial/shared-prefix difference, but failed independent fresh-environment trajectory equality: six RGB channel values differed by one gray level before action 226, after 225 identical physical actions. Its traces and zero-spend ledger remain intact.

The user approved strict same-input/RNG validation on every infer. The native production action is computed before diagnostics, computed again after diagnostics using exactly the original input/RNG, compared with `array_equal`, and the production RNG is restored. Every actual raw queue command is checked against its returned chunk, and live physical bytes must remain unchanged before execution. Reference-check time is reported separately from M=8 sampling. Independent fresh-environment trajectory comparison is retained as a warning because renderer RGB differences can precede action divergence; original RGB is never substituted. This is a conditional noninterference guarantee, not a claim of byte-identical trajectories across fresh renderer environments.

The observer adds data recording around the original pinned evaluator. It keeps its five executed actions per ten-action chunk, 20 settling actions, horizons, obstacle displacement collision proxy, collision continuation, native success logic and full six-axis AEGIS QP with unchanged gripper. `geometry.py` computes the signed distance using a separate MuJoCo data object and verifies live-state nonmutation. Nominal snapshots are save-only; arbitrary mid-rollout restoration remains unvalidated.

`worker.py` makes at most four fresh VLM logical calls in each smoke root, using the unchanged GLM-4.5V prompt and pinned Z.AI routing. It reads the pre-existing private key only when calling the existing transport. The shared campaign ledger allows at most $3 including actual costs and unresolved failed-attempt reservations. Each attempt reserves $0.10 before sending; only listed transient HTTP errors get one additional reserved attempt. Provider metadata is inspected first. Full usage/token metadata and raw responses are preserved. A key is never copied to the source, manifests or logs.

`runtime.py` records per-action traces and per-inference inputs/chunks, then `export.py` writes/readbacks `data/rollouts.parquet`, metrics and a metric-definition README. Inference measurements are held over the five executed actions and marked with `infer_boundary`; no task-2 fitting or prediction score is performed. The baseline timing rows are excluded. Never treat a failed/incomplete job as a scientific outcome.

Local checks: `python3 -m unittest discover -s analysis/uncertainty -p 'test_*.py' -v`, existing `tests/`, `git diff --check`, and `bash -n analysis/uncertainty/run.sbatch`. The integration source check uses the pinned evaluator read into `/tmp/crashbench_uncertainty_main_aegis.py`; native JAX/MuJoCo validation belongs on the allocated compute node, never a login node.

Publish a clean tested commit, run `scripts/quest_sync.sh check`, fetch only that branch through `quest_sync.sh exec`, archive the exact commit to a fresh `.../uncertainty/<timestamp_commit>/source`, write `SOURCE_COMMIT`, then invoke that archive's `launch.py` using the existing OpenPI Python. The launcher validates every archive file against Git, submits one held job, starts a bounded network worker in the project tmux session and releases only that job after worker readiness. Engineering repair uses a new source commit and new root; the campaign ledger and failed root remain intact.

The launcher accounts for every prior allocation in the same campaign using terminal `sacct` receipts. Remaining whole minutes are the floor of `(1800 - prior allocated seconds)/60`, so engineering repairs share the original cumulative 30-minute ceiling. Unknown or active accounting prevents submission. The first two jobs used 361+362 seconds, leaving a maximum 17-minute third job.

## Completed bounded smoke (2026-10-08)

Published compute commit `6ca85c909ebbe6b6a347b833d642e242835193f9`, Slurm `9089140`, root `/projects/p33100/siosio/crashbench_safelibero/uncertainty/20261008T042856Z_6ca85c909ebb`, completed with exit `0:0` in 883 seconds. Including the two retained failed jobs, actual allocation was 1606 seconds (26m46s), below the cumulative 30-minute ceiling. The frozen Quest checkout remains at `465f112ea18bf12204c2c87f1df99c75dd67bd4a`.

The eight smoke rollouts produced 1792 action rows and 360 diagnostic inferences. Every inference passed strict same-input/RNG native action equality, and every actual queue command matched its returned chunk. Fixed-noise shared-prefix versus serial sampling had maximum difference zero at unchanged `1e-6` tolerances. Independent NumPy/Parquet audit recomputed sample std, shifted churn, applied command norms, collision/TTC/outcomes and paired settled-state identity. All four AEGIS rollouts enabled the original QP; all nominal rollouts left it off. The two Spatial seeds crashed under both arms, and the two Object seeds safely succeeded under both arms; these eight runs do not support method-performance conclusions.

The latest no-diagnostic timing rollout took 105.09 seconds for 300 actions, after a startup model probe warmed compilation. The first retained cold timing took 157.49 seconds. Across smoke inferences after each rollout's first infer, median native time was 0.0760s, extra M=8 diagnostic time 0.2552s, and separate strict reference-check time 0.0750s. Sampling uses shared prefix and B=1 suffix kernels; these measurements do not demonstrate eight-wide suffix execution. Sampled allocated-GPU memory peaked at 13244 MiB on an A100-SXM4-80GB (2-second sampling).

Four fresh Z.AI calls, with no retry, cost `$0.00898748` total: 5674 prompt and 3485 completion tokens. All charges settled; no unknown reservation remains in the shared `$3` campaign ledger.

Fresh-environment trajectory equality remains false: before action 11 of the latest pair, three RGB channels differed by one gray level, while state, wrist image, prompt and RNG stream were equal. The earlier retained repair first diverged at action 226. Neither warning is hidden or converted into a claim of byte-identical renderer trajectories.

Local smoke delivery is `analysis/uncertainty/data/rollouts.parquet`, with its metric README, metrics, audit and provenance receipts. Full traces, snapshots, requests/responses and all nine original videos are retained at `results/uncertainty/20261008T042856Z_6ca85c909ebb/` and the immutable Quest root. Data are ignored by Git. M remains8. Full-series authorization followed on2026-10-09; no full outcome is inferred from this smoke.

The original proposal extrapolated1200 rollouts at the slowest smoke duration131.41s to43.8 GPUh, and600 AEGIS calls at the four-call mean to about$1.35. The user subsequently approved2A100×24h and removed the API dollar cap. These remain forecasts from two states; milk, provider latency and richer capture may differ. `full_launch.py` submits two held600-case shards; the first eight matrix cases form the stage smoke, and only a passing retained gate releases shard1. Each inference tensor is independently reduced during export. Cases checkpoint before the480s allocation margin; partial collection is preserved and does not enter primary fitting. Real submission/progress receipts live in unique Quest output roots, not this static document. Live Quest source stays unchanged.
# Task2 execution

`analyze.py` rejects anything other than the complete, hash-checked1200-case
independent audit. The first analysis smoke evaluates the two preselected heldout
states in `config.yaml` at repeats0/1, while using the full frozen train population
for every calibration/reference. The full pass must reproduce all smoke train
models, thresholds and reference medians byte-for-byte. No method is tuned from
smoke or heldout results. `analysis.sbatch` runs on2 CPU/8GB for at most60 minutes,
with no model inference or API calls.

Metric tables retain the predefined raw-score AUROC as `AUROC`, and also expose
`calibrated_AUROC` computed from the fixed train-fit probability. This makes any
slope reversal or probability ties visible. BA/Brier use calibrated probability.
Completion is evaluated on individual rollouts; the first-five landmark excludes
outcomes known before action21. Display calibration bins are ten equal-width
probability intervals, fixed before fitting. Every eligible/censored denominator
and bootstrap undefined count is written alongside estimates.
