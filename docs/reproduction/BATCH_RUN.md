# Official baseline matrix: launch protocol

The user authorized this on 2026-09-26 after both smoke arms completed. Scope:
nominal pi0.5-LIBERO and full AEGIS, four official suites × two safety levels ×
four tasks × 50 initial states = 1600 episodes per arm, 3200 overall. This is
baseline reproduction, with no learner, controller redesign or task expansion.
Smoke results are archived separately and not mixed into the matrix summary.

## Execution and provenance

64 Slurm cells each run one arm/scenario and all 50 episodes in the official
order. NumPy/environment seed 7, settling 20, replanning 5, horizons 300/550,
original collision proxy, pi05_libero and upstream 2457feed5968 remain fixed.
A new upstream policy server starts with its native JAX seed 0 for each cell;
its RNG advances normally across that cell's episodes. This is not exact-state
counterfactual pairing or paired random noise across policies.

The first array has cells 0 and 32 (Spatial/I/task0, one per arm). The remaining
62 cells depend on both completing successfully. Each array has concurrency 2;
the dependency prevents overlapping arrays, so total use is at most two A100s.
Each cell requests 8 CPUs, 64 GiB and 6 hours, p33100/gengpu. At the observed
smoke runtime, a full sweep is likely to take many hours to over a day, plus
queue time; no completion time is promised from one sample.

The checkout must remain clean and at the published batch commit until all its
jobs terminate. Jobs verify that commit before compute. Immutable roots include
batch ID, cell index, code commit and Slurm job ID. Per-episode JSON is atomically
published after its video is written; partial results survive an interrupted job.
The aggregator rejects mixed commits, duplicate cells/episodes and incomplete
runs incorrectly marked complete. Partial rates are explicitly marked partial.
Reports use completed action counts; they do not silently equate these with the
upstream successful episode's zero-based t counter.

`batch.json` is the authoritative matrix and launch receipt. `status.json`
contains partial counts/rates; `COMPLETE.json` requires all 64 cells / 3200
records. `STOP.json` records a budget, network or infrastructure stop. Failed
jobs are not scored as scientific failure outcomes. The worker cancels only its
two recorded array IDs on failure. Automatic retries are disabled to avoid
hidden charges or silently changing the server RNG sequence. Preserve failed
attempts and plan any recovery from their recorded progress.

## $5 OpenRouter budget

The user's total OpenRouter allowance is $100/week. This baseline run is limited
to $5, including $0.00279368 already reported for its two smoke calls. Previous
batch charges and unresolved reservations also carry into any later launch;
restarting does not grant another $5. No other project or key limit is changed.

A single writer owns an exclusive file lock. Before every paid call it durably
reserves $0.10; after a response, it replaces that reservation with reported
usage.cost. Unknown charges retain the reservation and stop. A remaining balance
below the reservation stops before the call, so some of the $5 may remain unused.
No more than 1600 paid requests are permitted in this batch. Responses are cached
by exact image, prompt and transport settings. Duplicate cached inputs are reused.

Observed Z.AI pricing on 2026-09-26: $0.60/M prompt, $1.80/M completion tokens,
65536 context, 16384 maximum completion. Even independently filling both bounds
costs $0.0688128, below the $0.10 reservation. The worker periodically verifies
these bounds through the public model endpoint and includes provider price caps
on each request. A native 16384-token ceiling is explicit; no shorter reasoning
budget is used. The $5 guard relies on the provider honoring these advertised
limits and price caps; it is not an account-wide billing control or GPU budget.
See [OpenRouter provider price caps](https://openrouter.ai/docs/guides/routing/provider-selection#max-price)
and [response usage](https://openrouter.ai/docs/api-reference/overview).

At the first uncached sample cost, 1600 calls plus smoke would be about $2.90.
This is a forecast, not a promise: scene complexity, reasoning length and service
failures can prevent completion under $5. The worker is bounded to seven days
and is independent of the local SSH socket. It performs only networking, metadata
checks and Slurm control on the login node; simulation and all inference other
than the remote GLM API run in GPU jobs.

## Local verification

17 tests passed before submission, covering fixed source/horizons/scoring, exact
matrix coverage, incomplete/mixed result rejection, budget blocking before any
network call, durable unsettled charges, exclusive locking, provider price/context
changes, truncated response rejection and cancellation scoped to our arrays.
Shell syntax and diff whitespace checks passed. The first full cells remain the
runtime acceptance check for the multi-episode path.
