# P2 dynamic first-crossing Router

## Paper question

P0/P1 establish that intervention value depends on both option and timing. P2
turns that matched-state result into the actual deployable policy: **when does
the Router first have enough evidence to leave Base, and which option does it
then execute?**

The primary method begins at episode reset. At every Base action opportunity it
computes

```text
Delta_hat_t(o) = U_lambda(o | h_t) - U_lambda(Base | h_t)
LCB_cal_t(o) = Delta_hat_t(o) - delta_cal
```

where `delta_cal` is the already-frozen calibration-set conservative margin.
This is deliberately named a calibrated lower score: the current frozen model
does not contain a bootstrap ensemble or posterior variance.

The primary dynamic point uses the router manifest's predeclared primary
intervention-rate target (`lambda=1`, `target=0.4`). The historical
`lambda=1,target=0.6` display point remains a diagnostic: its frozen margin is
zero, so it is not the conservative deployment point for first crossing.

The first development cohort exposed a repeated-look failure: even a nonzero
pointwise margin eventually crossed on every Base-success control. The revised
P2 boundary is therefore calibrated at the source/trajectory level. For each
original Router calibration source, it takes the maximum raw advantage over
all actions, non-Base options, and Base-success off-path/no-glass trajectories,
then freezes a one-sided split-conformal quantile. The primary sequential point
uses `alpha=0.1` and keeps the same strict first-crossing/latch rule; no warm-up
or hand-authored timing gate is added.

The trigger is the first strict crossing

```text
tau = min { t : max_{o != Base} LCB_cal_t(o) > 0 }.
```

At `tau`, the highest-scoring non-Base option is selected and latched. Detour or
FailSafeHold then runs until its fixed option budget completes or the episode
terminates. The Router is not queried again after the latch.

For causal evaluation, scores are computed in order on the matched Base
reference prefix. The first crossing branches from that exact serialized
simulator/controller/observation state. If no crossing occurs, the outcome is
the identical Base reference outcome rather than a second from-reset replay;
this prevents numerical replay drift from being misattributed to the Router.

## Experimental roles

| Role | Timing | Option choice |
|---|---|---|
| **Dynamic first crossing (primary)** | causal, learned from reset | frozen Router |
| Fixed T-20 Router | actual Base collision minus 20 actions | same frozen Router |
| Realized option Oracle | fixed T-20 matched branches | best realized option |

Fixed T-20 is an **oracle-timing upper bound**, not the method: its trigger time
uses the future Base collision. The realized option Oracle is retained only to
interpret remaining timing-versus-choice error.

## Required episode report

- intervention lead time in actions;
- Euclidean EEF distance from the first trigger state to the matched Base
  post-collision state;
- missed recovery window: Base catastrophes, T-20 Detour succeeds, the dynamic
  policy does not succeed, and its trigger is absent or later than T-20;
- unnecessary early intervention: a dynamic trigger where matched Base
  succeeds;
- selected option and intervention duration;
- task success, catastrophe, and safe noncompletion;
- within-episode contact-force p95/max and cohort p95/max of episode peaks.

The independent statistical unit remains the source state. Conditions and
per-action Router scores are repeated observations, not extra independent
samples.

## Implementation and Quest entrypoint

- runtime state machine: `crashbench/counterfactual_router.py`;
- online capture: `scripts/collect_dynamic_first_crossing_router.py`;
- sequential calibration: `scripts/calibrate_dynamic_first_crossing_router.py`;
- report builder: `scripts/analyze_dynamic_first_crossing_router.py`;
- trajectory morphology audit: `scripts/analyze_p2_trace_morphology.py`;
- calibration Quest job: `setup/dynamic_first_crossing_calibration.sbatch`;
- Quest job: `setup/dynamic_first_crossing_router.sbatch`;
- submission wrapper: `setup/submit_dynamic_first_crossing_router.sh`.

The default submission is a one-source execution smoke. A paper cohort should
set `CB_P2_MODE=full` and point `CB_P2_PLACEMENTS`/`CB_P2_ROUTER_MODEL` at the
frozen source-disjoint evaluation assets before submission.

## Development result

The five-source development execution is complete. Dynamic timing improved
success and catastrophe relative to Base and the fixed T-20 Router, but every
Base-success control also crossed the pointwise boundary. The compact result,
raw hashes, and next-method implication are in
[`results/P2_DYNAMIC_FIRST_CROSSING_DEV_20260818.md`](../results/P2_DYNAMIC_FIRST_CROSSING_DEV_20260818.md).

The final sequential-boundary development run reduces intervention from 100%
to 16.7% and adds a small Pareto improvement over Base, but misses both known
T-20 recovery windows because benign trajectories outrank them. The exact
result and paper decision are in
[`results/P2_SEQUENTIAL_FIRST_CROSSING_DEV_20260819.md`](../results/P2_SEQUENTIAL_FIRST_CROSSING_DEV_20260819.md).

## P2.5 trajectory-morphology closeout

P2.5 tests whether the two missed recoverable episodes were hidden by isolated
control spikes or whether their complete score trajectories also fail to
separate. It reconstructs Detour, Retreat/Hold, max non-Base, candidate-option,
and overall option-argmax sequences from the frozen traces. The primary ranks
contain the four requested groups (11 episodes); the twelfth stable episode is
a Base-safe-noncompletion no-glass trajectory and remains in per-episode
artifacts without entering the treatment/control comparison.

The two missed episodes rank as follows among those 11 episodes:

| Statistic | heldout_0010 | heldout_0019 |
|---|---:|---:|
| raw maximum | 9 | 6 |
| MA-3 maximum | 7 | 9 |
| MA-5 maximum | 6 | 10 |
| MA-8 maximum | 6 | 10 |
| top-5 mean | 8 | 10 |
| excess area | 9 | 11 |
| longest above-margin run | 6 | 11 |

Raw maximum gives missed-treatment-versus-control AUC 0.357; MA-3/5/8 each
give 0.286. Temporal smoothing partially promotes `heldout_0010`, whose T-20
window has an 11-action Detour-above-margin run, but demotes `heldout_0019`,
whose corresponding run is five actions. High control scores are a mixture of
spikes and sustained evidence; several remain above both treatments after
smoothing or accumulation.

No simple temporal statistic is justified for another freeze on this cohort.
The minimum next method experiment is recovery-window supervision. A temporal
value model is deferred until the supervision target, rather than model
capacity, has been tested. This closes P2 threshold and accumulator tuning
without converting the diagnostic into a new project-level GO/NO-GO. The full
report, flat episode table, machine-readable trajectories/rankings, and figures
are in
[`results/P2_TRACE_MORPHOLOGY_AUDIT_20260819.md`](../results/P2_TRACE_MORPHOLOGY_AUDIT_20260819.md).
