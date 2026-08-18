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
