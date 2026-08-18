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

The trigger is the first strict crossing

```text
tau = min { t : max_{o != Base} LCB_cal_t(o) > 0 }.
```

At `tau`, the highest-scoring non-Base option is selected and latched. Detour or
FailSafeHold then runs until its fixed option budget completes or the episode
terminates. The Router is not queried again after the latch.

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
- report builder: `scripts/analyze_dynamic_first_crossing_router.py`;
- Quest job: `setup/dynamic_first_crossing_router.sbatch`;
- submission wrapper: `setup/submit_dynamic_first_crossing_router.sh`.

The default submission is a one-source execution smoke. A paper cohort should
set `CB_P2_MODE=full` and point `CB_P2_PLACEMENTS`/`CB_P2_ROUTER_MODEL` at the
frozen source-disjoint evaluation assets before submission.
