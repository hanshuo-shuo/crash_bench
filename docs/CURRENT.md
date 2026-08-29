# Current paper state

> **ICLR 2027 gate notice (2026-08-29):** This file records the frozen
> pre-audit paper story. The [strongest-baseline audit](iclr27/BASELINE_AUDIT_RESULT.md)
> is now complete: its formal branch is INCONCLUSIVE and the current
> outcome-decomposition headline is a NO-GO. E16 remains descriptive evidence
> against the specifically evaluated Risk -> Retreat baseline, not evidence of
> superiority over strong scalar-risk or direct-value routing.

**Experiment discovery closed.**

**Paper consolidation active.**

## Frozen thesis

The working title is **Risk Is Not Intervention Value: Counterfactual Outcome
Routing for VLA Safety**.

The paper's organizing result is:

```text
Risk Detection
    !=
Intervention Value
    !=
Reliable Sequential Intervention
```

Failure risk alone cannot determine whether Base, Detour, or Hold is valuable.
E16 shows that option-conditioned counterfactual outcomes support a useful
fresh matched-state routing frontier. P2–P3.2 then establishes the boundary:
statewise/offline intervention-value improvements do not automatically become
reliable sequential first-crossing control.

The repository uses the C0–C14 vocabulary in [CLAIMS.md](CLAIMS.md). C14/E16 is
the headline learned-routing result. C1/C3/C5/C7/C13 provide causal and
mechanistic motivation. P2–P3.2 is a boundary claim, not a new headline method.

## Final evidence map

| Evidence | Final paper role | Status |
|---|---|---|
| E16 / C14 | Fresh matched-state option-routing headline | **POSITIVE / HEADLINE** |
| P2 | Limited fresh sequential improvement | **POSITIVE / SECONDARY** |
| P2.5 | Simple temporal aggregation does not repair score ordering | **BOUNDARY** |
| P3.0 | Dense exact-state recovery-window supervision | **BOUNDARY SUPPORT** |
| P3.1 | Strong offline/source-held-out statewise ranking improvement | **BOUNDARY SUPPORT** |
| P3.2 | Final fresh Direct Router collapse to Base | **NEGATIVE CLOSEOUT** |
| Recovery-window rescue line | No further modeling or evaluation | **CLOSED** |

## Headline result: E16 / C14

The deployed matched-state method is intentionally small:

```text
single-frame PCA-16 hidden + robot state + nominal action
    -> option-specific P(success/catastrophe/safe noncompletion)
    -> U_lambda(option) = P(success) - lambda * P(catastrophe)
    -> Base unless best non-Base advantage > delta
```

Training uses five source states, calibration seven disjoint sources, and
development eight diagnostic-only sources. The architecture and frontier were
frozen before fresh evaluation.

The independent random-reset cohort has eight source states, three matched
conditions per source, and 24 matched decisions. At the displayed
`lambda=1,target=0.6` point:

| Method | Task success | Catastrophe | Safe noncompletion | Intervention |
|---|---:|---:|---:|---:|
| Base | 66.67% | 33.33% | 0.00% | 0.00% |
| Hazard Prompt | 37.50% | 20.83% | 41.67% | 100.00% |
| Binary Risk -> Retreat | 41.67% | 8.33% | 50.00% | 50.00% |
| Always Detour | 70.83% | 4.17% | 25.00% | 100.00% |
| Always Retreat | 0.00% | 0.00% | 100.00% | 100.00% |
| **Counterfactual Router** | **87.50%** | 8.33% | **4.17%** | 58.33% |
| Counterfactual Oracle | 91.67% | 0.00% | 8.33% | 33.33% |

Router versus rate-matched Binary Risk has +45.83 task-success points, zero
catastrophe points, and +8.33 intervention points. Router versus Always Detour
has +16.67 success points, +4.17 catastrophe points, and -41.67 intervention
points. On 16 off-path/no-glass controls, Router retains 93.75% task success
versus 56.25% for Hazard Prompt.

Four predeclared points meet all four frontier criteria. The combined 13-source
frontier meets every criterion somewhere but has no joint all-criteria point.
The paper therefore claims a useful matched-state Pareto frontier, not one
universally dominant deployment setting.

Canonical tables and figures are in
[COUNTERFACTUAL_ROUTER_MAIN_RESULT.md](COUNTERFACTUAL_ROUTER_MAIN_RESULT.md).

## Sequential boundary: P2–P3.2

### P2: secondary positive evidence

On four stable development sources × three conditions, the frozen P2
source-calibrated sequential Router intervenes on 2/12 episodes, improves task
success from 58.3% to 66.7%, and reduces catastrophe from 33.3% to 25.0%.
It nevertheless misses 2/2 known T-20 recovery opportunities. This is a modest
selective benefit, not reliable recovery-window detection.

### P2.5: simple temporal summaries fail

Complete trajectory analysis gives missed-recovery-v-control AUC `0.357` for
raw maximum and `0.286` for MA-3/5/8. Smoothing, accumulation, excess area, run
length, option stability, and trend do not repair treatment/control ordering.

### P3.0–P3.1: statewise supervision improves offline ranking

P3.0 authors 80 dense exact-state anchors—28 recovery-open, 35 loss-control,
and 15 dense Base-preferred—plus 62 hard Base-success controls, with zero
FailSafeHold contract violations.

P3.1 fits a fixed linear head to the frozen 10-D Router output and evaluates it
with strict LOSO over nine sources. Recovery-open-v-hard-control AUC improves
from `0.351` to `1.000`; intervention-needed-v-hard-control AUC from `0.338` to
`0.995`. Source-macro accuracy is `0.632`, macro-F1 `0.540`, hard-control Base
retention `0.984`, and no class collapses globally in OOF predictions.

### P3.2: final negative closeout

P3.2 freezes the all-record direct head and a rank-10 source boundary
`2.0723795239` from the original ten calibration sources at fixed `alpha=0.1`.
It then runs exactly one fresh eight-source × three-condition closeout:

| Method | Task success | Catastrophe | Safe noncompletion | Intervention | Selected Base / Detour / Hold |
|---|---:|---:|---:|---:|---:|
| Base | 16/24 | 8/24 | 0/24 | 0/24 | 24 / 0 / 0 |
| Old P2 sequential Router | 17/24 | 6/24 | 1/24 | 5/24 | 19 / 5 / 0 |
| Direct Recovery Router | 16/24 | 8/24 | 0/24 | 0/24 | 24 / 0 / 0 |

Direct retains Base on 16/16 controls but recovers 0/8 glass episodes and
misses 2/2 known T-20 recoveries. The old P2 Router recovers 1/8 glass episodes
and retains task success on 16/16 controls. Direct-minus-P2 paired differences
are -4.2 task-success points and +8.3 catastrophe points.

No fresh direct trajectory crosses the frozen boundary. The known-recovery-v-
control trajectory-max AUC is `0.250`, so the null is not merely a promising
score hidden behind one conservative threshold.

Final boundary claim:

> Statewise counterfactual intervention value can improve matched-state routing,
> but neither simple temporal aggregation nor directly supervised
> recovery-window classification reliably transfers to fresh sequential
> first-crossing control.

The final closeout is
[`P3_2_FROZEN_DYNAMIC_CLOSEOUT_20260819.md`](../results/P3_2_FROZEN_DYNAMIC_CLOSEOUT_20260819.md).

## Paper boundaries

The paper may claim:

- fresh matched-state evidence for a learned intervention-value frontier;
- better task preservation than rate-matched binary risk routing and a hazard
  prompt at the displayed E16 point;
- learned routing over structured privileged options;
- a modest P2 selective dynamic improvement as secondary evidence;
- a statewise-to-sequential gap established by P2.5–P3.2.

It may not claim:

- universal dominance by one operating point;
- end-to-end learned action recovery;
- reliable recovery-window detection or reliable sequential first crossing;
- that “Knowing When to Intervene” has been solved;
- that Direct Recovery Router is a positive result;
- arbitrary task/layout coverage or production robustness.

The independent statistical unit is always source state. Conditions, anchors,
frames, and option branches are correlated observations.

## Current decisions and next action

1. Use E16/C14 as the abstract and main-result headline.
2. Present P2–P3.2 as one compact statewise-to-sequential boundary section.
3. Consolidate paper text, main figures, tables, captions, and appendix
   provenance.
4. The recovery line is **CLOSED**: no P3.3, no recovery/temporal model, no GRU
   or Transformer, no head refit, no threshold/alpha tuning, and no new
   recovery cohort.
5. Do not tune a deeper sequence model or create another GO/NO-GO gate.
6. The only optional experiment is matched-state replication of the frozen E16
   method on a second task family. It is not required for the current paper and
   does not reopen sequential recovery.

## Navigation

- Paper structure: [PAPER_PLAN.md](PAPER_PLAN.md)
- Exact claims and prohibitions: [CLAIMS.md](CLAIMS.md)
- E16 headline result: [COUNTERFACTUAL_ROUTER_MAIN_RESULT.md](COUNTERFACTUAL_ROUTER_MAIN_RESULT.md)
- P2 sequential evidence: [P2_SEQUENTIAL_FIRST_CROSSING_DEV_20260819.md](../results/P2_SEQUENTIAL_FIRST_CROSSING_DEV_20260819.md)
- P2.5 morphology: [P2_TRACE_MORPHOLOGY_AUDIT_20260819.md](../results/P2_TRACE_MORPHOLOGY_AUDIT_20260819.md)
- P3.0 supervision: [P3_RECOVERY_WINDOW_SUPERVISION_DEV_20260819.md](../results/P3_RECOVERY_WINDOW_SUPERVISION_DEV_20260819.md)
- P3.1 offline direct head: [P3_1_DIRECT_RECOVERY_HEAD_DEV_20260819.md](../results/P3_1_DIRECT_RECOVERY_HEAD_DEV_20260819.md)
- P3.2 final closeout: [P3_2_FROZEN_DYNAMIC_CLOSEOUT_20260819.md](../results/P3_2_FROZEN_DYNAMIC_CLOSEOUT_20260819.md)
- Data availability: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- Historical plans and logs: [archive/README.md](archive/README.md)
