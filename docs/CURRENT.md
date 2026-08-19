# Current paper state

## Thesis

The current working paper is **Knowing When to Intervene: Counterfactual
Outcome Routing for VLA Safety**.

Its central claim is that failure probability is not enough to choose a safety
response. A useful router must estimate how Base, Detour, and Retreat change the
probabilities of task success, catastrophe, and safe noncompletion, then keep
Base unless an intervention has sufficient predicted advantage.

The repository uses the C0–C14 vocabulary in [CLAIMS.md](CLAIMS.md). C14 is the
headline learned-routing result. C1/C3/C5/C7/C13 provide causal and mechanistic
motivation; the remaining claims are supporting evidence or boundaries.

## Headline result: E16 / C14

The deployed method is intentionally small:

```text
single-frame PCA-16 hidden + robot state + nominal action
    -> option-specific three-class outcome probabilities
    -> U_lambda(option) = P(success) - lambda * P(catastrophe)
    -> Base unless best non-Base advantage > delta
```

Training uses five source states, calibration uses seven disjoint sources, and
development uses eight diagnostic-only sources. The chosen single-frame
architecture was frozen before fresh online collection. The router JSON and NPZ
are byte-identical across the two fresh cohorts.

The independent random-reset cohort contains eight source states, three matched
conditions per source, and 24 matched decisions. At the predeclared
`lambda=1,target=0.6` display point:

| Method | Task success | Catastrophe | Safe noncompletion | Intervention |
|---|---:|---:|---:|---:|
| Base | 66.67% | 33.33% | 0.00% | 0.00% |
| Hazard Prompt | 37.50% | 20.83% | 41.67% | 100.00% |
| Binary Risk -> Retreat | 41.67% | 8.33% | 50.00% | 50.00% |
| Always Detour | 70.83% | 4.17% | 25.00% | 100.00% |
| Always Retreat | 0.00% | 0.00% | 100.00% | 100.00% |
| **Counterfactual Router** | **87.50%** | 8.33% | **4.17%** | 58.33% |
| Counterfactual Oracle | 91.67% | 0.00% | 8.33% | 33.33% |

Router versus rate-matched Binary Risk has +45.83 task-success points (source
cluster bootstrap 95% CI +25.00,+66.67), 0 catastrophe points (-12.50,+12.50),
and +8.33 intervention points (-8.33,+29.17). Router versus Always Detour has
+16.67 success points (+4.17,+29.17), +4.17 catastrophe points (0,+12.50), and
-41.67 intervention points (-62.50,-20.83).

On 16 off-path/no-glass controls, Router retains 93.75% task success versus
56.25% for Hazard Prompt. Router recovers 78.57% of Oracle value with mean
Oracle regret 0.125.

Four predeclared points meet all four frontier criteria in the independent
cohort: `(1,0.6)`, `(3,0.7)`, `(5,0.6)`, and `(8,0.6)`. The combined 13-source
frontier meets every criterion somewhere but has no single all-criteria point.
The original `lambda=5,target=0.4` and post-pilot `lambda=5,target=0.2`
fixed-point audits remain negative.

The canonical tables, figures, metric definitions, plain-language explanation,
and wording boundaries are in
[COUNTERFACTUAL_ROUTER_MAIN_RESULT.md](COUNTERFACTUAL_ROUTER_MAIN_RESULT.md).

## Why this result has a coherent mechanism

### Option value is state dependent

The sealed development capture contains 20 source states, 273 matched decision
states, and 819 exact-state option rollouts. Only 21/273 decisions have identical
outcomes under Base, Detour, and Retreat. There are 22 Base-catastrophe/Detour-
success decisions, 73 Base-catastrophe/Retreat-safe decisions, and 60 Base-
success/Detour-worse decisions. These branches show why no fixed intervention
can be uniformly correct.

The retrospective Oracle reaches 68.86% success and 3.66% catastrophe versus
57.51% and 33.70% for Base across the development capture. Oracle is an upper
bound and not a deployable competitor.

### Earlier diagnosis explains why routing is needed

- A matched visible wall produces 15/15 on-path crashes and 0/33 clear off-path
  crashes; the same contrast appears across OpenVLA, OFT, and pi0.
- Collision imminence is linearly decodable at T-5 from frozen OpenVLA and OFT
  states (AUC 0.998 and 0.903), while the final window has no sustained retreat
  in 25/25 wall episodes.
- A scoped `risk-readout -> RetreatHold` interface changes 15/15 wall crashes to
  0/15 and fires on 0/22 benign rollouts, while direct activation steering stays
  at 100% crash.

This is the supporting “decoded but not routed” mechanism. E16 advances the
question from whether a failure signal exists to whether intervention is
valuable and which structured option should be selected.

## Dynamic timing boundary: P2/P2.5

E16 evaluates option choice at an actual online T-20 anchor defined relative to
the matched Base collision. P2 asks the stronger from-reset question. After
source-level sequential calibration, the dynamic first-crossing Router
intervenes on 2/12 episodes across four stable development sources, improves
success from 58.3% to 66.7%, and reduces catastrophe from 33.3% to 25.0%.
This is a small development Pareto point, not a confirmatory timing claim: both
known T-20-recoverable glass episodes remain missed.

P2.5 reconstructs the complete score trajectories and tests raw maximum,
MA-3/5/8 maximum, top-5 mean, excess area, longest above-margin run, option
stability, and last-10 slope. Raw maximum has missed-treatment-versus-control
AUC 0.357; the best temporal variants reach 0.286. `heldout_0010` contains an
11-action Detour-above-margin run in its T-20 window, but `heldout_0019` has only
a five-action burst, and several benign controls remain high after smoothing.
Simple accumulation therefore does not repair the ordering.

The paper-facing interpretation is deliberately asymmetric: E16 supports
learned option-value routing at matched decision states; P2/P2.5 establishes
the unresolved deployment-timing boundary. Full evidence is in
[P2_SEQUENTIAL_FIRST_CROSSING_DEV_20260819.md](../results/P2_SEQUENTIAL_FIRST_CROSSING_DEV_20260819.md)
and [P2_TRACE_MORPHOLOGY_AUDIT_20260819.md](../results/P2_TRACE_MORPHOLOGY_AUDIT_20260819.md).

## Paper boundaries

The paper may claim:

- fresh online evidence for a learned intervention-value frontier;
- a new matched-decision Pareto tradeoff relative to fixed policies and binary
  risk;
- better control retention than a whole-episode hazard prompt;
- learned routing over structured, privileged options;
- a selective development Pareto point for source-calibrated dynamic first
  crossing, reported separately from the E16 confirmation.

It may not claim:

- universal dominance by one prespecified operating point;
- uniformly lower catastrophe than Always Detour;
- end-to-end learned action recovery;
- arbitrary glass-layout or task-family coverage;
- production robustness from eight independent confirmation sources;
- reliable from-reset recovery-window detection by the current single-frame
  score, or full separation of recoverable treatments from benign controls.

The statistical unit is always source state. Frames, conditions, anchors, and
rollout branches are correlated observations, not independent samples.

## Current decisions

1. Use E16/C14 as the abstract and main-result headline.
2. Use the wall causal/readout/intervention chain as motivation and mechanism,
   not as a competing paper thesis.
3. Keep Hazard Prompt, Binary Risk, Always Detour, Always Retreat, Base, Router,
   and Oracle together in the main comparison.
4. Show the full multi-`lambda` frontier; do not reduce the paper to one fixed
   `lambda=5` utility.
5. Do not tune a deeper sequence model, Transformer, ensemble world model,
   outcome-selected threshold, or simple evidence accumulator on these small
   cohorts.
6. If the dynamic timing claim is pursued, use explicit recovery-window
   supervision as the next method change; consider a temporal value model only
   after that target is validated on source-disjoint data.
7. If another experiment is needed only for E16 generalization, prefer a new
   task-family replication with the frozen method. Five-fold held-out wall guard
   and OFT-specific guard remain secondary mechanism upgrades.

## Navigation

- Main result: [COUNTERFACTUAL_ROUTER_MAIN_RESULT.md](COUNTERFACTUAL_ROUTER_MAIN_RESULT.md)
- Paper structure: [PAPER_PLAN.md](PAPER_PLAN.md)
- Exact claims: [CLAIMS.md](CLAIMS.md)
- Frozen fresh protocol: [FRESH_COUNTERFACTUAL_ROUTER_PROTOCOL.md](FRESH_COUNTERFACTUAL_ROUTER_PROTOCOL.md)
- Development/full-capture provenance: [COUNTERFACTUAL_ROUTER_HANDOFF.md](COUNTERFACTUAL_ROUTER_HANDOFF.md)
- Dynamic protocol and closeout: [P2_DYNAMIC_FIRST_CROSSING.md](P2_DYNAMIC_FIRST_CROSSING.md)
- P2.5 morphology audit: [P2_TRACE_MORPHOLOGY_AUDIT_20260819.md](../results/P2_TRACE_MORPHOLOGY_AUDIT_20260819.md)
- Experiment status: [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md)
- Data availability: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- Historical plans and logs: [archive/README.md](archive/README.md)
