# CrashBench

Working paper title: **Knowing When to Intervene: Counterfactual Outcome
Routing for VLA Safety**.

CrashBench studies a practical failure of robot safety systems: predicting that
the Base policy may fail does not reveal whether intervention will help, which
intervention to use, or whether safety comes at the cost of abandoning the
task. The paper therefore reframes safety routing from binary failure detection
to option-conditioned outcome prediction.

## Main result

The frozen router predicts

```text
P(task success | state, option)
P(catastrophe | state, option)
P(safe noncompletion | state, option)
```

for Base, structured Detour, and structured Retreat. It evaluates
`U_lambda(option) = P(success) - lambda * P(catastrophe)` and keeps Base unless
the best intervention has a calibration-frozen advantage greater than `delta`.

On an independent fresh cohort of eight source states and 24 matched decisions,
the paper display point (`lambda=1,target=0.6`) is:

| Method | Task success | Catastrophe | Safe noncompletion | Intervention |
|---|---:|---:|---:|---:|
| Base | 66.67% | 33.33% | 0.00% | 0.00% |
| Hazard Prompt | 37.50% | 20.83% | 41.67% | 100.00% |
| Binary Risk -> Retreat | 41.67% | 8.33% | 50.00% | 50.00% |
| Always Detour | 70.83% | 4.17% | 25.00% | 100.00% |
| Always Retreat | 0.00% | 0.00% | 100.00% | 100.00% |
| **Counterfactual Router** | **87.50%** | 8.33% | **4.17%** | 58.33% |
| Counterfactual Oracle | 91.67% | 0.00% | 8.33% | 33.33% |

At a similar intervention rate, Router improves task success over binary-risk
routing by 45.83 points with the same catastrophe point estimate. Relative to
Always Detour, it gains 16.67 success points and uses 41.67 fewer intervention
points, with a 4.17-point higher catastrophe estimate. On off-path/no-glass
controls, Router retains 93.75% task success versus 56.25% for Hazard Prompt.

![Fresh independent counterfactual-router frontier](results/counterfactual_router_fresh_online_n8_frontier_all_lambdas.png)

Four predeclared `(lambda,target)` points satisfy all four frontier criteria in
the independent cohort. The combined 13-source analysis meets each criterion
somewhere on the frontier but has no single all-criteria point. The result is a
new safety--success--intervention Pareto tradeoff, not a universally dominant
fixed operating point.

The complete seven-method outcome and intervention-quality tables, metric
definitions, plain-language interpretation, figures, confidence intervals, and
caveats are in [the canonical main-result document](docs/COUNTERFACTUAL_ROUTER_MAIN_RESULT.md).

## Paper spine

1. **Risk is not intervention value.** Exact-state branches show that Detour can
   rescue Base catastrophes but can also damage Base successes; Retreat is often
   safe without completing the task. No fixed option is uniformly correct.
2. **Predict consequences, then route conservatively.** A deliberately small
   single-frame model predicts three outcomes for every option and overrides
   Base only when expected advantage clears a frozen margin.
3. **Test the full frontier online.** Base, Hazard Prompt, Binary Risk, Always
   Detour, Always Retreat, Router, and Oracle are evaluated on the same fresh
   source placements, seeds, initializations, and exact T-20 branch states.
   Uncertainty is clustered by source state.
4. **Use diagnosis as mechanism support.** Earlier wall experiments show the
   motivating “decoded but not routed” gap: imminent collision is linearly
   readable while unsafe action continues, and an explicit
   `risk-readout → controller` interface can stop collision. The new headline
   advances from detecting failure to choosing valuable intervention.

The structured Detour controller uses privileged geometry, and Oracle observes
realized branches. Accordingly, the paper claims learned routing over structured
options, not an end-to-end learned recovery policy.

## Repository map

```text
crashbench/                 reusable scenario, policy, probe, and router code
scripts/                    experiment, training, and analysis implementations
setup/                      current environment and Quest submission guide
results/                    promoted summaries, full tables, frontiers, and figures
docs/                       current paper truth sources and frozen protocol
docs/appendix/              supporting and negative evidence
docs/archive/               superseded plans, execution records, and reports
legacy/                     path-stable historical orchestration helpers
```

Historical files remain at stable paths when manifests, tests, or provenance
records depend on them. They are not competing current entrypoints.

## Start here

- [Main result and interpretation](docs/COUNTERFACTUAL_ROUTER_MAIN_RESULT.md)
- [Current paper state](docs/CURRENT.md)
- [Paper plan](docs/PAPER_PLAN.md)
- [Claim ledger](docs/CLAIMS.md)
- [Fresh online protocol](docs/FRESH_COUNTERFACTUAL_ROUTER_PROTOCOL.md)
- [Experiment index](docs/EXPERIMENT_INDEX.md)
- [Reproducibility and data availability](docs/REPRODUCIBILITY.md)
- [Appendix index](docs/appendix/README.md)
- [Legacy archive](docs/archive/README.md)

## Zero-GPU verification

```bash
pip install -e .
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests -q
PYTHONDONTWRITEBYTECODE=1 python scripts/audit_repo.py
git diff --check
```

These checks validate tracked metadata, frozen claim values, scenario
fingerprints, promoted E16 assets, and local links. They do not reproduce GPU
rollouts or restore ignored hidden-state/video assets.

No archival citation or license file is currently supplied; add both before an
external release.
