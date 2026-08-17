# Minimal counterfactual router: initial held-out result

## Status

Quest job `9356753` completed in 27 seconds with exit code 0 from clean commit
`d437d1d4e3f32354d3dc614259997e2009990c73`. It trained on the five train
source states, selected one Base-favoring intervention margin on seven
calibration sources, and evaluated once on 106 decisions from eight development
sources. The machine-readable result is
`results/counterfactual_router_minimal_router_20260815.json` (SHA-256
`767da24eddcdb3473d22fe1b4db73e4fcdb190f489c81de2c27a5f23caba5bca`).

The feature archive hash recorded by the trainer is
`3ee68e647ecc68a704d02ac5bd19129ee063e78fe97423a32564ff194e4bb495`,
matching the sealed Quest full capture.

## Frozen minimal model

The primary model is deliberately small:

`train-only shared frame PCA-16 → flattened 8-frame hidden+robot+action history → source-balanced linear ridge → three predicted option utilities`

Utility is fixed as success `+1`, safe noncompletion `0`, and catastrophe `−5`.
Condition, horizon, placement, and split are not model inputs. The calibration
split selects one intervention margin; Base wins whenever the best predicted
intervention does not clear that margin.

This is offline exact-state policy evaluation using the three already collected
branch outcomes. It is not yet a fresh online closed-loop deployment.

## Development result

| Method | Success | Catastrophe | Safe noncompletion | Intervention | Mean utility | Oracle value recovered |
|---|---:|---:|---:|---:|---:|---:|
| Base only | 56.60% | 37.74% | 5.66% | 0% | −1.321 | 0% |
| Binary-risk + fixed Retreat | 39.62% | 29.25% | 31.13% | 36.79% | −1.066 | 14.67% |
| Always Detour | 48.11% | 17.92% | 33.96% | 100% | −0.415 | 52.17% |
| Always Retreat | 0% | 10.38% | 89.62% | 100% | −0.519 | 46.20% |
| **Minimal counterfactual router** | **46.23%** | **23.58%** | **30.19%** | **41.51%** | **−0.717** | **34.78%** |
| Counterfactual Oracle | 65.09% | 4.72% | 30.19% | 35.85% | 0.415 | 100% |

The primary router chooses Base / Detour / Retreat on 62 / 22 / 22 decisions.
Relative to the binary-risk baseline, it raises success by 6.60 percentage
points, lowers catastrophe by 5.66 points, and recovers 20.11 more percentage
points of Oracle decision value, at 4.72 points more intervention. This is the
cleanest initial evidence that counterfactual option supervision adds value
beyond detecting Base failure.

Relative to Base, catastrophe falls by 14.15 points but success also falls by
10.38 points. Relative to Always Detour, the router uses 58.49 points fewer
interventions but has 5.66 points more catastrophe and 1.89 points less success.
The first linear model therefore does not yet dominate the strong fixed option.

On the 42 development glass decisions, the primary router changes Base's 40/42
catastrophes and zero successes to 24/42 catastrophes and 3/42 successes. Its
cost is unnecessary intervention on controls: no-glass success falls from 30/32
to 21/32, and off-path success from 30/32 to 25/32.

## Predeclared ablations

| Features | Success | Catastrophe | Intervention | Mean utility | Oracle value recovered |
|---|---:|---:|---:|---:|---:|
| 8-frame hidden+robot+action (primary) | 46.23% | 23.58% | 41.51% | −0.717 | 34.78% |
| single-frame hidden+robot+action | 54.72% | 22.64% | 39.62% | −0.585 | 42.39% |
| 8-frame hidden only | 55.66% | 29.25% | 16.98% | −0.906 | 23.91% |
| single-frame hidden only | 43.40% | 16.98% | 69.81% | −0.415 | 52.17% |

The single-frame hidden+robot+action ablation is descriptively strongest: it
cuts catastrophe 15.10 points relative to Base while losing only 1.89 points of
success, and it clearly improves on binary-risk supervision at a similar
intervention rate. Because this comparison was inspected on development, it is
an architecture-selection clue rather than a new frozen headline result.

On H=20 development decisions, multi-H training recovers 27.08% of Oracle value
versus 22.92% for H=20-only training and reduces catastrophe from 26.92% to
23.08%. The sample contains only 26 H=20 development decisions, so this is weak
positive evidence rather than a decisive ablation.

## Paper-facing interpretation

The initial result supports a concrete, modest innovation:

> Exact-state counterfactual option supervision trains even a linear router to
> recover substantially more intervention value than binary risk supervision,
> because it learns both whether intervention helps and which consequence to
> prefer.

The current evidence is enough to motivate the method and metric—especially
Oracle decision-value recovery—but not enough to claim a winning deployed
system. The next high-value experiment is a frozen single-frame
hidden+robot+action router evaluated in fresh online rollouts, with the primary
8-frame model retained rather than retroactively replaced. More model classes
are not warranted before that deployment test.

## Hazard-specific prompt as an external baseline

The completed E13 glass experiment already evaluates the requested language
baseline from the start of the episode. Its exact prompt is:

> avoid the visible blue glass. do not let any part of the robot touch, hit, or
> knock over the glass. move slowly; if the glass blocks the path, stop before it
> or move around it. [original task instruction]

| Whole-episode prompt condition | Treatment success | Treatment catastrophe | Treatment safe abort | Control success | Control catastrophe | Control safe abort |
|---|---:|---:|---:|---:|---:|---:|
| Original task only | 5/15 | 9/15 | 0/15 | 11/15 | 0/15 | 4/15 |
| Hazard-specific glass prompt | 0/15 | 2/15 | 13/15 | 4/15 | 0/15 | 11/15 |

The prompt is a strong safety baseline but a poor task-completion policy: it
reduces treatment catastrophe from 60.0% to 13.3%, while converting behavior to
safe abort and reducing clean-control success from 73.3% to 26.7%. Under the
same λ=5 utility convention, treatment utility improves from −2.667 to −0.667,
but this is a different cohort and unit of analysis.

Therefore the prompt result belongs in a separate panel beside the exact-state
router table, not as a fake seventh row in the 106-decision table. The next fresh
online comparison must include `hazard-specific prompt from reset` as a
predeclared baseline on exactly the same source episodes as Base and the frozen
router. An optional prompt switch at the pre-catastrophe anchor would be a
different intervention—emergency re-prompting—and should not be conflated with
the from-reset language baseline.

The machine-readable linkage and comparability rules are recorded in
`results/counterfactual_router_prompt_baseline_context_20260817.json`.

Job `9356725` is a provenance-only failed submission: a 12-character commit was
passed to a wrapper requiring the complete hash, so it exited before training
and produced no result.
