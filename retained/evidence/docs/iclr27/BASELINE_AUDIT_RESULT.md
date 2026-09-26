# ICLR 2027 phase-2 strongest-baseline audit

- **Status:** complete on 2026-08-29
- **Formal machine gate:** **INCONCLUSIVE**
- **Paper decision:** **NO-GO for the current outcome-decomposition headline**
- **Next authorized step:** a scoped option-support / option-ambiguity pivot, not a
  new confirmatory cohort.

The formal gate is intentionally narrower than the paper decision. None of
GO-A, GO-B, PIVOT-C, or STOP-D cleared every frozen tolerance. Failure to clear
a positive gate is nevertheless a no-go for promoting the present method
claim or spending confirmatory rollout budget.

## Sealed run

- Code and protocol commit: `eed1feecb3bd298d870ab049976b7ac006d49633`
- Quest Slurm job: `5128781`, account `p33100`, partition `short`, `COMPLETED`,
  elapsed `00:00:23`, exit `0:0`
- Result root:
  [`baseline_audit_eed1feecb3bd_20260829T101548Z_job5128781`](../../results/iclr27/baseline_audit_eed1feecb3bd_20260829T101548Z_job5128781/)
- Machine decision: [`gate_decision.json`](../../results/iclr27/baseline_audit_eed1feecb3bd_20260829T101548Z_job5128781/gate_decision.json)
- Metrics and source inference:
  [`overall_metrics.csv`](../../results/iclr27/baseline_audit_eed1feecb3bd_20260829T101548Z_job5128781/overall_metrics.csv),
  [`exact_tests.json`](../../results/iclr27/baseline_audit_eed1feecb3bd_20260829T101548Z_job5128781/exact_tests.json)
- Calibration and preference transfer:
  [`calibration.json`](../../results/iclr27/baseline_audit_eed1feecb3bd_20260829T101548Z_job5128781/calibration.json)

The run used only the existing exact-state full capture: five train sources,
seven calibration sources, and eight development sources. The primary
preference was frozen at `lambda=1`, `eta=0`, with source-balanced calibration
target rate `0.60`. Fresh test outcomes were neither loaded nor used.

## Primary result

Gate values below are source-macro development estimates. Parentheses contain
raw decision counts over 106 repeated decisions.

| Method | Utility | Success | Catastrophe | Intervention |
|---|---:|---:|---:|---:|
| Risk -> Best Fixed (Detour) | **0.2111** | **54.10%** (63/106) | 32.99% (30/106) | **52.85%** (48/106) |
| Risk + Two Stage | **0.2028** | **51.18%** (59/106) | 30.90% (28/106) | **52.85%** (48/106) |
| Outcome Router | 0.1944 | 49.17% (57/106) | **29.72%** (25/106) | 65.76% (65/106) |
| Direct-Q | 0.1451 | 45.63% (53/106) | 31.11% (27/106) | 60.28% (59/106) |
| Pairwise Advantage | 0.1451 | 45.63% (53/106) | 31.11% (27/106) | 60.28% (59/106) |
| State/action-only Outcome | 0.1340 | 42.71% (49/106) | 29.31% (24/106) | 79.03% (82/106) |

Risk -> Best Fixed selected Detour on calibration. Relative to it, Outcome
Router has `-0.0167` source-macro utility, `-4.93` success points, `-3.26`
catastrophe points, and `+12.92` intervention points. This is a tradeoff, not a
method win. It misses the frozen same-rate tolerance by 2.92 points and the
catastrophe tolerance by 1.26 points, so formal PIVOT-C also does not fire.

Source-paired inference is correspondingly weak:

- Outcome minus Risk -> Best Fixed utility: `-0.0167`, bootstrap 95% CI
  `[-0.0938, 0.0417]`, exact sign-flip `p=0.875`.
- Success difference: `-0.0493`, exact `p=0.25`.
- Catastrophe difference: `-0.0326`, exact `p=0.3125`.
- Intervention difference: `+0.1292`, exact `p=0.1875`.

Outcome Router does descriptively outperform Direct-Q at a comparable rate:
utility `+0.0493`, success `+3.54` points, catastrophe `-1.39` points, and
intervention `+5.49` points. The utility interval still includes zero
(`[-0.0021, 0.1007]`; exact `p=0.15625`), so this does not rescue GO-A.
After excluding oracle diagnostics, Outcome Router is not on the development
utility/intervention frontier; that frontier contains Base and the two
identical Risk -> Detour / Best Fixed policies.

## Why each branch failed

- **GO-A failed:** Outcome Router did not beat frozen Risk -> Best Fixed, its
  benefit was not independent of a higher development intervention rate, and
  preference reweighting did not show broad added value.
- **GO-B failed:** Outcome, Direct-Q, and Pairwise did not all clearly beat the
  scalar-risk reference under the frozen utility, catastrophe, and rate
  tolerances.
- **PIVOT-C did not formally fire:** Risk -> Detour / Best Fixed was close in
  utility but outside the frozen same-rate and catastrophe tolerances.
- **STOP-D did not fire:** the deployable state/action-only ablation was worse
  and intervened more. The stronger geometry diagnostic used the capture
  condition label and is explicitly oracle-only.

Hidden-only is also not cleanly separated from the full Router (utility
`0.1819` versus `0.1944` with a wide paired interval), so this audit does not
establish that robot/action covariates add stable value either.

Preference transfer passed only `lambda=1` and `lambda=2`; it failed
`lambda=3,5,8`. The protocol required at least three non-inferior lambdas, so
`reweighting_value=false`.

## Diagnostic that determines the next experiment

The development outcomes contain real option ambiguity, but training support
for Retreat is too thin and every learned method mostly collapses to Detour.

| Split | Oracle Base | Oracle Detour | Oracle Retreat |
|---|---:|---:|---:|
| Train | 43 | 25 | **3** |
| Calibration | 71 | 18 | 7 |
| Development | 68 | 25 | **13** |

Among the 38 development decisions with a beneficial intervention, 11 strictly
prefer Detour, 13 strictly prefer Retreat, and 14 tie Detour/Retreat. The frozen
tie rule assigns those 14 to Detour, so the table does not manufacture the 13
strict Retreat cases. Ten of the 13 Retreat-optimal decisions occur at horizons
5 or 10, consistent with the intended late-state mechanism.

The three train Retreat-optimal decisions come from only two train sources.
On development, Outcome Router chooses Base/Detour/Retreat `41/59/6`; among 13
Retreat-optimal decisions it selects Retreat only once. Risk + Two Stage gets
only 2/13 correct. These counts use the frozen `lambda=1`, Base-first tie rule
over the tracked full capture and the sealed [`all_choices.csv`](../../results/iclr27/baseline_audit_eed1feecb3bd_20260829T101548Z_job5128781/all_choices.csv).

As an exploratory preference diagnostic, the separately trained `lambda=8`
Direct-Q selects Retreat correctly on 11/13 of those cases. This shows that the
capture contains a recoverable preference signal, but it is not a `lambda=1`
gate win and does not repair the missing train-source support.

The cohort also has a strong condition shortcut: Base catastrophes occur in
40/42 glass decisions, while off-path and no-glass controls each have 30/32
Base successes and zero catastrophes. An oracle condition gate therefore
reaches utility `0.3118`, catastrophe `24.79%`, and intervention `50.0%`. It is
not deployable evidence, but it shows why hazard recognition plus fixed Detour
is a serious explanation of the current result.

## Decision and claim control

Effective immediately:

1. Do not claim that outcome decomposition beats strong scalar-risk or direct
   value baselines.
2. Do not use the old `+45.83`-point comparison against Risk -> Retreat as the
   paper headline; it is structurally baseline-dependent.
3. Do not launch a new confirmatory test cohort, second task, or deeper model
   from this result.
4. If the ICLR method line continues, first author a small, mechanically
   selected option-ambiguity screen with adequate train-source support for
   Base-, Detour-, and Retreat-optimal states. Then rerun this exact audit
   contract before freezing any confirmatory protocol.

Safe current wording is:

> The exact-state protocol exposes option-level supervision and a descriptive
> routing tradeoff, but the frozen strongest-baseline audit does not yet
> establish outcome-decomposition or multi-option-value superiority.

The exact-state benchmark/protocol remains scientifically useful. The present
method headline does not pass its first hard ICLR gate.
