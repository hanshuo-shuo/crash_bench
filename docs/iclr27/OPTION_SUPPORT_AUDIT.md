# ICLR 2027 Phase 2.5A option-support audit

- **Evidence role:** exposed-development-only
- **Corpus:** the existing exact-state full capture (5 train + 7 calibration +
  8 development sources; 20 pooled sources total)
- **New rollout:** none
- **Fresh test outcomes:** prohibited
- **Primary preference:** `lambda=1`, `eta=0`

This audit asks whether the existing corpus contains source-diverse strict
support for Base, Detour, and Retreat, and whether a perfect hazard gate mapped
to one fixed non-Base option already exhausts the available Oracle value. It
does not train or evaluate a new router.

## Frozen label contract

For each decision, utility is `+1` for task success, `0` for safe
noncompletion, and `-1` for catastrophe. A strict label is assigned only when
one option is the unique utility maximizer. Detour/Retreat ties remain
`detour_retreat_tie`; every maximizing tie containing Base remains `base_tie`.
`no_good_option` is reserved for decisions where all three options catastrophize.

Source support uses a strict winner margin of at least `gamma=0.5`. With the
discrete utility above, this retains every genuine strict winner and excludes
all ties. No fixed tie order contributes to Base, Detour, or Retreat support.

## Fixed-option recovery contract

The primary fixed-option diagnostic gives a frozen intervention an idealized
hazard gate: for each decision it realizes the better utility of Base and that
one non-Base option. This deliberately upper-bounds any learned scalar-risk
gate while holding intervention choice fixed. Oracle value recovered retains
the repository's existing definition:

```text
sum(U(selected) - U(Base)) / sum(U(Oracle) - U(Base))
```

The Phase 2.5A gate uses this decision-level ratio exactly as frozen in the
revised plan. Equal-source macro recovery is reported as a sensitivity and is
not substituted into the gate after seeing the result.

## Diagnostics

- condition-only, horizon-only, and condition+horizon modal diagnostics use
  leave-one-source-out prediction and equal-source training weights;
- condition and horizon are oracle metadata diagnostics, never deployable
  features;
- risk entropy uses the sealed `Risk->BestFixed` Base-catastrophe probability;
- close-risk Detour/Retreat pairs are matched without replacement under the
  frozen absolute risk-score caliper of `0.05`;
- timing transitions are adjacent decisions within the same source, placement,
  and condition, ordered from more remaining actions to fewer. A tie between
  two strict states is not skipped to manufacture a transition.

## Gate interpretation

Only `PASS-SUPPORT` authorizes Phase 2.5B pooled leave-one-source-out cross-fit.
`SUPPLEMENT-SUPPORT` authorizes only the predeclared training-only source
supplement. `STOP-SUPPORT` stops method rescue. Any unmatched interval is
`INCONCLUSIVE` and does not authorize downstream work.

Even after `PASS-SUPPORT`, Screen A remains blocked until Phase 2.5B returns
`GO-SIGNAL` or `GO-VALUE-ONLY`. Phase 2.5A never authorizes confirmatory data or
new outcome-bearing rollout.

## Reproduction

From the repository root, after the frozen baseline result and its manifest-
hashed full capture have been pulled from Quest:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/iclr27/audit_option_support.py
```

The script creates a new non-overwriting result directory named
`results/iclr27/option_support_audit_<commit>_<utc>/`. The reviewed result and
machine gate are recorded below only after a clean, tested protocol commit is
published.

## Reviewed result

- **Status:** complete on 2026-08-29
- **Formal machine gate:** **PASS-SUPPORT**
- **Protocol and audit commit:** `787623226de03670130330dd7cdd5802694ae551`
- **Execution:** local CPU-only; no Slurm job and no new rollout
- **Result root:**
  [`option_support_audit_787623226de0_20260829T112149Z`](../../results/iclr27/option_support_audit_787623226de0_20260829T112149Z/)
- **Machine decision:**
  [`gate_decision.json`](../../results/iclr27/option_support_audit_787623226de0_20260829T112149Z/gate_decision.json)

The run consumed 273 decisions from the exact 5/7/8 historical split and
pooled all 20 sources as exposed development. Its manifest records a clean
worktree at the published protocol commit and the hashes of the frozen capture
and baseline prediction inputs.

| PASS-SUPPORT criterion | Actual | Threshold | Result |
|---|---:|---:|---|
| strict Base source support | 16 | at least 8 | pass |
| strict Detour source support | 13 | at least 6 | pass |
| strict Retreat source support | 9 | at least 6 | pass |
| sources with a qualifying strict flip | 13 | at least 4 | pass |
| best fixed non-Base Oracle value recovered | 0.7965 | at most 0.80 | pass |

The strict label counts are:

| Historical split | strict Base | strict Detour | strict Retreat | Detour/Retreat tie | Base tie | no good |
|---|---:|---:|---:|---:|---:|---:|
| train | 13 | 18 | 3 | 7 | 29 | 1 |
| calibration | 29 | 8 | 7 | 10 | 38 | 4 |
| development | 19 | 11 | 13 | 14 | 44 | 5 |
| **pooled** | **61** | **37** | **23** | **31** | **111** | **10** |

No tie contributes to the `16/13/9` source-support result. Five sources contain
both a strict Detour and a strict Retreat decision. Thirteen contain either
that flip or a strict Base/intervention flip, which is the exact gate rule.
The stricter adjacent-timing audit finds only three label-changing transitions:
one Detour-to-Retreat, one Retreat-to-Detour, and one Base-to-Detour. Thus the
gate passes source-profile support, but the old corpus is not a mechanically
balanced timing benchmark.

Retreat remains condition- and timing-concentrated. All 23 strict Retreat
decisions are glass decisions. Sixteen occur at horizons 5 or 10, and 21 occur
at horizon 20 or later in the trajectory (`horizon_actions <= 20`). The signal
is nevertheless source-diverse under the frozen minimum: those 23 decisions
come from nine sources.

The idealized Base-versus-Detour gate gains 90 of the 113 decision-level
utility points available over Base, for `90/113 = 0.79646` Oracle value
recovered. It leaves 23 points, or 20.35% of available value, that require a
different option under this upper-bound hazard gate. The equal-source
sensitivity is 0.82896, above the 0.80 PASS threshold; this does not change the
predeclared decision-level gate, but it makes the PASS margin narrow and must
remain visible in Phase 2.5B interpretation.

Risk-score diagnostics also show residual option ambiguity. In the frozen
Base-catastrophe-risk bin `[0.8, 1.0]`, the strict labels are 4 Base, 13 Detour,
and 12 Retreat across 14 sources; Detour/Retreat entropy is 0.9988 bits. The
caliper matcher finds 19 disjoint Detour/Retreat pairs with risk difference at
most 0.05, including 16 same-condition pairs. These are diagnostics from the
sealed historical risk model, not a cross-fitted method result.

The leave-one-source-out strict three-way balanced accuracies are 0.367 for
condition-only, 0.372 for horizon-only, and 0.463 for condition+horizon. These
oracle-metadata diagnostics do not solve strict option choice, but neither do
they establish that a deployable representation can learn it; that is the
purpose of Phase 2.5B.

## Decision

Phase 2.5B pooled source-cross-fitted support rescue is authorized. A
training-only source supplement is not needed. Screen A, confirmatory data,
fresh test outcomes, and new outcome-bearing GPU rollout remain unauthorized
until Phase 2.5B produces `GO-SIGNAL` or `GO-VALUE-ONLY`.
