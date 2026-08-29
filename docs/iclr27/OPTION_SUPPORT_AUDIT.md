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

Pending the clean protocol-commit run.
