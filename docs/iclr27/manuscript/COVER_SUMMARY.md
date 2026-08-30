# Venue-neutral cover summary

## Manuscript

**Risk Does Not Specify Intervention: Exact-State Diagnostics for an
OpenVLA--LIBERO Safety Case Study**

This submission is an empirical diagnostic and negative-result case study. It
does not propose a successful router and does not claim a broad VLA-safety
benchmark.

## Question and contribution

Runtime VLA safety is often reduced to predicting whether the nominal policy
will fail. This manuscript asks what that scalar label leaves unidentified when
a system can preserve Base behavior, attempt a task-preserving Detour, or
Retreat safely. The core methodological asset is exact-state realized-option
branching: all three fixed controllers are run from identical hash-verified
saved states, so risk, intervention benefit, best finite option, and regret can
be measured separately without a learned world model.

## Evidence

The paper uses one frozen OpenVLA--LIBERO `glass_recovery` design with 20
exposed-development sources and 273 eligible decisions. Risk and benefit
disagree on 19 decisions across 12 sources. Exhaustive tight matching finds
three same-risk, different-strict-action pairs across two sources; the exact
Detour/Retreat witness has one source. Risk to Best Fixed reaches source-macro
utility 0.2544, compared with a realized finite-option Oracle at 0.5642, leaving
an observed opportunity of 0.3097.

The method result is null. OutcomeRouter gains only +0.0178 over the risk
reference and fails the frozen method gate. Full tiny ADR reaches 0.2311. The
larger oracle-hybrid loss is localized to benefit gating, but every oracle
component is diagnostic and unavailable at deployment. Fresh direct sequential
selection chooses Base on 24/24 episodes, recovers 0/8 glass episodes, and
misses 2/2 known recovery opportunities.

## Scope

All 23 strict Retreat states and 51/60 strict Detour-or-Retreat states are
glass. `offpath` and `noglass` are controls in the same mechanical family, not
independent hazards. Detour uses privileged geometry. All sources are exposed
development. These limits are visible in the abstract, figures, tables,
appendix, artifact map, and reproducibility statement.

The submission therefore supports a narrow claim: within this frozen design,
binary Base catastrophe status does not uniquely determine intervention
benefit or the utility-maximizing finite option, and observed statewise
opportunity is not reliably realized by the evaluated learned selectors or by
fresh sequential first crossing. It does not support a deployable router,
cross-mechanism generality, reliable non-glass Retreat, or reliable sequential
intervention.

## Reproducibility package

The package includes a complete manuscript and appendix, three reviewed main
figures, full policy and witness tables, a machine-readable artifact map with
SHA-256 and commit provenance, a reproducibility statement, a limitations and
ethics statement, and a fail-closed paper-package audit. Non-glass v1 produced
zero option outcomes and is excluded from scientific denominators.
