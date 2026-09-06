# Paper blueprint: useful recovery rather than intervention frequency

**Working question:** Given a deployed VLA's observation and continuation context,
when does replacing its next behavior increase task utility without unnecessarily
destroying normal progress?

**Possible title:** *When Should a VLA Be Interrupted? Measuring Useful and Harmful
Recovery from Matched States.* This is a working title, not a priority claim.

## Objects and estimands

Let z be the full simulator, controller, observation-queue and policy continuation
state. Let x=phi(z) be the information available to the selector at deployment.
An option o produces an outcome Y(o;z,xi) under residual execution randomness xi.
The deployment target is the conditional expected difference

    A_o(x) = E[U(Y(o))-U(Y(Base)) | x].

A single paired rollout supplies one noisy training observation of this target.
An exact initial-state hash establishes equality only of the represented snapshot
components. It does not, without a replay check, establish equality of rendered
observations, all future stochasticity or terminal labels. When x omits a nominal
action proposal, observation age or policy continuation, fit to a realized
full-state label can be much easier than transfer across sources.

Report at least:

- nominal-success preservation and genuinely recovered task success;
- catastrophes avoided AND catastrophes newly introduced;
- intervention cost and source-macro utility;
- source support for positive and nonpositive benefit, allowing overlap;
- same-option repeat variance and branch start/continuation fidelity;
- statewise performance separately from online timing and end-to-end return.

## Proposed narrative and figures

1. **Introduction.** Failure risk does not by itself identify a beneficial action;
   recovery must also preserve normal task progress. Explain why exact branching
   is useful and why starts, observations and outcomes need different checks.
2. **Problem and protocol.** Define z, x, options, stochastic outcomes, source unit,
   horizon, costs and the information budget. State where intervention is offered.
3. **Measurement audit.** Neutral interventions and repeated Base branches; trace
   the earliest observation/action/state divergence. Figure 1: the intervention
   and null-repeat protocol. Figure 2: observed drift versus label reliability.
4. **Learning diagnosis.** Original additive head, interacting/per-option heads,
   direct paired gain, weighting, scaling and optimization budget. Figure 3:
   train fitting versus source-held-out transfer. Use round 1/2 as exposed development.
5. **Recovery versus stopping.** Preserve/Refresh/Stop compared with a common
   option catalog and explicit restricted-catalog ablation. Figure 4: successes
   recovered, nominal successes lost and new catastrophes, with source intervals.
6. **Deployable information and method extension.** Compare pooled x to measured
   age plus nominal action/chunk context. A conditional gain selector is a candidate,
   not a claimed completed contribution. Keep privileged metadata diagnostics separate.
7. **New evaluation if pursued.** Freeze after development; collect independent
   sources with repeat-aware paired outcomes and a small online timing evaluation.
8. **Limitations.** Two tasks/one current mechanism, exposed development, stochastic
   rendering/physics, structured options, observation limitations and uncertainty.

## Claim ledger for the next draft

| Claim | Present status | Evidence still needed |
|---|---|---|
| Exact-state branching exposes option-specific realized outcomes | Scoped implementation and artifact evidence exists | Quantify same-option replay variation and snapshot persistence |
| More careful source-support accounting changes historical scope interpretation | Supported retrospective correction | None for that correction; no implication of method superiority |
| Current 100-epoch failures do not show that real training samples are unlearnable | Supported by round-two tiny-fit and optimization comparisons | No generalization conclusion follows |
| Some model choices reach successful recovery branches | Supported only as exposed-development accounting | Exclude/diagnose neutral-control artifacts and quantify repeat uncertainty |
| A particular method improves expected recovery on new sources | Not established | A stable recipe, independent source evaluation and statistical uncertainty |
| Selector is safe or broadly deployable | Not established | Explicit guarantee target or repeat-aware empirical safety validation, latency and online tests |

## Smallest method direction worth testing

Use real deployment information (current/delivered observation relation, actual
age, nominal proposed action/chunk and proprioception) to predict action-specific
expected benefit. Keep task completion and catastrophic outcomes separately
inspectable. Treat Stop as a safety fallback with its own false-stop analysis,
not as evidence of successful task recovery. Compare against Base, AlwaysRefresh,
DirectQ, a properly evaluated risk-to-option baseline and a realized-outcome oracle.

Do not tune on a new confirmation cohort to repair a failed result. Do not require
arbitrary positive-label balance or a minimum override rate before learning can
proceed. The next experiment should answer one diagnosed uncertainty and retain
null/failed outcomes, rather than promise a publication outcome.
