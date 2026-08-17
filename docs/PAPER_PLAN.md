# Paper plan

## Working title and contribution

**Knowing When to Intervene: Counterfactual Outcome Routing for VLA Safety**

A safety router should not merely predict whether the current policy may fail.
It should predict how candidate interventions change task success,
catastrophe, and safe noncompletion, and should override the Base policy only
when the expected counterfactual benefit is sufficiently large.

The paper uses the C0–C14 vocabulary in [CLAIMS.md](CLAIMS.md), with C14 as the
headline result. Historical E14/E15 execution order is not a narrative outline.

## Four-act main text

### Act I — Risk detection is not intervention selection

Open with the decision problem rather than a catalog of hazards.

- A Base policy may complete the task or catastrophize.
- Retreat can prevent catastrophe but produce safe noncompletion.
- Detour can rescue a Base catastrophe but can also damage a Base success.
- A prompt can alter behavior everywhere without producing useful selective
  control.

The exact-state development capture makes this concrete: among 273 matched
decisions, only 21 have identical outcomes under Base, Detour, and Retreat; 91
have positive Oracle value over Base, while 60 are Base-success/Detour-worse.

Main claim: a scalar Base-risk score cannot encode whether intervention is
beneficial or which intervention should be chosen.

Use the older “decoded but not routed” evidence as motivation: collision risk is
readable from frozen representations, but unsafe action continues. The scoped
wall `risk-readout -> RetreatHold` result shows that explicit routing can close a
safety loop, while its safe-abort behavior also exposes why task completion must
be modeled explicitly.

### Act II — Predict option outcomes and route conservatively

Introduce the intentionally small method:

```text
x = single-frame hidden + robot state + nominal action
for option in {Base, Detour, Retreat}:
    predict P(success), P(catastrophe), P(safe noncompletion)
    U_lambda(option) = P(success) - lambda * P(catastrophe)
choose best non-Base option only if advantage over Base > delta
```

PCA and option heads use training sources only. `delta` and the rate-matched
binary-risk thresholds use calibration sources only. The full grid
`lambda in {1,2,3,5,8}` and target intervention rates 0.1 through 0.9 is frozen.
Development is diagnostic; fresh outcomes fit no parameter.

The method contribution is **intervention-value routing**, not a deeper sequence
model, a world model, or an end-to-end action policy.

### Act III — Fresh matched online evaluation

Every accepted source contributes glass, matched off-path glass, and no-glass
conditions. At the actual online T-20 anchor, Base, frozen privileged Detour,
and Retreat are evaluated from the identical simulator/controller state. The
hazard-specific prompt starts from the same episode reset.

Compare all seven methods together:

1. Base;
2. Hazard Prompt;
3. Binary Risk -> Retreat;
4. Always Detour;
5. Always Retreat;
6. frozen Counterfactual Router;
7. Counterfactual Oracle upper bound.

Use source state—not frame, condition, anchor, or branch—as the independent
unit. Report 5,000-replicate shared source-cluster bootstrap intervals and
paired differences.

### Act IV — A new frontier point, not universal dominance

In the independent eight-source/24-decision cohort, the
`lambda=1,target=0.6` display point reaches 87.5% success, 8.33% catastrophe,
and 58.33% intervention. Binary Risk reaches 41.67%, 8.33%, and 50%; Always
Detour reaches 70.83%, 4.17%, and 100%.

The result has three complementary readings:

- at a similar intervention rate and identical catastrophe point estimate,
  Router preserves far more task success than Binary Risk;
- versus Always Detour, Router trades a small catastrophe increase for higher
  success and much less intervention;
- on off-path/no-glass controls, Router retains 93.75% task success versus
  56.25% for Hazard Prompt.

Four predeclared frontier points jointly satisfy all four acceptance criteria.
The combined 13-source frontier has no single all-criteria point, so end with the
honest boundary: the method supplies a useful and adjustable Pareto frontier,
not a single universally dominant deployment setting.

## Main figures and tables

1. **Decision-problem figure:** identical state branching to Base, Detour, and
   Retreat; include one example where intervention helps and one where it hurts.
2. **Method figure:** option-conditioned outcome heads, `U_lambda`, conservative
   advantage margin, and Base-default handoff.
3. **Fresh frontier figure:** all five catastrophe costs; catastrophe on x,
   success on y, marker area as intervention, stars for all-criteria points.
4. **Control/decision-quality figure or compact table:** unnecessary
   intervention, missed beneficial intervention, regret, recovered Oracle value,
   and off-path/no-glass retention.
5. **Main seven-method table:** Base, Hazard Prompt, Binary Risk, Always Detour,
   Always Retreat, Router, and Oracle with success, catastrophe, safe
   noncompletion, and intervention.

The canonical numeric tables and current figures are in
[COUNTERFACTUAL_ROUTER_MAIN_RESULT.md](COUNTERFACTUAL_ROUTER_MAIN_RESULT.md).

## Supporting mechanism section

Keep the earlier wall chain compact:

- swept-corridor causal localization: 15/15 on-path versus 0/33 clear;
- representation readout: OpenVLA/OFT T-5 AUC 0.998/0.903;
- no sustained final-window retreat in 25/25 wall episodes;
- scoped probe-gated `RetreatHold`: 15/15 crashes to 0/15 and 0/22 benign fires;
- final-readout activation steering remains at 100% crash.

This section supports the premise that information can be decoded yet fail to
control behavior. It should not compete with the learned router for the paper
headline.

## Appendix

- full development router/ablation tables and split audits;
- original and post-pilot failed fixed-point audits;
- combined n=13 frontier and cohort heterogeneity;
- complete prompt matrix and whole-episode instructions;
- full wall, OFT, pi0, probe, confound, shield, and steering diagnostics;
- broad glass E14/E15 no-go, certification funnel, Oracle upper-bound traces,
  restore/hash details, and Slurm provenance;
- negative hazard mechanisms and low-wall existence demonstrations.

## Cut from the main narrative

- chronological E14/E15 Pilot A/B/C/D/E/F storytelling;
- deeper sequence models, Transformers, ensemble world models, or another
  outcome-led threshold search;
- claims that Hazard Prompt or Retreat constitutes task recovery;
- claims that Oracle is a deployable competitor;
- claims of universal safety, arbitrary-layout coverage, or production-scale
  robustness;
- separate headline figures for every historical diagnostic.

## Evidence discipline

- The independent confirmation has eight source states, not 24 independent
  samples; the three conditions are clustered within source.
- The display point is one of four all-criteria points in a fully predeclared
  frontier, not a prespecified single deployment point.
- The combined n=13 result passes criteria at different frontier locations but
  has no joint all-criteria point.
- Detour uses privileged geometry; the claim is learned option routing.
- Oracle observes realized branches and is excluded from deployable Pareto
  dominance.
- Negative fixed-`lambda=5` audits stay visible.

## Next experiment policy

Do not tune this cohort further. If review requires another positive result,
replicate the frozen router and option contract on a new task family. Five-fold
held-out wall guard and OFT-specific online guard remain useful mechanism
extensions, but they do not replace the current E16 headline.
