# Paper plan

> **ICLR 2027 gate notice (2026-08-29):** This is the frozen pre-audit writing
> plan. Use the [ICLR 2027 master plan](iclr27/MASTER_PLAN.md),
> [claim ledger](iclr27/CLAIM_LEDGER.md), and
> [reviewer-risk audit](iclr27/REVIEWER_RISK_AUDIT.md) for submission decisions.
> The method headline is conditional until the strongest direct, value, risk,
> and geometry baselines are audited without selecting on fresh test outcomes.

## Working title and frozen contribution

**Risk Is Not Intervention Value: Counterfactual Outcome Routing for VLA Safety**

The paper's central distinction is:

```text
Risk Detection
    !=
Intervention Value
    !=
Reliable Sequential Intervention
```

A failure probability does not say whether Detour or Hold will improve the
outcome, and a useful statewise intervention-value estimate does not by itself
produce a reliable repeated-look first-crossing policy. The positive
contribution is a frozen option-conditioned outcome Router at fresh matched
decision states. The sequential experiments define its deployment boundary.

The paper uses the C0–C14 vocabulary in [CLAIMS.md](CLAIMS.md), with E16/C14 as
the headline positive result. P2–P3.2 is one compact boundary section, not a
second method-development storyline.

## Five-minute paper story

1. **Risk detection is not intervention selection.** Collision risk can be
   decoded while the VLA continues an unsafe action, and fixed interventions
   trade catastrophe against task completion.
2. **Counterfactual option outcomes support useful matched-state routing.** E16
   predicts success, catastrophe, and safe noncompletion for Base, Detour, and
   Retreat, then exposes a fresh safety--success--intervention frontier.
3. **Statewise routing is not reliable sequential intervention.** P2 gives a
   modest selective first-crossing benefit, but P2.5–P3.2 show that neither
   simple temporal aggregation nor direct recovery-window supervision transfers
   reliably to fresh sequential control.
4. **The experimental line is closed.** The paper reports the positive
   matched-state result and the statewise-to-sequential gap without another
   recovery model, threshold search, or cohort.

## Main text

### 1. Risk detection is not intervention selection

Open with the decision problem rather than a hazard catalog:

- Base may complete the task or catastrophize.
- Hold can prevent catastrophe while guaranteeing noncompletion.
- Detour can rescue a Base catastrophe but can also damage a Base success.
- A hazard prompt can change behavior everywhere without routing selectively.

The exact-state development capture makes this concrete. Among 273 matched
decisions, only 21 have identical outcomes under Base, Detour, and Retreat; 91
have positive Oracle value over Base, while 60 are Base-success/Detour-worse.
A scalar risk score cannot encode whether intervention is valuable or which
option should be selected.

Use the wall mechanism chain as motivation: collision imminence is decodable
from frozen representations, the unsafe action persists, and a scoped
`risk-readout -> RetreatHold` interface can close a narrow safety loop. This
supports “decoded but not routed,” not a claim that risk detection solves task
recovery.

### 2. Counterfactual intervention-value routing

Introduce the intentionally small E16 method:

```text
x = single-frame hidden + robot state + nominal action
for option in {Base, Detour, Retreat}:
    predict P(success), P(catastrophe), P(safe noncompletion)
    U_lambda(option) = P(success) - lambda * P(catastrophe)
choose best non-Base option only if advantage over Base > delta
```

PCA and option heads use training sources only. `delta` and rate-matched
binary-risk thresholds use calibration sources only. The grid
`lambda in {1,2,3,5,8}` and target intervention rates 0.1 through 0.9 is frozen.
Fresh outcomes fit no parameter. The contribution is intervention-value
routing over structured options, not an end-to-end learned action policy.

### 3. Headline positive result: E16 / C14

Each accepted source contributes glass, matched off-path glass, and no-glass
conditions. Base, privileged Detour, and Retreat branch from the identical
actual online T-20 simulator/controller state. Source state—not frame,
condition, anchor, or branch—is the independent unit.

Compare Base, Hazard Prompt, Binary Risk -> Retreat, Always Detour, Always
Retreat, the frozen Counterfactual Router, and the Counterfactual Oracle upper
bound in one table.

In the independent eight-source/24-decision cohort, the
`lambda=1,target=0.6` display point reaches 87.5% task success, 8.33%
catastrophe, and 58.33% intervention. Binary Risk reaches 41.67%, 8.33%, and
50.0%; Always Detour reaches 70.83%, 4.17%, and 100%.

The positive result has three readings:

- at a similar intervention rate and identical catastrophe point estimate, the
  Router preserves substantially more task success than Binary Risk;
- versus Always Detour, it trades a small catastrophe increase for higher task
  success and much less intervention;
- on off-path/no-glass controls, it retains 93.75% task success versus 56.25%
  for Hazard Prompt.

Four predeclared points meet all four frontier criteria. The combined
13-source frontier has no single all-criteria point. The claim is a useful
matched-state Pareto frontier, not universal dominance by one deployment point.

### 4. Boundary: the statewise-to-sequential gap

Keep P2–P3.2 in one concise section. Do not present Direct Recovery Router as a
second positive method.

| Stage | Frozen evidence | Paper interpretation |
|---|---|---|
| P2 | Four stable development sources × three conditions. Source-calibrated first crossing improves Base success/catastrophe from 58.3%/33.3% to 66.7%/25.0% at 16.7% intervention, while missing 2/2 known T-20 recovery opportunities. | Modest but real selective dynamic benefit; secondary positive evidence only. |
| P2.5 | Complete score trajectories; raw missed-recovery-v-control AUC `0.357`; moving-average and other simple temporal summaries do not improve the ordering (`0.286` for MA-3/5/8). | Smoothing, accumulation, run length, option stability, and trend do not close the gap. |
| P3.0 | 80 dense exact-state anchors: 28 recovery-open, 35 loss-control, 15 dense Base-preferred; plus 62 hard Base-success controls and zero FailSafeHold contract violations. | Establishes explicit recovery-window supervision without adding a temporal model. |
| P3.1 | Nine-source strict LOSO direct head. Recovery-open-v-hard-control AUC improves from old scalar `0.351` to `1.000`; intervention-needed-v-hard-control from `0.338` to `0.995`; source-macro accuracy `0.632`, macro-F1 `0.540`, no globally collapsed class, hard-control Base retention `0.984`. | Statewise/offline supervision finds useful separation in the frozen 10-D output representation. |
| P3.2 | Fixed `alpha=0.1`, rank-10 source boundary `2.0724`; eight fresh sources × three conditions. Direct Router chooses Base 24/24, recovers 0/8 glass episodes, and misses 2/2 known recoveries. Old P2 Router chooses Detour 5/24, recovers 1/8 glass episodes, and retains task success on 16/16 controls. | The P3.1 improvement does not transfer to fresh sequential first crossing; Direct collapses operationally to Base. |

The post-closeout trajectory audit strengthens the boundary rather than opening
a tuning question: fresh known-recovery-v-control trajectory-max AUC is `0.250`,
and the known-recovery maxima lie below many controls. Lowering the frozen
boundary would cross benign trajectories first.

Paper-facing conclusion:

> Statewise counterfactual intervention value can improve matched-state routing,
> but neither simple temporal aggregation nor directly supervised
> recovery-window classification reliably transfers to fresh sequential
> first-crossing control.

The recovery rescue line is **CLOSED**. There is no P3.3, temporal recovery
model, GRU/Transformer follow-up, threshold/alpha tuning, or new recovery
cohort.

## Main figures and tables

1. **Decision-problem figure:** identical-state branching to Base, Detour, and
   Retreat, with one beneficial and one harmful intervention example.
2. **Method figure:** option-conditioned outcome heads, `U_lambda`, conservative
   advantage margin, and Base-default handoff.
3. **Fresh E16 frontier figure:** catastrophe versus success, marker area as
   intervention, with all predeclared frontier points.
4. **Main seven-method table:** success, catastrophe, safe noncompletion, and
   intervention for Base, prompt, fixed policies, Router, and Oracle.
5. **Compact boundary figure/table:** P2 modest benefit followed by P2.5, P3.0,
   P3.1, and the P3.2 dynamic null. One visual is enough; do not give each
   diagnostic a headline figure.

Canonical E16 tables and figures are in
[COUNTERFACTUAL_ROUTER_MAIN_RESULT.md](COUNTERFACTUAL_ROUTER_MAIN_RESULT.md).
The final sequential closeout is in
[`P3_2_FROZEN_DYNAMIC_CLOSEOUT_20260819.md`](../results/P3_2_FROZEN_DYNAMIC_CLOSEOUT_20260819.md).

## Supporting mechanism section

Keep the wall chain compact:

- swept-corridor causal localization: 15/15 on-path versus 0/33 clear;
- representation readout: OpenVLA/OFT T-5 AUC 0.998/0.903;
- no sustained final-window retreat in 25/25 wall episodes;
- scoped probe-gated `RetreatHold`: 15/15 crashes to 0/15 and 0/22 benign fires;
- final-readout activation steering remains at 100% crash.

This section motivates the distinction between readable risk and expressed
control. It does not compete with E16 for the headline.

## Appendix

- full development Router and ablation tables, split audits, and negative fixed
  operating-point audits;
- combined n=13 frontier and cohort heterogeneity;
- complete P2–P3.2 provenance, dense labels, OOF probabilities, sequential
  calibration, paired source differences, and trace audits;
- prompt matrix, wall/OFT/pi0/probe/shield/steering diagnostics;
- broad E14/E15 no-go evidence, exact restore hashes, and Slurm provenance.

## Evidence discipline and prohibited interpretations

- The E16 confirmation has eight independent source states, not 24 independent
  samples.
- The display point is one of four qualifying points in a predeclared frontier,
  not a prespecified universally dominant deployment point.
- Detour uses privileged geometry; the learned contribution is option routing.
- Oracle observes realized branches and is not deployable.
- P3.1 is an offline statewise result; P3.2 is the final sequential test.
- Do not claim reliable recovery-window detection, end-to-end learned recovery,
  arbitrary-layout robustness, or that “Knowing When” has been solved.

## Frozen experiment policy

Do not tune this cohort further. Experiment discovery is closed and paper
consolidation is active. Specifically:

- no P3.3;
- no temporal recovery model, GRU, or Transformer;
- no head refit, threshold tuning, alpha sweep, or additional recovery cohort;
- no new GO/NO-GO gate for the recovery line.

The only optional additional experiment is a matched-state replication of the
already-frozen E16 method on a second task family. It is not required for the
current paper and must not be framed as reopening sequential recovery.
