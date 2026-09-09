# Direct cost-gain learning: fixed development protocol

Authorized by the user after the completed candidate diagnosis. Existing data only:
18 candidate/control anchors, eight physical sources, 288 saved branches. No GPU,
policy call, new rollout, D8, extra source, architecture search or tuning sweep.
The time objective was proposed after B was seen. LOSO is development diagnosis,
not an independent confirmation. Preserve the old A freeze and every raw result.

## Fixed before fitting

Horizon 200. Four paired targets per anchor, each estimated from its four A repeats:
Refresh minus Base in (1) bounded completion cost (success steps; otherwise 200),
(2) actual inference calls, (3) success indicator, (4) accident indicator.
Normalize target 1 by 200 and target 2 by 40 for saved predictions; invert for reports.
No success-only deletion and no accident-as-fast-completion credit.

Two feature ablations of the SAME linear multioutput paired Ridge recipe, alpha=1:
- metadata: task-2 indicator, actual software age in steps, executed anchor steps;
- observation: metadata plus delivered proprio (8), 4x4 spatial average RGB from
  each delivered camera (96 total), existing chunk length (1). No image encoder
  download/training, fresh pixels, source ID, condition ID, future information,
  policy RNG, latent continuation, or unproduced action proposal.
Image pooling is fixed, no image-feature search. A coarse representation failure
cannot establish that observation content in general is useless.

Outer LOSO: all conditions, anchors and repeats of held-out source excluded from
fit. Train only A means from seven other sources, predict held-out anchor once,
evaluate its B repeats. Each source has equal total training weight (weights mean
one); feature mean/variance computed on training anchors only with these weights,
constant-feature scale=1. Each standardized feature block divided by sqrt(block
width), preventing image dimensionality alone from dominating Ridge regularization.
Unpenalized intercept, ridge=1 from existing _weighted_ridge. No inner tuning.
Freeze all eight fits, preprocessing, OOF predictions and choices before evaluation.

Main choice for either ablation: Refresh iff predicted completion delta < -1e-8,
call delta < -1e-8, success delta >= -1e-8 and accident delta <= 1e-8. These point
prediction checks are not safety guarantees. Report actual new accidents and lost
successes, including at 100 steps as a descriptive deadline check. Also report the
same fitted heads without success/accident gating as a clearly secondary error
analysis; it is not a second fitted model or a posthoc replacement for the main.

Comparators fixed now: Base, AlwaysRefresh, age>=1/3/5, period10 (anchor steps mod
10==0), and age==1 (motivated by already-seen severity pattern, explicitly posthoc
development). No selecting a best rule on B. Both fitted ablations share targets,
regularization and selection semantics.

Report every source and condition, source-macro success/accident/completion cost/
actual calls/refresh use, raw steps separately, and paired success losses/new
accidents. Show predictions vs B deltas and b00/b02 choices. Report removal of each
source from the aggregate as sensitivity (no refit), not eight extra experiments.
Eight shared-training-fold OOF outputs do not support naive independent-repeat CIs.
Especially disclose the sole age=3 source, shared by b06/b08, as extrapolation.

Measure local warm CPU feature extraction + preprocessing + four-head prediction
+ decision, excluding disk loading, for each held-out context: 10 warmups, 200
measurements. Also measure simple rules. Report median/p95 and model size. Raw
elapsed_seconds includes simulation/logging and is not pure VLA latency. Report
break-even VLA-call latency only if mean saved calls >0; cannot establish robot
end-to-end net speedup without deployment hardware measurements. All frozen gates
remain unchanged after timing measurements.

Deliver input hash manifest, executable feature extraction/LOSO fit/evaluation,
fold models and frozen choices, source-level metrics, safety/cost tradeoffs and
Chinese report. No automatic follow-up experiment or permanent GO/NO-GO label.
