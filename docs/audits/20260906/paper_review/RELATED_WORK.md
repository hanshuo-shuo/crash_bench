# Related work and positioning — checked 2026-09-06

This focused comparison uses primary paper pages/abstracts, the closest method sections (CheckVLA and CoWAM), and version metadata.
It is not an exhaustive systematic review or an independent reproduction of the
papers' experiments. Preprint status is not treated as acceptance, and numerical
headlines from different benchmarks are not compared directly with CrashBench.

| Primary source | What it already covers | Consequence for CrashBench |
|---|---|---|
| [SAFE](https://arxiv.org/abs/2506.09937), 2025, v2 October 2025 | Generalist VLA failure monitoring from policy features, including generalization across tasks. | Hidden-state failure prediction is not a novelty claim here. The research question must concern intervention-specific benefit and damage to nominal success. |
| [Recovery RL](https://arxiv.org/abs/2010.15920), RA-L/ICRA 2021 | Separates a task policy and a learned safety recovery policy using offline constraint data. | Separating task performance and safety is established. CrashBench must distinguish task-completing recovery from moving into a safe noncompletion state. |
| [Failure-Aware RL / FailureBench](https://arxiv.org/abs/2601.07821), January 2026 preprint | A failure benchmark and offline-to-online learning with a safety critic and recovery policy for manipulation. | “A benchmark with failures and recovery” is too broad a contribution. Exact state-matched option comparisons and their reliability/observability are the more specific asset. |
| [Hide-and-Seek in Trajectories](https://arxiv.org/abs/2605.30834), May 2026 preprint | Temporal discovery of VLA failure signals for runtime monitoring. | A future timing claim needs actual sequential evaluation; statewise AUROC alone cannot establish it. |
| [CheckVLA](https://arxiv.org/abs/2607.26789), July 2026 preprint | Action-conditioned world-model verification during chunk execution, calibrated first-intervention control and deployable suffix replacement. | Action-conditioned monitoring and recovery are already active research. Include nominal action/chunk evidence in a serious comparison; do not claim first VLA execution-time verification. |
| [CoWAM](https://arxiv.org/abs/2608.02578), August 2026 preprint | Conservative selection over identical candidate pools, nominal preservation, admissibility obligations and oracle-labeled comparisons. | Separating selector quality from proposal quality and measuring harmful overrides are already present. CrashBench should justify its actual simulator branching, deployable information and measured outcome uncertainty. |
| [CoRe: Imagining Recovery](https://arxiv.org/abs/2608.14822), August 2026 preprint | Training-free recovery of a frozen VLA using imagined counterfactual continuation and realignment. | Frozen-policy recovery and “counterfactual” language are not sufficient novelty. CrashBench uses physically executed option outcomes, which gives a distinct measurement opportunity but also an obligation to validate branching fidelity. |
| [Cawley & Talbot](https://www.jmlr.org/papers/v11/cawley10a.html), JMLR 2010 | Selection criteria can themselves be overfit; finite-sample model selection can substantially bias estimated generalization performance. | Repeatedly choosing models, epochs and modes on the same 12 development sources cannot be made confirmatory by attaching an ordinary bootstrap interval. |

The proposed research space is **measuring and learning useful recovery under a
specified deployable information set, while distinguishing intervention effect
from rollout variability and preserving nominal successes**. This is a direction
inferred from the comparison, not a verified priority or superiority claim.

A method paper would need a deployable selector plus fresh source-held-out and
closed-loop evidence after development. A diagnostic paper can instead make a
narrow, reproducible empirical claim about the failure of particular measurement
or learning assumptions. These routes can share the same audited experimental
foundation; the current nulls do not permanently force either route.
