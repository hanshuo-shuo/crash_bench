# Validation and reproduction

Full local suite: 558 passed in 10.34 s. Repository audit and git diff --check pass.
New tests cover forbidden fresh/continuation feature access, all held-out condition
labels excluded from fitting, train-only preprocessing, source-balanced intercept,
early-accident cost treatment, and fixed guarded/unguarded choice semantics.

All 18 context files were copied read-only from job 5753957 and their hashes were
compared against the Quest originals. They remain under
`results/direct_cost_learning/contexts/bXX/selector_context.pkl` (14 MB total).
The allowlisted derived data are in feature_dataset.json. Raw contexts contain
other keys, but those keys never enter feature extraction or the fitted models.

Run with NumPy and the repository available in PYTHONPATH; no sklearn/GPU/model
checkpoint needed. Select a NEW output path to preserve the frozen run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/expansion/direct_cost_learning.py fit --output /tmp/new-direct-cost-run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/expansion/direct_cost_learning.py evaluate --output /tmp/new-direct-cost-run
```

The fit stage never opens B. It writes freeze.json and freeze.sha256; evaluation
checks freeze and every training input hash before opening B. Results depend on
actual A costs from other sources only; B has already been exposed historically,
so this procedural separation is not a new confirmatory guarantee.

The frozen input manifest records full code, plan, A, anchors and context hashes.
The evaluation manifest records B, freeze, metrics and timing hashes. Fold models
include all coefficients and scalers. Timing is a local warm CPU microbenchmark,
not a deployable robot latency measurement. No rollout or policy inference was run.
