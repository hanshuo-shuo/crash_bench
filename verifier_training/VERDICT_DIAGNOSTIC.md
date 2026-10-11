# Saved B verdict diagnosis

The new user authorization first permits read-only diagnosis of the completed
Stage 1 B adapter. `launch_verdict_diagnostic.py` submits only that diagnosis;
it does not submit training or dependencies that could start training.

All 120 original examples retain their evidence labels and observations. Inspect
native tokenizer IDs both alone and in the actual verdict context. For each
example and each ground-truth/generated rationale prefix, teacher-force both
candidate verdict sequences. Report raw sequence log likelihoods with and
without EOS, two-candidate normalized likelihoods, the first differing class
token, continuation tokens, EOS, and rationale/format tokens separately. These
are language-model scores, not calibrated feasibility probabilities. Identical
prefixes share a computation and are explicitly recorded as identical.

Gradient attribution uses the saved final adapter, the original training-mode
dropout .05, seed713, and the original 30 four-example batches from epoch one.
Each component is weighted by the actual original objective: sum its token CE
divided by that example's total assistant token count, then average four
examples. Record component norms, pairwise cosines and signed dot-total shares.
Norms are not additive. This measures the old training objective at the final
checkpoint, not the unsaved historical gradient trajectory or Adam updates.
No optimizer is instantiated. Adapter tensors and stored files must remain
bitwise unchanged, with no visual/projector gradients.

Only after reviewing this diagnostic and passing corrected-loss tests may a
single controlled B correction proceed: 0=infeasible, 1=feasible; separately
normalized narrative and decisive verdict losses with equal component weights;
same observations, backbone and frozen vision/projector. A balanced small
overfit must pass before one fixed all-120 engineering fit. No hyperparameter
search or scientific generalization claim is authorized by this protocol.
