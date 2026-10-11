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

The first all120 diagnostic completed in job9453285 with unchanged weights.
Teacher-forced sequence classification (107/120) disagreed with the saved greedy
outputs (73/120), so a second read-only native-prefix check was required. Its
initial replay assertion stopped job9454018 before training: it compared raw
model probabilities with processed generation probabilities. Inspection found
the staged native generation default `repetition_penalty=1.05`. The revised
diagnostic explicitly reports both scores and replays that processor, including
whether each decisive class token already occurs in the full input/prefix. The
failed diagnostic artifact remains recoverable and is not a training result.

Job9454708 also stopped its handwritten-cache equality gate on the first row.
Those manual replay probabilities are invalid and excluded. Its native raw and
processed logits already demonstrate the penalty effect, but the complete
repeat now uses the library's own `generate` path with forced canonical prefix
IDs, capturing raw logits and processor scores before the forcing operation.
This removes the unvalidated handwritten cache implementation. Both failed
attempts leave the original adapter and all scientific results unchanged.

The controlled binary fit keeps the same generation penalty1.05 explicitly and
reports both raw and processed decision losses. Only the requested label tokens
and supervised objective change. The smoke uses the original balanced eight,
original LR2e-4/dropout0/cap240; the full fit uses original LR1e-4/dropout.05,
seed713, ten epochs/300 updates and the identical original example order. Both
start from fresh base weights. No extra decoding or training search is performed.

The completed native diagnosis requires distinguishing decision rules. There are
nine exact raw class-logit ties. Stored native token argmax gives109/120
(66/72 feasible,43/48 infeasible); full-verdict sequence scoring gives102/120.
The native repetition penalty changes42 correct negative token decisions to
feasible and corrects six positive decisions, yielding73/120. The reporting
readback uses actual stored token IDs, not Python dictionary order for ties.

The corrected balanced8 smoke passed at140 updates. One fresh, fixed300-update
full fit completed in job9459482:97/120 verdicts (49/72 feasible,48/48 infeasible),
104/120 grounded observability templates and96/120 exact full outputs. Its
joint exact-fit criterion is false. All120 predictions remain unchanged when
the command budget becomes zero. The primary artifact's legacy-named
`zero_flips_among_original_feasible` counted23 zero-correct positive-source
cases that were already predicted infeasible, not actual flips. The separate
readback reports the actual zero flips and preserves the immutable raw artifact.
The source reporter is corrected for subsequent use; no checkpoint is retrained.

A final read-only matched-prefix forward pass measures the residual corrected-B
errors under gold versus actual generated rationale token IDs. It does not create
an optimizer or alter observations. Final reporting has its own published source
archive and output root; the trained706fd140 checkpoint and historical scientific
results remain unchanged.
