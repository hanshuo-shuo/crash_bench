# Round 2: why does Refresh gain remain negative on train?

User authorized continuation on 2026-09-06 after reviewing the round-one results.
Base: c4d873a401dc80545c15f8592be612384437b3b4; branch codex/refresh-gain-round2.

This is exposed-development diagnosis and model iteration, not confirmation.
Use only existing D5 train/development labels and the existing 104-D cache.
Calibration outcomes and D8 outcomes are not used. Preserve all round-one files.

Fixed comparisons before running:

- Full train: outcome interaction and outcome per-option heads; paired-gain MLP
  with the same LayerNorm/128-D trunk dimensions; paired-gain positive-weight=5
  ablation; paired-gain train-standardized-feature ablation (no LayerNorm).
- Seeds 0–4; AdamW lr=0.001, weight_decay=0.0001. Record checkpoints at epochs
  100 and 500, including train loss and train/development readouts. Do not select
  an epoch using development and report only the winner.
- Outcome loss is unchanged CE+cost MSE. Gain targets are exact matched-state
  U(option)-U(Base), separately for Refresh and Stop; source-balanced MSE. The
  fixed positive-weight ablation changes the target-weighted objective and may
  increase false interventions; evaluate on natural development prevalence.
- Standardization uses only the 24 train sources, feature std floor 1e-6; it adds
  no information or privileged metadata. It is a separate preprocessing ablation,
  not a claimed pure loss comparison.
- Tiny fit: first four lexicographic train source IDs that each contain a true
  Refresh rescue (Base non-success to Refresh success) and non-beneficial Refresh.
  Retain ALL their decision blocks. Fit per-option outcome, paired gain, and
  train-standardized paired gain at seed 0 to 2,000 epochs; record 100/500/2000.
  Only these train sources are used to fit tiny standardization statistics.
  This is deliberately outcome-selected engineering diagnosis, never generalization evidence.
- Fit DirectQ on full train as the fixed strong comparator. Verify utility
  decomposition and inspect outcome-vs-cost contributions to Refresh gain.

Readout: paired-gain sign/ranking and MSE, true rescue opportunities recovered,
Base successes lost, Refresh-induced catastrophes, Stop counts, source-macro
success/catastrophe/U0 and paired source-bootstrap intervals. Evaluate both
all-options and Refresh-only (Base/Refresh) so Stop cannot hide failure to recover.
Report all seeds and source-level outputs. No calibrated-safety claim.

Maximum budget: 25 full-data fits ×500 epochs plus 3 tiny fits ×2000 epochs,
2 CPU cores, 8 GB, 30 minutes on Quest short/p33100. Reuse the verified shared
Git module. New output root results/repair_round2/<commit>_<job>/, ignored by Git;
promote reviewed summaries and output hashes here.

This round also inspects whether existing metadata contains deployable age or
history signals; it does not substitute severity_id/condition/anchor_steps for
observable inputs. New feature extraction or rollout is not required to finish
this diagnostic round. If the learning problem persists, report the smallest
next change supported by these results; do not add a permanent research gate.
