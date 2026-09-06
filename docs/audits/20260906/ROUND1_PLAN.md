# First repair round — authorized development revision

Date: 2026-09-06. Base: ODUR commit `c3378366d53cf514914ae837292126dbd806c4b2`.
Branch: `codex/audit-repair-round1`.

The user's request to implement the audit replaces the older publication-only
next action for this round. Historical results and claims remain frozen.

1. F1: report contains_B0, contains_B1, contains_both, entirely_B0 separately in
   all four D2 screens and D5/D8 analyzers. Append a retrospective erratum using
   committed data; no historical analysis is overwritten or new test authorized.
2. F3: retain the additive model as a versioned ablation; default to nonlinear
   state-option interaction; add independent option heads. Preserve old checkpoint
   loading by interpreting missing architecture metadata as additive.
3. F5: fit on train only, calibrate on disjoint calibration sources, evaluate on
   development. Reject source leakage and mismatched calibration/model hashes.
   Final train+development refit, if later needed, is a separate stage and cannot
   be evaluated as independent development evidence.
4. F2/F4: compare point choice, benefit gate at a fixed 0.05 predicted utility gain,
   and the historical conformal wrapper. Keep alpha=0.10 and the original safety
   limits. Report structural calibration failure explicitly; feasibility is not
   conditional risk certification. Remove development intervention quotas.
5. F7: preserve old README/CURRENT byte-for-byte; current entry points may evolve.
   Tests verify source support, architecture counterexamples and data roles.

Experiment budget: existing D5 data and the existing 104-D features, three
architectures × five seeds (0–4), 100 full-batch epochs each, AdamW lr=0.001,
weight decay=0.0001, unchanged losses. DirectQ and other fixed baselines use the
same train/development data. Two CPU cores, 8 GB, at most 30 minutes on Quest
short/p33100. No simulator rollout or GPU allocation is needed.

Report source-macro utility, task success and catastrophe, choices, B1 beneficial
intervention recall and B0 unnecessary intervention rate. Compare architectures
with source-paired bootstrap intervals (10,000 resamples, seed 20260906), explicitly
unadjusted for repeated development/model selection. Optimization seeds are not
independent physical sources. D8 is read only by the separate retrospective
support erratum; it is never fed into training, calibration or model comparisons.

New outputs: ignored `results/repair_round1/<commit>_<job>/`; promote reviewed
reports under this audit directory. Record code commit, input hashes, seeds,
training roles, Slurm job and runtime. No automated follow-on test collection.

Deferred: F6 non-glass geometry/physical preflight, richer features, new sources,
VLA hidden states, online timing, learned recovery actions and confirmatory tests.
The round ends with implementation checks and the bounded development report;
a weak method result prompts diagnosis rather than a permanent research ban.
