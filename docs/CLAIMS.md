# Claim ledger

This is the paper-facing wording for the C0–C13 machine-readable ledger in
`results/claims_ledger.json`. “Frozen” means a tracked summary exists; it does
not imply that ignored raw activations, logs, or videos are present locally.

## Primary chain

| ID | Exact supported claim | Evidence and scale | Scope / paper role |
|---|---|---|---|
| C1 | With a visible matched wall, collision depends on swept-corridor intrusion: 15/15 on-path trials crash and 0/33 clear off-path trials crash; the transition region is graded. | `ood_control_final.json`; 5 on-path and 11 clear wall geometries; K=3; wall-level Fisher p=0.0002289 | One task and wall family; causal-localization headline. |
| C2 | The matched on-path/clear-off-path behavioral contrast appears in OpenVLA base, OpenVLA-OFT, and pi0. | each model: 5/5 on-path, 0/10 clear; `path3_oft_summary.json` | Action-head replication in the same embodiment/task, not independent task diversity. |
| C3 | Collision imminence is linearly decodable from frozen OpenVLA and OFT hidden states while safe behavior is not expressed. | T-5 AUC 0.998 and 0.903; `selfreport*/probe_summary.json` | Use “decoded but not routed”; pair with C5 rather than claiming mental state. |
| C5 | In the final pre-impact window, the policy does not sustain wall-directed EEF braking. | 25 wall/no-wall pairs; 0/25 retreat; command projection rises/equal/falls in 22/2/1; probe summaries and `ANALYSIS_selfreport.md` | EEF-aligned behavior, not full-arm signed-distance causality. |
| C13 | Hidden states contain collision-predictive information beyond measured task progress, EEF pose, and action magnitude. | strict held-out-scenario analysis; hidden AUPRC 0.716 vs strongest observable baseline 0.442; `task_phase_confound.json` | Associational, five wall scenarios; supporting dissociation evidence. |
| C7 | In the scoped OpenVLA-base on-path-wall mode, a wall-risk readout routed to `RetreatHold` changes 15/15 crashes to 0/15, 321.7 N mean peak force to 0 N, and fires on 0/22 benign rollouts. | `intervention/{episodes,summary}.json`; 5 treatment walls, K=3; online threshold -0.422 | Closed-loop repair headline; one fitted wall family, structured safe-abort controller. |

## Supporting and boundary claims

| ID | Exact supported claim | Evidence and scale | Scope / placement |
|---|---|---|---|
| C0 | The OpenVLA–LIBERO bridge has a historical 400/500 nominal sanity result, but the expected manifest-backed gate artifact is not tracked. | old setup record; expected `m1_nominal_gate.json` | Entry condition only; exclude from abstract. |
| C4 | pi0 has partial, tap-dependent probe evidence rather than a settled architecture-wide decoding result. | T-5 AUC 0.728; action-expert T-3 0.902 and T-5 0.725 | Appendix until confounds and tap choice are closed. |
| C6 | Structured retreat safe-abort witnesses avoid all five wall scenarios at zero recorded wall force. | `witness.json`; one witness per scenario | Feasibility of safe abort, not task completion. |
| C8 | Frozen captures contain a wall-threshold operating window `[-0.7, 2.7]`; this does not validate threshold 1.0 online. | 53 positive and 4,400 negative frames; pooled AUC 0.7195; `shield/*.json` | Offline appendix support for C7. |
| C9 | A movable-glass hazard shows an along-path dose response: 30/50 on-path crashes and 0/50 matched off-path crashes across f30–f70. | `glass_prototype.json`; five matched pairs, K=10 | Second contact mechanism and safety–utility setup; do not report a selected f50–f70 average. |
| C10 | A glass-specific probe decodes glass risk, but the wall probe does not transfer to glass. | glass T-5 AUC 0.944; wall→glass AUC 0.359; `probe_glass_summary.json` | Supports hazard-specific readout; current repo lacks deployable glass probe weights. |
| C11 | Final-readout steering is a negative result: every tested alpha leaves wall crash at 100%, and only 0.742/7.688 of the probe-direction readout norm remains on the action-token slice. | 6 alphas × 10 treatment trials; `steering/*.json` | Compact mechanism ablation: detector direction ≠ controller direction. |
| C12 | A task-completing detour exists for a separately identified lowered d62 wall. | one low-wall replay/witness; 0 N | Existence demo only; never merge with the tall-wall treatment. |

## Non-numbered paper boundaries

E13 is a completed prompt-scope baseline, not a new claim ID. Hazard-specific
language changes behavior but yields 0/15 treatment task successes for both wall
and glass. The supported interpretation is conservative stopping.

E14/E15 glass recovery is supporting counterfactual evidence, not a learned
result. The broad frontier is a no-go. The scoped ledger certifies 3/15 accidents
(3/12 conditional on a Base catastrophe), and Oracle timing/actions produce 6/6
safe task successes on two development source states. The learned gate and learned
action head were not evaluated in a valid closed loop; the evaluation states were
also used for checkpoint validation.

The separate `glass_recovery_checkpoint_readiness_audit_20260812.json` records
why D/F are not active claims: the frozen gate has no threshold crossing in the
six saved episodes, and validation gripper-sign accuracy is below its gate.

Accordingly, the paper may say that a task-completing continuation physically
exists from exact matched states. It may not claim learned recovery, final-held-out
generalization, or arbitrary glass-layout coverage.

## Deprecated wording

- “The model knows it will crash.” Use “collision imminence is linearly
  decodable from the frozen representation.”
- “Architecture-independent.” Use “replicated across three action-head families
  in the same task/embodiment.”
- “Collision avoidance” for a safe abort.
- “Learned recovery” for Oracle actions or a structured controller.
- The selected-band category average in `headline_suite.json`.
