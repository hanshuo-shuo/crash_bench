# Claim ledger

This is the paper-facing wording for the C0–C14 machine-readable ledger in
`results/claims_ledger.json`. “Frozen” means a tracked summary exists; it does
not imply that ignored raw activations, logs, or videos are present locally.

## Headline claim

| ID | Exact supported claim | Evidence and scale | Scope / paper role |
|---|---|---|---|
| C14 | A frozen single-frame probabilistic outcome router contributes a new safety--success--intervention tradeoff at fresh matched T-20 decision states. At the independent-cohort `lambda=1,target=0.6` point it reaches 87.5% task success, 8.33% catastrophe, and 58.33% intervention versus 41.67%, 8.33%, and 50.0% for rate-matched Binary Risk -> Retreat; four predeclared frontier points meet all four acceptance criteria. | E16; 8 independent random-reset source states, 3 matched conditions per source, 24 decisions; 5,000 source-cluster bootstrap replicates; `counterfactual_router_fresh_online_20260817.json` and promoted n8 analysis | Learned routing over Base and privileged structured Detour/Retreat options. The display point is not a prespecified single deployment point; combined n=13 has no joint all-criteria point; P2/P2.5 does not establish reliable from-reset timing. |

## Causal and mechanism chain

| ID | Exact supported claim | Evidence and scale | Scope / paper role |
|---|---|---|---|
| C1 | With a visible matched wall, collision depends on swept-corridor intrusion: 15/15 on-path trials crash and 0/33 clear off-path trials crash; the transition region is graded. | `ood_control_final.json`; 5 on-path and 11 clear wall geometries; K=3; wall-level Fisher p=0.0002289 | One task and wall family; causal-localization headline. |
| C2 | The matched on-path/clear-off-path behavioral contrast appears in OpenVLA base, OpenVLA-OFT, and pi0. | each model: 5/5 on-path, 0/10 clear; `path3_oft_summary.json` | Action-head replication in the same embodiment/task, not independent task diversity. |
| C3 | Collision imminence is linearly decodable from frozen OpenVLA and OFT hidden states while safe behavior is not expressed. | T-5 AUC 0.998 and 0.903; `selfreport*/probe_summary.json` | Use “decoded but not routed”; pair with C5 rather than claiming mental state. |
| C5 | In the final pre-impact window, the policy does not sustain wall-directed EEF braking. | 25 wall/no-wall pairs; 0/25 retreat; command projection rises/equal/falls in 22/2/1; probe summaries and `ANALYSIS_selfreport.md` | EEF-aligned behavior, not full-arm signed-distance causality. |
| C13 | Hidden states contain collision-predictive information beyond measured task progress, EEF pose, and action magnitude. | strict held-out-scenario analysis; hidden AUPRC 0.716 vs strongest observable baseline 0.442; `task_phase_confound.json` | Associational, five wall scenarios; supporting dissociation evidence. |
| C7 | In the scoped OpenVLA-base on-path-wall mode, a wall-risk readout routed to `RetreatHold` changes 15/15 crashes to 0/15, 321.7 N mean peak force to 0 N, and fires on 0/22 benign rollouts. | `intervention/{episodes,summary}.json`; 5 treatment walls, K=3; online threshold -0.422 | Mechanism support for explicit routing; one fitted wall family and a structured safe-abort controller. |

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

E13 is a completed historical prompt-scope baseline, not a new claim ID.
Hazard-specific language changes behavior but yields 0/15 treatment task
successes for both wall and glass. E16 reruns the exact glass prompt from reset
inside the fresh matched cohort: it intervenes on 100%, reaches 37.5% success,
and has 66.67% unnecessary intervention. The supported interpretation is broad
behavioral caution rather than selective routing or recovery.

The old E14/E15 sequence-action recovery remains supporting counterfactual
evidence rather than a learned result. E16/C14 is the learned result, but its
Detour and Retreat actions remain structured options rather than a learned
action head. Complete interpretation is frozen in
[COUNTERFACTUAL_ROUTER_MAIN_RESULT.md](COUNTERFACTUAL_ROUTER_MAIN_RESULT.md).

The separate `glass_recovery_checkpoint_readiness_audit_20260812.json` records
why D/F are not active claims: the frozen gate has no threshold crossing in the
six saved episodes, and validation gripper-sign accuracy is below its gate.

P2/P2.5 defines an additional scope boundary for C14. Source-calibrated dynamic
first crossing produces a selective improvement on four stable development
sources, but misses both known T-20-recoverable episodes. The trajectory audit
finds missed-treatment-versus-control AUC 0.357 for raw maximum and 0.286 for
MA-3/5/8; control highs include sustained evidence rather than only one-step
spikes. C14 therefore supports learned option-value routing at matched decision
states, not reliable from-reset recovery-window detection. The diagnostic is
tracked in
[`P2_TRACE_MORPHOLOGY_AUDIT_20260819.md`](../results/P2_TRACE_MORPHOLOGY_AUDIT_20260819.md).

Accordingly, the paper may say that a task-completing continuation physically
exists from exact matched states and that a frozen learned router recovers a
useful fresh online frontier over those options. It may not claim end-to-end
learned recovery, universal fixed-policy dominance, or arbitrary glass-layout
coverage, and it may not claim reliable from-reset timing by the current score.

## Deprecated wording

- “The model knows it will crash.” Use “collision imminence is linearly
  decodable from the frozen representation.”
- “Architecture-independent.” Use “replicated across three action-head families
  in the same task/embodiment.”
- “Collision avoidance” for a safe abort.
- “Learned recovery” for Oracle actions or a structured controller.
- The selected-band category average in `headline_suite.json`.
