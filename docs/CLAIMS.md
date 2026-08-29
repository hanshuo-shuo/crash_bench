# Claim ledger

> **ICLR 2027 note (2026-08-29):** This C0--C14 document records the frozen
> pre-audit wording. The stricter [ICLR 2027 claim ledger](iclr27/CLAIM_LEDGER.md)
> adds evidence status, hashes, independent units, statistical limits, and
> forbidden stronger wording and is authoritative for the ICLR submission.

This is the paper-facing wording for the C0–C14 machine-readable ledger in
`results/claims_ledger.json`. “Frozen” means that a tracked summary exists; it
does not imply that ignored raw activations, logs, or videos are local.

Frozen working title: **Risk Is Not Intervention Value: Counterfactual Outcome
Routing for VLA Safety**.

## Headline claim

| ID | Exact supported claim | Evidence and scale | Scope / paper role |
|---|---|---|---|
| C14 | A frozen single-frame probabilistic outcome Router contributes a useful safety--success--intervention tradeoff at fresh matched T-20 decision states. At the independent-cohort `lambda=1,target=0.6` point it reaches 87.5% task success, 8.33% catastrophe, and 58.33% intervention versus 41.67%, 8.33%, and 50.0% for rate-matched Binary Risk -> Retreat; four predeclared frontier points meet all four acceptance criteria. | E16; 8 independent random-reset source states, 3 matched conditions per source, 24 decisions; 5,000 source-cluster bootstrap replicates; `counterfactual_router_fresh_online_20260817.json` and promoted n8 analysis | Learned routing over Base and privileged structured Detour/Retreat options. The display point is not a prespecified universal deployment point; combined n=13 has no joint all-criteria point. P2–P3.2 explicitly limits this claim to matched-state routing, not reliable sequential first crossing. |

One-sentence paper claim:

> At fresh matched decision states, option-conditioned counterfactual outcome
> prediction exposes useful safety--success--intervention tradeoffs that scalar
> risk routing cannot express.

## Causal and mechanism chain

| ID | Exact supported claim | Evidence and scale | Scope / paper role |
|---|---|---|---|
| C1 | With a visible matched wall, collision depends on swept-corridor intrusion: 15/15 on-path trials crash and 0/33 clear off-path trials crash; the transition region is graded. | `ood_control_final.json`; 5 on-path and 11 clear wall geometries; K=3; wall-level Fisher p=0.0002289 | One task and wall family; causal-localization headline. |
| C2 | The matched on-path/clear-off-path behavioral contrast appears in OpenVLA base, OpenVLA-OFT, and pi0. | each model: 5/5 on-path, 0/10 clear; `path3_oft_summary.json` | Action-head replication in the same embodiment/task, not independent task diversity. |
| C3 | Collision imminence is linearly decodable from frozen OpenVLA and OFT hidden states while safe behavior is not expressed. | T-5 AUC 0.998 and 0.903; `selfreport*/probe_summary.json` | Use “decoded but not routed”; pair with C5 rather than claiming mental state. |
| C5 | In the final pre-impact window, the policy does not sustain wall-directed EEF braking. | 25 wall/no-wall pairs; 0/25 retreat; command projection rises/equal/falls in 22/2/1; probe summaries and `ANALYSIS_selfreport.md` | EEF-aligned behavior, not full-arm signed-distance causality. |
| C13 | Hidden states contain collision-predictive information beyond measured task progress, EEF pose, and action magnitude. | strict held-out-scenario analysis; hidden AUPRC 0.716 vs strongest observable baseline 0.442; `task_phase_confound.json` | Associational, five wall scenarios; supporting dissociation evidence. |
| C7 | In the scoped OpenVLA-base on-path-wall mode, a wall-risk readout routed to `RetreatHold` changes 15/15 crashes to 0/15, 321.7 N mean peak force to 0 N, and fires on 0/22 benign rollouts. | `intervention/{episodes,summary}.json`; 5 treatment walls, K=3; online threshold -0.422 | Mechanism support for explicit routing; one fitted wall family and a structured safe-abort controller. |

## Supporting claims

| ID | Exact supported claim | Evidence and scale | Scope / placement |
|---|---|---|---|
| C0 | The OpenVLA–LIBERO bridge has a historical 400/500 nominal sanity result, but the expected manifest-backed gate artifact is not tracked. | old setup record; expected `m1_nominal_gate.json` | Entry condition only; exclude from abstract. |
| C4 | pi0 has partial, tap-dependent probe evidence rather than a settled architecture-wide decoding result. | T-5 AUC 0.728; action-expert T-3 0.902 and T-5 0.725 | Appendix until confounds and tap choice are closed. |
| C6 | Structured retreat safe-abort witnesses avoid all five wall scenarios at zero recorded wall force. | `witness.json`; one witness per scenario | Feasibility of safe abort, not task completion. |
| C8 | Frozen captures contain a wall-threshold operating window `[-0.7, 2.7]`; this does not validate threshold 1.0 online. | 53 positive and 4,400 negative frames; pooled AUC 0.7195; `shield/*.json` | Offline appendix support for C7. |
| C9 | A movable-glass hazard shows an along-path dose response: 30/50 on-path crashes and 0/50 matched off-path crashes across f30–f70. | `glass_prototype.json`; five matched pairs, K=10 | Second contact mechanism and safety–utility setup; do not report a selected f50–f70 average. |
| C10 | A glass-specific probe decodes glass risk, but the wall probe does not transfer to glass. | glass T-5 AUC 0.944; wall→glass AUC 0.359; `probe_glass_summary.json` | Supports hazard-specific readout; the repo lacks deployable glass probe weights. |
| C11 | Final-readout steering is a negative result: every tested alpha leaves wall crash at 100%, and only 0.742/7.688 of the probe-direction readout norm remains on the action-token slice. | 6 alphas × 10 treatment trials; `steering/*.json` | Compact mechanism ablation: detector direction ≠ controller direction. |
| C12 | A task-completing detour exists for a separately identified lowered d62 wall. | one low-wall replay/witness; 0 N | Existence demo only; never merge with the tall-wall treatment. |

## Paper-facing boundary claim

> Statewise counterfactual intervention value can improve matched-state routing,
> but neither simple temporal aggregation nor directly supervised
> recovery-window classification reliably transfers to fresh sequential
> first-crossing control.

This is a scope boundary for C14, not a negative headline claim.

| Stage | Key scale and result | Claim implication |
|---|---|---|
| P2 | 4 stable development sources × 3 conditions. Sequential Router improves task success from 58.3% to 66.7% and catastrophe from 33.3% to 25.0% at 16.7% intervention, but misses both known T-20 recoveries. | Limited but real secondary dynamic evidence; not reliable recovery-window detection. |
| P2.5 | Complete trajectory audit. Raw missed-recovery-v-control AUC `0.357`; simple moving-average variants `0.286`; accumulation, run length, stability, and trend do not repair ordering. | Simple temporal aggregation is ruled out as an adequate fix on these data. |
| P3.0 | 80 dense exact-state anchors: 28 recovery-open, 35 loss-control, 15 dense Base-preferred; 62 hard Base-success controls; 0 FailSafeHold contract violations. | Explicit dense supervision exists and is causally authored. |
| P3.1 | Strict LOSO over 9 sources. Recovery-open-v-hard-control AUC `1.000` versus old scalar `0.351`; intervention-needed-v-hard-control AUC `0.995` versus `0.338`; source-macro accuracy `0.632`, macro-F1 `0.540`, hard-control Base retention `0.984`, no global class collapse. | The frozen 10-D output representation supports useful offline statewise separation. |
| P3.2 | Fixed direct boundary `2.0724` from the original 10 calibration sources at `alpha=0.1`; 8 fresh sources × 3 conditions. Direct selects Base 24/24, recovers 0/8 glass episodes, and misses 2/2 known recoveries. Old P2 selects Detour 5/24, recovers 1/8 glass episodes, and retains control task success 16/16. | Direct Recovery Router is a final dynamic null and operationally collapses to Base. |

The P3.2 post-closeout known-recovery-v-control trajectory-max AUC is `0.250`.
Known-recovery maxima fall below many controls, so the result does not justify
another threshold or alpha search. Full evidence is in
[`P3_2_FROZEN_DYNAMIC_CLOSEOUT_20260819.md`](../results/P3_2_FROZEN_DYNAMIC_CLOSEOUT_20260819.md).

## Other non-numbered boundaries

E13 is a completed prompt-scope baseline, not a new claim ID. In the E16 fresh
cohort, the exact hazard prompt intervenes on 100%, reaches 37.5% success, and
has 66.67% unnecessary intervention. It supports broad behavioral caution, not
selective routing or recovery.

The old E14/E15 sequence-action recovery is supporting counterfactual evidence,
not a learned result. E16/C14 is the learned result, but Detour and Retreat
remain structured privileged options rather than a learned action head.

The checkpoint-readiness audit explains why D/F are not active claims: the
frozen gate has no threshold crossing in the six saved episodes, and validation
gripper-sign accuracy is below its gate.

## Explicitly prohibited paper claims

The paper must not claim:

- reliable recovery-window detection;
- end-to-end learned recovery;
- that “Knowing When to Intervene” has been solved;
- that Direct Recovery Router is a positive result;
- reliable fresh sequential first crossing by the E16 statewise Router;
- universal fixed-policy dominance, arbitrary glass-layout coverage, or
  production robustness;
- that Oracle or structured Detour/Hold is a learned action policy.

The recovery rescue line is **CLOSED**. No P3.3, temporal model, GRU,
Transformer, head refit, threshold/alpha tuning, or additional recovery cohort
may be implied as planned work.

## Deprecated wording

- “Knowing When to Intervene” as the paper title. Use **Risk Is Not Intervention
  Value** and treat sequential timing as a boundary.
- “The model knows it will crash.” Use “collision imminence is linearly
  decodable from the frozen representation.”
- “Architecture-independent.” Use “replicated across three action-head families
  in the same task/embodiment.”
- “Collision avoidance” for a safe abort.
- “Learned recovery” for Oracle actions or a structured controller.
- The selected-band category average in `headline_suite.json`.
