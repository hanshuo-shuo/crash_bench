# ICLR 2027 claim ledger

This is the paper-facing ledger for the ICLR 2027 submission decision. A
`supported` status means the exact scoped wording is backed by the listed
frozen artifact; it does not remove the stated limitations. `conditional`
means the descriptive result exists but the intended interpretation needs an
additional baseline, statistical, provenance, or scope check. `unsupported`
means the sentence must not appear as a result claim.

Artifact bytes and selected JSON values are pinned in
`results/iclr27/manifest.json`. Historical missing hashes are written as
`unavailable_historical_not_recorded`; they are never inferred.

## Headline and supporting claims

### C0 — nominal bridge gate

```yaml
claim_id: C0
paper_wording: >-
  The repository records a historical 400/500 OpenVLA--LIBERO nominal sanity
  result, but no manifest-backed result artifact is tracked.
status: unsupported
independent_unit: nominal episode
n_sources: unavailable_historical_not_recorded
cohort: historical LIBERO-Spatial tasks 0--9
method_hash: unavailable_historical_not_recorded
protocol_hash: unavailable_historical_not_recorded
result_files: []
statistical_test: none; expected result artifact is absent
limitations:
  - historical narrative only
  - no tracked manifest, checkpoint revision, or raw rows
forbidden_stronger_wording:
  - the OpenVLA--LIBERO bridge passes a reproducible 80% nominal gate
  - 400/500 is a claim-ready paper result
```

### C1 — swept-corridor wall causality

```yaml
claim_id: C1
paper_wording: >-
  In the evaluated matched wall family, 15/15 on-path trials crash and 0/33
  clear off-path trials crash; wall-geometry Fisher exact p=0.0002289.
status: supported
independent_unit: wall geometry; K=3 trials are repeats within geometry
n_sources: 16
cohort: 5 on-path and 11 clear wall geometries, OpenVLA base, LIBERO task 0
method_hash: unavailable_historical_not_recorded
protocol_hash: unavailable_historical_not_recorded
result_files:
  - results/ood_control_final.json
statistical_test: wall-level Fisher exact test, p=0.00022893772893772894
limitations:
  - one task and one wall family
  - historical run commit and exact checkpoint revision were not recorded
forbidden_stronger_wording:
  - wall presence alone causes collision
  - the causal result generalizes to arbitrary obstacles, tasks, or robots
```

### C2 — action-head replication

```yaml
claim_id: C2
paper_wording: >-
  The matched on-path/clear-off-path contrast appears for OpenVLA base,
  OpenVLA-OFT, and pi0: each has 5/5 on-path crashes and 0/10 clear crashes.
status: conditional
independent_unit: wall geometry nested within model
n_sources: 15 per model
cohort: same LIBERO task, embodiment, and matched wall/control geometry across three policy families
method_hash: unavailable_historical_not_recorded
protocol_hash: unavailable_historical_not_recorded
result_files:
  - results/path3_oft_summary.json
statistical_test: descriptive matched counts; no cross-task or cross-embodiment inference
limitations:
  - model results share task, embodiment, and geometry
  - exact checkpoint revisions and repeat provenance are incomplete
forbidden_stronger_wording:
  - architecture-independent causal law
  - replication across independent tasks or embodiments
```

### C3 — OpenVLA/OFT collision readout

```yaml
claim_id: C3
paper_wording: >-
  Collision imminence is linearly decodable from frozen OpenVLA and OFT hidden
  states at T-5, with compact-summary ROC-AUC 0.998 and 0.903 respectively.
status: supported
independent_unit: held-out scenario; frames are correlated observations
n_sources: unavailable_in_compact_summary
cohort: matched wall/no-wall frozen captures for OpenVLA base and OFT
method_hash: unavailable_historical_not_recorded
protocol_hash: unavailable_historical_not_recorded
result_files:
  - results/selfreport/probe_summary.json
  - results/selfreport_oft/probe_summary.json
statistical_test: held-out probe ROC-AUC; raw hidden/meta are not tracked
limitations:
  - compact summary does not retain every fold/source row
  - decoding is not evidence of a mental state or expressed safe control
forbidden_stronger_wording:
  - the VLA knows it will crash
  - hidden-state decoding proves causal control relevance
```

### C4 — pi0 probe evidence

```yaml
claim_id: C4
paper_wording: >-
  pi0 has partial tap-dependent decoding evidence: policy-tap T-5 ROC-AUC is
  0.728, while action-expert results vary by tap/horizon.
status: conditional
independent_unit: held-out scenario; frames are correlated observations
n_sources: unavailable_in_compact_summary
cohort: pi0 matched wall/no-wall frozen captures
method_hash: unavailable_historical_not_recorded
protocol_hash: unavailable_historical_not_recorded
result_files:
  - results/selfreport_pi0/probe_summary.json
  - results/selfreport_pi0_ae/probe_summary.json
statistical_test: held-out ROC-AUC summaries
limitations:
  - tap choice is unresolved
  - task-phase and architecture-wide confounds are not closed
forbidden_stronger_wording:
  - pi0 robustly encodes collision at all internal taps
  - the finding is architecture independent
```

### C5 — unsafe action persistence

```yaml
claim_id: C5
paper_wording: >-
  Across 25 matched wall/no-wall pairs, the final pre-impact EEF command does
  not show sustained retreat; retreat occurs in 0/25 wall episodes.
status: supported
independent_unit: matched wall/no-wall episode pair
n_sources: 25
cohort: final-window behavior across frozen base/OFT/pi0 probe captures
method_hash: unavailable_historical_not_recorded
protocol_hash: unavailable_historical_not_recorded
result_files:
  - results/selfreport/probe_summary.json
  - results/selfreport_oft/probe_summary.json
  - results/selfreport_pi0/probe_summary.json
statistical_test: descriptive paired final-window action audit
limitations:
  - EEF-aligned projection is not full-arm signed-distance causality
  - supporting summaries are historical and raw aligned frames are ignored
forbidden_stronger_wording:
  - the robot makes no avoidance motion of any kind
  - decoded risk can never affect policy behavior
```

### C6 — structured safe-abort witnesses

```yaml
claim_id: C6
paper_wording: >-
  One structured retreat safe-abort witness avoids each of five evaluated wall
  scenarios with zero recorded wall force.
status: conditional
independent_unit: wall scenario witness
n_sources: 5
cohort: five scoped on-path wall scenarios
method_hash: unavailable_historical_not_recorded
protocol_hash: unavailable_historical_not_recorded
result_files:
  - results/witness.json
statistical_test: five deterministic existence witnesses; no population inference
limitations:
  - one witness per scenario
  - safe abort does not complete the task
forbidden_stronger_wording:
  - RetreatHold is a general recovery policy
  - witness rollouts establish task-preserving safety
```

### C7 — scoped readout-to-controller intervention

```yaml
claim_id: C7
paper_wording: >-
  In the scoped OpenVLA-base on-path-wall mode, routing a wall-risk readout to
  RetreatHold changes 15/15 crashes to 0/15, mean peak force from 321.7 N to
  0 N, and fires on 0/22 benign rollouts at threshold -0.422.
status: supported
independent_unit: wall geometry for treatment; benign rollout for false-fire audit
n_sources: 5 treatment wall geometries plus 22 benign rollouts
cohort: OpenVLA base, LIBERO task 0, fitted wall family
method_hash: unavailable_historical_not_recorded
protocol_hash: unavailable_historical_not_recorded
result_files:
  - results/intervention/summary.json
  - results/intervention/episodes.json
statistical_test: raw scoped counts; treatment repeats K=3 within five geometries
limitations:
  - structured safe-abort controller, not task recovery
  - threshold and hazard family are scoped
forbidden_stronger_wording:
  - learned end-to-end collision avoidance
  - reliable online guard across tasks or hazard families
```

### C8 — offline wall threshold window

```yaml
claim_id: C8
paper_wording: >-
  Frozen wall captures contain an offline zero-false-positive threshold window
  [-0.7, 2.7] over 53 positive and 4400 negative frames; the independently
  validated online threshold is -0.422, not the window midpoint 1.0.
status: conditional
independent_unit: scenario/rollout; frames are not independent samples
n_sources: 5 on-path, 5 no-wall, and 21 off-path scenario captures
cohort: frozen OpenVLA wall-risk capture and sweep
method_hash: unavailable_historical_not_recorded
protocol_hash: unavailable_historical_not_recorded
result_files:
  - results/shield/summary.json
  - results/shield/sweep.json
statistical_test: descriptive threshold sweep; online validation belongs to C7
limitations:
  - pooled frame counts cannot support frame-independent inference
  - threshold 1.0 was not validated online
forbidden_stronger_wording:
  - threshold 1.0 is a validated deployment threshold
  - the window is a confidence interval
```

### C9 — glass dose response

```yaml
claim_id: C9
paper_wording: >-
  Across five matched along-path/off-path placement pairs with K=10, glass
  causes 30/50 on-path crashes and 0/50 off-path crashes.
status: supported
independent_unit: matched placement pair; K=10 trials are repeats
n_sources: 5
cohort: f30--f70 movable-glass placements in one LIBERO task
method_hash: unavailable_historical_not_recorded
protocol_hash: unavailable_historical_not_recorded
result_files:
  - results/glass_prototype.json
statistical_test: matched dose-response counts; no selected-band inference
limitations:
  - one task and one glass geometry family
  - outcome nondeterminism within placements
forbidden_stronger_wording:
  - a selected f50--f70 average is a preregistered result
  - glass safety generalizes across layouts
```

### C10 — glass-specific readout

```yaml
claim_id: C10
paper_wording: >-
  The compact capture reports glass-specific T-5 ROC-AUC 0.944 and wall-to-glass
  transfer ROC-AUC 0.359.
status: conditional
independent_unit: held-out placement/source; frames are correlated
n_sources: unavailable_in_compact_summary
cohort: frozen glass probe capture and wall-to-glass transfer audit
method_hash: unavailable_historical_not_recorded
protocol_hash: unavailable_historical_not_recorded
result_files:
  - results/selfreport_glass/probe_glass_summary.json
statistical_test: held-out ROC-AUC summaries
limitations:
  - raw hidden/meta and serialized deployable glass probe are not tracked
  - source count is absent from the compact summary
forbidden_stronger_wording:
  - a deployable glass detector is available
  - the wall detector transfers to glass
```

### C11 — final-readout steering null

```yaml
claim_id: C11
paper_wording: >-
  Final-readout activation steering is negative in the scoped wall study: all
  six tested alphas retain 100% crash, and only 0.742/7.688 of the full
  probe-direction readout norm lies on the action-token slice.
status: supported
independent_unit: source rollout block within alpha; alphas are not independent replications
n_sources: 10 treatment trials per alpha
cohort: six-alpha OpenVLA final-readout steering ablation on the wall family
method_hash: unavailable_historical_not_recorded
protocol_hash: unavailable_historical_not_recorded
result_files:
  - results/steering/summary.json
  - results/steering/diag.json
statistical_test: descriptive negative ablation across all pretested alphas
limitations:
  - one layer/readout direction and hazard family
  - a scoped null does not rule out other layers or steering methods
forbidden_stronger_wording:
  - VLA activation steering cannot change behavior
  - detector directions are never controller directions
```

### C12 — low-wall task-completing witness

```yaml
claim_id: C12
paper_wording: >-
  A task-completing, zero-recorded-force detour exists for one separately
  fingerprinted lowered d62 wall.
status: conditional
independent_unit: scenario witness
n_sources: 1
cohort: lowered d62 detour scenario, distinct from the tall-wall treatment
method_hash: unavailable_historical_not_recorded
protocol_hash: unavailable_historical_not_recorded
result_files:
  - results/phase2_task_witness/summary.json
statistical_test: one existence witness; no population inference
limitations:
  - separate easier geometry
  - one replay/witness only
forbidden_stronger_wording:
  - the tall-wall treatment is task-completing recoverable
  - Detour is validated generally
```

### C13 — hidden information beyond measured observables

```yaml
claim_id: C13
paper_wording: >-
  In five held-out wall scenarios, hidden-only macro-AUPRC is 0.716 versus
  0.442 for the strongest measured observable baseline; the exact grouped
  sign-permutation p-value for the paired difference is 0.25.
status: conditional
independent_unit: held-out wall scenario
n_sources: 5
cohort: frozen task-phase confound audit over measured progress, EEF pose, and action magnitude
method_hash: unavailable_historical_not_recorded
protocol_hash: unavailable_historical_not_recorded
result_files:
  - results/task_phase_confound/task_phase_confound.json
statistical_test: 20000-sample scenario bootstrap CI [0.0303,0.5162]; exact grouped sign permutation p=0.25
limitations:
  - associational and underpowered with five groups
  - only measured observables are excluded; unmeasured task phase can remain
forbidden_stronger_wording:
  - hidden state causally represents collision
  - the hidden advantage is statistically established at conventional alpha
```

### C14 — fresh matched-state outcome-routing frontier

```yaml
claim_id: C14
paper_wording: >-
  At fresh matched T-20 decision states, a frozen option-conditioned outcome
  router exposes a useful safety--success--intervention frontier relative to
  the evaluated baselines. At the exploratory display point lambda=1,
  target=0.6, it reaches 21/24 task successes, 2/24 catastrophes, and 14/24
  interventions versus 10/24, 2/24, and 12/24 for Binary Risk -> Retreat.
status: supported
independent_unit: source_state_sha256
n_sources: 8 independent confirmation sources; 13 in pooled descriptive analysis
cohort: 3 matched conditions per source; 24 independent-cohort and 39 pooled decisions
method_hash: 757e220bc2e561aa8207ae3778a3021554b3031df2ec3eef376e1e56db938052
protocol_hash: 5b50f42ce3dbfc16255e550939c4df918a77611cf06101f7cd44faffaddb93cd
result_files:
  - results/counterfactual_router_fresh_online_20260817.json
  - results/counterfactual_router_fresh_online_main_table.csv
  - results/counterfactual_router_fresh_online_n8_analysis.json
  - results/counterfactual_router_fresh_online_n8_frontier.csv
  - results/counterfactual_router_fresh_online_n13_analysis.json
  - results/counterfactual_router_fresh_online_n13_frontier.csv
statistical_test: 5000 shared source-cluster bootstrap replicates; exact paired source inference not yet reported
limitations:
  - display point selected from a predeclared 45-point test frontier
  - four n=8 all-criteria points, but no joint all-criteria point at pooled n=13
  - Binary Risk is restricted to Retreat and strongest direct/value baselines are absent
  - Detour uses privileged geometry
  - one task, one main VLA, and one main hazard family
forbidden_stronger_wording:
  - outcome decomposition beats risk routing in general
  - lambda=1,target=0.6 was a prespecified confirmatory operating point
  - the router is universally safer than Always Detour
  - end-to-end learned recovery
  - reliable sequential first-crossing intervention
```

## Sequential boundary claims

### C14-P2 — development sequential point

```yaml
claim_id: C14-P2
paper_wording: >-
  On four stable development sources and 12 condition episodes, the frozen P2
  sequential router changes task success from 7/12 to 8/12 and catastrophe
  from 4/12 to 3/12 at 2/12 intervention, while missing 2/2 known T-20
  recovery opportunities.
status: conditional
independent_unit: source_state_sha256
n_sources: 4
cohort: development-only stable sources, three conditions per source
method_hash: 757e220bc2e561aa8207ae3778a3021554b3031df2ec3eef376e1e56db938052
protocol_hash: 2764f2c823dc6b41e13102e7c2d6dc9770fcc524d15895762bfd83a8412e8533
result_files:
  - results/P2_SEQUENTIAL_FIRST_CROSSING_DEV_20260819.md
statistical_test: raw development source/episode counts; no confirmatory test inference
limitations:
  - development-only and n=4
  - misses both known matched-state recovery opportunities
forbidden_stronger_wording:
  - reliable sequential improvement
  - successful recovery-window detection
```

### C14-P2.5 — temporal aggregation closeout

```yaml
claim_id: C14-P2.5
paper_wording: >-
  On the frozen development trajectories, missed-recovery-versus-control AUC
  is 0.357 for raw maximum and 0.286 for MA-3/5/8; tested smoothing,
  accumulation, run-length, option-stability, and trend summaries do not repair
  the ordering.
status: supported
independent_unit: source_state_sha256 trajectory
n_sources: 4 stable primary sources
cohort: 12 stable development episodes; 3 unstable episodes retained as separate diagnostics
method_hash: ca158058f52d83d9a253a8bfda7120ec7249ac0b8def40a7d25b2978ea573b2d
protocol_hash: 5ecfd5f7de24ddc4d9a999ecc0fbddbd6c088d92c4017bb9e00fc1d58d7d6a4a
result_files:
  - results/p2_trace_morphology_audit_20260819.json
statistical_test: complete frozen-trajectory ranking audit; development-only descriptive AUC
limitations:
  - very small development cohort
  - rules out only the tested simple summaries on these trajectories
forbidden_stronger_wording:
  - all temporal models fail
  - sequential intervention is impossible
```

### C14-P3.0 — dense recovery-window supervision

```yaml
claim_id: C14-P3.0
paper_wording: >-
  Dense exact-state authoring yields 80 anchors (28 recovery-open, 35
  loss-control, 15 dense Base-preferred), 62 hard Base-success controls, and
  zero FailSafeHold contract violations.
status: supported
independent_unit: source_state_sha256; anchors within source are correlated
n_sources: 9
cohort: development-only exact-state recovery-window authoring
method_hash: unavailable_as_single_hash; all records are content-pinned in the ICLR manifest
protocol_hash: unavailable_as_single_hash; trajectory artifact is content-pinned in the ICLR manifest
result_files:
  - results/p3_recovery_window_records_20260819.csv
  - results/p3_recovery_window_trajectories_20260819.json
statistical_test: complete authored-anchor counts and controller-contract audit
limitations:
  - development supervision, not a fresh online result
  - anchors and controls are not independent samples
forbidden_stronger_wording:
  - 142 independent recovery examples
  - dense labels establish online recoverability
```

### C14-P3.1 — direct-head offline separation

```yaml
claim_id: C14-P3.1
paper_wording: >-
  Under strict leave-one-source-out evaluation over nine development sources,
  the direct head reaches recovery-open-v-hard-control AUC 1.000 and
  intervention-needed-v-hard-control AUC 0.995, with source-macro accuracy
  0.632, macro-F1 0.540, and hard-control Base retention 0.984.
status: supported
independent_unit: source_state_sha256
n_sources: 9
cohort: development-only dense recovery-window records with strict LOSO
method_hash: b957be866c35338e709a4eebaebf88257cc97772ef04dd44d82ba81fff931789
protocol_hash: 37af9140ab001b3bc3ad7c112dc91573bbd53a76b677393a0f83e6d9d1ce1f5f
result_files:
  - results/p3_1_direct_recovery_head_analysis_20260819.json
  - results/p3_1_direct_recovery_head_model.npz
  - results/p3_1_direct_recovery_head_oof_predictions_20260819.csv
statistical_test: strict leave-one-source-state-out predictions and source-macro metrics
limitations:
  - development-only statewise analysis
  - high AUC did not transfer to the P3.2 sequential closeout
forbidden_stronger_wording:
  - the direct head reliably detects fresh recovery windows
  - offline anchor-level performance is online sequential performance
```

### C14-P3.2 — fresh sequential negative closeout

```yaml
claim_id: C14-P3.2
paper_wording: >-
  With alpha=0.1 and the rank-10 source boundary 2.0723795239 frozen before
  evaluation, the Direct Recovery Router selects Base in 24/24 fresh episodes,
  recovers 0/8 glass episodes, and misses 2/2 known recoveries; the old P2
  router selects Detour in 5/24 and recovers 1/8 glass episodes.
status: supported
independent_unit: source_state_sha256
n_sources: 8
cohort: one fresh negative-closeout cohort, three conditions and three compared methods per source
method_hash: b957be866c35338e709a4eebaebf88257cc97772ef04dd44d82ba81fff931789
protocol_hash: 7ebd4e84d188c8026007851e98f716559b176de64b1014efcfc4857bdbfed8e4
result_files:
  - results/p3_2_direct_head_freeze_20260819.json
  - results/p3_2_direct_sequential_boundary_20260819.json
  - results/p3_2_frozen_dynamic_closeout_analysis_20260819.json
  - results/p3_2_frozen_dynamic_closeout_episodes_20260819.jsonl
  - results/p3_2_frozen_dynamic_closeout_paired_20260819.csv
  - results/p3_2_frozen_dynamic_closeout_trace_audit_20260819.json
statistical_test: complete source-paired closeout; known-recovery-v-control trajectory-max AUC 0.250
limitations:
  - one task and hazard family
  - negative result closes this protocol, not every possible sequential method
forbidden_stronger_wording:
  - Direct Recovery Router is a positive method result
  - lowering the threshold is a justified post-test repair
  - reliable fresh sequential first crossing
```

## Global prohibited wording

- “first selective VLA intervention method,” “first sequential VLA verifier,”
  or “first hidden-state VLA failure detector”;
- “causal estimator” in the observational-inference sense—the simulator gives
  full-information interventional labels at authored exact states;
- “confidence bound” for `delta`;
- “confirmatory operating point” for `lambda=1,target=0.6`;
- “general,” “robust,” “production-ready,” or “architecture-independent” for
  the E16 routing result;
- any sentence that treats frames, conditions, branches, anchors, horizons, or
  frontier points as independent samples.
