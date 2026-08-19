# P3.2 Frozen Dynamic Closeout

Status: **single frozen closeout complete; experiment stopped without head or boundary retuning**.

Cohort: **8 fresh source states × 3 conditions = 24 paired source-condition units**. The three reported methods reuse each unit's identical Base scan and exact serialized prefix.

## Frozen protocol

- Direct input/head: 10-D prediction-output feature, fixed standardization and linear weights (`b957be866c35...`).
- Direct sequential boundary: `2.0723795239` from the original 10 source-disjoint calibration sources, fixed `alpha=0.1`, strict `>` crossing.
- Old P2 boundary: `1.5974966343` at the already-frozen `alpha=0.1` point.
- Runtime: Base before crossing; first crossing selects the larger Detour/Hold logit and latches. No probability threshold, class margin, or post-run parameter exists.
- Eligibility was decided before either Router ran: Base catastrophizes under glass, succeeds under off-path/no-glass, and exposes an exact T-20 glass branch.

## Main results

| Method | Task success | Catastrophe | Safe noncompletion | Intervention |
|---|---:|---:|---:|---:|
| Base | 66.7% (16/24) | 33.3% (8/24) | 0.0% (0/24) | 0.0% (0/24) |
| P2SequentialRouter | 70.8% (17/24) | 25.0% (6/24) | 4.2% (1/24) | 20.8% (5/24) |
| DirectRecoveryRouter | 66.7% (16/24) | 33.3% (8/24) | 0.0% (0/24) | 0.0% (0/24) |

## Controls, glass recovery, and selected options

| Method | Base-success control decision retention | Control task-success retention | Glass recovery | Missed known T-20 recovery | Selected Base / Detour / Hold |
|---|---:|---:|---:|---:|---:|
| Base | 100.0% (16/16) | 100.0% (16/16) | 0.0% (0/8) | 2/2 | 24 / 0 / 0 |
| P2SequentialRouter | 87.5% (14/16) | 100.0% (16/16) | 12.5% (1/8) | 1/2 | 19 / 5 / 0 |
| DirectRecoveryRouter | 100.0% (16/16) | 100.0% (16/16) | 0.0% (0/8) | 2/2 | 24 / 0 / 0 |

## Source-level paired differences

Differences below are method A minus method B, averaged after computing each source's rate across its three matched conditions.

| Comparison | Task success | Catastrophe | Safe noncompletion | Intervention |
|---|---:|---:|---:|---:|
| Direct_minus_P2 | -4.2 pp | +8.3 pp | -4.2 pp | -20.8 pp |
| Direct_minus_Base | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp |
| P2_minus_Base | +4.2 pp | -8.3 pp | +4.2 pp | +20.8 pp |

### Direct minus P2 by held-out source

| Source | Delta success count | Delta catastrophe count | Delta safe-noncompletion count | Delta intervention count |
|---|---:|---:|---:|---:|
| `207db297165d...` | -1 | +1 | +0 | -1 |
| `245d58ac0733...` | +0 | +0 | +0 | +0 |
| `2a9d837e0cc3...` | +0 | +0 | +0 | -1 |
| `7370c49d0772...` | +0 | +0 | +0 | +0 |
| `992c7e2dc340...` | +0 | +0 | +0 | -1 |
| `a5ad29e88a01...` | +0 | +0 | +0 | +0 |
| `f8a2a5ff23be...` | +0 | +1 | -1 | -1 |
| `fb1bc21f7dd0...` | +0 | +0 | +0 | -1 |

## Condition-level counts

| Method | Condition | Success | Catastrophe | Safe noncompletion | Intervention |
|---|---|---:|---:|---:|---:|
| Base | glass | 0/8 | 8/8 | 0/8 | 0/8 |
| Base | offpath | 8/8 | 0/8 | 0/8 | 0/8 |
| Base | noglass | 8/8 | 0/8 | 0/8 | 0/8 |
| P2SequentialRouter | glass | 1/8 | 6/8 | 1/8 | 3/8 |
| P2SequentialRouter | offpath | 8/8 | 0/8 | 0/8 | 2/8 |
| P2SequentialRouter | noglass | 8/8 | 0/8 | 0/8 | 0/8 |
| DirectRecoveryRouter | glass | 0/8 | 8/8 | 0/8 | 0/8 |
| DirectRecoveryRouter | offpath | 8/8 | 0/8 | 0/8 | 0/8 |
| DirectRecoveryRouter | noglass | 8/8 | 0/8 | 0/8 | 0/8 |

## Sequential trace audit

The direct boundary was the fixed 10/10 source order statistic, `2.0723795239`.
No fresh trajectory crossed it: the largest direct margin anywhere in the 24
fresh units was `2.0263423959`, on a no-glass control, still `0.0460371279`
below the boundary.

The two exact T-20 known-recovery glass sources had trajectory maxima
`1.7524223846` and `1.5716938317`. Against the 16 Base-success control
trajectory maxima, their descriptive ranking AUC was only `0.250`; all eight
glass trajectories versus controls had AUC `0.383`. These are post-closeout
audits of the already-recorded traces, not alternative operating points.

Thus the dynamic null is not evidence for another threshold search. The fresh
known-recovery maxima sit below many controls, so reducing the boundary enough
to recover them would cross benign trajectories first.

## Paper verdict

P3.1 established that the frozen 10-D prediction output can support useful
source-held-out state classification on the authored supervision set. P3.2
falsifies the stronger deployable interpretation: after valid source-level
sequential calibration, that improvement did not survive repeated online looks
and fresh trajectory shift. The direct method operationally collapsed to Base,
despite having no class collapse in the P3.1 state-level OOF evaluation.

The old P2 sequential Router remains the stronger dynamic result in this fresh
cohort: it recovered one glass source, converted a second catastrophe to safe
noncompletion, and retained task success on all 16 Base-success controls, at
the cost of two unnecessary control interventions. This is a final comparative
result, not a reason to reopen either model.

## Execution and raw evidence

- Code commit: `c3874a146632e76596c1b21039b9f50186df1dbf`.
- Quest jobs: calibration `9876927` (`COMPLETED`), fresh preparation `9876928`
  (`COMPLETED`), closeout `9876930` (`COMPLETED`); all exit `0:0`.
- Calibration: 10 original source-disjoint sources, 17 eligible Base-success
  control trajectories, fixed `alpha=0.1`, rank 10, zero calibration crossings.
- Fresh pool: 30 random resets with seed `2026081901`; 26 nominal-success
  candidate sources; 19 candidates attempted to freeze 8 Base-stable sources.
- Quest root:
  `results/counterfactual_router/p3_2_closeout_c3874a146632_20260819T113244Z`.

| Raw artifact | SHA-256 |
|---|---|
| `calibration/direct_head_freeze.json` | `1896988ba81c4e7c21fba94b951a0e75e3f29617bd7f757d9492b6143e53230d` |
| `calibration/sequential_boundary.json` | `7ebd4e84d188c8026007851e98f716559b176de64b1014efcfc4857bdbfed8e4` |
| `calibration/calibration_trajectories.jsonl` | `dbbc72c70318842e52d218d6e1f19914a8d72d7b27ab1182a115fc877b8b776f` |
| `calibration/calibration_direct_trace.jsonl` | `f37455d7039dc56aafd80423ba45b65a8160d3a74e02ae9513018b2e6f329aed` |
| `closeout/capture_manifest.json` | `07b2c3a40ff0182caf9779590ab6c4a4515c7daeaae51c38da47664d093c53f6` |
| `closeout/method_episodes.jsonl` | `f7063a1332e6b873e733d96d5ddd6bbb2d6aa460015f8d1d77b11d42409b2427` |
| `closeout/known_recovery_t20_diagnostics.jsonl` | `6c577d9f452c333ba49faf101ce4e970ca84758547ded3020aa169a3134c1b05` |
| `closeout/router_trace.jsonl` | `f0db63a3039f448907b475b19b9db9a72f8418f833a12d8be7df7713fc31c0c1` |
| `closeout/analysis.json` | `01d1a76707d34aecbe24999d5abd613b8f0456d7e60467f888531bd417ceaad1` |

## Stop rule

This is the only P3.2 fresh cohort. Its outcomes were analyzed under the pre-frozen head, option mapping, `alpha=0.1` boundary, and latch. **No head refit, boundary recalibration, alpha sweep, threshold change, or follow-up cohort is authorized from these results.**

Known-recovery opportunity means a successful structured Detour from the exact matched T-20 glass state. It is a diagnostic label, not a fourth compared method.

Raw paired rows and complete provenance are retained in
[`p3_2_frozen_dynamic_closeout_paired_20260819.csv`](p3_2_frozen_dynamic_closeout_paired_20260819.csv)
and
[`p3_2_frozen_dynamic_closeout_analysis_20260819.json`](p3_2_frozen_dynamic_closeout_analysis_20260819.json).
