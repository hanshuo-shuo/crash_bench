# P2 Dynamic Router trace morphology audit

**Finding:** simple temporal aggregation does not repair the treatment/control ordering. It partially promotes `heldout_0010`, whose Detour signal is sustained, but demotes `heldout_0019`, whose evidence is a short burst. Several benign controls also carry sustained high scores.

This is a diagnostic analysis of existing artifacts only: no model training, new rollout, or instantaneous-threshold search was performed.

## Cohort and definitions

- Stable cohort: 4 sources, 12 episodes; 11 episodes belong to the four requested groups.
- The remaining stable episode is `heldout_0019/noglass`, whose matched Base outcome is safe noncompletion. Its episode statistics are preserved, but it is not a Base-success control and is excluded from ranks and AUCs.
- Primary score: max non-Base advantage. Pointwise margin: `0.1086698193`; frozen sequential margin: `1.5974966343`.
- Trend is fitted to the last up-to-10 actions ending at trigger or collision; a no-event control uses trace end.

## Ranking of the two missed recoverable episodes

Ranks are descending among the 11 stable episodes in the four requested groups. Parentheses give rank among the two missed treatments plus seven Base-success controls.

| Statistic | heldout_0010 | heldout_0019 | Missed-v-control AUC | All recoverable-v-control AUC |
|---|---:|---:|---:|---:|
| Raw max | 9/11 (7/9); 1.005 | 6/11 (5/9); 1.296 | 0.357 | 0.571 |
| MA-3 max | 7/11 (6/9); 0.941 | 9/11 (7/9); 0.906 | 0.286 | 0.524 |
| MA-5 max | 6/11 (5/9); 0.921 | 10/11 (8/9); 0.698 | 0.286 | 0.524 |
| MA-8 max | 6/11 (5/9); 0.808 | 10/11 (8/9); 0.516 | 0.286 | 0.524 |
| Top-5 mean | 8/11 (6/9); 0.921 | 10/11 (8/9); 0.794 | 0.214 | 0.476 |
| Excess area | 9/11 (7/9); 7.069 | 11/11 (9/9); 4.134 | 0.071 | 0.095 |
| Longest > margin | 6/11 (5/9, tie 2); 11.000 | 11/11 (9/9); 6.000 | 0.179 | 0.214 |
| Candidate run | 6/11 (6/9, tie 2); 34.000 | 11/11 (9/9); 12.000 | 0.107 | 0.119 |
| Last-10 slope | 11/11 (9/9); -0.127 | 7/11 (6/9); -0.030 | 0.143 | 0.429 |

Raw maximum gives missed-v-control AUC `0.357`. The best temporal alternatives (MA-3 max, MA-5 max, MA-8 max) reach only `0.286`. No listed temporal statistic improves the pairwise ordering of both missed episodes.

## Are the control highs only one-step spikes?

No. The strongest no-glass control (`heldout_0008`) is spike-heavy, but other false positives remain high after smoothing and accumulation.

| Control | Raw max | MA-5 max | MA-5/raw | Excess area | Longest > margin |
|---|---:|---:|---:|---:|---:|
| heldout_0008/noglass | 1.635 | 0.830 | 0.508 | 11.98 | 7 |
| heldout_0008/offpath | 1.240 | 0.927 | 0.748 | 9.34 | 8 |
| heldout_0009/noglass | 0.791 | 0.581 | 0.734 | 5.63 | 13 |
| heldout_0009/offpath | 1.514 | 1.007 | 0.665 | 12.18 | 16 |
| heldout_0010/noglass | 1.368 | 1.177 | 0.861 | 17.69 | 19 |
| heldout_0010/offpath | 1.404 | 1.181 | 0.841 | 10.36 | 11 |
| heldout_0019/offpath | 0.862 | 0.830 | 0.963 | 18.50 | 20 |

Smoothing helps only locally: controls above `heldout_0010` fall from 5/7 under raw max to 4/7 under MA-5. For `heldout_0019`, they increase from 4/7 to 6/7 because its own maximum is less persistent.

## Detour evidence in the T-20 window

| Episode | Detour max | Detour actions > margin | Longest Detour > margin | Detour candidate actions | Longest Detour-candidate run |
|---|---:|---:|---:|---:|---:|
| heldout_0010 | 1.005 | 13/20 | 11 | 18/20 | 18 |
| heldout_0019 | 1.296 | 7/20 | 5 | 17/20 | 9 |

`heldout_0010` contains a real sustained Detour segment: 11 consecutive actions above the pointwise margin and Detour is the best non-Base option for 18/20 actions. `heldout_0019` has only a five-action Detour-above-margin burst. Its 17/20 Detour candidate count is not strong positive evidence: the candidate comparison ignores Base and often just says Retreat is worse.

## Diagnostic decision

1. **Simple evidence accumulator:** do not freeze one now. MA-3/5/8, top-5, area, longest run, candidate stability, and last-10 slope all fail to improve the two-missed-v-control AUC over raw maximum. Area and run length are especially confounded by long benign trajectories.
2. **Recovery-window supervision:** this is the minimum justified next experiment. Train or calibrate Detour intervention value against explicit action-level recoverability windows, freeze once, and evaluate on a new source-disjoint cohort. The present score can be temporally persistent while still semantically wrong on controls.
3. **Temporal value model:** defer it. It is a larger change and these 12 episodes do not show that a learned sequence model is needed before fixing the supervision target.

This is not a new project-level GO/NO-GO. The partial result is that smoothing removes one spike-driven control and improves `heldout_0010`'s rank, but the failure mode is heterogeneous and no single simple statistic covers both recoverable treatments.

## Unstable heldout_0015 diagnostic (excluded from all main results)

This older capture used the pointwise first-crossing margin, not the source-level sequential boundary. All three traces end at their early trigger, and the source was node-unstable, so these values are descriptive only and are not added to ranks or AUCs.

| Condition | Trace actions | Trigger | Collision | Raw max | MA-5 max |
|---|---:|---:|---:|---:|---:|
| glass | 17 | 16 | 30 | 0.339 | -0.125 |
| noglass | 25 | 24 | n/a | 0.394 | -0.075 |
| offpath | 30 | 29 | n/a | 0.210 | -0.068 |

## Artifacts

- Full per-action reconstructed sequences and rankings: `p2_trace_morphology_audit_20260819.json`.
- Flat per-episode statistics: `p2_trace_morphology_episode_stats_20260819.csv`.
- Trajectory and ranking figures: `p2_trace_morphology_trajectories_20260819.png` and `p2_trace_morphology_rankings_20260819.png`.
