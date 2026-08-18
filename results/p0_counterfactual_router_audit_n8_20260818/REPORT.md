# P0 counterfactual-router survival audit

Verdict: **NO-GO for expanding the current cohort**

Display point: `lambda=1`, `target=0.6`; 8 source states / 24 matched decisions.

| Method | Success | Catastrophe | Safe noncompletion | Intervention | Choices B/D/R |
|---|---:|---:|---:|---:|---:|
| Base | 66.67% | 33.33% | 0.00% | 0.00% | 24/0/0 |
| Always Detour | 70.83% | 4.17% | 25.00% | 100.00% | 0/24/0 |
| Always Retreat | 0.00% | 0.00% | 100.00% | 100.00% | 0/0/24 |
| Risk -> Retreat (rate-matched) | 41.67% | 8.33% | 50.00% | 50.00% | 12/0/12 |
| Risk -> Detour (rate-matched) | 83.33% | 12.50% | 4.17% | 50.00% | 12/12/0 |
| Risk -> best fixed option | 66.67% | 33.33% | 0.00% | 0.00% | 24/0/0 |
| Two-threshold Risk | 66.67% | 33.33% | 0.00% | 0.00% | 24/0/0 |
| Geometry Gate | 91.67% | 4.17% | 4.17% | 33.33% | 16/8/0 |
| Counterfactual Router | 87.50% | 8.33% | 4.17% | 58.33% | 10/14/0 |
| Counterfactual Oracle | 91.67% | 0.00% | 8.33% | 33.33% | 16/7/1 |

## Decision

The fixed nominal-path Geometry Gate is at least as good on success, catastrophe, and intervention, and strictly better on at least one metric. Do not enlarge this cohort yet; the present router advantage can be explained by a simpler geometry-triggered Detour policy.

Geometry evidence mode: `nominal_eef_swept_path_surface_clearance`.

## Evidence files

- `audit.json`: full choice matrices, calibration, thresholds, and prevalence audit.
- `main_table.csv`: compact method comparison.

Direct classifier/advantage-regression fresh evaluation is not identifiable from the saved fresh artifact because raw fresh features were not retained. The geometry and risk baselines above require no new rollout.
