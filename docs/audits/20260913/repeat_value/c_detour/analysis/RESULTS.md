# Prospective C execution results

| Method | Selection / evaluation H | A gain | B gain | C gain | A-C [95% CI] | Source RMSE A / B |
|---|---|---:|---:|---:|---|---|
| real_full | 220/220 | 4.17 | 2.08 | 3.12 | 1.04 [0.00, 3.12] | 4.17/4.17 |
| real_one | 220/220 | 4.17 | 2.08 | 2.08 | 2.08 [0.00, 6.25] | 8.33/0.00 |
| pseudo_one | 220/220 | 2.08 | 0.00 | -2.08 | 4.17 [0.00, 12.50] | 16.67/8.33 |
| pseudo_one_swapped | 220/220 | 6.25 | 2.08 | 0.00 | 6.25 [0.00, 12.50] | 14.43/8.33 |
| real_full | 440/440 | 6.25 | 4.17 | 5.21 | 1.04 [0.00, 3.12] | 4.17/4.17 |
| real_one | 440/440 | 6.25 | 4.17 | 4.17 | 2.08 [0.00, 6.25] | 8.33/0.00 |
| pseudo_one | 440/440 | 2.08 | 0.00 | -2.08 | 4.17 [0.00, 12.50] | 16.67/8.33 |
| pseudo_one_swapped | 440/440 | 6.25 | 2.08 | 0.00 | 6.25 [0.00, 12.50] | 14.43/8.33 |

All displayed values are percentage points. C is a noisy new execution block on existing sources.
Separate-execution forecasts are not assumed to improve; signed source error contrasts and all controls are retained in overall.csv.
