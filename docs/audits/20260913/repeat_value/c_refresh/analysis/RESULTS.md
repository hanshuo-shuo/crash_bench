# Prospective C execution results

| Method | Selection / evaluation H | A gain | B gain | C gain | A-C [95% CI] | Source RMSE A / B |
|---|---|---:|---:|---:|---|---|
| real_full | 100/100 | 7.03 | 5.47 | 2.34 | 4.69 [-6.25, 14.06] | 15.93/20.73 |
| real_one | 100/100 | 12.50 | 3.12 | 0.00 | 12.50 [0.00, 31.25] | 25.00/19.76 |
| pseudo_one | 100/100 | 3.12 | 0.00 | 0.00 | 3.12 [0.00, 9.38] | 8.84/0.00 |
| pseudo_one_swapped | 100/100 | 0.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0.00/0.00 |
| real_two | 100/100 | 6.25 | 4.69 | 0.00 | 6.25 [0.00, 15.62] | 12.50/18.22 |
| pseudo_two | 100/100 | 7.81 | 3.12 | -3.12 | 10.94 [0.00, 29.69] | 26.88/17.68 |
| real_full | 200/200 | 1.56 | 0.00 | -0.78 | 2.34 [0.00, 7.03] | 6.63/2.21 |
| real_one | 200/200 | 3.12 | 0.00 | 0.00 | 3.12 [0.00, 9.38] | 8.84/0.00 |
| pseudo_one | 200/200 | 3.12 | 0.00 | 0.00 | 3.12 [0.00, 9.38] | 8.84/0.00 |
| pseudo_one_swapped | 200/200 | 3.12 | 3.12 | 0.00 | 3.12 [0.00, 9.38] | 8.84/8.84 |
| real_two | 200/200 | 1.56 | 0.00 | 0.00 | 1.56 [0.00, 4.69] | 4.42/0.00 |
| pseudo_two | 200/200 | 1.56 | 0.00 | 0.00 | 1.56 [0.00, 4.69] | 4.42/0.00 |

All displayed values are percentage points. C is a noisy new execution block on existing sources.
Separate-execution forecasts are not assumed to improve; signed source error contrasts and all controls are retained in overall.csv.
