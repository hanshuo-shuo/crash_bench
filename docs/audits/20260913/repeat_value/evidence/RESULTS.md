# Existing-data repeat-value results

Development diagnostics. All gains are safe-task-success percentage points, physical-source weighted.

| Panel | Horizon | Method | A reused gain | B separate gain | A minus B [95% descriptive CI] | Positive B sources |
|---|---:|---|---:|---:|---|---:|
| detour | 220 | real_full | 4.17 | 2.08 | 2.08 [0.00, 5.21] | 1/16 |
| detour | 220 | real_one | 4.17 | 2.08 | 2.08 [0.00, 6.25] | 1/16 |
| detour | 220 | pseudo_one | 2.08 | 0.00 | 2.08 [0.00, 6.25] | 0/16 |
| detour | 220 | pseudo_one_swapped | 6.25 | 2.08 | 4.17 [0.00, 10.42] | 1/16 |
| detour | 440 | real_full | 6.25 | 4.17 | 2.08 [0.00, 5.21] | 2/16 |
| detour | 440 | real_one | 6.25 | 4.17 | 2.08 [0.00, 6.25] | 2/16 |
| detour | 440 | pseudo_one | 2.08 | 0.00 | 2.08 [0.00, 6.25] | 0/16 |
| detour | 440 | pseudo_one_swapped | 6.25 | 2.08 | 4.17 [0.00, 10.42] | 1/16 |
| candidate_refresh | 100 | real_full | 7.03 | 5.47 | 1.56 [-6.25, 9.38] | 2/8 |
| candidate_refresh | 100 | real_one | 12.50 | 3.12 | 9.38 [0.00, 28.12] | 1/8 |
| candidate_refresh | 100 | pseudo_one | 3.12 | 0.00 | 3.12 [0.00, 9.38] | 0/8 |
| candidate_refresh | 100 | pseudo_one_swapped | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0/8 |
| candidate_refresh | 100 | real_two | 6.25 | 4.69 | 1.56 [-9.38, 14.06] | 1/8 |
| candidate_refresh | 100 | pseudo_two | 7.81 | 3.12 | 4.69 [0.00, 10.94] | 1/8 |
| candidate_refresh | 200 | real_full | 1.56 | 0.00 | 1.56 [0.00, 4.69] | 0/8 |
| candidate_refresh | 200 | real_one | 3.12 | 0.00 | 3.12 [0.00, 9.38] | 0/8 |
| candidate_refresh | 200 | pseudo_one | 3.12 | 0.00 | 3.12 [0.00, 9.38] | 0/8 |
| candidate_refresh | 200 | pseudo_one_swapped | 3.12 | 3.12 | 0.00 [0.00, 0.00] | 1/8 |
| candidate_refresh | 200 | real_two | 1.56 | 0.00 | 1.56 [0.00, 4.69] | 0/8 |
| candidate_refresh | 200 | pseudo_two | 1.56 | 0.00 | 1.56 [0.00, 4.69] | 0/8 |
| selection_retest | 100 | real_full | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0/12 |
| selection_retest | 100 | real_one | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0/12 |
| selection_retest | 100 | pseudo_one | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0/12 |
| selection_retest | 100 | pseudo_one_swapped | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0/12 |
| selection_retest | 100 | real_two | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0/12 |
| selection_retest | 100 | pseudo_two | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0/12 |

A/B are execution blocks on the same sources, not fresh-source generalization.
The pseudo-option columns must not be subtracted from real intervention columns as a noise correction.
Zero intervals reflect the recorded source contributions, not population safety certification.
