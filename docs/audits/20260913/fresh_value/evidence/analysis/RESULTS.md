# Fresh-source prospective execution results

Twelve prespecified fresh reset attempts; no nominal-success or recovery-success selection.
This population differs from the historical success/hazard-screened sources. Absolute episode actions include prefixes and recovery.

| Choice | H | A gain | B gain | C gain [95% source CI] | Source RMSE A / B |
|---|---:|---:|---:|---:|---:|
| real_full | 220 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0.00 / 0.00 |
| real_one | 220 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0.00 / 0.00 |
| pseudo_one | 220 | 2.78 | -2.78 | 0.00 [0.00, 0.00] | 9.62 / 9.62 |
| pseudo_one_swapped | 220 | 8.33 | 0.00 | 0.00 [0.00, 0.00] | 16.67 / 0.00 |
| Base | 220 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0.00 / 0.00 |
| AlwaysDetour | 220 | -44.44 | -45.83 | -47.22 [-65.28, -29.17] | 11.79 / 10.76 |
| BenefitGate | 220 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0.00 / 0.00 |
| RiskDetour | 220 | -6.94 | -9.72 | -9.72 [-20.83, 0.00] | 6.80 / 0.00 |
| real_full | 440 | 1.39 | 0.00 | 0.00 [0.00, 0.00] | 4.81 / 0.00 |
| real_one | 440 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0.00 / 0.00 |
| pseudo_one | 440 | 2.78 | -2.78 | 0.00 [0.00, 0.00] | 9.62 / 9.62 |
| pseudo_one_swapped | 440 | 8.33 | 0.00 | 0.00 [0.00, 0.00] | 16.67 / 0.00 |
| Base | 440 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0.00 / 0.00 |
| AlwaysDetour | 440 | -36.11 | -37.50 | -38.89 [-54.17, -22.22] | 11.79 / 10.76 |
| BenefitGate | 440 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0.00 / 0.00 |
| RiskDetour | 440 | -6.94 | -9.72 | -9.72 [-20.83, 0.00] | 6.80 / 0.00 |

Gains, intervals and RMSE are percentage points. The privileged references know A outcomes at each state; they are not deployable gates.
A/B/C are separate processes with identical software, checkpoint, GPU and restored bundle/RNG contracts; C remains a noisy execution block.
