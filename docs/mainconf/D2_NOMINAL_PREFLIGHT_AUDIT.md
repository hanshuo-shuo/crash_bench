# D2/D3 nominal preflight audit

Status: `NOMINAL_PREFLIGHT_GATE_A0_GO`

Quest array `5204803_[0-7]` executed the frozen π0 nominal-only plan at commit
`a1464cf4ddf03852143deab91339c8571844737d`. All eight cells completed with
exit `0:0` and accounted for all 64 planned source attempts.

| Task | Complete | Nominal successes | Rate |
|---|---:|---:|---:|
| `libero_spatial:0` | 32/32 | 27 | 84.375% |
| `libero_spatial:2` | 32/32 | 32 | 100% |

Every cell met the 70% nominal gate and had 8/8 path-ready trajectories. No
hazard was injected and no option outcome was opened. Gate A0 therefore permits
pre-outcome mechanical parameter derivation for fragile-path collision,
observation staleness, action drift, and narrow clearance. It does not authorize
unstable-placement v2.
