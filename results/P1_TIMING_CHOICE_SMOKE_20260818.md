# P1 timing-and-option-choice smoke

Status: **execution passed; class coverage incomplete**.

Quest jobs `9761440` and `9762786` ran at clean commit `7ac05defa388`.
Together they captured three source-disjoint placements, 36 matched decision
states, 108 option rollouts, and raw hidden/robot/action features.  Each
placement was evaluated at T-30, T-20, T-10, and T-5.

## Glass-condition outcome signatures

`B/D/R` denotes Base, Detour, and the historical directional RetreatHold.

| Placement | T-30 B/D/R | T-20 B/D/R | T-10 B/D/R | T-5 B/D/R |
|---|---|---|---|---|
| heldout_0000 | catastrophe / success / noncompletion | success / success / noncompletion | catastrophe / success / noncompletion | success / success / noncompletion |
| heldout_0003 | catastrophe / catastrophe / noncompletion | catastrophe / catastrophe / noncompletion | catastrophe / catastrophe / noncompletion | catastrophe / catastrophe / noncompletion |
| heldout_0004 | success / success / noncompletion | catastrophe / success / noncompletion | catastrophe / success / noncompletion | catastrophe / noncompletion / catastrophe |

## What this establishes

- The exact-state multi-anchor pipeline runs end to end and retains fresh raw
  features.
- The same hazard and source trajectory can change the Base/Detour outcome
  signature across phase; hazard presence alone is not the decision label.
- `heldout_0003` is a clean library-unsolved trajectory across all four anchors.
- `heldout_0004` shows Detour value at T-20/T-10, but the historical fixed `-x`
  RetreatHold is not a reliable late fail-safe at T-5.

The smoke does **not** yet establish a fresh Detour-to-Retreat timing pair.  The
next timing isolation run replaces directional retreat with zero-delta
`FailSafeHold` and captures only the glass condition.  Random source search
should not be expanded before that option contract is validated.

## Raw evidence

| Run | Manifest SHA-256 | Metadata SHA-256 | Rollouts SHA-256 | Features SHA-256 |
|---|---|---|---|---|
| r1 | `c9a81a22020ac7c4a2a343de379298e293ccecd27388dfec69a0fc63c3042a49` | `0814f8ee9a3a57beb667959459ad016a50a3a32820ac6c06f278ff9dbb3affb5` | `2facbd62f8a9856015bde4f0b265bd1ca8814c1cc9102520947566ee886a22a1` | `dfcdefa659aa2966222bf56939ef9cc19c7eb83f8f04dd2536cbca8ff21a03bf` |
| r2 | `a93feb50f870f66944ac9917e0242919f3807ce1af0bc700a8adb386478af4c6` | `828ec79cd6031b7bd2149c842ce34770992932dda9a1959703408bdd29e43088` | `ae7ff937b37b632c9e7925fffdaaa1a31f3e66823821bfb88db78843850280cd` | `f3c54b36292d78dfe5c3afba49effa008e75165b5f87e9750bb586bdd960c4ba` |

Quest roots:

- `results/counterfactual_router/p1_timing_choice_smoke_7ac05de_20260818_r1`
- `results/counterfactual_router/p1_timing_choice_smoke_7ac05de_20260818_r2`
