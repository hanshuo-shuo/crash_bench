# Path 1-1b — Activation steering at the readout · NEGATIVE (well-characterized)

> **Result:** subtracting `alpha · d_unit` (the probe's crash direction) from OpenVLA's final
> RMSNorm output does **not** brake the crash — crash stays **100 % at every alpha** (0→80),
> peak force is flat/noisy, action magnitude is unchanged (~0.8). A diagnostic proves this is a
> **true null, not a bug**: the hook fires (the action moves with alpha), but less than 10% of the
> probe-direction readout norm is retained on the action-token slice (‖W_action·d‖ = 0.74 of
> ‖W_full·d‖ = 7.69),
> so nudging it barely touches the action bins and, when alpha is large enough to matter, moves
> the action in an arbitrary (non-braking) direction. **The direction that *decodes* "I will
> crash" is not the direction that *controls* the action.** This is exactly why 1a's *structured
> gating* succeeds where naive readout-steering fails — and it cleanly differentiates the
> detector from a controller.

## Why we tried it

1a (Result 5) gated a hand-coded retreat ON the probe. 1b asked the deeper, mechanism-level
question: is the probe *direction itself* a steering knob — push the hidden state away from
"I will crash" and does the policy brake on its own, no controller? OpenVLA's action is a
discrete token off the LM head applied to the final post-norm hidden state, so the obvious
injection point is that norm output (the original Path 1-1b design).

## Method

- **Steering hook** (`crashbench/policies/openvla_policy.py`, `enable_steering=True` +
  `set_steering(vec, alpha)`): a forward hook on `language_model.model.norm` that returns
  `out − alpha · d_unit` at every token position (prefill context + each decoded action token).
  `alpha = 0` reproduces the bare policy exactly (verified: crash 100 %, force ~360 N, identical).
- **`d_unit`** = `Probe.steer_vector()` = the unit residual-stream direction that increases the
  crash logit, mapped back through PCA: `V_pca @ (w_lr / sd_lr)`, normalized.
- **Sweep** (`scripts/phase3_steering.py`, GPU job 5413534): `alpha ∈ {0,5,10,20,40,80}` over the
  5 on-path walls (K=2, does force fall?) and the 5 no-wall reaches (K=1, does the task survive?).
- **Diagnostic** (`scripts/phase3_steering_diag.py`, GPU job 5414190): one fixed observation,
  measures ‖Δaction‖ vs alpha (does the action move at all — bug check) and the readout projection
  ‖W·d‖ on the action-token slice vs the full vocab (why).

## Results

**Sweep (the null):**

| alpha | treat crash | treat peak force (N) | no-wall success | action mag |
|---|---|---|---|---|
| 0  | 100 % | 360.0 | 100 % | 0.81 |
| 5  | 100 % | 363.4 | 60 % | 0.80 |
| 10 | 100 % | 303.5 | 100 % | 0.81 |
| 20 | 100 % | 292.5 | 80 % | 0.85 |
| 40 | 100 % | 376.8 | 80 % | 0.86 |
| 80 | 100 % | 295.5 | 100 % | 0.77 |

No monotonic force drop, crash never avoided, action magnitude flat. (No-wall success wobble is
small-n noise, n=5; it does not trend with alpha.)

**Diagnostic (why it's a true null):**

- **Hook fires** — ‖action(alpha) − action(0)‖ = 0 → 0.11 → 0.44 → 0.37 → **0.73** at alpha=1000.
  The action *does* move, so steering is active (not a no-op bug). At alpha=1000 the action even
  *grows* (`[0.505, 0.873, …]`) — the opposite of braking.
- **Readout norm is concentrated outside action tokens** — ‖W_full · d‖ = **7.688**, but
  ‖W_action(256) · d‖ = **0.742**. Less than 10 % of the probe-direction readout norm is retained
  on the 256 action-token slice; this is a norm ratio, not a measured angle. The rest
  shifts non-action vocab. So the per-unit-alpha action-logit shift is tiny, and you only perturb
  actions at absurd alpha — in an uncontrolled, non-braking direction.

Figure: `setup/figures/fig_steering.png` — (1) flat null sweep, (2) ‖Δaction‖ vs alpha (fires only
at absurd alpha), (3) the readout-projection bar (0.74 vs 7.69).

## Interpretation & why this strengthens the paper

The crash is **linearly decodable** at the final layer (R4, AUC ~1.0) yet that same direction is
**causally inert at the readout** for the action. Detector ≠ controller: you cannot reuse the
linear probe as a steering knob by nudging the layer it was read from. This is the mechanistic
reason 1a's **structured intervention** (gate → hand off to a verified safe controller) is the
right design, and it differentiates CrashBench from SAE-steering work (Swann et al. find steerable
features by *training* an SAE; a single readout probe direction is not such a feature).

## Scope / fallback (not done)

The PLAN's predicted fallback is **mid-layer injection**: OpenVLA's world-model signal is
reportedly strongest in the middle layers (~15–22; Emergent World Representations in OpenVLA),
and a perturbation there flows through the remaining attention/MLP blocks (it changes the
*computation*, not just the *readout*), so it is a fundamentally different — and more promising —
intervention than the final-norm readout tested here. It requires (a) capturing mid-layer hidden
states, (b) re-fitting a probe there, (c) a steering hook at that block, (d) a new sweep
(~2 GPU jobs). **Deferred pending a decision** — 1a remains the locked headline; 1b at the readout
is a clean, informative negative.
