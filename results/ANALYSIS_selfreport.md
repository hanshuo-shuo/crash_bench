# Self-report probe — does OpenVLA *know* it is about to crash? (detailed analysis)

> 2026-06-27 · CrashBench Phase 2 · README §8 (analysis 2/3), §11 pilot 2.
> The "killer figure": separate a **policy/safety** failure ("knew it, did it anyway") from a
> **perception** failure ("never saw it coming").

## 0. TL;DR — it knows almost perfectly, and acts on it not at all

OpenVLA is action-only — you cannot *ask* it "are you about to crash?". So we test self-report
the rigorous way: train a **linear probe** on its own frozen hidden state to predict "crash within
T steps", under matched **wall vs no-wall** conditions on the 5 env_collision scenarios.

| evidence | result | reading |
|---|---|---|
| **Linear probe AUC** (crash within T, leave-1-scenario-out) | **0.99 (T-1) … 1.00 (T-10)** | the impending crash is **near-perfectly linearly decodable** from the representation → the model *knows* |
| **Behavior near impact** (action magnitude, last ≤2 steps) | **0.96 vs no-wall 0.55** | not only no braking — it drives **harder** into the wall → it does not *act* on what it knows |
| **Off-path confound control** (probe logit) | off-path **−5.0** ≈ no-wall **−5.3**, vs on-path-near **−0.3** | the probe decodes **crash-imminence, not "a wall is visible"** |

**Verdict:** the 100% crash is a **policy / safety-behavior gap, not a perception gap.** The crash
signal is present and linearly readable inside the frozen model; the action head simply ignores it.

## 1. Method

- **Frozen model, matched conditions.** Same checkpoint
  (`openvla-7b-finetuned-libero-spatial`), same 5 scenarios, same init states. The *only* change
  is the wall:
  - `nowall` — obstacles dropped (the normal reach; 1100 frames),
  - `wall` — wall on the grasp path (crash 100%; 38 pre-crash frames),
  - `offpath` — the 21 OOD-control walls, visible but off-path (3782 frames) — the confound set.
- **Feature = the LM's last hidden state.** A non-invasive forward hook on the language model's
  final RMSNorm captures the hidden vector at the **last prompt token of the prefill pass** — i.e.
  "having seen the scene, about to emit the action" (`OpenVLAPolicy(capture_hidden=True)`,
  hidden dim 4096). No edit to OpenVLA itself.
- **Probe = logistic regression**, PCA→50 dims, **leave-one-scenario-out** (train on 4 scenarios,
  test on the held-out one — no temporal frame leakage). Label `y_t = 1` iff a real crash occurs
  within T steps of frame t. Pool = `wall ∪ nowall`; `offpath` is held out entirely for the
  confound test. AUC via Mann-Whitney. (`scripts/probe_selfreport{,_analysis}.py`, all numpy — no
  sklearn dependency.)

## 2. Result 1 — the crash is linearly decodable (AUC ≈ 1.0)

| horizon T | probe AUC |
|---|---|
| within 1 step  | **0.993** |
| within 3 steps | 0.992 |
| within 5 steps | **0.998** |
| within 10 steps| **1.000** |

A simple linear read-out of the frozen activations predicts the crash essentially perfectly, out
to held-out scenarios. The information is **there**, and it's there early (AUC already 0.99 a
single step out, and 1.0 ten steps out — the model's representation "sees it coming").

## 3. Result 2 — zero behavioral avoidance (it speeds up, if anything)

Because the wall is tall and the collision is often **arm-body** (the eef stays behind the slab,
see `WITNESS.md`), we measure behavior by **action magnitude vs steps-until-crash**, not by eef
distance. As the crash approaches (steps→0):

- no-wall baseline: median action translation magnitude **0.55** (IQR shaded).
- wall, last ≤2 steps before impact: median **0.96** — **above** the normal band.

So there is **no anticipatory slowdown**; the policy commits its largest motions right as it drives
into the wall. Knowing (§2) and not-acting (§3) together = "knows but doesn't act."

## 4. Result 3 — it decodes the CRASH, not "a wall is visible" (confound killed)

The obvious objection: *"the probe just detects that a red wall is in the image."* The off-path
control set refutes it. Probe crash-logit (higher = "about to crash"):

| group | mean crash-logit |
|---|---|
| no wall | **−5.28** (safe) |
| **off-path wall** (wall visible, no crash) | **−5.00** (scored safe, like no-wall) |
| on-path wall, ≤5 steps to crash | **−0.30** (clearly elevated) |

The off-path walls are **just as visible** as the on-path wall, yet the probe scores them like the
no-wall scenes. The representation that the probe reads is **crash-imminence on the executed path**,
not mere wall presence. (This is the OOD-control set doing double duty — first it showed the crash
is path-encroachment not OOD; here it shows the probe decodes crashing, not seeing.)

## 5. Figures

- **`setup/figures/fig_selfreport_probe.png`** — 3 panels: (1) no braking as crash nears, (2) probe
  AUC ≈ 1.0 across horizons, (3) probe logit by group (the confound-killer).
- **`setup/figures/fig_selfreport_pca.png`** — raw 2-D PCA: on-path (crash) states occupy a distinct
  region; wall frames show a steps-to-crash gradient along PC1.

## 6. Caveats / next

- **Correlational, by construction.** The probe shows the crash info is *present and linearly
  readable*; the behavior (100% crash, no slowdown) shows it is *not used*. Together they make the
  "knows-but-doesn't-act" claim; neither alone does. A causal strengthening (activation steering:
  inject the crash direction and see if the action changes) is the natural follow-up.
- **N is small** (5 scenarios, 38 pre-crash frames). AUC is leave-one-scenario-out so it isn't
  in-sample, but the headline should be reported with the off-path confound control alongside it,
  not on its own. Scales naturally as the benchmark grows.
- **Framing decision (README §11 pilot 2):** AUC ≈ 1.0 ⇒ lean **hard** into the safety-reasoning
  framing ("the model represents the impending collision but has no policy to avoid it"), not a
  perception/anticipation framing.

Machine-readable: `results/selfreport/probe_summary.json`; raw activations
`results/selfreport/hidden.npz` (4920×4096, gitignored) + `meta.json`.
