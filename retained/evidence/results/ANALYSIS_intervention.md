# Path 1-1a — Probe-triggered safe-abort (causal intervention)

> **Headline:** reusing the frozen self-report probe (R4) to TRIGGER the witnessed retreat
> turns crash **100 % → 0 %** (15/15 → 0/15) and peak impact **322 N → 0 N**, while the guard
> fires on **0/12** off-path and **0/10** no-wall episodes. The same linear probe, with **no
> retraining and no model edit**, causally fixes the behaviour it was only able to *predict* in
> R4 — the step SALSA / SAFE / Basu et al. stop short of.

## Why this matters (positioning)

R4 established a *representation–behaviour gap*: OpenVLA's frozen hidden state linearly
decodes imminent collision (LOSO AUC 0.99–1.0) yet the policy does not brake. That gap is a
**detector**; the reviewer-facing question is "so what — can you USE it?". Path 1-1a answers
yes: the **same** probe gates a hand-off to the 0 N retreat the Phase-2 witness already proved
exists (5/5 safe-abort). The trigger threshold is fixed *offline* on the negative pool
(off-path + no-wall frames), so the near-zero false-trigger rate is **by construction** and
doubles as the confound control — the guard acts on *crash-imminence*, not *wall presence*.

## Method

- **Probe** (`crashbench/probe.py`): the T=5 model from `scripts/probe_selfreport_analysis.py`
  (PCA-50 + L2 logistic regression on the 4096-d last hidden state), dumped to
  `results/selfreport/probe_T5.npz` by `scripts/probe_dump_T5.py`. `logit(h) > thr` ⇒ "I will
  crash." Threshold **thr = −0.422** = negative-pool max (−0.70) + 10 % of the gap to the
  on-path≤5 median (+2.12). On the captured frames this gives fire-rate **1.0** on-path≤5,
  **0.0** off-path, **0.0** no-wall.
- **Recovery** (`crashbench/recovery.py`): `RetreatHold`, the witness retreat (−x 0.16 m,
  +z 0.10 m, gripper open) recomputed online from `obs` eef each step. No env / model access.
- **Guard** (`crashbench/policies/guarded_policy.py`): `GuardedPolicy(base, probe)` runs the
  base VLA, scores its hidden state, and LATCHES into `RetreatHold` the first step the logit
  crosses thr. Model-agnostic (Policy protocol), so `eval.run_episode` drives it unchanged and
  the two policies are directly comparable.
- **Run** (`scripts/phase3_intervention.py`, GPU job 5412688, A100, ~18 min):

  | condition | scenes | policy | expectation |
  |---|---|---|---|
  | treatment_baseline | 5 on-path walls ×K=3 | bare OpenVLA | crash ~100 %, ~250 N |
  | treatment_guarded  | 5 on-path walls ×K=3 | GuardedPolicy | crash ~0 %, ~0 N |
  | nowall_guarded     | same 5, wall dropped ×K=2 | GuardedPolicy | guard ~never fires |
  | offpath_guarded    | 6 'clear' walls (clr>0.18) ×K=2 | GuardedPolicy | guard ~never fires |

## Results

| condition | n | crash rate | peak force (N) | guard fire rate |
|---|---|---|---|---|
| treatment_baseline | 15 | **100 %** (15/15) | **321.7** (88–600) | – |
| treatment_guarded  | 15 | **0 %** (0/15) | **0.0** (all 15 exactly 0) | 15/15 |
| nowall_guarded     | 10 | 0 % | – | **0/10** |
| offpath_guarded    | 12 | 0 % | – | **0/12** |

- **Trigger timing (mechanism):** the guard fires a mean **3.4** steps into the episode; the
  baseline crashes a mean **6.6** steps in — i.e. the probe fires **before** impact in every
  case, with per-scenario lead time **0.3–5.3 steps**. Even the tightest case (wall_d85,
  lead 0.3 steps) still recovered to 0 N, because the retreat is immediate.
- **False triggers: 0/22** (0/10 no-wall + 0/12 off-path) → nominal behaviour preserved; the
  guard decodes "I will crash", not "a wall is visible" (off-path walls are fully visible yet
  never trip it). off-path / no-wall logit traces stay ≈4–9 below thr for all ~220 steps.

## Figure

`setup/figures/fig_intervention.png` — (1) crash rate 100 %→0 %, (2) per-episode peak force
strip (~88–600 N → 0 N, vs 75 N predicate line), (3) probe-logit traces: on-path crosses thr
and fires within a few steps; off-path (orange) and no-wall (green) stay below thr the whole
episode.

## Scope / caveats

- The guard's **behavioural** effect (CRASH → SAFE_ABORT at 0 N) is the headline; it converts a
  crash into a safe abort, not into task success. Task-completion recovery still needs the
  RRT*/teleop witness (deferred, doubles as recovery-finetune data).
- Probe is OpenVLA-specific; cross-policy replication of the *behavioural* on/off-path effect is
  Path 3. Activation steering (Path 1-1b) is the optional mechanism upside.
- The intervention is intentionally minimal (retreat-and-hold) to keep the causal claim clean:
  "the model's own crash signal, fed back, removes the crash." It is not proposed as a deployed
  safety controller.
