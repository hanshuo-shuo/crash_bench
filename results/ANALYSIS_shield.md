# Probe-gated shield — operating curve (Week-1, 2026-07-02)

> **Headline:** the probe-gated safe-abort shield is not knife-edge. Sweeping the trigger
> threshold, there is a **3.4-logit-wide window `[-0.7, 2.7]`** in which the shield holds
> **treatment crash rate = 0/5** *and* **benign false-abort = 0/20** simultaneously. The
> threshold fixed offline in Path 1-1a (**-0.42**) sits just inside the low edge of that
> window, with ~3.1 logits of headroom before it would start missing crashes. So the
> 100%→0% result is a **robust operating regime**, not a tuned coincidence.

This is the Week-1 upgrade of Path 1-1a (single fixed threshold) into the full shield
operating characteristic — the three numbers the plan (§3 baseline-3) asked for as a
function of threshold: **treatment crash↓, benign false-abort (off-path + nominal), and
success loss.**

## Method — post-hoc, no GPU

Path 1-1a already validated the *mechanism* closed-loop (fire before impact ⇒ witnessed 0 N
retreat ⇒ recovery; crash 15/15→0/15, false-trigger 0/22). Given that, the shield's
threshold trade-off is a **re-scoring** of the frozen R4 self-report capture set — no new
rollouts needed:

- `results/selfreport/hidden.npz` (4920 frames × 4096-d last hidden) aligned row-for-row
  with `results/selfreport/meta.json` (`cond, scenario_id, t, crash_step, ...`).
- Score every frame with the frozen probe (`crashbench.probe.Probe`, PCA-50 + LR), group
  into episodes by `(cond, scenario_id)`, order by `t`.
- Episode model:
  - **on-path (`wall`, n=5)** — POSITIVE. Saved iff the shield fires at some `t < crash_step`
    (substitutes the retreat before impact). The gating quantity is `max(logit | t<crash_step)`.
  - **no-wall (n=5)** — benign reach, NEGATIVE. Any fire = false abort = the **success-loss
    proxy** (a fired episode aborts the task). Gating quantity: `max(logit)` over the episode.
  - **off-path clear walls (n=15, non-crashing)** — NEGATIVE confound (wall fully visible, no
    crash coming). Same gating quantity.
  - *(off-path gradient-band walls that did crash, n=6, are genuine positives; folded into the
    frame-level ROC only, not the shield-metric denominators.)*

Script: [`scripts/probe_shield_sweep.py`](../scripts/probe_shield_sweep.py) →
`results/shield/{sweep.json,summary.json}`, `setup/figures/fig_shield_sweep.png`.

## Result

The shield thresholds on a single per-episode logit, so the safe window is *exactly*
`( highest benign max-logit , lowest on-path pre-crash max-logit )`:

| pool | quantity | value |
|---|---|---|
| benign (20 neg.) | **highest** episode max-logit (the FP ceiling) | **-0.70** (no-wall d85) |
| on-path (5 treat.) | **lowest** pre-crash max-logit (the detection floor) | **2.73** (wall_d85) |

→ **safe window `[-0.7, 2.7]`, width 3.4 logits**; midpoint thr≈1.0.

| threshold | treatment crash | benign false-abort | nominal false-abort |
|---|---|---|---|
| **-0.42 (1-1a, validated)** | **0/5** | **0/20** | **0/5** |
| 1.0 (window midpoint) | 0/5 | 0/20 | 0/5 |
| anywhere in [-0.7, 2.7] | 0/5 | 0/20 | 0/5 |
| < -0.7 | 0/5 (still saved) | >0 (benign fires creep in) | >0 |
| > 2.73 | >0 (starts missing on-path) | 0/20 | 0/5 |

- The three headline numbers **at the validated threshold: crash 0.0 / benign-FP 0.0 /
  nominal-FP 0.0** — reproducing 1-1a from the frozen captures and locating it on the curve.
- `fig_shield_sweep.png`: **(left)** crash rate (red) and false-abort (blue/green) vs
  threshold, with the safe window shaded and the validated thr marked; **(right)** the raw
  separation — every benign episode's max-logit sits ≤ -0.7, every on-path pre-crash max-logit
  sits ≥ 2.73, the 3.4-logit gap shaded.

## Scope / honesty

- **Small episode n** (5 on-path, 5 no-wall, 15 off-path-clear). The *frame* pool behind it is
  large (509 imminent / 4400 benign), but the shield metrics are episode-level, so the window
  edges are set by single worst-case episodes (both edges happen to be the d85 wall). The
  closed-loop 1-1a run (K-repeats → 15/10/12) is the statistically harder validation; this
  sweep shows the *slack around* that operated point.
- **Post-hoc assumption:** "fires before `crash_step` ⇒ crash prevented," taken as established
  by 1-1a's closed-loop 15/15→0 (retreat is immediate; recovered even at 0.3-step lead). The
  sweep does not re-simulate the retreat; it inherits that guarantee.
- A pooled frame-level detection AUC is emitted as a secondary robustness stat
  (`frame_roc_auc_pooled` in summary.json) — it is looser than R4's LOSO-CV AUC 0.99–1.0
  because it scores whole traces without cross-validation; R4 remains the probe's headline AUC.
- **Optional closed-loop confirmation:** re-run 2–3 threshold points (e.g. thr=1.0 midpoint,
  thr=2.5 near the upper edge) through `scripts/phase3_intervention.py` with
  `GuardedPolicy(base, probe, thr=…)` to confirm the offline window holds in simulation. Cheap
  (~18 min/point) but not required for the Week-1 number.
