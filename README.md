# CrashBench: A Benchmark for VLA Crash Recovery

## Status (updated 2026-06-20)

Pilot infrastructure is up and the VLA bridge is de-risked; authoring real pre-crash
scenarios is the next step.

- **Environment** — reproducible OpenVLA eval env on Northwestern Quest (Python 3.10 /
  torch 2.2 / transformers 4.40.1 / flash-attn 2.5.5 + LIBERO). One-command rebuild and
  every dependency fix documented in [`setup/`](setup/README.md).
- **Nominal baseline** — OpenVLA × LIBERO-Spatial = **80.0% (400/500)**, matching the
  official ~84.7%, so the observation/action bridge is correct (the sanity gate in
  PLAN.md Phase 0). Per-task: `[.90 .92 .86 1.0 .68 .44 .90 .86 .82 .62]`.
- **`crashbench/` package skeleton** — `Scenario` / predicates / LIBERO adapter / OpenVLA
  policy / closed-loop eval / metrics, with passing pure-logic unit tests
  ([`crashbench/README.md`](crashbench/README.md)). The closed loop runs end-to-end and
  reuses OpenVLA's verified `run_libero_eval` bridge.

**Next:** author the 5 pilot pre-crash scenarios (PLAN.md §11) — needs an interactive GPU
session to pin down the `set_init_state` vector layout and the mujoco body names (currently
marked `TODO(verify)`), then run `scripts/run_pilot.py` for the go/no-go crash-rate number.

## 1. One-line pitch

VLAs are trained almost exclusively on successful demonstrations, so they have never seen — and have no policy for — the states that immediately precede a crash. **CrashBench** is a small, hand-curated benchmark that drops a VLA into pre-crash states and measures whether it can recover.

## 2. Motivation

Current VLA benchmarks (LIBERO, SimplerEnv, RoboArena, etc.) measure nominal task completion. None of them stress-test the failure mode practitioners actually observe: VLAs drift into walls, push objects off tables, jam pegs, and drop things. Because demos rarely contain recovery behavior, the policy has no idea what to do when a crash is imminent.

We want to:

1. Quantify how often current SOTA VLAs crash from realistic pre-crash states.
2. Distinguish *perception* failures ("didn't see the wall") from *prediction* failures ("didn't model where the arm would go") from *policy* failures ("saw it, knew it, did it anyway").
3. Establish baselines so future methods (including ours, e.g. Cold Diffusion on the Replay Buffer) have something to beat.

## 3. Connection to prior lab work

This benchmark is the natural evaluation target for *Cold Diffusion on the Replay Buffer: Learning to Plan from Known Good States* (Wang, Oba, Yoneda, Shen, Walter, Stadie). That paper argues plans should be routed through known-good states to stay in the feasible region. CrashBench provides scenarios *outside* the feasible region by construction, where "return to nearest known-good state" is the right recovery strategy. The benchmark and the method are mutually reinforcing.

## 4. Benchmark design

### 4.1 Scope

- **50 hand-curated scenarios × 3 crash horizons = 150 trials**
- **Simulator:** MuJoCo (MJX preferred for parallelism). Single sim — do not multi-sim.
- **Robot:** Franka Panda for the initial pass. Add a second embodiment only if time permits.

### 4.2 Crash horizons

The crash-horizon axis is the key independent variable. It separates reflex from anticipation.

| Horizon | Meaning | Capability tested |
|---------|---------|-------------------|
| T-1     | next action will crash if executed nominally | reactive safety |
| T-5     | crashes within 5 steps if policy continues unchanged | short-horizon correction |
| T-20    | crashes within 20 steps if nothing changes | anticipation / "did the model even notice?" |

### 4.3 Scenario categories (~7 each)

1. Environment collision (gripper drifting into wall, plunging at table, swinging into shelf)
2. Object collision (about to knock over glass, sweep cluttered objects)
3. Self / dual-arm collision
4. Joint / force-limit violation
5. Grasp instability (held object slipping, tilted, about to drop)
6. Unsafe terminal state (pushing object off table edge, toppling stack)
7. Constraint violation (peg about to bind in misaligned insertion, door about to slam)

### 4.4 Scenario authoring pipeline

For each scenario:

1. Record a nominal demo or a scripted bad trajectory.
2. Use a hand-engineered detector (SDF distance, contact prediction, LangSAM mask + depth for "object near table edge", joint-limit margin, force prediction, etc.) to find frames where a T-{1,5,20} crash is imminent.
3. Snapshot the MuJoCo state.
4. **Feasibility filter.** A sample-based planner (or human teleoperator) must find *at least one* recovery trajectory. If no recovery exists, drop the scenario. Every scenario in the benchmark must have a witness.
5. Save: initial state, oracle-verified recovery trajectory (the *witness*), crash predicate, task-success predicate, category label, horizon label.

The witness trajectories serve two purposes: they prove recoverability, and they become the recovery-demo dataset for the fine-tuning baseline.

### 4.5 Crash predicates

Predicates must be objective and computed from sim state, not vision:

- contact force > threshold on disallowed bodies
- penetration depth > ε
- object COM below table surface (fell off)
- joint angle within δ of limit
- grasped object dropped (no contact + below initial height)

Each scenario specifies which predicates apply.

## 5. Metrics

Report all four per trial:

- **Crash rate** (binary; primary headline)
- **Recovery + task success** (didn't crash *and* completed the original task)
- **Safe abort rate** (didn't crash, didn't complete, ended in a stable state)
- **Impact severity** (peak contact force, conditional on crash) — distinguishes soft from catastrophic

**Headline number:** crash rate at T-5, averaged across categories.
**Diagnostic plots:** crash rate vs horizon; crash rate by category.

## 6. Models to evaluate (out-of-the-box)

- π0 / π0.5 (Physical Intelligence)
- OpenVLA + OpenVLA-OFT
- GR00T N1 (NVIDIA)
- Octo
- A non-VLA control: diffusion policy trained on the same demos

## 7. Baselines (the interesting comparisons)

Apply to at least OpenVLA (open weights, easy to modify):

1. **Vanilla** (zero-shot)
2. **Prompted-careful** — prepend "move slowly, avoid collisions" to the instruction. Tests latent safety knowledge.
3. **VLM-as-monitor** — at each step, ask the VLM "will the next action cause a collision?"; if yes, substitute a zero-velocity action. Tests whether monitoring is the bottleneck.
4. **Safety filter** — CBF / signed-distance shield on the executed action. Tests the "just bolt on classical safety" objection.
5. **Recovery-finetuned** — fine-tune on the oracle witness trajectories from a held-out subset of scenarios.
6. **Cold-diffusion-on-replay-buffer** (our prior method) — exactly the case it was designed for.

## 8. Analyses

These are the figures the paper will be remembered for:

1. **Crash rate vs horizon.** If all models flat-line across horizons, they aren't using the extra time — i.e., they don't anticipate.
2. **Self-report calibration.** At each step, prompt the VLM with "are you about to crash?" and compare to ground truth. If AUC is high but crash rate is also high → model *knows* and *doesn't act*. This is the killer figure.
3. **Linear probe on VLA hidden states** for "crash within T steps." Same question, more rigorous.
4. **Scaling.** Small vs large variants within a family. Does crash rate decrease with scale? Hypothesis: barely.
5. **Category breakdown.** Which crash types are hardest? Grasp instability vs collision will diverge.
6. **Held-out generalization.** Train recovery fine-tune on 35 scenarios, evaluate on 15.

## 9. Real-robot validation

Do **not** attempt all 50 on real hardware. Pick **10**: 2 per category, all at T-5, on a single Franka tabletop setup. Report crash rate for 2–3 of the strongest sim-evaluated models. Just enough to argue sim results transfer.

## 10. Execution plan

| Week  | Goal |
|-------|------|
| 1–2   | Pick MuJoCo stack, build 5 scenarios end-to-end, lock in scenario authoring pipeline + crash predicate API |
| 3–4   | Scale to 50 scenarios; implement feasibility filter; collect witness trajectories |
| 5     | Wire up evaluation harness for OpenVLA + π0; first crash-rate numbers |
| 6     | Add remaining 3 models + the 5 baselines |
| 7     | Self-report / probe analyses, scaling, category breakdown |
| 8     | Real-robot subset (10 scenarios) |
| 9–10  | Writing and figures |

## 11. Two cheap pilot experiments to run FIRST

Before committing the full 10 weeks, do these in a weekend. They tell you the paper's actual story.

1. **Take 5 scenarios, run OpenVLA, measure crash rate.**
   - <20% → no paper.
   - 20–50% → standard "VLAs need safety" framing.
   - >50% → very easy paper.
2. **Run the self-report probe on the same 5.**
   - AUC > 0.8 → "model knows but doesn't act" framing is real; lean into it.
   - AUC ≈ chance → reframe as perception/anticipation, not safety reasoning.

These two pilots decide the framing of the whole paper. Do not skip them.

## 12. Deliverables

- `crashbench/` Python package with scenario loader, predicate API, and eval harness
- 50 scenario files (MuJoCo XML + initial state pickle + predicate + witness trajectory)
- A README with reproduction instructions for each baseline
- A leaderboard table (model × horizon × category)
- Companion dataset: ~50 witness recovery trajectories (small, but novel)

## 13. Open questions for the student to answer in week 1

1. MuJoCo + MJX vs Isaac Lab — which gives faster iteration for *our* lab's existing infra?
2. Are there existing crash-scenario assets in `robosuite` / `robocasa` / `mimicgen` we can reuse, or do we author from scratch?
3. How should the witness trajectory be generated — sample-based planner (RRT*?) or human teleoperator? Probably both, depending on scenario complexity.
4. What's the cheapest VLA to iterate on locally (OpenVLA at 7B is the obvious answer, confirm GPU budget).

## 14. Anticipated reviewer objections and responses

- *"This is just OOD generalization."* → include an OOD-but-not-crash control condition; if crash recovery fails much more than general OOD, the safety framing holds.
- *"Just use a CBF / safety filter."* → it's a baseline; show where it works and where it doesn't (grasp instability, tipping, soft constraints).
- *"Just collect recovery demos."* → that's a solution to our benchmark, not a critique; the recovery-finetuned baseline tests exactly this and likely only half-works.

---

