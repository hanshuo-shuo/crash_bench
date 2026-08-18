# P2 dynamic first-crossing Router: development cohort

Status: **superseded pointwise-boundary diagnostic; dynamic timing helps,
selective intervention fails**. The trajectory-calibrated follow-up is in
[`P2_SEQUENTIAL_FIRST_CROSSING_DEV_20260819.md`](P2_SEQUENTIAL_FIRST_CROSSING_DEV_20260819.md).

Quest job `9776682` ran at commit `196d867d5dbd` on five frozen development
source states, with glass, off-path, and no-glass conditions for each source.
The primary point was the router manifest's predeclared `lambda=1,target=0.4`
point (`delta_cal=0.1086698193`). The unit is the source state; the 15 episodes
are repeated condition observations.

## Main result

| Method | Task success | Catastrophe | Safe noncompletion | Intervention |
|---|---:|---:|---:|---:|
| Base from reset | 66.67% | 33.33% | 0.00% | 0.00% |
| **Dynamic first crossing** | **73.33%** | **6.67%** | 20.00% | **100.00%** |
| Fixed T-20 Router (oracle timing) | 53.33% | 26.67% | 20.00% | 26.67% |

Dynamic routing reduced catastrophe by 26.67 points versus Base and by 20.00
points versus the same Router at privileged fixed T-20 timing. It also improved
task success by 6.67 and 20.00 points, respectively. This is real timing value:
on source `heldout_0019`, the fixed T-20 Router chose Base and crashed, whereas
the dynamic score first crossed at action 14, ten actions and 0.118 m before the
matched Base collision, selected Detour, and completed the task.

The present Router nevertheless does **not** yet establish selective “knowing
when to intervene.” It triggered in all 15 episodes. All 10 Base-success control
episodes were unnecessary interventions; eight retained task success and two
became safe noncompletion. A confirmatory dynamic cohort is not justified until
the conservative margin is calibrated over whole causal trajectories rather
than isolated decision states.

## Required P2 diagnostics

- selected options: 14 Detour, 1 Retreat, 0 Base;
- intervention lead time on the five collision references: median 25, mean
  23.6 actions;
- first-trigger distance to matched Base collision state: median 0.173 m, mean
  0.211 m;
- missed recovery window: 0/15 overall and 0/3 among T-20-recoverable episodes;
- unnecessary early intervention: 10/15 overall and 10/10 among Base-success
  controls;
- control task-success retention after intervention: 8/10; safe noncompletion:
  2/10;
- intervention duration: median 202 actions, p95 900 actions;
- episode-peak glass contact force: p95 4.783 N, max 15.944 N;
- dynamic glass outcomes: 3/5 task success, 1/5 catastrophe, 1/5 safe
  noncompletion;
- dynamic off-path/no-glass outcomes: 8/10 task success, 0/10 catastrophe,
  2/10 safe noncompletion.

## Interpretation for the paper

P2 validates the implementation and the “timing can change option value” part
of the story. It also exposes the multiple-look failure hidden by fixed T-20
evaluation: a margin calibrated on single decision states eventually crosses
on every control trajectory. T-20 remains an oracle-timing comparison, not the
deployable method, but the current dynamic Router should remain development
evidence rather than replace the frozen headline result.

The next method change should preserve the exact first-crossing rule while
replacing the pointwise margin with a source-disjoint, episode-level sequential
margin, for example a calibration quantile of
`max_t max_o Delta_hat_t(o)` on Base-success trajectories. This directly
controls the false first crossing created by repeated online looks; it is not a
new low-level option or a hand-authored warm-up gate.

## Raw evidence on Quest

Root:
`results/counterfactual_router/p2_dynamic_full_196d867d5dbd_20260818T143934Z`

| Artifact | SHA-256 |
|---|---|
| `capture_manifest.json` | `218601afca49c1825b7586080c221517ac18c3f45fdf591bfc08736760f3104b` |
| `dynamic_episodes.jsonl` | `96f8524a545c50deebe79f48e34cf08c0fa92e115ddf017c644f74a9264fb31e` |
| `router_trace.jsonl` | `e4dfe9db49426de0879b77901d7dc632b77dc75fca7b41ef56d8022e62d5ee54` |
| `t20_option_rollouts.jsonl` | `79bd76fb370ab0cd7279de3832e3a75d0e7beaad9731e9efad122e86dfa9f283` |
| `analysis.json` | `416ec085eb3b4466a48c58066090e77e1dddda1b902b0fc9c77a07556865ea61` |
| `REPORT.md` | `6cfca2a04735aab57d20041697a27648f12b78b5091a0452db3c1e48a029c8c8` |
