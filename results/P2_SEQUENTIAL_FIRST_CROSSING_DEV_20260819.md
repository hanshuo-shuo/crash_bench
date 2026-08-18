# P2 sequential first-crossing Router: final development result

Status: **selective development Pareto point established; strong “knowing when”
claim not yet established**.

## Protocol

The pointwise Router margin failed under repeated online looks, firing on every
control episode. P2 therefore freezes a source-level sequential boundary using
the original Router calibration split, not evaluation outcomes:

```text
delta_seq(alpha) = split-conformal quantile across source maxima of
                   max_condition max_t max_non-Base Delta_hat_t(option)
```

- calibration: 10 source-disjoint sources, 18 Base-success off-path/no-glass
  trajectories, 1 invalid-initial trajectory excluded;
- primary boundary: `lambda=1`, `target=0.4`, `alpha=0.1`, strict `>` crossing;
- pointwise margin: `0.1086698193`; frozen sequential/effective margin:
  `1.5974966343`;
- development evaluation: 4 stable sources × 3 conditions = 12 episodes;
- causal execution: score the matched Base prefix in order, branch Detour or
  Retreat from the first exact serialized crossing state, and reuse the
  identical Base outcome when no crossing occurs;
- fixed T-20 is an oracle-timing comparison only.

Calibration job `9779232` completed at commit `4cbe033e7850`; exact-prefix
evaluation job `9782983` completed at commit `0f92a1b420dd`. Report-only T-20
Base matching was rebuilt at commit `e974e12` without rerunning GPU episodes.

## Primary result

| Method | Task success | Catastrophe | Safe noncompletion | Intervention |
|---|---:|---:|---:|---:|
| Base from reset | 58.33% | 33.33% | 8.33% | 0.00% |
| **Sequential dynamic first crossing** | **66.67%** | **25.00%** | 8.33% | **16.67%** |
| Fixed T-20 Router (oracle timing) | 58.33% | 33.33% | 8.33% | 0.00% |

This is a strict development Pareto improvement over Base: one Base
catastrophe becomes task success, no outcome worsens, and only 2/12 episodes
intervene. The useful glass crossing occurs at action 25, nine actions and
0.103 m before the matched Base collision; Detour completes the task with a
2.15 N maximum glass-contact force. The one unnecessary no-glass intervention
occurs at action 77 and still completes the task.

## Required P2 diagnostics

- selected options: 10 Base, 2 Detour, 0 Retreat;
- glass outcomes: 1/4 success and 3/4 catastrophe;
- off-path outcomes: 4/4 task success, 0/4 intervention;
- no-glass outcomes: 3/4 no intervention, 1/4 unnecessary Detour; all four
  retain task success except one matched Base safe noncompletion outside the
  Base-success control denominator;
- unnecessary early intervention: 1/12 overall and 1/7 among Base-success
  episodes; task-success retention on Base-success episodes is 7/7;
- intervention lead time: 9 actions; trigger-to-collision EEF distance:
  0.103 m;
- missed recovery window: 2/12 overall and **2/2 among T-20-recoverable
  episodes**;
- intervention duration: median 117.5 actions, p95 220.55 actions;
- episode-peak contact force: p95 34.45 N, max 40.65 N, driven by the three
  non-intervened glass catastrophes.

## Why the strong claim still fails

The failure is score ordering, not a badly chosen sequential alpha. On the
complete exact-prefix traces:

| Frozen boundary | Glass crossings | Control crossings |
|---|---:|---:|
| alpha 0.1, margin 1.5975 | 1/4 | 1/8 |
| alpha 0.2, margin 1.5951 | 1/4 | 1/8 |
| alpha 0.3, margin 1.4807 | 1/4 | 2/8 |
| alpha 0.4, margin 1.4573 | 1/4 | 2/8 |

The two missed T-20-recoverable glass trajectories have maximum advantages
`1.0048` and `1.2960`. Several benign controls rank above them (`1.3681` through
`1.6347`). Lowering a scalar threshold enough to recover those treatments must
first trigger on many controls. No point on the frozen sequential-alpha
frontier repairs this inversion.

One additional source (`heldout_0015`) alternated between on-path catastrophe
and no catastrophe across GPU nodes and is excluded from the stable four-source
table. This is another reason not to promote the development point directly to
a confirmatory claim.

## Paper decision

What is supported now:

> A causal first-crossing Router with source-level sequential calibration can
> add a selective safety--success Pareto point on a small development cohort,
> while preserving Base on most controls.

What is not supported now:

> The current single-frame outcome score reliably knows when recoverable
> interventions are available.

Further threshold tuning is not justified. To make the strong title claim, the
next model change must learn temporal intervention value from complete
trajectories or explicit recovery-window supervision, then freeze once and run
a new source-disjoint confirmatory cohort. The conservative paper option is to
keep the established fixed-T20 E16 result as the headline and present P2 as a
development result plus a precise limitation.

## Raw evidence on Quest

Calibration root:
`results/counterfactual_router/p2_sequential_calibration_4cbe033e7850_20260818T152624Z`

| Artifact | SHA-256 |
|---|---|
| `capture_manifest.json` | `06dc8dd4dd921587364cd17a1fe36269c396e42d5b494a22b873bb3dbb270c7e` |
| `calibration_trajectories.jsonl` | `3591d9bd711aa855e215f63f5768f135782886003561dbd5d781ab871a577ab8` |
| `calibration_router_trace.jsonl` | `cb05cbf2aa1371ad3066914bd5985a2c1a0a9135ec9705d27733b690b762e997` |
| `sequential_boundary.json` | `2764f2c823dc6b41e13102e7c2d6dc9770fcc524d15895762bfd83a8412e8533` |

Evaluation root:
`results/counterfactual_router/p2_dynamic_exact_seq_0f92a1b420dd_20260818T162140Z`

| Artifact | SHA-256 |
|---|---|
| `capture_manifest.json` | `1ef6b64abfe3e7c06f3a2b8d3358114118fb22fe01dbe3a1d6f2dce2c3008b42` |
| `dynamic_episodes.jsonl` | `9ca55ea9d02110659cffd08913c01365112b8e8b439f3496434fcb0d2da3223e` |
| `router_trace.jsonl` | `e88e0c31f611262ac1a8ae4fec5979a60a2f7ab526b6370af43039ad09ca136d` |
| `t20_option_rollouts.jsonl` | `6fab7b2c057982d481f75d9bc3dfbf98fbf19651089a980449a94bbad8ce8ea8` |
| `analysis.json` | `14adc407d2eb7f21fecc86a507ecc5033b2afb5849dcf872e101daba9bb16b84` |
| `REPORT.md` | `a207d04457b08055c6451e9229909f3ba614a5fd839188a1a6652eb0bf716587` |
