# CrashBench: ODUR Negative Result

> **Final status (2026-08-31):** the Option-Conditioned Distributional Utility
> Router (ODUR) experiment completed correctly, but the learned method and the
> confirmatory benchmark claim did not pass their frozen gates. The terminal
> release is `SCOPED_TEST_SCOPE_FAILURE_RELEASE`, not a successful router or a
> general safety benchmark.

CrashBench asks a simple question: if a robot may be in danger, what should it
actually do? A binary risk score cannot by itself tell us whether intervention
will help, which intervention is best, or whether a safe action merely abandons
the task. We therefore tested **option-conditioned outcome prediction** and
conservative action selection using exact-state realized outcomes.

中文概括：实验和数据管线都成功运行了，但想证明的科学结论失败了。模型没有稳定
超过简单基线；保守校准后又退化成几乎总是安全停止。最终测试还严重缺少“其实不该
干预”的对照场景，所以不能证明系统同时知道什么时候该干预、什么时候不该动。

## Core method / 核心方法

ODUR stands for **Option-Conditioned Distributional Utility Router**. For every
decision state, the benchmark restores the same simulator, controller, policy
continuation, sensor queue, and random-number state, then executes each option
from that identical branch point.

本次 observation-staleness 实验使用三个可部署选项：

1. `base_continue`：继续执行冻结的 π0 策略；
2. `observation_refresh`：清理延迟视觉队列、刷新观测并重新查询策略；
3. `safe_stop`：停止运动并接受安全但未完成任务的结果。

For each state–option pair, ODUR predicts:

```text
P(task success | state, option)
P(catastrophe | state, option)
P(safe noncompletion | state, option)
normalized duration, path, force-exposure, and latency costs
```

These predictions are converted to the frozen utility `U0`. Five independently
seeded models form an ensemble. Source-level conformal residuals produce
simultaneous lower confidence bounds on pairwise option advantages and upper
bounds on absolute and relative catastrophe risk. A non-Base option is selected
only if it strictly beats Base and every other admissible option while satisfying
both safety bounds. Otherwise the router preserves Base or fails closed to safe
stop.

简单说：我们不是只预测“危险不危险”，而是预测“继续、刷新视觉、停止”各自会发生
什么，然后只有在证据足够强时才允许切换动作。

## What we tried / 我们做了什么

- Built an exact-state branch engine for π0 on two LIBERO-Spatial tasks.
- Screened several mechanisms; only observation staleness had sufficient valid
  support for the formal scoped pilot.
- Collected 48 train/calibration/development physical sources: 1,296 decision
  anchors and 3,888 complete realized option outcomes.
- Trained five ODUR selection seeds, froze the strongest deployable baseline
  (`DirectQ`), refit five final seeds, and calibrated on 12 disjoint sources.
- Opened one immutable 32-source confirmatory split, collected all 864 anchors
  and 2,592 option branches, audited every content-addressed blob, and permanently
  sealed the test.
- Evaluated the three frozen catastrophe-cost settings plus a separately frozen,
  explicitly non-gating 108-setting utility sensitivity grid.

## What failed / 为什么失败

### 1. The learned method did not beat the simple baseline

Only 2 of 5 train-only ODUR seeds exceeded `DirectQ`; median development
`Delta U0` was `-0.0116`. The method gate therefore failed before any method
superiority test was opened.

No method superiority test was opened.

只有 2/5 个随机种子的 ODUR 超过 DirectQ，开发集效用中位差是 `-0.0116`。因此我们
没有资格在最终测试集上宣称或检验“ODUR 比基线更好”。

### 2. Conservative calibration collapsed to safe stop

The source-level conformal bounds were extremely wide. On development data the
calibrated selector chose safe stop for 100% of decisions. Catastrophe decreased
by 6.79 percentage points, but source-macro utility decreased by `0.698` because
the robot abandoned many tasks it could otherwise finish.

校准后模型过于保守，所有状态都选择安全停止。碰撞确实少了，但机器人也放弃了大量
本来能完成的任务，所以总体效用大幅下降。这不是一个有用的恢复策略。

### 3. The confirmatory test lacked enough non-intervention controls

In the once-opened 32-source test, 31 sources benefited from at least one
intervention and only 1 source was a `B=0` control. Even the prospectively scoped
gate required at least 3 controls. The same deficiency appeared under all three
catastrophe-cost settings (`0/3` settings passed the complete support contract).

最终 32 个测试场景里，31 个都能从某种干预中获益，只有 1 个属于“不干预更合适”。
最低要求仍是 3 个。正负场景太不平衡，因此无法证明 benchmark 能同时考察“何时该
干预”和“何时不该干预”。其他很强的结果不能抵消这个缺口。

## What worked / 哪些部分成功了

| Evidence | Result |
|---|---:|
| D8 physical sources | 32/32 complete |
| Tasks | 16 sources per task |
| Exact branch-start equality | 100% |
| Mechanical-invalid rate | 0% |
| Missing planned branches | 0 |
| B=1 support | 31 sources |
| Strict observation-refresh support | 21 sources |
| Strict safe-stop support | 22 sources |
| Same-risk benefit/strict-option flips | 27 sources |
| B=0 support | **1 source; required >=3** |

The pipeline, exact restoration, option execution, data accounting, and one-time
test protocol all worked. The failure is scientific: the evidence does not
support the desired balanced benchmark or learned-router claim.

实验程序没有坏，数据也没有损坏。失败的是研究假设，而不是运行本身。

## Final claim boundary / 最终结论边界

This repository does **not** claim:

- a successful, superior, or deployable ODUR method;
- a general multi-mechanism VLA-safety benchmark;
- reliable online recovery or sequential safeguarding;
- permission to reopen, replace, or top up the confirmatory test.

The valid output is a complete, reproducible **single-policy,
single-observation-staleness scope-failure and method-null release**.

## Historical glass diagnostic

The publication-first glass case study remains a separate preserved result. Its
working history includes **Knowing When to Intervene: Counterfactual Outcome
Routing for VLA Safety**, Base/Detour/**Always Retreat**, and the finding that
risk does not specify intervention value. The current canonical scope and claim
lock are in
[`docs/iclr27/PUBLICATION_FIRST_RESOLUTION.md`](docs/iclr27/PUBLICATION_FIRST_RESOLUTION.md).
The ODUR negative result does not rewrite or inflate that frozen glass evidence.

## Final artifacts / 最终产物

- [D8–D10 scope-failure audit](docs/mainconf/D8_D10_SCOPE_FAILURE_AUDIT.md)
- [Scoped manuscript draft](docs/mainconf/SCOPED_STALENESS_BENCHMARK_MANUSCRIPT_V2.md)
- [D10 release manifest](results/expansion/d10_release/e2a5264a802a_9c869dc8f97d_scope_failure_v2/release_manifest.json)
- [D10 release audit](results/expansion/d10_release/e2a5264a802a_9c869dc8f97d_scope_failure_v2/release_audit.json)
- [Final support figure](results/expansion/d10_figures/e2a5264a802a_9c869dc8f97d_scope_failure_v2/figure_scoped_support.svg)
- [Confirmatory analysis](results/expansion/d8_benchmark/e2a5264a802a_9c869dc8f97d_20260830T124059Z/postprocess_pinned_9c869dc/benchmark_test_analysis.json)
- [Full 108-setting sensitivity](results/expansion/d8_benchmark/e2a5264a802a_9c869dc8f97d_20260830T124059Z/postprocess_pinned_9c869dc/full_utility_sensitivity.json)
- [One-time test completion seal](results/expansion/d8_benchmark/e2a5264a802a_9c869dc8f97d_20260830T124059Z/authorization/test_complete.seal)

## Repository map

```text
crashbench/                 exact-state, option, model, and runtime libraries
scripts/expansion/          source, collection, training, gate, and release tools
setup/                      Quest/Slurm wrappers and environment instructions
results/expansion/          frozen machine artifacts and compact releases
docs/mainconf/              scoped expansion audit and manuscript package
docs/iclr27/                publication-first glass truth sources
docs/archive/               superseded historical plans and reports
```

## Verification

```bash
pip install -e .
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests -q
PYTHONDONTWRITEBYTECODE=1 python scripts/audit_repo.py
git diff --check
```

These checks validate tracked metadata, hashes, gates, test sealing, exposure
lineage, release claim boundaries, and code invariants. They do not rerun GPU
rollouts or reconstruct ignored raw observation blobs.
