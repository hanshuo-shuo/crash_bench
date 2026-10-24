# V5 交接（2026-10-24 收尾）

读者：接手的下一个 Claude/Codex 会话。先读 [AGENTS.md](../../AGENTS.md) 和 [CURRENT.md](../CURRENT.md)，再读本文。

## 用户的决定（按时间）

| 日期 | 用户原话 | 含义 |
|---|---|---|
| 10-24 | "推送+开始第一步 故事是最重要的" | V4 收口批不做；转向 V5，一步步来，故事优先 |
| 10-24 | "做吧 然后采用v5计划" | 做第 2 步之后的设计修改，采用 V5 计划；付费 API 仍需另行批准 |
| 10-24 | "3吧" | 用反事实布局（挪动锚点）构造测试 |
| 10-24 | "限定一下吧 然后暂时不要扩展太多了 就是故事都还没走通呢" | 只保留能产出孪生的任务组合；不补源、不扩规模 |
| 10-24 | "我想做1"，"这一轮先收尾" | 先用 object/II/task1 一个任务走通故事；本轮到此为止，没有正在运行的作业 |

## 故事（当前版本）

在自然场景里，约束是一句指向真实物体的话，在机器人做到一半时加入（"别让任何东西越过 A 和 B 的连线"）。

- 训练数据用自然布局。这里只看文字或"规则 × 进度"就能猜对大部分标签。
- 测试用反事实布局：挪动机器人碰不到的锚点，同样的动作带回放仍是合法见证；同一张图配两条规则，一条分隔、一条不分隔。
- 预期：直接判断（A）学到的是捷径，在反事实孪生上失败；先写关系事实的模型（B）必须看图，能够推广。

这延续了 V4 的主线（关系事实监督 vs 直接判断），也对应 HazardArena 的"孪生几乎一样时，拒绝训练退化成表面匹配"。走偏检查见报告页 <https://claude.ai/artifact/DJPGiauxhPAC8BYE4L14Y5> 的"本周进展"一节（第 5 版）。

## 已完成（结果只读）

| 步 | 报告 | 运行目录（`/projects/p33100/siosio/crashbench_safelibero/verifier_dataset/`） |
|---|---|---|
| 第 1 步 CPU 几何摸底 | [V5_STEP1_REPORT.md](V5_STEP1_REPORT.md) | `20261024T201429Z_v5-scan_1d8a7cd84182` |
| 第 2 步 渲染与盲看 | [V5_STEP2_REPORT.md](V5_STEP2_REPORT.md) | `20261024T205732Z_v5-render_35fc6c5029bf` |
| C0 修正后重新回放 | [V5_C0_REPORT.md](V5_C0_REPORT.md) | `20261024T212428Z_v5-trace_0b0573ded5e0` |
| 反事实布局 smoke v1–v3 | [V5_LAYOUT_PROTOCOL.md](V5_LAYOUT_PROTOCOL.md) | `…220018Z…3229b5377671`、`…221104Z…5fac839b7681`、`…221853Z…7c9958eb72a7` |
| S0 筛查 | [V5_S0_REPORT.md](V5_S0_REPORT.md) | `20261024T224456Z_v5-screen_31bbc56ab8e2` |

S0 的关键数字：
- 38 条源、6 个任务组合可用（训练 21、验证 9、测试 8 条源）。
- 在"任务 × 规则 × 进度段"组内配平后，能挡住捷径的测试几乎全是 object/II/task1：验证、测试各约 87 个孪生单元、各 4 条源。

## 下一步：最小故事闭环（每步开始前向用户确认）

1. **S1：补布局，再盲看一次**
   - 给 object/II/task1 的 18 条源各多做几个设计式反事实布局，例如 8 个。只在 CPU 上回放，不补源。
   - 从验证源里选 Claude 和任何人都没看过的单元，用 `v5_render` 渲染，再按 [V5_STEP2_PROTOCOL.md](V5_STEP2_PROTOCOL.md) 的流程盲看。
   - `v5_render` 现在读的是第 1 步的 scan 行；要改成读布局运行的行和 LAYOUT.json：在规范初始状态挪好锚点后再回放渲染，可以复用 `v5_layouts.run_layout` 的挪动代码。
2. **S2：建数据**
   - **训练**：object/II/task1 训练源的自然布局行。是否加入其他 5 个组合的自然行，待用户决定（建议先只用 object）。
   - **验证和测试**：对应切分源的反事实静置孪生，在组内配平。在任何模型看到之前冻结测试集。
   - **B 的中间事实还没定。** 候选做法是以机器人一侧为参照："物体在连线靠机器人的一侧吗""终点呢"，判定 = 两者的异或。机器人底座离连线太近的行要排除。写进协议后再生成目标。
   - 渲染两张对向图（gengpu A100，排除 qgpu2014）。
3. **S3：A 和 B 的 pilot**
   - 各 1 个 seed，只看验证集，沿用 `verifier_training/` 的管线和损失设置。
   - 加零样本 VLM 对照：本地 Qwen，不花 API。
   - 看到方向后再定 3 个 seed 和测试评估。测试集只评估一次。

**暂缓**：补源（C1–C5、nominal 重新执行）、挪动固定件、AEGIS 闭环、付费 API。

## 代码入口（`verifier_dataset/`）

| 文件 | 作用 |
|---|---|
| `v5_geometry.py` | 锚点规则（排除终点所属物体、相像物体）、连线标签与余量（5 cm + 0.1 × 抬起高度）、配平与孪生统计、选样 |
| `v5_scan.py` | 第 1 步：初始状态的物体几何 |
| `v5_trace.py` | 带物理哈希核验的 CPU 回放，记录每个 checkpoint 时刻的几何（自然布局行的来源） |
| `v5_layouts.py` | 反事实布局：叠放物体整体挪动、按设计摆放、回放见证核验、筛查与捷径迁移读数 |
| `v5_render.py` | 两张对向俯视图加诊断图（第 2 步用；S1 要接上布局） |
| `v5_launch.py` | 启动器，阶段为 `scan`、`render <scan>`、`trace`、`layout <trace>`、`screen <trace>` |

测试：`/projects/p33100/siosio/envs/openpi/bin/python -m unittest verifier_dataset.tests.test_v5_geometry verifier_dataset.tests.test_v5_trace verifier_dataset.tests.test_v5_layouts verifier_dataset.tests.test_v5_render`，共 31 个。

## 容易踩的坑

- **启动器要求工作区完全干净，代码已发布。** 用 `/projects/p33100/siosio/envs/openpi/bin/python -m verifier_dataset.v5_launch …` 启动，提交时用 hanshuo-shuo 的身份。
- **Quest 会复用作业号。** `sacct -j` 可能列出别人的旧作业；启动器里的 `completed()` 已按 p33100 账号过滤，手动查询时加 `-u $USER` 并看 Account 列。
- **容器里是 Python 3.8**（`aegis-py38.sif` 加 `envs/aegis_sim`），不要用 3.9 以后的语法。
- **盲看纪律**：先只读 `REVIEW_SHEET.json` 和学生图，判断写进 `verifier_dataset/reviews/` 并提交，之后才读 UNITS.json、RENDER_GEOMETRY.json 和诊断图。
- **报告页**：`docs/generative_verifier/PAPER_READING.html` 是 DJPGiauxhPAC8BYE4L14Y5 的源文件，中英两份内容都要改。主报告页 Ya7FQbZC9yEuNNyYhPBHHT（PHASE2_REPORT.html）本轮没有同步 V5。
