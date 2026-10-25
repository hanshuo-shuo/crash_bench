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
| 10-24 | "训练集只用 object 一个任务" | S2 训练只用 object/II/task1 训练源，不加其他 5 个组合的自然行 |
| 10-24 | "哦哦有道理 继续吧 ……都加" | 三项都采用：训练加入训练源的反事实布局；B 沿用 V4 写法（复述连线、物体靠机器人一侧、篮子靠机器人一侧、Separated、Verdict）；S1 渲染加正俯视图。S1 协议：[V5_S1_PROTOCOL.md](V5_S1_PROTOCOL.md) |

## 故事（当前版本）

在自然场景里，约束是一句指向真实物体的话，在机器人做到一半时加入（"别让任何东西越过 A 和 B 的连线"）。

- 训练数据用自然布局。这里只看文字或"规则 × 进度"就能猜对大部分标签。
- 测试用反事实布局：挪动机器人碰不到的锚点，同样的动作带回放仍是合法见证；同一张图配两条规则，一条分隔、一条不分隔。
- 预期：直接判断（A）学到的是捷径，在反事实孪生上失败；先写关系事实的模型（B）必须看图，能够推广。

这延续了 V4 的主线（关系事实监督 vs 直接判断），也对应 HazardArena 的"孪生几乎一样时，拒绝训练退化成表面匹配"。走偏检查见主报告 <https://claude.ai/artifact/Ya7FQbZC9yEuNNyYhPBHHT> 的"下一阶段：V5"一节（第 13 版）。

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

1. **S1：补布局，再盲看一次**（2026-10-25 完成，见 [V5_S1_REPORT.md](V5_S1_REPORT.md)）
   - 144 个布局全部合法；验证、测试各 392 / 384 个严格孪生单元，读法 `s1_supply_ok`。
   - 两轮盲看在判定、物体侧、篮子侧三项上都是 40/40；读法 `relation_visible`、`facts_readable`。
   - 俯视图没有带来提升，而且 13/20 个单元的布丁被夹爪挡住。S2 用两张斜视图。
   - 运行目录：`20261025T002929Z_v5-s1-layout_c7329566dbbb`（`c732956` 后来改写为 `d38420c`，内容相同）、`20261025T005258Z_v5-s1-render_d38420c96208`。
2. **S2：建数据**（用户 10-25 "继续做s2 一起更新"；协议 [V5_S2_PROTOCOL.md](V5_S2_PROTOCOL.md)；2026-10-25 完成，见 [V5_S2_REPORT.md](V5_S2_REPORT.md)）
   - `s2_dataset_ready`：训练 1,792 行，验证 1,538 行，测试 1,154 行，另有自然诊断各约 80 行；丢弃 0 行。
   - "规则 × 进度"查表：训练 0.760；验证、测试 0.500；自然诊断 0.747 / 0.912。
   - 数据：`/projects/p33100/siosio/crashbench_safelibero/verifier_dataset/20261025T013358Z_v5-s2_d1d8d4129223/dataset.jsonl`。`TEST_FREEZE.json` 冻结了测试，只在 S3 最后评估一次。
   - **训练**：只用 object/II/task1 的 10 条训练源，自然布局加上它们的反事实布局（用户 10-24 决定）。
   - **验证和测试**：对应切分源的反事实静置孪生，在组内配平。在任何模型看到之前冻结测试集。
   - **B 的中间事实**（用户 10-24 采用）：复述连线，"物体在连线靠机器人底座的一侧吗""篮子呢"，Separated = 两者不同，Verdict。机器人底座离连线 < 10 cm 的行排除。S1 的盲看检验这两个事实看不看得出，写进 S2 协议后再生成目标。
   - **10-24 只读摸底**（C0 回放和 S0 布局，不跑作业；脚本未入库，正式采用时写进 S2 协议）：
     - object/II/task1 的 18 条源自然布局完全相同：杂货位置的标准差 0.0 cm，篮子 0.6–1.0 cm；只有障碍物种类不同（5 种，都在约 (0, 0.10)）。所以"只用自然布局训练"等于只见过一个布局。
     - 以机器人为参照的两个事实（物体侧 T、篮子侧 G），在自然训练行上按"规则 × 进度"查表能拟合 0.985 和 1.000，到反事实验证行上是 0.517 和 0.462。和判定一样都是捷径，只用自然布局训练时 B 也没有理由看图。
     - 加入训练源的反事实布局后，查表上限：判定 0.767、T 0.698、G 0.674（只用自然布局时 0.985 / 0.985 / 1.000）。
     - 机器人底座（约 (-0.60, 0)）离连线不到 10 cm 的只有 `alphabet_soup|orange_juice` 一条规则（5.7 cm，自然行 536/3216）；反事实布局里有 456/6404 行。
     - "图上标位置 + 程序判定"这一候选：用第 2 步渲染的相机参数，按物体可见中心的投影在图上判断哪一侧，余量 ≥ 5 cm 时只有 60/74 与真实标签一致；改用地面接触点的投影是 73/74。斜视图里的视差很大，这种事实要换俯视图或改标接触点。
   - 渲染两张对向图（gengpu A100，排除 qgpu2014）。
3. **S3：A 和 B 的 pilot**（等用户确认）
   - 各 1 个 seed，只看验证集，沿用 `verifier_training/` 的管线和损失设置；要把 `fd1000_contract` 里 V4 的事实键和强制前缀换成 V5 格式（`Line:`、`Target on robot side`、`Goal on robot side`、`Separated`）。
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
- **报告页**：
  - **主报告**是 <https://claude.ai/artifact/Ya7FQbZC9yEuNNyYhPBHHT>，源文件 `docs/generative_verifier/PHASE2_REPORT.html`，公开链接，改动立刻可见。第 13 版新增 `#v5`"下一阶段：V5"一节，和上周的结论、后续工作逐条对照。中英双语用 `data-l="zh"` / `data-l="en"`，两种语言都要改。
  - 精读页 DJPGiauxhPAC8BYE4L14Y5（`PAPER_READING.html`）第 5 版也有一节"本周进展"。那是用户一开始给错了链接时加的，是否保留由用户决定。
