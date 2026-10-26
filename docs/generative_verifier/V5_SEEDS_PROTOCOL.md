# V5 加 seed：A 和 P 各补 714、715（只看验证）

2026-10-25 起草。加 seed 此前已获批；用户 2026-10-25 说"收尾下 然后让下一个agent开始干下一步"，交给新的 agent 做。
**已于 2026-10-25 冻结**，见文末冻结记录。本文件和 `verifier_training/v5_seeds_contract.py` 在作业提交前一起提交。
前情见 [V5_S3_REPORT.md](V5_S3_REPORT.md) 和 [V5_S4_REPORT.md](V5_S4_REPORT.md)。

## 1. 要回答的问题

S4 只有 seed 713：反事实孪生验证上 P 0.940，A 0.501。这一步看换了 seed 之后，这个差距还在不在。

- P 换 seed 后还能不能保持在高位？
- A 换 seed 后是不是还在随机水平？会不会在某个 seed 上也学会看图？

- **授权**：用户 2026-10-25（见上）。
- **不做的事**：
  - 不读测试集（`test`、`test_natural`）；
  - 不改任何训练设置，不做超参数搜索；
  - 不重训 seed 713，不再跑零样本（S3、S4 已有）；
  - 不重新渲染，不采集新数据，不调用 API。

## 2. 设计

- **新 campaign**：`v5-seeds`，新建运行目录。S3、S4 的运行目录只读。
  - `RECEIPT.json` 记录读取的 S3 文件（10 个）和 S4 文件（9 个）的哈希，每个作业开始时逐个核对。
- **3 个 seed**：
  - 713 直接用已有记录：S3 的 `A_seed713`、S4 的 `P_seed713`；
  - 714、715 新跑，A 和 P 各一次，共 4 次训练。
- **A**：与 S3 的 A 完全相同。
  - 目标 `Verdict: 0/1`；选模值是自由生成的判定准确率；最多生成 8 个 token。
  - 代码是 `v5_run.train_arm`：把 S3 的 `train` 函数体原样抽出来，S3 的入口行为不变。
- **P**：与 S4 的 P 完全相同。
  - 目标先写两个锚点、布丁、篮子在第 1 张图上的位置，程序判定；选模值是程序判定的准确率；最多生成 128 个 token。
  - 代码是 `v5_pos_run.train_p`：同样是 S4 的 `train` 函数体原样抽出。
- **配置**：
  - `v5_seeds_contract.configuration()` 原样嵌入 S3、S4 的配置。它们的 `seeds` 仍是 [713]，不改；新 seed 写在本 campaign 自己的配置里。
  - 每个作业开始时核对：S3、S4 的 `config.json` 与嵌入的配置相同。
- **样本顺序**：
  - 每个新 seed 用 `v5_contract.schedule` 生成：一个 epoch，448 次更新，每次 4 行。
  - 同一个 seed 的 A 和 P 用同一个顺序。P 用 `schedule` 给 B 的那一份（目标类型 facts），这和 S4 对 713 的做法相同。
  - preflight 先用同一个函数重算 713 的顺序，要求与 S3 的 `ORDERS.json` 完全相同，并且 S4 的 P 顺序等于 S3 的 B 顺序。
  - 714、715 的顺序都覆盖每个训练行恰好一次，彼此不同，也和 713 不同。
- **选模子集、训练拟合子集、查表**：preflight 重算，要求与 S3 的 `SUBSETS.json`、`LOOKUP.json` 完全相同，再写进新目录。
- **位置真值**：preflight 用 `v5_pos_contract.image_points` 对 1,332 张图重算，要求与 S4 的 `POINTS.json` 完全相同；每一行用真值点跑程序，结论都要与标签一致（沿用 S4 的检查）。
- **输入**：
  - 每一行 A、P 的推理输入（提示 token 加像素）都要与 S3 preflight 记录的 `input_sha256` 逐行相同；
  - 用原生 processor 核对快速编码；不截断。

## 3. 训练与评估（沿用 S3、S4，不改任何一项）

- 模型、LoRA、优化器、损失（0.5 判定 + 0.5 其余 token）、448 次更新；
- 选模时刻 150 / 300 / 448，选模子集 384 行，平分时选较早的 checkpoint；
- 选定后在完整验证集（1,538 行）、验证自然（79 行）和训练拟合子集（256 行）上各评估一次；
- 贪心解码，不给任何金标前缀；
- 每个 seed、每组都从新的基座和新的 LoRA 开始。seed 决定样本顺序、LoRA 初始化和 torch 的随机数。
- 和 S3、S4 一样，选模子集是验证集的一部分：读数的 1,538 行里包含选模用过的 384 行。

## 4. 读数（只看验证）

| 读数 | 定义 |
|---|---|
| R1 | 每个 seed 在完整验证集（1,538 行）上：A 的判定准确率；P 的程序判定准确率（主读数）和 P 自己写的 Verdict 的准确率（次读数） |
| R1 配对 | 每个 seed 的 P − A 配对差，以图像为单位的 bootstrap 95% 区间；再按 S4 的读法单独给出该 seed 的读法（只作参考） |
| R1 平均 | 每一行先按 3 个 seed 求 A、P 各自的平均正确率，再对 P − A 求行平均；以图像为单位做 bootstrap，求 95% 区间（1,000 次，seed 1014）；按源（4 条）的区间只作参考 |
| R2 | 每个 seed 的孪生图全对率（程序判定） |
| R3 | 每个 seed 的 P 四个点的像素误差（中位数，15、30 像素内的比例）；P 自己写的 `Separated`、`Verdict` 的分布 |
| R4 | 每个 seed 在验证自然（79 行）上的准确率 |
| R5 | 每个 seed 在训练拟合子集（256 行）上的准确率 |
| 稳定性（只作解释） | 3 个 seed 的最小值和最大值；3 个 seed 都对、都错的行数；3 个 seed 预测完全相同的行的比例 |

- **区间的含义**：3 个 seed 是固定的，区间只反映"换一批图像"的不确定性。seed 之间的差别看逐 seed 的数和最小、最大值。

## 5. 读法（事先写定）

设 P_s 为 seed s 的 P 程序判定准确率，A_s 为 A 的准确率，Δ_s = P_s − A_s（s = 713、714、715）。
Δ̄ 为 3 个 seed 平均后的 P − A，区间指 Δ̄ 的图像 bootstrap 95% 区间。

| 读法 | 条件 | 含义 |
|---|---|---|
| `stable` | 3 个 seed 的 P_s 都 ≥ 0.65，每个 seed 的 Δ_s 都 ≥ 0.10，并且区间下限 > 0 | "先写位置、再用程序判定"的优势不依赖 seed。下一步：在测试集上评估一次（需要用户确认，还要写测试阶段的代码） |
| `unstable` | 至少一个 seed 的 P_s ≤ 0.60 | 713 的结果不稳。如实报告，讨论原因 |
| `unclear` | 其余情况 | 报告后讨论 |

- `stable` 要求每个 seed 的差距都至少 0.10。如果 A 在某个 seed 上也学会了看图（比如 0.85 对 0.94），读法就不会是 `stable`。
- 逐 seed 的 S4 读法和 A 的最大值只用来解释，不进入读法。

## 6. 运行与资源

- **启动**：`python -m verifier_training.launch_v5 seeds --s3 <S3 运行目录> --s4 <S4 运行目录>`。
  - 前提：
    - 工作区干净，commit 已推送；
    - 本协议已冻结；
    - S3 的 preflight、A_seed713、B_seed713、report，以及 S4 的 preflight、P_seed713、report，都是 COMPLETED 0:0（按 p33100 账号过滤）；
    - 没有其他 CrashBench GPU 作业。
  - 新建 `verifier_training/runs/<UTC>_v5-seeds_<commit>/` 并归档源码。
- **作业**：
  1. CPU preflight（short 分区）；
  2. preflight 成功后，两条 GPU 线同时开始，各用 1 张 A100：
     - 线 1：A_seed714，结束后 P_seed714；
     - 线 2：A_seed715，结束后 P_seed715；
     - 所以任何时候最多 2 张 GPU。
  3. 四个训练都结束后跑 CPU report，写 `SEEDS_READING.json`。
- **预计用量**：
  - preflight 约 20–30 分钟 CPU；
  - A 每个约 0.8 小时，P 每个约 4.5 小时；每条线约 5.3 小时，合计约 10.6 个 GPU 小时；
  - GPU 作业时限 8 小时（S4 的 P 用了 4 小时 28 分）。

## 7. 实现

| 文件 | 内容 |
|---|---|
| `verifier_training/v5_seeds_contract.py` | 配置 `V5-object-seeds-v1`（嵌入 S3、S4 配置）；新 seed 的样本顺序和核对；P 的程序判定视图；3 个 seed 的行平均和图像 bootstrap（`seed_average`）；读法 `seeds_reading` |
| `verifier_training/v5_seeds_run.py` | 阶段 `preflight`（CPU）、`train ARM SEED`（GPU，A 调 `v5_run.train_arm`，P 调 `v5_pos_run.train_p`）、`report`（CPU） |
| `verifier_training/v5_run.py`、`v5_pos_run.py` | `train` 拆成"核对 + 训练函数体"两部分；新增的可选参数 `provenance`、`scope` 只往 `PROVENANCE.json`、`RESULT.json` 里加字段；S3、S4 的入口行为不变 |
| `verifier_training/launch_v5.py` | 新动作 `seeds` |
| `verifier_training/tests/test_v5_seeds_contract.py` | 配置不变项；样本顺序覆盖和 A、P 相同；`schedule` 能复现 S3、S4 对 713 的顺序；读法；行平均（只放 713 时能复现 S4 的配对差和区间）；S3 的子集和查表能重算出来；静态检查不加载测试行；启动器检查冻结 |

## 8. 冻结记录

**状态：已冻结**（2026-10-25）。之后不再改读数、读法、子集、样本顺序和训练设置。

- **实现 commit**：`1a056d6`（`Implement the V5 seed campaign: A and P with seeds 714 and 715 (draft)`）。冻结提交只改本文件的文字，代码不变。
- **测试**：`verifier_training/tests` 共 153 个全部通过（原有 141 个，新增 12 个）。
- **smoke**：在草稿目录里，用 `1a056d6` 的源码，不进入读数，不读测试。
  - 作业 5341754（CPU，18 分钟）：用真实配置和真实的 S3、S4 目录，跑完整的 preflight。
    - 3,409 行、1,332 张图全部通过；A、P 的输入与 S3 逐行相同；最长 717 token，不截断；
    - 713 的样本顺序复现了 S3、S4 的记录；
    - 选模子集、训练拟合子集、查表、位置真值与 S3、S4 相同，写出的文件逐字节相同。
  - 作业 5341755（1 张 A100，10 分钟）：缩小的 S3、S4 目录，取自真实文件（16 个训练行、3 张孪生验证图、4 行验证自然）。
    - 每次训练 4 次更新，跑完 preflight，A、P 的 714、715，以及 report；
    - 梯度探针都非零，显存峰值约 9 GB；
    - 4 次更新后模型还没学会输出格式，按规定计为错；
    - report 能读到 713 的 S3、S4 记录，并按读法给出结果。
