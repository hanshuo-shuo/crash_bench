# V5 测试评估：6 个选定的 adapter 在冻结测试集上各评估一次

2026-10-26 起草。用户 2026-10-26："都改一下 然后做测试级"（批准加 seed 之后的测试评估）。
**已于 2026-10-26 冻结**，见文末冻结记录。本文件和 `verifier_training/v5_test_contract.py` 在作业提交前一起提交。
前情见 [V5_SEEDS_REPORT.md](V5_SEEDS_REPORT.md)（读法 `stable`）。

## 1. 要回答的问题

验证集上 3 个 seed 的 P（先写位置、再用程序判定）都是 93–95%，A（直接判断）都是 50%。测试集来自另外 4 条源，至今没有读过。

- P 的优势在没见过的源上还在不在？
- A 在测试集上是不是仍然在 50% 左右？

- **授权**：用户 2026-10-26（见上）。
- **范围**：S3 的 `A_seed713`、S4 的 `P_seed713`、加 seed 的 `A_seed714`、`A_seed715`、`P_seed714`、`P_seed715`，共 6 个选定的 adapter，各评估一次。
- **不做的事**：
  - 不训练，不重新选模，不改任何生成设置；
  - 不评估 B（S3 的 `B_seed713`）和零样本模型，它们不在这次授权里；
  - 不重新渲染，不采集新数据，不调用 API；
  - 测试结果不用于任何后续的选模或调参。

## 2. 设计

- **新 campaign**：`v5-test`，新建运行目录。S3、S4、加 seed 的运行目录只读。
  - `RECEIPT.json` 记录读到的每个旧文件的哈希（3 个运行目录共 42 个文件，包括 6 个 adapter 和它们的验证记录），每个作业开始时逐个核对。
  - `FREEZE.json` 在启动时写定 6 个 adapter：所在目录、选定的更新次数（A 713/714/715 是 448/150/448，P 是 448/448/300）、adapter 和 `RESULT.json` 的哈希、验证准确率。
- **测试行**：S2 冻结的 `test`（1,154 行反事实孪生，378 张图，4 条源）和 `test_natural`（80 行自然布局，40 张图）。
  - 只有本 campaign 的 `v5_test_run.context` 会读测试行，`validate(include_test=True)` 只在这里调用；读之前用 `TEST_FREEZE.json` 核对全部 1,234 行的哈希。
- **preflight（CPU）**：
  - 测试行与 `TEST_FREEZE.json` 的行列表、图像哈希逐个相同；每个测试单元的 `CAMERA.json`、`RENDER_GEOMETRY.json` 与该单元的 `HASHES.json` 相同；
  - 每一行 A、P 的推理输入（提示 token 加像素）相同，用原生 processor 核对快速编码，不截断；
  - 用 `v5_pos_contract.image_points` 把仿真几何投影成第 1 张图上的真值点，每一行用真值点跑程序，结论都要与标签一致；
  - 用训练集学"规则 × 进度"查表，测试集上的准确率要等于 S2 报告的值（`test` 0.500，`test_natural` 0.9125）。
- **重新加载核对（每个 GPU 作业读测试行之前）**：
  - 用 `PeftModel.from_pretrained` 把选定的 adapter 加载到 bf16 基座上，和训练时的评估一样打开 TF32 矩阵乘，再在 32 行固定的验证行上重新生成（按 `sha256(行号|v5-test-reload)` 排序取前 32 行）；
  - 要求每一行的判定（A 的 Verdict，P 的程序判定）和输入摘要都与该 adapter 记录的验证结果相同，生成的文字最多允许 0 行不同；
  - 不满足就停止，不读测试行，写 STOP 文件。

## 3. 评估（沿用 S3、S4，不改任何一项）

- A 用 `v5_run.item` 和 `v5_run.evaluate`（读法 `direct_free`，最多 8 个新 token）；P 用 `v5_pos_run.p_item` 和 `v5_pos_run.evaluate`（最多 128 个新 token）。这两段代码就是验证集上用的那两段。
- 贪心解码，不给任何金标前缀。
- 每个 adapter 在 `test` 和 `test_natural` 上各生成一次，逐行写进记录文件；作业结束时写 `COMPLETE.json`（行数、记录哈希）。打分全部放在 CPU report 里做。
- **只评估一次**：每个 adapter 的输出目录用 `mkdir()` 新建，已存在就报错；启动器不允许同一标签重复提交。
  - 只有在 `COMPLETE.json` 写出之前出了基础设施故障（节点故障、超时），才可以用同一份代码和同一个 adapter 重跑一次，失败的那次记录保留在旁边，并在报告里写明。

## 4. 读数（测试）

| 读数 | 定义 |
|---|---|
| T1 | 每个 seed 在 `test`（1,154 行）上：A 的判定准确率；P 的程序判定准确率（主读数）和 P 自己写的 Verdict 的准确率（次读数） |
| T1 配对 | 每个 seed 的 P − A 配对差，以图像为单位的 bootstrap 95% 区间 |
| T1 平均 | 每一行先按 3 个 seed 求 A、P 各自的平均正确率，再对 P − A 求行平均；以图像为单位做 bootstrap，求 95% 区间（1,000 次，seed 1014）；按源（4 条）的区间只作参考 |
| T2 | 每个 seed 的孪生图全对率（程序判定） |
| T3 | P 四个点的像素误差（中位数，15、30 像素内的比例）；P 自己写的 `Separated`、`Verdict` 的分布 |
| T4 | 每个 seed 在 `test_natural`（80 行）上的准确率 |
| 对照 | 每个 seed 从验证到测试的变化（A、P 各一个数）；漏判率、误拒率；与查表的一致率；按源、按进度段的准确率 |

- 区间的含义与加 seed 时相同：3 个 seed 是固定的，区间只反映换一批图像的不确定性。

## 5. 读法（事先写定）

设 P_s 为 seed s 在 `test` 上的 P 程序判定准确率，A_s 为 A 的准确率，Δ_s = P_s − A_s（s = 713、714、715）。
Δ̄ 为 3 个 seed 平均后的 P − A，区间指 Δ̄ 的图像 bootstrap 95% 区间。门槛与加 seed 相同。

| 读法 | 条件 | 含义 |
|---|---|---|
| `test_confirms` | 3 个 seed 的 P_s 都 ≥ 0.65，每个 seed 的 Δ_s 都 ≥ 0.10，并且区间下限 > 0 | 验证集上的结论在没见过的测试源上成立 |
| `test_fails` | 至少一个 seed 的 P_s ≤ 0.60 | P 的优势没有推广到测试源，如实报告 |
| `unclear` | 其余情况 | 报告后讨论 |

- T2–T4 和对照只用来解释，不进入读法。
- 测试只评估这一次。无论读法是什么，都不回头改模型或设置后再测。

## 6. 运行与资源

- **启动**：`python -m verifier_training.launch_v5 test --s3 <S3 运行目录> --s4 <S4 运行目录> --seeds <加 seed 运行目录>`。
  - 前提：
    - 工作区干净，commit 已推送；
    - 本协议已冻结；
    - 加 seed 的读法是 `stable`；
    - S3 的 preflight、A_seed713、report，S4 的 preflight、P_seed713、report，加 seed 的 preflight、4 个训练和 report，都是 COMPLETED 0:0（按 p33100 账号过滤）；
    - 没有其他 CrashBench GPU 作业。
  - 新建 `verifier_training/runs/<UTC>_v5-test_<commit>/` 并归档源码。
- **作业**：
  1. CPU preflight（short 分区）；
  2. preflight 成功后，两条 GPU 线同时开始，各用 1 张 A100：
     - 线 1：P_seed713 → A_seed713 → A_seed714 → A_seed715；
     - 线 2：P_seed714 → P_seed715；
  3. 6 个评估都结束后跑 CPU report，写 `TEST_READING.json`。
- **预计用量**：P 在验证集上约 4.8 秒一行，每个 P 约 1.7 小时；A 约 0.5 秒一行，每个约 12 分钟。合计约 6 个 GPU 小时，两条线约 3.5 小时。GPU 作业时限 8 小时。

## 7. 实现

| 文件 | 内容 |
|---|---|
| `verifier_training/v5_test_contract.py` | 配置 `V5-object-test-v1`（嵌入加 seed 的配置）；6 个目标和两条线；旧文件清单与哈希；adapter 冻结；重新加载核对的行和规则；读法 `test_reading` |
| `verifier_training/v5_test_run.py` | 阶段 `preflight`（CPU）、`evaluate TARGET`（GPU）、`report`（CPU）；要评估的划分写在 `RECEIPT.json` 里（正式是 `test`、`test_natural`，smoke 用 `validation_natural`） |
| `verifier_training/launch_v5.py` | 新动作 `test`；`test_root` 建目录并写冻结和收据，`test_jobs` 提交作业 |
| `verifier_training/tests/test_v5_test_contract.py` | 配置；目标和线；重新加载核对；读法；冻结读到的选定更新次数与 adapter 哈希；静态检查：V5 代码里只有 `v5_test_run.context` 会读测试行，每个 adapter 只评估一次，启动器检查冻结 |

## 8. 冻结记录

**状态：已冻结**（2026-10-26）。之后不再改读数、读法、重新加载核对的规则和生成设置。

- **实现 commit**：`72d825b`（草稿），`f071896`（评估时打开 TF32，见下）。冻结提交只改本文件的文字，代码不变。
- **测试**：`verifier_training/tests` 共 164 个全部通过（原有 153 个，新增 11 个）。
- **smoke**：在草稿目录里跑，评估的划分是 `validation_natural`（79 行，开发集）。两次 smoke 都没有读测试行（每个结果文件都记了 `test_rows_loaded: 0`），也不进入读数。
  - **第一次**（源码 `72d825b`，作业 5415751–5415758）：
    - preflight 通过：79 行、40 张图，查表 0.747 与 S2 相同，最长 717 token；
    - A 的 3 个 adapter，重新加载后的 32 行逐字复现；
    - P 的 3 个停在重新加载核对：坐标差几个像素，文字相同的只有 2、5、3 行（共 32 行），判定相同 30、31、32 行。
    - 原因：训练时的评估打开了 TF32 矩阵乘，LoRA 是 fp32，评估阶段没有打开。修正为 `f071896`。
  - **第二次**（源码 `f071896`，作业 5425363–5425370，全部 COMPLETED 0:0）：
    - 6 个 adapter 的重新加载核对都是 32/32 行逐字相同；
    - 在 `validation_natural` 上，6 个 adapter 的 79 行生成文字和判定都与训练时的记录逐字相同；
    - 这在不同型号的卡之间也成立：A713（SXM 训练、PCIe 评估）、P714（PCIe 训练、SXM 评估）、P715（SXM 训练、PCIe 评估）；
    - report 写出 `SMOKE_READING.json`，每个 seed 的准确率与加 seed 报告的 R4 相同。这次 smoke 的读法（`unclear`）是在 79 行自然布局上演练，不是结果。
- **据此确定**：重新加载核对允许的文字差异保持 0 行；不需要指定 GPU 型号。P 每行约 5 秒，与第 6 节的估计一致。
