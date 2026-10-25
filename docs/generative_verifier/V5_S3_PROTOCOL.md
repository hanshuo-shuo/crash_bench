# V5 最小故事闭环：第 S3 步，A 对 B 的 pilot

2026-10-25 起草，用户当天批准开始（"可以 开始写"）。**启动前必须冻结**：实现完成并通过测试后，在本文件末尾记下实现的 commit 和"冻结"字样，之后不再改读数和读法。
数据见 [V5_S2_REPORT.md](V5_S2_REPORT.md)，交接见 [V5_HANDOFF.md](V5_HANDOFF.md)。

## 这一步回答什么

故事的核心问题：训练数据里有捷径（只看"规则 × 进度"就能猜对 76%），测试按构造挡住这条捷径（只能得 50%）。在这种情况下：
- 直接判断（A）会不会学成捷径，在反事实孪生上掉回 50%？
- 先写关系事实（B）的模型，会不会因为要写"布丁在哪侧""篮子在哪侧"而真的去看图，从而做对？

## 数据

- S2 数据集：`/projects/p33100/siosio/crashbench_safelibero/verifier_dataset/20261025T013358Z_v5-s2_d1d8d4129223/dataset.jsonl`（`V5-object-v1`）。
- 启动时核对 `HASHES.json`、`S2_COMPLETE.json`（`passed`、`s2_dataset_ready`）和 `TEST_FREEZE.json`。
- 训练只读 `train`；选模和读数只读 `validation` 和 `validation_natural`。
- `test`、`test_natural` 只在 3 个 seed 全部训练、选模冻结之后，由单独的评估阶段读一次。本 pilot 不读测试。

## 两组

| 组 | 目标文本 | 选模读数 |
|---|---|---|
| A | `Verdict: 0/1` | 自由生成的判定准确率 |
| B | `Line: …` / `Target on robot side: …` / `Goal on robot side: …` / `Separated: …` / `Verdict: 0/1` | 自由生成的判定准确率（不给任何金标前缀） |

V4 的 C 组不做。

## 训练设置（沿用 V4 Phase 2，只改与数据规模有关的两项）

- **与 V4 相同**：
  - 模型：Qwen2.5-VL-3B-Instruct（已暂存的同一份），LoRA r16、α32、dropout 0.05，只挂在语言层的 252 个线性层上，视觉和投影冻结；
  - 优化：AdamW，学习率 1e-4，权重衰减 0，梯度裁剪 1.0，micro batch 1，有效 batch 4；
  - 损失：0.5 × 判定 token 的交叉熵 + 0.5 × 其余 assistant token 的平均交叉熵；
  - 输入：448×448 两张图，最长 2048 token。
- **改动 1，更新次数**：448 次，恰好一个 epoch（1,792 行 ÷ 4）。V4 是 512 行 × 3 epoch = 384 次。
  - 样本顺序是按 seed 打乱的训练行排列；A 和 B 用同一个顺序。
- **改动 2，选模评估**：在第 150、300、448 次更新时，用固定的选模子集评估。
  - 子集：验证集里两种答案都有的图像，按 `sha256(图像|v5-s3-select)` 排序取前 192 张；每张取一条分隔、一条不分隔的规则，共 384 行。
  - 平分时选较早的 checkpoint。
  - 选定后，在完整验证集（1,538 行）和验证自然（79 行）上各评估一次。
- **生成**：贪心解码，最多新 token 数：A 8 个，B 96 个。
- **seed**：pilot 只用 713。方向清楚后再加 714、715，那是下一次提交，需要另行确认。

## 读数（只看验证）

| 读数 | 定义 |
|---|---|
| R1 判定准确率 | 完整验证集（反事实孪生，组内配平）上 |
| R2 孪生图全对率 | 两种答案都有的图上，这张图的所有行都对的比例 |
| R3 B 的事实 | `Target on robot side`、`Goal on robot side`、`Separated` 各自的准确率；格式合规率；判定与"模型自己写的两个事实按程序算出的判定"一致的比例 |
| R4 自然诊断 | 验证自然上的判定准确率 |
| R5 训练拟合 | 固定的 256 行训练子集（按 sha256 取）上的判定准确率，对照查表上限 0.760 |
| R6 零样本对照 | 在选模子集（384 行）上，用直接判断提示和写事实提示各测一次：(a) Qwen2.5-VL-3B 基座（与 V4 相同）；(b) 本地 Qwen3-VL-32B-Instruct |

- **不确定性**：以图像为单位做 bootstrap，报告 95% 区间。验证只有 4 条源，按源聚类的区间只作参考。
- **R6 的资源**：(b) 的权重 63 GB，需要两张 A100-40GB（`device_map=auto`）或一张 80 GB / H100。不花 API。

## 读法（事先写定）

设 A、B 为 R1，Δ = B − A：

| 读法 | 条件 | 含义 |
|---|---|---|
| `b_grounds` | B ≥ 0.65，且 Δ ≥ 0.10，且 Δ 的 95% 区间下限 > 0 | 写事实让模型看了图，A 没有。下一步：3 个 seed，然后测试评估一次 |
| `both_ground` | A ≥ 0.65，且 Δ < 0.10 | 训练里有反事实布局时，直接判断也学会了。故事要改成"起作用的是数据里的反事实布局" |
| `both_shortcut` | A ≤ 0.60 且 B ≤ 0.60，并且至少一组的 R5 ≥ 0.70 | 两组都学成了捷径，写事实挡不住。下一步讨论（例如提高反事实比例，或改用"标位置 + 程序判定"的后备方案） |
| `unclear` | 其余情况 | 报告后讨论 |

R3 用来解释原因：
- B 的判定错了而事实对了，说明问题在组合；
- 事实错了，要看错在物体侧还是篮子侧。篮子侧有先验，查表就能猜对 0.676。

## 要实现的代码（下一个会话做）

V4 的 `verifier_training/fd1000_*` 和 V4 格式绑得很紧。V4 的代码不改动行为，新代码另起文件：

1. **`verifier_training/v5_contract.py`**：
   - 配置（版本 `V5-object-AB-v1`）；
   - `FACT_KEYS = ('line', 'target_robot_side', 'goal_robot_side', 'separated')`，B 的强制前缀 `Line:`；
   - 目标文本直接复用 `verifier_dataset.v5_dataset.facts_text`，加上 `\nVerdict: ` 和数字；
   - 程序判定：Separated 为 yes 时判 0；
   - 数据校验：核对哈希，确认训练阶段不加载测试行；
   - 样本顺序：一个 epoch、A 和 B 相同；
   - 选模子集和训练拟合子集的选取。
2. **`fd1000_run.encode`**：加一个可选参数，替换强制前缀的边界检查。现在写死成 V4 的 `Protected:`，V5 的目标里没有这个前缀，检查会失败。默认值保持 V4 行为不变。`answer=`、`keys=` 参数已经存在，可以直接用。
3. **`verifier_training/v5_metrics.py`**：
   - 解析 B 的输出，严格格式为 `Line: (.+)\nTarget on robot side: (yes|no)\nGoal on robot side: (yes|no)\nSeparated: (yes|no)\nVerdict: [01]`；
   - 计算 R1–R5 和图像级 bootstrap。
4. **`verifier_training/v5_run.py`**：
   - 阶段：`preflight`（编码、token 边界、不截断、A/B 推理输入相同）、`train ARM SEED`、`zero-shot-3b`；
   - 复用 `fd1000_run` 的 `vision_inputs`、`adapter_model`、`update`、`run_generation`、`losses`。
5. **`verifier_training/v5_zero_shot_32b.py`**：用 `qwen-vlm` 环境（transformers 5.7）；两种提示与 3B 零样本相同。
6. **启动器 `launch_v5.py` 和 sbatch**：
   - 工作区干净且已发布；按 S2 根目录建运行目录；
   - `preflight` 通过后再提交 A、B 两个训练作业，同时最多 2 张 GPU；
   - 每个运行目录记录 commit、配置、seed、Slurm 号。
7. **测试**：
   - 目标文本与 S2 的 `target.rationale` 一致；
   - 强制前缀边界；
   - 解析器能识别格式不合规的输出；
   - 选模子集是平衡的；
   - 训练顺序覆盖全部训练行各一次，A 和 B 顺序相同；
   - 训练阶段拿不到测试行。

## 资源

- 训练：每组约 1–1.5 个 A100 小时，B 的评估较慢，预计再加约 1 小时。
- 零样本 3B 约 0.5 小时；32B 需两张卡，约 1 小时。
- 合计约 5–6 个 GPU 小时，同时最多 2 张卡。不调付费 API。

## 冻结记录

（实现完成、测试通过、启动前在这里写：实现 commit、冻结日期。）
