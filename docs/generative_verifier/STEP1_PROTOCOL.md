# 第 1 步协议：A 基线能否拟合训练数据

2026-10-17。对应 [NEXT_STEPS.md](NEXT_STEPS.md) §7.2 第 1 步。
本文件和 `verifier_training/fit_contract.py` 在 GPU 作业提交前一起提交到 Git，作为这一步的预注册。

## 1. 要回答的问题

Phase 2 里，A（直接判断）三个 seed 在训练集上的判定损失都停在 0.70 左右（ln 2），验证和测试都是常数预测。
这一步要分清：A 是因为**实现问题**学不会，还是**优化停滞**（需要学"图像 × 规则"的交互，在当前预算内没学起来）。

- **授权**：用户 2026-10-17 同意按本协议执行，1a 和 1b 放在同一个 GPU 作业里。
- **不做的事**：
  - 不读测试集，也不再评估锁定测试集；
  - 不评估验证集；
  - 不调用 API；
  - 不做超参数搜索。

## 2. 设置

| 子步 | 内容 |
|---|---|
| 1a | 用 Phase 2 保存的 A adapter（seed 713/714/715，第 128 和 384 步，共 6 个），在全部 512 个训练行上做 greedy 判定，eval 模式（不开 dropout），在判定 token 处读 P(1) |
| 1b | 从 4 个构造格子各取 1 个训练单元，共 16 行（8 可行、8 不可行，来自 4 个不同的源）。A 和 B 各 3 个 seed，都从头训练新的 LoRA：40 个 epoch，共 160 次更新；在第 40、80、120、160 次更新后对这 16 行做 greedy 评估 |

- **1b 选哪些单元**：每个格子里按 `sha256("v4-fit16/" + unit_id)` 排序，取第一个所在源还没被选过的单元。
  实际选中 `unit_130`、`unit_067`、`unit_165`、`unit_042`。
- **超参数**：与 Phase 2 完全相同，包括 LoRA r16、α32、dropout 0.05、学习率 1e-4、AdamW、梯度裁剪 1、有效批量 4、损失 0.5 判定 + 0.5 其他 token。
  启动器会检查 Phase 2 的 `config.json` 和本次配置里记录的 Phase 2 配置逐项相等。
- **代码路径**：
  - 编码、掩码、损失和每次更新都直接调用 Phase 2 的代码（`fd1000_run.encode`、`losses`、`update`）；
  - `update` 是从 `train()` 原样抽出来的函数，行为不变；
  - 输入（prompt token 和像素）逐行和 Phase 2 preflight 的哈希核对。

## 3. 判定规则（作业开始前冻结在 `config.json`）

**拟合损失**：只看决定性的 token。

- A：判定数字的交叉熵；
- B：四个事实值的交叉熵。B 的判定在 teacher forcing 下由它自己的事实决定，判定损失本来就接近 0，所以不用它。

**1b 每次运行**（看第 160 次更新后的结果，"最后一个 epoch"指最后 4 次更新）：

| 类别 | 条件 |
|---|---|
| 能拟合 | greedy 准确率 16/16，并且最后一个 epoch 的平均拟合损失 < 0.1 |
| 不能拟合 | greedy 准确率 ≤ 10/16，或者最后一个 epoch 的平均拟合损失 ≥ 0.6 |
| 部分 | 其他情况 |

- **每个组的结论**：3 个 seed 中至少 2 个同类，就取该类；否则记为 mixed。

**1a**（看第 384 步的 adapter）：

| 类别 | 条件 |
|---|---|
| 停在随机水平 | 3 个 seed 的训练集准确率都 ≤ 0.6 |
| 能拟合 | 至少 2 个 seed 的训练集准确率 ≥ 0.9 |
| 部分 | 其他情况 |

**决策表**：

| A 16 行 | B 16 行 | A 512 行 | 结论 | 下一步 |
|---|---|---|---|---|
| 不能 | 能 | — | `implementation_suspect` | 先逐项检查 direct 目标的编码、掩码和损失，再做第 2 步 |
| 不能 | 不能 | — | `shared_pipeline_suspect` | 先检查 A、B 共用的训练路径 |
| 能 | — | 随机水平 | `optimization_stall` | 第 2 步加一个事先声明的 A 训练预算变体 |
| 能 | — | 能 | `fits_training_data` | A 能拟合训练数据，验证集上的失败属于泛化问题；第 2 步按原样跑 A |
| 其他组合 | | | `inconclusive` | 如实报告，讨论后再定第 2 步 |

## 4. 资源与记录

- **预计用量**：约 1.25 A100 小时，在一个 `gengpu` 作业里完成（`v4_gpu.sbatch`，时限 4 小时）。
- **启动**：`python -m verifier_training.launch_fd1000 fit --root <Phase 2 campaign>`。
  - 启动器会新建输出目录 `verifier_training/runs/<UTC>_step1-fit_<commit>/`，并在那里归档源码；
  - `RECEIPT.json` 记录读取的 Phase 2 文件的哈希（6 个 adapter、preflight、训练曲线）；
  - Phase 2 目录只读，不写入任何东西。
- **输出**：
  - `train_evaluation/`：1a 的逐行记录和汇总；
  - `overfit/<组>_seed<N>/`：1b 的训练曲线、各次评估、最终 adapter 和 `RESULT.json`；
  - `REPORT.json`、`REPORT.md`、`COMPLETE.json`。
- **附带记录**：报告里还附有 Phase 2 九次运行最后 32 步的训练损失，只作背景，不参与判定。
