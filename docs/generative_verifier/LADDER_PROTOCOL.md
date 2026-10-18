# 第 2 步协议：B 的输出里哪一部分让判定可学、能不能推广到没见过的源

2026-10-18。接在 [CONTROLS_REPORT.md](CONTROLS_REPORT.md) 之后。
本文件和 `verifier_training/ladder_contract.py` 在 GPU 作业提交前一起提交到 Git，作为这一步的预注册。

## 1. 要回答的问题

单通路对照发现：判定前先写出受保护颜色（A+P），16 行就能拟合。这一步回答两件事：

1. 起作用的是写出的颜色，还是判定前多出来的几个输出 token？
2. 在 512 行上训练后，A+P 在没见过的源上能不能达到 B 的水平？

## 2. 授权与设计

- **授权**：用户 2026-10-18 要求把"只加 token 的对照"和第 2 步对照放在同一批。
- **相对讨论稿的两处调整**（提交前需要用户确认）：
  - **A 和 B 复用 Phase 2 的运行**：数据、样本顺序、超参数和验证集读法都与这次相同，三个检查点的验证记录都在，不重训。
  - **R 换成 AF**：B 的判定在每一行上都等于"它自己预测的关系 + 程序规则"，所以 R 这个读法就是 B 本身；R 也没有判定 token，需要另写一条损失路径。换成 AF 后，每一级只加一样东西：

| 级 | 模型要输出的文字 | 比上一级多了 | 来源 |
|---|---|---|---|
| A | `Verdict: d` | — | Phase 2 |
| AF | `Protected: box`，换行，`Verdict: d` | 判定前 4 个固定 token，不带任何输入信息 | 本次 |
| AP | `Protected: <颜色>`，换行，`Verdict: d` | 这 4 个 token 里换成真实的受保护颜色 | 本次 |
| B | 四个结构化事实，再 `Verdict: d` | 两个包含关系和 Separated | Phase 2 |

- AF 与 AP 的 token 位置完全相同（`Protected` `:` ` box/orange/purple` `\n`），只有第 3 个 token 不同。
- 不读测试集；只评估验证集；不调用 API；不做超参数搜索。

## 3. 设置

**第 1 部分：AF 的 16 行拟合**

与单通路对照完全相同：同样的 16 行、样本顺序、超参数和训练循环，3 个 seed，160 次更新。AP 的 16 行结果已在单通路对照中得到。

**第 2 部分：512 行训练、验证集评估**

- AF 和 AP 各 3 个 seed，用 Phase 2 的训练配置：384 次更新（3 个 epoch），有效批量 4，学习率 1e-4，LoRA r16。
- **样本顺序**用 Phase 2 `ORDERS.json` 里每个 seed 的顺序，与 Phase 2 的 A、B、C 完全相同；作业开始时逐批核对。
- 在第 128、256、384 次更新后，对 192 个验证行做 greedy 自由生成，每次保存 checkpoint。
- A 和 B 的数字读自 Phase 2 同一批验证行、同样三个检查点的记录。
- **输入**：704 行的推理输入与 Phase 2 preflight 的哈希逐行核对；16 行子集另做 native processor 校验。

## 4. 判定规则（作业开始前冻结在 `config.json`）

**第 1 部分**，分类规则与第 1 步相同：

| AF | 记为 | 含义 |
|---|---|---|
| 能拟合 | `extra_tokens_restore_fit` | 判定前多 4 个固定 token 就足以拟合，颜色内容对拟合不是必需的 |
| 不能拟合 | `colour_needed_for_fit` | 起作用的是写出的颜色，不是多出的 token |
| 部分或 mixed | `inconclusive` | 如实报告 |

**第 2 部分**（主要读第 384 次更新）：

- **方法**：每行先对 3 个 seed 平均，再在 14 个验证源上做源级 bootstrap（1000 次，seed 1014）和精确 sign-flip 检验（`fd1000_metrics.clustered_difference`）。
- **读法**：差值（处理组 − 对照组）的 95% 区间：
  - 下界 > 0，记为 `treatment_higher`；
  - 上界 < 0，记为 `treatment_lower`；
  - 其余记为 `no_detected_difference`。这**不等于两者相同**，报告时给出区间宽度。
- **预先确定的四个比较**：

| 比较 | 回答的问题 |
|---|---|
| AP − B | 两个包含关系在写出颜色之外，是否还提高了验证集准确率 |
| AP − A | 写出颜色是否提高了验证集准确率 |
| AF − A | 只多几个固定 token 是否提高了验证集准确率 |
| AP − AF | 颜色内容在多出的 token 之外是否还有作用 |

- **参照**：用 Phase 2 记录试跑报告代码，B − A 在验证集上是 +34.5 个百分点，区间 [+29.8, +39.5]，半宽约 5 个百分点。所以在 14 个源上，大约 10 个百分点以上的差距应能测出。
- **同时报告**：各级在三个检查点的准确率（平均和单个 seed）、单元四行全对、规则翻转一致、false-feasible，以及训练集最后 32 次更新的判定损失（看在 512 行上是否拟合）。

## 5. 资源与记录

- **预计用量**：约 2.5 A100 小时，一个 `gengpu` 作业（`v4_gpu.sbatch`，时限 4 小时）。
  - 第 1 部分约 20 分钟；
  - 第 2 部分每次运行约 18–19 分钟，共 6 次；
  - 编码和检查约 8 分钟。
- **启动**：`python -m verifier_training.launch_fd1000 ladder --root <单通路对照 campaign>`。
  - 新建输出目录 `verifier_training/runs/<UTC>_step2-ladder_<commit>/`，并在那里归档源码；
  - `RECEIPT.json` 记录读取的 Phase 2、第 1 步和单通路对照文件的哈希；
  - 这些目录都只读。
- **输出**：
  - `overfit/AF_seed<N>/`：第 1 部分；
  - `train512/<级>_seed<N>/`：训练曲线、三个检查点的验证记录和 adapter、`VALIDATION_SCORES.json`、`RESULT.json`；
  - `REPORT.json`、`REPORT.md`、`COMPLETE.json`。
- **限制**：
  - 这是开发数据上的探索性验证结果；锁定测试集已用过，不再评估。确认性结论需要新的锁定源。
  - AF、AP 与 Phase 2 的 A、B 在不同 commit 上运行；编码、损失和更新函数相同，此后的改动都不改变原有行为。
- **暂缓**：R、初始化时的梯度测量、新的锁定源（需要单独授权数据采集）。
