# 生成式可行性 Verifier：研究与实验计划

2026-10-16（已吸收外部评审意见）。读者：项目负责人，以及之后执行本计划的 Codex/Claude 会话。
状态（2026-10-17）：**Phase 0–2 已完成**。结果见 [PHASE1_REPORT.md](PHASE1_REPORT.md) 和 [PHASE2_REPORT.md](PHASE2_REPORT.md)：
测试集上 B 比 A 高 29.7 个百分点（CI [+22.3, +36.8]），科学标准达成；开发里程碑未达成。Phase 3 等用户决定。所有 Quest 操作遵守 [AGENTS.md](../../AGENTS.md) 和
[QUEST_WORKFLOW.md](../../QUEST_WORKFLOW.md)：账号 p33100，GPU 分区 gengpu，CPU 分区 short，
最多同时 2 张 GPU，登录节点不跑重计算，每次运行使用新的输出根目录并记录 commit、配置、seed 和 Slurm ID。

## 0. 一页摘要

- **目标**：训练一个 VLM verifier，在执行前判断"在给定安全约束下，这个操作任务是否存在满足约束的解"，
  并比较"直接判别"和"先推理再判断"（GenRM 式生成式验证）。
- **数据决定（用户，2026-10-16）**：V4 的 896 行（224 个四行单元、64 条源轨迹）直接作为**开发数据**
  用于训练。原定的 shortcut 闸门（联合读数验证 AUROC ≤ 0.75）没有通过（0.780），不再作为训练前提；
  失败结果如实披露，该读数作为模型必须超过的基线报告。
- **为什么可以这样做**：诊断显示闸门信号是局部的盒子摆放伪迹（EXIT × checkpoint 0 上 AUROC 1.000，
  ENTRY × checkpoint > 0 只有 0.636），不是"整个任务不需要关系推理"；锁定测试集从未被任何模型看过。
- **这批数据能回答什么**：能检验"中间事实监督是否帮助判断"；不足以证明模型获得了通用的机器人规划推理能力。
- **四个研究问题**：
  1. 先生成事实再判断（B），是否比直接判断（A）更准？
  2. 如果更准，收益来自"rationale 作为训练信号"，还是"测试时显式推理"？（C 组的两种读法回答）
  3. 瓶颈在感知（看清包含关系），还是规则应用？（"模型预测事实 + 程序应用规则"的读法回答）
  4. verifier 能否作为 AEGIS 之前的拒绝执行门，减少无解场景中的违规？
- **文献先验**（§3）：
  - 在短的、偏感知的二分类上，测试时 CoT 的收益通常只有 0–3 分，甚至为负。
  - 大部分收益在"只训练、不推理"时也保留。
  - 带坐标的 grounded CoT 比纯文本 CoT 更可能有用。
  - GenRM 的优势很大程度来自多条 rationale 的投票。
- **"有效果"分两种**：
  - (a) 开发里程碑：任一组在 held-out 源上准确率高、反事实一致性高。它本身不足以当安全门用（§5）。
  - (b) 科学：B 显著优于 A；C 的推理读法优于它自己的直接读法。
- **下一步**：
  - Phase 0：导出数据和改代码，只用 CPU；顺手输出 32 行 rationale 样例。
  - Phase 1：零样本诊断，加一次小规模 A/B 工程试跑（测能否学会输出格式和实际耗时），只用 validation。
  - Phase 2：A/B/C 三组 × 3 个 seed；全部训练完、选模规则冻结后，再统一评估一次锁定测试集。
  - 之后按结果进入 Phase 3/4。B′ 和更大的模型按验证集发现再追加。

## 1. 问题定义

### 1.1 输入与输出

- **student 可见的输入**：
  - 两张 448×448 外部视角 RGB（`overview_a`、`overview_b`）。
  - 任务文本。
  - 约束文本，其中写明哪个颜色的虚拟盒子受保护。
- **输出**：`Verdict: 1`（可行）或 `Verdict: 0`（条件不可行）。推理组在 verdict 前先输出结构化事实（§4.0）。
- **student 不可见、只放在 metadata**：几何、证书、安全轨迹、源 ID、相机参数、分割掩码。

### 1.2 标签语义

- **正例**：一条真实的安全轨迹（源轨迹后缀）满足全部约束，所以解的存在性已被证明。
- **负例**：受保护的封闭虚拟盒子把目标和终点分在两侧（closed-separator 证书），所以在声明的合同下不可行。
- **在当前经过见证筛选的数据分布内**，这个 verifier 实际上是"证书检测器"：
  - 找到一个分隔目标和终点的受保护盒子，就判 0；找不到，就判 1。
  - "找不到"只说明这道边界没有把两者分隔，**不能**证明机器人够得着、抓得住、能在时间限制内完成。
    这条规则之所以在数据里成立，是因为每个正例都另有一条经过核验的安全轨迹；离开这个分布不一定成立。
- **特权信息的使用边界**：
  - 用证书、安全轨迹、几何来**制作**监督标签是允许的（GenRM 也用参考答案生成训练 rationale，推理时移除）。
  - 模型输出不能把特权信息说成自己看到的事实，例如"我观察到一条成功轨迹"、精确距离等。
- 所有结论都限定在声明的合同之内。

### 1.3 与 π0.5 / AEGIS 的关系

- 训练 Qwen verifier 只改变它自己的 LoRA，不会让 π0.5 获得推理能力。
- 本计划把 verifier 当作独立模块，用作执行前的拒绝执行门（Phase 4）。
- 蒸馏进 VLA 不在本计划范围内，理由见 §3.6。

## 2. 现状

### 2.1 V4 数据

来源：[verifier_dataset/CURRENT.md](../../verifier_dataset/CURRENT.md)。

| 项 | 值 |
|---|---|
| 行数 | 896（448 可行 / 448 条件不可行） |
| 单元 | 224 个四行单元：同一张图配两条规则；把盒子颜色互换后再渲染一张，同样配两条规则 |
| 源 | 64 条原始轨迹（每条一个 ancestry group），12 个任务组合，4 个 family |
| 按源切分 | train 512 行 / 35 个源；validation 192 / 14；test 192 / 15（锁定，从未评分） |
| 构造格子 | EXIT/ENTRY（分隔盒包住目标 / 包住终点）× BOTH/NEITHER（另一个盒子同时包住两者 / 都不包）；每个 split 内四格等量 |
| 图像审核 | 由 Codex 逐单元做过前提可见性审核，不是独立的人工审核 |
| 原始记录 | 本地 `results/verifier_dataset/factorial_v4_final_896_20261015/AUDIT_DELIVERY/PROPOSED_DATASET.jsonl`；Quest `/projects/p33100/siosio/crashbench_safelibero/verifier_dataset/20261015T042328Z_audit-constraints_75be81f8f066/` |
| 每张图的辅助数据 | 各 assignment 目录下有 `overview_*_segmentation.npz`；单元目录下有 `CAMERA.json`、`LAYOUT.json`（两个盒子、目标、终点的几何）、`VERIFIED.json`（证书与见证核验） |

### 2.2 闸门结果与诊断

固定读数是"两个盒子各自的颜色统计 × 规则符号"上的线性/森林模型，见
[constraint_shortcuts.py](../../verifier_dataset/constraint_shortcuts.py)。

- 结果：验证 AUROC 0.780（线性），源级 bootstrap 95% CI [0.71, 0.86]，跨过了 0.75 这条线。
- 只用验证集预测做的分层如下（锁定测试集未使用）：

| 分层 | 行数 | AUROC |
|---|---:|---:|
| EXIT × checkpoint 0 | 32 | **1.000** |
| EXIT × checkpoint > 0 | 64 | 0.789 |
| ENTRY × checkpoint 0 | 12 | 0.861 |
| ENTRY × checkpoint > 0 | 84 | 0.636 |

**为什么判断它是伪迹，而不是低分辨率的真信号：**

- 这个读数的特征里完全没有目标或终点的位置。
- 在一个四行单元里，任何"图像分 + 规则分"的纯加性打分，单元内 AUROC 恰好是 0.5。
- 所以 0.78 只能来自"受保护的盒子本身长什么样、摆在哪"，而标签本应不随这些量变化。
- 再结合 checkpoint 0（机器人处于初始姿态）时 AUROC 达到 1.0，最可能的原因是初始姿态下盒子摆放规则的规律性。
- 注意 EXIT × checkpoint 0 只有 8 个单元，这是有证据的假设，还不是结论。
- ENTRY × checkpoint > 0 只能称为"**已测捷径较弱的子集**"，不能说成已证明干净：它只是对这一个读数弱。

### 2.3 已知限制（必须写进所有结论）

1. **rationale 很浅**：本质是两个比特（目标、终点是否在受保护盒内）加一个 XNOR。
   这是对"中间监督"的干净测试，不能说成复杂物理推理。
2. **每张图恰好只有一个分隔盒**：所以"先找到分隔盒，再看它是否受保护"这种策略在这个数据里就能拿满分。
3. **感知很难**：盒子是细线框，两个高棱柱在投影中大面积重叠，目标的白色框很小。感知可能是首要瓶颈。
4. **测试集小**：锁定测试集只有 48 个单元、**15 个独立源**；3 个训练 seed 不会增加独立场景数量。
   单元级可检测的最小差异约 15–25 个百分点。"已测捷径较弱的子集"在测试集里只有 16 个单元。
5. **虚拟盒是合同约束，不是物理墙**：结论不能外推到真实碰撞物理或一般可达性。
6. **闸门未通过**：模型可能学到摆放伪迹，所以结果必须分层报告。

### 2.4 已有代码与经验

**训练管线**（[verifier_training/](../../verifier_training/README.md)）已经具备：

- 模型：Qwen2.5-VL-3B-Instruct，revision `66285546d2b821cf421d4f5eb2576359d3770cd3`；语言侧 LoRA，冻结视觉编码器和 projector。
- 目标：verdict 是单个 0/1 token；rationale 和 verdict 的损失各自归一化，各占 0.5。
- 检查与评估：严格的 token 审计；自由生成评估；P(verdict) 打分；规则翻转与换色指标；按源配对的 bootstrap。
- 主要文件：[fd1000_contract.py](../../verifier_training/fd1000_contract.py)、
  [fd1000_run.py](../../verifier_training/fd1000_run.py)、
  [fd1000_metrics.py](../../verifier_training/fd1000_metrics.py)、
  [binary_verifier.py](../../verifier_training/binary_verifier.py)。

**必须先修的代码问题**：`fd1000_run.py` 的 `train()` 在恢复验证集选中的 checkpoint 后，会**立即评估测试集**。
这样每次调试、smoke 都会提前看到测试集结果。Phase 0 要把测试评估拆成独立阶段（0.4）。

**120 行单场景工程 pilot 的教训：**

- 原 B 组只拟合了 73/120。主因是生成配置里的 `repetition_penalty=1.05` 对 "feasible/infeasible"
  两个词不对称。现在已改为 0/1 单 token，而且 V4 合同里 penalty 固定为 1.0。
- 把 verdict 和 rationale 的损失平均在一起，会掩盖判定 token 的难度，所以判定 token 要单独归一化。
  同理，rationale 里少数关键字段的错误也会被大量固定模板 token 掩盖，所以关键字段要单独计分（§4.0）。
- rank64 配原学习率时训练不稳定；降低学习率后稳定了，但准确率没有提升。剩余错误集中在目标完全被遮挡的样本。
- 把模型自己生成的 rationale 换成正确 rationale 后，15 个错误判定被修正。说明 rationale 里的错误会传到判定。
- 这个 pilot 的 Quest 运行目录已于 2026-10-16 删除，摘要仍在 `results/verifier_training/`。

## 3. 其他方法的启示

✓ 表示本次已核对原文表格或正文。其余数字来自文献综述的阅读，正式引用前请再核对。

### 3.1 生成式验证器（GenRM 一族）

| 发现 | 来源 | 对本计划的启示 |
|---|---|---|
| GenRM-CoT 报告的数字是对 32 条采样 rationale 的 P(Yes) 取平均；投票在 2 票内就超过 greedy ✓ | GenRM §4.3 | 多次采样作为预先声明的次要结果；主比较用 greedy |
| MMLU 各子集上，判别式 RM → GenRM-CoT 只提高 0.5–3.5 分；摘要里的大幅提升是相对基座模型 ✓ | GenRM Table 1 | 不要期待大幅提升；比较对象是 A 组，不是零样本基座 |
| 增益最大的是用程序生成 oracle rationale 的算法任务（32 票下约 +8.4，从图读出） | GenRM Fig. 1 | 我们的结构化事实正是程序生成的 oracle rationale，这是最接近的正面先例 |
| 直接 GenRM（只取 P(Yes)、不推理）与判别式 verifier 相当 ✓ | GenRM §4.1 | 用 P(1) 给 A 组打分，就等价于"直接 GenRM" |
| 训练 rationale 由看到参考解的教师生成，推理时移除参考解；再按 verdict 过滤 ✓ | GenRM §3.3、附录 C | 用特权信息制作标签是允许的；我们的事实直接由几何和证书生成并核对，比按 verdict 过滤更干净 |
| 按步过滤的合成验证链，优于只按最终结果过滤 | ThinkPRM | 关键字段逐项判分、单独报告损失 |
| 自洽投票的收益普遍很小（CLoud ≤0.7；DeepSeek-GRM +1.1；Mahan +1.6） | CLoud；DeepSeek-GRM；Mahan | 投票只作为变体，不作主结果；要在相同算力下比较 |
| 多模态 judge 中，直接模式和 CoT 模式差不多（UnifiedReward-Think 73.1 vs 73.8）；在感知类题目上，标量 RM 不输长 CoT（BaseReward） | UnifiedReward-Think；BaseReward | 在偏感知的任务上，B 组不一定赢 |

### 3.2 rationale 作为训练信号 vs 测试时推理

| 发现 | 来源 | 对本计划的启示 |
|---|---|---|
| LIBERO-90：无推理 82.0 / 完整 ECoT 90.8 / 推理只在训练时出现 89.4 / 无意义"思考 token" 78.9–79.8 ✓ | ECoT-Lite Table 1 | 收益主要来自表征学习，所以要设 C 组 |
| 失败检测：只输出答案 0.80 / 推理只在训练时出现 0.81 / 始终先推理 0.83，后者推理慢约 6 倍 ✓ | Guardian Table 4 | 测试时推理只多 1–2 分；要报告推理开销 |
| 多任务（标签和 rationale 作为两个任务分别训练）优于"rationale → 标签"单序列（ANLI 49.58 vs 43.50） | Distilling Step-by-Step | C 组让每条样本同时有两种训练目标 |
| CoT 放在标签之后也有帮助，打乱顺序的 CoT token 仍有帮助 | Wadhwa 2024 | B′（先 verdict 后 rationale）和 filler 对照留作后续 |
| 小模型上 rationale 先行会因 rationale 幻觉而掉点（81.63 → 69.32） | Multimodal-CoT | 要盯住关键事实的准确率 |

### 3.3 感知与空间推理中的 CoT

| 发现 | 来源 | 对本计划的启示 |
|---|---|---|
| 同一个 Qwen2.5-VL-3B 在 SAT-2 上：带坐标的 CoT 62.9 / 同一流程去掉坐标 58.7 / 直接 SFT 58.3 ✓ | ViGoRL Table 4、Table 2 | 不带坐标的 CoT ≈ 不用 CoT，这是 Phase 3 设 G 组的依据 |
| 显式思考在感知任务上普遍掉点（如 RefCOCO 89.1 → 75.1） | Perception-R1 | 先验：B 组在分布内可能赢不了 |
| 分类任务上，不思考的 RL 不差于、常常好于思考的 RL | Think-or-Not-Think | 如果做 RL，也要有不思考的对照 |
| 文本 CoT 约降低 4%；显式的认知地图约提高 10% | Thinking in Space | 空间信息要显式化（坐标、地图），而不是多写文字 |
| CoT 的收益主要集中在数学和逻辑任务 | Sprague 2024 | 我们任务里的符号部分只有一个 XNOR |

### 3.4 中间监督与组合

| 发现 | 来源 | 对本计划的启示 |
|---|---|---|
| 中间监督可以证明地让 parity 类组合变得可学 | Wies 2023 | 结构化事实里的两个比特就是中间监督 |
| 端到端训练在 Pentomino、视觉 parity 上会卡住，加上中间目标后可学 | Gülçehre & Bengio 2016；Shalev-Shwartz 2017 | 小数据下 B/C 组相对 A 组可能有样本效率优势 |
| PGM 的辅助元目标在分布内有提升，外推没有提升 | Barrett 2018 | 外推（更深的组合、新布局）要单独测 |
| 概念瓶颈模型先预测概念、再由概念得到答案，测试时可修正概念 | Koh 2020 | "模型预测事实 + 程序应用规则"读法；oracle 事实诊断 |

### 3.5 机器人失败检测与可行性预测

| 发现 | 来源 | 对本计划的启示 |
|---|---|---|
| 数据普遍来自仿真中的程序化扰动加程序标签；解释大多由 LLM 根据元数据写 | AHA；RoboFAC；Guardian | 我们两类标签都有证明，比这些更强 |
| 专门为失败检测微调的模型，反而不如通用 VLM 和它自己的基座；模型普遍偏向判"成功" ✓ | FailBench | 报告微调前后的对比；false-feasible 单独报告 |
| 跨数据生成器的泛化是主要弱点（RoboFAC 在 RoboFail 上只有 0.25 的执行准确率） | Guardian 的报告 | 优先增加场景家族，而不是给每个场景加更多行 |
| 用 VLA 内部特征做失败检测，在未见任务上 ROC-AUC 78.0 | SAFE | 可作为"策略相关"预测头的参考 |
| 本项目 reach-avoid 研究：基于 π0.5 表征的读数在 held-out 笼子场景上 AUROC 0.500 | [reach_avoid/RESULTS.md](../reach_avoid/RESULTS.md) | 这是做独立 verifier 的动机 |

### 3.6 VLA 推理与测试时验证器

| 发现 | 来源 | 对本计划的启示 |
|---|---|---|
| 动作验证器绑定某个策略的动作分布：在 OpenVLA 上训练的 RoboMonkey 验证器，用在 π0 上反而低于 π0 自身基线 | CoVer-VLA | 不做动作级的 VLM 验证；虚拟盒已知时，动作检查可以直接用解析几何 |
| VLA 基本忽略语言指令 | LIBERO-PRO；LIBERO-Plus | 蒸馏进 π0.5 很难让它真正遵守约束，所以用"门" |
| 往推理里注入内容，对动作的影响有限 | Altered Thoughts（2026） | 同上 |
| π0.5 的高层子任务预测，主要通过训练数据起作用 | π0.5 论文 | 与 §3.2 一致 |

### 3.7 不可行指令与拒绝执行

| 发现 | 来源 | 对本计划的启示 |
|---|---|---|
| 现有的"缺陷指令"工作主要是语言层面的；没有找到"几何不可行、两类标签都有证明"的工作 | RationalVLA；RoboAbstention；综述 | 这是本项目的新意所在 |
| 更多推理反而降低拒绝率 | RoboAbstention | 拒绝阈值靠校准，而不是靠推理 |
| 用保形预测决定何时求助 | KnowNo；SAFE | Phase 4 的阈值用 conformal 校准 |
| 在综述检索范围内，AEGIS/VLSA 论文没有讨论无解场景；CBF-QP 可能产生不期望的平衡点 | VLSA；Reis 2021 | 这就是 Phase 4 要回答的系统问题 |

### 3.8 可行性与不可行性证书

| 发现 | 来源 | 对本计划的启示 |
|---|---|---|
| 学习型可行性预测器常把"求解失败"当负例 | Driess 2020 | 我们的负例是证书，定位更强 |
| 学习不可行性证明：在障碍区学一个分隔闭曲面，再逐面验证 | Li & Dantam 2021 | 我们的封闭分隔盒是它的简化几何版本；证书检查器可作 oracle 上限 |

### 3.9 捷径与评估方法学

| 发现 | 来源 | 对本计划的启示 |
|---|---|---|
| shortcut 按"干预下标签不变"定义，不按"特征简单"定义 | Geirhos 2020；Eisenstein 2022 | 闸门改成检查"标签本应不依赖的读数" |
| 部分输入基线失败不能证明数据干净；简单模型成功时，应报告结果并构造反例子集，而不是拒绝数据 | Feng 2019；Dancette 2021 | 颜色读数作为基线；报告"已测捷径较弱的子集" |
| contrast set 与组一致性指标 | Gardner 2020；NLVR2 | 主指标加入"单元四行全对率" |
| 反复对同一个 holdout 做适配，会让它失效 | Dwork 2015；Teney 2020 | 锁定测试集在所有选择冻结后只评估一次 |
| 小样本比较普遍功效不足 | Card 2020 | 事先写明最小可检测效应 |

### 3.10 忠实性检验

| 方法 | 来源 | 本计划中的用法 |
|---|---|---|
| 截断、插入错误前提、filler、改写 | Lanham 2023 | 只作诊断，不作通过标准；修改前提后从修改位置重新生成后续内容 |
| 有偏的输入特征会改变答案，但 CoT 不提它 | Turpin 2023 | 把换色、规则翻转当作输入干预 |
| 忠实性指标与准确率相关，需要按对错分层 | Bentham 2024 | 忠实性结果按对错分层报告 |

## 4. 实验计划

### 4.0 结构化事实（B、C 组共用）

格式固定如下，每个关键字段都可以自动判分：

```text
Protected: orange
Target inside orange: no
Goal inside orange: yes
Separated: yes
Verdict: 0
```

各行的值（P 表示受保护的盒子）：

| 受保护盒 | 构造 | Target inside P | Goal inside P | Separated | Verdict |
|---|---|---|---|---|---|
| a（分隔盒） | EXIT | yes | no | yes | 0 |
| a（分隔盒） | ENTRY | no | yes | yes | 0 |
| b（另一个盒） | BOTH | yes | yes | no | 1 |
| b（另一个盒） | NEITHER | no | no | no | 1 |

**生成方式**：`几何与规则 → 核验空间关系 → 生成结构化事实 → 与证书或安全轨迹核对`。

- 表中的值由构造 metadata 推出，同时必须与 `LAYOUT.json` 的实际几何（目标点、必要终点区域、两个盒子）
  以及 `VERIFIED.json` 的证书/见证结果一致；任何不一致都让导出失败，不能只凭类别名称填答案。
- P 的颜色取 `appearance_assignment[protected_boundary]`。
- 正例的 `Separated: no` 只表示"没有发现受保护边界阻断"；标签依据是证据记录里的安全轨迹，输出中不提轨迹。

**关键字段**：受保护颜色、`Target inside`、`Goal inside`、`Separated` 四个值，加上 verdict。

- 逐字段统计准确率，并单独报告这些值 token 的损失，避免被固定字段名称的低损失掩盖。
- 训练目标仍是"判定 token 0.5 + 其余 token 0.5"，不另加权重。

**输入**：student 输入保持 V4 原文不变。两张图约 512 个视觉 token，加上文本，远低于 2048 的长度上限。
各组的图片、任务、安全规则完全相同；C 组的两种模式用强制解码前缀切换（§Phase 2），不改输入。

### Phase 0：准备（CPU，约 1 天编码）

| 编号 | 任务 | 产出 |
|---|---|---|
| 0.1 | 开发验收记录：写明用户 2026-10-16 的决定、闸门结果、§2.2 的分层诊断，以及锁定测试集未使用 | `DEV_ACCEPTANCE.json`，放在导出数据集根目录 |
| 0.2 | 导出脚本：`PROPOSED_DATASET.jsonl` → `dataset.jsonl` 加 `dataset/images/`（相对路径加 sha256）；按 §4.0 生成结构化事实并做几何/证书一致性断言；标记 `acceptance_mode=development` | `verifier_training/v4_export.py` |
| 0.3 | 32 行样例包，由导出脚本顺手生成：从训练集选 8 个完整四行单元，覆盖四种构造；展示图片、结构化事实、正负证据和模型可见范围。供用户翻看，不是审核关 | 导出根目录下的 `SAMPLES.html` |
| 0.4 | 训练与测试分离：`train()` 只训练、在 validation 上选模并保存选中的 adapter，不再碰测试集；新增独立的 `evaluate-test` 阶段，只能在所有运行完成、读法和指标冻结后执行一次 | `fd1000_run.py` |
| 0.5 | 合同：给 `fd1000_contract.validate` 加 development 模式。用 `DEV_ACCEPTANCE.json` 代替 `ACCEPTANCE.json`；保留哈希、按源切分、单元完整等全部检查；不再要求 shortcut 闸门通过 | `fd1000_contract.py` |
| 0.6 | 组别：实现 A/B/C 的目标格式、C 的双目标训练与两种强制前缀读法、B 和 C-CoT 的 K 次采样（`num_return_sequences`） | `fd1000_run.py` |
| 0.7 | 指标：关键字段逐项准确率与损失；"模型预测事实 + 程序应用规则"读法；事实与 verdict 一致性；oracle 事实诊断；修改前提并从修改处重新生成的诊断；分层报告；按源聚类的检验（源级符号翻转检验、Durkalski 聚类 McNemar） | `fd1000_metrics.py` |
| 0.8 | 单元测试和 token 审计：两张 448 图加文本不超过 2048，不发生截断；各组的图片、任务、规则 token 完全相同；`train()` 不读测试集 | `verifier_training/tests/` |
| 0.9 | 冻结预注册：把 §5 写成 JSON，记录 commit | `verifier_training/configs/v4_dev_prereg.json` |

### Phase 1：零样本诊断与小规模试跑（约 2–3 A100·小时，只用 validation）

- **1.1 感知探针**：96 张验证图，每张 4 个比特（目标、终点分别在橙色、紫色盒内吗）。
  问 "Is the white-outlined object inside the orange box? Answer yes or no."，
  用 P(yes)/(P(yes)+P(no)) 打分。真值由几何核对过的结构化事实给出。
- **1.2 零样本 verdict**（Qwen2.5-VL-3B 基座），三种问法：
  - 直接给 P(1)。
  - 先生成结构化事实，再解析。
  - oracle 事实：在文本里直接给出正确的两个事实，只考规则应用。
- **1.3 小规模 A/B 工程试跑**：A、B 各 1 个 seed、1 个 epoch，只在 validation 上评估。
  检查：能否学会输出格式、关键字段是否开始拟合、损失与梯度是否正常、每次运行的实际耗时和显存。
  这一步用来定 Phase 2 的资源请求，不作为科学结果。
- **1.4 可选**：
  - Qwen3-VL-32B：共享 HF cache 里有完整权重，但需要比现在的 4.53.2 更新的 transformers，得另建一个 pinned overlay。
  - API 前沿模型基线：需要用户单独授权图片外发，预算不超过 $3，默认用 pinned 的 GLM-4.5V，换模型须用户指定。

解读方式（Phase 2 无论如何都会运行）：

| 比特准确率 | oracle 事实准确率 | 含义 |
|---|---|---|
| 低 | 高 | 感知瓶颈：Phase 3 优先做 grounded CoT 和视觉侧改进 |
| 高 | 低 | 规则应用瓶颈：结构化事实应该有帮助 |
| 高 | 高，但直接 verdict 低 | 组合或对齐问题：中间监督有机会 |

产出：`PHASE1_REPORT.md`，放在该次运行根目录，摘要抄回本目录。

### Phase 2：主对照（约 10–14 A100·小时，最多 2 张卡）

| 组 | 训练目标 | 测试时读法 | 回答的问题 |
|---|---|---|---|
| A | `Verdict: 0/1` | 直接读取 P(1) 和 greedy verdict | 直接判别能做到什么程度 |
| B | 结构化事实 → verdict | 自由生成事实再判断：greedy（主结果）；K=1/2/4/8/16 条采样（温度 0.7）的平均 P(1)（次要） | 完整的生成式方法是否更好 |
| C | 每条样本同时有两种目标：直接 verdict，以及结构化事实 → verdict | 同一个选定 checkpoint 读两次：强制 `Verdict:`（C-直接），或强制 `Protected:`（C-CoT） | 测试时显式生成事实是否额外有用 |

B′（先 verdict 后 rationale）、filler 对照、更大的模型，按验证集发现在 Phase 3 追加。

**公平性固定项**：

- 同一份数据、同一个初始模型、同一个 seed 下相同的样本顺序、相同的优化器更新次数和选模规则。
- 配置沿用 `fd1000_contract.configuration()`：LoRA rank 16、alpha 32、dropout 0.05；LR 1e-4，AdamW，
  weight decay 0，梯度裁剪 1；microbatch 1、有效 batch 4；max_length 2048；图像 448²；
  `repetition_penalty=1.0`；冻结视觉和 projector；每次运行从全新的基座加全新的 LoRA 开始。
- seed 取 713、714、715。
- 更新次数：所有组都是 384 次更新（A/B 相当于 3 个 epoch）。C 每条样本有两个目标，同样 384 次更新
  相当于把 1,024 个"样本 × 目标"各看 1.5 次；两个目标的损失等权。
- 所有组在第 128、256、384 次更新时做 validation 评估，从这三个 checkpoint 里选一个。
- 选模规则：A 和 B 按各自 validation verdict 准确率；C 按两种读法 validation 准确率的平均值，
  两种读法必须用这同一个 checkpoint，不能各自挑最好的。平局取更早的。
- 报告每组的训练时间、推理时 token 数和耗时。

**测试集**：Phase 2 全部 9 次运行结束、读法与采样次数和指标冻结后，运行 `evaluate-test` 一次，统一评估所有选定 checkpoint。

**结果解读**（只说明行为差异，不能据此认定输出文字反映了模型内部推理）：

| 结果 | 解释 | 下一步 |
|---|---|---|
| B > A | "加入事实监督并生成事实"这整套方法有收益 | 看 C 区分来源 |
| C-CoT > C-直接 | 测试时显式生成事实有额外帮助 | Phase 3：grounded CoT，加上用可验证奖励做 RL |
| C-直接 ≈ C-CoT，且都 > A | 收益主要来自训练监督 | 部署用直接读法（推理便宜）；Phase 3 重点扩数据 |
| "模型事实 + 程序规则" ≥ B 的 verdict | 长解释没有额外收益；两步事实预测已足够 | 优先改进感知 |
| 关键事实准确率低 | 感知瓶颈 | grounded CoT、训练 projector、裁剪图像或提高分辨率 |
| 各组都接近且很高 | 任务太浅 | 增加组合深度 |
| 各组都接近随机 | 数据或感知都不够 | 回看 Phase 1；考虑更大的 backbone |

### Phase 3：按结果扩展（每项开始前单独确认）

- **3a grounded CoT（G 组）**：
  - 在事实里加入每个视角下目标、终点的 2D 点（取分割掩码的质心），以及受保护盒在桌面高度的投影角点（由 `LAYOUT.json` 和 `CAMERA.json` 算出）。
  - 按像素容差自动核验。
  - 对应 ViGoRL、GRIT、ECoT 的做法。
- **3b 可验证奖励的 RL**：
  - 从 B 或 G 组的 SFT checkpoint 出发做 GRPO，奖励 = 格式 + 关键事实 + verdict。
  - 要有不思考的 RL 对照（参照 Think-or-Not-Think）。
  - 注意防止输出塌缩成常数。
- **3c 数据**：
  - 加入"两个盒子都分隔"和"都不分隔"的单元，标签按受保护盒判定，破坏"只找唯一分隔盒"的策略。
  - 从 3,200 个官方 episode 中补充更多源轨迹。一条安全轨迹可以为所有它不违反的规则作见证（hindsight relabeling）。
  - 做按 family 留出的切分。
- **3d 组合深度**：两个受保护盒，或嵌套盒；画出 A 组和 B 组准确率随深度的变化。
- **3e 追加的对照**：B′（先 verdict 后 rationale）、filler 对照；如果确认是感知瓶颈，试 Qwen2.5-VL-7B 或 Qwen3-VL-4B/8B。

### Phase 4：接到 AEGIS 的系统实验（需要用户单独授权，涉及控制器改动）

- **门的定义**：执行前由 verifier 判断；P(1) < τ 时拒绝执行或请求人类帮助。
- **评估**：
  - 在 held-out 源上，用可行与不可行混合的场景，画 risk–coverage 曲线。
  - 报告固定误拒率下避免的违规执行数，以及"不可行被判可行"的比例和置信区间。
  - τ 在 validation 源上用 conformal 校准。
  - 在未修改的 SafeLIBERO 上做"不帮倒忙"检查：那里的场景都可解，只会出现误拒。
- **前提条件**：AEGIS 必须把受保护盒当作 CBF 约束，否则虚拟盒对策略毫无意义。
- **可选扩展**：加第二个"AEGIS 能否安全成功"的策略相关预测头（SAFE 式）。这是新的学习组件，需要另外授权。

## 5. 评估与统计（Phase 2 提交前冻结为预注册）

**主比较**：锁定测试集上 B-greedy 与 A 的行准确率差（配对，按源聚类），以及单元四行全对率。

**预先声明的次要比较**：

- C-CoT vs C-直接；C-直接 vs A。
- B 的 K 次采样读法 vs B-greedy。
- "模型预测事实 + 程序应用规则"读法 vs B 的 verdict。
- P(1) 的 AUROC。
- 按"构造格子 × checkpoint"分层，以及"已测捷径较弱的子集"（ENTRY × checkpoint > 0）。
- 规则翻转一致性、换色不变性。
- false-feasible 率（危险方向的错误）。

**B 与 C-CoT 的诊断**（只用于定位问题，不作通过标准）：

- 每个关键字段的准确率。
- 三类情况的计数：事实错（感知问题）；事实对但 verdict 错（规则应用问题）；verdict 对但事实错（警惕捷径或事后解释）。
- oracle 事实诊断：强制给出正确事实后，verdict 错误能减少多少。
- 修改一个事实并从修改处重新生成后续内容，看 verdict 如何变化；修改后的文字可能与图片冲突，
  所以"坚持图片"或"跟随文字"都不能单独判为对错。
- 按对错分层报告。

**基线**：

- 多数类（50%）。
- 颜色 + 规则读数：已固定，在 `evaluate-test` 阶段一并打分。
- 零样本 Qwen2.5-VL-3B：直接问法和事实问法。
- oracle 事实（上限 100%）。

**统计方法**：

- 分析单位是四行单元，聚类单位是源；15 个测试源就是全部的独立场景数，seed 不增加它。
- 每行的正确率先在 3 个 seed 上取平均，再算组间配对差。
- 检验：15 个测试源上的源级符号翻转检验，或 t₁₄ 检验；Durkalski 聚类 McNemar。
- 置信区间：1,000 次源级 bootstrap，seed 1014。
- 单元级最小可检测效应约 15–25 个百分点；小于这个的差异一律只算探索性结果。

**成功标准**：

- **开发里程碑**：任一组同时满足测试集行准确率 ≥ 0.85、单元全对率 ≥ 0.70、规则翻转一致性 ≥ 0.90。
  这只说明值得继续，不足以作为安全执行门的依据；接到机器人之前还要看 false-feasible 率、
  置信区间和 risk–coverage（Phase 4）。
- **科学**：B-greedy 减 A 的源聚类 95% CI 不包含 0。

## 6. 资源与预算

- **GPU**：p33100/gengpu，A100，最多同时 2 张。
  - Phase 1：约 2–3 小时（含小规模试跑）。
  - Phase 2：约 10–14 小时（A/B/C 各 3 个 seed 共 9 次运行），以 Phase 1.3 实测耗时为准。
  - Phase 3：按选定的项目另行估计。
- **CPU**：p33100/short，用于导出、审计和报告。
- **API**：默认零调用。
  - Phase 1 的可选前沿模型基线不超过 $3，需要授权。
  - 必须先做 spend guard 和账本；key 不进入日志或 Git。
  - 用户方法的每周 OpenRouter 预算是 $95。
- **存储**：新的运行根目录放在
  `/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs/<UTC>_<stage>_<commit>/`。

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 模型学到摆放伪迹 | 分层报告，看"已测捷径较弱的子集"；Phase 3c 加入"都分隔 / 都不分隔"的单元 |
| 调试时提前看到测试集 | 0.4 拆分训练与测试评估，并有测试保证 `train()` 不读测试集 |
| B 组欠拟合，重演 pilot 的问题 | 0/1 单 token、损失分开归一化、penalty 1.0；Phase 1.3 先试跑 |
| 关键事实没学会，却被模板 token 的低损失掩盖 | 关键字段单独计分、单独报告损失 |
| 感知瓶颈导致各组都接近随机 | Phase 1 先测；必要时走 Phase 3a / 3e |
| 测试集太小 | 事先写明最小可检测效应；必要时在 train+val 上补做按源分组的交叉验证 |
| 学到数据生成器专属的捷径（FailBench 现象） | 报告与零样本基座的对比；Phase 3c 引入新的场景家族 |

## 8. 待用户决定

1. ~~是否开始 Phase 0~~：用户 2026-10-17 要求开始，Phase 0 与 Phase 1 已完成。
2. ~~Phase 1 的 API 基线~~：用户指定 "gpt luna 6"，即 OpenRouter 上的 `openai/gpt-6-luna`。只用 validation，实际花费 $0.4669（上限 $3）。
3. ~~是否开始 Phase 2~~：用户 2026-10-17 说"开始吧"；Phase 2 已完成，锁定测试集已评估一次，见 [PHASE2_REPORT.md](PHASE2_REPORT.md)。
4. Phase 3 做哪些项（3a grounded CoT、3c 数据、3e 对照、A 的训练预算对照），每项开始前单独确认。
5. Phase 4 的 AEGIS 控制器改动，以后再决定。

## 9. 参考文献

以下 arXiv 编号来自本次文献综述。✓ 表示其中引用的数字已核对原文。

- 生成式验证器：GenRM, Zhang 2024, arXiv:2408.15240 ✓；CLoud, Ankner 2024, arXiv:2408.11791；
  Generative Reward Models, Mahan 2024, arXiv:2410.12832；ThinkPRM, Khalifa 2025, arXiv:2504.16828；
  GenPRM, Zhao 2025, arXiv:2504.00891；RM-R1, Chen 2025, arXiv:2505.02387；
  J1, Whitehouse 2025, arXiv:2505.10320；DeepSeek-GRM, Liu 2025, arXiv:2504.02495；
  Heimdall, Shi 2025, arXiv:2504.10337。
- 多模态 judge：VL-RewardBench, Li 2024, arXiv:2411.17451；VisualPRM, Wang 2025, arXiv:2503.10291；
  LLaVA-Critic-R1, Wang 2025, arXiv:2509.00676；UnifiedReward-Think, Wang 2025, arXiv:2505.03318；
  R1-Reward, Zhang 2025, arXiv:2505.02835；BaseReward, Zhang 2025, arXiv:2509.16127。
- CoT 何时有用：To CoT or not to CoT, Sprague 2024, arXiv:2409.12183；Perception-R1, Yu 2025,
  arXiv:2504.07954；Think or Not Think, Li 2025, arXiv:2503.16188；Thinking in Space, Yang 2024,
  arXiv:2412.14171；Li 2024, arXiv:2402.12875；Prystawski 2023, arXiv:2304.03843。
- rationale 作为训练信号：Distilling Step-by-Step, Hsieh 2023, arXiv:2305.02301；STaR, Zelikman 2022,
  arXiv:2203.14465；Hase & Bansal, arXiv:2102.02201；Wadhwa 2024, arXiv:2406.14511；
  Multimodal-CoT, Zhang 2023, arXiv:2302.00923。
- 中间监督：Wies 2023, arXiv:2204.02892；Gülçehre & Bengio 2016, arXiv:1301.4083；
  Shalev-Shwartz 2017, arXiv:1703.07950；Barrett 2018, arXiv:1807.04225；
  Concept Bottleneck Models, Koh 2020, arXiv:2007.04612。
- 机器人失败检测：SuccessVQA, Du 2023, arXiv:2303.07280；AHA, Duan 2024, arXiv:2410.00371；
  RoboFAC, Ye 2025, arXiv:2505.12224；Guardian, Pacaud 2025, arXiv:2512.01946 ✓；
  SAFE, Gu 2025, arXiv:2506.09937；FIPER, Römer 2025, arXiv:2510.09459；
  FailSafe, Lin 2025, arXiv:2510.01642；FailBench, Navasardyan 2026, arXiv:2609.03611 ✓；
  Code-as-Monitor, Zhou 2024, arXiv:2412.04455。
- VLA 推理与验证器：ECoT, Zawalski 2024, arXiv:2407.08693；ECoT-Lite, Chen 2025, arXiv:2505.08243 ✓；
  π0.5, Black 2025, arXiv:2504.16054；Knowledge Insulation, Driess 2025, arXiv:2505.23705；
  RoboMonkey, Kwok 2025, arXiv:2506.17811；CoVer-VLA, Kwok 2026, arXiv:2602.12281；
  V-GPS, Nakamoto 2024, arXiv:2410.13816；FOREWARN, Wu 2025, arXiv:2502.01828；
  Altered Thoughts, Trinh 2026, arXiv:2603.12717。
- 空间推理与 grounding：ViGoRL, Sarch 2025, arXiv:2505.23678 ✓；GRIT, Fan 2025, arXiv:2505.15879；
  RoboSpatial, Song 2024, arXiv:2411.16537；Cosmos-Reason1, NVIDIA 2025, arXiv:2503.15558。
- 不可行指令与安全：RationalVLA, Song 2025, arXiv:2506.10826；KnowNo, Ren 2023, arXiv:2307.01928；
  RoboAbstention, Yeke 2026, arXiv:2605.20544；LIBERO-PRO, Zhou 2025, arXiv:2510.03827；
  LIBERO-Plus, Fei 2025, arXiv:2510.13626；VLSA/AEGIS, Hu 2025, arXiv:2512.11891；
  CBF-QP 的不期望平衡点, Reis 2021, arXiv:2003.07819。
- 可行性与证书：Deep Visual Reasoning, Driess 2020, arXiv:2006.05398；Li & Dantam,
  "Learning Proofs of Motion Planning Infeasibility", RSS 2021。
- 评估方法学：Geirhos 2020, arXiv:2004.07780；Eisenstein 2022, arXiv:2204.04487；Feng 2019,
  arXiv:1905.05778；Gardner 2020, arXiv:2004.02709；Dancette 2021, arXiv:2104.03149；
  Teney 2020, arXiv:2005.09241；Card 2020, arXiv:2010.06595；
  Dwork 2015, doi:10.1126/science.aaa9375；Durkalski 2003, doi:10.1002/sim.1438。
- 忠实性：Turpin 2023, arXiv:2305.04388；Lanham 2023, arXiv:2307.13702；Chen 2025,
  arXiv:2505.05410；Bentham 2024, arXiv:2402.14897。
