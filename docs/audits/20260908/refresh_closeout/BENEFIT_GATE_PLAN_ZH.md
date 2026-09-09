# 固定恢复选项下的 benefit-gate 开发方案

状态：**设计完成，未执行**。本次授权是收束 Refresh 诊断、制定下一份方案；不代表已授权新训练、Quest 作业、rollout、确认测试或新增来源。旧冻结结果保持不变。执行前应将下面的数据核验结果和具体配置写入独立运行清单；本文件不是自动实验队列。

## 1. 目标与已有依据

目标：给定执行前可得信息，判断是否值得切换到一个已固定 choice 机制选出的恢复选项，在保留任务完成和正常成功的同时，超过强 Risk→BestFixed 基线。

依据为历史 [ADVANTAGE_ROUTER_RESOLUTION](../../../iclr27/ADVANTAGE_ROUTER_RESOLUTION.md) 的 tiny-choice 分解：

| 组合 | source-macro utility | 距完全 Oracle |
|---|---:|---:|
| OracleGate + OracleChoice | 0.5642 | 0 |
| OracleGate + LearnedChoice | 0.4917 | 0.0725 |
| LearnedGate + OracleChoice | 0.2714 | 0.2928 |
| LearnedGate + LearnedChoice | 0.2311 | 0.3331 |

gate 的缺口较大，提供了下一步优先级；这些项不能当可独立相加的因果贡献，组合还有约 −0.0322 的交互项。该表来自 **20 来源、273 决策的整体 exposed-development capture**，并非 glass-only 分析。51/60 个严格 D/R 状态来自 glass，控制条件和非 glass 状态仍须保留。历史 machine decision `CHOICE_CAPACITY_BOTTLENECK` 源于其有序判定规则，保留原名，不改写为新胜利或旧实验的事前许可。

## 2. 首先固定选项、choice 和测量对象

沿用 capture `d4751330395e`，选项顺序为 Base、`detour_complete`、`retreat_hold`。不改控制器、动作时长、事故谓词或允许输入。Retreat 的 safe stop 不能自动叫完成型恢复；每个被选恢复分支都必须另报 task success。

choice 固定为既有 tiny hidden-only 配方（32 tanh 单隐层，L-BFGS-B，L2=0.01，最多 1000 次，seed=2027）；不再搜索 choice 架构、宽度或种子。固定的是训练配方和来源边界，不是用全数据拟合一个 choice 后跨折复用。每个外层折的 choice 只在该折 14 个 fitting sources 上拟合；预处理同侧。14 来源内产生训练标签时进一步采用来源留一 choice 预测，防止 choice 在自己拟合过的标签上显得过好。有限端点不收敛沿用历史记录规范，单独报告，禁止通过改配方追结果。

主问题应对应**实际将执行的选项**。设固定配方选择为 c(x)，主目标为

`ΔU_c(x) = U(c(x)) − U(Base)`。

分别保存 Δsuccess、Δcatastrophe，以及可验证的执行成本。历史 `max(U_D,U_R)−U_B` 只作 oracle-option 可恢复性诊断和旧 gate 参照；不能把它等同于实际 c(x) 的可兑现收益。该区别可避免“值得采取某个选项”被误当成“当前 choice 选到的选项值得采取”。

## 3. 数据核验先于拟合

以历史 resolver 的 `config_resolved.yaml`、manifest 和 fold_assignments 为入口，核对以下文件存在且哈希、来源定义一致：capture_manifest、decision_metadata、option_rollouts、decision_features，以及固定选项实现和 controller 版本。

建立 source×option×condition 的支持表，含任务成功、事故、正常成功保持、正负收益与已有重复数。原配置明确每个 decision-option 只有一个冻结结局；因此现阶段不能假设 glass 已有同 bundle 多重复，也不能套用 Refresh 的 A/B 样本设计。

核查已交付 hidden/state/action 的生成时点、策略成本是否已有记录、nominal action 是否本来就已产生。来源 ID、condition、horizon、placement、未来结局和 oracle 标签只能作分组或监督，不能作部署输入。缺失 action 不额外调用策略补齐。不可验证的特征先标记不可用，不用未来字段替代。

对每个旧决策说明：完整 bundle 是否可用、终止是否事故/成功/时域、时长是干预时长还是完整任务完成时长、是否存在右截断。只能从确实保存的轨迹导出较短截止结果；不能把某个 anchor horizon 当作 rollout 完成时域，也不能伪造延长后的结果。

若缺乏重复或长时域，照实标注“未知”，第一阶段仅做单次结局上的开发诊断。未来补测需要另定来源、同 bundle 重复、原时域/长时域和新增预算；本方案不自动授权。

## 4. 一次固定的 gate 对照设计

第一轮只比较一个 gate 学习配方的两个监督对象，不做架构搜索：

- 历史目标参照：Ridge 预测 max-option advantage。
- 主候选：同一 Ridge 预测固定 choice 的可兑现 ΔU_c。

Ridge α=1，使用已验证的 future-free full features；hidden PCA 16 维、scaler 均在对应 fitting sources 内拟合。源等权，不以图像帧数增加来源权重。utility 沿用历史 success=+1、safe noncompletion=0、catastrophe=−1，λ=1、η=0；不靠事后成本权重取得优势。另报成功/事故差，utility 增加并不等价于完成型恢复。

外层沿用20来源 LOSO；其余19来源使用历史确定性14 fitting/5 calibration划分（seed=2027）。同来源的全部状态、条件、时域和已有重复整体归侧。内层 choice 产生主候选训练目标时，不得看到所属来源；每次内层 PCA 也只拟合内层训练来源。标记内层13来源 choice 与最终14来源 choice 的分布差异，不冒充完全一致。

五个 calibration sources 只供 gate 阈值校准，最大化来源平均实际组合 utility，干预率 cap 0.60；平局依次更少事故、更少干预、更高阈值。明确允许永远 Base 的候选阈值。外层 held-out 标签不参与特征、choice、gate或阈值选择。两个监督对象同用这一校准过程，完整报告，不选好看的版本替代主候选。

如某折训练没有正/负收益或风险变化，记录为支持缺口，不把零输出当风险认证。第一轮不临时加事后“来源识别保护”来挽救成绩；未知支持的保守处理如需作为方法，须单独固定可部署规则与评价范围。

## 5. 必须同时报告的比较器

- Base、AlwaysDetour、AlwaysRetreat。
- 强 Risk→BestFixed：固定选项的选择与风险阈值仅使用对应训练/校准来源，沿用历史边界，不能用 held-out 找 best fixed。
- Risk gate + 相同固定 learned choice：分离 gate 改进与 choice 差异。
- 同输入、同来源边界的 DirectQ 与历史完整路由结果作为参照；若旧预测的数据边界不同，不混在同一严格对比表。
- OracleGate/OracleChoice hybrids 只作为诊断上界。

所有比较器用同一选项池、终局语义和完整任务计分。除历史 cap 下的主要评价外，可在 calibration 中匹配 Risk→BestFixed 的干预预算，作为预先声明的次要成本对照；匹配不得依赖外层结局，也不承诺外层实际调用率完全相同。

## 6. 评价顺序与交付

先报每来源的 task success、catastrophe、safe noncompletion、相对 Base 的原成功损失和新增事故，再报 utility、调用率和可测成本。正常控制保持必须单列；净事故差不能替代新增事故。glass/非 glass/控制分组为诊断，不能只挑恢复子集做总体分母。

gate 指标同时面向可兑现 benefit 和 oracle-option benefit，给出排序、符号误差、预测幅度误差及支持缺失来源。固定 choice 的条件正确率保持可追踪。若已有重复，按来源和 bundle 保留层次，汇报均值波动；无重复不能给“稳定期望收益”的标签。

沿用来源平均、共享来源重采样用于描述性不确定性，并说明外层训练集重叠；不把20折或273状态当全新的独立确认。报告逐来源差和删除来源的敏感性；不得以删掉困难来源后的均值充当主结果。

交付应包含：数据支持和可重放性表、输入/选项/choice/fold 哈希、唯一固定配置、完整 OOF 选择和结局、正常任务保持、gate/choice 分解、预算与开销、中文报告。执行规模、确切文件哈希与运行入口在数据核验后填写；无合格数据时报告具体缺口，而非追加隐式采集。

## 7. 后续决策边界

- 若更准确的 benefit 目标在保持正常成功的同时带来额外价值，才值得讨论独立的重复/时域验证计划。
- 若支持不足、标签波动未知或 choice 仍限制收益，先记录具体限制，不通过架构 sweep 掩盖。
- 若只减少事故但不能完成任务，称为避险收益，不能称任务恢复。
- 在线触发是后续独立问题；给定状态 gate 的开发结果不能替代在线时机验证。

该方向承接原始恢复目标，但历史数据已暴露，所有新分析都仍是开发。阶段性收束 Refresh 与继续研究恢复干预并不矛盾。
