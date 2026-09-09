**CrashBench 全历史进度审查｜截至 2026-09-07，提交 `0b3eb75`**

我的判断是：你已经完成了一条从碰撞诊断、结构化避险、学习选项，到干预收益测量的研究链。现有资产足以支撑有内容的研究稿和很具体的方法开发；尚未建立的是一个能在新来源、真实在线时机上，稳定超过强简单基线并保留任务完成能力的干预方法。

最关键的当前问题是：**我们正在预测的干预收益，是否在重复执行中稳定存在，是否能由部署时可获得的信息预测，以及是否必须通过状态选择才能获得。**这三个问题需要分别回答。历史 bug 已有明确修复，最新实验的限制也不能继续全部归到旧 bug 上。

本次审查覆盖本地保存的四个 Git 引用、339 个可达提交的历史、主要实验与论文记录、当前模型/校准/采集实现、最新 A/B 原始终局表，并重跑本地测试。没有 SSH 到 Quest、刷新远端引用、训练模型、运行仿真、开启确认测试或读取 D8 原始结果。Quest 作业状态采用已保存的溯源记录；分支状态限定于本地引用。下面对研究动机的重建依据记录，不把某个历史决策擅自归因于你个人或某一次助手建议。

**1. 先解决分支问题：四个引用是一条历史链**

| Git 引用 | 本地提交与日期 | 新增提交数，相对上一行 | 该阶段保留的研究资产 |
|---|---|---:|---|
| `main` | `a0fae91`，8/12 | — | 原始 wall/glass 诊断、probe、safe-abort、glass oracle、早期恢复学习与失败记录 |
| `origin/codex/iclr27-exact-state-intervention-routing` | `cdc4802`，8/29 | 72 | E16 matched-state Router、P2/P3 在线时机实验、强基线审计、source cross-fit、风险/收益/选项误差分解 |
| `origin/codex/odur-negative` | `c337836`，8/31 | 95 | glass 论文包、新分支引擎与 π0 机制实验、D3/D4 来源冻结、D5、ODUR、D8/D10 历史 release |
| `codex/odur-repair`，当前 HEAD | `0b3eb75`，9/7 | 16 | 第一轮修复、第二轮收益学习、观测重放、论文审查、384 分支选择复验 |

本次 `git merge-base --is-ancestor` 对每一对相邻引用均通过。当前分支比 main 多 183 个提交，包含另外三个 tip 的全部历史。无需先做跨分支合并才能继续研究。当前仅有一个 worktree，检查开始时工作区干净，HEAD 与本地 `origin/codex/odur-repair` 一致。

9/6 报告里的 `codex/audit-repair-round1`、`codex/refresh-gain-round2`、`codex/paper-direction-audit` 是执行当时的名称；9/7 已统一到 `codex/odur-repair`。这些旧名称不是三份待收拢的代码。本次比较也确认：从 `c337836` 到当前 HEAD，Git 跟踪的 `results/` 没有改动，修复报告以追加文档保存。

因此，main README 只能解释到 8/12。即便当前 README，也仍以 9/6 的 `PAPER_DIRECTION_AUDIT` 为顶层状态；最新状态以 [CURRENT.md](/Users/hanshuo/Desktop/crash_bench/docs/CURRENT.md) 和 [9/7 结果](/Users/hanshuo/Desktop/crash_bench/docs/audits/20260907/selection_retest/RESULTS_ZH.md) 为准。

**2. 6 月：你先把“机器人可能有危险”变成了可复现的问题**

起点是一个实质性疑问：VLA 在正常示范附近能完成任务，但进入示范中很少出现的临近失败状态时，能否识别后果并恢复？最初还担心，给 VLM 输入手工 clearance、安全标签和速度等抽象量，会把研究变成读取预先算好的安全答案。见 [最初动机](/Users/hanshuo/Desktop/crash_bench/docs/archive/motivation.md)。

6/17 建仓；6/20 建立 OpenVLA–LIBERO 桥接；6/21 先试桌边碗，发现机器人不按预想触发危险，再转向在原动作路径上放墙。重要进展是把危险构造成可检验的行为差异。

- 同类可见墙，进入动作走廊时 15/15 撞击，明确移出走廊时 0/33。独立单位是 5 个 on-path 和 11 个 clear wall 几何，K=3 是几何内重复。
- OpenVLA、OpenVLA-OFT、π0 都在同一任务/机器人/墙族上表现出 on-path/clear 的差别。它增加了策略实现的覆盖，不能解释成跨任务或跨机器人普遍规律。
- hidden-state probe 在 T−5 上达到 OpenVLA AUC 0.998、OFT 0.903；行为侧却没有持续的、朝离墙方向的制动。
- probe 接到结构化 `RetreatHold` 后，既有 on-path 的 15/15 事故变为 0/15，平均峰值墙力约 321.7 N 变为 0 N；22 条 benign rollouts 没有触发。
- 直接沿 probe 方向做 activation steering 的已测配置仍然撞击，说明可读取的风险方向不自动等于能控制动作的方向。

这一阶段得到的是“特定失败机制可定位、风险可读出、外部避险可生效”。它没有证明模型具有人类意义的风险理解，也没有证明原任务恢复成功。名义 400/500 成功记录缺少对应 manifest-backed 结果，不应作为已经完整复现的论文主结果。严格边界见 [ICLR claim ledger](/Users/hanshuo/Desktop/crash_bench/docs/iclr27/CLAIM_LEDGER.md)。

**3. 7 月至 8/12：你开始追求完成原任务，困难转移到 controller 与数据产率**

墙可以展示避险，但停止并不完成任务。高墙绕行还有整臂碰撞；降低后的 d62 墙出现成功绕行 witness，但几何已经改变，不能拿低墙成功替代高墙结果。glass 更适合表达“避开危险以后继续抓取和放置”。

7/30 的 oracle-stop LoRA 说明行为可以被少量监督改变：两个 held-out wall 场景的 6 条运行，Base 6/6 撞击，微调后 1/6 撞击、5/6 safe abort。它主要学会了停；control 的事故由 2/6 变为 3/6，没有显示正常任务保护。见 [当时的机器汇总](/Users/hanshuo/Desktop/crash_bench/results/oracle_recovery/report_assets/analysis_summary.json)。

8 月的 glass recovery 同时承担了三个难题：造出干净而可恢复的事故锚点、找出确实能完成任务的恢复 controller、学习触发与恢复动作。实际数据产率很低：

| 环节 | 已完成结果 | 应当怎样解释 |
|---|---|---|
| E14 acceptance smoke | 109 次 rollout attempts 接受 3 个 placements | 证明这种可恢复状态存在，产率较低 |
| Broad Pilot B | 20 个场景 × 6 个 H，120 cells；没有共同 H 通过原 gate | 未建立原计划规模的固定-H population |
| 修复后的 fresh H20 筛选 | 15 候选，12 次 Base 事故，3 个 accepted pairs | 总体产率 20%，不是 15 个都可恢复 |
| 可用于该 scoped 学习线的来源 | 1 个 train pair，2 个 development pairs | 帧数多，独立物理状态极少 |
| Pilot C oracle-timed oracle recovery | 2 个 development source × 3 seeds，6/6 成功、0/6 事故 | oracle 恢复可行，独立 n 仍为 2 |

这里确实消耗了很多工程工作：controller/runtime snapshot、gripper 标定、抓取姿态、玻璃与目标物的间距、终局判断、时间预算。两个 oracle 轨迹分别要 232 和 300 actions；使用 220-action 预算时会超时，改回原协议的 360 才得到完整终局。

学习恢复动作本身并没有完成独立在线验证。旧 learned gate 在保存的六条 episode 上没有 crossing，动作头的 gripper 指标也低于当时门槛。这个结论是“该版本未闭环验证成功”；不能拿未执行的 Pilot E/F 证明所有学习恢复路线无效。另一方面，oracle task failure 是真实 controller 问题，放宽文书 gate 无法自动解决它。见 [glass 进度与筛选账本](/Users/hanshuo/Desktop/crash_bench/docs/archive/glass_recovery_20260812/GLASS_RECOVERY_PROGRESS_REPORT_20260812.md)。

**4. 8/13–17：E16 的进展来自把动作生成与选项选择拆开**

你固定 Base、结构化 Detour、结构化 Retreat，只让模型估计选项结局并作选择。同一决策状态执行各选项，监督 `P(success)、P(catastrophe)、P(noncompletion)`，再按偏好计算效用。这让“现在该选什么”第一次成为可直接训练的问题。

独立 fresh random-reset cohort 有 8 个 source，每源 3 条 matched conditions，24 个相关决策。展示点如下：

| 方法 | 任务成功 | 事故 | 干预 |
|---|---:|---:|---:|
| Base | 66.67% | 33.33% | 0% |
| Risk→Retreat | 41.67% | 8.33% | 50.00% |
| AlwaysDetour | 70.83% | 4.17% | 100% |
| Outcome Router | 87.50% | 8.33% | 58.33% |

这是值得保留的正面记录：在给定 matched state 时，选择合适的结构化干预可以同时考虑成功、事故和干预负担。但是它的范围有四个关键限定：

1. 独立样本量是 8，不能把 24 conditions 当 24 个独立场景。
2. 展示点属于预声明 frontier 中的点，并不是在所有结果之前就确定的唯一部署点；合并到 n=13 后没有单一点同时满足全部标准。
3. Detour/Retreat 是结构化、包含特权信息的选项，模型学的是选择，没有学会完整恢复动作。
4. Risk→Retreat 固定用一个容易损失任务的 fallback。后来的强基线审计表明，这个比较不足以建立一般的方法优越性。

所以 87.5% 既不应被抹掉，也不能继续单独承担今天的论文 headline。见 [E16 原始主结果及范围](/Users/hanshuo/Desktop/crash_bench/docs/COUNTERFACTUAL_ROUTER_MAIN_RESULT.md)。

**5. 8/18–19：从“给你一个时刻”走向“自己找时刻”，在线分布暴露了问题**

P2/P3 回答的是更难的任务：从 reset 开始运行，监测连续轨迹，并在恢复窗口关闭前触发。

- P2 在 4 个稳定 development sources 上有小幅改善，成功 58.3%→66.7%、事故 33.3%→25.0%，但两个已知 T−20 可恢复 episode 都错过。
- P2.5 检查了最大值、移动平均、累计面积、连续越界长度、趋势与选项稳定性。错过的恢复轨迹与 controls 的排序并没有通过简单聚合修好。
- P3.0 新增 80 个 dense exact-state anchors，并记录 recovery-open、loss-control、Base-preferred 和 hard controls。
- P3.1 的 source-LOSO 离线分类很强：recovery-open vs hard-control AUC 1.000。但这是一个离线状态判别任务。
- P3.2 用冻结阈值，在 8 个 fresh source × 3 conditions 上运行，Direct Router 24/24 保留 Base，glass 恢复 0/8，错过 2/2 已知恢复机会。

关键事实是：已知恢复轨迹相对于 controls 的 trajectory-max AUC 只有 0.250。仅仅降低同一分数的阈值，可能先在 controls 上触发。该版本的问题包含评分排序与在线访问状态的差异，不能概括成阈值略高。

这段历史留下了一条重要边界：**离线能把已有好状态和坏状态分开，不保证从真实轨迹上能及时遇到、识别并利用这些状态。**见 [P3.2 closeout](/Users/hanshuo/Desktop/crash_bench/results/P3_2_FROZEN_DYNAMIC_CLOSEOUT_20260819.md)。

**6. 8/29：更强比较器改变了方法判断，也留下了具体误差定位**

第一轮 strongest-baseline audit 用 5 train、7 calibration、8 development sources：Outcome Router 的 source-macro utility 为 0.1944，Risk→BestFixed(Detour) 为 0.2111。差 −0.0167，95% CI [−0.0938,+0.0417]。因此原来相对 Risk→Retreat 的大幅成功差不能证明 outcome decomposition 普遍优于风险路由。

随后把 20 个旧来源作为 exposed development 做 source cross-fit，确保被评价来源不参与特征处理、拟合、选参和校准：Outcome Router 0.2722，Risk→BestFixed 0.2544，差 +0.0178，区间 [−0.0747,+0.1000]。结果仍没有建立强基线优势。这两组数字属于不同评估设计，不能挑其中较好一组覆盖较差一组。

支持审计同时发现：严格 Base、Detour、Retreat 各有 16、13、9 个 source 支持。历史 5-source train 只有 3 个严格 Retreat-optimal 决策，训练支持确实偏薄；合并已有 source 后可以开展更合理的开发诊断。

最值得继续利用的是 gate/choice 分解：

| 组成，tiny nonlinear choice 版本 | source-macro utility | 相对完整 Oracle 的缺口 |
|---|---:|---:|
| OracleGate + OracleChoice | 0.5642 | 0 |
| OracleGate + LearnedChoice | 0.4917 | 0.0725 |
| LearnedGate + OracleChoice | 0.2714 | 0.2928 |
| LearnedGate + LearnedChoice | 0.2311 | 0.3331 |

tiny choice 在严格 Detour/Retreat 子集的 balanced accuracy 是 0.7253；benefit gate 的 AUROC 仅 0.6062。就这份误差分解看，改善“该不该干预”比单纯再增强“干预后选 D 还是 R”更值得优先关注。两个 Oracle 混合器都用到了部署时不可得的真实信息，不能算部署基线。

历史 machine label 为 `CHOICE_CAPACITY_BOTTLENECK`，因为规则优先检查 tiny-vs-linear 条件选择。这个名称不等于最大剩余效用损失全部来自 choice。相同数据下 gate 的损失更大，论文和后续方法应读误差表，而不是只读一个终止状态名称。

另一个范围限制是：23 个严格 Retreat 状态全在 glass，60 个严格 D/R 状态中 51 个来自 glass。三种 condition 不是三个独立 hazard family。见 [强基线审计](/Users/hanshuo/Desktop/crash_bench/docs/iclr27/BASELINE_AUDIT_RESULT.md)、[source cross-fit](/Users/hanshuo/Desktop/crash_bench/docs/iclr27/SUPPORT_CROSSFIT_RESULT.md)、[gate/choice 分解](/Users/hanshuo/Desktop/crash_bench/docs/iclr27/ADVANTAGE_ROUTER_RESOLUTION.md:102)。

**7. 8/30–31：ODUR 与更大协议补充了实验基础，但没有形成方法胜利**

扩展线新增 source registry、曝光账本、策略 continuation、exact branching、option information boundary、D4 freeze、分角色评估及 Slurm 溯源。主要机制最终收缩到 π0 × observation staleness × LIBERO-Spatial tasks 0/2。

这里的“100 sources”是 126 个预声明 nominal candidates 经成功筛选和场景去重后得到的 100 个独立初始来源，不是 100 个任务。每任务 50 个，分配如下：

| 角色 | 两任务合计 | 实际进展 |
|---|---:|---|
| Train | 24 | D5 已执行，后续两轮模型只在这些 source 上拟合 |
| Calibration | 12 | D5 已执行，用于校准 |
| Development | 12 | D5 已执行，已反复暴露用于开发 |
| Confirmatory statewise test | 32 | D8 已执行并封存；没有 ODUR superiority test |
| Future sequential sources | 20 | 当时冻结的未来角色；本次审查没有运行或读取其新结果 |

D5：48 sources × 27 configurations = 1,296 个决策，三个 option 合计 3,888 条结局。D8：32 sources、864 anchors、2,592 branches。大部分数量增长来自同 source 的多个 anchor/severity/condition，不能替代任务、机器人或机制覆盖。

ODUR 的输入是两张相机的 4×4 RGB 均值与 8 维 proprio，共 104 维；不包含 VLA hidden、显式 observation age、历史序列或 nominal chunk。原模型 additive 结构把 option embedding 放到最后一个线性 head 前，限制了状态相关 option effect。原开发 ensemble utility 0.3853，DirectQ 0.3965，没有优势；校准后所有状态都 Stop。

D8 原来以“31 个 B1 source、只有 1 个 B0 source”记作 scope failure。9/6 后发现这个统计数实际是在数“整个 source 所有状态都 B0”，没有数“含 B0 状态的 source”。32 个 source 都含 B0，其中 31 个还含 B1。旧 release 保留为历史记录，原解释需要勘误；勘误不把已暴露 test 变回一个新确认集，也不创造 learned method 的优势。

其他机制不能统一叫失败：fragile 的唯一支持口径问题在更正后消失，回顾分析变 GO；action drift 仍有选项支持问题；narrow clearance 仍有 control catastrophe 有效性问题；unstable placement v1 则在几何预检就失败，8 候选、0 个物理 block、0 条 option outcome，根本没有检验恢复方法。后者用碗整体 AABB 当底部支撑尺寸、姿态相关 AABB 当固定几何身份，且失败初始化影响后续候选，仍是未修的历史工程入口。见 [D5–D7 历史记录](/Users/hanshuo/Desktop/crash_bench/docs/iclr27/D5_D7_SCOPED_PIPELINE_AUDIT.md)、[D8 历史 release](/Users/hanshuo/Desktop/crash_bench/docs/mainconf/D8_D10_SCOPE_FAILURE_AUDIT.md)、[非 glass 预检审计](/Users/hanshuo/Desktop/crash_bench/docs/iclr27/NON_GLASS_UNSTABLE_PLACEMENT_V1_AUDIT.md)。

**8. 9/6 第一轮：哪些旧问题已经修好，哪些只定位了**

| 项目 | 现在的状态 | 不应继续使用的解释 |
|---|---|---|
| B0/B1 source 支持 | 已区分 contains_B0、contains_B1、contains_both、entirely_B0，追加勘误 | “几乎没有不干预状态” |
| 状态×选项交互 | 新增 joint nonlinearity 和 per-option heads，保留 additive 对照 | “当前所有 ODUR 都只有选项偏置” |
| development refit 泄漏 | 训练编排与本轮实验改为 train-only 拟合、独立 calibration、开发读数 | “修复结果来自在 development 上拟合后回看” |
| 40%–70% 干预率门槛 | 已移除，报告有益召回和无益干预 | “多触发一些就更好” |
| 校准恒停 | 已识别、显式报告 CALIBRATION_FAILURE；保证方法仍未修好 | “校准已可部署”或“所有价值学习都无效” |
| 非 glass unstable placement | 本轮没有修复和重采 | “修 ODUR 时物理构造也已修好” |

24 train sources 拟合，12 development sources 上比较，同样 104-D 输入、5 seeds、100 epochs：

| 模型/策略 | U0 | 成功率 | 事故率 | 干预率 |
|---|---:|---:|---:|---:|
| Base | 0.3932 | 68.21% | 9.88% | 0% |
| DirectQ | 0.3965 | 65.12% | 6.79% | 28.09% |
| Additive ODUR | 0.3853 | 65.43% | 5.86% | 91.67% |
| Interaction ODUR | 0.3849 | 64.20% | 7.41% | 8.33% |
| Per-option ODUR | 0.4144 | 68.21% | 8.64% | 2.16% |

Per-option 的 324 次选择是 317 Base、0 Refresh、7 Stop。事故由 32 减到 28，成功保持 221；相对 DirectQ 的 ΔU0=+0.01785，95% CI [−0.01746,+0.05616]。这里观察到的小收益来自少量安全停止，尚未增加任务完成。

三个版本的 calibration correction 仍约 0.95–0.98，而要求 `predicted_accident + correction ≤ 0.10`。合法概率最小为零，根本没有选项能通过。显式识别这个数学不可行性是完成的修复；获得实用的统计安全保证仍是另外一个未完成问题。

第一轮 CPU 作业 `5627464` 用时 2:34。此前两个 Git 不可用的 launcher failure 在训练前退出，不是方法实验失败。详见 [第一轮结果](/Users/hanshuo/Desktop/crash_bench/docs/audits/20260906/ROUND1_RESULTS_ZH.md)。

**9. 9/6 第二轮：拟合能力改善了，跨来源的收益排序仍很弱**

第二轮直接预测 Refresh/Stop 相对 Base 的 gain，比较训练时长、正样本权重与 train-only 标准化。

四源工程小集的标准化模型在 2,000 epochs 把 gain MSE 降到约 0.000042，找回 4/4 已记录任务成功转换。Per-option outcome 在更长训练后也能拟合。这证明“100 epochs 没选 Refresh，所以这种模型不能拟合真实训练数据”的推论不成立。

但是完整数据的 500-epoch gain AUROC：标准化模型 train=0.998，development=0.449。训练标准化值没有明显在 development 上数值爆炸。现有证据支持训练—开发差距；没有单独证明唯一根因是样本量、缺少 age、视觉表示太粗或某种过拟合。

为什么第一轮预测 Refresh gain 总是负？在 29 个训练正收益状态上，per-option 平均预测终局收益差仅 +0.017908，连续成本差 −0.000079，再扣固定干预成本 −0.05，净值约 −0.032171。主要缺口是没预测出终局改变，不能用随手删除成本来掩盖它。

第二轮最值得注意的两组结果：

| 模式 | 已记录行为 | 正面信号 | 限制 |
|---|---|---|---|
| Interaction-500，Base/Refresh/Stop | U0 0.4370；成功 64.81%；事故 4.32% | 效用点估计与事故率改善 | 成功低于 Base 68.21%，损失 13 个原成功；11 个损失在 controls |
| 标准化 gain-500，只选 Base/Refresh | 221→227 次成功，68.21%→70.06%；121 次 Refresh | 选到 6/11 已记录成功转换，未损失原 Base 成功 | 新增 2 次事故；ΔU0=+0.00976，95% CI [−0.02060,+0.03828]；无益调用多 |

同一个标准化模型允许 Stop 时，会损失 49 个原 Base 成功，41 个在 controls。只看“总体事故少了”容易漏掉大量任务被停止的问题。当前开发数据的 privileged `真实 Base 事故→Stop` 诊断已经覆盖约 80.7% 的完整 Oracle 效用提升空间，说明 U0 很容易主要奖励安全中止。

CPU 作业 `5627906` 用时 2:07。所有这些仍是已暴露 development 上的离线选择，不能替代新来源、在线接管或可部署保证。见 [第二轮报告](/Users/hanshuo/Desktop/crash_bench/docs/audits/20260906/round2/RESULTS_ZH.md)。

**10. 9/6 晚些时候：测量链路的问题改变了“恢复标签”的解释**

旧 D5 的两个理论中性条件中，Base 与 Refresh 终局不同的计数：train 为 17/432，development 为 10/216。由于 anchor 位于五步 chunk 边界，且 controls 给到的是当前观测，队列刷新在软件语义上应中性。这里的差异不能全部当作观测陈旧被修好。

开发集 11 个“Base 未成功→Refresh 成功”记录中，8 个来自 stale、3 个来自 matched-buffer control；第二轮选中的 6 个是 4 stale + 2 control。数字作为记录没有错，因果解释需要收缩。识别出 controls 的差异也没有自动认证其余 stale 标签。

训练源工程探针给出更直接的证据：

| 检查 | 实测 | 能说明什么 |
|---|---|---|
| 6 新锚点 × 2 options 的同选项重复 | 12/12 对动作与状态轨迹不完全相同；1/12 对终局改变 | 分支起点恢复通过，仍不足以保证未来结局确定 |
| Task 0 同进程观察 | step 1 wrist 仅 3 个 RGB 数值相差至多 1/255；step 10 动作/状态分叉 | 微小观察差异出现在物理分叉上游 |
| Task 2 同进程观察 | step 2 wrist 6 个 RGB 数值有差异；step 10 分叉 | 第二个锚点也看到相同层次问题 |
| 重放完整策略输入 | 两锚点 35 步动作/状态一致 | 在这些具体检查中，输入重放可消除后续差异 |
| 跨作业比较 | 起始 observation/continuation hash 一致，初始动作仍相差约 0.00128 | 根因尚不能只归到相机，运行时/推理身份仍须核验 |

作业 `5628711` 和 `5629071` 分别耗时 5:05、2:43。它们使用两个训练来源的新 bundle，没有重放历史 D5 原始锚点，也不能外推为所有 OpenVLA 历史结果无效。

两个具体的持久化/采集缺口仍需牢记：旧 D5 不同 condition 只 `policy.reset()` 清队列，没有恢复统一 JAX RNG 起点；同一个 block 内选项之间则有 continuation 恢复。这是两种不同配对。旧表还只留了 feature blob、bundle ID 和 queue hash，没有可供直接重放的完整历史 bundle。hash 能校验身份，不能重建丢失内容。新的探针与复验入口补充了完整保存，但没有把旧历史追溯性补齐。见 [论文审查报告](/Users/hanshuo/Desktop/crash_bench/docs/audits/20260906/paper_review/COMPREHENSIVE_REPORT_ZH.md)。

由此产生一个合理的研究区分：单次结果后选赢家 `E[max U]`，与用部署输入选期望赢家 `E_X[max E(U|X)]` 一般不相等。执行波动与输入信息缺失均可能扩大差距。但这个一般关系不是新的理论贡献，也不能未经实测就说旧 Oracle gap 全部由噪声造成。

**11. 9/7 最新复验：流程完成了，任务救回与学习选择优势尚未出现**

这轮的设计已经落实了部分测量改进：12 个 train sources，按任务内 source ID 顺序取前六个；每源 stale/matched-buffer 各一新 bundle；每 bundle 两 options、每 option 八次，共 384 条计分分支。A/B 各四次，全部 A 完成并冻结选择后再跑 B；选项顺序交替；记录完整输入、动作、队列、RNG、checkpoint 与运行时。

最新 B 的每方法评价是 96 个配对机会，不是 384 个独立 source：

| 方法 | 成功次数 | 事故次数 | Refresh 调用 | Base-success controls 保持 |
|---|---:|---:|---:|---:|
| Base | 64/96 | 12/96 | 0 | 48/48 |
| AlwaysRefresh | 61/96 | 10/96 | 96 | 47/48 |
| A 选出的 period10 | 64/96 | 8/96 | 32 | 48/48 |
| A_reference，逐状态用 A 终局决定 | 64/96 | 8/96 | 4 | 48/48 |
| B 同表事后挑赢家，特权诊断 | 64/96 | 8/96 | 4 | 48/48 |

`A_reference` 没有训练参数，用的是同一状态之前四次执行的真实终局；它比 learner 能用到的信息多，不能称为已部署的学习选择器。`period10` 是在指定锚点步数能被 10 整除时做一次 Refresh，也不是全程每十步不停刷新的 controller。

唯一持续被 A_reference 选择的 Refresh 是 b02：task 0、stale、anchor=10、delay=3。A 中 Base 4/4 事故、Refresh 0/4 事故；B 中相同。但是 B 的 Refresh 4/4 都在 100 步超时，任务成功仍 0/4。降低事故转成安全未完成，不能写成救回四个任务。事故差 −4.17 percentage points 的 source CI 是 [−12.50,0.00]，收益只落在一个源的一处锚点。

本次按任务/condition 重算进一步看到：Base 在 B 的 controls 是 48/48 成功，stale 是 16/48 成功；AlwaysRefresh 的对应数是 47/48 和 14/48。因此总体 66.67% 同时包含很容易的 controls 和较难的 stale states，没有测出对后者的任务成功救回。

4 次调用与 period10 的 32 次调用，是一个值得记录的效率候选：同样事故/成功记录，更少刷新。但还没有可部署 predictor，没有调用预算匹配的强规则，没有完整成本对齐，也没有运行 Risk→BestFixed 与 DirectQ。不能将这个差别写成学习方法已胜出。

运行期间 home 配额不足，`5680180` 在完成 A 192 条、B 139 条后中断。`5695873` 只补 53 条 B，原结果迁往项目存储，完整 trace/bundle/A freeze/checkpoint 哈希经过核对；总计 384 scored + 1 条保留未计分截断尝试。分配用时合计 69:50。b17/repeat5 两 options 跨作业边界；预声明排除该对的诊断不改变结论。本次仅核验本地小型证据，没有重新访问约 43 GB 的远端原始轨迹。见 [最新结果](/Users/hanshuo/Desktop/crash_bench/docs/audits/20260907/selection_retest/RESULTS_ZH.md) 与 [存储恢复记录](/Users/hanshuo/Desktop/crash_bench/docs/audits/20260907/selection_retest/RECOVERY.md)。

**12. 本次新增核对：最新面板没有覆盖旧的成功转换配置**

这是本次在 9/6、9/7 既有报告之外新增的回顾性描述分析。连接键为 `physical_source_id + anchor_steps + severity(delay) + condition`；只读取 D5 训练记录，未选新样本、未改本轮面板、未训练或重新跑仿真。

| 历史数据范围 | 旧配置数 | 记录中的 Base 未成功→Refresh 成功 |
|---|---:|---:|
| 所有 D5 train，24 sources | 648 | 15：9 stale、5 fresh control、1 matched-buffer control |
| 最新面板选用的 12 sources，全部旧配置 | 324 | 9 |
| 与最新 24 bundles 对应的旧元数据配置 | 24 | **0** |

也就是说，最新复验使用的来源本来有部分旧成功转换，但那些转换位于该来源的其他 anchor/delay/condition。此次每源只取两个配置，覆盖该 12-source 旧配置网格的 24/324≈7.4%。这体现的是重复精度与状态覆盖的取舍。

这个事实进一步限制了最新零结果的解释：

- 可以说“事前、非结果筛选的这个小面板没有捕获任务救回”。
- 不能说“已经把旧成功状态全部重跑，并证明成功消失”。
- 也不能说“只要换到旧成功配置就会稳定成功”；旧单次标签仍可能有波动或控制条件解释问题。
- 元数据配置匹配不等于完整物理/策略状态一致。新旧采集编排和 continuation 不同，旧完整 bundle 不能直接重放；bundle ID 本身也包含身份信息。

A/B 合计八次中，24 个 bundle 的 Base 终局类别均未发生变化；Refresh 在 5 个 bundle 有终局变化。这个局部描述不能推出“Base 普遍确定”或给全库估计一个统一噪声率。

本次还重新调用现有分析器，从 A 重建 freeze，从 B 重算包括 10,000 次 task-stratified source bootstrap 在内的全部主统计与跨作业诊断，与保存的 `metrics.json` 完全一致；13 个小型证据文件哈希全部匹配。详见 [本次机器证据](/Users/hanshuo/Desktop/crash_bench/docs/audits/20260907/progress_review/evidence.json) 与 [复算脚本](/Users/hanshuo/Desktop/crash_bench/docs/audits/20260907/progress_review/reproduce_review.py)。

**13. 现在的难点，按研究依赖关系排序**

| 优先问题 | 已有证据 | 尚不知道的具体问题 | 判断何时取得进展 |
|---|---|---|---|
| 稳定任务收益是否存在于目标状态分布 | 旧 D5 有单次成功转换；新面板没有捕获；b02 仅事故转未完成 | 哪些可定义、非事后挑选的状态有可复验成功增益？ | 相同声明执行分布中，选择与评价隔离后仍保留任务收益 |
| 单次标签是否代表干预效应 | 中性 controls 终局不同、同选项重复波动 | 状态、观察流、policy continuation、运行时各贡献多少？ | 有完整记录，可将收益稳定性与执行波动分别报告 |
| 部署输入能否预测期望收益 | 104-D 模型 train AUROC 0.998、dev 0.449；旧导出缺 age/chunk/history | 特征不足、标签不稳、source 差异各占多少？ | 在 held-out source 上，相对同输入强基线有可复现改进 |
| 是否需要 learned state selection | 旧 glass Risk→BestFixed 很强；新 period10 达到 A_reference 的同样成功/事故 | 状态相关选择能否在同调用预算和成本下超过规则？ | 避免用弱固定 fallback、特权 reference 或额外调用制造优势 |
| 任务收益与安全停止的目标冲突 | Stop 消除事故，同时大量丢失原成功；风险权重会改变 utility 排名 | 真正要优化事故减少、完成增加还是调用效率？ | 主张、主指标和实际行为一致，分别保留成功/新增事故/成本 |
| 从给定状态到在线接管 | P3.1 离线强，P3.2 fresh 24/24 Base | 在线访问状态和触发窗口是否能被正确学习？ | 从 reset 的真实闭环中及时利用收益、保留 controls |
| 实用校准保证 | q≈0.95 与事故上限 0.10 结构冲突 | 校准对象、保证目标和可行动性怎样同时成立？ | 保证语义明确，且合法输入域中允许有用动作 |
| 覆盖与独立样本 | glass 主要一任务，staleness 两任务；新复验仅 12 sources | 效果是否只属于当前机械构造与来源？ | 开发结论清楚以后，独立 source 与适当新任务证据支持外推 |

需要特别具体地理解 Refresh：当前只是锚点时**一次**软件观测队列清空与 fresh observation 重查询。下一步仍走同一个延迟队列，缓存会再次积累陈旧；它不永久修复传感器，也不负责规划新的完整恢复轨迹。分支最长 100 步，锚点固定 5/10/15，事故代理为机器人接触力 ≥75 N。上述定义限制了它能证明的能力。见 [一次性调用](/Users/hanshuo/Desktop/crash_bench/scripts/expansion/selection_retest.py:177) 与 [队列语义](/Users/hanshuo/Desktop/crash_bench/crashbench/mechanisms/observation_staleness.py:40)。

另一个统计难点是重复开发：12 个 development sources 已经承载多个模型、epoch、权重与 option mode 比较；普通 source bootstrap 不会消除这种模型选择偏差。5 个训练 seeds、324 个决策或 384 条执行分支，都不等于相同数量的独立环境。零宽 bootstrap 区间也只说明观测配对中没出现差异，不能证明总体差异精确为零。

**14. 哪些结论站住了，哪些应当撤回或保持开放**

| 说法 | 当前判断 |
|---|---|
| 路径中注入墙会引发特定碰撞，移出路径后显著改变行为 | 有限定机制的实证支持 |
| 风险可从某些 frozen hidden 表示读出 | 有支持；不等于风险理解、动作因果作用或全架构规律 |
| 结构化避险和某些任务完成恢复存在 | 有支持；要区分 safe abort、低墙 witness、筛选后的 glass oracle |
| E16 在 fresh matched-state 上出现有用的 tradeoff | 保留原证据；强基线和在线范围限制仍生效 |
| ODUR bug 说明整个 intervention-value learning 无效 | 不成立，关键 bug 已修；修复后也尚未建立方法优势 |
| D8 只有一个 source 含不干预状态 | 错；一个是 entirely-B0，32 个都 contains-B0 |
| 已学会可靠在线 recovery timing | 未建立，P3.2 是该版本的明确负结果 |
| 第二轮记录中模型选到了一些成功转换 | 成立；不能把中性 control 转换都称为 staleness recovery |
| 最新 384 条分支证明旧成功都是噪声 | 不成立，未覆盖旧成功配置，也没有重放原始 bundle |
| 最新 A_reference 是已经训练好的低成本选择器 | 不成立，它使用 A 的真实终局，是特权参考 |
| 最新复验显示“事后成功赢家的收益在 B 消失” | 不成立，这个面板连事后赢家也没有任务成功提升 |
| 一切实验都需要绝对 bit-exact 才能研究 | 不成立；可声明执行分布，用重复与隔离评价估计收益 |
| 项目只能写永久 negative paper | 无此证据；可写范围明确的诊断稿，方法主张取决于后续证据 |

**15. 项目离论文与可用系统分别还有多远**

工程上：当前提交的本地测试是 **545 passed，20.97 秒**，`scripts/audit_repo.py` 通过。模型交互、统计定义、source 角色、A/B 冻结、恢复缺失后缀等已有测试。测试通过说明实现满足被测试的契约；其中很多使用假环境或保存记录，不证明 GPU/仿真未来轨迹确定，也不证明研究方法有效。

可复现性上：较新的模型、输入、hash、Slurm 和恢复证据较完整。最早的 nominal 与部分 checkpoint provenance 仍不全，旧 D5 完整锚点持久化不足。不能把新入口的完善反向写成所有历史数据都可精确重放。

写作上：已有两份不同阶段的稿件资产：

- [glass 完整 LaTeX 稿](/Users/hanshuo/Desktop/crash_bench/docs/iclr27/manuscript/manuscript.tex) 最后实质提交为 8/30 `cb79556`，题为 *Risk Does Not Specify Intervention: Exact-State Diagnostics for an OpenVLA–LIBERO Safety Case Study*。它有 appendix、artifact map、审计与构建流程，研究范围是旧 glass 诊断。
- [staleness 稿](/Users/hanshuo/Desktop/crash_bench/docs/mainconf/SCOPED_STALENESS_BENCHMARK_MANUSCRIPT_V2.md) 最后提交为 8/31 `3b0e27f`，仍保存当时 scope-failure/旧 B0 解释。
- [9/6 新论文蓝图](/Users/hanshuo/Desktop/crash_bench/docs/audits/20260906/paper_review/PAPER_BLUEPRINT.md) 已围绕 useful/harmful recovery 与 measurement audit 重写问题，但还不是包含 9/7 复验的完整新稿。

所以写作资产并不少；**当前缺的是把修复、旧正负结果、测量发现和最新复验组织成同一个准确主张的新稿**。本次没有重新编译或视觉检查 PDF，也没有进行新的近邻文献检索，不能据此宣布投稿就绪或优先权。

一种当前可成立的稿件重心是：给定状态的干预选择、风险与任务价值区分、正常成功保持、重复执行测量和来源隔离的具体案例研究。若坚持任务恢复方法论文，关键缺口是 stable expected success gain、强基线增益和在线实现。若转向成本效率方法，4 vs 32 的特权参考只是候选线索，还需要真正的部署选择器与预算匹配比较。

**16. 我建议怎样继续，而不把研究又变成一串无限入口修复**

先明确一个近期主要命题，任务成功、事故代理、调用效率三者中选一个做主贡献，其余作为必须透明报告的代价。现在最容易发生的混淆是：方法为了减少事故而 Stop，报告却说成任务恢复；或者特权 reference 少调用，报告却说成 learned routing 胜利。

现有证据支持的工作顺序如下。这是研究建议，本次没有启动额外实验：

1. **把当前已知事实统一成状态与证据索引。**本报告已完成跨分支进度复盘和旧配置覆盖核对。README、旧主结果页、两个历史稿与 CURRENT 的时间层次仍应在后续写作中明确，避免继续把旧 gate 当当前实验结论。
2. **若继续任务恢复主线，先定义可重复收益的对象。**分别解释固定早期面板的覆盖、旧记录中的候选机制，以及新 bundle 与历史 bundle 的差异。旧成功可作为机制解释组，不能未经区分地作为代表性总体；任何新增采集都要单独预声明来源、重复、执行分布和预算。
3. **有稳定收益后，再研究部署信息和预测。**当前输入、真实软件 age、当前/已交付观测关系和已有 nominal action/chunk 是不同候选。不能用 condition/severity 或未来结果代替部署特征，也不能因缺失 proposal 而多调用一次政策、悄悄改变比较成本。
4. **比较要覆盖简单规则与强学习基线。**至少明确 Base、AlwaysRefresh、预算匹配规则、Risk→BestFixed、DirectQ 与 proposed selector；相同来源、option pool、时域、成本与选择/评价分离。当前 384 面板没有完成这一整组比较。
5. **在线时机作为单独问题验证。**旧 glass 的 P3.2 已说明这一步有真实困难。只有给定状态的收益/选择足够清楚，才能解释在线失败究竟来自窗口、分数、状态分布还是 option 本身。

这些步骤用于隔离不确定性，不是把某个数字设成研究永久关闭条件。已有 gate 名称只能约束对应冻结实验的主张；旧数据保护、source 隔离与新确认独立性则应持续保留。

从整条历史看，持续目标是清楚的：让机器人遇到危险以后仍能完成有用的工作。转向主要来自发现“检测、避险、恢复动作、选项收益、触发时机、执行稳定性”各自是独立难题。现在已经把这些问题拆开，并对其中几项完成了修复；最值得集中投入的剩余环节，是**稳定收益、可观测收益和超越简单规则的选择价值**。

**17. 本次新增交付与验证范围**

- 本报告：`docs/audits/20260907/progress_review/PROJECT_PROGRESS_REVIEW_ZH.md`。
- 描述性复算脚本：`reproduce_review.py`。读取本地 Git、D5 与最新训练源复验小型证据；无拟合、仿真或 D8 原始读取。
- `evidence.json`：分支祖先关系、A 冻结重建、B 完整统计一致性、13 文件 hash、按任务/condition 的计数、同选项终局变化、历史配置覆盖及 54 个读取输入的 SHA-256。
- 本次重新执行全仓 545 项测试与 repository audit。历史所有 tracked results 保留。工作区只增加本次报告和复算交付，没有修改现有实验代码、结果、研究计划或权限。

复算命令，输出应写到新文件以保留本次快照：

```bash
PYTHONDONTWRITEBYTECODE=1 /private/tmp/crashbench-repair-venv/bin/python \
  docs/audits/20260907/progress_review/reproduce_review.py
```

临时 Python 环境路径仅代表本次可用运行环境，不应当成为可长期部署的依赖声明。
