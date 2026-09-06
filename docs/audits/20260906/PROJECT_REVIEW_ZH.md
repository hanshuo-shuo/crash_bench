**CrashBench 三分支研究历程与代码审计｜2026-09-06**

我的判断是：这个项目保留了有价值的实验资产，现有记录不足以宣布“恢复／干预价值学习这条方向失败”。代码中确实存在会改变科学结论的统计口径错误、让策略必然停止的校准组合，以及限制方法表达能力的实现。研究流程还把多个探索结果过早绑定到了永久停止和论文降级上。

因此，优先工作应当是纠正评估与方法实现，恢复可迭代的开发流程。保留历史结果，同时重新界定它们究竟否定了什么。直接把数值门槛调低到能过，不会解决这些问题。

本次审查覆盖本地保存的三个 Git 引用、提交历史、主要计划和结果记录、采集／模型／校准／门控代码及相关测试。没有运行新 GPU 实验、访问 Quest、修改模型或覆盖冻结结果。所有新增复算都是回顾性审计，不是新的确认性实验。对于“心路历程”，下面是依据记录重建的研究动机；Git 本身不能判断每项决定究竟出自你还是某次助手建议。

**一、三个分支其实是一条演进链**

| 引用 | 本次审查的提交 | 时间 | 当时的主线 |
|---|---|---|---|
| `main` | `a0fae915c37a13527fe8926ce870d52a4958ff2b` | 2026-08-12 | 碰撞诊断、内部风险读出、结构化避险；glass 学习恢复受阻后收缩叙事 |
| `codex/iclr27-exact-state-intervention-routing` | `cdc4802494a7c0e540fe80bfebc2dfc304af35d0` | 2026-08-29 | exact-state 反事实选项路由；E16 正面结果、序贯时机失败、强基线审计后再转诊断 |
| `codex/odur-negative` | `c3378366d53cf514914ae837292126dbd806c4b2` | 2026-08-31 | 补充 π0 精确分支和新机制；ODUR／保守选择器；最终标为 scope failure |

`git merge-base` 证实：main 的 tip 是 routing 的祖先，routing 的 tip 又是 ODUR 的祖先。它们不是三个相互独立的方案，也不存在需要先合并三个互相分叉分支的问题。当前工作区停在最早那个 tip，所以只看当前 README 会漏掉后面两周的工作。

**二、我理解你的研究动机是逐渐变具体了**

| 阶段 | 你在追问什么 | 得到了什么／为何转向 |
|---|---|---|
| 6 月 17–21 日：benchmark 起点 | VLA 平时能做任务，快出事故时能否识别并救回来？ | 最初把碗移到桌边，机器人却没有按预想接触危险；改成在既有动作路径注入墙，终于得到稳定碰撞。 |
| 6 月 22–29 日：定位失败机制 | 它是没看见，还是看见了仍沿原动作走？ | 同类墙移出路径，事故消失；风险能被 probe 读出；接上 `RetreatHold` 后 15/15 撞击变 0/15。项目从目录型 benchmark 变成了机制诊断和干预。 |
| 7 月：不满足于只停下来 | 安全停止以后，原任务怎么办？ | 高墙绕行遇到整臂碰撞；降低 d62 墙后有任务完成实例，但几何已经改变。glass 成为较适合研究任务恢复的对象。 |
| 7 月 30 日：尝试把避险学进策略 | 能否通过少量恢复监督改变 VLA？ | oracle-stop LoRA 在两种 held-out wall 场景的六次运行中，把 6/6 碰撞变成 1/6，其余主要是安全中止；控制组没有稳定改善。它证明少量学习可改变行为，但没证明学会任务恢复。 |
| 8 月 3–12 日：glass 学习恢复 | 能否同时学“什么时候接管”和“接管后怎么动”？ | 大量工作花在状态恢复、抓取姿态、oracle 绕行和有效样本筛选。最后只有一个训练 pair、两个开发 pair；oracle 6/6 成功，learned gate 不触发，动作头未完成独立在线验证。 |
| 8 月 13–17 日：拆开难题 | 先固定会绕行／会停的 controller，只学选择，能否走通？ | E16 从同一状态比较 Base、Detour、Retreat，得到 fresh matched-state 的有用结果：展示点成功率 87.5%，事故率 8.33%。 |
| 8 月 18–19 日：补上在线时机 | 从 reset 开始，能否自己找到恢复窗口？ | 早期 P2 有小幅改善；P3.1 离线分类很好，P3.2 fresh 在线却 24/24 选 Base。问题从“选什么”变成“何时选，以及训练状态与在线轨迹是否一致”。 |
| 8 月 29 日：提高比较强度 | 相对 Risk→Detour、DirectQ 等强基线，方法还成立吗？ | 旧的 Risk→Retreat 比较不够强。20-source OOF 的 OutcomeRouter 对 Risk→BestFixed 只有 +0.0178 效用，区间跨零；但 oracle 仍明显更强，conditional choice 也有信号。 |
| 8 月 30–31 日：尝试扩展和 ODUR | 换更完整的分支引擎、机制和选择器，能否形成方法论文？ | 两任务 π0 exact branching 跑通，D5 有 48 sources；ODUR 接近简单基线，校准后全停；D8 因 B0 统计被标为测试范围失败。 |

这条线背后有一个持续的目标：**让机器人从即将失败的状态返回有用的任务执行。** benchmark、probe、oracle、router 都是你为这个问题寻找可实验的切入点。转向大多有技术原因；真正消耗人的部分，是每次刚找到一点进展，工作目标就又被扩成“先满足更完整的证明和门控体系”。

史料依据：[最初动机](/Users/hanshuo/Desktop/crash_bench/docs/archive/motivation.md)、[oracle-stop 实验汇总](/Users/hanshuo/Desktop/crash_bench/results/oracle_recovery/report_assets/analysis_summary.json)、[8 月 24 日进展记录](https://github.com/hanshuo-shuo/crash_bench/blob/cdc4802494a7c0e540fe80bfebc2dfc304af35d0/CrashBench_Advisor_Progress_Update_2026-08-24.md)。

**三、最需要先处理的七项发现**

这里的 P1 表示会实质改变实验解释或使核心方法不可用；P2 表示重要设计／流程问题。不同问题不会被统一叫作“模型失败”。

**F1 · P1 · B0 源支持的聚合口径错位，最终“缺少不干预对照”的解释不成立。**

[D8 分析器 L74、L91](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/scripts/expansion/analyze_benchmark_test.py#L74) 做的是：

```python
source_benefit = any(row["benefit"] for row in rows)
benefit_zero_sources = len(source_rows) - benefit_one_sources
```

于是，一个源只要有一个 B1 状态，它所有 B0 状态都会从“B0 source support”中消失。实际统计的是“全部 anchor 都不受益的纯 B0 源”。[原 benchmark 规范](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/docs/iclr27/BALANCED_INTERVENTION_BENCHMARK_SPEC.md#L173) 要求 B0、B1 各自在多个源中出现；同一个源本来可以同时支持两种标签。

我读取完整已提交的 anchors／branches，复算如下：

| 数据 | 独立源数 | 旧代码：全部为 B0 的源 | 至少含一个 B0 的源 | 至少含一个 B1 的源 | 同时含 B0/B1 的源 | B0 决策数 |
|---|---:|---:|---:|---:|---:|---:|
| D5 train+development | 36 | 4 | 36 | 32 | 32 | 849/972 |
| D5 development | 12 | 2 | 12 | 10 | 10 | 284/324 |
| D8 已封存测试 | 32 | 1 | 32 | 31 | 31 | 752/864 |

D8 中 13 个独立源还包含 gap≥0.10 的严格 Base 最优状态，共 16 个这样的决策。严格大间隔 Base 支持较少，应当报告；但这与“没有不干预对照”是两回事。

**影响：**旧数字在“纯 B0 源”的定义下计算正确；问题在于把它命名并解释成 B0 标签的源支持。“31 个都受益、只有 1 个对照，所以无法考察不干预”把两种统计对象混在了一起。事实上，31 个源同时拥有有正干预收益和没有正干预收益的状态，正符合 within-source intervention-choice 研究的需求。D5 同样受此口径影响。

**处理：**分别报告 `contains_B0`、`contains_B1`、`contains_both` 和 `entirely_B0`；不要用互补计数替代标签支持。原机器决定保留，新增带修订说明的统计报告。这个修正不等于自动获得新的确认性胜利，也不证明 learned method 成功；它足以撤回“数据几乎没有不干预情形”的解释。

**追查上游后的补充：同一口径还影响了 D2 机制筛选。**我读取四个 screen 的 64 个已提交 source shard，并调用各自真实的 block 汇总函数：

| D2 机制 | 有效源 | 旧 B0 源计数 | 含 B0 状态的源 | 原决定及其他问题 |
|---|---:|---:|---:|---|
| fragile-path | 16 | 2 | 16 | SCOPED_CONTINUE；唯一 scope miss 就是 B0 |
| staleness | 16 | 6 | 16 | GO |
| action drift | 16 | 0 | 16 | SCOPED_CONTINUE；另有两种 strict intervention winner 的支持不足 |
| narrow clearance | 16 | 2 | 16 | NO_GO；另有 matched-control catastrophe 硬有效性失败 |

所以“只有 staleness 值得进入正式机制”的历史结论也应重审：fragile 的唯一降级理由受这个口径影响。action drift 和 narrow 的其他问题仍在；修正 B0 不会自动补齐三个有效机制。相关源码包括 [fragile analyzer L94](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/scripts/expansion/analyze_fragile_screen.py#L94) 和 [action-drift analyzer L85](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/scripts/expansion/analyze_action_drift_screen.py#L85)。

**F2 · P1 · 当前 conformal 参数使整个选择器恒等于 safe_stop。**

[selector L122–141](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/crashbench/models/selective.py#L122) 要求：

```text
predicted_catastrophe(option) + q_absolute <= 0.10
```

冻结文件中的 `q_absolute = 0.9769746549`。对于任何合法概率 p∈[0,1]，p+0.97697 都大于 0.1。所以没有任何非 Base 选项可以通过。Base 也必定触发回退条件，因此所有输入都选 safe_stop。

这不是只在当前 324 个开发决策上恰好出现的结果。我调用真实 selector，输入 10,000 组随机合法概率／效用，全部 safe_stop；即使给它“Base 必撞，Refresh 必成功且零风险”的理想预测，它也停止。

另一个惩罚是 `q_utility=58.2551`，乘最小 scale 0.05 后，任意 pairwise 下界至少扣 2.9128。它还要求干预同时胜过 Base 和其余选项，远强于“选择有正收益的安全干预”。

根源可见于 [calibration_source_scores](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/scripts/expansion/calibrate_selector.py#L89)：先取每源所有 anchor／option 的最大残差，再在 12 个源、α=0.1 上取第 12 个次序统计量，即最大值。一次预测不准的事故就可能推高整个源的修正量，随后这个修正量又应用到所有选项。

这不能通过把 α 从 0.1 随手改成 0.2 解决。在保存的这组残差上，要使绝对修正量降到 ≤0.1，α 至少需约 0.846。这说明要检查校准对象、预测器和保证目标。

Conformal 对结果集合的边际覆盖，也不能直接解释为每个状态下事故概率的条件保证；这里对 0/1 事故标签的残差上界尤其需要明确统计含义。[Angelopoulos 与 Bates 的原始教程](https://arxiv.org/html/2107.07511v6) 区分了这两类覆盖。

**影响：**“校准后全部停止”是这个冻结组合真实的失败。但它主要检验了一个不可用证书／回退策略，不能用于判断更一般的干预价值学习是否无效。当前检查只要求 quantile 有限，没有检查是否还允许任何有用动作。

**处理：**先在开发实验中分别比较点预测选项选择、对 Base 的收益门控、以及保守校准；逐层测量带来的成功率、事故率和效用变化。正式保证只绑定适用的版本。把不可用证书标成 calibration failure，保留 safe-stop 作为该版本的行为记录。

**F3 · P1 · ODUR 的选项只改变最后一层偏置，缺少关键状态×选项交互。**

[OptionOutcomeModel L93–119](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/crashbench/models/option_outcome.py#L93) 先独立计算 state trunk，再拼 option embedding，直接接一个线性 outcome head：

```text
logits(x,o) = W_x h(x) + W_o e(o) + b
```

拼接之后没有非线性，也没有 option-specific state head。因此，选项造成的任意两类 outcome 的 log-odds 位移与状态无关。这明显限制了“在状态 A 刷新有用、在状态 B 刷新无用或有害”的建模。

我检查 seed 0 已保存的 972 个 anchor 预测：Refresh 相对 Base 的“成功／事故 log-odds 位移”始终约 0.0333865，跨状态波动只有 1.64×10⁻⁶，与这个代数限制一致。这里没有重新训练，也没有用合成结果代替原数据。

准确边界：softmax 和三类加权效用仍可能让最优选项随状态改变，所以不能说它“完全无法路由”。可以确定的是，它缺少一般的状态相关选项作用。相比之下，[DirectQ](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/crashbench/models/advantage.py#L63) 每个 option 有独立状态系数，这一维表达能力反而更合适。

另外，ODUR 实际输入是两张相机图的 4×4 RGB 均值，加 8 维状态，共 104 维。它没有使用 VLA hidden features，也没有 observation age、历史序列、动作 chunk 等输入。粗特征是否决定了失败尚未验证，不能当作已证实因果；但当前实验也不能概括成“充分利用 VLA 内部表示仍学不会”。

**处理：**最小修复是在拼接之后加非线性层，或使用每选项独立 outcome/cost head；保持同样输入和训练预算，比较是否改善。用确实需要相反选项作用的小样本验证表达能力，再比较 DirectQ。不要直接跳到更大模型。

**F4 · P2 · 最低 40% 干预率与现有效用目标冲突，Oracle 自己也过不了。**

[Gate B L159](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/scripts/expansion/analyze_scoped_gate_b.py#L159) 硬要求 `0.40 <= intervention_coverage <= 0.70`。

开发集 324 个决策中，仅 40 个有正干预收益。知道全部结果、在没有收益时保留 Base 的效用最优 Oracle，干预率是 40/324=12.35%，也会被该门槛拒绝。满足 40% 至少要干预 130 次，即至少追加 90 次没有正收益的干预。

我还计算了这个约束本身的最小代价：开发集 Oracle 效用由 0.604805 降至最多 0.598930。这个代价约 0.00588，远小于全部停止造成的损失；因此我没有把它夸大成当前崩溃的主因。问题在于，门槛奖励干预数量，目标函数却已经对无效干预收费。

**处理：**取消研究探索的最低干预配额，报告 B1 条件下的有益干预召回、B0 上的无效干预和总体效用。若确有部署配额要求，应单独定义该约束下的最优策略和评估任务。

**F5 · P2 · Gate B 在 refit 已见过的 development 上评估。**

[训练编排 L79–106](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/scripts/expansion/run_scoped_training.py#L79) 先在 train+development 上 refit，然后把这些 refit 模型送入 development Gate B。五个 refit manifest 都写明 `fit_on=train_development`，拟合 2,916 行，其中包括 972 行 development。

因此，后续“calibrated development”是对拟合数据的回看，不是独立泛化评估；其 DirectQ 比较器仍来自 train-only 训练。校准源与 test 源未被用于拟合，这一点是正确的；但 `test_rows_read=0` 不能证明所有验证角色都合理。

**影响：**它没有制造本次负结果，甚至可能偏袒模型。它削弱的是 Gate B 作为独立发展判断的意义。

**处理：**比较使用 train-only、独立 calibration 后的模型；或做完整 source-level cross-fitting。选定方案后，再把 development 并入最终 refit，并将那次 development 回看明确标为训练内诊断。

**F6 · P1／已知缺陷复核 · 非 glass preflight 确实在方法开始前被实现错误挡住。**

仓库已经有一份诚实的 [v1 preflight 审计](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/docs/iclr27/NON_GLASS_UNSTABLE_PLACEMENT_V1_AUDIT.md)，记录 8 个候选、0 个有效物理 block、0 个 option outcome。本次核对了相关源码：

- [_measure_geometry L257](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/scripts/iclr27/run_non_glass_option_ambiguity_screen.py#L257) 用整个碗的 world AABB 尺寸当底部接触半径，可能把可放置区域算为空。
- [L623–625](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/scripts/iclr27/run_non_glass_option_ambiguity_screen.py#L623) 先写 `raw_geometry` 再 derive；derive 抛异常后，后续候选会跳过初始化，继承半完成状态。
- [L644](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/scripts/iclr27/run_non_glass_option_ambiguity_screen.py#L644) 对受姿态影响的 world AABB 使用 0.2 mm 的不变性要求，将正常姿态变化判成 geometry drift。
- [分析器 L329](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/scripts/iclr27/analyze_non_glass_option_ambiguity_screen.py#L329) 把 selected source 数写成 attempted source 数，进一步隐藏了失败链。

这些是已有记录且源码可证的实现缺陷，不是本次新跑出的模拟结果。v1 的零结果不能否定 unstable placement 机制。后续文档已经正确把它叫作工程失败，但实现仍留在分支里；将研究永久停掉，不能代替修复这些缺陷。

**F7 · P2 · 一些测试锁住的是研究结论和措辞，而没有验证科学语义。**

[main audit 的 REQUIRED_CURRENT_TEXT](/Users/hanshuo/Desktop/crash_bench/scripts/audit_repo.py:53) 强制要求“Pilots D/F 不是当前目标”等文字。[ODUR 的 README 测试](https://github.com/hanshuo-shuo/crash_bench/blob/c3378366d53cf514914ae837292126dbd806c4b2/tests/test_readme_odur_negative.py#L7) 要求出现“31 sources”“only 1 source”“required at least 3 controls”。

相反，B0 测试没有验证“同一个源应同时计入 B0 和 B1 支持”；模型测试主要检查 shape、有限 loss 和能 backward，没有覆盖状态相关选项作用。

我分别执行三个分支的 `scripts/audit_repo.py`，全部通过。**这证明现有文档、指纹和规则互相一致，不能证明规则本身表达了正确的研究问题。**

**处理：**冻结报告可以测试不可意外改写；当前研究状态应允许带理由的版本更新。单元测试优先验证统计定义、数据角色、选项语义和反例。把“未来不得训练”的句子写进测试，不应成为新研究的技术前提。

**四、早期 gate：有些必要，有些把探索难度抬成了最终验收难度**

| Gate／规则 | 审计判断 | 建议用途 |
|---|---|---|
| 每个保留样本的状态、控制器、action buffer 和 RNG 恢复正确 | 必要；否则 option 差异可能来自恢复误差 | 保留为数据有效性检查；失败样本明确排除和记账 |
| source 不跨训练／验证／确认测试；原始结果可追溯 | 必要 | 保留；关注实际数据依赖而非只有标志位 |
| 有 oracle 恢复 witness 才宣称“可恢复状态” | 对可恢复子集成立 | 有效 oracle 样本可供训练；完整总体另报成功率和筛选率 |
| 所有候选的 nominal replay 必须 100% 通过才训练 | 过度聚合 | 采集器成熟度、有效样本数量和样本自身有效性分别报告 |
| 只能用一个固定 H，其他 H 的有效 pair 无法合并使用 | 对固定-H 主张合理，对学习开发过窄 | 允许 horizon 多样的训练数据，保存真实时间标签；确认性固定-H 表另做 |
| 1 个 train pair、2 个 validation pair 上，用 gripper sign≥95% 否决动作路线 | 代理指标过重 | 看真实接管后的闭环成功、关键抓取／松爪时刻和错误位置 |
| 先过 80% timely trigger、≤10% control FPR 才做任何组合尝试 | 是一种部署目标，不是唯一探索准入规则 | 先测可实现的 recall–FPR–utility 曲线以及触发后结果 |
| 完整 episode 的控制分数最大值做阈值 | 有其序贯误报控制意义，但很严格 | 明确它控制什么，并测真实恢复窗口内触发；不能只用单个 T−20 frame 代替在线效果 |
| +0.03／+0.06 效用、每类 recall、4/5 seeds、coverage 等全部同时过 | 可作某个强主张的标准；不适合作为所有研究能否继续的判断 | 一个主指标配关键安全指标；其他指标用于解释 tradeoff |
| “只允许一次 corrective-data”“不得加 MLP／GRU”“no P3.3” | 属于当时的时间／资源选择，不是科学定律 | 在明确开发预算下允许针对瓶颈修复；保存完整探索记录 |
| 旧 test 不准覆盖、不准悄悄补样本直到过关 | 正确 | 保留旧测试；错误更正另出说明；新协议使用新确认数据 |
| 一次失败后未来永久禁止新测试／新模型 | 对未来研究过强 | 一个冻结实验结束，不等于研究问题永久关闭 |

两个早期实例特别能说明问题：

1. [Pilot B 汇总器](/Users/hanshuo/Desktop/crash_bench/scripts/summarize_glass_pilot_b.py:121) 的 replay 分母是全部 Base catastrophe 候选，而非已接纳有效 pair。某些候选失败可以说明采集质量不够稳定，但不应自动抹去其他样本的训练价值。实际 120 个候选/H 单元里有 5 个有效 pair，分散在 H20/H15/H10/H5。它们不够支持原固定-H 规模主张，却可以用于明确标记的学习／控制器调试。来源重复仍须去重处理。
2. [Pilot E](/Users/hanshuo/Desktop/crash_bench/scripts/summarize_glass_pilot_cdef.py:112) 即使闭环指标达到要求，也会因全帧 gripper sign<95% 被否决。全帧动作一致性不能替代关键接触事件和最终任务成功。当前动作头 0.846 的分数说明有值得检查的问题；在 Pilot E 根本未运行的情况下，无法据此宣布“学习恢复已失败”。

还要保留另一面：早期 oracle 本身确实很弱。[fresh H20 记录](/Users/hanshuo/Desktop/crash_bench/docs/archive/glass_recovery_20260812/PILOT_B_FRESH_H20_20260812.md) 中，四个明确 oracle task failure 都跑完 48 个配置，主要在最终下探阶段碰玻璃或未抓起碗。降低 gate 无法修复 controller。这些失败需要 controller 改进或改变研究子集定义。

**五、哪些负结果仍然应当承认**

可以承认当前实现不成功，同时拒绝过度外推。

- **E16 的 87.5% 是有用的小样本匹配决策结果。**它来自 8 个独立 source、24 个相关条件决策；不能把它当作 24 个独立场景，也不能自动扩成完整在线恢复。Risk→Retreat 是较弱比较器，后续加 Risk→BestFixed 和 DirectQ 是合理补充。[强基线审计](https://github.com/hanshuo-shuo/crash_bench/blob/cdc4802494a7c0e540fe80bfebc2dfc304af35d0/docs/iclr27/BASELINE_AUDIT_RESULT.md) 对旧 headline 的收缩有依据。
- **P3.2 不只是阈值略高。**两个已知可恢复 glass 轨迹的最大分数相对 controls 的 AUC 为 0.250。仅降低同一分数的阈值会带来更早的控制误触发。这足以否定该版本的可靠序贯识别，不能推成“时间模型、原始特征、learner-visited 数据都不值得再试”。[P3.2 记录](https://github.com/hanshuo-shuo/crash_bench/blob/cdc4802494a7c0e540fe80bfebc2dfc304af35d0/results/P3_2_FROZEN_DYNAMIC_CLOSEOUT_20260819.md)。
- **ODUR 当前未优于强基线。**未校准 ensemble 的开发效用 0.385347，DirectQ 为 0.396533。我的回顾性 source-paired bootstrap 得到差值 −0.01119，95% 区间约 [−0.07038, +0.04730]，n=12。它不支持方法优势，也不足以证明一切 option-conditioned 学习无效；该区间不校正模型选择，仅作开发诊断。
- **重复训练五个 seed 不能补足独立场景。**seed 反映优化不稳定，source 才承载环境泛化。小样本没有证明有效，并不等于已经精确证明无效。
- **直接恢复动作头和选择结构化 controller 是两条不同主张。**用 frozen Detour 并不低级，也可以形成方法，但应准确命名为 learned selection over structured options。
- **最初的“模型知道但不做”需要谨慎。**probe 可解码、动作未避险和外接 controller 有效是观测；它们不等于已经证明原策略内部存在完整物理理解，或找到唯一因果机制。

本次也排查了一个未得到支持的猜测：D5 的 1,296 个 104-D 特征向量没有完全相同的向量重复。因此不能声称已经发现“完全相同输入却有冲突标签”的硬不可辨识问题。更细的感知信息不足仍是待实验的假设。

**六、为什么你会感觉“方法还没开始，它就不干了”**

从执行链看，有四种压力叠在一起。

第一，环境构造、oracle controller、风险检测、选项选择、时机识别、跨机制泛化、论文叙事同时被当成一条串联流水线。前面任意一项不够成熟，后面就停止。最终用户感受到的是不停修入口，很少真正迭代方法。

第二，探索和确认被混用。探索应该允许在开发数据上看失败、修标签、改表达、收 learner-visited 状态；确认阶段才需要固定主要假设并保护测试。仓库有些规则在非常早的开发阶段就冻结了架构、H、α、特征、数据源和下一步行动。

第三，部分 gate 把“没有通过所有强条件”翻译成一条自动研究建议。这尤其体现在 [Phase 2.5B-R resolver](https://github.com/hanshuo-shuo/crash_bench/blob/cdc4802494a7c0e540fe80bfebc2dfc304af35d0/scripts/iclr27/run_advantage_router_resolver.py#L716)：tiny choice 过了、linear 没过，就优先返回 `CHOICE_CAPACITY_BOTTLENECK`。但同一结果中，OracleGate+LearnedChoice 的损失只有约 0.0725，LearnedGate+OracleChoice 的损失约 0.2928，显示 learned benefit gating 是更大的剩余损失来源。规则优先级不能取代对误差分解的判断。

第四，“冻结事实”与“冻结今后的决定”被写进同一套规范。到 ODUR 分支，README 是 staleness negative，AGENTS 的默认任务仍是 glass 论文，而 CURRENT 又保留 E16 旧主线及后加 gate notice。多个真相入口并存。下一次助手很容易执行历史禁令，而没有重新理解你正在提出的研究问题。

这些问题来自研究流程和实现，不能归结为你“不够坚持”，也不能凭此断言只要多跑一定能发论文。

**七、我建议下一步这样推进**

**先固定一个短期问题：现有反事实数据中的干预收益，能否被一个实现正确、输入合理的选项模型学到。** 用当前已有数据回答它，暂时不扩 benchmark，不把写论文需要的所有性质同时塞进模型准入条件。

建议以 ODUR tip 的分支引擎和数据工具为基础，保留 routing 分支的 glass 结果与 controller 作为恢复任务资产。不要整套覆盖 main，也不需要重做全部历史实验。

| 顺序 | 具体工作 | 完成后应能回答的问题 |
|---|---|---|
| 1 | 修正 B0 源支持口径，追加勘误；统一 CURRENT／研究入口，保留历史 README 和决策 | 到底是数据支持不足，还是指标定义错误？ |
| 2 | 修 ODUR 的状态×选项交互，保持 104-D 特征不变，比较原模型、交互模型、per-option head、DirectQ | 修实现本身能否改善？哪些 option／source 改善？ |
| 3 | 使用 train-only 或完整 cross-fit 预测比较开发效果；报告源级差值和区间 | 改善是否来自真实泛化，而非 development refit？ |
| 4 | 将点预测、收益阈值、保守校准分成单独消融；去掉最低干预配额 | 价值预测与 conservative wrapper 各损失了多少收益？ |
| 5 | 如果点预测仍很弱，再比较更有信息的冻结图像／VLA 特征和可部署历史信号 | 数据是否有信号，模型是否能看到？ |
| 6 | 在已验证 controller 的小范围 glass 任务上做 causal online timing，记录失败轨迹并做 targeted corrective data | 从“给定状态会选”到“在线找到时机”的具体差距在哪里？ |

这里的顺序是为了隔离变量，不是新增一串“差一点就永久停止”的门。开发阶段看到不理想结果，应产出可解释的错误归因和下一项最小修改；是否继续投入，由改进空间、剩余资源和论文目标决定。

动作恢复如果要重新做，应先在已有已认证样本上检查 teacher forcing 与闭环的差别、gripper 离散动作、controller phase／接管时间，再决定是否学习全部 7-D 动作。单个训练 pair 上连闭环都不能重现，通常首先是实现、目标表达或数据分布问题；做不到泛化则需要更多独立源。学习策略访问到的状态可用于有明确记录的 corrective-data 迭代；这种分布问题正是 [DAgger 原论文](https://proceedings.mlr.press/v15/ross11a.html) 讨论的背景。

论文工作可以同步整理已存在的 exact-state protocol、选项收益、强基线和序贯局限。最终是否成为方法论文，取决于修复后的证据。现在就把项目永久定性成 negative paper 过早；现在承诺主会一定可发也没有依据。已有 [SAFE](https://arxiv.org/abs/2506.09937) 等工作说明“读取 VLA 特征做 failure detection”已有充分先例，因此值得继续围绕干预后果与任务恢复提出具体方法，而不是仅靠旧 probe 故事或新术语。

**八、建议给后续 Codex 的工作约定**

以下是可直接使用的任务约定，不会自动修改仓库 AGENTS 或发起任务：

> 本项目当前允许在明确的开发数据和预算内进行研究迭代。请区分工程失败、数据有效性失败、当前模型失败、证据不足和论文主张不足。历史 NO-GO 只约束对应的冻结实验及其主张，不能自动禁止新版本探索。发现实现错误、不可用校准或指标口径问题时，先修复并用最小反例验证。模型对比保持 source 隔离并报告实际效用、事故、完成率和不确定性。数值目标未达到时，报告哪些信号存在、瓶颈在哪里以及最小下一步，不自行扩大 benchmark 或宣布整个方向永久结束。保留旧结果，新的开发修改使用新版本；任何新的确认性测试必须有独立于当前开发的新证据。

**九、本次交付和复算方法**

- 本报告：[PROJECT_REVIEW_ZH.md](/Users/hanshuo/Desktop/crash_bench/docs/audits/20260906/PROJECT_REVIEW_ZH.md)。
- 机器证据：[audit_evidence.json](/Users/hanshuo/Desktop/crash_bench/docs/audits/20260906/audit_evidence.json)，包括四个 D2 screen 及 D5/D8 源支持、决策标签、coverage 最优性检查、selector 实际调用、保存预测的 log-odds 验证、source bootstrap、角色 manifest、输入 SHA-256。
- 可复算脚本：[reproduce_audit.py](/Users/hanshuo/Desktop/crash_bench/docs/audits/20260906/reproduce_audit.py)。需要 NumPy；无需 Torch、GPU 或原始图像。

本次使用的 ODUR／routing 树分别是 Git archive 导出的只读研究快照，位于 `/private/tmp/crashbench-audit-20260906/odur` 和 `/private/tmp/crashbench-audit-20260906/routing`。临时目录删除后，可从上表精确提交重新导出。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/hanshuo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  /Users/hanshuo/Desktop/crash_bench/docs/audits/20260906/reproduce_audit.py \
  --odur-root /private/tmp/crashbench-audit-20260906/odur
```

验证：三个分支现有 `audit_repo.py` 均通过；新增数值审计成功执行；Git diff 格式检查通过。当前本地 Python 环境没有 pytest／PyTorch，未运行全部单元测试，也未重新训练或重放物理仿真。该限制不影响 F1 的逐行数据重算、F2 的真实 selector 调用或 F3 的结构证明与保存预测验证；涉及 controller 物理行为的判断采用仓库已有实验记录并明确标注。

本次没有改写历史的负结果，也没有将回顾性修正包装成新的确认性通过。需要改变的是现有结论的适用范围，以及接下来能否合理开展研究的工作方式。
