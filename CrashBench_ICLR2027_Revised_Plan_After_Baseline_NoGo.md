# CrashBench ICLR 2027 修订执行计划

## Phase 2 最强基线审计后的 Option-Support / Option-Ambiguity Pivot

**基线分支：** `codex/iclr27-exact-state-intervention-routing`
**审计记录提交：** `ac88aeffc990b8edfeab46b73d4aac9a332760e6`
**冻结审计协议提交：** `eed1feecb3bd298d870ab049976b7ac006d49633`
**当前决定：** 旧的 outcome-decomposition headline 已停止；不得直接启动 confirmatory cohort。
**新的唯一授权方向：** 先验证“训练 option support 不足 + benchmark option ambiguity 不足”是否确实解释失败，再决定是否恢复方法论文路线。

---

# 1. 这次 NO-GO 到底说明了什么

这次审计已经否定了下面这个论文主张：

> 在当前数据、当前训练 split 和当前 `lambda=1, eta=0` 偏好下，Outcome Router
> 明显优于强 scalar-risk、two-stage 和 direct-value baselines。

它没有证明下面这个更窄的命题为假：

> 当训练数据在 source 层面对 Base、Detour、Retreat 三种严格最优 option 都有
> 足够支持，而且 evaluation 不能被“识别 glass 后固定 Detour”解决时，
> option-conditioned consequence supervision 可能学会真正的多技能选择。

现有证据正好指出两个可检验的瓶颈：

1. **Option-support failure**
   - Train 中 Oracle Retreat 只有 3 个 decision，来自 2 个 source。
   - Development 中有 13 个 strict Retreat-optimal decision。
   - Outcome Router 在这 13 个中只正确选择 Retreat 1 次。
   - 因而当前失败很可能至少部分来自 train-source support 缺失。

2. **Option-ambiguity / shortcut failure**
   - 40/42 glass decisions 的 Base 会 catastrophe。
   - off-path 和 no-glass 各有 30/32 Base success，且 0 catastrophe。
   - Risk -> Best Fixed 在 calibration 上选择 Detour 后已经很强。
   - 因而当前 benchmark 很容易退化为：`识别 glass -> Detour；否则 Base`。

新的计划不能通过改 tolerance、换 headline point 或增加模型深度来“让结果过关”。
它必须直接检验以上两个解释。

---

# 2. 修订后的论文问题

## 2.1 新的研究问题

> **When does a VLA safety benchmark genuinely require intervention choice rather
> than hazard gating, and can source-supported counterfactual supervision learn
> that choice?**

中文：

> **什么情况下，VLA 安全系统真正需要在多种干预之间做选择，而不是识别危险后
> 执行一个固定动作；在训练源对各种 option 都有充分支持时，反事实结果监督能否
> 学会这种选择？**

## 2.2 新的潜在标题

方法门通过时：

> **Beyond Risk Gating: Support-Balanced Counterfactual Routing under Exact-State Option Ambiguity**

方法门未通过、但 benchmark 诊断成立时：

> **CrashBench-Ambiguity: Exact-State Evaluation of Intervention Choice for VLA Safety**

## 2.3 新故事的四步结构

1. **Risk gating can exploit a benchmark shortcut.**
   当危险状态几乎总对应同一个有效干预时，Risk -> Best Fixed 就足够。
2. **Intervention choice needs option ambiguity and source support.**
   同样高风险的状态必须包含 Detour-optimal 与 Retreat-optimal 两类结果，且每类都要由多个训练 source 支持。
3. **CrashBench-Ambiguity isolates the selector.**
   从相同 exact state 执行 Base、Detour、Retreat；用 matched geometry × timing 构造 within-source optimal-option flips。
4. **A support-balanced router is tested against the strongest alternatives.**
   比较 Risk -> Best Fixed、Risk + Two Stage、Direct-Q、Pairwise Advantage 和支持平衡的 outcome/ranking model。

旧的 P2–P3.2 sequential failure 只保留为边界：本文研究给定候选 decision state 时的 selector，不声称解决 from-reset trigger。

---

# 3. 新的阶段总图

| 阶段 | 目的 | 是否使用新 GPU rollout | 通过后授权 |
|---|---|---:|---|
| **2.5A** Existing-corpus support audit | 判断 pooled source support 能否解释现有失败 | 否 | 2.5B 或停止方法救援 |
| **2.5B** Source-cross-fitted support rescue | 在旧 20 个 source 上做严格 OOF 学习检验 | 否 | 小型机械 authoring screen |
| **2.5C** Mechanical ambiguity authoring | 找到能够稳定产生 Detour/Retreat 区分的 geometry × horizon cells | 是，极小规模 | 冻结 ambiguity protocol |
| **2.5D** External development screen | 冻结模型和 protocol 后，在新 source 上做一次外部 development gate | 是，小规模 | confirmatory cohort |
| **3** Confirmatory ambiguity cohort | 对唯一模型、唯一偏好、唯一预算做一次确认 | 是 | ICLR method paper |
| **Fallback-B** Benchmark paper | 若 option ambiguity 成立但 learned method 不胜出 | 视范围决定 | 扩任务/VLA 后投稿 |
| **STOP** | support 或 ambiguity 本身不成立 | 否 | 不再为 ICLR 强行救模型 |

原计划中的“第二任务、30+20 source confirmatory、conformal、深模型”全部后移到 2.5D 通过之后。

---

# 4. Phase 2.5A：现有 20-source option-support 审计

## 4.1 目的

在不运行任何新 rollout 的情况下，先回答：

1. 把现有 train/calibration/development 全部视为已暴露的 development corpus 后，三种严格最优 option 分别来自多少独立 source？
2. Retreat signal 是否集中在极少 source、极少 horizon 或单一 condition？
3. 在同一 source 内，是否真的出现最优 option 随 timing/geometry 改变？
4. fixed Detour 能恢复多少 Oracle value；理论上还剩多少“必须选择”的价值？

## 4.2 新文件

```text
configs/iclr27/option_support_audit.yaml
scripts/iclr27/audit_option_support.py
tests/iclr27/test_option_support_audit.py
docs/iclr27/OPTION_SUPPORT_AUDIT.md
results/iclr27/option_support_audit_<commit>_<utc>/
```

## 4.3 严格最优 option 定义

主偏好继续冻结为：

```text
lambda = 1
eta = 0
U(success) = +1
U(safe_noncompletion) = 0
U(catastrophe) = -1
```

对每个 decision：

```text
A*(i) = argmax_a U_i(a)
```

但必须把以下标签分开：

- `strict_base`
- `strict_detour`
- `strict_retreat`
- `detour_retreat_tie`
- `base_tie`
- `no_good_option`

**不能再用固定 tie order 把 Detour/Retreat tie 计作 Detour support。**
主 choice-learning 指标只使用 strict labels；ties 单独报告。

## 4.4 必做表格

1. decision count by `split × condition × horizon × strict label`；
2. distinct-source count by strict label；
3. 每个 source 是否包含 Base/Detour/Retreat strict state；
4. within-source optimal-option transition matrix；
5. Detour/Retreat strict advantage 分布；
6. 最强 fixed option 的 Oracle value recovered；
7. condition-only、horizon-only、condition+horizon diagnostics；
8. risk-score bins 内 strict optimal-option entropy；
9. Base risk 相近但 strict intervention 不同的 matched pairs。

## 4.5 Support 指标

定义：

```text
SourceSupport(a) = number of distinct sources containing at least one
                   strict-a decision with advantage >= gamma
```

主 `gamma=0.5`。由于当前 utility 离散，这会排除 ties，而不会删除真实 1-point advantage。

## 4.6 2.5A 门槛

### PASS-SUPPORT

现有 20-source corpus 至少满足：

- `SourceSupport(Base) >= 8`
- `SourceSupport(Detour) >= 6`
- `SourceSupport(Retreat) >= 6`
- 至少 4 个 source 存在 within-source `Detour <-> Retreat` 或 `Base <-> intervention` strict flip
- Best fixed non-Base option 的 Oracle value recovered `<= 0.80`

通过后直接进入 2.5B pooled cross-fit。

### SUPPLEMENT-SUPPORT

若 Retreat distinct-source support 在 3–5：

- 允许一个 **training-only source supplement**；
- 不允许启动 test 或 confirmatory；
- 新 source 在任何 outcome 前通过 hash 分配为 `support_train` 或 `support_cal`；
- 收集直到 Retreat strict support 达到 8 个 source，或达到预声明的最大 18 个 attempted source；
- 所有 attempted source 和失败原因都保留。

### STOP-SUPPORT

若 pooled 20 sources 的 strict Retreat support少于 3 个 source，或 Best Fixed 已恢复 `>0.90` Oracle value：

- 当前任务/option library 不足以支撑多-option方法论文；
- 不再训练更复杂 Router；
- 直接转 benchmark authoring，或停止 ICLR 2027 method submission。

---

# 5. Phase 2.5B：现有 corpus 上的 source-cross-fitted support rescue

## 5.1 为什么要先做这一步

当前 split 只有 5 个 train source。直接收集新 test 无法判断失败是模型问题还是 split-support 问题。

因此将现有 20 个 source 全部视为 **development-only corpus**，做严格 source OOF：

- 每个 held-out source 从未参与 PCA、标准化、模型 fitting 或 threshold calibration；
- 所有模型选择仍然是 development；
- fresh n=8/n=13 outcomes 不得加载。

## 5.2 推荐交叉拟合

主方案：**20-fold leave-one-source-out**。

对每个 outer held-out source：

1. 剩余 19 source 按固定 hash 排序；
2. 14 source 用于 fitting；
3. 5 source 用于 calibration；
4. PCA、scale、model、threshold 全部只在 14/5 内完成；
5. 对 held-out source 只预测一次；
6. 合并 20 个 OOF source predictions。

每个 fold 必须记录三种 strict class 的 train-source support。任何 fold 若缺某 strict class，标记为 `support_invalid`，不能静默继续。

## 5.3 新文件

```text
configs/iclr27/support_crossfit.yaml
scripts/iclr27/train_support_crossfit_suite.py
scripts/iclr27/analyze_support_crossfit_suite.py
scripts/iclr27/plot_support_crossfit_suite.py
tests/iclr27/test_support_crossfit_suite.py
results/iclr27/support_crossfit_<commit>_<utc>/
docs/iclr27/SUPPORT_CROSSFIT_RESULT.md
```

## 5.4 新的公平 calibration 规则

旧 protocol 使用“最接近 target rate”，导致 Outcome Router 在 development 实际达到 65.76%，超过 60% target。

新的所有方法统一使用约束式 calibration：

```text
maximize source-balanced calibration utility
subject to source-balanced intervention rate <= rho
```

主 `rho=0.60`。若多个 threshold utility 相同：

1. 更低 catastrophe；
2. 更低 intervention；
3. 更高 threshold。

这条规则对 Risk、TwoStage、Direct-Q、Pairwise 和 Outcome 方法完全相同，且在运行前冻结。

## 5.5 必做模型

### M0 Current Outcome Router

当前 source-balanced three-outcome softmax，作为原方法。

### M1 Support-Balanced Outcome Router

保留 outcome NLL，但训练权重同时满足：

- 每个 source 总权重相等；
- strict Base/Detour/Retreat 三个 decision stratum 总权重相等；
- tie/no-good 作为独立辅助 stratum，不进入 strict-choice平衡；
- 单个 decision weight 必须 cap，避免极小类无限放大。

### M2 Support-Balanced Outcome + Ranking Router（首选候选）

训练目标：

```text
L = L_outcome + beta * L_pairwise_rank
```

其中 `L_pairwise_rank` 只对真实 utility 不同的 option pairs 计算：

```text
log(1 + exp(-(Uhat(a)-Uhat(b)) * sign(U(a)-U(b)) / tau))
```

要求：

- pair 以 source 和 strict-optimal class 平衡；
- tie pairs 不赋予人为顺序；
- outcome probabilities 仍然输出并接受 calibration 检查；
- `beta` 只允许预声明小网格 `{0.25, 1.0}`，在 inner source folds 选择；
- 不增加深层 encoder，不新增 temporal model。

论文若采用该方法，名称可暂定：

> **Support-Balanced Counterfactual Outcome Router (SB-COR)**

它不再声称“纯 outcome decomposition 自动优越”，而是明确解决 option-support imbalance。

### 强 baseline

必须原样保留：

- Base
- Always Detour
- Always Retreat
- Risk -> Retreat
- Risk -> Detour
- Risk -> Best Fixed
- Risk + Two Stage
- Direct Choice
- Direct-Q
- Pairwise Advantage
- Hidden-only
- State/action-only

新增公平 baseline：

- Support-balanced Direct-Q
- Support-balanced Risk + Two Stage
- Horizon-only diagnostic（不可部署）
- Condition-only diagnostic（不可部署）

## 5.6 新的 primary metrics

### Natural-distribution metrics

- source-macro utility；
- task success；
- catastrophe；
- intervention；
- harmful intervention；
- Oracle value recovered。

### Strict-choice metrics

只在 strict beneficial intervention states 上：

- Detour recall；
- Retreat recall；
- Detour/Retreat balanced accuracy；
- strict-choice macro-F1；
- wrong-intervention catastrophe；
- fixed-option regret。

### Base-preservation metrics

- Base-optimal recall；
- control task success；
- unnecessary intervention；
- Base success overridden。

## 5.7 2.5B 门槛

### GO-SIGNAL

至少一个非特权 learned method 必须同时满足：

1. OOF strict Detour recall `>= 0.55`；
2. OOF strict Retreat recall `>= 0.55`；
3. OOF Base-optimal recall `>= 0.80`；
4. 相对 Risk -> Best Fixed：
   - source-macro utility `>= +0.03`；
   - intervention rate 不高于 `+0.05`；
   - catastrophe 不高于 `+0.02`；
5. 至少 12/20 source 的 paired utility difference 非负；
6. 相对 support-balanced TwoStage 或 Direct-Q，至少有一个：
   - utility `>= +0.02`，或
   - strict-choice macro-F1 `>= +0.08`，且 natural utility 不降低超过 0.01。

通过：冻结 candidate architecture，进入 2.5C。

### GO-VALUE-ONLY

若 Direct-Q / Pairwise 明显胜出，但 outcome variants 不胜出：

- 论文改为 **multi-option intervention value learning**；
- Outcome decomposition 仅为 baseline/ablation；
- 仍可进入 2.5C，但不得恢复 outcome headline。

### BENCHMARK-PIVOT

若所有 learned selector 都无法达到 Detour/Retreat recall 0.55，但 Oracle/future-free observable diagnostic 能明显区分：

- 方法线暂时停止；
- 说明当前 representation 或模型不足；
- 只继续 author option-ambiguity benchmark，不运行 confirmatory method test。

### STOP-RESCUE

若 pooled/cross-fitted 后 Risk -> Best Fixed 仍恢复 `>=0.85` Oracle value，或没有方法在 strict Retreat 上超过随机水平：

- 不再补数据、调 lambda、加 MLP/Transformer；
- 当前任务不能支撑 ICLR 方法主张。

---

# 6. Phase 2.5C：机械 Option-Ambiguity Authoring Screen

此阶段只有 2.5B 的 GO-SIGNAL 或 GO-VALUE-ONLY 才能执行。

## 6.1 核心目标

构造这样的 matched decision states：

- Base catastrophe risk 都高；
- 但一部分状态 Detour 能完成任务；
- 另一部分状态 Detour 已不可行而 Retreat 能安全退出；
- controls 中 Base 应保持；
- condition 或 horizon 单独不能确定正确 option。

## 6.2 2×2 机械设计

优先尝试正交化：

```text
Detour clearance: wide / constrained
Decision timing:  early / late
```

每个 source 尽量贡献：

1. wide + early on-path；
2. wide + late on-path；
3. constrained + early on-path；
4. constrained + late on-path；
5. matched off-path control；
6. matched no-glass control。

预期但不强制的机制：

| Cell | 期望结构 |
|---|---|
| wide + early | Detour-optimal 较多 |
| wide + late | Retreat-optimal 增加 |
| constrained + early | Retreat 或 no-good，打破纯 horizon shortcut |
| controls | Base-optimal |

`wide/constrained` 必须由 workspace、lane clearance 或固定 placement parameters 定义，不能由 Router score 定义。

## 6.3 Authoring screen 与模型 evaluation 分离

### Screen A：mechanical authoring only

- 4–6 个已声明为 exposed-development 的 source；
- 可以查看所有 option outcomes；
- 只用于选择 geometry/horizon cells；
- 禁止报告 learned method performance；
- 输出完整 candidate grid，不只保留成功 cell。

### Freeze

冻结：

- geometry cells；
- horizon cells；
- option controller configs；
- source eligibility；
- failure/exclusion rules；
- maximum supplement rule；
- all artifact hashes。

### Screen B：external development gate

- 12 个全新 source；
- frozen model、frozen threshold、frozen cells；
- 不允许看中间结果后更换 cell；
- 若机械有效 source 不足，最多按预声明规则补 6 个 source；
- supplement 由 eligibility count 触发，不由 method comparison 触发。

## 6.4 新文件

```text
configs/iclr27/ambiguity_authoring.yaml
configs/iclr27/ambiguity_external_dev.yaml
crashbench/ambiguity_protocol.py
scripts/iclr27/author_option_ambiguity.py
scripts/iclr27/collect_option_ambiguity.py
scripts/iclr27/analyze_option_ambiguity.py
setup/iclr27_ambiguity_authoring.sbatch
setup/iclr27_ambiguity_external_dev.sbatch
setup/submit_iclr27_ambiguity_pivot.sh
tests/iclr27/test_ambiguity_protocol.py
docs/iclr27/OPTION_AMBIGUITY_PROTOCOL.md
```

## 6.5 Exact-state 与公平性契约

所有 options：

- 完全相同 simulator/controller/RNG/observation state；
- 相同 remaining environment horizon；
- 相同 catastrophe predicate；
- 相同 task success predicate；
- 记录实际 steps、path length、control effort、latency；
- structured Detour 的 privileged geometry 明确记录；
- source 是统计单位。

## 6.6 Ambiguity dataset gate

Screen B 必须满足：

1. strict Base、Detour、Retreat 每类至少来自 6 个独立 source；
2. 至少 4 个 source 内部存在两个 on-path states 的 strict optimal option 不同；
3. strict beneficial states 中，最大固定 intervention 的占比 `<= 0.70`；
4. Risk -> Best Fixed 恢复的 Oracle available value `<= 0.75`；
5. condition-only strict-choice balanced accuracy `<= 0.70`；
6. horizon-only strict-choice balanced accuracy `<= 0.75`；
7. ties 单独报告，不得填入稀缺 class；
8. no-good-option 比例 `<=0.30`，否则 option library 本身太弱。

失败时不运行 confirmatory。允许回到 Screen A 修改一次 mechanical design，但不得用 Screen B 的 method result选择参数。第二次仍失败则停止 ICLR method line。

---

# 7. Phase 2.5D：冻结模型后的外部 development gate

同一个 Screen B 同时提供一次 external method gate，但只有在 ambiguity dataset gate 通过后才解释方法结果。

## 7.1 候选模型冻结

来自 2.5B，只能冻结一个 primary：

- SB-COR；或
- Support-balanced Direct-Q；或
- 其他在 2.5B 预声明 suite 中胜出的模型。

不能在 Screen B 上重新选择模型、beta、lambda、feature、threshold 或 target rate。

## 7.2 方法 GO 门

Primary 相对 Risk -> Best Fixed 必须满足：

- source-macro utility `>= +0.05`；
- task success 不降低超过 0.02；
- catastrophe 不增加；
- intervention 不增加超过 0.05；
- strict Detour recall `>=0.60`；
- strict Retreat recall `>=0.60`；
- Base-optimal recall `>=0.85`；
- 至少 8/12 source paired utility 非负。

相对最强 learned baseline（TwoStage/Direct-Q/Pairwise 中最佳者）：

- utility `>= +0.02`，或
- strict-choice macro-F1 `>= +0.08` 且 natural utility 非劣于 -0.01。

这是 authorize-confirmatory gate，不要求 12-source development p-value 显著；要求 effect、方向和 source consistency。

## 7.3 结果分支

### METHOD-GO

- 允许主任务 confirmatory；
- 论文讲 support-balanced multi-option selection；
- 原 n8 E16 仅作历史 pilot/appendix。

### VALUE-GO

- Direct-Q 胜出；
- 论文讲 exact-state multi-option value learning；
- outcome decomposition 不再作为贡献。

### BENCHMARK-GO

- ambiguity protocol 成立，但 learned methods 都未胜出；
- 论文可转 benchmark/diagnostic，但 ICLR 主会需要再扩至少第二任务或第二 VLA；
- 不得把失败模型包装成正方法。

### STOP

- ambiguity 和 learned method 都失败；
- 冻结该线，不再做 confirmatory。

---

# 8. Phase 3：只有 METHOD-GO / VALUE-GO 后才执行的新确认实验

## 8.1 主确认集

建议：

- 24–30 个全新 independent source；
- frozen ambiguity cells；
- 唯一 primary model；
- `lambda=1, eta=0`；
- intervention budget `rho=0.60`；
- 唯一 threshold/calibration protocol；
- Base、Risk -> Best Fixed、Risk + Two Stage、Direct-Q/Pairwise、primary、Oracle。

旧 n8/n13 不能加入 primary confirmatory statistic。

## 8.2 Primary hypotheses

### H1：Beyond fixed risk mapping

Primary 在 `rho<=0.60` 下相对 Risk -> Best Fixed：

- source-macro utility 提高；
- catastrophe non-inferior；
- intervention budget 不超限。

### H2：Correct intervention choice

在 strict beneficial states：

- Detour/Retreat balanced accuracy 高于 Risk -> Best Fixed；
- wrong-skill catastrophe 降低。

### H3：Base preservation

在 strict Base-optimal/control states：

- Base retention 达到冻结下限；
- unnecessary intervention 不高于冻结上限。

## 8.3 统计

- source-block exact randomization；
- shared source bootstrap；
- raw source-paired table；
- natural-distribution 与 strict-class-macro 两套结果；
- 一个 primary operating point；
- preference grid 只作 secondary，不重新选择 headline。

## 8.4 第二任务

第二任务不再是 confirmatory 前置条件。

只有以下均成立才做：

- 主 confirmation 已完成；
- controller adaptation 在机械 smoke 中通过；
- 9 月 8 日前仍有执行窗口；
- 不影响主文完成。

否则第二任务作为 appendix pilot 或后续 CoRL/TMLR 扩展。

---

# 9. 数学部分的修订

## 9.1 保留 potential outcomes

```text
Z: complete simulator/controller/RNG state
X = phi(Z): deployable representation
A = {Base, Detour, Retreat}
Y_Z(a): exact-state option outcome
```

## 9.2 增加 option-support 条件

定义 strict-support：

```text
Supp(a) = |{source s : exists decision i in s,
                         a is strictly optimal,
                         advantage_i(a) >= gamma}|
```

论文明确：没有 source-diverse support 时，经验风险最小化可能把稀缺 option 当成从不应选择；这正是旧 split 的实际问题。

## 9.3 增加 benchmark ambiguity

定义 risk-only information gap：

```text
Delta_risk = E[max_a V(X,a)]
             - E[max_a E[V(X,a) | r_base(X)]]
```

定义 fixed-mapping gap：

```text
Delta_fixed = V(Oracle selector)
              - max_{a in {Detour, Retreat}} V(RiskGate -> a)
```

只有 `Delta_fixed` 足够大，benchmark 才真正测试 intervention choice。

## 9.4 方法不再依赖“outcome 天然优越”

SB-COR 的主张是：

> outcome calibration 提供 consequence model；source/class-balanced pairwise
> ranking 修复稀缺 option 的 decision support。

需要由 stronger baselines 实证，而不是通过定理宣称。

---

# 10. 修订后的 ICLR 故事

## 若 METHOD-GO

主张：

> Common safety evaluations can be solved by hazard gating followed by one
> dominant intervention. CrashBench-Ambiguity constructs exact-state matched
> decisions where equally risky states require different responses. A
> source-support-balanced consequence-and-ranking router improves utility and
> correct skill selection over risk-to-fixed, two-stage, and direct-value
> baselines while preserving Base behavior.

贡献：

1. 最强基线审计揭示 risk-to-fixed shortcut；
2. exact-state option-ambiguity protocol；
3. source-support-aware consequence/ranking method；
4. selector 与 sequential trigger 的明确边界。

## 若 VALUE-GO

把第三条改为：

> source-supported direct intervention-value learning

不再强调 outcome decomposition。

## 若 BENCHMARK-GO

主张：

> existing routing evaluations confound hazard detection with option choice;
> CrashBench-Ambiguity isolates the latter and exposes systematic failures of
> current selectors.

此路线需要更广 task/VLA 覆盖，不应只靠一个模型正结果。

---

# 11. 截止日前的压缩日程

## 8 月 29–30 日

- 完成 2.5A option-support audit；
- 完成 2.5B LOSO cross-fit 实现与 CPU run；
- 得到 GO-SIGNAL / GO-VALUE-ONLY / STOP。

## 8 月 31 日–9 月 1 日

仅在 GO 后：

- 完成 Screen A mechanical authoring；
- 冻结 2×2 ambiguity cells；
- 完成 exact-state tests。

## 9 月 2–4 日

- 运行 12-source Screen B；
- 得到 ambiguity gate 和 external development method gate。

## 9 月 5 日

硬决定：

- METHOD/VALUE GO：启动 confirmatory；
- BENCHMARK GO：决定是否有能力扩范围；
- STOP：停止 ICLR 主会方法线。

## 9 月 5–10 日

仅 GO：

- 24–30 source confirmatory collection；
- sealed analysis；
- 不做 test frontier 选点。

## 9 月 10–17 日

- 主文、主图、主表；
- related work；
- appendix provenance；
- anonymous release。

## 9 月 18 日

- 提交真实 abstract；
- 冻结作者列表。

## 9 月 19–24 日

- 完成全文、复现说明、AI-use、匿名检查；
- 不再启动会改变主故事的新实验。

## 9 月 25 日

- full paper deadline。

---

# 12. 立即交给 Codex 的任务文档

```text
你正在仓库 hanshuo-shuo/crash_bench 的分支
codex/iclr27-exact-state-intervention-routing 上工作。

先阅读：
- AGENTS.md
- QUEST_WORKFLOW.md
- setup/README.md
- docs/iclr27/MASTER_PLAN.md
- docs/iclr27/BASELINE_AUDIT_RESULT.md
- configs/iclr27/baseline_suite.yaml
- scripts/iclr27/train_router_baseline_suite.py
- scripts/iclr27/analyze_router_baseline_suite.py

背景：phase-2 strongest-baseline audit 对当前 outcome-decomposition headline
给出 NO-GO。不得改 tolerance、不得加载 fresh n8/n13 outcomes、不得启动新
confirmatory cohort。下一步只允许 option-support / option-ambiguity pivot。

本任务只完成 Phase 2.5A 和 2.5B，优先 CPU-only，不提交新的 outcome-bearing
GPU rollout。

一、Option-support audit
创建：
- configs/iclr27/option_support_audit.yaml
- scripts/iclr27/audit_option_support.py
- tests/iclr27/test_option_support_audit.py
- docs/iclr27/OPTION_SUPPORT_AUDIT.md

在现有 full capture 的全部 20 个已暴露 source 上：
1. 用 lambda=1, eta=0 定义 strict_base / strict_detour / strict_retreat；
2. Detour/Retreat ties 必须独立成类，不能按 tie order 计入 Detour support；
3. 报告每类 decision 数与 distinct-source support；
4. 报告每个 source 的 optimal-option transition；
5. 报告 condition-only、horizon-only、condition+horizon diagnostics；
6. 报告 risk bins 内 optimal-option entropy；
7. 报告 Risk->BestFixed 的 Oracle value recovered 与 fixed-mapping gap；
8. 生成 machine-readable gate。

二、20-source source-cross-fitted audit
创建：
- configs/iclr27/support_crossfit.yaml
- scripts/iclr27/train_support_crossfit_suite.py
- scripts/iclr27/analyze_support_crossfit_suite.py
- tests/iclr27/test_support_crossfit_suite.py
- docs/iclr27/SUPPORT_CROSSFIT_RESULT.md

使用 20-fold leave-one-source-out：
- outer held-out source 不得进入 PCA、scale、fit、calibration；
- 剩余 source 按固定 hash 产生 fit/cal split；
- 每 fold 记录 strict class source support；
- fresh test outcomes 禁止读取。

所有方法使用相同 intervention constraint：
maximize calibration source-macro utility subject to rate <= 0.60。

实现并比较：
- current Outcome Router；
- source/strict-class-balanced Outcome Router；
- Outcome + source-balanced pairwise-ranking auxiliary；
- Risk->BestFixed；
- support-balanced Risk+TwoStage；
- Direct-Q；
- support-balanced Direct-Q；
- Pairwise Advantage；
- hidden-only；
- state/action-only。

主报告：
- source-macro natural utility/success/catastrophe/intervention；
- strict Base/Detour/Retreat recall；
- Detour/Retreat balanced accuracy 和 macro-F1；
- harmful intervention；
- Oracle value recovered；
- per-source paired differences；
- exact sign-flip 和 shared source bootstrap。

按修订计划中的 GO-SIGNAL / GO-VALUE-ONLY / BENCHMARK-PIVOT /
STOP-RESCUE 规则生成 gate_decision.json。

禁止：
- 新 MLP/GRU/Transformer；
- 使用 horizon/condition 作为 deployable feature；
- 用 tie 扩充 Retreat/Detour 类；
- 用 development OOF 结果改 lambda=1 或 rho=0.60；
- 读取任何 fresh n8/n13 outcome；
- 覆盖旧 baseline audit。

输出使用：
results/iclr27/option_support_audit_<commit>_<utc>/
results/iclr27/support_crossfit_<commit>_<utc>/

完成后运行：
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests -q
PYTHONDONTWRITEBYTECODE=1 python scripts/audit_repo.py
git diff --check

最后只提交以下之一：
GO-SIGNAL / GO-VALUE-ONLY / BENCHMARK-PIVOT / STOP-RESCUE，
并说明下一步是否获准运行 Screen A。不要把 INCONCLUSIVE 自动解释为 GO。
```

---

# 13. 最终决策原则

这份修订计划所谓“能走通”，不是保证 Outcome Router 一定赢，而是保证每种结果都有科学上合法的出口：

- **support rescue 成功**：进入真正 option-ambiguity 数据和 confirmatory；
- **Direct-Q 成功**：改讲 multi-option value；
- **benchmark 成功、方法失败**：改讲 benchmark/diagnostic；
- **support 和 ambiguity 都失败**：及时停止，不再浪费 confirmatory rollout。

唯一不能再走的路线是：在当前数据上继续换 threshold、换 lambda、加深模型，然后仍用旧 `+45.83pp vs Risk->Retreat` 作为主结果。
