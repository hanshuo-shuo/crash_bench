# CrashBench → ICLR 2027：Codex 详细执行计划

**目标分支基线**：`codex/glass-recovery-d0-fprime-rescue`
**建议新分支**：`codex/iclr27-exact-state-intervention-routing`
**建议论文题目**：**CrashBench: Exact-State Potential Outcomes for Selective VLA Intervention**
**核心问题**：给定一个候选干预时刻，安全层不只要判断“危险吗”，还要判断“继续、绕行或撤退分别会导致什么结果，以及哪一种接管真正改善任务—安全权衡”。

---

## 0. 全局执行契约

### 0.1 Codex 开始前必须阅读

1. `AGENTS.md`
2. `QUEST_WORKFLOW.md`
3. `setup/README.md`
4. `docs/CURRENT.md`
5. `docs/PAPER_PLAN.md`
6. `docs/CLAIMS.md`
7. `docs/FRESH_COUNTERFACTUAL_ROUTER_PROTOCOL.md`
8. `docs/COUNTERFACTUAL_ROUTER_MAIN_RESULT.md`
9. `docs/REPRODUCIBILITY.md`

### 0.2 不可违反的规则

- 不覆盖任何 frozen result。
- 所有新结果写到带 commit 和 UTC 时间戳的新目录。
- 所有 GPU 运行来自 clean、tested、published commit。
- 独立统计单位始终是 source；frame、condition、horizon、branch 不能冒充独立样本。
- 新 test cohort 只能使用一次；不得在 test outcome 上选模型、阈值、λ、η、κ、任务或图中“最好点”。
- 任何实验失败都保留并进入 manifest；不得静默删行、删 source 或换定义。
- Oracle 只能作为不可部署上界。
- privileged Detour 必须明确标记，不能写成端到端 learned recovery。
- 不重新开启旧 P3 recovery-window 调参线；新的实验只服务于 fixed-decision intervention routing。
- 所有 test eligibility 只能由 Base/机械契约决定，不能依赖 Router 与 baseline 的相对结果。
- 每一阶段结束必须执行：
  ```bash
  PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests -q
  PYTHONDONTWRITEBYTECODE=1 python scripts/audit_repo.py
  git diff --check
  ```

### 0.3 新目录建议

```text
configs/iclr27/
docs/iclr27/
paper/
results/iclr27/
scripts/iclr27/
tests/iclr27/
```

### 0.4 新证据等级

- `protocol-frozen`：协议、source IDs、任务、阈值、主要假设均已 hash。
- `development-only`：可用于模型/协议选择，不能作最终确认。
- `confirmatory-test`：只执行一次，禁止选择性报告。
- `exploratory-test`：明确标记，不能与 confirmatory 结论混写。
- `negative-closeout`：失败但可复核，禁止再次调同一 test。

---

# 第一阶段：建立 ICLR 论文真相源

## 1.1 任务

创建：

- `docs/iclr27/MASTER_PLAN.md`
- `docs/iclr27/CLAIM_LEDGER.md`
- `docs/iclr27/RELATED_WORK_MATRIX.md`
- `docs/iclr27/REVIEWER_RISK_AUDIT.md`
- `results/iclr27/manifest.json`

## 1.2 CLAIM_LEDGER 格式

每条 claim 必须包含：

```yaml
claim_id:
paper_wording:
status: supported | conditional | unsupported
independent_unit:
n_sources:
cohort:
method_hash:
protocol_hash:
result_files:
statistical_test:
limitations:
forbidden_stronger_wording:
```

## 1.3 首轮 reviewer-risk 审计必须至少列出

1. Binary Risk 只接 Retreat，baseline 结构性偏弱。
2. 没有 Risk→Detour、Risk→best fixed option、risk+option 两阶段 baseline。
3. 没有 Direct Choice、Direct Utility/Q、Pairwise Advantage baseline。
4. n=8 confirmation 太小；n=13 合并后无单一 all-criteria point。
5. `λ=1,target=0.6` 是在完整 test frontier 中识别出的展示点，不是预先冻结的单点。
6. 45 个 frontier 点涉及 post-selection / multiple comparison 风险。
7. Fresh T−20 cohort 中 Base outcome 基本由 condition 决定，可能退化为 hazard/control 分类。
8. Detour 使用 privileged geometry。
9. 不同 option 的时间步预算、路径长度和控制代价可能不公平。
10. Hazard Prompt 从 reset 运行，不是 exact-anchor same-state selector baseline。
11. 当前 δ 是 calibration margin / switching penalty，不是统计意义的 lower confidence bound。
12. 预测 safe noncompletion，但主 utility 令其权重为 0，需给出一般化效用或敏感性分析。
13. source-cluster bootstrap 只有 8 或 13 个 cluster，必须补 exact paired inference。
14. 只有单任务、单 VLA、单主 hazard family。
15. 同期 CoWAM、CheckVLA、SAFE 等已覆盖 selective intervention、sequential verification 或 hidden-state failure detection；论文必须重新定位。

## 1.4 阶段交付标准

- 所有当前 paper-facing 数字能从 claim ledger 追到 frozen artifact。
- 所有不支持的强表述被列为 forbidden。
- 不改任何 frozen result。
- 输出一个 1 页的“当前项目是否值得继续 ICLR”的判断。

---

# 第二阶段：最强 baseline GO/NO-GO 审计（最高优先级）

这是整条 ICLR 线的第一个硬门。先完成它，再决定是否做更多 rollout。

## 2.1 新脚本

创建：

```text
scripts/iclr27/train_router_baseline_suite.py
scripts/iclr27/analyze_router_baseline_suite.py
scripts/iclr27/plot_router_baseline_suite.py
tests/iclr27/test_router_baseline_suite.py
configs/iclr27/baseline_suite.yaml
```

## 2.2 所有 baseline 必须共享

- 相同 source split。
- 相同 train-only PCA。
- 相同 hidden/robot/action feature contract。
- 相同训练 source weighting。
- 相同 calibration source。
- 相同 candidate option set。
- 相同 test outcomes。
- 尽可能相同模型容量；容量不同时报告参数量。
- 每个 baseline 的超参数只能在 train/calibration/development 上确定。

## 2.3 必做 baseline

### B0. Base

永不接管。

### B1. Always Detour

固定使用 Detour。

### B2. Always Retreat

固定使用 Retreat。

### B3. Binary Risk → Retreat

复现当前 baseline。

### B4. Binary Risk → Detour

同一个 Base-catastrophe score 和同一个 rate calibration，但报警后固定执行 Detour。

### B5. Binary Risk → Best Fixed Option

在 calibration source 上，在 `{Detour, Retreat}` 中选择 source-balanced utility 更高的固定技能；test 前冻结。

### B6. Risk Gate + Two-Stage Option Classifier

- 第一阶段：预测 Base catastrophe / intervention-needed。
- 第二阶段：仅在 calibration/train 的 beneficial-intervention 样本上预测 Detour 或 Retreat。
- test 时先 gate，再选 option。
- 必须避免用 test Oracle label。

### B7. Direct Oracle-Choice Classifier

直接预测 `{Base, Detour, Retreat}` 中的 realized-utility 最优 option。

处理 tie 的预先规则：

1. Base 优先；
2. 其次较低 intervention cost；
3. 再按固定 option order。

### B8. Direct Utility / Q Regression

为每个 option 直接预测 realized utility。
至少做：

- single λ direct-Q；
- multi-λ separate-Q；
- shared model + λ 输入（可选）。

### B9. Pairwise Advantage Router

直接预测：

```text
A_detour = U(Detour) - U(Base)
A_retreat = U(Retreat) - U(Base)
```

仅在最大预测 advantage 大于 switching cost 时接管。

### B10. Outcome-Decomposed Router

当前三类 outcome softmax 方法。

### B11. Outcome Router without VLA hidden

仅 robot state + nominal action。

### B12. Outcome Router hidden-only

仅 frozen VLA hidden。

### B13. Simple observable geometry

若 capture 中可获得且不泄漏未来：

- end-effector position；
- nominal action；
- hazard-relative distance；
- corridor/clearance；
- time-to-contact proxy。

明确标记 oracle geometry 与 deployable geometry。

## 2.4 必做分析

每个方法报告：

- task success；
- catastrophe；
- safe noncompletion；
- intervention；
- unnecessary intervention；
- harmful intervention；
- missed beneficial intervention；
- Oracle regret；
- Oracle value recovered；
- calibration NLL/Brier/ECE；
- predicted advantage calibration；
- parameter count；
- inference latency。

统计：

- source-level shared bootstrap；
- source-block exact sign-flip/randomization test；
- discordant source count；
- raw numerator/denominator；
- per-condition、per-horizon、per-source 表；
- 不把 45 个 grid 点当 45 个独立发现。

## 2.5 强制 GO/NO-GO 逻辑

### GO-A：Outcome decomposition 真正成立

同时满足：

1. Outcome Router 在 development/source-held-out 上优于 Risk→Best-Fixed。
2. Outcome Router 不被 Direct Choice、Direct-Q、Pairwise Advantage 全面覆盖。
3. outcome calibration 或 preference reweighting 提供清晰额外价值。
4. 提升不是仅来自更高 intervention rate。
5. 至少一个同 rate / 同 cost 比较方向稳定。

此时可以写“outcome decomposition provides a useful inductive bias and preference-reweightable consequence model”。

### GO-B：只有 multi-option value learning 成立

若 Outcome Router 与 Direct-Q/Pairwise 类似，但均明显优于 scalar risk：

- 把贡献改为 **multi-option intervention value learning**；
- outcome decomposition 只作为一种实现；
- 不声称 outcome head 本身优于所有直接决策方法。

### PIVOT-C：Risk→Best-Fixed 已追平 Router

若 Risk→Detour 或 Risk→Best-Fixed 在同 rate 下基本追平：

- 当前 fresh cohort 没有证明“which intervention”问题；
- 必须建立 option-ambiguity cohort；
- 旧 +45.8pp 只能解释为“Retreat 映射过于保守”，不能作为方法 headline。

### STOP-D：Direct/simple geometry 全面追平

若机器人状态或简单几何 baseline 全面追平：

- 不把 VLA representation 作为核心；
- 论文转为 benchmark/protocol/diagnostic；
- 若 benchmark 也不扩展，则不以当前形式冲 ICLR main track。

## 2.6 阶段产物

```text
results/iclr27/baseline_audit_<commit>_<utc>/
  manifest.json
  all_predictions.npz
  all_choices.csv
  overall_metrics.csv
  per_source_metrics.csv
  exact_tests.json
  calibration.json
  frontier.csv
  figures/
docs/iclr27/BASELINE_AUDIT_RESULT.md
```

---

# 第三阶段：重写问题定义与数学方法

## 3.1 正确的潜在结果定义

令完整 simulator/controller/RNG 状态为 `Z`，部署可见特征为：

```text
X = φ(Z) = frozen VLA hidden + robot state + nominal action
```

option 集合：

```text
A = {Base, Detour, Retreat}
```

每个 option 的 potential outcome：

```text
Y(a) ∈ {Success, SafeNoncompletion, Catastrophe}
```

Exact-state branching 在 simulator 中观察同一 `Z` 下所有 `Y(a)`。
因此这是 full-information interventional supervision，不需要 observational causal inference 中的 propensity/ignorability 假设。论文不得暗示真实世界中可以同时观测所有 counterfactual。

## 3.2 一般效用

实现并在论文中使用：

\[
V_{\lambda,\eta}(x,a)
= p(S\mid x,a)-\eta p(N\mid x,a)-\lambda p(C\mid x,a),
\quad \lambda>\eta\ge 0.
\]

加入 intervention/switching cost：

\[
Q_{\lambda,\eta,\kappa}(x,a)
= V_{\lambda,\eta}(x,a)-\kappa\,\mathbf 1[a\ne Base].
\]

Router：

\[
\pi(x)=\arg\max_a Q_{\lambda,\eta,\kappa}(x,a)
\]

tie 时 Base 优先。

这使旧 `delta` 有清楚解释：它是 switching-cost / intervention-budget 的对偶变量，不是神秘“保守阈值”。

## 3.3 约束优化解释

论文给出：

\[
\max_\pi \mathbb E[V_{\lambda,\eta}(X,\pi(X))]
\quad
\text{s.t.}\quad
\mathbb E[\mathbf 1\{\pi(X)\ne Base\}]\le \rho .
\]

其 Lagrangian 的逐状态 plug-in 解等价于上面的 `κ`-penalized argmax。
Calibration 的目标 intervention rate `ρ` 用来选择 `κ`。

## 3.4 三个理论命题

### Proposition 1：Outcome sufficiency

完整 outcome distribution 对任意 outcome-based utility 都足够：

\[
\mathbb E[u(Y(a),a)\mid X=x]
=\sum_y u(y,a)\,p(y\mid x,a).
\]

因此同一个 outcome model 可以在不重训的情况下改变 λ、η、κ。

### Proposition 2：Scalar-risk insufficiency

令：

```text
r0(x) = P(Catastrophe under Base | x)
```

仅依赖 `r0` 的最优 policy 与拥有完整 `X` 的最优 policy 之间存在 information gap：

\[
\Delta_{\text{risk}}
=
\mathbb E[\max_a V(X,a)]
-
\mathbb E[\max_a\mathbb E[V(X,a)\mid r_0(X)]] .
\]

当同一 risk level 内最优 option 不一致且 advantage 非零时，gap 严格为正。

实验上必须给出：

- risk bin 内 Oracle option entropy；
- 同 risk bin 中 Detour-optimal / Retreat-optimal / Base-optimal 的比例；
- cross-fitted risk-only oracle ceiling；
- Router 与该 ceiling 的比较。

### Proposition 3：Conservative advantage bound

若所有 option 的 value estimation error 不超过 ε：

\[
|\hat V(x,a)-V(x,a)|\le \epsilon,
\]

并仅在：

\[
\hat V(x,a)-\hat V(x,Base)>\kappa
\]

时接管，则所选 option 的真实 Base-relative advantage 大于：

\[
\kappa-2\epsilon .
\]

若 `κ ≥ 2ε`，则在该 uniform-error event 上不会选择真实 utility 低于 Base 的接管；pointwise regret 上界为 `κ+2ε`。

必须明确：当前 calibration margin 并未估计 ε，因此旧结果没有该统计保证。

## 3.5 可选增强：Group-Conformal Advantage Router

只有在 development 不塌缩时采用。

对 calibration source `s` 定义 one-sided group score：

\[
R_s
=
\max_{i\in s,\,a\ne Base}
\left[
\hat A_i(a)-A_i(a)
\right],
\]

其中：

```text
A_i(a) = realized utility(a) - realized utility(Base)
```

取 split-conformal quantile `q_(1-α)`，形成 lower bound：

\[
L_i(a)=\hat A_i(a)-q_{1-\alpha}.
\]

仅当：

```text
max_a L_i(a) > κ
```

时接管。

所需条件：

- calibration unit 是 source；
- score 在 source 内对所有 condition/horizon/option 取 max；
- train/cal/test source 完全不重叠；
- α 与 quantile rule 在 test 前冻结；
- 不把该保证外推到 from-reset sequential stopping time。

### 采用门槛

- α=0.1 时 calibration source 至少 10 个。
- development 上 intervention 不能塌缩为 0。
- harmful intervention 显著下降，且 Oracle value recovered 仍有实际意义。
- 若塌缩，保留 Proposition 3 的条件性理论，方法仍用 cost-calibrated plug-in router，不伪造保证。

## 3.6 代码重构

创建：

```text
crashbench/potential_outcomes.py
crashbench/intervention_policy.py
crashbench/conformal_advantage.py   # 可选
tests/iclr27/test_potential_outcomes.py
tests/iclr27/test_intervention_policy.py
tests/iclr27/test_conformal_advantage.py
```

旧接口保持 backward-compatible，不修改 frozen artifact 的解释。

---

# 第四阶段：建立真正的 option-ambiguity benchmark

当前 T−20 fresh cohort 很可能主要是：

```text
glass -> Base catastrophe -> Detour useful
controls -> Base success
```

这不足以证明“which intervention”。

## 4.1 新 benchmark 必须包含的机会类型

每个 task 尽量构造四类：

1. **Base-preferred**
   - Base success；
   - Detour/Retreat 至少一个更差。

2. **Detour-preferred**
   - Base catastrophe 或 failure；
   - Detour task success；
   - Retreat safe noncompletion。

3. **Retreat-preferred**
   - Base catastrophe；
   - Detour 也 catastrophe/failure；
   - Retreat safe noncompletion；
   - 在高 λ 下 Retreat 为最优。

4. **No-good-option / abstention diagnostic**
   - 所有 option 均失败，或只有极小差异；
   - 检查 Router 是否过度自信。

## 4.2 利用 horizon 制造 option heterogeneity

优先复用现有 exact-state 基础设施：

- Early：`T−40` 或开发集确认可绕行的早期点；
- Mid：`T−20`；
- Late：`T−10` 或 `T−5`，绕行可能来不及但 Retreat 仍可避免 catastrophe。

任务选择、horizon 和 eligibility 只能依据 Base 与 controller mechanics，不得依据 Router test 表现。

## 4.3 TaskSpec 重构

创建：

```text
crashbench/task_specs.py
crashbench/option_library.py
configs/iclr27/tasks/*.yaml
```

`TaskSpec` 至少包含：

```yaml
task_id:
instruction:
target_object:
goal_object_or_region:
success_predicate:
hazard_object:
hazard_placement_generator:
detour_controller:
retreat_controller:
max_remaining_steps:
control_conditions:
```

移除 controller 中对 bowl/plate/task-0 的隐式硬编码。

## 4.4 同步公平性

所有 option：

- 从完全相同的 restored state 开始；
- 恢复 simulator、controller、RNG、policy cache；
- 使用相同剩余环境 horizon；
- 报告实际执行 steps；
- 报告 wall-clock latency；
- 报告 path length、control effort；
- catastrophe predicate 统一；
- task success predicate 统一。

如果 structured skill 需要额外 planning compute，作为 intervention cost 报告，不隐藏。

## 4.5 Exact-state integrity 测试

增加：

- restore 后 observation hash 相同；
- Base branch 重跑一致；
- RNG state hash；
- policy reset/cache contract；
- controller state hash；
- same seed + same state 的 determinism test；
- option execution 不修改其他 branch 的共享状态；
- failed eligibility whole-source exclusion test。

产物：

```text
docs/iclr27/EXACT_STATE_CONTRACT.md
results/iclr27/exact_state_integrity_<commit>.json
```

---

# 第五阶段：第二任务与新确认队列

## 5.1 任务筛选原则

从 LIBERO 中筛选至少两个任务，不能根据 Router outcome 选任务。筛选只看：

- Base nominal success 足够；
- hazard 可插入；
- same option semantics 可适配；
- success predicate 可复用；
- source reset 有足够多样性；
- controller contract 无异常。

## 5.2 建议规模

### Primary task

- 30 independent source states。
- 3 conditions：on-path / off-path / no-hazard。
- 3 horizons：early / mid / late。
- 每个 decision 执行 3 options。
- source 是统计单位。

### Replication task

- 20 independent source states。
- 同样 3 conditions × 3 horizons × 3 options。

总计约：

```text
50 sources
450 matched decisions
1350 option branches
```

若资源受限，最低不能低于：

```text
20 primary sources + 15 replication sources
```

但不得把 condition/horizon 充当独立 source。

## 5.3 Source allocation

在任何 outcome-bearing run 之前生成并 hash：

```text
configs/iclr27/source_allocation_v1.json
```

必须分开：

- train
- calibration
- development
- confirmatory test

test IDs 可以由脚本加密/隐藏，但 allocation hash 必须提前提交。

## 5.4 单一 confirmatory operating point

旧 n=8/n=13 全部降为 development/pilot。
在旧数据上确定并冻结：

```yaml
primary_lambda:
primary_eta:
primary_kappa_or_target_rate:
primary_alpha_if_conformal:
primary_model_variant:
primary_feature_contract:
primary_baselines:
primary_noninferiority_margin:
```

新 confirmatory test 只能报告该单点为 primary。

Secondary：

- 完整 preference/frontier 可报告；
- 必须标记 exploratory；
- 不再从 secondary grid 选一个“新 headline”。

## 5.5 主要假设建议

### H1：相对 Risk→Best-Fixed 的任务价值

在 frozen intervention budget 下，Router 提高：

```text
U_(lambda,eta) 或 task success
```

并不超过预先定义的 catastrophe non-inferiority margin。

### H2：相对 Direct Choice/Q 的 outcome-model 价值

Outcome Router 的：

- utility；
- calibration；
- preference transfer；
- Oracle regret

至少在其中一项有预先定义的优势。

### H3：跨任务一致性

第二任务上差异方向一致，不要求每个任务单独显著。

### H4：技能选择确实发生

test 中 Detour 和 Retreat 都必须有非零选择/Oracle-optimal count；否则该 cohort 不能证明多技能选择。

## 5.6 Quest wrapper

创建：

```text
setup/iclr27_task_screen.sbatch
setup/iclr27_capture_train_cal.sbatch
setup/iclr27_confirmatory_primary.sbatch
setup/iclr27_confirmatory_replication.sbatch
setup/submit_iclr27_pipeline.sh
```

wrapper 必须：

- fail closed；
- 检查 clean commit；
- 记录 job ID、commit、checkpoint、scenario fingerprint；
- 不覆盖结果；
- completion 后自动运行 integrity + analysis；
- 若 source 数不足，只能按预声明 supplement rule 补充；
- 不读取 method comparison 决定是否补 source。

---

# 第六阶段：分析与统计

## 6.1 Primary analysis

只针对冻结单点：

- source-level mean difference；
- raw counts；
- source-block exact randomization test；
- cluster bootstrap CI；
- catastrophe non-inferiority；
- intervention budget difference；
- per-task interaction；
- per-horizon interaction。

## 6.2 Frontier analysis

替换当前手工“四条 acceptance criteria 到处找点”的主要叙述。

新增：

1. 3D Pareto set：success、−catastrophe、−intervention。
2. Predefined reference point 下的 dominated hypervolume。
3. Router family 相对 baseline family 的 hypervolume gain。
4. source bootstrap hypervolume CI。
5. empirical attainment / point survival frequency。
6. simultaneous max-T bootstrap CI，控制 45 点多重比较。
7. 不使用 test outcome 重新定义容差。

## 6.3 Preference robustness

对：

```text
lambda ∈ predefined grid
eta ∈ predefined grid
kappa ∈ predefined grid
```

画：

- best deployable method preference map；
- Router 优于每个 baseline 的区域；
- utility regret heatmap；
- preference transfer：同一 outcome model 无需重训。

Direct-Q baseline：

- train at λ=5；
- test reweight at other λ 时不能直接重用，需显示其限制；
- 另给每个 λ 单独训练的 upper control，避免不公平。

## 6.4 Calibration

每 option × outcome：

- NLL；
- Brier；
- ECE；
- reliability diagram；
- sharpness；
- advantage calibration；
- expected vs realized intervention value。

必须按 source cross-fitting 或 held-out source 计算。

## 6.5 机制分析

1. risk bin 中 Oracle action entropy。
2. risk 相似但 optimal option 不同的 paired examples。
3. harmful intervention case study。
4. missed-beneficial case study。
5. Base/Detour/Retreat confusion matrix。
6. hidden-only、state-only、combined ablation。
7. option outcome head 的特征重要性/线性系数稳定性。
8. task/horizon transfer。
9. privileged vs perceptual information boundary。

## 6.6 Sequential boundary

不做新恢复模型。只重写旧证据：

- fixed-state selection distribution `D_anchor`；
- from-reset repeated-look distribution `D_traj`；
- stopping time：
  \[
  \tau=\inf\{t:s_t>q\}
  \]
- statewise calibration 不控制：
  \[
  P(\exists t:s_t>q)
  \]
- 旧 P3.1 offline ranking 与 P3.2 fresh trajectory inversion 作为 boundary evidence。
- 主文最多半页或一张小表；完整历史进 appendix。
- 不以 “we solve when to intervene” 为题或摘要结论。

---

# 第七阶段：论文结构

## 建议标题

首选：

> **CrashBench: Exact-State Potential Outcomes for Selective VLA Intervention**

备选：

> **Beyond Failure Detection: Exact-State Intervention Value Learning for VLA Safety**

不建议继续单独使用 “Risk Is Not Intervention Value” 作为全部 novelty，因为同期 selective intervention 工作已经非常接近。

## 7.1 九页主文预算

### 1. Introduction — 0.9 页

只讲一个问题：

> Failure detection reports what may go wrong under Base; intervention routing must estimate what each available response will cause.

贡献四条：

1. Counterfactual Intervention Routing 形式化。
2. exact-state full-potential-outcome benchmark/protocol。
3. preference-reweightable conservative outcome router。
4. multi-task fixed-decision evidence + honest sequential boundary。

### 2. Related Work — 0.7 页

四组：

- VLA failure detection；
- runtime safety constraints；
- failure recovery/selective intervention；
- safety benchmarks。

CoWAM 和 CheckVLA 必须作为 closest work 正面比较，不能藏在 appendix。

### 3. Problem Formulation — 1.0 页

- `Z, X, A, Y(a)`；
- utility；
- intervention budget；
- risk sufficiency gap；
- fixed decision versus sequential trigger。

### 4. CrashBench Exact-State Protocol — 1.0 页

- identical-state branching；
- conditions/horizons；
- source split；
- full outcome labels；
- integrity hashes；
- Oracle role。

### 5. Method — 1.1 页

- outcome model；
- preference reweighting；
- switching cost；
- optional conformal advantage；
- propositions。

### 6. Experimental Setup — 0.7 页

- tasks/models/hazards；
- options；
- strongest baselines；
- primary endpoint；
- statistics。

### 7. Results — 2.4 页

- confirmatory table；
- strong baseline comparison；
- second-task consistency；
- option ambiguity；
- calibration/preference robustness；
- one main frontier figure。

### 8. Boundary and Limitations — 0.6 页

- privileged option；
- fixed decision timing；
- small/finite task scope；
- sequential negative；
- simulator potential outcomes。

### 9. Conclusion — 0.3 页

一句核心结论，不做项目史总结。

## 7.2 主图

### Figure 1：Problem + exact-state branches

左：

```text
same restored state
 -> Base
 -> Detour
 -> Retreat
```

右：

- 同 risk、不同 optimal option 的两个例子。

### Figure 2：Method

```text
X + option
 -> outcome distribution
 -> arbitrary utility weights
 -> intervention-cost/conformal lower bound
 -> Base / Detour / Retreat
```

### Figure 3：Confirmatory results

- primary fixed point；
- strongest baselines；
- source paired differences。

### Figure 4：Generalization and ambiguity

- two tasks × three horizons；
- Oracle-optimal option distribution；
- preference robustness。

Sequential negative只放小表或 appendix figure。

## 7.3 主表

必须包括：

- Base
- Risk→Retreat
- Risk→Detour
- Risk→Best-Fixed
- Risk+Two-Stage
- Direct Choice
- Direct-Q
- Pairwise Advantage
- Outcome Router
- Oracle

Hazard Prompt 放单独 panel，避免与 exact-anchor same-state selector混淆。

---

# 第八阶段：同期工作定位

`docs/iclr27/RELATED_WORK_MATRIX.md` 至少包含以下列：

```text
Work
Primary problem
Failure/risk detection
Chooses among interventions
Candidate futures or realized branches
Same-state/same-pool audit
Sequential trigger control
Formal guarantee
Tasks/models scale
CrashBench distinction
```

必须覆盖：

- SAFE
- SafeVLA
- AEGIS/VLSA
- Neuro-Symbolic Safety Guidance
- ProbeAct
- FailSafe
- CycleVLA
- FLARE
- Imagining Recovery / CoRe
- CheckVLA
- CoWAM
- TOWN-VLA
- SafeVLA-Bench
- ForesightSafety-VLA
- LIBERO-Safety

论文定位：

- 不声称 first failure detector。
- 不声称 first runtime shield。
- 不声称 first recovery method。
- 不声称 first selective intervention layer。
- 可以主张的独特组合是：
  1. exact-state complete option outcomes；
  2. potential-outcome / treatment-value formulation；
  3. task success、catastrophe、safe noncompletion 分解；
  4. preference-reweightable routing；
  5. selector quality 与 option quality 的分离；
  6. fixed-decision 与 sequential stopping 的明确边界。

---

# 第九阶段：代码、匿名化与复现

## 9.1 论文工程

创建：

```text
paper/main.tex
paper/sections/
paper/figures/
paper/tables/
paper/references.bib
paper/appendix.tex
paper/Makefile
```

使用 ICLR 2027 模板，主文不超过 9 页。

## 9.2 自动表格

所有数字由脚本生成 `.tex`，禁止手抄：

```text
scripts/iclr27/render_main_tables.py
scripts/iclr27/render_appendix_tables.py
```

每个 `.tex` 顶部写 source JSON hash。

## 9.3 匿名发布

创建：

```text
scripts/iclr27/build_anonymous_release.py
```

检查并移除：

- 用户名；
- Northwestern/Quest 路径；
- GitHub owner；
- Slurm account；
- job IDs（论文可放匿名 provenance ID）；
- commit 中可识别作者的信息；
- 图片 metadata；
- 文档 author metadata。

输出：

```text
dist/crashbench_iclr27_anonymous_<hash>.zip
```

## 9.4 复现层级

- CPU-only synthetic tests；
- tracked summary reanalysis；
- full simulator rollout recipe；
- model checkpoint requirements；
- raw data availability；
- exact-state hash verification；
- seed/nondeterminism policy。

## 9.5 ICLR 必备

- AI-use statement。
- reproducibility statement。
- ethics statement：physical safety benchmark、simulator limitations、misuse/overclaim boundary。
- 所有作者在 abstract deadline 前确认 OpenReview profile。
- genuine abstract，不用 placeholder。

---

# 第十阶段：按日期执行

## 8 月 29–31 日

- 完成第一阶段 truth-source。
- 完成强 baseline suite。
- 得到 GO/NO-GO。
- 同时搭建 LaTeX paper skeleton。

## 9 月 1–3 日

- 冻结最终问题定义和方法。
- 完成 TaskSpec/option library 重构。
- 完成 exact-state integrity tests。
- 完成第二任务机械筛选。
- 冻结 confirmatory protocol 和 source allocation。

## 9 月 3–10 日

- 收集新的 train/cal（如方法改变）。
- 冻结模型。
- 执行 primary + replication confirmatory runs。
- 严禁看中间结果后改 point。

## 9 月 8–13 日

- 完成统计分析、强 baseline、preference map、calibration。
- 生成所有主图主表。
- 写 Methods/Experiments/Results 初稿。

## 9 月 13–17 日

- 完成 9 页论文。
- 做 adversarial reviewer audit。
- 完成 related work 和 appendix。
- 内部检查所有 claim hash。

## 9 月 18 日

- 提交真实 abstract。
- 冻结作者列表。

## 9 月 18–23 日

- 完成匿名代码包、reproducibility、AI-use、ethics。
- 最终文字压缩和图表检查。
- 不新增会改变核心故事的大实验。

## 9 月 24 日

- 内部 final freeze。
- 运行完整 audit。
- 生成最终 PDF 和 supplementary。

## 9 月 25 日

- 提交 full paper。

---

# 第十一阶段：Codex 分阶段提示词

## Prompt A：只做审计，不改实验

```text
你正在仓库 hanshuo-shuo/crash_bench 的分支
codex/glass-recovery-d0-fprime-rescue 上工作。

先读取 AGENTS.md、QUEST_WORKFLOW.md、setup/README.md、
docs/CURRENT.md、docs/PAPER_PLAN.md、docs/CLAIMS.md、
docs/FRESH_COUNTERFACTUAL_ROUTER_PROTOCOL.md、
docs/COUNTERFACTUAL_ROUTER_MAIN_RESULT.md 和 docs/REPRODUCIBILITY.md。

目标：建立 ICLR 2027 truth source，不运行新 GPU 实验，不修改 frozen result。

创建：
- docs/iclr27/MASTER_PLAN.md
- docs/iclr27/CLAIM_LEDGER.md
- docs/iclr27/REVIEWER_RISK_AUDIT.md
- docs/iclr27/RELATED_WORK_MATRIX.md
- results/iclr27/manifest.json

重点审计：
1. binary-risk baseline 是否只固定接 Retreat；
2. 是否缺 Risk→Detour、Risk→best-fixed、two-stage、direct choice、
   direct-Q、pairwise advantage；
3. λ=1,target=.6 是否是 test-frontier post-selection；
4. n=8/n=13 的统计与多重比较；
5. fresh cohort 是否主要退化为 hazard/control 分类；
6. privileged Detour、option horizon fairness；
7. delta 是否被错误解释为 confidence bound；
8. 同期 CoWAM/CheckVLA/SAFE 等的重合。

所有判断必须指向具体文件、函数、行或 artifact。
不要美化项目，不要新增 unsupported claim。
完成后运行测试、audit_repo.py、git diff --check，并提交一份简洁结论。
```

## Prompt B：实现强 baseline suite

```text
继续在新的 codex/iclr27-exact-state-intervention-routing 分支工作。
遵守 AGENTS.md，不覆盖 frozen artifacts。

实现统一的 source-disjoint baseline suite：
Base、AlwaysDetour、AlwaysRetreat、Risk→Retreat、Risk→Detour、
Risk→BestFixed、Risk+TwoStage、DirectChoice、DirectQ、
PairwiseAdvantage、OutcomeRouter、hidden-only、state/action-only。

要求：
- identical features/splits/source weights；
- 训练和 calibration 无 test leakage；
- 参数量、NLL、Brier、ECE、success、catastrophe、safe noncompletion、
  intervention、harmful/unnecessary intervention、Oracle regret/value；
- source-block exact randomization test + cluster bootstrap；
- per-source/per-condition/per-horizon raw table；
- 所有 output 新目录并写 manifest/hash；
- tests 覆盖 tie-break、rate matching、best-fixed selection、no leakage。

先只在现有 development/full-capture 上运行 CPU analysis。
不要读取 fresh test 后选择超参数。
最后根据预先定义的 GO-A/GO-B/PIVOT-C/STOP-D 给出结论。
```

## Prompt C：数学接口与可选 conformal advantage

```text
把现有 outcome router 重构为 potential-outcome intervention policy，
保持旧 frozen artifacts backward-compatible。

新增：
- crashbench/potential_outcomes.py
- crashbench/intervention_policy.py
- 可选 crashbench/conformal_advantage.py

实现：
V_{lambda,eta}=pS-eta*pN-lambda*pC
Q=V-kappa*I[option!=Base]
Base-favoring tie break
target intervention budget 与 kappa calibration
preference reweighting
risk-sufficiency-gap analysis
uniform-error conservative bound 的数值单测

可选 group-conformal：
calibration score 必须按 source 内 max 聚合，
alpha/quantile rule 明确，coverage test 使用 synthetic exchangeable groups。
如果 development 上 conformal policy 零接管，诚实报告并不采用为 primary；
不要通过调 alpha/test threshold 强行制造正结果。
```

## Prompt D：建立 option-ambiguity benchmark

```text
目标不是重新解决 from-reset timing，而是在外部给定 decision state 时，
建立真正要求 Base/Detour/Retreat 选择的 benchmark。

重构 TaskSpec 和 option library，移除 task-0 bowl/plate 硬编码。
构造 early/mid/late horizons，使数据中同时包含：
Base-preferred、Detour-preferred、Retreat-preferred、No-good-option。

任务和 horizon 的选择只能依据 Base scan、controller mechanics 和
eligibility contract，不能依据 Router outcome。

所有 option 使用相同 restored simulator/controller/RNG state 和相同
remaining horizon；记录 steps、path length、latency、control effort。
增加 deterministic restore、RNG hash、Base rerun、whole-source exclusion tests。

先做 mechanical smoke 和 task compatibility report；
不要在协议冻结前运行 outcome-bearing confirmatory test。
```

## Prompt E：冻结并执行新确认实验

```text
读取已完成的 baseline audit、TaskSpec、exact-state integrity report。
把所有旧 n=8/n=13 结果视为 development/pilot。

在 configs/iclr27/confirmatory_protocol.yaml 中冻结：
- primary model；
- primary lambda/eta/kappa；
- alpha（如使用 conformal）；
- primary/replication tasks；
- horizons/conditions；
- source allocation；
- attempted/eligible source rule；
- primary baselines；
- noninferiority margin；
- statistical tests；
- no-retuning clause。

协议和 source allocation 提交并 hash 后，才允许执行 Quest pipeline。

目标规模：
- primary 30 sources；
- replication 20 sources；
- 每 source 3 conditions × 3 horizons；
- source 为独立单位。

执行期间不得根据 Router-vs-baseline 中间结果补 source、改 threshold、
换 task 或改 headline point。
完成后自动生成 sealed manifest，并停止实验发现。
```

## Prompt F：最终统计与论文

```text
基于 sealed confirmatory artifacts 生成 ICLR 2027 paper。

Primary 只报告冻结单点。
Secondary frontier 明确 exploratory，并用 hypervolume、simultaneous CI、
source exact tests，不能再从 test grid 选一个新 headline。

论文题目优先：
CrashBench: Exact-State Potential Outcomes for Selective VLA Intervention

主文 9 页：
Introduction / Related Work / Formulation / Exact-State Protocol / Method /
Setup / Results / Boundary & Limitations / Conclusion。

必须正面比较 CoWAM、CheckVLA、SAFE、AEGIS、ProbeAct、FailSafe、
CycleVLA、FLARE、CoRe、SafeVLA-Bench、ForesightSafety-VLA、
LIBERO-Safety。

wall probe 只作短动机或 appendix；
P2–P3.2 sequential history 主文最多半页；
不声称 solved when-to-intervene、end-to-end recovery、formal safety guarantee
（除非 conformal theorem 的条件与范围完全满足）。

所有表格自动从 JSON 生成，不手抄数字。
完成匿名 release、AI-use statement、reproducibility、ethics 和最终 audit。
```

---

# 第十二阶段：最后的决策规则

## 适合冲 ICLR main track

至少满足：

1. 新的单点 confirmatory cohort，而不是旧 test frontier 选点。
2. Risk→Best-Fixed、Two-Stage、Direct-Q/Choice 后仍有清楚价值。
3. 两个任务或一个任务 + 明确不同 option-ambiguity families。
4. Detour 与 Retreat 在 test 中都真正成为最优/被选择。
5. source 数量明显高于 8，CI 不再极宽。
6. exact-state protocol 和 integrity audit 可复核。
7. 数学表述不把 margin 假装成 confidence bound。
8. 与 CoWAM/CheckVLA 的差异写得具体且有实验支撑。

## 适合改投 CoRL / RA-L / TMLR 或 workshop

若：

- exact-state benchmark 很扎实；
- 但只有单任务/单模型；
- 或 outcome router 与 direct-Q 类似；
- 或新 confirmatory 规模不足；
- 或 sequential 部署仍是主要期待但没有正结果。

## 不应以当前主张直接提交

若：

- Risk→Best-Fixed 已追平；
- Direct/simple geometry 全面追平；
- 新 frozen single point 不复现；
- 只能靠 test frontier 重新选点；
- 第二任务方向反转；
- 仍需把 privileged Detour 写成 learned recovery 才显得强。

在这些情况下，保留负结果与 benchmark，重新定题；不要靠换措辞掩盖证据。
