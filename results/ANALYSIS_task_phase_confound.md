# 任务阶段混淆诊断（严格 OOF 版本）

本分析只读取已有的 `results/selfreport/hidden.npz` 和 `meta.json`，不重跑
VLA，也不把全数据拟合的 frozen probe 作为主结果。可复现脚本是
[`scripts/task_phase_confound_analysis.py`](../scripts/task_phase_confound_analysis.py)，
结果为 [`task_phase_confound.json`](task_phase_confound/task_phase_confound.json)，
主图为 [`task_phase_confound_summary.png`](task_phase_confound/task_phase_confound_summary.png)。

## 设计

- `wall` 中 `0 <= steps_to_crash <= 5` 的 29 帧是 collision-window，分布在 5
  个 on-path wall scenario 中。
- 严格 OOF 的每一折留出一个 wall scenario；probe 只在其余 4 个 wall
  scenario 和对应 4 个 nowall scenario 上拟合。训练内做 PCA-50 和标准化
  L2 logistic；held-out wall、同 ID nowall 和 off-path 控制都只用该折 probe
  评分。
- off-path 控制先排除发生过碰撞的 5 个 scenario，保留 16 个全程 safe
  scenario。每个 held-out source timestep 下，对每个控制 scenario 单独取
  最近 timestep 帧；先在 scenario 内取均值，再在 scenario 间做 macro mean。
- 控制使用量按实际 row index 审计：off-path 共 464 次 assignment、256 个
  unique rows、208 次 reuse；nowall 共 29/29/0。16 个 safe off-path scenario
  各有 29 次使用、16 个 unique rows、13 次 reuse；逐 scenario 计数也写入
  JSON 的 `control_usage.*.per_scenario`。
- 分类池只含 wall + paired nowall，仍然是按 held-out wall scenario 做 5
  折 OOF。每个 scenario 单独计算 ROC-AUC 和 AUPRC，再报告 macro mean、
  median、range；连续帧没有被当成独立显著性样本。
- baseline 包括 linear logistic 和单隐层 16-unit tanh MLP。输入覆盖
  timestep、EEF xyz、action magnitude、全部 pairwise/complete combinations，
  以及 `hidden only`、`hidden + covariates`。

## 严格 OOF matched logits

| held-out wall | n | wall logit mean | off-path scenario-balanced mean | paired nowall mean | wall−nowall |
|---|---:|---:|---:|---:|---:|
| wide | 5 | -2.171 | -6.413 | -7.256 | 5.085 |
| d62 | 6 | -5.454 | -5.210 | -5.832 | 0.378 |
| d70 | 6 | -0.702 | -5.749 | -6.556 | 5.854 |
| d78 | 6 | -1.188 | -5.236 | -4.979 | 3.791 |
| d85 | 6 | -0.493 | -5.003 | -5.129 | 4.637 |

跨 5 个 held-out wall scenario 的 wall−nowall 差值 macro mean 为 **3.949**，
median 为 **4.637**，range 为 **0.378–5.854**。wall−off-path 的 scenario-balanced
差值 macro mean 为 **3.521**，median 为 **4.242**，range 为 **−0.244–5.047**。
d62 fold 的差值较小，因此结果不应被写成每个 scenario 都同样强。

## Scenario-level 分类

AUPRC 是主指标，ROC-AUC 是辅助指标。下表中的 `macro / median / range` 均
先对 5 个 held-out scenario 计算，再汇总；括号内为 ROC-AUC 的对应三项。

| model | input | AUPRC macro / median / range | ROC-AUC macro / median / range |
|---|---|---:|---:|
| linear | timestep | 0.260 / 0.251 / 0.034–0.500 | 0.901 / 0.960 / 0.637–0.989 |
| linear | EEF xyz | 0.187 / 0.179 / 0.023–0.339 | 0.848 / 0.933 / 0.468–0.964 |
| linear | action magnitude | 0.042 / 0.053 / 0.016–0.064 | 0.558 / 0.731 / 0.205–0.779 |
| linear | timestep + EEF xyz | 0.209 / 0.184 / 0.027–0.425 | 0.867 / 0.936 / 0.539–0.979 |
| linear | timestep + action magnitude | 0.213 / 0.231 / 0.032–0.325 | 0.887 / 0.951 / 0.609–0.974 |
| linear | EEF xyz + action magnitude | 0.261 / 0.209 / 0.017–0.596 | 0.820 / 0.942 / 0.272–0.992 |
| linear | all measured covariates | 0.329 / 0.198 / 0.018–0.967 | 0.824 / 0.940 / 0.293–0.999 |
| linear | hidden only | **0.716 / 0.915 / 0.026–1.000** | 0.903 / 0.998 / 0.531–1.000 |
| linear | hidden + covariates | 0.743 / 0.915 / 0.025–1.000 | 0.900 / 0.998 / 0.511–1.000 |
| MLP | timestep | 0.260 / 0.251 / 0.034–0.500 | 0.901 / 0.960 / 0.637–0.989 |
| MLP | EEF xyz | 0.150 / 0.164 / 0.027–0.236 | 0.846 / 0.926 / 0.540–0.954 |
| MLP | action magnitude | 0.024 / 0.020 / 0.014–0.047 | 0.321 / 0.306 / 0.089–0.667 |
| MLP | timestep + EEF xyz | 0.188 / 0.157 / 0.015–0.413 | 0.770 / 0.922 / 0.128–0.980 |
| MLP | timestep + action magnitude | **0.442 / 0.245 / 0.036–1.000** | 0.907 / 0.953 / 0.665–1.000 |
| MLP | EEF xyz + action magnitude | 0.310 / 0.202 / 0.014–0.596 | 0.789 / 0.942 / 0.099–0.992 |
| MLP | all measured covariates | 0.194 / 0.103 / 0.041–0.561 | 0.871 / 0.871 / 0.697–0.989 |
| MLP | hidden only | 0.487 / 0.489 / 0.013–1.000 | 0.789 / 0.981 / 0.066–1.000 |
| MLP | hidden + covariates | 0.612 / 0.789 / 0.016–0.958 | 0.839 / 0.994 / 0.234–0.999 |

按 macro AUPRC 从所有 observable candidates 中选择的 strongest observable
baseline 是 nonlinear MLP `timestep + action magnitude`，AUPRC 为 **0.442**；
primary hidden 是与 OOF matched probe 同族的 linear `hidden only`，AUPRC 为
**0.716**。对应 macro ROC-AUC 为 **0.903 vs 0.907**，所以这里的主要差异在
AUPRC，而不是简单地把 ROC-AUC 当作唯一结论。完整 measured covariates 的
linear baseline 为 AUPRC **0.329**，它不是主对照。

主比较的每个 held-out scenario 指标如下；完整 18 个模型/输入组合的逐
scenario ROC-AUC 和 AUPRC 都保存在 JSON 的 `feature_sets.*.per_held_out_scenario`。

| held-out scenario | hidden AUPRC | hidden ROC-AUC | strongest observable AUPRC | strongest observable ROC-AUC |
|---|---:|---:|---:|---:|
| wide | 1.000 | 1.000 | 1.000 | 1.000 |
| d62 | 0.026 | 0.531 | 0.036 | 0.665 |
| d70 | 0.944 | 0.998 | 0.774 | 0.992 |
| d78 | 0.693 | 0.990 | 0.158 | 0.922 |
| d85 | 0.915 | 0.998 | 0.245 | 0.953 |

## Scenario-level inference

primary hidden − strongest observable 的 scenario-level AUPRC 差值为
**0.273**。五个 scenario 的 paired differences 为 `0.000, -0.010, 0.171,
0.535, 0.670`。scenario bootstrap（20,000 次）95% percentile interval 为
**0.030–0.516**；5 个 scenario 的 exact grouped sign-permutation 双侧
`p = 0.25`。因此统计量的独立单位是 scenario，不是 1,185 个连续帧；5 个
scenario 也意味着这不是大样本显著性声明。

## 如何区分两层解释

1. measurable task-phase information 确实存在：timestep、EEF pose、action
   magnitude 及其组合都能在 held-out scenario 上提供不同程度的预测力。
2. measured task progress、EEF pose 和 action magnitude 不能完全解释 hidden
   signal：primary hidden OOF AUPRC 高于经过 nonlinear observable candidate
   比较后的 strongest baseline，并且 strict OOF matched logits 在大多数
   held-out scenario 中保留 wall/control gap。

这仍是 frozen capture 上的关联性分析，不能推出已经完全排除 task-phase
confound，也不能声称存在纯粹或因果性的 collision representation。

## 最终结论

Hidden states contain additional collision-predictive information beyond measured task progress, EEF pose, and action magnitude.

## 复现

```bash
MPLCONFIGDIR=/tmp/crashbench-mpl-taskphase \
envs/openvla/bin/python3.10 scripts/task_phase_confound_analysis.py \
  --input results/selfreport \
  --output results/task_phase_confound \
  --T 5
```

命令只读取现有 hidden/meta，在 `results/task_phase_confound/` 写入 JSON 和 PNG，
不会修改 `results/selfreport/` 或重跑 VLA。
