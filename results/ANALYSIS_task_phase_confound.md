# 任务阶段混淆诊断

本分析只读取已有的 `results/selfreport/hidden.npz`、`meta.json` 和已经拟合的
`probe_T5.npz`，没有重新运行模型，也没有覆盖原始结果。可复现脚本是
[`scripts/task_phase_confound_analysis.py`](../scripts/task_phase_confound_analysis.py)，结果为
[`results/task_phase_confound/task_phase_confound.json`](task_phase_confound/task_phase_confound.json)，汇总图为
[`results/task_phase_confound/task_phase_confound_summary.png`](task_phase_confound/task_phase_confound_summary.png)。

## 设计

- `wall` 中 `0 <= steps_to_crash <= 5` 的 29 帧作为 on-path collision-window。
- timestep 匹配：对每一帧碰撞窗口，分别取 `offpath` 和 `nowall` 中最近 timestep 的一帧；匹配是确定性的，平局按 scenario ID 和原始行号解决。
- 最近邻匹配：在 pooled 数据上对 `(timestep, EEF x/y/z, act_xyz_norm)` 按列标准差标准化，再分别取 off-path 和 no-wall 的欧氏距离最近邻。
- off-path 的 5 个实际发生碰撞的边界 episode（共 318 行）不进入主控制池；否则 “visible wall, no collision” 会混入另一个碰撞标签。JSON 中保留了含这些行的 timestep sensitivity 检查。
- LOSO 分类池为所有 on-path wall 行、所有 paired no-wall 行和 safe off-path 行；正类是 collision-window，负类是其余行。每个 held-out wall scenario 同时留出同 ID 的 no-wall 行；LOSO 指标在这 5 个 paired held-out scenario 上计算，safe off-path 只作为每折的 wall-visibility control 进入训练。
- 五组特征使用同一个 standardized L2 logistic regression。hidden-state 分支在每个训练折内先做确定性的 PCA-50（seed 0、2 次 power iteration），再用相同 logistic classifier；没有使用测试折拟合 PCA 或标准化参数。

## 结果

### Frozen probe 的 matched logit

Timestep 完全匹配（29/29，最大 `|Δt|=0`）时，on-path collision-window 的 probe logit 中位数是 **1.482**，safe off-path 是 **−3.669**，no-wall 是 **−7.793**。配对差值的中位数分别是 **5.095** 和 **9.351**。

在 timestep + EEF xyz + action magnitude 最近邻匹配后，平均标准化距离为 off-path **0.128**、no-wall **0.188**；on-path 与两组控制的 logit 中位数差仍为 **5.102** 和 **6.227**。因此，probe 的高 logit 并不能由这三类可观测量的近邻关系解释掉。

### LOSO 分类

| feature | ROC-AUC | AUPRC |
|---|---:|---:|
| timestep | 0.907 | 0.134 |
| EEF xyz | 0.793 | 0.094 |
| timestep + EEF xyz | 0.860 | 0.103 |
| timestep + EEF xyz + action magnitude | 0.806 | 0.128 |
| hidden state | **0.917** | **0.324** |

LOSO 的评估集为 1,185 个 paired held-out 行，其中 29 个正类，正类比例为 0.0245；因此 AUPRC 的随机基线约为 0.0245。hidden 相比完整可观测协变量基线 `timestep + EEF xyz + action magnitude` 的增益为 ROC-AUC **+0.112**、AUPRC **+0.196**。动作幅值加入后没有带来稳定的增益，说明它不是这批数据中主要的额外阶段解释量。

## 如何区分三种解释

### Wall visibility confound

这是“模型只知道画面里有一面墙”的解释。safe off-path 同样有墙但没有碰撞；它的 frozen-probe logit 仍接近低值，且显著低于 on-path collision-window。因此单纯 wall visibility 不能解释 probe 的升高。需要注意，边界 off-path 碰撞已从主控制池排除，避免把 visibility control 变成碰撞 control。

### Generic task-phase confound

这是“probe 只读取了任务进度、机器人姿态或动作强度”的解释。timestep、EEF xyz 和它们的组合确实有可观的 LOSO 预测力，说明原始 AUC 不能被直接称为纯粹的 collision code。它们的结果是本诊断必须报告的阶段/姿态基线；动作幅值并没有增强到超过 timestep 的水平。

### Collision-specific information

在这份已有 capture 上，hidden-state LOSO 的 AUPRC 明显高于 timestep + EEF xyz + action magnitude，且在两种匹配后 frozen-probe 的 on-path logit 差仍然存在。综合起来，结果支持 hidden state 含有超出任务阶段、EEF 姿态和动作幅值的 collision-specific 信息；但这是相关性证据，不是因果证据，也不能排除其它未记录的视觉或场景因素。若要升级为因果结论，仍需要跨场景的干预/表示 steering 或更强的场景与视觉条件匹配。

## 复现

在仓库根目录运行：

```bash
MPLCONFIGDIR=/tmp/crashbench-mpl-taskphase \
python scripts/task_phase_confound_analysis.py \
  --input results/selfreport \
  --probe results/selfreport/probe_T5.npz \
  --output results/task_phase_confound \
  --T 5
```

脚本只会在新的 `results/task_phase_confound/` 下写 JSON 和 PNG；它不会修改
`results/selfreport/`、`setup/figures/` 或既有 summary。
