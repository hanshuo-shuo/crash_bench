# 玻璃碰撞恢复实验进度与证据盘点

> **LEGACY / SUPERSEDED — 2026-08-12.** 这是冻结的执行审计，不是当前论文叙事或运行队列。
> 文中“Pilot D/E/F 仍然开放”以及补完 learned recovery 的建议仅代表当时状态；当前决定是
> **D no-go、F 不运行**。请以
> [GLASS_SAFETY_UTILITY.md](../../appendix/GLASS_SAFETY_UTILITY.md) 和
> [CURRENT.md](../../CURRENT.md) 为准。

日期：2026-08-12  
范围：E14、E15 Pilot A、Broad Pilot B、Scoped Pilot B、Pilot C  
当前结论：完成了“可恢复事故认证 + Oracle 任务完成上界”的最小闭环；尚未验证 learned closed-loop recovery。

## 1. 执行摘要

目前最可靠、最适合论文的故事是：

1. 既有实验已经表明，VLA 在可见障碍进入动作走廊时会发生碰撞；隐藏表示中可以解码碰撞临近性，但原始动作未充分利用该信息。
2. 玻璃恢复线建立了严格的事故准入条件：同一 H=20 状态必须能复现 Base 碰撞，同时存在独立重捕获的安全任务完成 Oracle，并且 matched off-path control 也能安全完成任务。
3. 完整的 scoped Pilot B 账本从 15 个候选中认证出 3 个 controller-compatible、robustly recoverable accidents；总体产率为 3/15，在 Base accident 条件下为 3/12。
4. Pilot C 在两个认证的 development source states 上，以 exact anchor、Oracle timing 和 Oracle recovery 运行 3 个 seeds，共 6/6 safe task success、0/6 catastrophe、6/6 exact simulator/controller restore。
5. 这证明了认证事故仍然可恢复，并给出了任务完成型恢复的 Oracle upper bound；它不证明 learned trigger、learned recovery action、final-heldout generalization 或任意玻璃布局覆盖。

![实验故事轨迹](report_assets/glass_recovery_progress_20260812/fig_evidence_trajectory.png)

## 2. 实验进度轨迹

| 阶段 | 目的 | 结果 | 当前角色 |
|---|---|---|---|
| E14 acceptance smoke | 验证“Base 会撞、Oracle 能安全完成任务”的环境是否存在 | 109 次 rollout attempts 中接受 3 个 placements | 环境有效性与存在性证据 |
| E15 Pilot A | 修复并审计旧状态的 exact H=20 replay/recapture | 3/3 exact replay，3/3 independent Oracle recapture | 状态恢复与协议工程验证；不直接迁移为 v2 数据 |
| Broad Pilot B | 在 20 个场景、6 个 H 上寻找可扩展 population | 120/120 cells 完成，但没有共同 H 通过 gate | 广泛 population claim 的 no-go |
| Fresh scoped H=20 ledger | 用修复后的 snapshot/oracle 对 15 个新候选做完整认证 | 3 个 accepted pairs；3/15 overall，3/12 conditional Base accidents | 论文中的认证/yield screen |
| Stricter 0.17/0.30 authoring | 尝试更宽 bowl clearance、更远 off-path 和 35 N margin | 0 个候选通过联合预筛；未进入 live frontier | 失败的 authoring feasibility，不是恢复负结果 |
| Pilot C | 测试认证事故在 exact anchor 上是否仍能稳定安全完成任务 | 6/6 safe task success，0/6 catastrophe | development-only Oracle upper bound |
| Pilot D/E/F | learned timing、learned action、完整闭环 | 未执行 | 仍然开放 |

## 3. Pilot B 筛选过程

![Pilot B 筛选漏斗](report_assets/glass_recovery_progress_20260812/fig_pilot_b_funnel.png)

完整 H=20 ledger 包含 15 个终止结果：

| 筛选节点 | 数量 | 相对上一节点 |
|---|---:|---:|
| authored candidates | 15 | 100% |
| live Base catastrophes | 12 | 80% |
| exact H=20 replay | 9 | 75% of Base accidents |
| complete accepted pairs | 3 | 25% of Base accidents；20% overall |

12 个未被接受的候选全部保留在 ledger 中：

| 拒绝原因 | 数量 | 含义 |
|---|---:|---|
| `no_base_crash` | 3 | 没有形成要研究的事故 |
| `nominal_replay_failure` | 2 | captured Base suffix 未在 exact H=20 重现事件 |
| `invalid_initial_state` | 1 | H=20 anchor 已经不是干净任务起点 |
| `off_path_catastrophe` | 2 | matched control 仍碰撞，不能支持路径特异性 |
| `oracle_task_failure` | 4 | Oracle 没有在预算内完成原始任务 |

三个 accepted pairs 为：

| Pair | Split/role | Geometry | Stored Base frames | Stored Oracle frames | Stored off-path frames |
|---|---|---|---:|---:|---:|
| `glass_recovery_train_0003` | train | nominal | 20 | 183 | 66 |
| `glass_recovery_heldout_0000` | development evaluation | tall_narrow，radius 0.024 m | 20 | 232 | 59 |
| `glass_recovery_heldout_0004` | development evaluation | tall_narrow，radius 0.024 m | 20 | 300 | 57 |
| **合计** | 3 independent source states |  | **60** | **715** | **182** |

三条 accepted pairs 合计保存 957 张 224×224 observation frames。每条 accepted pair 都满足：Base crash、exact H=20 replay、Oracle no-crash + task success、off-path no-crash + task success。

需要特别说明：成功 ledger 实际使用的最小 target clearance 是 0.12 m，off-path offset 是 0.20 m。后续提出的 0.17 m / 0.30 m / 35 N margin 严格 authoring 没有生成候选，因此不能写成已执行成功的几何设置。

## 4. Pilot C 结果

![Pilot C 成功矩阵](report_assets/glass_recovery_progress_20260812/fig_pilot_c_success_matrix.png)

Pilot C 的评估条件是：

- `evaluation_mode = exact_anchor`
- `regime = treatment`
- `condition = oracle_timed_oracle_recovery`
- H=20，最大 360 actions
- 两个 development source states，每个 seeds `{101, 202, 303}`

结果如下：

| Pair | seed 101 | seed 202 | seed 303 | 步数 | Peak glass force |
|---|---|---|---|---:|---:|
| `heldout_0000` | safe task success | safe task success | safe task success | 232 | 16.62 N |
| `heldout_0004` | safe task success | safe task success | safe task success | 300 | 11.57 N |

所有六条 episode 均低于 25 N glass-catastrophe threshold，并在完整任务成功时终止。每个 pair 的三个 seeds 轨迹完全重叠，因此 K=3 是嵌套重复，独立样本数仍是两个 source states。

![Pilot C 玻璃力轨迹](report_assets/glass_recovery_progress_20260812/fig_pilot_c_force_traces.png)

第一次使用 220-action evaluation budget 时，两个已知成功轨迹分别需要 232 和 300 actions，因此出现 timeout。将评估预算改为 primary protocol 已授权的 360-action Oracle budget 后，6/6 全部成功；这是 right-censoring 修正，不是对 controller 的结果后调参。

## 5. 真实轨迹图片

以下 filmstrip 直接从 accepted pair 的 `nominal_catastrophe.npz`、`oracle_recovery.npz` 和 `off_path_control.npz` 中等距抽取 `images`，不是示意图，也不是生成式图片。

### heldout_0000

![heldout_0000 真实轨迹](report_assets/glass_recovery_progress_20260812/fig_heldout_0000_filmstrip.jpg)

### heldout_0004

![heldout_0004 真实轨迹](report_assets/glass_recovery_progress_20260812/fig_heldout_0004_filmstrip.jpg)

## 6. 视频与动态图现状

当前 Pilot B/C 原始作业没有直接写出 MP4。原始证据是 NPZ 中逐帧保存的 224×224 `images`，所以可以重建可视化，但必须标注为 derived media。

本报告已从 `heldout_0000` 的 232-frame Pilot B Oracle trajectory 重建：

- [Oracle recovery GIF](report_assets/glass_recovery_progress_20260812/heldout_0000_oracle_recovery.gif)
- [Oracle recovery MP4](report_assets/glass_recovery_progress_20260812/heldout_0000_oracle_recovery.mp4)

![heldout_0000 Oracle recovery](report_assets/glass_recovery_progress_20260812/heldout_0000_oracle_recovery.gif)

这段动态图是 Pilot B accepted-pair 的 Oracle recapture，不是 Pilot C 每个 seed 的直接录像。Pilot C 保存了 6 条逐 action JSON trace 和 trigger-state snapshot，但没有保存逐 seed RGB video。因为 Pilot C 的三个 seeds 对同一 pair 产生相同长度、相同 force trace 和相同结果，可用该 accepted-pair Oracle trajectory 作为定性动作展示，但论文 caption 必须准确说明来源。

仓库中还存在 `setup/figures/` 下 22 个 GIF 和 39 个 PNG，主要覆盖历史 wall、glass dose response、control、safe-abort、probe 和 intervention 实验。它们可用于整篇论文的前半部分，但不能冒充本轮 Pilot B/C 的新视频证据。

## 7. 已保存证据清单

### 7.1 轨迹数组

三个 accepted-pair 目录在 Quest 的 ignored result root 下保存：

```text
results/glass_recovery_v2/pilot_b_fresh_h20_20260812_r2/
  frontier_task0/h_20/{train,heldout}/<pair>__attempt_<hash>/
```

核心文件包括：

- `nominal_catastrophe.npz`：H=20 Base suffix；
- `oracle_recovery.npz`：安全任务完成 Oracle continuation；
- `off_path_control.npz`：matched off-path task-success continuation；
- `controller_state.npz`：环境、机器人 controller、observable/cache 状态；
- `precrash_onpath_state.npy`、`offpath_start_state.npy`：simulator state；
- `pair.json`、`nominal_replay.json`、`oracle_search.json`：准入与搜索记录。

每个 trajectory NPZ 同时包含：

- `images: [T,224,224,3]`；
- `hidden: [T,4096]`；
- `robot_state: [T,8]`；
- `nominal_action`、`target_action`、`executed_action: [T,7]`；
- 五个 risk horizons 的 targets/masks；
- `glass_force_after`；
- `time_to_catastrophe_actions`；
- `oracle_recoverable_from_this_state`、`latest_verified_recoverable_state`；
- `runtime_trigger_eligible`。

因此当前不仅有结果标签，也有训练/诊断需要的 observation、representation、state、action、force 和 temporal labels。

### 7.2 筛选与账本

- `frontier_rows.jsonl`：候选级 frontier 记录；
- `attempts.jsonl`：15 个 terminal attempts；
- `frontier_summary.json`：15→12→9→3 汇总；
- `collection_summary.json`：accepted/rejected accounting 和完整准入验证；
- `train.jsonl`、`heldout.jsonl`：三个 accepted pairs 的 schema-v2 trajectory manifests；
- `placements.json`：几何、source state、scene/geometry fingerprints 和预筛 metadata。

本地 tracked 摘要为：

- `results/glass_recovery_pilot_b_scoped_20260812.json`
- `results/glass_recovery_pilot_c_scoped_20260812.json`

### 7.3 Pilot C 记录

`eval_pilot_c_dev360_treatment_seed_17.json` 约 3.1 MB，保存：

- 6 个完整 episodes；
- 1596 个 action-level trace rows；
- 每步 nominal/executed/recovery action；
- 每步 risk vector、threshold 和 trigger ownership；
- 每步 glass/global force；
- task-success 和 catastrophe predicates；
- source-state、pair、seed、checkpoint/protocol identities；
- exact anchor restore evidence。

另外，每条 episode 都写出 `simulator_state.npy` 和 `controller_state.npz` trigger-state snapshot。当前没有 Pilot C RGB frame array。

### 7.4 日志与 provenance

| 项目 | 值 |
|---|---|
| Pilot B job | Quest `9095054`；完整 15-scene ledger 后停止 |
| stricter authoring job | Quest `9096065`；0 candidate，未进入 frontier |
| checkpoint training | Quest `9098769`，exit `0:0` |
| Pilot C evaluation | Quest `9099316`，exit `0:0`，elapsed `00:09:47` |
| Base revision | `962318cec55ac10993ff0f5f43eda9a270b4c873` |
| primary protocol SHA-256 | `0f531b939bbd02fec0f2ca5c3af9f82c386bf7ed0ee660b08f6f87a39dea0ae6` |
| checkpoint SHA-256 | `87ab4bee583b5792c930d2810be520b5e0049e5b8f849a1ec16b7fa56a08ca95` |
| Pilot C evaluation protocol SHA-256 | `5b748cdde10798a299658a9b11c8a94fb46cb5e08f4f8c812c5284f0c3d93af3` |
| Pilot C evaluation JSON SHA-256 | `1d7e071e2b2c0bdaddf8fbae08204df5d15940ad5754748f30f705e9e950f64f` |
| Pilot C run commit | `b16bce3a93aa7819a237ba376bb50a8281d4f066` |

Slurm logs和大体积 raw trajectories 保留在 Quest 的 gitignored workspace；tracked summaries、protocol/checkpoint/result hashes 和 pair/scene identities 已写入仓库。

## 8. 论文现在可以验证的具体命题

可以写：

> 在一个严格认证的、controller-compatible 的玻璃事故子类中，Base policy 的碰撞不是不可避免的。相同 H=20 状态存在安全且完成原任务的 recovery continuation；在两个 development source states 上，Oracle-timed recovery 的 6/6 重复均安全完成任务。

还可以写：

- 认证筛选是严格且可审计的，完整保留失败分母；
- 路径位置很重要，因为 accepted pair 要求 matched off-path task success；
- 恢复并非只做 safe-abort，而是完成原始 pick-and-place 任务；
- exact state/controller restore 在 6/6 Pilot C episodes 中稳定。

不能写：

- learned recovery controller 在 held-out 上成功；
- learned risk gate 能及时触发；
- 任意玻璃布局都能恢复；
- K=3 seeds 等于 6 个独立场景；
- 0.17 m / 0.30 m 严格几何已经跑通。

## 9. 投稿前建议准备

若按当前 scoped proof-of-concept 投稿，不再追加 GPU 实验也能形成报告闭环。建议完成：

1. 将筛选漏斗、Pilot C success matrix、force trace 和一张真实 filmstrip 选为正文/appendix figures；
2. 在 supplementary 中提供本报告重建的 GIF/MP4，并注明 derived from stored accepted-pair RGB frames；
3. 将独立样本数写为两个 development source states，seeds 作为 nested repeats；
4. 将 `controller-compatible, robustly recoverable glass accidents` 作为唯一恢复 claim；
5. 将 Broad Pilot B no-go 和 stricter authoring 0-candidate 结果放入 limitation/appendix，显示筛选边界而非隐藏失败；
6. 发布前归档 Quest raw roots 和 Slurm logs，生成外部备份清单，避免只有集群单副本。

如果摘要必须声称 learned recovery，则本报告明确显示仍缺 Pilot D/E/F：需要额外 disjoint train/validation/final-test pairs、修复 control exact-anchor observation restore、重新校准 learned risk gate，并分别验证 learned timing、learned actions 和完整闭环。

## 10. 本报告派生媒体说明

本报告的统计图由 tracked JSON 数值生成；filmstrip/GIF 由 accepted-pair NPZ 的真实 `images` 生成。生成脚本是：

```text
scripts/build_glass_recovery_progress_assets.py
```

派生媒体只用于展示，不替代 raw JSON/NPZ、hashes 或 Slurm provenance。
