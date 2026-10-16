# 当前状态（2026-10-16）

## 正在进行：生成式可行性 verifier

- **计划**：[generative_verifier/PLAN.md](generative_verifier/PLAN.md)。内容包括其他方法的启示、实验各阶段、预算和待决定事项。
- **用户 2026-10-16 的决定**：
  - V4 的 896 行（224 个四行单元、64 条源轨迹）作为开发数据，直接用于 verifier 训练。
  - 不再要求通过 0.75 的 shortcut 闸门。闸门失败（0.780）如实披露，并作为基线报告。
- **当前阶段**：Phase 0（数据导出与代码改动）尚未开始；没有正在运行的 CrashBench 作业。

## 已完成的研究（结果只读）

| 时间 | 研究 | 一句话结论 | 详细记录 |
|---|---|---|---|
| 09-26 – 09-28 | SafeLIBERO 复现：π0.5 与完整 AEGIS 的官方矩阵 | 64/64 cells、3,200 个 episode 全部完成；API 实报加预留共 $3.80（上限 $5） | [REPRODUCTION.md](REPRODUCTION.md)、[reproduction/BATCH_RUN.md](reproduction/BATCH_RUN.md)、`results/safelibero_batches/20260927T204911Z_d22035816647/COMPLETE.json` |
| 10-01 – 10-02 | 安全但未完成的状态：可行性见证与受控续接 | 共 1,209 次执行；Object5 上，参考策略在等预算条件下成功率下降。这是"相对专家能力"的证据，不能证明真的不可行 | [feasibility/CURRENT.md](feasibility/CURRENT.md) |
| 10-02 – 10-03 | 可行性契约 pilot | 5 个 held-out 布局有合法、无接触的成功见证；没有得到非平凡的"合法但不可达"负例 | [feasibility_contract/CURRENT.md](feasibility_contract/CURRENT.md) |
| 10-03 | 红色围栏 | 得到一对经过认证的开/闭对照；两个带安全提示的 π0.5 都碰到了受保护的酒瓶 | [red_enclosure/CURRENT.md](red_enclosure/CURRENT.md) |
| 10-03 | FR-1 / FR-1B | 6 个有安全见证的状态上，π0.5 全部碰到酒瓶；支持区分"解是否存在"和"策略会不会失败" | [feasibility_risk/CURRENT.md](feasibility_risk/CURRENT.md) |
| 10-03 | Reach-avoid 分组研究 | 168 个状态：72 可行、48 条件不可行、48 UNKNOWN。π0.5 在 72 个可行状态中只完成 13 个；基于 π0.5 表征的读数在 held-out 笼子场景上 AUROC 0.500 | [reach_avoid/RESULTS.md](reach_avoid/RESULTS.md)、[中文摘要](reach_avoid/RESULTS_ZH.md) |
| 10-10 – 10-12 | Verifier 工程 pilot（120 行，单一源场景） | 直接判别组训练集 120/120。生成式组 73/120，修正 verdict token 和损失后 97/120。rank64 和低学习率都没有带来改善 | [verifier_training/README.md](../verifier_training/README.md) |
| 10-14 – 10-15 | FD1000 数据集 V1–V4 | 四个版本都没有通过 0.75 闸门；V4 最终 896 行；诊断显示伪迹集中在 EXIT × checkpoint 0 | [verifier_dataset/CURRENT.md](../verifier_dataset/CURRENT.md) |

## 规则与资源

- **工作规则**：[AGENTS.md](../AGENTS.md)、[QUEST_WORKFLOW.md](../QUEST_WORKFLOW.md)、[setup/README.md](../setup/README.md)。
- **存储与代码**：大文件根目录是 `/projects/p33100/siosio/crashbench_safelibero/`；唯一的代码仓库是 Quest 上的 `$HOME/crash_bench`。
- **OpenRouter 预算**：
  - 每周 $100，其中 $95 留给用户自己的方法。
  - SafeLIBERO 复现已用 $3.80（含预留，上限 $5）。
  - FD1000 campaign 已用 $0.0588（上限 $20）。
  - key 不进入 Git，也不进入日志。

## 历史与清理记录

- 2026-10-16 之前的完整时间线（逐条记录作业、失败与修复）：[STATUS_HISTORY_until_20261016.md](STATUS_HISTORY_until_20261016.md)。
- 2026-10-16 的清理记录：[CLEANUP_20261016.md](CLEANUP_20261016.md)。
- 2026-09-26 的 CrashBench 归档：`codex/archive-crashbench-20260926` 分支，以及 [CLEANUP_MANIFEST.json](CLEANUP_MANIFEST.json)。
