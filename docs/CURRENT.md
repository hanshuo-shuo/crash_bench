# 当前状态（2026-10-17）

## 正在进行：生成式可行性 verifier

- **计划**：[generative_verifier/PLAN.md](generative_verifier/PLAN.md)。内容包括其他方法的启示、实验各阶段、预算和待决定事项。
- **用户 2026-10-16 的决定**：
  - V4 的 896 行（224 个四行单元、64 条源轨迹）作为开发数据，直接用于 verifier 训练。
  - 不再要求通过 0.75 的 shortcut 闸门。闸门失败（0.780）如实披露，并作为基线报告。
- **当前阶段**：Phase 0–2 已于 2026-10-17 完成，见 [PHASE1_REPORT.md](generative_verifier/PHASE1_REPORT.md) 和 [PHASE2_REPORT.md](generative_verifier/PHASE2_REPORT.md)。
  - 锁定测试集（评估一次）：B（事实→判定）79.7%，A（直接）50.0%，差 +29.7 个百分点，源聚类 95% CI [+22.3, +36.8]，科学标准达成。
  - 开发里程碑未达成：B 行准确率 79.7%、单元全对 54.2%、规则翻转 64.2%，门槛分别是 85% / 70% / 90%。
  - B 的错误全部出在两个包含关系的预测上（判定与"模型自己的关系 + 程序规则"逐行一致）；A 和 C 的直接读法都是常数预测（50%），在训练集上就没拟合；C-CoT 62.8%。
  - 2026-10-17 Astra 独立评审：复算结果一致；"收益来自测试时推理"等过度表述已撤回；调整后的计划见 [NEXT_STEPS.md](generative_verifier/NEXT_STEPS.md) §7。
  - 原始测试报告误判"里程碑达成"，已用只读更正记录修正（PHASE2_REPORT §5）。Phase 3 等用户决定；没有正在运行的 CrashBench 作业。
  - 带图的报告：[PHASE2_REPORT.html](generative_verifier/PHASE2_REPORT.html)（含方法的文献来源；10-18 加入第 1 步和新的下一步，与 claude.ai 上的报告页同步）。
  - **下一步讨论稿**（用户、Astra、Claude 三方）：[NEXT_STEPS.md](generative_verifier/NEXT_STEPS.md)。
- **第 1 步（A 基线能否拟合）**：2026-10-18 完成，见 [STEP1_REPORT.md](generative_verifier/STEP1_REPORT.md)。
  - 用了 1.15 A100 小时，只读训练集，没有调用 API。
  - A 在 512 个训练行上停在随机水平，连 16 行都拟合不了；B 在第 80 次更新时就拟合了 16 行。
  - 按决策表记为 `implementation_suspect`。按规定检查了 direct 目标的编码、掩码和损失，没有发现错误。
  - 10-18 独立复核后的结论：中间事实监督改善了这种图文联合判断在当前设置下的可训练性；不能说直接判别普遍较弱，原因（梯度机制、显式条件化或其他）未定。初稿"缺少一阶梯度信号"的说法过强，已改。
  - 补充核对：随规则翻转的那个包含事实在全部 512 个训练行上等于判定或其取反，B 先学会不变的事实、再学会翻转的事实。
  - 下一步：16 行单通路对照 t、s、A+P（用户 10-18 认可范围，梯度测量后置），约 1.1–1.2 A100 小时；协议见 [CONTROLS_PROTOCOL.md](generative_verifier/CONTROLS_PROTOCOL.md)，计划见 [NEXT_STEPS.md](generative_verifier/NEXT_STEPS.md) §8。代码已准备，作业等用户确认后提交。

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
  - V4 verifier 的 Phase 1 API 基线（`openai/gpt-6-luna`，384 次调用）已用 $0.4669（上限 $3），计入用户方法的每周预算。
  - key 不进入 Git，也不进入日志。

## 分支（2026-10-17 整理）

| 分支 | 用途 | 状态 |
|---|---|---|
| `main` | 默认分支，稳定线 | 快进到 verifier Phase 0–2 的最新状态 |
| `codex/generative-verifier-pilot` | verifier 工作线 | 与 `main` 相同；后续工作可以继续在这里，或每个阶段开新分支 |
| `codex/archive-crashbench-20260926` | 2026-09-26 之前的 CrashBench 计划存档 | 只读，AGENTS.md 引用 |
| `codex/feasibility` | 10 月初可行性研究的发布分支 | 只读；`analysis/uncertainty/`、`experiments/reach_avoid/`、`experiments/feasibility_contract/` 的启动器靠它核对来源，不要删除 |
| `codex/paired-repeats` | paired repeats 实验，以及用户在网页上写的 `docs/new_report.md`（10-06 至 10-13） | 有 7 个独有提交，没有并入 `main`；是否合并由用户决定 |

Quest 上只在本地存在的 `agent/task-phase-transfer`（1 个独有提交和 1 个 stash）保留未动。
三个已完全并入、远端已删除的本地分支已删除，不丢任何提交：`codex-repo-hygiene-provenance`、`codex/glass-recovery-d0-fprime-rescue`、`codex/odur-repair`。

## 历史与清理记录

- 2026-10-16 之前的完整时间线（逐条记录作业、失败与修复）：[STATUS_HISTORY_until_20261016.md](STATUS_HISTORY_until_20261016.md)。
- 2026-10-16 的清理记录：[CLEANUP_20261016.md](CLEANUP_20261016.md)。
- 2026-09-26 的 CrashBench 归档：`codex/archive-crashbench-20260926` 分支，以及 [CLEANUP_MANIFEST.json](CLEANUP_MANIFEST.json)。
