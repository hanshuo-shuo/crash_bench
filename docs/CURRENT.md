# 当前状态（2026-10-25）

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
- **单通路对照**：2026-10-18 完成（作业 4371785，58 分钟，只读训练集），见 [CONTROLS_REPORT.md](generative_verifier/CONTROLS_REPORT.md)。
  - t（只看文字）、s（只看图像）、A+P（先复述颜色再判断）三组的 9 次运行全部拟合了 16 行；预注册读法为 `composition_bottleneck` 和 `restatement_restores_fit`。
  - 结论（16 行）：direct 路径学得会两个单一模态的标签，学不会它们的 XOR；先写出受保护颜色，16 行就能拟合。
  - 10-19 更正：判定前只加 4 个固定 token（AF）也能拟合这 16 行，所以 A+P 的拟合靠的是多出的输出位置，不是颜色（见下一条）。
- **第 2 步梯级对照**：2026-10-19 完成（作业 4399441，2 小时 36 分，只读训练集和验证集），见 [LADDER_REPORT.md](generative_verifier/LADDER_REPORT.md)。
  - 512 行上，AF 和 AP 在训练集和验证集上都停在 50%，B 是 84.5%（AP − B = −34.7 个百分点，区间 [−39.7, −30.1]）。
  - 结论：让判定可学并能推广的，是关系事实的中间监督，不是输出结构或复述颜色。
- **初步故事线已整理收尾**，见 [PRELIMINARY_STORY.md](generative_verifier/PRELIMINARY_STORY.md)。
- **2026-10-24 方向调整：V5**，见 [V5_PLAN.md](generative_verifier/V5_PLAN.md)。
  - 用户决定不做"收口批"（基本是反复验证），改为把环境做得不那么 toy：规则用自然语言指向真实物体、一个场景多条规则、自然分布训练、反事实孪生测试。
  - 第 0 步（只读盘点）已完成：1,600 个官方初始状态里有 838 个有历史 proxy 安全成功，V4 只用了 67 个源。
  - 第 1 步（CPU 几何摸底）2026-10-24 完成，见 [V5_STEP1_REPORT.md](generative_verifier/V5_STEP1_REPORT.md)：
    初始布局上只看文字就能猜对 99.9%；"做到一半加一条新规则"让标签依赖图像，按文字 × 进度段配平后 616 行、43 个源，读法 `proceed_to_render_pilot`。
    产出集中在 object 任务，只有"两物体连线"模板可用。
  - 第 2 步（小样本渲染看图）2026-10-24 完成，见 [V5_STEP2_REPORT.md](generative_verifier/V5_STEP2_REPORT.md)：
    Claude 盲看 20 个孪生单元的 40 个判断，回答 34 个、全部正确，读法 `relation_visible`。
    6 个"看不出"对应三处设计修改（终点所属物体排除出锚点、用 checkpoint 时刻的几何、提高余量或加俯视图）。
  - 用户 2026-10-24 批准设计修改并采用 V5 计划，正式采集协议见 [V5_COLLECTION_PROTOCOL.md](generative_verifier/V5_COLLECTION_PROTOCOL.md)。
  - C0（修正后重新回放 V4 的 67 条源）2026-10-24 完成，见 [V5_C0_REPORT.md](generative_verifier/V5_C0_REPORT.md)：
    修正都生效了，但严格层孪生从 53 降到 0。固定布局里，"标签要看图"与"图上看得清"互相冲突。
  - 用户选了方向 3（反事实布局）。三轮 smoke 见 [V5_LAYOUT_PROTOCOL.md](generative_verifier/V5_LAYOUT_PROTOCOL.md)：
    v3 的合法率 0.72，每个合法布局 8.3 个静置孪生，只看文字的上限 0.705，严格层孪生 249 个。
    但 8 条源只有 4 条能产出孪生，其余 4 条在结构上做不到（可挪锚点太少或规则太少）。
  - 用户决定只保留能产出孪生的任务组合、不扩展，先走通最小故事闭环（[V5_SCREEN_PROTOCOL.md](generative_verifier/V5_SCREEN_PROTOCOL.md)）。
  - S0 筛查 2026-10-24 完成，见 [V5_S0_REPORT.md](generative_verifier/V5_S0_REPORT.md)：
    38 条源、6 个任务组合可用；预注册读法 `shortcut_survives`（查表 0.615 > 0.60）。
    捷径只在 goal 和 spatial 任务上残留；严格配平后能挡住捷径的测试几乎只有 object/II/task1（验证、测试各 87 个孪生单元）。
  - 用户 10-24 选择先用 object/II/task1 一个任务走通故事，本轮收尾。**交接见 [V5_HANDOFF.md](generative_verifier/V5_HANDOFF.md)**。
  - 10-24 用户决定训练只用 object/II/task1，并采用三项建议：训练加入训练源的反事实布局；B 沿用 V4 写法（物体、篮子是否在连线靠机器人一侧）；S1 加俯视图。
  - S1 于 2026-10-25 完成，见 [V5_S1_REPORT.md](generative_verifier/V5_S1_REPORT.md)：
    验证、测试各 392 / 384 个严格孪生单元（`s1_supply_ok`）；两轮盲看在判定和两个事实上都是 40/40（`relation_visible`、`facts_readable`）；
    俯视图没有带来提升，S2 用两张斜视图。
  - S2 于 2026-10-25 完成，见 [V5_S2_REPORT.md](generative_verifier/V5_S2_REPORT.md)：
    4,643 行、1,750 对图像（`s2_dataset_ready`）；训练集的"规则 × 进度"查表 0.760，验证和测试按构造 0.500，自然诊断 0.747 / 0.912；测试已冻结。
    S3（A 对 B 的 pilot）等用户确认。没有正在运行的 CrashBench 作业。
  - 主报告页 Ya7FQbZC9yEuNNyYhPBHHT 第 13 版加了"下一阶段：V5"一节，和上周的结论、后续工作逐条对照（结论：主线没有走偏，覆盖面暂时变窄）。精读页 DJPGiauxhPAC8BYE4L14Y5 第 5 版也有一节同样内容。
  - 没有正在运行的 CrashBench 作业。

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
