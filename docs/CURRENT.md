# 当前状态（2026-11-15）

这一页只放重点。逐条的历史记录在 [STATUS_HISTORY_until_20261112.md](STATUS_HISTORY_until_20261112.md)。

## 一句话

问题已经找到并量化：π0.5 在任务中途换指令，只有 22.8% 换得过去，从起点换是 98.0%；"停下""回原点""别做了"照做不到 1%。现在的目标是**在模型里解决它**，做成一篇方法型论文。11-15 第 1 步出了结果：换指令时先用脚本把手臂退回起始位姿，再让 π0.5 做新任务，88.3% 做成。11-14 用户选定方向：用一次后训练同时解决几种中途指令（换任务是主结果，停下、回原点、放下是覆盖面），并要求先把地基打稳。

## 下一步

计划和交接都在 [HIL_METHOD_PLAN.md](generative_verifier/HIL_METHOD_PLAN.md)，下一个 agent 先读它的第 0 节。

| 步 | 做什么 | 状态 |
|---|---|---|
| 0 | 普查报告、两处更正、文献核对、提交推送 | **11-14 做完** |
| 1 | "退回再做"：换指令时用脚本放下并退回起始位姿，再让 π0.5 做新任务（[方案](generative_verifier/HIL_RETREAT_PROTOCOL.md)） | **11-15 做完**（[报告](generative_verifier/HIL_RETREAT_REPORT.md)）：退回起始位姿 88.3%，只放下、抬高 30.5%，不干预 22.8% |
| 2 | 用模型自己生成的数据做后训练，一次教几种中途指令 | **没批准**，只有设计草图。**等用户定**：要不要开始训练 |
| 旁支 | 单遍修复（[设计](generative_verifier/HIL_MECHANISM_SINGLEPASS_HANDOFF.md)） | 11-14 挂起（用户同意），用户说了再开始 |

队列里现在没有 CrashBench 作业。

## 几个词

| 词 | 意思 |
|---|---|
| A、B | 旧任务、新任务。"A→B"是一个有序的任务对，LIBERO-Goal 有 10 个任务、90 对 |
| f | 换指令的时点：旧任务自己做完所需步数的比例。f = 0 是起点，普查测了 0.5 和 0.75 |
| 状态 | 一个（任务对，初始布置，时点） |
| 格 | 一个（任务对，时点），里面有若干初始布置。不干预时换成功率 ≤ 0.5 的叫失败格，≥ 0.9 的叫成功格 |
| 试点对 | 最早研究的三对：1→8、7→8、2→9。层和剂量都是在它们上面定的 |
| 印记、γ | 同一张图配新旧两条指令，动作头读到的图像 token 上的值相差多少，叫印记；γ 是它相对起点的比例，小于 1 就是变淡 |
| amp2，"放大 2 倍" | 把这个差放大到 2 倍。每个控制步要多跑一遍旧指令 |
| 第 12–14 层 | 闭环实验里动的三层。开环里单层作用最大的是第 14 层，最好的三层窗口有 12–14 和 14–16 两个 |

## 现在知道什么

哪些站得住、哪些不能写进论文，以 [地基审查](generative_verifier/HIL_FOUNDATION_AUDIT.md) 为准。

| 结论 | 稳不稳 |
|---|---|
| LIBERO-Goal 全部 90 对：中途换 22.8%，起点换 98.0%；同一物体的对 59.6%，不同物体的对 16.2% | 稳。Causeway 在同一个模型和任务集上也报告了这个现象（26.0%） |
| 这些中途状态下新任务做得成：先用脚本退回起始位姿，88.3% 做成（不同物体的对 91.0%）；只放下、抬高 10 cm 是 30.5% | 稳（冻结读法，1,764 个状态）。它是上限和最简单的基线，不是方法。不说明模型在原状态下"听没听见" |
| 旧任务改了场景时退回也没用：旧任务是"打开上层抽屉并把碗放进去"时 29.6%，其余九个旧任务 94.3% | 事后分析。原因是推测（抽屉开着的画面训练里没见过），没有对照 |
| 动作头经第 12–14 层的图像 token 读任务；把旧指令的值写进去，机器人就做旧任务（230 → 6 / 232） | 稳。没有别的层的对照 |
| 失败格里这条路上的指令印记更淡（0.66 对 1.22） | 弱。在不同物体的任务对里几乎分不出成败 |
| 把印记放大 2 倍能修复 | 只在试点对上（17 → 64 / 90；同一批状态重跑三次是 17–20 → 64–67）。全部状态上是 22.8% → 26.7% |
| 不干预时做完旧任务的 810 个状态，放大后 379 个不再做完旧任务；其中 93 个做成新任务，286 个两个都没做成 | 事后分析。这 286 个多数没有去新任务那边（到过新任务干活位置附近的 26%，和不理新指令的运行一样）。意思是：放大多数时候只是把旧任务打断。11-12 写的"听见了但做不了"撤回 |
| 别人已经做到：换任务，Causeway 不训练 26.0% → 61.6%，ReSteer 靠训练（做法和我们计划的后训练几乎一样）；停下，ELLSA、VITA-E 靠专门结构做到 94–100% | 11-14 读全文核对过这五篇（含 Hi Robot）。没有哪一篇在同一个扁平 VLA 上同时做换任务、停下、回原点 |

## 已完成的研究（结果只读，不重跑）

**中途换指令的机制（11-04 至 11-12）**

| 批次 | 一句话 | 报告 |
|---|---|---|
| 开环定位 | 动作头在第 14 层读图像 token，在第 16 层读文字 token | [报告](generative_verifier/HIL_MECHANISM_REPORT.md) |
| 闭环因果 | 试点失败格放大 2 倍 64 / 90，不干预 17，随机方向 22；放大 4 倍过量 | [报告](generative_verifier/HIL_MECHANISM_CLOSED_LOOP_REPORT.md) |
| 修复批次 | 动作上匹配的推动 21 / 90；留出对的增益约为试点的四成 | [报告](generative_verifier/HIL_MECHANISM_REPAIR_REPORT.md) |
| 神经元 | 新指令经第 9–13 层 MLP 整体写入，分散在几千个神经元上；前几名可解释但不是修复位置 | [报告](generative_verifier/HIL_MECHANISM_NEURONS_REPORT.md) |
| 留出对开环 | 印记变淡在三个任务集上都有；开环的动作变化预测不了闭环 | [报告](generative_verifier/HIL_MECHANISM_HELDOUT_OPEN_REPORT.md) |
| 单元补测 | 写进新指令一半要约 1 万个单元，挡住一半约 300 个 | [报告](generative_verifier/HIL_MECHANISM_UNITS_REPORT.md) |
| 反向剂量 | 拿掉这条路 230 → 6 / 232；只调弱一半 201 / 232 照样换 | [报告](generative_verifier/HIL_MECHANISM_DOSE_REPORT.md) |
| 任务对普查 | 90 对里 75 对在后段失败；放大在失败格上 8.0% → 13.1%，多数时候只是把旧任务打断 | [报告](generative_verifier/HIL_MECHANISM_CENSUS_REPORT.md) |
| 退回再做（11-15） | 先退回起始位姿再做新任务 88.3%；只放下、抬高 30.5%；旧任务改了场景时退回没用 | [报告](generative_verifier/HIL_RETREAT_REPORT.md) |

**人在任务中途干预（10-27 至 11-04）**

| 研究 | 一句话 | 报告 |
|---|---|---|
| 中途加规则 | π0.5 不理会中途加的规则 | [报告](generative_verifier/V5_HIL_REPORT.md) |
| 离线闸门 | 把判定放在策略之外能拦住 | [报告](generative_verifier/V5_GATE_REPORT.md) |
| 回原点、绕路 | 让它回原点、让它绕开禁区，都不照做 | [回原点](generative_verifier/V5_RETURN_REPORT.md)、[绕路](generative_verifier/V5_DETOUR_REPORT.md) |
| HIL 套件 | π0.5 和 OpenVLA-OFT 不理会停止、返回、取消；π0.5 被推开后能恢复 | [报告](generative_verifier/HIL_SUITE_REPORT.md)。其中 Goal 上"早换基本能换"是手挑的 8 对，普查不支持（报告开头有 11-14 的更正说明） |
| 语言消融 | 指令换成空的或乱码，成功率 96.75% → 约 50%；Goal 上 96% → 约 9% | [报告](generative_verifier/LANGUAGE_ABLATION_REPORT.md) |

**生成式 verifier（10-16 至 10-27）**

| 研究 | 一句话 | 报告 |
|---|---|---|
| V4，Phase 0–2 | 关系事实让题目学得会；锁定测试已评一次，不再评 | [Phase 2](generative_verifier/PHASE2_REPORT.md)、[初步故事](generative_verifier/PRELIMINARY_STORY.md) |
| V5 线规则 | 先写位置再用程序判定 0.940，直接判 0.501；测试集确认，已评一次，不再评 | [测试报告](generative_verifier/V5_TEST_REPORT.md)、[交接与历史](generative_verifier/V5_HANDOFF.md) |

**更早（09-26 至 10-15）**：SafeLIBERO 复现、可行性研究、红色围栏、reach-avoid、verifier 工程 pilot、FD1000 数据集。表格在[历史文件](STATUS_HISTORY_until_20261112.md)的"已完成的研究"一节。

**论文方向与文献**：[11-02 的方向底稿](generative_verifier/PAPER_DIRECTIONS.md)；[10-31 的文献定位](literature/reports/VLA%20中途人类干预文献定位.md)；11-14 核对过的文献对照在[计划](generative_verifier/HIL_METHOD_PLAN.md)的附录 A。报告页是 `generative_verifier/PHASE2_REPORT.html`。

## 规则与资源

- **工作规则**：[AGENTS.md](../AGENTS.md)、[QUEST_WORKFLOW.md](../QUEST_WORKFLOW.md)、[setup/README.md](../setup/README.md)。
- **存储与代码**：大文件根目录是 `/projects/p33100/siosio/crashbench_safelibero/`；唯一的代码仓库是 Quest 上的 `$HOME/crash_bench`。
- **分支**：现在的工作分支是 `hil-mechanism`（已推送，比 `main` 多 46 个提交，`main` 还没快进）。`codex/archive-crashbench-20260926` 和 `codex/feasibility` 只读，不要删。分支清理的细节在历史文件里。
- **OpenRouter 预算**：每周 $100，其中 $95 留给用户自己的方法。已用：SafeLIBERO 复现 $3.80，FD1000 $0.0588，V4 的 API 基线 $0.4669。这条线不调 API。key 不进 Git，也不进日志。

## 历史

- 到 2026-11-12 为止的逐条状态记录：[STATUS_HISTORY_until_20261112.md](STATUS_HISTORY_until_20261112.md)。
- 每次授权的原话和范围：[AUTHORIZATION_HISTORY_until_20261112.md](AUTHORIZATION_HISTORY_until_20261112.md)。
- 2026-10-16 之前的时间线：[STATUS_HISTORY_until_20261016.md](STATUS_HISTORY_until_20261016.md)；清理记录 [CLEANUP_20261016.md](CLEANUP_20261016.md)。
