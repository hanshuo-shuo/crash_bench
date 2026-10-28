# 当前状态（2026-10-27）

## 正在进行：生成式可行性 verifier

- **计划**：[generative_verifier/PLAN.md](generative_verifier/PLAN.md)。内容包括其他方法的启示、实验各阶段、预算和待决定事项。
- **用户 2026-10-16 的决定**：
  - V4 的 896 行（224 个四行单元、64 条源轨迹）作为开发数据，直接用于 verifier 训练。
  - 不再要求通过 0.75 的 shortcut 闸门。闸门失败（0.780）如实披露，并作为基线报告。
- **10-27 晚：HIL 完成，预注册读法 `ignores_rule`**（见 [V5_HIL_REPORT.md](generative_verifier/V5_HIL_REPORT.md)）。用户："写运行器和测试 → 草稿目录 smoke 16 次 → 填第 10 节、冻结 → 正式 640 次 → V5_HIL_REPORT.md"。160 个中途状态 × 4 条提示（同一种子）= 640 次 π0.5 执行：不可行规则下 160/160 抓着布丁越线（156 次照样完成），不给规则完成 157/160，中性句 156/160（没被搞乱），可行规则 156/160；可行与不可行规则的结果相同 0.950 [0.904, 0.987]；时机（4 个进度段）没有影响。可行规则"不越线"只是因为原路线本来就不越（C0 自然越线 0/160），"可行却不遵守"这次没测到。探索性：加一句话会让动作在约 18 步后分开，句子说了什么几乎不起作用。协议冻结于 `8d06be7`（实现 `5eb71fb`、`2f81374`、`0105502`）；运行目录 `/projects/p33100/siosio/crashbench_safelibero/verifier_dataset/20261028T024002Z_v5-hil_8d06be7c60a4`，作业 5697456（两个 A100 分片，各 1 小时 40 分）→ 5697474，全部 COMPLETED 0:0；不调 API，测试源未碰。没有正在运行的 CrashBench 作业。用户随后说"加上然后写下一步"：主报告页第 25 版加了 `#v5-hil` 和新的下一步；推荐的下一步是离线闸门（HIL 的 60 个验证源状态上拼出"π0.5 + S/P/A 闸门"，只读已有记录），协议草稿 [V5_GATE_OFFLINE_PROTOCOL.md](generative_verifier/V5_GATE_OFFLINE_PROTOCOL.md)，等用户点头。
- **10-27 晚：导师的问题，HIL 设计交接**。导师问 VLA 对任务中途人在环干预的鲁棒性。决定：不换题，把它当系统论文的动机和基线行；设计了"策略孪生"实验（同一中途状态、同一种子，四条提示：无规则 / 中性句 / 可行规则 / 不可行规则；160 个状态 × 4 = 640 次 π0.5 执行；读法 `ignores_rule` / `obeys_rule` / `breaks_on_prompt`，预测 `ignores_rule`），用户批准设计，交给下一个 agent 实现。协议 [V5_HIL_PROTOCOL.md](generative_verifier/V5_HIL_PROTOCOL.md)，结果见上一条。
- **10-27 S5 完成，预注册读法 `self_computes`**（见 [V5_S5_REPORT.md](generative_verifier/V5_S5_REPORT.md)）：组 S 在 P 的四个点之后自己写出叉积、正负号和结论，测试时没有代码参与。验证集（1,538 行反事实孪生）上 S 自己写的结论 0.923 [0.906, 0.938]，A 0.501，S − A +0.421 [+0.397, +0.446]；P 自己写的结论 0.500，P 的程序判定 0.940（S − P −0.017 [−0.030, −0.005]，1 个 seed）。判错的 119 行里 112 行是点找错了，计算出错只有 7 行；乘法是近似的（逐位对 59%，错的中位差 0.5%），但只要正负号，所以够用。1 个 seed，只看验证，测试集已用掉。没有正在运行的 CrashBench 作业。下一步（加 seed 714、715，约 22 GPU 小时；或别的选项）要用户决定。
- **方向（2026-10-27，用户决定）**："我还是觉得应该是先继续原来的想法 但是2可以保留下"。主线回到生成式 verifier：模型自己推理、自己下结论。P（模型定位、代码判定）不当作方法，只作诊断（它说明找位置学得会，卡在"从位置推出结论"）；用 P 做 AEGIS 闸门或系统保留为备选。主线的第一步是 S5（模型自己写出从位置到结论的计算），已完成，见上一条。
- **10-27**：检测器基线完成，预注册读法 `training_needed`（见 [V5_DET_REPORT.md](generative_verifier/V5_DET_REPORT.md)）：不训练、直接用 AEGIS 自带的 GroundingDINO 找四个物体再跑同一段程序，验证集只有 0.521，P 是 0.940（P − 检测器 +0.419 [+0.395, +0.443]）；主要是几乎找不到布丁（392 张图命中 3 张）。P 的位置监督是必要的，AEGIS 闸门的感知要用 P。
- **10-26**：V5 测试评估完成，预注册读法 `test_confirms`（见 [V5_TEST_REPORT.md](generative_verifier/V5_TEST_REPORT.md)）：在没见过的 4 条测试源上，"先写位置、再用程序判定"（P）3 个 seed 是 0.932 / 0.942 / 0.924，直接判断（A）都是 0.50；P − A +0.432 [+0.416, +0.449]。测试集只评估这一次，不再重评。没有正在运行的 CrashBench 作业。本轮已收尾，**接手先读 [V5_HANDOFF.md](generative_verifier/V5_HANDOFF.md) 开头的接手指南**（现状、可选的下一步、每一步的做法、关键路径）。
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
    主报告页第 14 版已同步到 S2。
  - S3（A 对 B 的 pilot，seed 713，只看验证）于 2026-10-25 完成，见 [V5_S3_REPORT.md](generative_verifier/V5_S3_REPORT.md)。协议 [V5_S3_PROTOCOL.md](generative_verifier/V5_S3_PROTOCOL.md) 在启动前冻结（`bdcf277`）。
    预注册读法 `both_shortcut`：反事实孪生上 A 0.501、B 0.503（B − A 区间 [−0.015, +0.020]），自然诊断上都是 0.89。
    两组连训练集都没超过"规则 × 进度"查表；B 的组合全对，错在两个"机器人侧"事实。零样本 3B、32B 也都在随机水平。
    用户同意先做拟合诊断（"做一下吧"）。
  - 拟合诊断于 2026-10-25 完成，见 [V5_FIT_REPORT.md](generative_verifier/V5_FIT_REPORT.md)（协议 [V5_FIT_PROTOCOL.md](generative_verifier/V5_FIT_PROTOCOL.md)，冻结 `ea76dab`）。
    预注册结论 `inconclusive`：16 行 3 个 seed 都背到 100%，64 行背到 94–98%（差一点没到"完全拟合"）；留出训练图上的迁移增益 +0.004，等于查表。
    B 能背但学不会读线；S3 的 B 在同样的行上也是查表水平。
  - **S4 于 2026-10-25 完成，本轮收尾**，见 [V5_S4_REPORT.md](generative_verifier/V5_S4_REPORT.md)（协议 [V5_S4_PROTOCOL.md](generative_verifier/V5_S4_PROTOCOL.md)，冻结 `20913c7`）。
    用户："换吧 继续 然后这一步干完先收尾 反正要尽力跑通故事线"。组 P 先写两个锚点、布丁、篮子在图上的位置，再由程序判断是否隔开。
    预注册读法 `positions_ground`：反事实孪生验证上 P 0.940，A 0.501，P − A +0.438 [+0.416, +0.461]；自然诊断 0.962。
    P 自己写的结论是常数，起作用的是"模型定位、程序判定"的分工。1 个 seed，只看验证集。
    交接见 [V5_HANDOFF.md](generative_verifier/V5_HANDOFF.md) 开头。
    用户 10-25："收尾下 然后让下一个agent开始干下一步"。加 seed（A、P 各补 714、715，只看验证）已获批，交给下一个 agent；测试评估仍需用户确认。
  - 主报告页 Ya7FQbZC9yEuNNyYhPBHHT 第 13 版加了"下一阶段：V5"一节，和上周的结论、后续工作逐条对照（结论：主线没有走偏，覆盖面暂时变窄）。精读页 DJPGiauxhPAC8BYE4L14Y5 第 5 版也有一节同样内容。第 18 版（10-26）在 S4 结果后加了"3 个 seed"一节（`#v5-seeds`）。第 19、20 版（10-26）在页首加了"现在在做什么（白话版）"：用一道真实的题讲当前任务，并说明 P 的"程序判定"是训练过的 VLM 写坐标、固定代码判定。
  - **加 seed 于 2026-10-26 完成**，见 [V5_SEEDS_REPORT.md](generative_verifier/V5_SEEDS_REPORT.md)（协议 [V5_SEEDS_PROTOCOL.md](generative_verifier/V5_SEEDS_PROTOCOL.md)，启动前冻结 `ee20b20`，实现 `1a056d6`）。
    A、P 各补 seed 714、715，只看验证；713 用 S3、S4 的记录。预注册读法 `stable`：
    P 的程序判定 0.940 / 0.947 / 0.933，A 0.501 / 0.499 / 0.500；3 个 seed 按行平均后 P − A = +0.440，图像 bootstrap 区间 [+0.427, +0.452]。
    A 每个 seed 都塌成一个常数（方向随 seed 变）；P 自己写的结论在 3 个 seed 上都是随机水平，结论仍来自程序。
    运行目录 `/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs/20261026T001735Z_v5-seeds_ee20b2033465`，作业 5343157–5343162 全部 COMPLETED 0:0，约 9.8 GPU 小时。
    下一步：在测试集上评估一次，需要用户确认，并先写测试协议和测试阶段的代码。交接见 [V5_HANDOFF.md](generative_verifier/V5_HANDOFF.md) 开头。
  - **测试评估于 2026-10-26 完成**，见 [V5_TEST_REPORT.md](generative_verifier/V5_TEST_REPORT.md)。用户 10-26 "都改一下 然后做测试级"。协议 [V5_TEST_PROTOCOL.md](generative_verifier/V5_TEST_PROTOCOL.md) 冻结于 `5594ceb`（实现 `72d825b`、`f071896`）。
    两次 smoke 只用 `validation_natural`：第一次发现评估时没开 TF32、P 的坐标复现不了；修正后 6 个 adapter 的输出逐字复现训练时的记录。
    预注册读法 `test_confirms`：测试集（1,154 行反事实孪生，4 条没见过的源）上 P 的程序判定 0.932 / 0.942 / 0.924，A 0.503 / 0.501 / 0.500；3 个 seed 按行平均后 P − A = +0.432，图像 bootstrap 区间 [+0.416, +0.449]。
    从验证到测试 P 只降 0.005–0.009；A 仍塌成常数，P 自己写的结论仍是随机水平。测试自然（80 行）P 0.958、A 0.713。
    运行目录 `/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs/20261026T173919Z_v5-test_5594cebeb616`；作业 5433550–5433770 共 8 个全部 COMPLETED 0:0，约 6 GPU 小时，6 个 adapter 的重新加载核对都是 32/32 逐字相同。
    **测试集已用过这一次，不要再评估或重跑。** 下一步（AEGIS 闸门、教模型自己算隔开、扩任务组合）都要用户另行决定。主报告页第 22 版加了测试结果一节（`#v5-test`）。
  - **检测器基线于 2026-10-27 完成**，见 [V5_DET_REPORT.md](generative_verifier/V5_DET_REPORT.md)。用户 10-26 "先做1"。协议 [V5_DET_PROTOCOL.md](generative_verifier/V5_DET_PROTOCOL.md) 冻结于 `380d632`（实现 `988f9fe`，sbatch 修正 `293a299`）。
    问题：不训练，直接用 AEGIS 自带的 GroundingDINO（同一份权重）在第 1 张图上找布丁、篮子和两个锚点，再跑 P 的同一段程序，能不能判对孪生题。
    预注册读法 `training_needed`：验证集 1,538 行上检测器 0.521 [0.499, 0.546]，P（3 个 seed 平均）0.940，P − 检测器 +0.419，图像区间 [+0.395, +0.443]。
    布丁几乎找不到（最高分框命中 392 张里的 3 张，前 5 个框里有布丁的也只有 19%）；锚点命中 47%（红杯子、酒瓶、橙汁盒约九成，字母汤罐、沙拉酱瓶、黄色书 6–21%）；篮子 95%。训练集上 48 组框到点的比例都在 0.515–0.536；AEGIS 原规则 0.500。
    运行目录 `/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs/20261027T031559Z_v5-det_380d63210941`；作业 5520821（preflight）、5520823（检测，A100，15 分钟）、5520824（report）全部 COMPLETED 0:0。不调 API、不训练、不读测试集。
    下一步（AEGIS 闸门设计稿，感知用 P；或补更强的现成检测器）要用户决定。
  - **S5（模型自己算出是否隔开）于 2026-10-27 完成**，见 [V5_S5_REPORT.md](generative_verifier/V5_S5_REPORT.md)。用户定方向："我还是觉得应该是先继续原来的想法 但是2可以保留下"，然后"我觉得很好 帮我开始吧"，"弄完后自动帮我交gpu的训练"。协议 [V5_S5_PROTOCOL.md](generative_verifier/V5_S5_PROTOCOL.md) 冻结于 `f20304e`（实现 `e62ce81`、`017232a`）。
    组 S 的目标：P 的四个点，然后向量、两个位移、两个叉积（两个乘法、一个减法）、两个正负号、`Separated`、`Verdict`；读数看模型自己写的 `Verdict`，程序只作诊断。其余设置和 S4 的 P 相同（seed 713、448 次更新、同一样本顺序）。
    预注册读法 `self_computes`：验证 0.923 [0.906, 0.938]，A 0.501，S − A +0.421 [+0.397, +0.446]；验证自然 0.949；孪生图全对 0.794（P 0.813，A 0.003）。
    自己写的结论与"用自己的点跑程序"99.0% 一致；错的 119 行里 112 行是点错，7 行是计算错。乘积逐位正确只有 59%，错的中位差 0.5%，叉积正负号 3,076 个里只翻了 11 个。比 P 的程序判定低 0.017（区间 [−0.030, −0.005]），差在定位，1 个 seed 分不清是否是波动。
    运行目录 `/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs/20261027T050745Z_v5-s5_f20304e222b9`；作业 5554027（preflight）、5554028（qgpu0403，A100，10 小时 50 分；每次生成约 12.4 秒）、5554029（report）全部 COMPLETED 0:0。不调 API，不读测试集。
    下一步要用户决定：加 seed 714、715（只看验证）；干净的测试要新的留出源（要另批数据）。

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

## 分支（2026-10-26 清理）

| 分支 | 用途 | 状态 |
|---|---|---|
| `main` | 默认分支，也是工作线 | 10-26 快进到 verifier 的全部工作（第 1 步、对照、梯级、V5 到测试评估），并收入了用户在网页上写的 `docs/new_report.md`；10-27 快进到检测器基线（`v5-det` 分支）、S5（`v5-s5` 分支）和 HIL（`v5-hil` 分支） |
| `codex/archive-crashbench-20260926` | 2026-09-26 之前的 CrashBench 计划存档 | 只读，AGENTS.md 引用；内容已在 `main` 的历史里 |
| `codex/feasibility` | 10 月初可行性研究的发布分支 | 只读；`analysis/uncertainty/`、`experiments/reach_avoid/`、`experiments/feasibility_contract/` 等 20 多个启动器按这个名字核对来源，不要删除 |

- **以后的新工作**：从 `main` 开一个新分支，做完、推送后再快进 `main`。V5 的启动器只要求 commit 在某个 `origin/` 分支上，`main` 就满足。
- **10-26 用户决定的清理**（"main 直接推进吧"，"删了吧 就这样清理"），不丢任何提交：
  - 删除 `codex/generative-verifier-pilot`（本地和远端）：快进后和 `main` 完全相同。以前记录里的 commit 号都还在 `main` 上。
  - 删除 `codex/paired-repeats`（远端）：它比 `main` 多的 7 个提交只改了 `docs/new_report.md`，已用 cherry-pick 放进 `main`（作者保留为 CG）。原来的分支头保存为标签 `archive/paired-repeats-20261013`。旧的 paired-repeats 启动器按名字读取 `origin/codex/paired-repeats`，现在跑不了；要复跑需先改启动器，并需要用户批准。
  - 删除只在本地的 `agent/task-phase-transfer`，以及两个 7 月的 stash、一个 8 月的游离工作树 `~/crashbench-pilotb-repair-20260811`。删除前都存成了标签并推到 GitHub：`archive/task-phase-transfer-20260719`、`archive/stash-selfreport-probe-20260719`、`archive/stash-cluster-notes-20260725`（stash 用 `git stash apply <标签>` 恢复）、`archive/pilotb-repair-20260811`（8 个不在任何分支上的 Pilot B 修复提交）。
- 更早（10-17）删除的三个已并入的本地分支：`codex-repo-hygiene-provenance`、`codex/glass-recovery-d0-fprime-rescue`、`codex/odur-repair`。

## 历史与清理记录

- 2026-10-16 之前的完整时间线（逐条记录作业、失败与修复）：[STATUS_HISTORY_until_20261016.md](STATUS_HISTORY_until_20261016.md)。
- 2026-10-16 的清理记录：[CLEANUP_20261016.md](CLEANUP_20261016.md)。
- 2026-09-26 的 CrashBench 归档：`codex/archive-crashbench-20260926` 分支，以及 [CLEANUP_MANIFEST.json](CLEANUP_MANIFEST.json)。
