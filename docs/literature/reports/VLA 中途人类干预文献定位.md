# 推得回来，叫不停：HIL 发现的文献坐标

**结论先说**：我们这组 HIL 结果大约一半已经有人做过，另一半在本轮检索里找不到先例。论文应当把后一半写进标题，把前一半当作复现和对照。已经有人做过的有三块。第一，LIBERO 上的 VLA 在 t = 0 就不怎么看指令：LIBERO-PRO、LIBERO-Plus 都报过，Grant et al. 还测到 π0.5 被告知 "do not {task}" 时 **96.7%** 照做。第二，任务中途换成另一条训练过的 LIBERO-Goal 指令会失败，而且越晚越难：SwitchVLA、ReSteer、Causeway 都报过。其中 Causeway 在 71 个跨物体对上的平均裸切换成功率，π0.5 只有 **26.0%**，OpenVLA-OFT 只有 **3.4%**。第三，闭环策略能从挪物体、推手臂中恢复：从 Diffusion Policy 到 PI 自己的 π0、π0.5 博客都展示过，2026 年 VLA-Corrector 还在同一个 π0.5 LIBERO checkpoint 上给出了计数。MotorMind 的一张附表甚至已经给出同一个 π0.5 "中途挪物体 70%、中途换指令 20%" 的不对称。

看起来新的有四块。一是在没有专门训练的现成 VLA 上，于任务中途说"停下 / 回原点 / 别做了"这类训练外命令，并把"原任务照样完成"当主指标：每个策略约 0/235 照做，原任务完成率是不干预时的 0.88–1.01 倍。二是任务中途追加空间规则：越线 160/160，进入禁区 168/168。三是用 policy twin 在同一步把说话式干预和动手式干预并排比较。四是按种子噪声归一化、带时间走向的动作块探针。

推荐的主张是：**现成的、按基准微调的 VLA，被推开能恢复，被叫停却不停，所以纠正和停止必须在策略之外执行。** 这给 verifier 加外部 gate 的主线补上了文献里缺的那条证据。已有很多工作主张在策略外执行安全约束，但没有一篇的理由是"VLA 不执行人善意地说出的话"。代价是三处必须收窄。第一，物理恢复只能说到 π0.5。第二，"不听"只能指自然、善意的人话，因为对抗文本能操纵 VLA，我们自己的探针也显示新句子确实改变了动作块。第三，必须正面回应 Causeway 的 "task islands" 解释，并说明为什么我们归一化后的切换数字远高于三篇竞品。

## 语言这一半大多已有先例：t = 0 不看指令是定论，中途换任务有三篇直接竞品

"LIBERO 上的 VLA 不怎么看指令"在 2025–2026 年已被多个基准反复确认，不能再当成发现。

- **LIBERO-PRO**：模型在标准 LIBERO 上超过 90%，换到泛化设置后 "collapse to 0.0%"；给乱码指令时输出 "remain unchanged"（[arXiv 2510.03827](https://arxiv.org/abs/2510.03827)）。它的 GitHub 排行榜上，把指令改成同场景另一个物体或动作的 "Task" 扰动，**π0.5 只有 0.00–0.01**（[LIBERO-PRO GitHub](https://github.com/Zxy-MLlab/LIBERO-PRO)）。各模型在乱码指令下的数值只画在图里。
- **LIBERO-Plus**：语言改写让 OFT 从 97.1 降到 81.5，π0 从 94.2 降到 61.0；作者说模型 "tend to ignore language instructions completely"；OFT 在 LIBERO-Object 上给空指令，成功率也 "largely unchanged"（[arXiv 2510.13626](https://arxiv.org/html/2510.13626v1)）。但空指令和换目标两组的具体数值只在 Fig. 3 里，笔记没有读出，而且 LIBERO-Plus 没测 π0.5。
- **Grant et al.，*Not All Features Are Created Equal***：和我们最贴近，用的也是 π0.5 和 LIBERO（[arXiv 2603.19233](https://arxiv.org/html/2603.19233)）。
  - 线性探针能以 >99% 的准确率读出指令类别。
  - 给空指令时，π0.5 在 LIBERO-Goal 上从 92.6% 掉到 5.8%，在 LIBERO-Object 上却还有 31.6%（同文另一张表写 22.0%，两表不一致）。
  - 被告知 "do not {task}" 时，**π0.5 仍有 96.7% 照做**（n = 30，同批基线 86.7%）。
- **LIBERO-CF**：在 CF-Object 上，OFT 只有 2.0% 跟随新指令，99.6% 照做训练任务（[arXiv 2602.17659](https://arxiv.org/html/2602.17659v1)）。
- **Paraphrase Consensus**：用"流畅但错误的指令"去测，OFT 在 Goal 上的成功 100% 依赖指令，在 Object 上只有 2%（[arXiv 2610.19438](https://arxiv.org/html/2610.19438v1)）。
- **LIBERO-Para** 说清了套件间差异的来由：LIBERO-Goal 的所有任务共用同一个初始场景，指令是区分任务的唯一线索（[arXiv 2603.28301](https://arxiv.org/html/2603.28301v1)）。

所以，我们的"LIBERO-Object 那一对从 t = 0 就不跟"和"do not <task> 不听"都只是复现。它们真正的用处，是说明"中途听不听"只能在 Goal 这种必须读指令才能区分任务的套件上测。

真正的直接竞品是三篇专门研究"中途换任务"的论文。它们覆盖了我们阶段 1 的同一个套件，以及我们两个模型族中的至少一个。

- **SwitchVLA**（2025-06）：在 LIBERO-Goal 上，分别在接触前、接触中、动作完成后三个时点把任务 A 换成 B（[arXiv 2506.03574](https://arxiv.org/html/2506.03574v1)）。

  | 时点 | π0 | OFT |
  |---|---|---|
  | 不换 | 92.3 | 98.0 |
  | 早换 | 40.7 | 40.6 |
  | 中换 | 8.3 | 11.1 |
  | 晚换 | 10.2 | 13.0 |

  在真机上，π0 的中换在两个工位都是 0.0。
- **ReSteer**（2026-03）：在 LIBERO-Goal 上测 π0.5、OFT、MolmoAct（[arXiv 2603.17300](https://arxiv.org/html/2603.17300)）。结论是 "none exhibits reliable cross-task steering under prompt switches"，换任务只在执行早期可行（"feasible only early in execution"）。它给出的 steerability 分数是 0.252 / 0.295 / 0.403；第一列对应 π0.5 是从 HTML 里的数学符号推断的，需要在 PDF 里核对。
- **Causeway**（2026-09）：同样用 π0.5、OFT，外加 GR00T N1.5；在 LIBERO-Goal 的 71 个跨物体对、三个时点上测，平均裸切换成功率是 **π0.5 26.0%、OFT 3.4%、GR00T 9.9%**；作者把失败归因于 "task islands"（[arXiv 2609.30913](https://arxiv.org/html/2609.30913)）。

三篇的共同结论与我们阶段 1 完全同向：现成 VLA 中途换到另一条训练过的指令大多失败，而且越晚越难。下表把它们和我们并排。各列的定义不同，数字不能直接相比，第四节会专门讨论。

| 研究 | 模型 | 换法与时点 | 成功定义 | 报告数字 |
|---|---|---|---|---|
| SwitchVLA (2506.03574) | π0、OFT（仿真） | LIBERO-Goal，A → B，接触前 / 接触中 / 动作后；每格 12 次 | 严格：A 要先按要求收尾（如放下手里的物体），再完成 B | 早 / 中 / 晚：π0 40.7 / 8.3 / 10.2；OFT 40.6 / 11.1 / 13.0（不换时 92.3 / 98.0） |
| ReSteer (2603.17300) | π0.5、OFT、MolmoAct | LIBERO-Goal，从中间状态换指令 | steerability 分数 | 0.252 / 0.295 / 0.403（列与模型的对应为推断） |
| Causeway (2609.30913) | π0.5、OFT、GR00T N1.5 | LIBERO-Goal 71 个跨物体对；第 30 步 / 第 60 步 / A 完成后 | 裸切换后完成 B | π0.5 29.9 / 22.9 / 25.1；OFT 5.1 / 2.9 / 2.2（列名由正文推断） |
| MotorMind (2609.38078) | π0.5 等 | 物体被举起后、运送途中换成训练外的新说法 | 完成新要求 | π0.5 20%（每任务试次数未能确定） |
| 我们（阶段 1） | π0.5、OFT | LIBERO-Goal 8 对训练过的原句；任务进度 25% / 50% / 75%；policy twin | 完成 B 的比例 ÷ 同场景 t = 0 上限 | π0.5 .84 / .78 / .70；OFT .50 / .50 / .20 |

另有几篇用定性或小样本的方式说过同样的话：

- **Hi Robot**（PI 自己的工作）写明，扁平的 π0 基线 "does not react to real-time feedback"，提示词中途改变时 "often revert to default behaviors"（[arXiv 2502.19417](https://arxiv.org/html/2502.19417)）。flat π0 的具体数值只画在 Fig. 5 里。
- **SIMPLER-IN** 每 8 步让一个 VLM 插入纠正（[arXiv 2610.21251](https://arxiv.org/html/2610.21251)）。π0 等通用 VLA 有纠正时成功率是 68.2%，没有纠正时是 69.4%。对失败回合做离线纠正时，VLM 给出的纠正能救回 12.8%，故意给反的纠正是 11.8%，几乎一样，而单纯重试反而有 28.2%。
- **π0.7**：PI 在论文里承认 "Prior models generally lack the language following ability needed to follow the coaching commands"，但没说是指哪些模型（[arXiv 2604.15483](https://arxiv.org/html/2604.15483)）。
- **VITA-E** 直说当前 VLA "assume that user instructions are provided once at the beginning of a task and remain static"（[arXiv 2510.21817](https://arxiv.org/abs/2510.21817)）。
- **Xu and Wu** 在 π0.5 上发现，执行结果 "sensitive to command granularity and invocation state"（[arXiv 2610.17746](https://arxiv.org/abs/2610.17746)）。
- **CoRe** 用一个推理时"先想象、再重新对齐"的外部机制，同时处理中途换指令和物理扰动，相当于默认冻结的 VLA 自己吸收不了指令变化（[arXiv 2608.14822](https://arxiv.org/abs/2608.14822)）。

最需要警惕的是 **MotorMind 的 Table 5**（[arXiv 2609.38078](https://arxiv.org/abs/2609.38078)）。它用的是同一个 OpenPI π0.5 `libero_base` checkpoint：**指令不变、中途挪物体（Scene Shift）成功 70%，中途换指令（Prompt Shift）成功 20%**；GR00T N1.5 是 10% / 0%，MolmoAct2 是 50% / 0%。这就是我们"推得回、说不动"的不对称。不过它有几点局限：

- 它埋在一篇 VLM 脚手架论文的附表里，作者没把它当结论。
- 只有 10 个和 5 个任务，每任务的试次数没能确定。
- 换上去的是训练外的新说法。

这篇必须引用，定位为"小样本的同向佐证"。

"读到了但不照做"在表征层面也已经有人说过。除了 Grant et al. >99% 的探针结果，还有两篇：

- **Chen et al.，*When Instructions Retrieve Trajectories***（[arXiv 2609.39971](https://arxiv.org/html/2609.39971)）：在 π0.5/LIBERO 上，381 个人工标注的探测中有 69% 属于"回放熟悉的示范行为"；把目标任务的 prefix-KV 换进去，有 109/200 次 rollout 转向目标物，对照组是 0/200。作者明确说 "language is not simply ignored"。
- ***Encoded but Not in Control***（[arXiv 2610.06235](https://arxiv.org/abs/2610.06235)）：探针能读出指令指定的目标，但策略仍然去拿场景里原来的那个。

因此，"VLA 编码了指令却不执行"不能写成我们的首创。

## 动手这一半也有先例：在论文里它是对照，而且只能说到 π0.5

闭环视觉运动策略能从人为扰动中恢复，至少从 2023 年起就有公开展示：

- **Diffusion Policy**：在真机 Push-T 上，人在策略微调时挪动 T 块，策略 "immediately re-planned to push from the opposite direction"，作者称这是示范中从未出现过的行为（[arXiv 2303.04137](https://arxiv.org/abs/2303.04137)）。但只有单个任务、只有定性描述，没有计数也没有对照。
- **PI 的 π0 博客**：机器人 "is able to recover when someone tries to intervene"（[PI blog](https://www.pi.website/blog/pi0)）。
- **PI 的 π0.5 博客**：策略 "are reactive, and can handle both variability in the environment and perturbations"（[PI blog](https://www.pi.website/blog/pi05)）。两篇博客都只有视频，没有数字；π0.5 的页面也没有谈到中途改口令。
- **DeepMind 的 Gemini Robotics**：说法更进一步，称模型能 "detects changes to its environment or instructions"，有人挪动东西时会 "quickly replans and carries on"（[DeepMind blog](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/)）。它声称两半都能做到，但笔记在论文（[arXiv 2503.20020](https://arxiv.org/abs/2503.20020)）里没找到量化的中途换指令实验。

审稿人大概率会说"物理恢复是已知的"，这一点他们是对的。

2026 年开始有人在开源 VLA 上计数，结果与我们同向：

- **VLA-Corrector（stage-aware）** 用的正是冻结的 `pi05_libero`，每个条件跑 1,000 次（[arXiv 2609.06508](https://arxiv.org/abs/2609.06508)）。干净条件下 93.7%；目标物在 XY 平面挪 10 cm 后 78.3%；目标区域挪 10 cm 后 77.6%。按笔记的换算，这分别是干净条件的约 **0.84** 和 **0.83**。这两个比值是笔记自己算的，论文没有直接给；挪动发生在哪一步也没读出来。它们比我们 6 cm 条件下的 .93–1.00 略低，暗示位移越大掉得越多。
- **CycleVLA** 干脆把"闭环策略会自然吸收早期扰动"当作已知前提，原话是 "Since the closed-loop policy naturally absorbs early perturbations ... we inject errors near subtask completion"（[arXiv 2601.02295](https://arxiv.org/abs/2601.02295)）。

反方向的数字同样存在，它们决定了我们的物理结论只能说到"π0.5、仿真、中等幅度"：

- **VLA-Corrector（adaptive horizon）**：在真机 AgileX PiPER 上由人手动挪物体或目标，π0.5 只有 **40.0%**，普通抓放是 70.0%（3 个任务 × 20 次）（[arXiv 2607.01804](https://arxiv.org/abs/2607.01804)）。
- **ReViP**：在 LIBERO 上把已抓起的物体强行掉落、互换位置或重新摆放，平均成功率 **OFT 19%、π0 36%**（笔记只可靠地读出了平均值）（[arXiv 2601.16667](https://arxiv.org/abs/2601.16667)）。这和我们的 DROP（物体被抓起后拿走、放回桌上）很接近，而我们的 OFT 阶段 3 没有跑。
- **LIBERO-Plus**：在开局扰动机械臂的初始关节角（qpos 幅度 0.1–0.5），OFT 从 97.1 掉到 37.2，π0 从 94.2 掉到 6.6（[arXiv 2510.13626](https://arxiv.org/abs/2510.13626)）。

笔记列出的可能调节因素有五个：位移大小（6 cm 对 10 cm）、扰动时机（早期对临近子任务完成）、仿真对真机、π0.5 对 π0/OFT、"放回桌上"对"掉落"。另外，笔记没有找到任何在任务中途推 VLA 机械臂并计数的基准，所以我们的 PUSH 在方法上算一个小补充。

因此，物理这一半在论文里应当写成**对照条件**，而不是发现。它的作用是排除"这个策略中途根本不会做出反应"这个解释，从而把"不听人话"的原因限定在语言通道上。π0 论文给了一个可用的机制说法：多样的预训练数据让模型学到了 "a repertoire of recoveries and corrections"（[arXiv 2410.24164](https://arxiv.org/abs/2410.24164)）。笔记据此推测：被推、被挪之后的状态接近训练数据里的恢复片段；而训练数据里每段示范只配一条固定指令，"停下"这类话在其中没有对应物。这是推断，不是论文结论。

## 真正的空白：没人拿"停下、回原点、别做了"测过现成 VLA

五组笔记各自独立检索到 2026 年 10 月底，都没有找到这样的论文：把训练外的"停下 / 别动 / 回原点 / 取消"命令交给扁平 VLA（π0、π0.5、OpenVLA-OFT、Octo、GR00T 等），无论是在中途还是 t = 0，并报告照做率。最近的几篇各差一步：

- **Grant et al.** 只在 t = 0 测了否定句（n = 30）。
- **SIMPLER-IN** 测的是"怎么做"类指令，在成功的回合中，路径指令被遵守的比例是 0.0%（[arXiv 2610.21251](https://arxiv.org/abs/2610.21251)）。
- **LIBERO-VIFO** 报告真机上的 π0.5 "follows the language instruction rather than the unauthorized STOP cue"（[arXiv 2608.17600](https://arxiv.org/html/2608.17600v1)）。但那是开局就写在纸上的视觉 STOP 标志，而且作者把"不理它"算作安全行为。
- **VITA-E 和 ELLSA** 能停下，是因为专门造了系统级的特殊 token 或全双工架构；笔记没有读出它们的停止成功率（[VITA-E](https://arxiv.org/abs/2510.21817)；[ELLSA](https://arxiv.org/abs/2510.16756)）。

同样没有人把"收到冲突命令后原任务照样完成"当主指标，SwitchVLA 和 Causeway 都只报切换成功率。所以我们的阶段 2 在本轮检索里是空白：约 0/235 照做，原任务完成率是不干预时的 0.88–1.01 倍，"停下"即使在 t = 0 说也不听。唯一需要盯的潜在撞车是 GitHub 项目 physical-ai-safety-learning：它计划在 LIBERO 上对 SmolVLA 测"中途换目标"和裸 "stop" 指令，但只跑过一次 smoke，而且笔记只在搜索摘要里见过它（[GitHub](https://github.com/Barap1/physical-ai-safety-learning)）。

任务中途追加空间规则同样没有先例：

- **NegaAlign/NegaBench** 让 GR00T、π0、π0.5 遵守"排除某物"一类条件；但从摘要看，这些条件是开局指令的一部分，而且是靠插入并训练新层实现的（[arXiv 2610.11952](https://arxiv.org/abs/2610.11952)）。
- **SafeStage** 在执行期统计 "region entries"（进入区域的次数），但规则不是中途加的（[arXiv 2609.21223](https://arxiv.org/abs/2609.21223)）。
- **ReStruct** 显示，用提示词约束 π0.5 的偏好，效果不如外部的时序逻辑自动机（[arXiv 2606.26588](https://arxiv.org/abs/2606.26588)）。
- **Sharma et al.** 早在 2022 年就把"避开某物"一类语言翻译成规划器的代价函数，而不是交给学到的策略（[arXiv 2204.05186](https://arxiv.org/abs/2204.05186)）。

我们在 V5 阶段测到的"不许越线"（160/160 越线）和需要绕路的"别进这个区域"（168/168 进入），可以写成同一结论在空间规则上的延伸。

设计层面有三处新意，但每一处都要说准：

- **policy twin 本身不新。** ReSteer 从中间状态开始评估，本质上就是 twin 设计（[arXiv 2603.17300](https://arxiv.org/html/2603.17300)）；MAIL-Bench 在参考轨迹的多个阶段中断摄像头，用的也是"回放到检查点"的思路（[arXiv 2609.35003](https://arxiv.org/abs/2609.35003)）。新的是：在同一个模型、同一个时刻，把说话式干预和动手式干预并排比较，并配上同步对照。笔记没有找到这样做的论文；MotorMind 的那两列不是同步配对的。
- **按 t = 0 上限归一化、按任务进度 25/50/75% 画出的切换曲线没有先例。** 最接近的只有零碎结果：SwitchVLA 的接触阶段效应；Grant et al. 发现 OFT 在 libero_10 上，早期或中期把提示通路置空，会让 9 个原本成功的任务全部归零，晚期置空则保留 2 个；以及 GR00T 在轨迹起始时最敏感（早期 −29.7 pp，中期 −13.8，晚期 −13.6）（[arXiv 2603.19233](https://arxiv.org/html/2603.19233)）。
- **"换指令带来的首个动作块变化 ÷ 换种子带来的变化"这个比值没有先例**，笔记专门检索过也没找到。最接近的有三个：驾驶 VLA 里用"遮掉摄像头"作参照尺度（[arXiv 2610.30953](https://arxiv.org/abs/2610.30953)）；Paraphrase Consensus 统计每次重新规划时动作块之间的分歧（[arXiv 2610.19438](https://arxiv.org/html/2610.19438v1)）；Gaze Prompts 测到 π0 换指令后注意力图的 JSD 只有 0.047（[arXiv 2609.34550](https://arxiv.org/html/2609.34550v1)）。我们的探针在输出层面证明 π0.5 读到了新句子（变化是种子噪声的 2.4–9.4 倍），但没有转化成闭环行为。而且敏感性在 Goal 上持续、在 LIBERO-10 场景上衰减，与两者的行为结果对得上，这个时间走向也没人报过。

论证层面的新意对主线最重要。"安全约束要在策略外面执行"这个立场已有很多人持有：

- **AEGIS/VLSA** 和 **VLPSA** 用 CBF 包住 VLA；VLPSA 在 SafeLIBERO 上把基础 π0.5 的避碰率从 23.1% 提到 91.2%（[arXiv 2512.11891](https://arxiv.org/abs/2512.11891)；[arXiv 2609.22462](https://arxiv.org/abs/2609.22462)）。
- **Multi-Link** 给 π0.5 加上一个从感知拟合出的禁入椭球（[arXiv 2609.40007](https://arxiv.org/abs/2609.40007)）。
- **Safety Chip** 把人说的"不要"编译成 LTL，再剪掉违规动作（[arXiv 2309.09919](https://arxiv.org/abs/2309.09919)）。
- **RoboGuard** 在越狱攻击下，把不安全计划的执行率从 92% 以上压到 3% 以下（[arXiv 2503.07885](https://arxiv.org/abs/2503.07885)）。
- **SafeHarness** 指出，提示词里写了禁止，规划时仍然不当回事（[arXiv 2609.20822](https://arxiv.org/abs/2609.20822)）。
- ***What Can One Sentence Fix?*** 发现，人说的一句话只有被编译成策略之外的动作偏置才起作用；让 VLM 直接写纠正，结果是 "recovers none"（[arXiv 2608.29967](https://arxiv.org/abs/2608.29967)）。

但这些工作的理由是"学到的策略没有安全保证"或"语言模型可以被越狱"。没有一篇的出发点是"人在任务中途善意地叫 VLA 停下或给它定规则，它不执行"这条证据；也没有任何防护层会接收人在中途说出的自然语言规则或停止命令，再施加到 VLA 上。现有工作分成两类：Safety Chip 和 RoboGuard 把自然语言规则交给外部检查器，但作用对象是 LLM 规划器；AEGIS、VLPSA、Multi-Link 直接作用在 VLA 上，但只处理几何约束。我们的 verifier 加 gate 正好落在两者之间的空位上。verifier 一侧的竞品已在 2026-10-20 的检索中覆盖，这里不重复。

| 我们发现中的一块 | 最近的先例与数字 | 判断 |
|---|---|---|
| t = 0 时不怎么看指令；LIBERO-Object 靠场景认任务 | LIBERO-PRO（π0.5 在 Task 扰动下 0.00–0.01）；LIBERO-CF（OFT 在 CF-Object 上跟新指令 2.0%）；Grant et al.（空指令下 π0.5 Goal 5.8%） | 已知，属于复现 |
| t = 0 说 "do not <task>" 仍照做 | Grant et al.：π0.5 96.7%（n = 30） | 已知（小样本） |
| 中途换成训练过的指令会失败，而且越晚越难 | SwitchVLA、ReSteer、Causeway（π0.5 26.0%、OFT 3.4%） | 已知；我们的增量是 twin、t = 0 归一化、进度曲线、四个套件 |
| 扁平 VLA 不理中途的话（定性） | Hi Robot；SIMPLER-IN（68.2 对 69.4） | 部分已知 |
| 训练外的 stop / return / cancel 在中途说，原任务照做 | 无 | 看起来新 |
| 中途追加越线 / 禁区规则 | 无（NegaBench 是开局条件；SIMPLER-IN 的路径 0.0% 是开局指令） | 看起来新 |
| 物理扰动后能恢复 | Diffusion Policy、π0/π0.5 博客（定性）；VLA-Corrector 约 0.84 / 0.83（笔记推算） | 已知；我们的增量是四类干预加同步对照 |
| 同一个模型推得回、说不动 | MotorMind：π0.5 70% 对 20% | 有一个小样本先例 |
| 编码了指令却不执行 | Grant et al. 探针 >99%；Chen et al. KV 互换 109/200 | 表征层已知；输出层的种子归一化与时间走向看起来新 |
| 以"不执行善意人话"为理由主张外部执行 | 无（已有工作的理由都是"无保证"或"可越狱"） | 看起来新 |

## 审稿人会提的替代解释，以及各自怎么回应

### Causeway 的 "task islands"：换不过去也许是"够不着"，而不是"没听"

Causeway 的核心论点是：任务 A 执行到一半留下的状态，可能让 B 从这里"够不着"，作者称之为 task islands。A 完成后再换，成功率在 3% 到 50% 之间，而且与退出时相对目标的接近轴误差强负相关（[arXiv 2609.30913](https://arxiv.org/html/2609.30913)）。这是对我们阶段 1 最有力的替代解释，而我们自己的数据也部分支持它：在项目报告里，两个策略都表现为"t 时刻末端离 B 的参考路线越远，越换不过去"（[项目报告](../../generative_verifier/HIL_SUITE_REPORT.md)）。

笔记给出的拆法，是把这种"惰性"分成两个来源。一是语义上的：微调后的 VLA 本来就少用指令。二是几何上的：手臂一旦朝 A 投入，B 就落在示范分布之外。前者预言训练外命令完全无效，后者预言即使是训练过的切换，也会随进度衰减。我们的数据两样都有。

对阶段 2 的回应更硬一些：保持不动和回到起点在任何状态下都物理可行，不存在"够不着"的问题。但也要老实承认，"停下"在 t = 0 说也不听，所以阶段 2 区分不了"因为是中途"和"因为没学过这类句子"。阶段 2 能支持的说法是"用这些话停不下它"。"中途"本身的效应，要靠阶段 1 的训练过的原句加 t = 0 上限来量。

### 为什么我们的切换数字比 SwitchVLA、ReSteer、Causeway 高这么多

π0.5 在 25/50/75% 进度时保有 t = 0 上限的 .84/.78/.70；而 Causeway 的 π0.5 平均只有 26.0%，SwitchVLA 的 π0 在中、晚期只有 8–10%。审稿人一定会问为什么。

先排除最容易想到的原因：我们的 t = 0 上限本身接近满分，除以上限只会把数字抬高一点，解释不了这么大的差距。笔记列出的真正原因有四条：

1. **配对不同。** Causeway 只用跨物体的对；我们 LIBERO-Goal 的 8 对里，有 4 对是"同一个碗、只换目的地"。按项目报告的逐对表，能换过去的几乎都是这 4 对；同一个酒瓶换目的地的两对，π0.5 参差不齐，OFT 基本换不过去。OFT 在碗以外的 4 对上从 25% 起就基本换不过去，这和 Causeway 中 OFT 的 3.4% 吻合。但对 π0.5 来说，即使只看我们那两对跨物体、跨任务的对（开炉子 ↔ 碗放盘子），早换的完成率也明显高于 Causeway 的平均值，所以配对只能解释一部分差距。
2. **时点定义不同。** 我们按任务进度的比例取点，Causeway 用绝对步数（第 30 步、第 60 步、A 完成后），SwitchVLA 用接触阶段。三者落到同一条轨迹上，可能对应差得很远的位置。
3. **成功判据不同。** SwitchVLA 要求 A 先按要求收尾（例如放下手里的物体），再完成 B，比只看 B 是否完成更严格。
4. **模型与样本量不同。** SwitchVLA 仿真里的基线在 HTML 中显示为 π0，不是 π0.5；每格只有 12 次；其基线是否用标准 LIBERO-Goal 数据训练也没确认。ReSteer 报的是另一种 steerability 分数，列与模型的对应还是推断的。

处理办法是在论文里补一张表：未归一化的原始切换成功率，按配对类型分列；同时把我们的进度点换算成绝对步数和接触阶段，与上面三篇逐项对齐。对齐之前，不能笼统地说"π0.5 早换基本能换过去"，只能说"在同一个碗换目的地的配对上能换过去"。这与探针的读数也一致：Goal 上的敏感性在碗已在手里、正搬往 A 的目的地时最高。

### LIBERO-Plus 说开局扰一下就崩，为什么中途推一把反而没事

在 LIBERO-Plus 里，机械臂初始关节被扰动后，π0 从 94.2 掉到 6.6，OFT 掉到 37.2；而我们的 π0.5 被推开 10 cm 后，完成率仍是同步对照的 .93–1.00。两者并不矛盾，但要讲清楚原因：

- LIBERO-Plus 没测 π0.5，比的不是同一个模型。
- 它扰动的是开局时的关节角，幅度以 qpos 0.1–0.5 给出；我们是在任务中途给末端施加 10 cm 位移。两者不是同一种量，无法直接换算。
- 同一个 π0.5 checkpoint 在 VLA-Corrector 的中途挪动实验里也保住了约 0.83–0.84，与我们一致。

真正危险的反例是 ReViP（OFT、π0 在强制掉落后平均只有 19%、36%）和真机上的 VLA-Corrector（π0.5 40% 对 70%）。它们说明恢复能力取决于模型、扰动类型，以及是仿真还是真机。论文的写法应当是"在 LIBERO 仿真中，π0.5 能从四类中等幅度的脚本化扰动中恢复"，并把 OFT、π0 和真机列为未覆盖。

### 攻击论文说文字能操纵 VLA：措辞要从"不听语言"收窄到"不执行善意的人话"

几篇工作表明，语言通道确实能影响 VLA 的动作：

- **Jones et al.**：开局时一次性施加的文本攻击能 "facilitate full reachability of the action space"，而且攻击文本不必在语义上与伤害相关（[arXiv 2506.03350](https://arxiv.org/abs/2506.03350)）。
- **Puthumanaillam et al.**：通过在线搜索提示词，找到"看上去仍在描述原任务、却改变了最终物理结果"的提示（[arXiv 2606.12978](https://arxiv.org/abs/2606.12978)）。
- ***Learning What to Say to Your VLA***：搜索出能改善冻结 VLA 闭环表现的语言序列，仿真中提升 24.7%，真机上提升 65.0%（具体用了哪些 VLA，笔记没读出）（[arXiv 2606.12299](https://arxiv.org/abs/2606.12299)）。
- **Häon et al.**：在 π0 和 OpenVLA 上，改提示词的效果弱于直接改激活（[arXiv 2509.00328](https://arxiv.org/abs/2509.00328)）。

再加上我们自己的探针，结论很清楚：语言通道连到了动作，只是自然、善意的命令驱动不了它。笔记把这概括为"单向杠杆"：优化过的 token 能操纵 VLA，构成攻击面；善意的安全命令却不能，起不到安全手柄的作用。笔记没有找到任何论文明确说出这一不对称。

**FreezeVLA** 提供了威胁模型上的镜像：对抗图像能让 VLA "ignore subsequent instructions"，在三个 VLA、四个基准上的平均攻击成功率是 76.2%（[arXiv 2509.19870](https://arxiv.org/abs/2509.19870)）。这意味着，如果停止要靠指令通道传达，攻击者还能把它关掉，这是"停止通道必须独立于策略"的又一个理由。反方向上，带 LLM 规划器的机器人则过度服从，RoboPAIR "often achieving 100% attack success rates"（[arXiv 2410.13691](https://arxiv.org/abs/2410.13691)）。两类系统的问题方向相反，但都说明语言不是可靠的安全接口。

### 其余会被问到的质疑

- **延迟。** RTC 和 BID 说明，动作分块会降低反应速度，新指令最早也要到下一次推理才能生效（[arXiv 2506.07339](https://arxiv.org/abs/2506.07339)；[arXiv 2408.17355](https://arxiv.org/abs/2408.17355)）。但它们解释的是延迟，不是完全不响应。论文应写明 STOP 的判定窗口跨越了几次重新推理，以排除"只是还没轮到新指令生效"。
- **checkpoint 的范围。** PI 的 π0.5 在推理时先生成文字形式的子任务再执行，训练时还混入了 "verbal instruction" 类示范（[PI blog](https://www.pi.website/blog/pi05)）。openpi 的 README 没说明 `pi05_libero` 在 LIBERO 微调时是否保留了 knowledge insulation（[openpi](https://github.com/Physical-Intelligence/openpi)）；PI 自己也说，用了 knowledge insulation 之后语言遵循仍然 "still far from perfect"（[arXiv 2505.23705](https://arxiv.org/abs/2505.23705)）。Breaking Lock-In 和 Flatness 两篇都指出，低数据量的微调会让 VLA 不再响应新指令（[arXiv 2604.23121](https://arxiv.org/abs/2604.23121)；[arXiv 2606.23641](https://arxiv.org/abs/2606.23641)）。所以我们测的是"公开的、按基准微调的 checkpoint"，不是 PI 内部带高层推理的 π0.5，也不是 Gemini Robotics 这类闭源模型。同时，Interactive Language 用约 60 万条带语言标注的片段训练出了能实时改口令的扁平策略（[arXiv 2210.06407](https://arxiv.org/abs/2210.06407)），说明专门训练过的策略可以做到。
- **CANCEL 句子点名了任务。** Grant et al. 的否定句结果，以及文本 LLM 研究中"点名被禁止的词反而起了提示作用"占违规案例 87.5% 的发现（[arXiv 2601.08070](https://arxiv.org/abs/2601.08070)），都提示 "do not <task>" 可能反过来提示了 A。这应当写成 CANCEL 的局限。
- **空指令的混淆。** Paraphrase Consensus 指出，空提示会把"不看语言"和"文本无意义时退回先验行为"混在一起。我们换上的是流畅的、训练过的原句，空指令（1.4–2.9 倍）只当对照，正好避开了这个问题。
- **pHRI 的反向框架。** Bajcsy、Losey 等人把人对机器人的推碰看作关于任务目标的有意信息（[CoRL 2017 PDF](https://collab.me.vt.edu/pdfs/bajcsy_corl2017.pdf)；[arXiv 2107.02349](https://arxiv.org/abs/2107.02349)）。笔记据此推测，"对物理扰动鲁棒"和"听不懂身体上的提醒"可能是同一种性质的两面。因此我们的物理条件要写成"指令不变时的扰动"。
- **术语冲突。** 在 HG-DAgger、HIL-SERL、π*0.6 RECAP 这条研究线里，"intervention"指人通过遥操作接管机器人、产出训练数据（[arXiv 1810.02890](https://arxiv.org/abs/1810.02890)；[arXiv 2410.21845](https://arxiv.org/abs/2410.21845)；[arXiv 2511.14759](https://arxiv.org/abs/2511.14759)）；这些工作都没有评估冻结的策略对旁人行为的反应。论文应主动给出三分法：遥操作接管、语言插话、身体接触（接触又分扰动和交流两种）。
- **我们自己的范围限制。** 只有仿真；物理干预是脚本化的瞬时扰动；每个（状态，时刻）只有一个种子（[项目报告](../../generative_verifier/HIL_SUITE_REPORT.md)）。

## 推荐写法：一句话主张、必引清单、该补的分析和候选实验

推荐的标题级主张是："**现成的、按基准微调的 VLA，被推开能恢复，被叫停却不停。**"英文可以写成 *Pushed, They Recover; Told, They Don't Stop: Off-the-Shelf VLAs Ignore Mid-Task Human Commands*。

正文里的一句话版本是：在标准 LIBERO 的四个套件上，π0.5 和 OpenVLA-OFT 读到了任务中途的新指令（首个动作块的变化是种子噪声的 2.4–9.4 倍），却不执行训练外的停止、回原点和取消命令（每个策略约 0/235，原任务照常完成）；它们只在早期、且多半只在"同一物体换目的地"的配对上切换到训练过的指令；与此同时，π0.5 能从四类物理干预中恢复（完成率是同步对照的 .93–1.00）。因此，人或 verifier 的纠正与停止必须在策略之外执行。

要避开四种写法：

- **"首次证明 VLA 编码了指令却不执行"**：Grant et al. 和 Chen et al. 已经先说了。
- **"VLA 忽略语言"**：会被攻击论文和我们自己的探针反驳。
- **"在策略内部不可能做到"**：SwitchVLA、ReSteer、Causeway、CAG、Learning What to Say 和 Interactive Language 都在策略内部、推理时或通过训练部分做到了。
- **"VLA 对物理扰动鲁棒"**：只能说 π0.5。

这条主张接上主线的最好方式，是把它和安全标准、外部执行文献连起来。根据笔记读到的二手摘要，ISO 13850 要求急停"由单个人的动作发起"、优先于其他一切功能、可靠性至少 PL c；ISO 10218:2025 新增了 "normal stop" 安全功能，用来减少对急停的误用（[ibf-solutions](https://www.ibf-solutions.com/en/seminars-and-news/news/new-standards-for-industrial-robots-en-iso-10218-1-and-2)；[machinerysafety101](https://machinerysafety101.com/2026/05/18/iso-13850-emergency-stop-requirements/)）。按这些标准，即使 VLA 听得懂"停"，它也不能算急停；我们的结果更进一步，说明它连普通停止都做不到。所以口头的"停下 / 回去"应当经过 verifier 前端，路由到控制器的 normal stop 或 protective stop，而不是写进策略的提示词。gate 的误停率可以拿 RESCUE 作外部参照：在 stop-and-reset 模型下，即使是理想检测器，也只能承受 2.0–3.3% 的误报率（[arXiv 2610.13516](https://arxiv.org/abs/2610.13516)）。

| 角色 | 文献 | 用法 |
|---|---|---|
| 直接竞品（中途换任务） | SwitchVLA 2506.03574；ReSteer 2603.17300；Causeway 2609.30913；CoRe 2608.14822 | 把阶段 1 定位为复现与扩展；正面讨论 task islands |
| 同向佐证 | MotorMind 2609.38078；Hi Robot 2502.19417；SIMPLER-IN 2610.21251；VITA-E 2510.21817；π0.7 2604.15483 | 作为不对称现象和"扁平 VLA 不理中途的话"的先例 |
| t = 0 时不看指令，及其机制 | LIBERO-PRO 2510.03827；LIBERO-Plus 2510.13626；Grant et al. 2603.19233；LIBERO-CF 2602.17659；LIBERO-Para 2603.28301；Paraphrase Consensus 2610.19438；LangForce 2601.15197；Breaking Lock-In 2604.23121 | 作背景和机制，不当成我们的发现 |
| 读到但不做 | Chen et al. 2609.39971；Encoded but Not in Control 2610.06235；Manalu et al. 2610.30953 | 给探针定位；归一化思路的出处 |
| 物理恢复 | Diffusion Policy 2303.04137；π0 2410.24164 及博客；π0.5 博客；Gemini Robotics 2503.20020；VLA-Corrector 2609.06508、2607.01804；CycleVLA 2601.02295；ReViP 2601.16667 | 物理这一半作对照；正反两个方向都引 |
| 文本可以操纵 VLA | Jones et al. 2506.03350；Puthumanaillam et al. 2606.12978；Learning What to Say 2606.12299；FreezeVLA 2509.19870 | 用来收窄措辞，支撑"单向杠杆"的说法 |
| 外部执行 | AEGIS/VLSA 2512.11891；VLPSA 2609.22462；Multi-Link 2609.40007；Safety Chip 2309.09919；RoboGuard 2503.07885；SafeHarness 2609.20822；What Can One Sentence Fix? 2608.29967；RESCUE 2610.13516；LIBERO-VIFO 2608.17600；ISO 10218 / 13850 | 主线的动机，以及我们占据的空位 |
| 术语与反向框架 | HG-DAgger 1810.02890；HIL-SERL 2410.21845；π*0.6 2511.14759；Bajcsy et al. CoRL 2017；Losey et al. 2107.02349 | HIL 三分法；pHRI 讨论 |

不新跑任何实验，只在现有记录上用 CPU 重算，就能补出最关键的几样东西（是否动手仍由用户决定）：

1. **未归一化的原始切换成功率**，按配对类型分列（同一个碗换目的地、同一个酒瓶换目的地、跨物体的开炉子 ↔ 碗放盘子），并附上 t = 0 上限，方便读者与 Causeway 的跨物体对直接比较。
2. **把 25/50/75% 进度换算成绝对步数和接触阶段**（未接触 / 已抓起 / 已放下），并尽量按 SwitchVLA 的严格判据（A 有没有收尾）重新判一次。
3. **把"收到冲突命令后原任务仍完成"作为阶段 2 的主指标单独报告**，因为笔记里没有人报过这个量。
4. **说明 STOP 的判定窗口跨越了几次重新推理**，堵住"只是延迟"的质疑。
5. **在相关工作里放一张"已知 / 新"的表**（即本报告第三节那张），并采用 HIL 三分法。

下表列出候选的新实验。每一项都是新的 GPU 执行或新的干预方法，都需要用户批准，这里只列候选，不作决定。

| 候选 | 回答的问题 | 文献依据 |
|---|---|---|
| OFT 阶段 3（物理干预） | 物理恢复是不是只属于 π0.5 | ReViP 中 OFT 在物体掉落后平均只有 19% |
| OFT 阶段 5（动作块探针） | OFT 是否也"读到但不做" | LIBERO-Plus 称 OFT 的行为更像只看视觉、不看语言的模型 |
| π0 分支（需要偏离冻结的环境门槛） | 加入第三个模型族 | SwitchVLA、Hi Robot 用的都是 π0 |
| Causeway 式的跨物体配对，按绝对步数换 | 让数字直接对齐 | Causeway 2609.30913 |
| 位移幅度梯度（如 6 cm 对 10 cm） | 恢复能力是否随位移幅度下降 | VLA-Corrector 在 10 cm 时约 0.83–0.84 |
| "推一下，意思是停" | 对扰动鲁棒和听不懂身体提醒是否是同一种性质 | Bajcsy / Losey 的 pHRI 工作 |
| 逐时刻的探针，或 prefix-KV 互换 | 中途给的指令在模型哪一层丢失 | Grant et al.；Chen et al. |
| 换指令时加语言 CFG（CAG） | 推理时放大语言项能否救回切换 | LIBERO-CF 只在开局指令上验证过 CAG；Causeway 发现 CFG 式提示对比对切换 "no consistent recovery" |
| 在线 gate 在策略外执行停止 | 主线的第三个问题 | AEGIS、VLPSA、RESCUE |

还要盯几处可能的撞车：

- **Grant et al.** 已经定义了 "temporal switch" 条件（中途换提示词），但笔记在 v5 的 HTML 中没找到它的结果。一条搜索摘要称它"只恢复策略本来就在追求的目标"，这一说法未经核实。
- 上面提到的 **GitHub SmolVLA 停止实验**。
- 只在搜索结果中出现、还没打开的 ***Decoding Task Progress from VLA Representations***（2608.13474）和 ***Dynamic Execution Commitment of VLA Models***（2605.11567），可能涉及"语言的作用随任务进度变化"。

## 结论

这轮文献定位改变了我们对这项结果的理解。它的价值不在现象本身："VLA 不太看指令"和"闭环策略能从推碰中恢复"都已是共识。它的价值在于：第一次把两者放在同一个模型、同一个时刻、同一套对照下比较，而且用了一类没人测过的命令，即人在任务中途让机器人停下、退回或取消。这个组合接通了两条原本不相交的文献。一条是诊断"VLA 忽略语言"的文献，它们只说"不看"，从不谈后果。另一条是"在策略外执行安全约束"的文献，它们主张外部执行，理由却是"没有保证"或"会被越狱"。我们的结果给后者补上了一个更直接、更朴素的理由：问题不是 VLA 会被坏人说动，而是好人叫它停时它根本不停。

另一个值得带走的判断是：竞品的存在反而让主张更稳。SwitchVLA、ReSteer、Causeway、CoRe 和 Learning What to Say 都表明，经过专门训练、推理时改写或外部搜索，语言可以重新起作用。所以论文不该说"在策略内部做不到"，而应该说："现成的 checkpoint 默认做不到；而且即使把切换修好了，停止仍需要一条不经过策略、可以审计的通道。"这与安全标准把急停和任务逻辑分开的思路一致，也正是 verifier 加 gate 存在的理由。

## 附：核实说明

| 内容 | 核实情况 |
|---|---|
| ReSteer 的 0.252 对应 π0.5 | 列头在 HTML 中是数学符号，这一对应是推断，需查 PDF |
| Causeway 分时点的数字 | 列名（第 30 步 / 第 60 步 / 完成后）由正文推断；作者单位与发表 venue 未提取 |
| SwitchVLA | 仿真基线 [9] 在 HTML 中显示为 π0；真机上的 π0 是复现版；基线的训练数据与 venue 均未确认 |
| MotorMind 70% / 20% | 每任务试次数未能确定，只能作指示性参考 |
| VLA-Corrector 约 0.84 / 0.83 | 比值由笔记换算，论文没有直接给出；挪动的触发步未读出；合并条件正文写 70.9，表中写 70.7 |
| ReViP | 只可靠地读出了平均值 |
| LIBERO-Plus 空指令 / 换目标 | 数值只在 Fig. 3 里；换目标测的是哪个套件，正文与附录说法不一；没测 π0.5；CVPR 2026 版本以 CVF 页面为准 |
| LIBERO-PRO 乱码指令 | 各模型的数值只在图中；Task / Sem 两列的数字来自 GitHub 排行榜 |
| Grant et al. | 空指令下 π0.5 在 Object 上的 31.6% 与另一张表的 22.0% 不一致；摘要与正文措辞不完全一致；temporal switch 的结果未找到 |
| LIBERO-CF 仅视觉输入的结果 | 两份笔记的记录不同（逐套件的 68.2 / 56.6，对比 51.7），未核对，本报告未使用 |
| Hi Robot | flat π0 的数值只在 Fig. 5；ICML 2025 这一 venue 未完全确认 |
| FreezeVLA 分模型数字、AEGIS 的 77.9 对 18.7、DropVLA 的结论 | 来自搜索摘要，未核对原文 |
| RESCUE | 一份笔记标注为只见过搜索结果与 HTML 片段 |
| CoRe 2608.14822 | ID 经 arXiv API 核对过；另一份笔记未能把"Case Western 的 CoRe"对应到这个 ID |
| VITA-E、ELLSA、YAY Robot | 停止 / 插话的成功率未提取 |
| Learning What to Say | 用了哪些 VLA 未提取 |
| π0.7 中的 "prior models" | 具体指哪些模型未提取 |
| ISO 10218 / 13850 | 只读了二手摘要；ISO/TS 15066 的内容在 Part 1 还是 Part 2，各来源说法冲突 |
| Gemini Robotics | 只有博客层面的说法，未找到量化实验 |
| GitHub physical-ai-safety-learning | 只在搜索摘要中见过，未经同行评审 |
