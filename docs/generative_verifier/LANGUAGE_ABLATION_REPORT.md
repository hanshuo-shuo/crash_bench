# π0.5 从头语言消融：正式报告

2026-11-04 归档。协议：[LANGUAGE_ABLATION_PROTOCOL.md](LANGUAGE_ABLATION_PROTOCOL.md)。
运行代码与冻结协议均为 `e6dee13`；以下数字来自已完成、哈希核对通过的正式批次。

**主要结果：任务文字有显著作用，而且套件间差异很大。**
原指令 387/400（96.75%），空指令 201/400（50.25%），随机字符 215/400（53.75%）。
空指令比原句低 46.5 个百分点，随机字符低 43.0 个百分点；两个配对区间均不跨零。

**如何解释这次实验与此前 HIL 的关系：**

- LIBERO-Goal 最依赖任务文字：96% 降到 9% / 8%。因此不能把此前“停止 / 回原点 / 绕路不服从”
  统一解释成“π0.5 不读文字”。任务选择能力和服从新类型命令应分别讨论。
- Spatial、Object 和 LIBERO-10 中，即使没有任务语义，仍能完成 62–75% 的原任务。
  这与视觉场景和初始状态提供较强任务线索一致，是解释，不是已分离验证的因果机制。
  这三套中原句仍提高成功率 20–36 个百分点，也不能说文字完全无用。
- 本批 Object 中的“布丁放篮子”是标准任务 8：原句 10/10、空指令 7/10、随机字符 8/10。
  它支持该任务存在较强的无任务文字完成能力；标准 LIBERO 与旧 SafeLIBERO 绕路 / 回原点的状态和干预不同，
  不能用这 10 个状态直接重解释旧批次的所有结果。
- 随机 / 空指令本身分布外。本实验发现原任务在丢失任务文字时明显退化，
  **没有排除**此前新命令因分布外而不被执行的解释，也不能据此断言模型能理解所有新命令。
  已有的、训练过的任务原句在中途切换时会怎样，仍看 HIL suite 的阶段 1。

每任务只有 10 个初始状态，单个策略检查点、每状态一个策略种子、10 条共享随机字符串。
区间仅覆盖固定任务内初始状态的变化，不覆盖检查点、策略种子或随机银行的额外变化。
没有新增执行、付费 API、训练或锁定测试集访问。

模式：`formal`。400 个状态，1240 次执行。

| 套件 | 状态 | 原指令 | 空指令 | 随机字符 | 空 − 原（95% CI） | 随机 − 原（95% CI） |
|---|---:|---:|---:|---:|---|---|
| libero_spatial | 100 | 98.0% | 62.0% | 64.0% | -36.0 [-42.0, -30.0] pp | -34.0 [-40.0, -28.0] pp |
| libero_object | 100 | 98.0% | 63.0% | 68.0% | -35.0 [-44.0, -26.0] pp | -30.0 [-38.0, -21.0] pp |
| libero_goal | 100 | 96.0% | 9.0% | 8.0% | -87.0 [-92.0, -82.0] pp | -88.0 [-94.0, -81.0] pp |
| libero_10 | 100 | 95.0% | 67.0% | 75.0% | -28.0 [-36.0, -20.0] pp | -20.0 [-26.0, -14.0] pp |
| 总体（等任务权重） | 400 | 96.8% | 50.3% | 53.8% | -46.5 [-50.0, -43.0] pp | -43.0 [-46.3, -39.5] pp |

三组使用相同场景、官方初始状态和策略种子，从头给对应文字。随机银行为 12 个均匀 A–F / + 符号，
同一初始状态编号的字符串在所有任务共用。原指令失败的状态照样计入；重复原指令只作噪声诊断。

区间按每个固定任务内的初始状态配对 bootstrap，10,000 次，seed 1103，任务等权。
区间不覆盖随机文本银行和任务集合变化的不确定性。随机 / 空文字本身也分布外，
因此性能下降不能单独区分缺少任务信息和输入分布变化；本结果不能直接推出中途指令服从能力。

## 完整性与运行

- 全部计划单元和轨迹核对通过；起始物理 / 本体 / 策略 RNG 配对通过。
- 400 个配对组；图像最大差 2 级、25 个值。
- 原指令重复：40 / 40 成功判定一致，18 / 40 逐步哈希一致。
- 执行总耗时 4.80 GPU h（不含服务启动 / 汇总）；重放输出 680，动作越界 140068 步。
- 代码 `e6dee13fc7a1c23c04f25684046884349dfc2870`；上游 `2457feed5968ae803926e178c8ce8243b9ecdcf9`；LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`。
- GPU 作业 `7109431`；CPU 汇总 `7109433`。
- 根目录：`/projects/p33100/siosio/crashbench_safelibero/language_ablation/20261104T004227Z_formal_pi05_libero_e6dee13fc7a1`；配置 / 种子 / 全部文字见 `CAMPAIGN.json`、`PLAN.json`。
- 无 API、无 verifier、无训练，不读 SafeLIBERO 或 V4/V5 锁定测试。

## 逐任务结果

| 套件 | 任务 | 原句 | C0 | EMPTY | RANDOM |
|---|---:|---|---:|---:|---:|
| libero_spatial | 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 10/10 | 8/10 | 10/10 |
| libero_spatial | 1 | pick up the black bowl next to the ramekin and place it on the plate | 9/10 | 2/10 | 4/10 |
| libero_spatial | 2 | pick up the black bowl from table center and place it on the plate | 10/10 | 10/10 | 9/10 |
| libero_spatial | 3 | pick up the black bowl on the cookie box and place it on the plate | 10/10 | 10/10 | 9/10 |
| libero_spatial | 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 10/10 | 2/10 | 2/10 |
| libero_spatial | 5 | pick up the black bowl on the ramekin and place it on the plate | 10/10 | 10/10 | 9/10 |
| libero_spatial | 6 | pick up the black bowl next to the cookie box and place it on the plate | 10/10 | 10/10 | 10/10 |
| libero_spatial | 7 | pick up the black bowl on the stove and place it on the plate | 10/10 | 1/10 | 0/10 |
| libero_spatial | 8 | pick up the black bowl next to the plate and place it on the plate | 10/10 | 8/10 | 9/10 |
| libero_spatial | 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 9/10 | 1/10 | 2/10 |
| libero_object | 0 | pick up the alphabet soup and place it in the basket | 9/10 | 5/10 | 4/10 |
| libero_object | 1 | pick up the cream cheese and place it in the basket | 10/10 | 4/10 | 7/10 |
| libero_object | 2 | pick up the salad dressing and place it in the basket | 10/10 | 8/10 | 10/10 |
| libero_object | 3 | pick up the bbq sauce and place it in the basket | 10/10 | 9/10 | 9/10 |
| libero_object | 4 | pick up the ketchup and place it in the basket | 10/10 | 3/10 | 3/10 |
| libero_object | 5 | pick up the tomato sauce and place it in the basket | 10/10 | 10/10 | 10/10 |
| libero_object | 6 | pick up the butter and place it in the basket | 10/10 | 6/10 | 5/10 |
| libero_object | 7 | pick up the milk and place it in the basket | 10/10 | 7/10 | 9/10 |
| libero_object | 8 | pick up the chocolate pudding and place it in the basket | 10/10 | 7/10 | 8/10 |
| libero_object | 9 | pick up the orange juice and place it in the basket | 9/10 | 4/10 | 3/10 |
| libero_goal | 0 | open the middle drawer of the cabinet | 10/10 | 0/10 | 0/10 |
| libero_goal | 1 | put the bowl on the stove | 10/10 | 1/10 | 2/10 |
| libero_goal | 2 | put the wine bottle on top of the cabinet | 10/10 | 0/10 | 0/10 |
| libero_goal | 3 | open the top drawer and put the bowl inside | 7/10 | 0/10 | 1/10 |
| libero_goal | 4 | put the bowl on top of the cabinet | 10/10 | 0/10 | 0/10 |
| libero_goal | 5 | push the plate to the front of the stove | 10/10 | 0/10 | 0/10 |
| libero_goal | 6 | put the cream cheese in the bowl | 10/10 | 0/10 | 0/10 |
| libero_goal | 7 | turn on the stove | 10/10 | 1/10 | 1/10 |
| libero_goal | 8 | put the bowl on the plate | 10/10 | 7/10 | 4/10 |
| libero_goal | 9 | put the wine bottle on the rack | 9/10 | 0/10 | 0/10 |
| libero_10 | 0 | put both the alphabet soup and the tomato sauce in the basket | 10/10 | 0/10 | 0/10 |
| libero_10 | 1 | put both the cream cheese box and the butter in the basket | 10/10 | 5/10 | 4/10 |
| libero_10 | 2 | turn on the stove and put the moka pot on it | 10/10 | 8/10 | 10/10 |
| libero_10 | 3 | put the black bowl in the bottom drawer of the cabinet and close it | 10/10 | 10/10 | 10/10 |
| libero_10 | 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 9/10 | 9/10 | 9/10 |
| libero_10 | 5 | pick up the book and place it in the back compartment of the caddy | 10/10 | 10/10 | 10/10 |
| libero_10 | 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 9/10 | 3/10 | 9/10 |
| libero_10 | 7 | put both the alphabet soup and the cream cheese box in the basket | 10/10 | 10/10 | 10/10 |
| libero_10 | 8 | put both moka pots on the stove | 7/10 | 6/10 | 5/10 |
| libero_10 | 9 | put the yellow and white mug in the microwave and close it | 10/10 | 6/10 | 8/10 |

## 作业与收尾核对

Slurm 核对：以下全部 `COMPLETED 0:0`，账号 p33100；四个 GPU 分片最多两片同时运行。

| 阶段 | 作业 | 耗时 |
|---|---|---|
| smoke，16 次 | 7109400 | 6 分 11 秒 |
| CPU 完整性 gate | 7109401 | 3 秒 |
| 正式分片 0（数组母号 7109431） | 7115635 | 1 小时 20 分 44 秒 |
| 正式分片 1 | 7115671 | 1 小时 12 分 45 秒 |
| 正式分片 2 | 7123501 | 1 小时 16 分 30 秒 |
| 正式分片 3 | 7109431 | 1 小时 16 分 29 秒 |
| CPU 汇总 / 自动报告 | 7109433 | 1 分 52 秒 |

- 四片 320 / 300 / 320 / 300 次，共 1,240 次；C0、EMPTY、RANDOM 各 400，C0R 40。
- 正式 GPU 分配耗时合计约 5.11 小时；逐次执行累计 4.80 小时，差额含服务启动等开销。
- 400 个状态的初始物理 / 本体 / RNG 配对检查通过，图像差未超过预设容差。
- C0R 与 C0 的成功判定 40/40 一致，但只有 18/40 物理轨迹逐步完全相同。
  重复对的最大末端偏差中位 3.12 mm，最大 11.95 cm；结果一致不代表动作完全相同。
- 本次收尾重新核对正式 `HASHES.json` 中的全部汇总文件，并从原始逐次表独立核对条件计数，全部一致。
- 没有修改原始输出，没有重跑任何评测。首个排队但未执行的 smoke 7108220 已取消，原目录及 ABORTED.json 保留。
- 收尾时用户队列为空。后续更多种子、其他策略或新条件需另行决定。
