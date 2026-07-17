# HISTORICAL — not the current execution plan

The current execution plan is [docs/PAPER_PLAN.md](docs/PAPER_PLAN.md). This
file is retained as a dated planning record and may contain superseded scope,
status, external-reference statements, and the deprecated selected-band
cross-category headline.

# CrashBench — 顶会冲刺计划 v2(ICLR 2027 主攻 · 2026-07-13 审计修订版)

> **这是什么:** 从"手里的结果"到"一篇 ICLR/CoRL 级论文"的执行计划,**取代 [ROADMAP.md](ROADMAP.md) 的
> 排期部分**。v2 = 在 v1(07-11)基础上做了一轮全面审计(逐条数字对账 + 3 路模拟审稿 + 竞品联网核实)
> 后的修订;v1→v2 改了什么见 [§8](#8-v1v2-差异这次改了什么为什么)。
>
> **底稿:** 进度 [STATUS.md](STATUS.md) · 文献 [STRATEGY.md](STRATEGY.md) · 报告 [REPORT.md](REPORT.md)。
> v1 备份在 scratchpad(未入库)。

---

## 0. 一句话战略

**手里已经是一篇 80% 完成的顶会论文;剩下 20% 不是"再加一类 hazard",而是 ①把唯一的因果结论从
1 个任务推广到 ≥4 个任务(外部效度),②用三个零 GPU 的对照/基线把"探针混淆"和"shield 平凡性"
两个最大攻击面堵死,③把 π0 收口,④在 10 周内写完投出去。** 竞品速度比 v1 估计的更快
(SafeLIBERO 生态 6–7 月一个月内出了 ≥4 篇,见 §2),**速度仍是第一约束**——凡是 2 周内做不完、
又不挡出版的,一律砍;凡是零 GPU 就能堵住一条致命攻击的,一律做。

**论文主张(定稿版 v2,口径已按证据收窄):** *固定障碍外观、只变扫掠走廊归属的受控 on/off-path
对照,因果证明 VLA 碰撞源于"缺安全策略"而非 OOD——行为层跨 3 种动作头架构复现(离散 token /
L1 回归 / flow-matching,同一具身与训练数据分布,见 §7-14);冻结线性探针解码"我会撞"
(OpenVLA base AUC 0.99–1.0,OFT 0.90–0.99,π0 待 M2 收口)而策略零减速——**信号"可解码但未被
使用"(decoded-but-not-used)**:crash 方向在动作读出投影中保留不足 10% 的范数(steering 失败的
机制解释);同一探针门控恢复(base,在线闭环)把 on-path crash 100%→0%、冲击 321.7N→0N、
良性 0/22 零误触发,离线操作窗口 3.4 logit——并以 trivial-trigger 基线(M7)界定探针的真实增量。*

> **措辞纪律(全文执行):** 正文精确措辞用 **"decoded but not used / represented but not read out"**;
> "knows but doesn't act" 只作 informal hook(intro 一次)。理由:steering 正交性诊断本身说明信号
> 未进入动作子空间——"knows"会被解读为认知宣称,被 interp 背景的审稿人反打(§7-8)。

---

## 1. Venue 决策(2026-07-12 已联网核实)

| 选项 | deadline(核实状态) | 判断 |
|---|---|---|
| **ICLR 2027 ★主攻** | **官方未公布**(iclr.cc/2027 404;Future Meetings 页只写 "West Coast North America")。按 ICLR'26 模式推断 abstract ≈ 9/18–19、full ≈ 9/23–24;第三方 tracker 列 9/19+9/24(未经官方证实)。**内部安全线定 9/15** | 距今 ~9 周。唯一近期顶会窗口;本文智力核心(因果对照、表征-行为解离、探针干预)正是 ICLR 品味;sim-only 完全可接受 |
| RSS 2027(备份 1) | 未公布;RSS'26 全文 deadline 为 2026-01-30,惯例 1 月底 → 预计 **~2027-01 底** | ICLR 被拒后的第一落点;届时补真机 10-trial 演示 |
| CoRL 2027(备份 2) | 未公布;CoRL'26 = 5/26+5/29(已过,会期 2026-11 Austin)→ 预计 ~2027-05 下旬 | 等 10 个月 = scoop 风险不可接受,只作二级备份 |

**arXiv 策略(提前):** 实验冻结 + 初稿成型即挂 arXiv,目标从 W8 **提前到 W7 末**。SafeLIBERO
生态一个月 4 篇的速度意味着"晚一个月,C2/C3 都可能被别人做掉"不再是修辞,是实测出版速率。

**W1 必办:** 每周一查一次 iclr.cc(CFP 通常 6–8 月上线),官方日期一出立即重排 §6 锚点。

---

## 2. 竞品时间线(2026-07-12 联网复核 + 新增 8 篇)

### 2.1 已知三篇(v1 核实过,维持判断)

| 论文 | 提交日 | 与我们的关系 |
|---|---|---|
| SafeVLA-Bench (2606.00773) | 2026-05-30(无 v2) | STL 规约度量 success-safety gap;**不做根因区分** → 我们正面回答它悬置的问题 |
| SALSA (2606.10495) | 2026-06-09(v2 06-14) | **确认 v2 = 社交导航域后训练对齐框架**。差异化四点成立:①manipulation vs 导航 ②几何因果对照 vs 语义对齐 ③冻结权重推理时门控 vs 改权重后训练 ④牛顿力+可恢复性 |
| LIBERO-Safety (2606.23686) | 2026-06-22(v2 06-26) | 程序化广度 benchmark;无 on/off 因果、无探针、无干预。**注意:它报 collision rate + 接触力**(STRATEGY §4.4 自己就写了)——Tab 1 的"无牛顿力"列必须改成"无 per-impact 力表征 + 无可恢复性 witness"(M6) |

### 2.2 新增威胁(v1 不知道,2026-07-12 检索发现;全部需进 M6 related work)

| 论文 | 日期 | 重叠 / 差异化 |
|---|---|---|
| **Your Model Already Knows**: Attention-Guided Safety Filter for VLA (2606.09749) | 06-08 | **标题直接撞 "knows" 叙事**;training-free,用 VLA 内部 attention 定位障碍 + CBF 避碰。差异:读 attention 做障碍定位 ≠ 监督探针解码"我会撞";无因果对照、无剂量-响应、无力学。**必须进 Tab 1**;也是标题策略改变的直接原因(§4) |
| **ProbeAct**: Probe-Guided Training-Free Failure Recovery (2606.09740) | 06-08 | "frozen VLA + 隐层探针 + 运行时干预"组合被**部分占位**。差异:其探针回归物体 3D 位置(空间状态),非危险/碰撞预期;针对抓放失败非碰撞力;无因果场景设计 |
| VLA-FAIL (2606.21386) | 06-19 | 隐层 Mahalanobis OOD 式 failure detection。差异:无监督 vs 我们的监督 crash 探针;无干预闭环、无因果设计 |
| SafeDojo (2606.20698) | 06-15 | 世界模型 + constrained GRPO 训练路线,与冻结权重路线正交,但抢占 "VLA 碰撞安全" 空间 |
| Constrained Flow Matching guidance (2607.01378) | 07-01 | π0.5 去噪过程约束修正避碰——与我们 π0 线直接相关;符号约束修正 vs 探针解码 |
| ForesightSafety-VLA (2606.27079) | 06-25 | 又一篇广度诊断 benchmark → 佐证"我们定位是 controlled study 不是 benchmark" |
| OopsieVerse / DAMAGESIM (2606.31993) | 06-30 | **也做接触力→损伤量化**,与"牛顿级力"卖点部分重叠;写作时区分:我们的力是因果对照的因变量,非损伤评分 |
| Hide-and-Seek trajectories (2605.30834) | 05-29 | trajectory-level runtime monitoring;M6 与 SAFE 同段落处理 |

**⚠ SafeLIBERO 生态警报:** AEGIS/VLSA(2512.11891,v2 07-02)不只是"引数字即可的 CBF baseline"——
它的 **SafeLIBERO**(LIBERO+障碍,Level I=障碍近目标 / Level II=障碍挡运动路径)已成为上面 ≥4 篇的
标准评测台。Level II 与我们的 on/off-path **表面相似**,related work 必须显式区分:他们变"难度档位"
做方法评测,无因果归因、无剂量-响应、无探针。→ 新增 **S5**:在 SafeLIBERO 1–2 个场景上跑 corridor
对照以示可比性(time-box 2 天,加分不挡出版)。

**含义(定向检索阴性结果):** 截至 07-12,**没有任何工作做"受控障碍摆放的因果归因(on/off/剂量-响应)"
或"监督探针解码碰撞预期 + 探针门控干预"的闭环**——核心组合仍未被做掉。但窗口关闭速度是 v1 估计的
约 2 倍,所有排期以"能提前就提前"执行。

---

## 3. 现状审计 v2:哪些资产够硬、哪些有暴露面

> 07-13 审计结论:**§3.1 的每一个数字都 1:1 对上了 `results/` 冻结文件**(base probe AUC、shield 三
> 元组、操作窗口、玻璃 S 曲线、跨架构行为数字、task_id=0 全部程序化复核)。资产是真的;问题集中在
> **口径外推、选择性分母、和几个文档级事实错误**。

### 3.1 够硬(论文直接用;口径已修正)

- **C1 行为因果**:on-path 100%(15/15,pilot 力 108.9–544.8N)vs clean off-path 0/33,clearance
  剂量-响应,**墙级 Fisher p=2.3e-4**(5/5 墙 vs 0/11 墙——这本来就是诚实的聚类级检验,写作时
  明说"推断单位是墙,trial 数只作描述");twin-wall(同 clearance 0.158、corridor 内外)3/3 vs 0/3
  ——**注意单独 Fisher p=0.10,不显著**,只能作为剂量-响应的 illustrative 案例或按 M5 补 2–3 对。
  **行为层跨 3 架构复现**(base/OFT/π0 各 5/5 on-path、0/10 clear;各 K=1,写作时明说)。
- **C2 表征-行为解离**:base LOSO AUC 0.993–1.0 / OFT 0.903–0.987(**T=5 仅 5 个正样本,
  Hanley-McNeil 95% CI 约 [0.72,1.0],必须带 CI 报**);off-path logit ≈ nowall ≪ on-path;临撞动作
  幅度 0.96 vs 0.55;horizon 图 T-5 穿阈值、动作反升。π0:感知层 0.56–0.78,AE 层 T=3 0.902(**8 个
  数里挑的最大值,horizon 非单调 0.71/0.90/0.73/0.87,T=5 正样本 ~7,off-path 混淆未跑**)→ M2。
  steering NEGATIVE + 正交性诊断:**crash 方向在动作读出投影中保留 <10% 范数**(‖W_action·d‖=0.742
  vs ‖W_full·d‖=7.688;不要写成"~90°"——那是把范数比错译成角度)。
- **C3 因果利用(scope = base 模型、on-path wall 类、wall-trained probe)**:probe-gated shield
  **在线闭环** 15/15 crash→0/15、321.7N→0N、误触发 0/22(nowall 10 + offpath 12),阈值 -0.422
  离线预定;**离线**操作窗口 [-0.7,2.7](3.4 logit)内 crash 0/5 + benign FP 0/20(两端均由 d85 单
  episode 决定,写作时明说);nominal-under-guard 现有 8/10(M3 扩到 N≥20)。d62 task-completing
  detour(RECOVERY_SUCCESS)存在性证明——**必须带三个 qualifier 写**:①墙降高 0.22→0.12(降后仍撞
  206N,有效性已验)②恢复是 witness 开环回放(脚本轨迹,非探针驱动规划)③触发点=episode 起始态;
  n=1/5 墙。d70/d78/d85 = **已实测负结果**(18 配置全败:147N / avoided-but-task-fail 22N / 419N),
  不是"大概率负"——这反而是"OSC 动作空间下高墙无全臂无碰撞完成轨迹"的**发现**,值得一段。
- **玻璃杯(cat-2)**:S 曲线 0/10→1/10→9/10→10/10→10/10 vs 配对 off-path 0/50;contact_force 归因
  30/30;within-glass LOSO AUC 0.944(且随 horizon 递减 0.971→0.929,与墙相反——可写);跨 hazard
  零迁移(0.356/0.466,**且 wall probe 在 glass pre-crash 帧的 fire rate = 0.0**)但 joint 0.888
  → "hazard-specific code" 是发现,同时是 shield scope 的边界证据(§4 叙事)。

### 3.2 暴露面 v2(审稿人会打的地方,按疼痛重排)

1. **⚠ 全部证据在 `libero_spatial task 0`**(41 个 scenario.json 程序化复核,task_id 全=0)。
   → **M1,第一优先级,不可砍。** 并且措辞收窄:三架构共享同一 Franka/OSC/相机/成功演示分布,
   claim 写 **"action-head-independent(单具身、成功演示 BC 范围内)"**,正文加一段共享混淆
   acknowledgment(§7-14)。
2. **⚠ probe 视觉混淆(v1 完全没列——最危险的新暴露面)**:off-path 对照排除了"有墙",但排不掉
   "墙在走廊内的视网膜-几何特征可线性解码"这个 deflationary 解读;horizon 逼近 ramp 同样改变像素;
   **跨 hazard 零迁移恰好是"hazard-specific 视觉特征"假说的预测**,§6d 现在是在给这个攻击递刀。
   → **M8(零 GPU)+ 措辞降级**(decoded-but-not-used)+ S1 层图(输入层 vs motor 层可解码性)。
3. **⚠ shield 三重暴露(v1 只列了在线/离线一条,且写反了)**:
   ①**选择性分母**:6 面 gradient-band off-path 墙真实撞了(力可达 939N),其 episode logit 全程
   ≈ -4.1,**任何窗口内阈值都看不见它们**,而它们不在 shield 分母里;`results/shield/summary.json`
   里躺着 **pooled frame AUC 0.7195**,审稿人打开 JSON 十分钟就能重构这个故事。
   ②**循环性**:阈值在训练 capture 上定+扫+验,在线闭环重跑的是**同 5 面训练墙**;无 held-out wall
   shield 测试。③**无 trivial-trigger 基线**:"距离阈值/接触力阈值 + 同一 retreat 也能 100%→0%,
   探针加了什么?"目前无任何数字回应。→ **M7(大部分零 GPU)+ scope 措辞**。
4. **π0 probe 未闭合**(同 v1)→ **M2**,capture 提前到 W2,off-path 混淆拆成独立短作业绕开 3h 墙。
5. **headline "98.3%" cherry-pick 确认属实**:= mean(env 12/12, glass f50–f70 29/30),f30/f40 事后
   排除;全档口径 = 80.0%。→ **M5 拆开报 + M0 立即把 REPORT.md/README 里的 98.3% 洗掉**(arXiv 前
   必须完成,否则文档自相矛盾被抓)。
6. **shield 在线/离线口径(v1 §3.2-4 写反了,已修正)**:headline 三元组(100%→0%/321.7N→0N/0 误触发)
   **是在线闭环**(thr=-0.422);**离线的是操作窗口**(复用 capture 扫阈值);thr=1.0 是窗口中点、
   恰是 M3 还没跑的点。另:**d62 场景被 commit 961571e 原地改矮**(0.22→0.12),而 shield/capture
   数字全部产于原几何——M3 重跑前必须固定几何(M7d),否则在线-离线对不上还说不清。
7. **小 n 统计(v1 #6 扩容)**:zero-numerator 全部要报 Clopper-Pearson 上界(0/33 → 试次级 UB 8.7%,
   墙级 0/11 → UB 23.8%;FP 0/22 → 12.7%;0/20 → 13.9%;M3 若只跑 5 条,0/5 的 UB 高达 45%——所以
   M3 扩到 10–15 条);75N 谓词的 gap 叙述与自家数据矛盾(pilot crash 最低 108.9N,非"≥150";实际
   gap ≈ [69,81.5]N)→ M5 用力阈值敏感性曲线替换该叙述;π0 T=3=0.90 是 forking paths → M5 预登记
   T=5 为主终点;ANALYSIS_shield.md 的 "509 imminent" 实为 53(509 是全 pre-crash 帧)→ M0 改;
   nondeterminism 无 seed、无来源表征 → M5 半天补。
8. **文档级事实错误(M0 卫生包,半天)**:REPORT §6e "BORDER 墙各架构不重叠" 是错的
   (base∩OFT={v3_01,v3_11},只有 π0 disjoint——改成"π0 与 OpenVLA 家族 disjoint;base/OFT 部分
   重叠,与共享 backbone 一致");REPORT.md:110 残留编辑句 "It still might just be ood still.";
   §0 主张句 ~250N 与 §3.1 322N 不一致(shield 用 321.7N,~250N 留给 pilot 描述);"90°正交"改
   "<10% 范数";**PAPER_PLAN 本身未入库、2 个 π0 probe 结果 commit 未 push、phase2_task_witness/
   summary.json 未提交且每次运行会被覆写、phase3 detour 的 PASS 证据只活在被 gitignore 的 log 里**
   ——全部一次性入库。

---

## 4. 论文形态(ICLR analysis paper)

**叙事弧不变:diagnose → localize → explain → exploit。** 开篇用问题不用现象:"VLA 撞障碍时,
是感知、预测、还是策略坏了?"(引 SafeVLA-Bench 明确悬置)——100% crash 只是 motivating observation,
贡献是因果归因 + 解离 + 机制 + 因果利用。

**标题策略(v2 新增):** 不用 "CrashBench" 或 "-Bench" 进标题(会触发与 LIBERO-Safety/ForesightSafety
的广度对比,且 "Your Model Already Knows" 已占 attention 路线的 knows 命名空间)。用发现命名,候选:
*"Seeing Is Not Avoiding: A Controlled Causal Diagnosis of Why VLAs Collide"* /
*"Decoded but Not Used: Collision Foresight in VLA Representations and Its Causal Utility"*。
CrashBench 作为 released artifact 名在正文出现一次。

**图表清单(8 页主文;状态按审计修正)**

| # | 内容 | 状态 |
|---|---|---|
| Fig 1 | teaser:场景 + 走廊示意 + crash/no-crash 帧 + 力曲线 | 素材全有,拼版 |
| Fig 2 | 剂量-响应主图 + twin-wall 内嵌 + 三架构条形 | **数据有、两块面板待画**(三架构条形无图、twin 内嵌无图;0 GPU,W1 画) |
| Fig 3 | probe:AUC × horizon × 3 架构(**每格标 n_pos**);on/off/nowall logit 分布;不减速面板 | 有,π0 待 M2;n 标注待 M5 |
| Fig 4 | layer map(S1)+ π0 VLM-vs-AE 对比(现有 PCA 图可用) | AE 对比有;层图待 S1 |
| Fig 5 | shield 操作曲线 + horizon + **trigger-baseline 对比面板(M7a)** | 曲线有;baseline 面板待 M7a |
| Tab 1 | 竞品对比(**扩到含 2606.09749/ProbeAct/AEGIS-SafeLIBERO;修 LIBERO-Safety 力列**) | 改两行 + 新增行(M6) |
| Tab 2 | 跨任务 × 跨架构 corridor 矩阵 | 待 M1 |
| Tab 3 | shield:在线三元组 + 窗口 + nominal-under-guard + **always/never-fire + 距离/接触 trigger 行(M7a)+ prompted baseline(M4)+ held-out wall 行(M7b)** | 待 M3/M4/M7 |

**§4.5 诚实边界段(写进正文的清单):** π0 感知层弱、正样本稀;d62 detour 三 qualifier(降墙/开环
witness 回放/起始态触发)+ d70/78/85 实测负 = OSC 动作空间几何限制的发现;shield scope = on-path
wall 类 + wall-trained probe,border-band 撞墙与 glass hazard 对该探针不可见(→ hazard-specific
code);sim-only;单具身/共享演示分布;力的数值是 MuJoCo solver 量,只作 labeling device
(任何 [75,105]N 阈值给出相同标签——M5 敏感性曲线),不宣称物理校准。

---

## 5. 实验清单 v2:MUST / SHOULD / CUT

### MUST(挡出版;做不完不投)

| # | 内容 | 预算 | 验收标准 |
|---|---|---|---|
| **M0 卫生包(新)** | ①commit/push:PAPER_PLAN、2 个未 push commit、phase2_task_witness/summary.json(+gif 加 .gitignore)、phase3 detour 结果导出 JSON(从 log 提取);②REPORT/README 洗 98.3%、修 §6e 不重叠句、删 :110 残句、修 shield "509→53"、统一 321.7N 与 <10% 范数措辞;③给 d62 建 `scenarios_detour/` 副本存降墙版+witness,`scenarios/` 恢复原几何(M3/M7 依赖) | **半天,0 GPU** | repo 内不存在与论文口径矛盾的数字;git 干净 |
| **M1 ★跨任务外部效度** | corridor 协议 × **≥4 个新任务**:承诺范围 = libero_spatial t1–t9 选 3–4;libero_object/goal 各 1 = **checkpoint-contingent stretch**(W1 先在登录节点预下载 per-suite checkpoints,gate 过不了立即降级)。每任务:nominal gate(≥60%)→ 录 nominal 轨迹 → 2–3 面 on-path + 4–6 分层 off-path,K=3;base 全跑,OFT/π0 各抽查 1–2 任务;**顺手存 hidden(供 S4/M8)、顺手跑 M7b**。工程注意:builder 的 HOME/BOWL/TARGET 全是 t0 硬编码手测常数,先花 1–2 天把 build 脚本参数化(从 obs 程序化读物体坐标) | 2–4 GPU-day + **1–2 天工程**(v1 漏算) | Tab 2 成型:每任务 on≫off、方向一致。新任务 crash 不必 100%,70–90% 完全够——claim 是 on/off 分离 |
| **M2 π0 probe 收口** | ①chunk 内每步记状态/慢逼近场景,T=5 正样本 7→30+;②AE off-path 混淆拆**独立短作业**(offpath-only,绕 3h 墙);③capture 全部**前置到 W2**与 M1 同批排队 | 1–2 GPU-day | 两种结局都能写:(a) AE 全档 ≥0.9 + off-path 干净 →"三架构 motor 流都可解码";(b) 仍弱 → 诚实 scope"OpenVLA 家族 + π0 方向性证据"。**悬着不能写** |
| **M3 shield 在线闭环扩容** | ①固定几何后(M0③),GuardedPolicy 在窗口中点 thr=1.0 在线跑 treatment **10–15 条**(0/5 的 CP 上界 45%,不够;10–15 条压到 <20%);可选加 thr≈2.5 上边缘点;②nominal-under-guard N≥20;③**把 6 面 border-band 墙加进在线评测**(预期 shield 看不见它们——这是 scope 数据,不是失败) | <1 GPU-day | Tab 3:在线数字与离线曲线一致;nominal 成功率差 ≤ 噪声;border-band miss 如实进 scope 段 |
| **M4 prompted-careful baseline** | "move slowly and avoid obstacles" 前缀 × 5 treatment × K=3(base;`run_pilot.py --prompt_prefix` 已有) | 半天 | 预期照撞 →"缺策略非缺指令";若有效也是重要结果 |
| **M5 统计与口径硬化(扩容)** | 每 claim 的 n/K/CI 表(做成 `results/claims_ledger.json`,论文数字可程序化对账);全部 zero-cell 报 CP 上界(trial 级+cluster 级);**预登记 T=5 为各架构主终点**,其余 horizon 进同一表 + episode 级 permutation p;episode 级 bootstrap 作 AUC 不确定度;glass 拆出 headline(全 S 曲线,98.3% 删除);glass 用 Cochran-Armitage 趋势检验替代 pooled Fisher,删 "textbook";**75N 敏感性曲线**(从冻结力迹重算 40–150N 下的墙级 crash rate,替换错误 gap 叙述);twin-wall:补 2–3 对同 clearance 内/外配对(K=3,~2 GPU-h)**或**降级为 illustrative;nondeterminism 半天表征(来源+固定 cuDNN flags+2–3 墙 ×5 rollout 稳定性)+ methods 一句话 | ~0.3 GPU-day | 论文里没有任何数字需要审稿人自己去分母里找;每个 0% 都带上界 |
| **M6 related work 定稿(扩容)** | 基于 §2 重写:新增 8 篇(2606.09749 与 ProbeAct 必须进 Tab 1 正面区分);SafeLIBERO 生态定位段(Level II ≠ 因果对照);修 LIBERO-Safety 力列;SALSA v2 定位;FailSafe/SAFE/Basu/DAgger 按 STRATEGY §4.4 落位;三个 negative 重写为 scope 决策;**§7 预答辩新增 4 行(11–14)同步埋进正文** | 1 天 | related work 一次写死 |
| **M7 shield 硬化包(新;堵 §3.2-3)** | **a)** trivial-trigger 离线基线(**0 GPU,~1 天**):在冻结 capture 上扫 ①always/never-fire ②eef-墙距离阈值(特权态 skyline)③接触力 onset 阈值,输出与 probe 并排的操作窗口/触发时刻分布/off-path 误停率 → Fig 5 面板 + Tab 3 行;**b)** held-out-wall shield(**搭 M1 顺风车,<0.5 GPU-day**):t0 训的 probe + GuardedPolicy 在 M1 新任务的 on-path 墙上在线跑(= S4 的因果版,升级为 MUST);**c)** scope 措辞:凡 100%→0% 出现处都带 "on-path wall 类、wall-trained probe";报 border-band 6 撞 + glass fire-rate 0.0 为边界;一句话解释 pooled AUC 0.72 vs LOSO ≈1.0(混合 fold 校准 + border-band 属不同 crash 模式);**d)** OFT 离线 shield sweep(**0 GPU**:复用 selfreport_oft capture 跑 probe_shield_sweep)→ 堵"shield 只有一个架构"(§7-11) | ~1.5 天,≤0.5 GPU-day | Tab 3 有 baseline 行;shield claim 有明确 scope;至少一个 held-out 设置下 shield 仍 work |
| **M8 probe 混淆对照(新;堵 §3.2-2)** | **全离线 0 GPU。** trial-level within-scene 预测:同场景多 rollout、结局离散的 bin 里,probe 早期 logit 能否预测**哪条** rollout 会撞——①glass f40(1/10)/f50(9/10)bin(selfreport_glass capture 在手);②border-band 墙 crash/no-crash 同墙对比(capture 在手;**预期阴性**——logit 平坦 -4,这本身是"corridor-class code"的 scope 证据,预写两种结局的措辞);③S1 层图作输入级 vs motor 级可解码性对照。**无论结果如何,正文措辞已按 decoded-but-not-used 降级,两种结局都可写** | 1 天,0 GPU | 混淆攻击有数据回应;若 within-scene 预测阳性 → 最强卖点,升级进 abstract |
| **M9 release/repro(新)** | 匿名代码/场景包(scenario JSON + witness + 冻结 probe + 离线复现脚本);claims_ledger 对账 CI;seed/nondeterminism 政策段;license 表(LIBERO MIT / OpenVLA MIT / openpi Apache+Gemma 条款 / OFT checkpoint);GPU-day 汇总披露 | 1 天,0 GPU(W8) | 审稿人可零 GPU 复算论文全部统计数字 |

### SHOULD(加分;每项 time-box,超时即弃)

| # | 内容 | 预算/box | 回报 |
|---|---|---|---|
| S1 layer-wise probe map | 多层 hook 重 capture(vision encoder/LLM 若干层 × horizon),base+OFT;与 π0 AE/VLM 合成 Fig 4 | 1 GPU-day / box 3 天 | "where is it decodable" 图;同时是 M8 的输入级-vs-motor 级对照 |
| S2 mid-layer steering | 中间层(~15/22)注入,扫 alpha | 1 GPU-day / box 2 天 | 正=激活级刹车;负=补全 detector≠controller 跨深度 |
| S3 glass × OFT/π0 | 玻璃协议在 OFT/π0 复现(K 减半) | 1 GPU-day / box 2 天 | 2 hazard × 3 arch 矩阵;缺格诚实标 |
| S4 探针跨任务迁移(离线部分) | 用 M1 顺手存的 hidden:t0 探针在新任务 zero-shot(在线因果部分已并入 M7b) | 0 GPU | 同 hazard 跨任务 vs 跨 hazard 不迁移的对照,深化 hazard-specific code |
| S5 SafeLIBERO 可比性对照(新) | 在 SafeLIBERO 1–2 场景跑 corridor 协议 | 0.5 GPU-day / box 2 天 | 与 6–7 月 4 篇竞品同台可比;堵"为什么不用社区评测台" |

### CUT(明确砍掉,写进 future work;理由全部有实测背书)

- **Category 5 grasp no-reset 管线** —— 3 个实测负结果在先;corridor 因果逻辑不适用;吃掉 M1/M2 预算。
- **d70/d78/d85 更多 task-witness** —— **已实测负结果**(18 配置全败,147N/22N-task-fail/419N),
  比 v1 的"大概率负"更硬;d62 存在性证明 + "OSC 无全臂解"发现段够用。
- **Phase 4 recovery-finetune(LoRA)** —— 1–3 条 demo 必被过拟合攻击;shield 已是干预;留下一篇。
- **Octo / GR00T** —— 三种动作头已覆盖;第 4 个边际收益趋零。
- **Franka 真机** —— ICLR 不需要;RSS/CoRL 备份版再补。
- **CBF/SDF 完整 shield、VLM-monitor baseline** —— 引 AEGIS/Sentinel 数字;**但注意:M7a 的
  trivial-trigger 离线基线不在此列,必须做**(它是零 GPU 且堵最大攻击,v1 把它和 CBF 混在一起砍是错误)。
- **per-hazard 通用 shield 宣称** —— 跨 hazard 零迁移已证;论文只 claim per-hazard 校准。

---

## 6. 排期 v2(10 周,2026-07-13 起;写作 W1 双轨;全部 capture 前置)

| 周 | 实验轨 | 写作轨 | 出口判据 |
|---|---|---|---|
| W1 7/13 | **M0 卫生包**;M1 选型(gate 扫 t1–t9)+ **per-suite checkpoint 预下载**+ builder 参数化动工;M4 提交;M3①②提交(固定几何后);**M7a 离线基线动工** | 查 ICLR CFP;contributions 一页 + 图表清单钉死;Fig 2 缺的两面板画掉;intro 骨架 | M1 选出 ≥4 合格任务;M7a 有初版数字 |
| W2 7/20 | **M1 主跑**(base × 4–5 任务,顺存 hidden)+ **M2 capture 同批前置**+ M7b 搭车 | related work 初稿(M6,含 8 新竞品) | base 跨任务矩阵首列成型;M2 数据在手 |
| W3 7/27 | M1 收尾(OFT/π0 抽查);M2 分析;**M8 离线分析** | method 节(场景/谓词/协议形式化) | Tab 2 完整;M8 两种结局落定其一 |
| W4 8/3 | M2 收口;M3③ border-band 在线;S1 capture+分析;M7d OFT 离线 sweep | results 节随图就位 | π0 结论二选一落定;shield scope 段成文 |
| W5 8/10 | S2/S3/S5(选做,超 box 即弃);M5 统计硬化 + claims_ledger;**全部图表定稿** | figure 终版;abstract v1 | **8/16 实验冻结**(唯一破例:M 级返工) |
| W6 8/17 | (冻结)补漏 rerun | 全文初稿完成 | 初稿给导师 |
| W7 8/24 | — | 导师意见修改;红队自审(§7 十四条逐条埋进正文);**W7 末争取挂 arXiv** | v2 + arXiv 时间戳 |
| W8 8/31 | — | 打磨;M9 release 包;(arXiv 若 W7 未挂,此周必挂) | 匿名包可复算 |
| W9 9/7 | — | 可复现附录;终稿 | camera-ready 质量 |
| W10 9/14 | — | buffer;**abstract(~9/18-19)** | abstract 投出 |
| (W11) | — | **full paper(~9/23-24,若官方日期如推断——full deadline 大概率落在 W10 之外,别按 W10 排满)** | 投出 |

**止损规则 v2:** ①**0 GPU 项永不砍**(M0/M5/M7a/M7c/M7d/M8/M9——它们堵的恰是致命攻击);
②MUST 超预算 → 砍 SHOULD 保 MUST,砍序 **S3 → S2 → S5 → S4**(S4 离线部分零成本保留);
③M1 唯一不可砍;④GPU 排队不可控 → 一切 capture(M1/M2/S1)W2 前置一次性排队,分析纯离线随时做;
⑤ICLR 官方日期一出,若早于推断 → W10 buffer 消失,立即执行砍序。

---

## 7. 预答辩:审稿攻击 × 回应(v2 扩到 14 条,写作时逐条埋进正文)

| # | 攻击 | 回应(来源) |
|---|---|---|
| 1 | "单任务单场景" | M1:base ≥4 任务全矩阵 + OFT/π0 抽查确认迁移(**别再写"≥5×3 全矩阵"——M1 产出是稀疏矩阵,措辞要匹配**);Tab 2 |
| 2 | "LIBERO-Safety 7603 场景,你几十个?" | 轴不同:受控因果 vs 程序化广度;Tab 1;它无 on/off、无探针、无干预、**无 per-impact 力表征+witness**(措辞已修) |
| 3 | "探针预测失败,SAFE/SALSA/ProbeAct/2606.09749 做过了" | 监督 crash 探针+几何因果对照+**因果干预闭环**是组合独占(07-12 定向检索阴性);ProbeAct 探针=物体位置回归、无因果设计;09749 读 attention 定位障碍、无解码"我会撞";SALSA=导航域后训练 |
| 4 | "sim-only" | LIBERO 是社区标准底座;结论是因果机制不是部署性能;真机列 future(RSS/CoRL 补) |
| 5 | "shield 是手写 retreat,不算方法" + "trivial trigger 也能 100%→0%" | claim 是"信号因果可用"非 SOTA;**M7a 给出并排数字**:probe 无特权态、预接触触发、off-path 可见墙 0/22 误停 vs 距离阈值的特权态需求与误停率;always-fire 行给出 trade-off 下界 |
| 6 | "98.3% 怎么算的" | 已删;env 100% + glass 全 S 曲线分开报(M5);repo 内旧口径已洗(M0) |
| 7 | "AUC 建立在个位数正样本上" | 显式 n+CI+episode bootstrap(M5);M2 把 π0 正样本 ×4;T=5 预登记主终点,不再挑 horizon |
| 8 | "steering 失败,凭什么说 knows?" | 措辞已降级:**decoded but not used**;正交性(<10% 范数)是机制发现:解释 gap 存在、steering 失败、gating 有效;不做认知宣称 |
| 9 | "跨 hazard 不迁移 → 不是通用安全信号" | 重新框架:hazard-specific codes 是发现;joint 0.888 证可共训;实践含义 per-hazard 校准;S4 补跨任务维度 |
| 10 | "玻璃 60% 不到 100%" | S 曲线全报 + 趋势检验;衰减档正是剂量-响应证据 |
| 11 | **(新)"shield 只在 1 个架构上"** | M7d:OFT 离线 sweep(0 GPU);正文明说 C3 是 base 上的因果可用性存在性证明,per-arch 校准 future work |
| 12 | **(新)"BC 当然撞——expected by construction"** | 开篇即问题化(感知/预测/策略三分,引 SafeVLA-Bench 悬置);tautology 预测不了剂量-响应结构、twin-wall 走廊判据、off-path probe 静默、以及安全动作存在(witness 5/5);贡献=归因+解离+机制+利用,不是 crash rate |
| 13 | **(新)"探针解码的是走廊内墙的视觉特征,不是碰撞预期"** | M8:within-scene trial-level 预测(阳性=混淆死;阴性=如实写 corridor-class code 的 scope)+ S1 输入层 vs motor 层对照 + 措辞已降级 |
| 14 | **(新)"三架构共享具身/控制器/成功演示分布,不是独立证据"** | 措辞收窄为 action-head-independent;正文 acknowledgment 段;π0 是第三方独立微调 checkpoint + BORDER 撞墙 π0-vs-家族 disjoint(**只能这么说,base∩OFT 有重叠**);M1 的 object/goal stretch 若成,部分缓解 |

---

## 8. v1→v2 差异(这次改了什么、为什么)

| 项 | v1(07-11) | v2(本版) | 理由 |
|---|---|---|---|
| §0 主张口径 | 探针 AUC≈1.0、正交、shield 不分架构一锅端 | 按架构分开写;knows→decoded-but-not-used;90°→<10% 范数;250N→321.7N | 审计发现 §0 把 base-only 结果外推到三架构;正交措辞是范数比不是角度 |
| shield 暴露面 | 只写"数字全离线、在线仅 thr=1.0 单点" | **写反了,已纠正**(headline 在线 thr=-0.422;窗口离线;1.0 恰是没跑的);新增选择性分母/循环性/无基线三条 | 对账 `intervention/summary.json`、`shield/summary.json`(pooled AUC 0.72 就在里面) |
| trivial-trigger 基线 | 与 CBF 一起砍 | **M7a MUST**(0 GPU 离线) | robotics 审稿第一攻击;砍错了 |
| probe 混淆 | 未识别 | **M8 MUST**(0 GPU)+ 措辞降级 | novelty 审稿的 fatal 级攻击;跨 hazard 零迁移在给它递刀 |
| held-out shield | S4(选做) | **M7b MUST**(搭 M1) | 循环性无 held-out 测试,审稿可打"selective reporting" |
| d62 detour | 只写 "1/5 墙" | 三 qualifier 明写(降墙/开环回放/起始触发);d70/78/85 = 已实测负 | scenario.json 里 metadata.wall_lowered 审稿人看得见;藏=concealment |
| M3 规模 | 5 条 @中点 | 10–15 条 + border-band 6 墙 + 先固定 d62 几何 | 0/5 的 CP 上界 45%;d62 已被原地改矮,不固定则在线-离线对不上 |
| M5 范围 | n/K/CI | + CP 上界/主终点预登记/bootstrap/75N 敏感性曲线/twin 补对或降级/趋势检验/nondeterminism | 统计审稿逐条实测出的窟窿(twin p=0.10;crash 最低力 108.9N) |
| 竞品 | 3 篇 | 3+8 篇 + SafeLIBERO 生态警报 + S5 | 07-12 联网检索;窗口关闭速度 ×2 |
| venue | "~9 月中下旬(W1 核实)" | 官方未公布(已核);推断 9/18–24;安全线 9/15;full 大概率落 W11 | iclr.cc 实查;W10 排期原本装不下 full deadline |
| arXiv | W8 | W7 末争取 | 竞品速率实测 |
| M0/M9 | 无 | 新增(卫生包 + release/repro) | PAPER_PLAN 未入库、结果证据活在 gitignore log 里、REPORT 与新口径矛盾、ICLR repro checklist 无人负责 |
| §7 | 10 条 | 14 条 | 新攻击面(11–14)全部来自模拟审稿实测 |
| M1 预算 | 2–4 GPU-day | + 1–2 天工程(builder 参数化)+ checkpoint 预下载 | build 脚本 t0 常数硬编码,实查 `phase1_build_env_collision.py` |

**没变的(v1 做对的、审计确认):** M1 第一优先级;M2/M4/M6 原样;CUT 全部维持(且 d70/78/85 的
砍现在有实测背书);框架锁因果对照为脊梁;负结果诚实入文;离线分析复用 capture 的习惯;
"图表清单锁定后一切不在清单上的实验冲动自动失效"。

---

## 9. 立刻可动手(下周一开始)

1. **M0 卫生包(登录节点,半天):** commit/push 全部漂流资产;洗 REPORT(98.3%/§6e/:110);
   建 `scenarios_detour/`、恢复 `scenarios/` d62 原几何;phase3 detour PASS 从 log 导出 JSON。
2. **提交 4 个 sbatch(排队并行):** M1 gate 扫描(t1–t9 各 K=5)· M4 prompted(5 墙×K=3)·
   M3①(thr=1.0 在线 10–15 条,固定几何后)· M3② nominal-under-guard(N≥20)。
   同时登录节点:per-suite checkpoint 预下载 + builder 参数化动工。
3. **离线双杀(0 GPU,登录节点即可):** M7a trivial-trigger 基线脚本(仿 `probe_shield_sweep.py`
   扫距离/力阈值)+ M8 within-scene trial-level 分析(glass capture 在手)。
   这两个是本周期性价比最高的两件事——**各一天,各堵一条 fatal/major 级审稿攻击。**
4. **写作轨:** contributions 一页 + 14 条预答辩表发导师对齐;Fig 2 两块缺失面板画掉(0 GPU)。
