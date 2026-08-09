# HISTORICAL — not the current execution plan

See [PAPER_PLAN.md](../PAPER_PLAN.md) for the current plan.

# CrashBench — 战略分析与文献定位(2026-06-28)

> 这份文档回答三个问题:**①我们到哪了、②为什么会"卡住"、③结合文献该往哪走。**
> 它是一次完整调研的产物:用一个多 agent workflow 从 6 个角度检索了 ~55 篇 2021–2026 的相关论文,
> 再对 3 个"新颖性关键 claim"做了对抗式(try-to-refute)核查,最后综合成战略建议。
> 配套:总览 [REPORT.md](REPORT.md) / 进度 [STATUS.md](STATUS.md) / 计划 [PLAN.md](PLAN.md)。
> 本文同时充当 PLAN.md §10 所说的 `RELATED.md`(related-work 定位)。

---

## ⚠️ 关于文献时效的诚实声明(务必先读)

本文献部分由网络检索 agent 于 2026-06 生成。**多篇最具威胁的竞品带 2026 年(`2606.*`/`2605.*`/`2603.*`/
`2601.*`)的 arXiv 编号,日期就在本月前后**;检索 agent 自己已多处标注"待核实"。在把它们写进论文 related work
或据此声称差异化**之前**,你必须**逐一 `WebFetch` 核对**:编号是否真实、作者、venue、**确切提交日期**(早于本工作 =
prior art 需差异化;晚于/同期 = concurrent work,声明即可)。下面凡是 2026 编号一律视为"待核实"。

---

## 1. 一句话结论

**你不是科学上卡住了,是在追错误的轴。** 手里的四件套已是一篇逻辑闭环、统计扎实的 paper;真正的问题是
**2025–2026 这一年 VLA 安全这块挤进来一大批工作**,你下意识想靠"加第二类 hazard(广度)"补强,而广度恰恰是
你最该回避、也比不过的轴。文献核查后,你**唯一没被预占、且能扛住审稿的新颖性 = 几何因果对照**(固定障碍外观、
只变它在手臂扫掠走廊里的归属 → crash 100%→0%)。升级方向不是加 hazard,而是 **Path 1(把探针变成因果干预)+
Path 3(跨策略复现)** —— 详见 §6。

---

## 2. 现状回顾:从 motivation 到四件套

**Motivation([motivation.md](motivation.md)):** VLA 几乎只在**成功演示**上训练,没见过"快出事"的状态;
把它丢进 pre-crash 态,失败属于**感知 / 预测 / 策略**三者中的哪一个?CrashBench 要把答案钉死。

**四个 load-bearing 结论(全部 ✅,详见 [REPORT.md](REPORT.md)):**

| # | 结论 | 证据 |
|---|---|---|
| R1 | VLA 对 pre-crash **没有避让策略** | 注可见墙挡路 → crash **100%(15/15)**,impact 均值 ~250 N,3–9 步逼近(非瞬时退化) |
| R2 | 是**安全策略缺口,不是 OOD 泛化差**(核心贡献) | 同一面墙挪开路 → **0/33**,clearance 剂量-响应,墙级 **Fisher p=0.0002**;判据是**走廊**非直线距离 |
| R3 | 场景**公平**(崩溃可避免) | 5/5 safe-abort 可恢复(retreat+hold,**0 N**) |
| R4 | 是**安全缺口不是感知缺口**(知道却不避) | 冻结隐层线性探针 LOSO **AUC 0.99–1.0**;临撞**不减速**(动作幅度 0.96 vs 0.55);off-path 对照证明解码"我会撞"非"有墙" |

motivation 的三分法被 R2/R3/R4 干净地钉在了 **policy** 上。**科学链条没断。**

---

## 3. 诊断:为什么"卡住"

卡点来自 Phase 2-④ 那 **3 个连续 NEGATIVE**(libero-10 太菜 / 矮物体被垂直下压让过 / 抓取状态过不了 reset)。
你把"加第二类 hazard"误当成"项目还没完成"。但:

1. **加 hazard 追的是广度,而广度是错误的轴。** LIBERO-Safety 用 10 模型 × 5 suite × 7600 场景碾压你;你赢不了。
2. **第二类 hazard 本是为堵"红墙 OOD 假象"的反驳,而那个反驳早被 R2(off-path 对照)严格堵死了。** 所以你在为
   一个**已经不需要的保险**反复烧 GPU —— 这才是"卡住"的体感来源。
3. **3 个 negative 不是项目失败,是 simulator 在告诉你:静态注墙是你手里唯一干净、高胜任度、可因果控制的工具。**
   它们各自破坏了 env_collision 成功的一个条件(①模型胜任、②障碍挡在行为路径、③状态可 round-trip),反而**反证**了
   静态注墙为什么强。把这 3 个 negative 重写成一段**主动的 scope 决策**("刻意限定单一高胜任度 hazard 以实现干净的
   因果归因"),四个尴尬就变成一段 methods。

---

## 4. 文献调研全景(本文重点)

### 4.1 调研方法
一个多 agent workflow:**Phase A** 6 个并行检索 agent,各管一个角度(每个跑 18–25 次网络检索/抓取),共得
**55 篇** 2021–2026 论文,逐篇标 `scoop_risk ∈ {none, low, medium, high}`;**Phase B** 对 3 个"新颖性关键 claim"
各派一个**对抗式审稿 agent**(扮 harsh CoRL/RSS reviewer,任务是**找到把你 scoop 掉的 prior art**);**Phase C** 综合。
总计 10 个 agent、~39 万 token、189 次工具调用。

### 4.2 六个角度的 landscape(各自的"水有多深、缺口在哪")

**① VLA 安全 / 碰撞 benchmark —— 2025–2026 爆发、已拥挤。**
至少 4 个 LIBERO/RoboCasa 安全 suite(LIBERO-Safety、SafeVLA-Bench、SafeLIBERO/AEGIS、Safety-CHORES)+ 一批鲁棒性
压测(Eva-VLA、VLATest、LIBERO-Plus)。"success-safety gap"已是被命名、被测量的现象;**在 OpenVLA 上注障碍、量
collision rate / contact force 已经存在**(LIBERO-Safety 的 Tabletop Spatial Avoidance 重叠最大)。
→ **"VLA 会撞墙""success≠safety"本身已不新颖。** 缺口 = 机制 + 因果归因的组合,没有任何一个 benchmark 提供。

**② 失败检测 / 运行时监控 —— "策略内部信号能预测自己即将失败"基本已成立。**
SAFE(冻结 VLA 末层特征分出 success/failure"failure zone")、FIPER / FAIL-Detect / Sentinel / INSIGHT(用 OOD /
不确定性 / 一致性信号、仅靠成功数据 + conformal 校准预判失败)。
→ **R4 的"裸命题"(隐层能预测撞)不新颖。** 缺口 = 把检测推进到**因果干预**(不只 detect/abort,而是触发返回
known-good state),并证明"知道却不做"这个 gap 是**可因果干预的**。

**③ 探针 / 可解释性 —— 三条线逼近但都不重合。**
(a) 探 OpenVLA 隐层已成立(Emergent World Representations、Tufts 符号态探针),但只解码**良性**状态、从不碰碰撞、
也无行为/可操作性 claim;(b) 读 VLA 内部预测**失败**很热(SAFE 最危险);(c)"知道却不做"的探针 gap 只在**文本 LLM**
上被干净表述(Basu et al.:0.982 AUROC 探针 vs 45% 行为检出,steering 只修好 24%),无机器人、无碰撞、无因果操控。
→ 缺口 = 单个**线性**探针在冻结 VLA 残差流解码**即将碰撞**(AUC~1.0)+ on/off-path 因果分离(读"我会撞"非"有墙")
+ 不刹车的行为证据。**SALSA 是这块最危险的预占**(见 §4.3)。

**④ 失败恢复 / 重规划 —— 拥挤且在快速收敛。**
"VLA 只学成功 → 缺 pre-crash 策略"这个**核心 thesis 已被同期工作明说**(FailSafe、Failing Forward/AFIL、RACER),
**不再新颖,应作为既定 motivation 引用**。恢复侧:RecoveryChaining、Recovery RL、SC-VLA、Back to the Manifold、
Latent Policy Barrier、本实验室的 Cold Diffusion = "返回 known-good state"。
→ CrashBench 的位置 = 给这些恢复方法提供**可证可恢复的 out-of-feasible-region 测试场景** + 干净的因果归因。

**⑤ BC / 协变量漂移 thesis —— 直接祖先是 DAgger(2011)。**
Ross-Bagnell 的复合误差(T² regret)就是 CrashBench thesis 的理论原型;墙 = 专家演示从未访问的状态。
→ **抽象 thesis 不能当新颖点**;新颖点是把它**物理化 + 因果隔离**。

**⑥ VLA 鲁棒性 / OOD / 对抗 —— 标准做法是单维扰动测成功率下降。**
LIBERO-Plus 是 OOD 扰动的标准 taxonomy(视角/光照/布局/语言/噪声);对抗侧有 Wang 2024、AttackVLA;BYOVLA 做
distractor 鲁棒。**它们都不注良性挡路障碍、不量接触力、也不区分"OOD-感知失败"与"缺动作策略"。**
→ CrashBench 的 off-path 对照正是这个区分,且证明墙是**被正确感知**的(探针读"我会撞"),所以 mask 像素式的修法不适用。

### 4.3 对抗式新颖性核查:三个核心 claim 的判决

| 核心 claim | 判决 | 最该对标的 prior work | 怎么办 |
|---|---|---|---|
| **C1** 没有 benchmark 把 VLA 丢进 pre-crash 态、测它避不避**自身执行路径上**可见障碍(on-path 100%) | **partially-anticipated** | **LIBERO-Safety**(2606.23686*)同 suite/同 OpenVLA/注障碍/量 collision;**SafeVLA-Bench**(2606.00773*)量牛顿力 | 别声称"又一个 VLA 安全 benchmark"新颖;紧扣三件没人合起来做的:受控因果隔离 + 可恢复性 + 探针 |
| **C2** 同一面墙 on/off 走廊 → 100%→0% 剂量-响应,**因果隔离"安全策略缺口 vs OOD"** | **novel-as-worded(最强、唯一能站住的"第一")** | benchmark 重叠仍是 LIBERO-Safety;方法学先例可引 KAGE-Bench("固定其它、只变一个 nuisance 维") | **这是论文脊梁。** 一句差异化:"不像 LIBERO-Safety/SafeVLA-Bench 把几何/布局/外观一起变并报聚合 collision rate,我们固定外观、只变扫掠走廊归属" |
| **C3** 线性探针解码即将碰撞(AUC~1.0)却**不刹车**,off-path 对照证明读"我会撞"非"有墙" | **partially-anticipated → 若把探针当 headline 则滑向 scooped** | **SALSA《Act on What You See》**(2606.10495*):冻结 7B VLA 线性探针解码"逼近危险"、明说 representation-behavior gap、测不减速、有解耦对照 —— **几乎一模一样(导航域)**;次级:SAFE、Basu et al. | **探针绝不能当"发现"写**,否则被当成不了解同期工作直接 reject。重写为"representation-behavior gap 在 **manipulation 域的首次实例 + 更强的几何(非语义)对照 + 近乎完美 LOSO AUC + 进一步做了因果干预**",显眼引 SALSA |

> **一句话:** benchmark 轴、"VLA 会撞"轴、探针"知道却不避"框架**都被预占**;**唯一活下来的是 C2 的几何因果对照**,
> 而它正好回答了 **SafeVLA-Bench 公开承认没回答的问题**("我们不区分 unsafe 来自 distribution shift 还是缺安全目标")。

### 4.4 关键竞品深读(必须 cite-and-differentiate)

- **LIBERO-Safety**(arXiv 2606.23686*,Tsinghua/BAAI/Beihang)—— **最近的 benchmark 竞品**。同 LIBERO + OpenVLA,
  程序化注静/动/新障碍,7603 场景、10 模型,报 collision rate + 接触力。**差异:** 它随机放障碍、几何+布局+外观一起变、
  违例即终止、报聚合率;**不做** on/off-path 同墙因果对照、**不做** 隐层探针、**不**量每次冲击牛顿、**不**证可恢复性。
  且 OpenVLA 在其难 suite 上近 0% 成功,**无法**像你的高胜任度路径那样干净隔离碰撞行为。
- **SafeVLA-Bench**(2606.00773*)—— 正式定义并量化 "success-safety gap",**用牛顿力**(200 N / ISO-TS 15066)。
  **差异:** 后验打分现有 rollout、**不**注挡路墙、**无** on/off-path 对照、**无**探针,且**明说不区分** OOD vs 缺安全目标。
- **SAFE**(NeurIPS 2025,2506.09937,Toyota/Toronto)—— **R4 最近的 prior**。在 OpenVLA/pi0 冻结特征上训轻量检测器,
  失败 rollout 落入共享"failure zone"。**差异:** 用非线性 MLP/LSTM(非严格线性探针)、目标是**通用失败**非碰撞、
  是 reactive 检测、**不**做 on/off-path 分离、**不**提"不刹车"的行为 dissociation。
- **SALSA《Act on What You See》**(2606.10495*)—— **C3 最危险的预占**。导航 VLA(OmniVLA)上冻结线性探针解码"逼近
  危险"、明说"编码了危险却既不刹车也不改动作"、测 clearance、还有把危险信号与障碍存在解耦的对照。**差异(你更强):**
  域不同(manipulation vs 导航)、对照是**几何 swept-corridor**(非语义 inpaint)、AUC~1.0(vs 其 ~74%),且你**进一步做了
  因果干预**(Path 1)。**必须显眼引用。**
- **FailSafe**(2510.01642,2025)—— **thesis 重叠最高**,几乎逐字说出 CrashBench 核心命题、还用 OpenVLA。**差异:** 它注
  kinematic 扰动(平移/旋转/no-op)非可见物理障碍,测**任务成功**非碰撞力/剂量-响应,无探针。→ thesis 引它,别声称 thesis 新颖。
- **ActProbe**(2606.08508*)与 **Failing Forward/AFIL**(2605.08434*)—— 前者用 action-chunk magnitude 预判失败(正好碰到你
  "临撞动作幅度更大"的统计),后者重述"success-only BC → 脆弱"。两者均 2026、**待核实日期**(可能同期)。

---

## 5. 定位:6 列对比表(CrashBench 是唯一一行全勾)

| 工作 | 同一障碍 on/off 对照 | clearance 剂量-响应 | 统计隔离检验 | 牛顿级力 | 隐层探针 | 可恢复性证明 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| LIBERO-Safety (2606.23686*) | ✗(随机放) | ✗ | ✗ | ✗ | ✗ | ✗ |
| SafeVLA-Bench (2606.00773*) | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| SAFE (NeurIPS'25, 2506.09937) | ✗ | ✗ | ✗ | ✗ | ✓(通用失败) | ✗ |
| SALSA (2606.10495*) | ✗(语义 inpaint) | ✗ | ✗ | ✗ | ✓(导航) | ✗ |
| **CrashBench(本工作)** | ✓ | ✓ | ✓(Fisher p=2e-4) | ✓ | ✓(+因果干预) | ✓ |

**论文主张一句话:** *固定障碍外观、只变扫掠走廊归属的受控 on/off-path 对照,首次**因果地**证明 VLA 碰撞是"缺安全
策略"而非 OOD;并用解码"我会撞"(而非"有面墙")的隐层探针佐证、策略不刹车;**进一步用该探针触发恢复 / steering,
把冲击从 ~250 N 降到接近 0 N**(跨多个 VLA 复现行为效应)。* **这张表 + 这句话,就是你的贡献。**

---

## 6. 三条前进路径 + 推荐(已选 Path 1 + Path 3,冲 main-track)

| | 做什么 | 成本 | 回报 | 连接的工作 |
|---|---|---|---|---|
| **Path 1 ★核心** | 把现成"我会撞"探针**拿来用**:① 触发已证 0 N 的 safe-abort;② activation steering 诱导刹车;测冲击力下降 | ~1.5–2 周,**<1 GPU-day** | **最高**:把论文从"诊断"升级为"诊断+因果干预",正好补上 SALSA 没做的一刀 | SALSA、**SAE steerable features(2603.19183,steering 先例)**、Basu et al.(LLM steering 只修 24%,在机器人上超它)、Cold Diffusion / Latent Safety Filters(恢复目标) |
| **Path 3 顺手做** | π0 / Octo / diffusion policy 跑同一套 on/off-path 协议 | ~1.5–2 周,1–3 GPU-day,纯推理 | 中:堵"只测 OpenVLA"软肋;跨架构复现 → 因果 claim 架构无关 | LIBERO-Safety / SafeVLA-Bench(多 VLA 评测是常态)、SAFE(跨策略 failure-zone 先例) |
| Path 2 后置 | 跑 README 全套干预(prompted / VLM-monitor / CBF shield / recovery-finetune) | ~3–4 周,2–5 GPU-day | 中高但工程量大、部分已被占 | AEGIS(CBF)、Sentinel(VLM monitor)、RACER/RecoveryChaining/SC-VLA(recovery-finetune) |

**下周第一件事(保底必成):** Path 1-1a,<1 GPU-day,拿"**探针触发把冲击 ~250 N → ~0 N**"这一个数字 —— 这是
SALSA/SAFE/Basu 都没有的、把"知道却不避"升级为"用这个信号因果地修好行为"的关键证据。实现细节见 [PLAN.md](PLAN.md) §11。

---

## 附录 A:完整文献清单(55 篇去重后按主题分组)

> 格式:`[scoop_risk] 标题 — 作者/venue/arXiv`,后接一句"它做了啥 / 与我们的关系"。**所有 2026 编号待核实。**

### A.1 VLA 安全 / 碰撞 benchmark(benchmark 轴,scoop 主战场)
- **[HIGH] LIBERO-Safety** — Cui, Zhang 等(Tsinghua/BAAI/Beihang),arXiv 2606.23686*(2026)。LIBERO 注障碍、10 模型、
  collision rate + 接触力。**最近 benchmark 竞品**;无 on/off 因果、无探针、无牛顿级单撞、无可恢复性。
- **[HIGH] SafeVLA-Bench** — Fan, Xu, Sokolsky, Lee, Kong,arXiv 2606.00773*(2026)。定义 success-safety gap,牛顿力
  (200 N ISO-TS 15066)。后验打分、无注墙、无 on/off、无探针、**明说不区分 OOD vs 缺安全目标**。
- **[MED] VLSA / AEGIS + SafeLIBERO** — Hu 等,arXiv 2512.11891*(2025)。CBF 安全层 bolt-on 冻结 VLA + SafeLIBERO
  障碍 benchmark。重心是**缓解方法**;可作 Path 2 的 CBF baseline。
- **[MED] SafeVLA / Safety-CHORES** — B. Zhang, Y. Wang, J. Ji 等,arXiv 2503.03480(2025)。约束 RL(CMDP)安全对齐 +
  Safety-CHORES(导航重)。"训练侧修复"分支;共享 motivation。
- **[MED] SafeManip** — arXiv 2605.12386*(2026)。LTLf 时序安全 8 属性(含"撞后回退到安全态"= 与你 safe-abort 近)。
  评 pi0/GR00T;**不**注挡路墙、**不**测 OpenVLA。
- **[MED] SafeDojo** — arXiv 2606.20698*(2026)。world-model 想象 + 约束 RL,从不安全态恢复,跑 SafeLIBERO。可作
  benchmark 上的 method baseline;注意 SafeLIBERO 与你场景集是否重叠。
- **[LOW] Eva-VLA** — arXiv 2509.18953(2025)。连续优化压测 OpenVLA(3D 变换/光照/对抗 patch),>60% 失败。**对比项**:
  扰动视觉、无障碍、无力、把鲁棒性与安全混为一谈。
- **[LOW] VLATest** — Z. Wang 等,FSE 2025,arXiv 2409.12894。fuzzing(障碍数为 operator 之一)。把障碍当 clutter 测成功率,
  非挡路 hazard。
- **[LOW] LIBERO-Plus** — Fei 等(OpenMOSS/Fudan),arXiv 2510.13626(2025)。OOD 扰动**标准 taxonomy**(7 单维因子)。
  **你要对比的"标准 OOD 测法"**:无障碍、无力、不分感知 vs 策略。

### A.2 失败检测 / 内部信号读出(R4 轴,scoop 主战场)
- **[HIGH] SAFE** — Gu, Ju 等(Toyota/Toronto),NeurIPS 2025,arXiv 2506.09937。冻结 VLA 特征预测失败、共享 failure zone,
  含 OpenVLA。**R4 最近 prior**;非线性头、通用失败、无 on/off、无"不刹车"。
- **[HIGH/MED] ActProbe** — Huang, Li, Wang, Cao 等,arXiv 2606.08508*(2026)。action-space 信号(chunk 一致性 + 幅度)早期
  预判失败。**正撞你"临撞动作幅度更大"**;但用动作空间非 4096d 隐层、非碰撞、无 epistemic claim。待核实日期。
- **[MED] Sentinel** — Agia, Sinha, Yang 等(Stanford),CoRL 2024,arXiv 2410.04640。扩散策略运行时监控(STAC + VLM video-QA)。
  监控动作分布/进度非隐层;须 cite 为 monitoring baseline。
- **[MED] FIPER** — Roemer 等(TUM),NeurIPS 2025,arXiv 2510.09459。生成策略**运行前**预判失败(RND OOD + chunk 熵)。
  框成 OOD/不确定性(与你"非 OOD"相反)→ 好用的 foil。
- **[MED] FAIL-Detect** — Xu 等,RSS 2025,arXiv 2503.08558。无失败数据的序贯 OOD 失败检测。"仅用成功数据监控"已非新颖。
- **[MED] INSIGHT** — arXiv 2510.01389(2025)。VLA(pi0-FAST)token 级不确定性触发求助;时序不确定性比静态分数更预测失败。

### A.3 探针 / 可解释性(R4 方法学先例 + "知道却不做"血脉)
- **[KEY] SALSA《Act on What You See》** — Q. Wang, X. Wu, G. Shi, D. Chen, X. Yang, D. Manocha,arXiv 2606.10495*(2026)。
  **C3 最危险预占**(导航 VLA 线性探针 + representation-behavior gap + 不刹车 + 解耦对照)。**必须显眼引用并差异化。**
- **[MED] Interpretability without actionability** — Basu 等,arXiv 2603.18353*(2026)。**"知道却不做"的经典表述**:0.982 AUROC
  探针 vs 45% 行为检出,steering 只修 24%。文本 LLM/临床,无机器人 → CrashBench 是其 embodied 实例。
- **[MED] Probing OpenVLA for Symbolic States(DIARC/Tufts)** — H. Lu, H. Li, Scheutz 等,arXiv 2502.04558(2025)。OpenVLA 33 层
  线性探针解码符号态(>0.90)。**你探针法的同模型/同 setup 先例**;但解码良性态、无安全/可操作性。
- **[MED] Emergent World Representations in OpenVLA** — Molinari 等,arXiv 2509.24559(2025)。线性探针 > MLP,信号在中层
  (~15,22);OpenVLA 有内部世界模型。**可行性先例。**
- **[LOW] VLA Already Has Attention Heads for Path Deviation Detection** — Jeong 等,arXiv 2603.13782*(2026)。冻结导航 VLA
  (NaVILA)注意力熵 head 检测偏离。同精神(信号已在模型内),但注意力 head 非线性探针、导航非碰撞。
- **[LOW] SAE Reveal Interpretable & Steerable Features in VLA** — Swann 等(Stanford),arXiv 2603.19183*(2026)。SAE 提取
  **可因果 steering** 的特征(LIBERO + 真机 DROID)。**Path 1b steering 的直接先例 → 重点引、可借其方法。**
- **[LOW] CoFreeVLA** — 作者未确认,arXiv 2601.21712*(2026)。短时自碰撞风险估计 + 引导恢复。最 collision-specific,但建外部
  风险模块而非诊断"已编码却忽略"。作者待核实。
- **[NONE] Othello-GPT 线性涌现世界表示** — K. Li 等,ICLR 2023,arXiv 2210.13382;线性化 follow-up Nanda 等 2309.00941。
  **R4 的概念祖先**(冻结隐层线性可解码世界模型)。纯血脉引用。

### A.4 失败恢复 / 重规划(CrashBench 作为其 eval 目标)
- **[NONE] Cold Diffusion on the Replay Buffer** — Wang, Oba, Yoneda, Shen, Walter, Stadie,CoRL 2023,arXiv 2310.13914。
  **本实验室锚点**:经 known-good states 规划、留在 feasible region。CrashBench = 提供 region 外、"返回最近 known-good"
  为正解的场景。
- **[NONE] Recovery RL** — Thananjeyan, Balakrishna 等(UC Berkeley),RA-L/ICRA 2021,arXiv 2010.15920。任务策略 + 恢复策略
  回安全集。**经典 safe-set/recovery 引用。**
- **[LOW] RecoveryChaining** — Vats, Likhachev, Kroemer(CMU),IROS 2025,arXiv 2410.13979。分层 RL 把失败态链回 known-good。
  **最贴你"retreat+hold 返回 known-good"**;可作 benchmark 的 baseline/target。
- **[LOW] SC-VLA** — Liu, Chi 等,arXiv 2405.17418(2024/25)。单 VLA 快路径 + 慢纠错路径(检测到碰撞→CoT 纠错)。
  正是你证明 vanilla OpenVLA 缺的恢复环;可压测。
- **[LOW] ReflectVLM(Reflective Planning)** — Feng 等,CoRL 2025,arXiv 2502.16707。VLM planner + 扩散动力学想象 + 反思纠正。
  高层 planner 层,非低层动作策略。
- **[LOW] Uncertainty-aware Latent Safety Filters** — Seo, Bajcsy 等,arXiv 2505.00779(2025)。隐空间世界模型安全滤波,接近
  OOD/不安全态即介入留在 feasible region。**与锚点同源**;CrashBench 提供其需要的 region 外测例。
- **[LOW] Back to the Manifold** — Reichlin, Marchetti, Kragic 等,IROS 2022,arXiv 2207.08673。BC 离流形不能恢复 → 学恢复策略
  steer 回 in-distribution。**锚点的概念孪生。**
- **[LOW] Latent Policy Barrier** — arXiv 2508.05941(2025)。学隐式 barrier 让 visuomotor BC 留在训练分布。同"留在 feasible
  region"族。

### A.5 BC / 协变量漂移 thesis(motivation,非新颖点)
- **[LOW] DAgger(Reduction of IL to No-Regret Online Learning)** — Ross, Gordon, Bagnell,AISTATS 2011。**thesis 理论祖先**:
  BC 复合误差 T² regret;墙 = 专家未访问态。**必引为命题起源**;抽象命题不能当新颖点。
- **[HIGH] FailSafe** — Lin, Duan, Fang, Fox, Krishna, Tan, Wen,arXiv 2510.01642(2025)。**几乎逐字** CrashBench thesis、用 OpenVLA。
  注 kinematic 扰动非物理障碍、测成功非力、无探针。→ **thesis 的主锚点。**
- **[MED] Failing Forward / AFIL** — Zheng 等,arXiv 2605.08434*(2026)。success-only BC → 无纠错信号;Dual Action Generators 作
  负向引导。**thesis 最强同期重述**;待核实日期。

### A.6 VLA 鲁棒性 / 对抗(对比项,确立你"非 OOD/非对抗")
- **[LOW] Adversarial Vulnerabilities of VLA** — T. Wang, Han, Liang 等,arXiv 2411.13587(2024)。对抗 patch 劫持轨迹(至 100% 退化)。
  **对比**:它是 worst-case 对抗 artifact,你的墙是良性可见障碍、失败是常规策略缺口。
- **[LOW] AttackVLA** — arXiv 2511.12149(2025)。对抗/后门攻击统一 benchmark。确认这一族**全是恶意注入**,无良性 hazard 碰撞 benchmark。
- **[LOW] BYOVLA** — Hancock, Ren, Majumdar(Princeton),arXiv 2410.01971(2024)。运行时编辑掉无关视觉区→提鲁棒(OpenVLA/Octo)。
  **感知侧隔离的类比**:它 mask 像素证明"哪些视觉内容重要",你移墙证明"哪些空间内容重要";但你的失败在墙被正确感知下仍发生,
  mask 式修法不适用。

---

## 附录 B:本分析的可信度与局限
- 文献由检索 agent 于 2026-06 网络抓取;**2026 编号需人工核实**(见顶部声明)。SALSA、LIBERO-Safety、SafeVLA-Bench、ActProbe、
  AFIL 这几个**最影响定位**的,优先逐一 WebFetch 核对真伪与日期。
- 对抗式核查只针对 3 个 claim;若投稿前再扩,建议补一轮"completeness critic"(专找"还漏了哪一类工作")。
- 原始数据:workflow 输出与各 agent 完整结构化结果已留痕于本 session 的 task 输出/transcript;55 篇全表存于
  scratchpad `lit_full.json`(如需可再导出为 BibTeX)。
