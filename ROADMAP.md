# CrashBench — ROADMAP(唯一的"下一步"入口 · 更新 2026-07-02)

> **这是什么:** 项目**前瞻/计划**的唯一权威文档。把原先散在三处的下一步(旧 `下一步计划_v2`、
> `STRATEGY.md §6` 的 Path、`STATUS.md §8` 的交接)整合到这里,并更新到 **Week-1 已完成**之后的状态。
>
> **配套底稿(被引用,不重复):** 进度快照 [STATUS.md](STATUS.md) · 带图技术报告 [REPORT.md](REPORT.md) ·
> 傻瓜版 [OVERVIEW.md](OVERVIEW.md) · 导师原始 spec [PLAN.md](PLAN.md) · 文献定位/竞品 [STRATEGY.md](STRATEGY.md) ·
> thesis [motivation.md](motivation.md)。
>
> **性质:执行 PLAN.md 的修订版,不是改方向。** framing 已由两个 pilot 锁定为 **"knows but doesn't act"**
> (crash 100%>50%;probe AUC 0.99–1.0>0.8,均触发 PLAN §11 最高档)。下一步 = 在此 framing 下,
> **以最小 breadth 支撑最强 analysis + 干预**,而非堆场景广度。

---

## 0. 一句话战略(来自 STRATEGY.md)

**不是科学卡住,是别追广度这条追不赢的轴。** LIBERO-Safety(7603 场景/10 模型)在广度上碾压;
唯一没被预占、能扛审稿的新颖点 = **C2 几何因果对照**(固定障碍外观、只变它在手臂扫掠走廊里的归属 →
crash 100%→0%)。升级方向 = **Path 1(把探针变成因果干预)+ Path 3(跨策略复现)**,不是加 hazard。

**论文主张一句话:** *固定障碍外观、只变扫掠走廊归属的受控 on/off-path 对照,首次因果地证明 VLA
碰撞是"缺安全策略"而非 OOD;隐层探针解码"我会撞"(而非"有墙")+ 不刹车佐证;进一步用该探针触发恢复 /
steering,把冲击 ~250 N → ~0 N。* (定位对比表见 STRATEGY.md §5,必引竞品 SALSA / LIBERO-Safety /
SafeVLA-Bench / SAFE 见 §4.4。)

---

## 1. 现在到哪了(Week-1 后快照)

| 模块 | 状态 | 一句话 |
|---|---|---|
| R1 headline(注墙 crash 100%) | ✅ | 15/15,impact 均值 ~250 N |
| R2 OOD 对照(**核心贡献**) | ✅ | 剂量-响应,off-path 0/33,Fisher **p=0.0002**,REFUTED |
| R3 场景公平(可恢复) | ✅ | 5/5 safe-abort,0 N |
| R4 自我报告探针(知道却不避) | ✅ | LOSO AUC 0.99–1.0,临撞不减速 |
| **Path 1-1a** 探针→因果干预 | ✅ | crash 100%→0%,impact 322N→0N,误触发 0/22 |
| Path 1-1b 激活 steering | 🟡 NEGATIVE | detector≠controller(crash 方向~正交 action readout) |
| **Week-1① probe-gated shield 阈值扫** | ✅ **2026-07-02** | **3.4-logit 安全窗口 [-0.7,2.7]** 内 crash 0/5 + benign FP 0/20;纯离线。[`ANALYSIS_shield.md`](results/ANALYSIS_shield.md) |
| **Week-1② horizon 重标注(§8.1)** | ✅ **2026-07-02** | on-path logit **T-5 穿阈值**、action 反升 → horizon 轴的"知道却不刹车"。[`fig_horizon.png`](setup/figures/fig_horizon.png) |
| Week-1③ Category 2 玻璃杯 → **并入 headline** | ✅ **全量 07-02 · 并入 07-07** | on-path **30/50 (60%)** vs 5 组配对 off-path **0/50**;S 形剂量-响应(f30 0→f40 10%→f50 90%→f60/f70 100%)。谓词判对:30/30 由 `contact_force`(robot-vs-glass ≥25N,中位 41N)触发,位移/倾倒为下游后果;clean pass 不误报。修正 cookies 负结果(短→**加高**)。**已并入 headline suite**(口径 B:挡路档 f50–f70=29/30=96.7%)→ **跨类别平均 headline 98.3%**(env 100%+obj 96.7%),`scripts/headline_suite.py` 纯离线。[`ANALYSIS_glass.md`](results/ANALYSIS_glass.md) · [`fig_glass.png`](setup/figures/fig_glass.png)(job 5843331) |
| **Probe-on-glass**(自我报告探针扩到 cat-2) | ✅ **2026-07-07** | within-glass LOSO **AUC 0.94**(墙同管线 1.00)→"知道却不避"跨 hazard 成立;joint 双 hazard 一个探针 **0.89**(有共享危险方向)但单 hazard 零样本**不迁移**(墙→杯 0.36/杯→墙 0.47)。job 6238135。[`probe_glass_summary.json`](results/selfreport_glass/probe_glass_summary.json) · [`fig_glass_probe.png`](setup/figures/fig_glass_probe.png) |
| **Path 3 跨架构复现(OpenVLA-OFT)** | ✅ **2026-07-07** | 同场景 apples-to-apples,**按墙几何分箱**:on-path 墙(x≈−0.1)**100%(5/5)**、off-path CLEAR(x≥0.22)**0/10**,与 base **不可区分**(base 同场景 100% / 0/10)。BORDER 带(x<0.22 贴走廊边)base 4/11≈OFT 3/11 是轨迹噪声,非 OFT 弱点。→ 几何因果对照**架构无关**。OFT env 在 p33100(不占 home)。[`ANALYSIS_path3_oft.md`](results/ANALYSIS_path3_oft.md) · `scripts/path3_oft_compare.py`(纯离线) · jobs 6253349/6253555/6253963 |
| Phase 2-⑤ task-completion witness | ✅ 1/5 | d62(需降墙);d70/d78/d85 几何受限 |
| Phase 3 绕行 recovery | ✅ d62 | `RECOVERY_SUCCESS`(bare 对照 CRASH) |
| Phase 4 微调数据导出 | ⬜ 未做 | 卡在只有 1 条 witness |

---

## 2. 三条前进路径(STRATEGY.md §6;已选 Path 1 + Path 3)

| | 做什么 | 成本 | 状态 |
|---|---|---|---|
| **Path 1 ★核心** | 把"我会撞"探针拿来用:①触发 0 N safe-abort ②steering 诱导刹车 ③操作曲线/horizon | ~1.5–2 周,<1 GPU-day | 1-1a ✅ / shield ✅ / horizon ✅ / 1-1b steering 🟡NEG |
| **Path 3 顺手** | π0 / Octo / OpenVLA-OFT 跑同一套 on/off-path 协议 → 因果 claim 架构无关 | 1–3 GPU-day,纯推理 | 🟢 **OpenVLA-OFT ✅ done(07-07)**:同场景 on 100%/clean-off 0%,与 base 不可区分。π0/Octo 待做 |
| Path 2 后置 | README 全套 baseline(prompted / VLM-monitor / CBF shield / recovery-finetune) | 2–5 GPU-day | 部分:shield=Path1 已覆盖;其余见 §5 |

---

## 3. 场景族:最小 breadth 方案(修订 PLAN §4.3 七类)

- **Category 1 环境碰撞 ✅ done** — 26 场景 + 全套控制/witness/probe。
- **Category 2 物体碰撞 ✅ 原型 done(2026-07-02)** — 注入式**可倒玻璃杯**放 nominal path 上、目标物不动
  (不触发 OOD);predicate = 倾倒角(`object_toppled`)/位移(`object_displaced`)/接触力,均单测+GPU-free 验过。
  全量(K=10,5 组配对):on-path **60%** vs off-path **0%**,S 形剂量-响应,30/30 由 contact_force(≥25N)触发。
  基元(`inject_movable_objects_xml` + `reset_to` 拼接)复用现有 pipeline。**✅ 已并入 headline suite(07-07,
  口径 B:挡路档 f50–f70=96.7%)→ 跨类别平均 headline 98.3%(`scripts/headline_suite.py`)。下一步 =
  Path 3 跨策略复现(OpenVLA-OFT/π0)。**
- **Category 5 抓取失稳(新增,需修管线)** — snapshot-reset 存不下夹爪挤压力(**已 documented failure**,
  STATUS §6.2)。**正确做法 = no-reset:** 改物理属性(降摩擦/加重)+ 从 t=0 活体抓取,或换 `joint_force_limit`
  (状态=关节角,可干净 round-trip);**不要**再走"快照重放 + 注入外力"。预计 1.5–2 周。
- **Category 3/4/6/7 — blocked**:自碰撞单臂构造不出;joint-limit 目标出工作空间→OOD;edge 类已有 documented failure。

## 4. 模型矩阵(缩围 PLAN §6)

OpenVLA(✅)→ **OpenVLA-OFT(✅ 2026-07-07)**→ **π0**(openpi LIBERO checkpoint,复用 harness)。
GR00T N1 / Octo 标 **stretch**,不承诺。核心主张至少要在 **2 个模型**上复现,才从个案变 VLA 性质(= Path 3)。

### 4.1 环境/目录清单(⚠ home 配额已满 → 一切新 env/仓库/checkpoint 全落 `/projects/p33100/siosio`,不占 home)

| 组件 | 位置 | 说明 |
|---|---|---|
| **base OpenVLA env** | `~/crash_bench/envs/openvla`(home,8.5G,**已有**) | torch2.2.0+cu121 / transformers4.40.1 / flash-attn2.5.5 / mujoco3.9.0。`source activate ~/crash_bench/envs/openvla` |
| base OpenVLA 仓库 | `~/crash_bench/third_party/openvla` | |
| **OFT env** | `/projects/p33100/siosio/envs/openvla-oft`(8.6G) | 与 base 同栈;`source activate /projects/p33100/siosio/envs/openvla-oft`。crashbench **未** pip 装进来 → 跑作业须 `export PYTHONPATH=$HOME/crash_bench` |
| OFT 仓库 | `/projects/p33100/siosio/third_party/openvla-oft` | `export OPENVLA_OFT_ROOT` 指这;跑 OFT 时 `export CRASHBENCH_OPENVLA_ROOT=$OPENVLA_OFT_ROOT`(让 env 与 policy 用同一 `experiments.robot`) |
| OFT checkpoint | `HF_HOME=/projects/p33100/siosio/huggingface_cache` 内 | `moojink/openvla-7b-oft-finetuned-libero-spatial` |
| pip cache / 构建 tmp | `/projects/p33100/siosio/{pip_cache,tmp}` | 安装时 `export PIP_CACHE_DIR TMPDIR` 指这,别爆 home /tmp |

**装 env(登录节点,需外网):** `bash setup/install_openvla_oft_env.sh` · **验证(GPU):** `sbatch setup/verify_openvla_oft.sbatch`
**跑 OFT:** `sbatch setup/run_pilot_oft.sbatch`(默认 walls;换场景 `--export=ALL,SCEN=...,OUT=...`)。
**踩坑:** mujoco 3.10 会崩 robosuite 1.4.1 的 `mj_fullM` → 必 pin `mujoco==3.9.0`。

### 4.2 π0(openpi)接入 —— 给下一个执行者的落点

- **必须新建独立 env**(π0 是 **JAX** 栈,base/OFT env 都没 jax;完全隔离,零污染)。**同样落 p33100**:
  建议 env=`/projects/p33100/siosio/envs/openpi`、仓库=`/projects/p33100/siosio/third_party/openpi`、
  checkpoint 走 `HF_HOME`(已在 p33100)。安装/验证/运行脚本照 `setup/*openvla_oft*` 那三件套改写。
- **harness 已通用**:policy 工厂 `crashbench.policies.build_policy("pi0", ...)` 已注册占位
  (`crashbench/policies/registry.py`),只差写 `crashbench/policies/pi0_policy.py`(实现 `Policy`
  协议:`resize_size` + `act`;若 π0 也吐 action chunk,照 OFT wrapper 的 deque + `reset()` 写)。
  `--policy pi0` 即可复用 `run_pilot.py` 与全部场景。
- **协议不变**:跑 `--scenarios scenarios`(on-path 墙)+ `--scenarios scenarios_control`(位置扫描),
  再 `python scripts/path3_oft_compare.py` 同款离线对照(可加一列 π0)。
- **对照分箱按几何**(墙 x 坐标),别按场景名 —— 详见 [`ANALYSIS_path3_oft.md`](results/ANALYSIS_path3_oft.md)。

## 5. Baseline 表(solo 版,修订 PLAN §7)

1. Vanilla ✅
2. Prompted-careful("move slowly, avoid collisions" 前缀)— 极便宜,半天
3. **Probe-gated shield** ✅(= Path 1-1a + Week-1 阈值扫;三数字 + 3.4-logit 安全窗口已出)
4. CBF/SDF shield — 引 AEGIS 结果或做简化 SDF 版
5. Recovery-finetuned — **前置:先有 ≥3 条 witness**(见 §9),否则 1 条必过拟合;35/15 held-out
6. Cold-diffusion-on-replay-buffer — 与导师对齐排期,衔接后续 method 论文

进阶:分层 probe(vision encoder vs LLM 各层,堵"像素层可解码"攻击);activation steering 已试(🟡NEG)。

## 6. Solo 排期(8–10 周,已划掉完成项)

| 周 | 交付物 | 状态 |
|---|---|---|
| 1 | ~~probe-gated shield 三数字~~ + ~~horizon §8.1 图~~ | ✅ 两件均完成(纯离线) |
| 2 | Category 2(物体碰撞)搭建 + 跑通 | ✅ done(全量 60% vs 0%,S 形剂量-响应) |
| 3–4 | Category 5 no-reset 管线;OpenVLA-OFT 复现 | ⬜ |
| 5 | π0 接入;分层 probe;prompted-careful baseline | ⬜ |
| 6 | Recovery-finetuned(35/15)+ 外观 ablation(墙色/纹理,加固反 OOD) | ⬜(卡 witness) |
| 7 | (有 Franka)真机 10 trials @T-5;否则 steering/分层 probe | ❓ 待确认 Franka |
| 8–10 | 写作 + 场景 suite / witness 数据集打包发布 | ⬜ |

## 7. 立刻可动手(下一步优先级)

1. ~~**Category 2 玻璃杯原型 + 全量 + 并入 headline suite**~~ ✅ **done(全量 07-02;并入 07-07)** —
   on-path 60% / off-path 0%,S 形剂量-响应,contact_force 归因 30/30。**已并入 headline suite**(口径 B:
   只算挡路档 f50–f70 = 29/30 = 96.7%)→ headline 升级为 **跨类别平均 98.3%**(env_collision 100% +
   object_collision 96.7%),`scripts/headline_suite.py` 纯离线复现。**下一步 = Path 3 跨策略复现。**
2. Path 3 起步:OpenVLA-OFT(同库近零成本)在现有 on/off-path 协议上复现 crash 100%/0%。
3. (可选,便宜)shield 闭环复核:`GuardedPolicy(base, probe, thr=1.0)` 跑安全窗口中点,~18 min/点确认离线曲线。

## 8. 开会 checklist(对导师)

- [ ] 汇报 §11 两个 gate 触发 → framing 锁定(引原文 "These two pilots decide the framing")
- [ ] §14.1 OOD 反驳已完成(p=0.0002)
- [ ] **新:** probe-gated shield 操作曲线 + 3.4-logit 安全窗口(100%→0% 是稳健工作区非调参巧合)
- [ ] **新:** horizon §8.1 图(T-5 就知道、却越逼越用力)
- [ ] Category 5 no-reset 修订提案(定性为 PLAN §4.4 bug fix)
- [ ] 出示 SALSA / LIBERO-Safety / SafeVLA-Bench / SAFE,讨论 related work 重定位(STRATEGY §4.4)
- [ ] 确认 Franka 可用性(决定第 7 周内容)
- [ ] 对齐 cold-diffusion baseline 排期与分工

## 9. 交接:未决工作 + 踩坑(细节见 STATUS.md §8)

**选项 A — 多拿 witness(d70/d78/d85):** 这三面墙 `frac` 大、+y 边≈0.25 贴着碗,抓取时前臂必穿墙,
OSC 管不到肘 → 降到 h=0.10 仍撞(d70 fmax=147、d85 fmax=419)。大概率负结果,诚实记录即可。

**选项 B — Phase 4 微调导出:** 前置最好 ≥3 条 witness(否则 1 条必过拟合)。步骤:`witness_to_hdf5.py`
(照抄 `regenerate_libero_dataset.py:161-199` 字段)→ 外部 rlds builder → TFDS → LoRA(rank 32,
**必须** mixture 混入原始 libero_spatial demo)。

**踩坑(别再犯,详见 STATUS §8):** ①绕行不要控末端姿态(OSC 发散)②读碗/盘坐标前先 10 步 settle
③放置要补偿抓取偏移④witness(~332 步)> max_steps 要设 500⑤闭环 `DetourComplete` 未调通,现用开环 `WitnessReplay`。

**跑作业:** `sbatch setup/xxx.sbatch`,account `p33100`,partition `gengpu`,`MUJOCO_GL=egl`,
env `~/crash_bench/envs/openvla`;登录节点无 GPU,只能写代码。结果 GIF/MP4 在 `results/`(gitignore)。
