# Path 3 — 跨架构 probe fit:「模型自己知道要撞了吗?」(2026-07-08)

> **一句话:** 把 R4 的自报告 probe(从冻结策略**自身隐状态**线性解码「T 步内会撞」)从单一架构
> 推到三个架构。**OpenVLA 家族(base + OFT)强成立**(AUC 0.90–1.0,off-path 混淆干净);
> **π0 只部分成立**(VLM prefix 层 AUC ~0.73)—— 而且**不是样本少的锅**,是 tap 层/架构的真差异。

## 结果(三模型,同一 probe:PCA-50 + L2 logreg,**leave-one-scenario-out**,标签「crash within T」)

| 模型 | tap 层 | AUC T=1 | T=3 | T=5 | T=10 | off-path 混淆(logit)| 刹车? |
|---|---|---|---|---|---|---|---|
| **OpenVLA (base)** | LLM 末 token(post-norm)| 0.993 | 0.992 | **0.998** | 1.0 | on −0.30 ≫ off −5.00 ≈ nowall −5.28 | 不刹 0.96 vs 0.55 |
| **OpenVLA-OFT** | LLM 末 token(post-norm)| null | 0.987 | **0.903** | 0.919 | on −1.92 ≫ off −4.80 ≈ nowall −4.84 | 不刹 0.97 vs 0.50 |
| **π0 / openpi** | VLM(PaliGemma)末层 prefix 末 token | 0.561 | 0.768 | **0.728** | 0.784 | on −3.11 > off −4.92 ≈ nowall −4.90 | 不刹 0.82 vs 0.66 |

- **三个模型都不刹车**:撞前 ≤2 步的动作平移幅度不降反升 —— 行为上零回避,与 R4 主结论一致。
- **off-path 混淆**:三者的 off-path logit 都 ≈ nowall、远低于 on-path,probe 解的是「我要撞了」而非
  「看见墙了」。但 on/off 的分离度逐级变弱:base Δ≈5 logit,OFT Δ≈3,**π0 只 Δ≈1.8**。

## 为什么 π0 弱 —— 而且**不是样本太少**(关键的诚实点)

三个跨架构模型的正样本都稀疏(chunked 执行 + 撞得快 → 撞前 query 帧很少),但对照下来这不是主因:

| 模型 | wall 帧 | **T=5 正样本** | AUC T=5 |
|---|---|---|---|
| base | 38 | 28 | 0.998 |
| **OFT** | 13 | **5** | **0.903** |
| **π0** | 20 | **7** | **0.728** |

**OFT 只有 5 个正样本却仍 0.90;π0 有 7 个反而只 0.73。** 同样极稀疏,OpenVLA 家族的 LLM 末 token
照样线性可解,π0 的 VLM prefix 不行。所以 π0 的弱**不能全归因于样本少** —— 更像是 **tap 层的真差异**:

- OpenVLA/OFT 的动作是从 **LLM 末 token 的 post-norm 态**直接解出(离散 token / L1 回归头都读它),
  所以「运动意图 / 我要撞」自然编码在那一层 → 线性可读。
- π0 是 flow-matching:**动作意图在 action expert 的去噪过程里**,而我们(按既定选择)tap 的是
  **VLM 感知 prefix**——它更偏「看到了什么」,未必线性编码「我这一步要撞」。有分离信号(on-path
  logit 中位 1.44 vs 负样本 −3.0),所以是**弱而真实**,不是零。

## 边界 / 如实声明(不藏)

- **π0 结论是「部分成立」,不是「成立」**:AUC ~0.73,且 T=1 只 0.56(≈随机)。
- **正样本个位数**:OFT/π0 的 LOSO 每折留出场景只有 1–2 个正样本,AUC 方差大 —— 报出来,别当定论。
- **π0 的 T=1 时间墙教训**:首跑 90min 被 CANCEL(offpath 21 场景 + 每 requery 双 prefix 前向太慢);
  已改为**每 condition 增量落盘** + walltime 3h,重跑完整拿到 1108 帧。
- probe 是「自身隐状态线性可解码」,**不是**模型能被问答自报;这是 action-only VLA 唯一严谨的自报告测法。

## 这step证明了什么

「VLA 撞墙是缺安全策略、不是 OOD」这个**行为几何因果**此前已跨 3 架构(base/OFT/π0,均 on-path 100% /
clean off-path 0%)。**本step补的是「内部可解码性」这条正交证据**:
- **在 OpenVLA 家族里,「知道却不做」得到跨架构确证**(base→OFT,AUC 都 ≥0.90);
- **π0 上,「知道」在我们所 tap 的感知层只弱成立** —— 悬而未决:是信号不在 VLM(而在 action expert),
  还是采样太稀。下一步可消歧。

## 下一步(消歧 π0)

1. **换 tap 层**:改取 π0 **action expert** 的输入/中间态(运动意图所在),同一 probe 重跑。若 AUC 跳到
   0.9+,则证实「π0 也知道,只是知道在别处」——把跨架构结论补全成 3/3。
2. **加密采样**:给 π0 造更多「撞得慢」的场景(更远起手、更小步)或对 chunk 内每步都记状态,把 T=5
   正样本从 ~7 提到几十,消掉方差这个混淆。

## 复现

```bash
# 三模型各自:capture(GPU)→ analysis(CPU)
sbatch setup/probe_selfreport.sbatch                     # base  -> results/selfreport/
sbatch setup/probe_selfreport_oft.sbatch                 # OFT   -> results/selfreport_oft/
sbatch setup/probe_selfreport_pi0.sbatch                 # π0    -> results/selfreport_pi0/
# analysis 可单独在登录节点重跑(纯 numpy/matplotlib):
python scripts/probe_selfreport_analysis.py results/selfreport_oft
python scripts/probe_selfreport_analysis.py results/selfreport_pi0
```

capture 脚本 `scripts/probe_selfreport_capture.py` 模型无关(`--policy {openvla,openvla-oft,pi0}`);
hidden hook:OFT = `language_model.model.norm`(每 8 步 requery 记一帧),π0 = 复刻 openpi 输入变换后
单跑一次 prefix 前向取 `prefix_out` 末 token(每 5 步,不改 openpi 源码)。
