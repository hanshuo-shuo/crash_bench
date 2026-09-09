> 后续开发已完成并核验，见[完整实验报告](RESULTS_ZH.md)。以下保留资产核验各阶段的发现与当时状态。

# 固定 Detour benefit gate：资产核验

日期：2026-09-09。第一项交付，本地及 Quest 资产核验完成；新实验实现和运行尚未完成。没有新训练、rollout、Slurm 提交或 D8 操作。
本地分支 `codex/odur-repair`，HEAD `f6a927d2684d06c9d09a1da3de564be29754b0e1`，与用户参考一致。
已有未跟踪目录 `docs/audits/20260908/candidate_refresh_assessment/`、`results/direct_cost_learning/` 原样保留。
用户本轮方案取代旧文档对本轮开发工作的停止授权描述；历史结论不改写。

## 1. 实际文件与身份

实际本地目录 `results/counterfactual_router/full_d4751330395e_20260814T152231Z/` 包含原始
`capture_manifest.json`、`decision_metadata.json`、`option_rollouts.jsonl`、`decision_features.npz`。
四项 SHA-256 **全部匹配**冻结 resolver 的 `manifest.json:input_sha256`，详见 `assets.csv`。
它们是本机现存资产，并非根据 GitHub 链接补造。Git 默认文件搜索忽略了该结果目录，因此必须检查实际磁盘。

采集提交 `d4751330395eec11691f4830761b76f5f1ac36b7`，manifest 声明当时工作区 clean。
Backbone 为 `openvla/openvla-7b-finetuned-libero-spatial`，revision / model config commit 均为
`962318cec55ac10993ff0f5f43eda9a270b4c873`，do_sample=false、num_beams=1。
远端记录位置为 `/gpfs/projects/p33100/siosio/huggingface_cache/hub/models--openvla--openvla-7b-finetuned-libero-spatial/snapshots/962318cec55ac10993ff0f5f43eda9a270b4c873`。
该 revision 是模型身份声明，不等于本次核验了权重文件字节；权重、依赖版本及实际运行环境尚未远端验证。

控制器为该历史提交的 `crashbench/recovery.py:DetourComplete`。
`assets.csv` 同时记录历史 Git blob 和当前文件 SHA-256，不能把当前文件哈希冒充旧执行版本。
固定参数见 `audit_summary.json:protocol.detour`；side=-1、lane_margin=.12、lift_offset=.38、
descend_offset=.04、grasp_xy_offset=[.009,-.04]、departure_clearance=.06、leg_cap=140、path_aligned=true。
protocol SHA-256 为 `b487d8b5cc20c1fe932d8ee4ff67f05883e2b010b6ee9eb155eb9b93c2d1a4c3`。

Detour 使用特权模拟器几何：glass 的位置/尺寸，碗和盘的位置，以及搬运时碗相对 EEF 的偏移。
no-glass 条件仍注入 on-path glass 的虚拟几何。这些是控制器依赖，不能描述为纯视觉部署能力。

## 2. 完整 bundle 与重放

本地 capture 只有四个主文件及 `capture_progress.jsonl`，没有序列化完整 bundle。
历史 collector 只在内存保留 XML、sim state、controller state、observation；写盘的是五类哈希与特征、终局。
缺失持久化字段：原始 XML/环境状态、控制器 payload、policy continuation、全部 RNG、传感缓存、输入图像与上下文、pending proposal/队列、逐步轨迹。
`continuation_state_sha256` 是环境 controller runtime 字典的哈希，并非完整 policy/RNG bundle 的证明。
本地其他 Refresh bundle 来自 π0 和不同任务设置，不能代替这批 glass 锚点。
因此不能精确重放旧 glass 状态，也不能从终局摘要恢复完整成功累计曲线。

新锚点可以在用户授权范围内另行采集，但不能命名为旧状态精确重放；须先验证实际环境和来源资产。
下列历史 placement 清单本地不存在，其相对引用的 source state 文件目前也无法解析核验：

- `results/glass_recovery_v2/pilot_b_task0_20260811_r5/frontier_placements/placements.json`
- `results/glass_recovery_v2/pilot_b_fresh_h20_20260812_r2/frontier_placements/placements.json`

Quest 首条命令 `scripts/quest_sync.sh check` 返回 `SSH socket not found: /tmp/quest.sock`。
失败后未运行任何远程命令。Quest checkout、项目存储中的 bundle、placement、checkpoint 均标记为尚未核验，而不是不存在。

## 3. 输入时序和队列

历史 `_scan_condition` 顺序：保存 pre-action 状态/观测 → policy.act → 读取 last_hidden → 保存 robot_state/nominal_action → env.step。
所以 hidden 和 nominal action 在对应动作执行前生成；hidden 已包含生成动作的计算，不能称为推理前信号。
`glass_force_after` 在执行后写入扫描内存，没有进入 `decision_features.npz`。
特征形状为 hidden [273,8,4096]、robot_state [273,8,8]、nominal_action [273,8,7]、mask [273,8]，全部有限。
本轮仅可使用当前有效帧；8 帧历史不能自动成为新输入，也不能用它虚增 PCA 秩。

旧 Base OpenVLA 每次输出一个动作，没有 π0 的五动作 chunk。决策时当前 proposal 已生成、执行长度为0、待执行动作1、后续缓存队列0；之前执行长度见 matched_scan_index。
但旧 Base 分支 reset policy 后重新调用 act，没有直接复用扫描 proposal，也没有完整记录该重复调用的观测/RNG一致性和成本。
新 B 应执行已保存的 proposal 并保持 continuation；R 消耗同一决策机会、丢弃该 proposal，Detour 完成后用新观测恢复 Base，不再触发。
这属于新 runner 必须实现并验证的合同，不是旧数据已有语义。拒绝也必须锁定候选机会，不能用旧 FirstCrossingRouter 的“仅接受后 latch”直接替代。

## 4. 来源、条件、重复、终止与预算

273 决策、819 终局、20 物理来源（历史 train=5、calibration=7、development=8）；每决策每选项恰好一次。
23 有效 placement，35 attempted。101 glass 决策、86 offpath、86 noglass；控制组各仅17来源。
完整 `source_condition_table.csv` 保留计划清单27来源×3条件（其中20来源有终局），缺失条件填空值/0数量，不填零成功率。
`terminal_records.csv` 包含全部三选项终局；`fixed_detour_pairs.csv` 包含每个锚点的 Base/Detour 对比。
原 capture 的排除清单保存在 `audit_summary.json`：14 catastrophe_before_horizon、30 condition_terminal_before_matched_anchor、
7 onpath_no_catastrophe、5 scan_invalid_initial_state。层级不同，不能将这些数直接作为 episode 总数相加。
这批数据按未来碰撞选择锚点且排除了部分早终局，不能估计新协议的无触发率或总体 episode 成功率。

`anchor_index = collision_step - horizon_actions + 1`，所以 [40,30,20,10,5] 是碰撞相对距离，绝不是任务预算 H。
历史 collector / sbatch 默认：scan=220，Base branch=220，Detour branch=900，Retreat=80，settle=10。
源码检查事故优先于同一步成功；另外检查环境 episode_terminated。
终局共158事故、285成功、6显式 robosuite_episode_horizon、370未写明确终止原因的 noncompletion。
默认参数和记录步数可辅助解释后者，但不能伪造其缺失的 termination 字段。
manifest 未保存解析后的总步数参数或真实环境 horizon；Slurm 实际命令/依赖未核验，不能将默认值当运行时完整配置。
旧 Detour 从头到尾运行结构化控制器，走完 legs 后原地开夹保持，没有返回 Base。
**旧分支预算不对称，且缺轨迹。旧摘要不能生成有效的共同 H/2H 比较，也不能沿用 Refresh 的100/200。**
新 H 必须从该任务和运行时配置核实，验证环境支持2H，再冻结；不以最有利截止时间选 H。

## 5. 固定 Detour 相对 Base 重算

每来源先平均现存决策，再在来源间等权。下表是暴露旧数据的描述，不使用 Detour/Retreat 的事后 max。

| 子组 | 来源/决策 | Base 成功 | Detour 成功 | 成功差 | 事故差 |
|---|---:|---:|---:|---:|---:|
| 全部 |20/273|51.97%|43.47%|−8.50 pp|−20.92 pp|
| glass |20/101|2.25%|24.25%|+22.00 pp|−54.75 pp|
| offpath |17/86|88.73%|61.76%|−26.96 pp|+4.71 pp|
| noglass |17/86|92.06%|58.43%|−33.63 pp|0.00 pp|

原始成功计数157→128；60次原成功损失、31次正向成功转换；7次新增事故。
总体来源等权原成功损失率20.36%，新增事故率3.17%。净事故降低不能掩盖新增事故。
完整数据见 `fixed_detour_summary.csv`。子组来源支持不同，不能将三组宏平均简单平均复原总体。
一次配对不认证状态效应，预算不对称和碰撞筛选也限制解释。此次没有验证稳定收益、可学习信号或风险基线优势。

## 6. 可因果调用的风险候选与成本

已有 `FrozenOutcomeRouter.predict` 计算 `base_catastrophe_probability`，输入为当前 hidden PCA、robot_state、nominal_action。
在正常 Base act 后调用不增加 VLA 次数；代价为4096×k投影、scaler、线性风险头及sigmoid，实际毫秒数须在目标环境测量。
本地未找到可验证的对应 frozen router 权重/manifest，旧全来源/LOSO模型也不能直接充当新12/4划分训练侧冻结的触发器。
可选训练侧固定风险分数，必须先核实其训练来源边界、权重及阈值；当前不凭空冻结一个阈值。
已有传感状态/此前力信号可作为另一个因果触发器候选，但原 capture 没有保存可直接在线复用的完整历史传感轨迹和校准产物。
禁止用未来 collision_step/horizon、option结局、condition/source ID 代替风险。事故后触发也不是预防候选。
过去已生成 hidden 可复用；禁止额外调用 VLA 补特征。候选扫描本身和重复 branch 全成本均需单列。

## 7. Ridge 代数核验与配置边界

相同 X、来源权重W、scaler/PCA与正则P时，T=(XᵀWX+P)⁻¹XᵀW 与目标无关，
因此 T(yR−yB)=TyR−TyB；未惩罚截距也不破坏等价。
现有 `fit_weighted_ridge` 正是来源等权、共享标准化、alpha=1的线性多输出解。
合成矩阵代数诊断最大误差8.33e−17，没有拟合研究标签。
本轮同输入两臂 DirectQ 与直接差值 Ridge 合并为 sanity check。历史三臂 utility、max选择与校准不同，不能整套宣称等价。

新12 fitting sources至多36候选，但可能少得多；保持PCA16前必须对实际 fitting 当前帧矩阵检查 n−1和秩。
当前旧矩阵尺寸不能证明新样本支持16维；不加重复帧充秩，不按结果挑维度。
待资产核验后固定可行维度，若输入/秩不足则记录支持缺口。

## 8. 本轮尚未完成的实施工作

`development_contract.json` 保存用户已固定的范围与待定字段，明确不是可提交配置。
来源物理清单、12/4分配、候选触发器及阈值、真实H/2H、环境依赖/权重字节哈希、恢复语义实现、成本计数与质量测试尚未完成。
必须先建立来源清单并排除D8/未暴露验证来源，再按固定非结局规则选择；不能拿旧成功配置补足16来源。
之后按用户上限实施384 scored branches+48前缀；无触发/早成功/早事故保留总体分母，不补人为锚点。
后续288 branches+24前缀条件验证不自动排队。

本次仅验证审计脚本的哈希匹配、计数、唯一配对、终局互斥与Ridge代数；这些不是新 runner 的单元测试。
新 runner 仍须覆盖用户列出的无触发、早终局、同一步事故优先、连续H/2H、控制器时间、队列恢复、标签隔离、来源隔离及失败不算加速。
资产不完整和连接缺口不表示永久终止研究。

复现本地审计：`python docs/audits/20260909/detour_benefit_gate/audit_assets.py`。


## 9. Quest 连接恢复后的补充核验（取代上述“远端尚未核验”状态）

使用沙箱外已授权的 SSH socket 后，`scripts/quest_sync.sh check` 成功。
Quest 路径 `/gpfs/home/shv7753/crash_bench`、origin `git@github.com:hanshuo-shuo/crash_bench.git`，分支 `codex/odur-repair`。
Quest HEAD `f67e28457a58c557ecccf98cd7c189d0a12150f9` 是本地参考 HEAD 的祖先，落后3个提交，没有分叉；主仓库工作区干净。本次没有修改远端源码或同步提交。

两份 placement 清单及全部27个不同物理状态文件在 Quest 均存在，已取回明确文件并验证所有35条 placement 的 source array SHA-256。
详见 `available_sources.csv` 与 `quest_asset_evidence.json`。27是尝试清单的来源数；20是保留273个终局决策的来源数，不能混淆。
这更正早期文字“20来源×3条件表”：实际表含27来源×3条件，缺结局来源保留空值。
所有这些来源都在历史开发清单中，不能作为未来“未暴露8来源”。此前已经有 hazard screening，也必须作为来源选择历史披露。

四个 capture 主文件在 Quest 的哈希全部与本地及 resolver 一致；该精确目录同样只有五个摘要/特征文件，没有 bundle payload。
不能排除另有未知备份，但不能假设存在，更不能用哈希重建旧环境。

实际 OpenVLA task runner (`third_party/openvla/experiments/robot/libero/run_libero_eval.py`) 明确 libero_spatial max_steps=220。
因此新协议固定原任务预算H=220、长预算440，从 episode 的连续任务动作时钟读出，前缀和干预均占用任务预算；reset settle=10另记，不在候选处重新给完整预算。
LIBERO wrapper源码默认horizon=1000、ignore_done=false；运行前必须断言实际环境至少支持settle+440，不能仅凭源码默认值声称已完成运行验证。

依赖 Git HEAD：LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`，OpenVLA `c8f03f48af692657d3060c19588038c7220e9af9`。
检查未发现依赖已跟踪源码改动；OpenVLA存在未跟踪 `experiments/logs/`、`rollouts`，保留不动。
环境包与权重清单见 `quest_checkpoint_inventory.json`：torch2.2.0+cu121、transformers4.40.1、numpy1.26.4、mujoco3.9.0、robosuite1.4.1、libero0.1.0。
四个safetensors shard均存在；大权重仅记录实际大小和缓存blob名称，尚未在计算节点重新计算字节哈希，不把缓存名充当本次校验。

已有风险模型实际可用：
`results/counterfactual_router/fresh_online_2eab4a4a53dc_20260817T080708Z/router.json` 与 `router.npz`。
权重SHA-256 `46fd264b5258cb1be20a4c16f259025dc6c5ab9650afc75d65742eed16412e69` 与manifest完全一致。
PCA16与风险头只在历史5来源/71状态拟合；旧阈值使用7来源/96状态校准。新12/4来源表必须考虑这12个既有来源的使用记录，不能将其假装未用于模型选择。
该分数可以因果调用；历史阈值不能不经边界核对就用于新calibration。
本机预热100次完整predict调用中位约0.030 ms，P95约0.048 ms；包含outcome头，不是Quest/机器人端到端时延，不包含VLA，额外VLA调用0。

固定模型do_sample=false意味着策略分布本身退化；不同后续种子不能自动称为不同随机策略样本。本轮需要如实区分同bundle数值/运行时波动与策略随机性，不能更改解码设置以制造随机性。

第一项资产核验现已交付。后续仍需完成16来源冻结、候选阈值、输入秩约束、runner与单元测试、计算节点权重校验及正式执行；本次没有训练或新轨迹，也没有排入条件性验证。
