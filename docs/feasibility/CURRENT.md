# 安全未完成状态的可行性见证与受控续接实验 — 2026-10-01

用户明确授权本诊断实验，所有仿真/推理/图表在Quest；没有训练授权。当前活动goal的全部原始
需求见本聊天；protocol.py固定六个诊断状态与两个正常对照，不做总体评估或概率认证。

## 已完成

- 本地3项新协议测试、22项复现测试、27项配对/恢复测试通过；py_compile及Bash语法检查通过。
- 预检8169406因目标实体应为akita_black_bowl_1而失败，没有GPU实验或科学结果，记录保留。
- 修正后CPU/short预检8169759、8170141均完成八状态，PREFLIGHT.status=passed。
  目标/目标容器存在、碰撞网格读取与单椭球包含性通过，普通OSC平移.05m、旋转.5rad/action。
  普通机器人能力不改；参考轨迹只通过7D OSC/gripper。不能直接写qpos或移动障碍/对象。
- 初始旧提交e27cfc2的GPU作业8169055在PENDING时取消；无动作执行，原根有SUPERSEDED.json。

## 活动作业（必须读最新Quest记录，不从本文推断完成）

1. GPU冒烟8170183，16次，源码f8d89948b49aaba70335b59277d5d6fc34811114。
   Quest根`/projects/p33100/siosio/crashbench_safelibero/feasibility/20261001T181952Z_smoke_f8d89948b49a/`。
   两状态Spatial03/Object00 × (5条件+3固定参考方案)。提交后PENDING，Slurm曾估计15:44芝加哥开始。
   检查plan/progress/STOP/SMOKE_COMPLETE及slurm_8170183.log。源archive不可改。
2. CPU旧证据图8170378，源码0b606242c6744ff39cafc539501644878b1cbf35。
   Quest根`/projects/p33100/siosio/crashbench_safelibero/feasibility/20261001T182246Z_prior_0b606242c674/`。
   只读取已完成600行summary/raw.csv，输出report/PRIOR_REPORT.md、PNG、JSON、HTML。
   这些是旧证据，不回答新见证与续接问题。

Quest live checkout仍是main/d22035816647321b8c9c44674efb76273916db95，clean；不触碰其他项目jobs。
新源码均在本项目Git发布commit的独立archive；局部修复不能覆盖先前root或STOP。
本轮机制实验冻结600次实验repeat0 VLM文字，不调用API、不读取key、不新增费用。
六诊断状态repeat0身份已正确；ObjectII02正常对照repeat0误答yellow rectangular book。

## 待完成（不要把提交作业说成已完成研究）

1. 检查GPU冒烟所有行/视频/首输入、原生差异、受控首chunk，核查参考是否有独立安全完成。
   源码日志policy.jsonl保存原生首chunk和重复推理数值差异，不删不一致重复。
   同输入与同JAX RNG执行原生infer后用首次输出作受控common randomness，明确报告此诊断控制。
2. 若接口或规划器报错，作为InfrastructureError停根，不记未知/失败科学标签；修复后新根。
   原AEGIS QP无解导致undefined a的已知上游行为单独记录method exit。
3. 发布clean测试commit后归档新initial根，继承通过且preflight/geometry/protocol/serve四文件
   byte-identical的CPU预检（保存PREFLIGHT_INHERITED源hash、原job/commit及报告hash），或重做CPU预检。
   通过脚本quest_sync.sh exec调用该归档launch.py initial <根>。新initial执行304次，8h单A100。
4. 读INITIAL_COMPLETE/BRANCH_GATE，仅为“独立新验证有安全见证+identity_geometry仍安全未完成”
   的诊断状态逐一执行launch.py branch_<状态> <同根>，6h单A100，顺序提交。
   如果资格集合为空，报告不知道可行性；没有分叉不等于候选都失败。
5. 分叉采用同原始raw AEGIS前缀重执行，不恢复qpos；每个分叉核验sim state/qpos/qvel/ctrl/
   warmstart、controller、marker、action deque、JAX/Python/NumPy RNG、AEGIS内部、观察完整一致。
   检查点0/50/150/250未碰撞且未成功；固定raw预算对照/参考/nominal/release5/lift_then_nominal。
   比较原300步总预算与每检查点300步suffix。持久policy_cache支持跨Slurm阶段相同受控输出。
6. 本轮成果生成report/REPORT.md、runs.csv、statistics.json、perception_conditions.png、
   safety_traces.png、execution_examples.jpg、符合资格才有continuation_opportunities.png、HTML。
   活跃GPU时自动审批禁止在Quest checkout创建结果symlink，已实际拒绝两次，不能再试。
   使用只读SSH读取确切report文件并保存本地results副本；GPU终止后才可结果链接/pull-result。
   只读取回已通过自动审批，PNG来源/副本SHA256相等；不修改任何Quest源文件或目录。
   检查图/表/视频并分析四路判断。
   独立参考10次新seed验证仍属确定性sim同状态重复，不能认证高成功概率或真实不可行。
7. 仅当研究和最终图文表交付全部完成时mark goal complete；GPU排队不应标完成。

任何Quest操作先读QUEST_WORKFLOW、setup/README并quest_sync.sh check。冻结根源码来自已发布
codex/feasibility。不得在login运行模型/仿真/绘图；轻量Git、结果metadata及Slurm查询可以。

## 既有证据已交付（仍不是本轮新结果）

8170378输出report已生成并只读取回本地results/feasibility/prior_20261001/，其中
PRIOR_REPORT.md与prior_outcomes.png；PNG SHA256两边一致为
62bdcde2823458e399be7ece46ab83b0137d2ebea37d22ac405625b123e2bfcf。
取回新图表可通过quest_sync.sh exec base64 <明确run/report文件>仅远程读，再本地解码，
无需Quest checkout结果软链接；不要将这类只读输出复制误称源目录同步。
截至本轮末8170183仍PENDING，Slurm预计15:44:50 Chicago；未产生新科学episode。
Goal保持active，完整initial/条件branch与新图文结论均未完成，不mark complete。

## 实际发现与新作业 — 2026-10-01 后续

- **8170183已终止**：Slurm FAILED/1:0，6m05s，qgpu2006。仅spatial03/nominal完成，
  新feasibility-v1 seed948064852、300步、碰撞action60、最大L1=.1252346m、未完成；raw在
  第一策略输入之前失败。STOP为First policy observation differs across conditions，不能当raw科学失败。
  原根所有文件保留；raw第一输入没有被旧程序保存，但checkpoint与检测PNG均存在。
- CPU探针8179539完成；源b1e00d5，根
  `/projects/p33100/siosio/crashbench_safelibero/feasibility/20261001T193750Z_observation_audit_b1e00d5c3cf1/`。
  记录的sim/qpos/qvel/ctrl/warmstart/controller/队列/RNG/marker一致，仅agent/back RGB不同。
  raw检测PNG的感知前图像与raw checkpoint感知后图像逐字节一致，排除本次感知阶段图像修改。
  缩放224后agentview仅1像素/3通道差1灰度级，mean absolute=1.9929847e-5。
  这定位本轮失配，不能宣称解释旧600行的11对全部来源。
- 旧加强CPU预检8179541通过八状态各三个新环境，完整物理/控制器/力状态指纹相同。
- 新预检**8185435 COMPLETED/0:0，2m12s**，源码a7fd556ac7fbf9d6a5754a168fb3e98d21371cc6。
  验证RGB模板同状态复用/异状态分离/原生变体保存、模型渲染签名属性、目标抓持检查接口、
  8×3 fresh no-render环境物理指纹与几何包含性/普通动作范围。
- **新GPU冒烟8186026**，16次，当前PENDING，最后squeue估计17:45:41 Chicago；不是完成时间。
  新根`/projects/p33100/siosio/crashbench_safelibero/feasibility/20261001T204139Z_rgb_control_a7fd556ac7fb/`。
  读plan/progress/STOP/slurm_8186026.log/SMOKE_COMPLETE判断实际执行；其source archive不能修改。
  旧live Quest main/d220358保持clean。

### 协议修正（最终按实际源码执行）

- 阶段三前缀改为**identity_geometry感知受控AEGIS**，候选基线名aegis。问题B与分叉共享
  同一正确身份/已验证仿真几何，不再换回raw几何。原始raw仍作为问题B的单独条件。
- 初始和每个策略查询/检查点增加明确RGB控制：只在完整sim/力/相机/渲染模型/非RGB观测
  精确签名相同时共享第一次真实RGB采样。原生哈希/差异/变体NPZ留存；不同状态不复用。
  这是机制诊断的共同观测控制，不是部署系统性能，也不通过删重复解决失配。
- 所有env返回的观测立即复制为自有数组；感知前后观测哈希再次核验。全预测chunk与实际
  执行前5动作单独分析；相同输入的原生infer仍执行并保留与控制输出的差异。
- 报告分开配对身份/几何效果；同300步suffix早晚比较只用同种子、后期未碰撞的共同前缀。
  特权参考与普通有限干预候选分开计。只有特权参考成功不能支持学习普通候选选择器。
- 固定参考补偿夹持物体的XY偏移；从中途已抓住目标时保留夹爪/姿态并继续运输；未抓持时
  先保持当前姿态提高，再在安全高位转到抓取姿态。改动在任何参考新成功结果产生前冻结。
- 每run最终manifest现在保留完整configuration；第一输入不匹配时也保存输入；参考命令限
  原native action_spec，并保存reference_actions.npy。视频完整解码/shape/帧数检查恢复。

本地取回只读证据在`results/feasibility/observation_audit_20261001/`：OBSERVATION_PROBE.json及
initial_rgb_difference.png；PNG两边SHA256为
d641a740242083246cb0d6797680b484203b474369c88d6c1bdae6fc81169b84。

后续仍必须完成8186026实际模型/视频审查、304次initial、符合资格的感知受控同前缀分叉及
最终新图文表/路线判断。7项新协议分析测试、22复现、27配对测试通过（新数据尚未完成）。
Goal保持active；不能把本次观察失配诊断或提交作业当作整体实验完成。

## 后续完整阶段已准备（仍未提交）

发布0ec56db9f869a1eadc487f7ebe271caf96b4ce1e增加完整分叉schedule/coverage验证，已到达却缺失
的checkpoint会报错；未到达/已碰撞/基线已完成的跳过均列原因。COMPLETE在report导出后写，
report/REPORT_COMPLETE.json记录每个图表/CSV/JSON的SHA256。8项新协议测试通过。

准备根：`/projects/p33100/siosio/crashbench_safelibero/feasibility/20261001T211455Z_initial_ready_0ec56db9f869/`。
INITIAL_READY.status=prepared_not_submitted；继承8185435 preflight，四文件byte-identical及报告hash
已核验，source所有文件逐字节等于Git发布提交。**尚无initial Slurm作业**。
必须先核验8186026终止成功、16条及输入/物理配对门、视频正常，再通过该root/source/launch.py
initial提交304次。冻结8186026的a7fd556 archive不变；Quest main/d220358 clean。
本轮最后8186026在squeue可见且PENDING/Resources，属于对已确认live句柄的verified wait，
不是终止或失联；不因调度时间推迟取消/重启，不mark goal complete。
