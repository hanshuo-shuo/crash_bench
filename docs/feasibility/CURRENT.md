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
   将明确results路径symlink到对应output/report再quest_sync pull-result；检查图/表/视频并分析四路判断。
   独立参考10次新seed验证仍属确定性sim同状态重复，不能认证高成功概率或真实不可行。
7. 仅当研究和最终图文表交付全部完成时mark goal complete；GPU排队不应标完成。

任何Quest操作先读QUEST_WORKFLOW、setup/README并quest_sync.sh check。冻结根源码来自已发布
codex/feasibility。不得在login运行模型/仿真/绘图；轻量Git、结果metadata及Slurm查询可以。
