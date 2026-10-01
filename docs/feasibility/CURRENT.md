# 安全未完成状态的可行性见证与受控续接实验

用户于2026-10-01明确授权本诊断，全部仿真、推理、分析与绘图在Quest执行；无训练、阈值搜索或新任务。六个诊断状态为Spatial I/task1的3、9、15与Object I/task2的0、2、5；正常对照为Spatial I/task1的1与Object II/task1的2。不是总体评估或概率认证。

## 最新作业（必须读取实际receipt/status）

- 完整initial **8189966**已提交：304次，单A100，8h上限。源提交
  `327205258fd48e531b04441e62f90a045fc2fd4b`，根
  `/projects/p33100/siosio/crashbench_safelibero/feasibility/20261001T215021Z_initial_327205258fd4/`。
  查看plan.json、progress.json、STOP.json、INITIAL_COMPLETE.json、BRANCH_GATE.json及Slurm记录。
  不得修改source archive；Quest live main保持`d22035816647321b8c9c44674efb76273916db95`且clean。
- 同根CPU预检 **8189615 COMPLETED/0:0**，2m52s：八状态各三个fresh物理环境、动作范围、碰撞几何包含性、共同RGB控制、服务器控制输出/原生差异保留通过。
- 同根CPU合法命令回放 **8189617 COMPLETED/0:0**，2m10s：四条轨迹的最大障碍位移和任务结果逐条复现。Spatial3/rim参考和identity_geometry都记录到robot0_link5与酒瓶接触。Object0两条在动作端点没有记录到机器人/目标接触；不能据此排除物理子步接触。
- 旧准备根`20261001T211455Z_initial_ready_0ec56db9f869`未提交且已被新根替代，不能启动。

已提交CPU依赖衔接 **8192029**，afterany:8189966。控制器源7a1e69eec4b8，独立根
`/projects/p33100/siosio/crashbench_safelibero/feasibility/20261001T221249Z_controller_7a1e69eec4b8/`。
先检查前置Slurm COMPLETED/0:0、INITIAL_COMPLETE、BRANCH_GATE与STOP，再逐一调用完整根原327提交的归档launcher；每个branch结束后依赖触发下一个。没有资格就不提交；缺失证据、失败、重复尝试即停。
读取控制器SUBMITTED.json、receipts/*.json、CONTROLLER_STOP_*.json与controller日志确定新句柄。
控制器不执行仿真/分析，仅短CPU任务提交；运行中的initial与后续branch源码完全不改。

## 已验证烟测

烟测 **8186026 COMPLETED/0:0**，15m31s，qgpu2005，16/16行、全部视频解码/帧数与初始物理/观察/首chunk配对门通过。源a7fd556ac7fb，根
`/projects/p33100/siosio/crashbench_safelibero/feasibility/20261001T204139Z_rgb_control_a7fd556ac7fb/`。

|状态|raw/identity|geometry/identity_geometry|独立固定参考筛查|
|---|---|---|---|
|Spatial3|安全未完成|不安全未完成，最大L1 87.948mm|center/side安全未完成；rim完成但最大L1 1.067mm，不算安全见证|
|Object0|安全未完成|不安全未完成，最大L1 1.373mm|center/rim安全完成227步，side安全完成230步|

两个状态原策略均不安全完成。Object0合法安全见证已找到，尚须完整initial的新重复验证；Spatial3不能判为不可行。原生同输入infer在该烟测内没有非零输出差异；17条原生RGB变体保留。视频末帧已实际查看，不放松1mm官方代理边界。

## 执行可比性与保留的失败

旧烟测8170183 FAILED/1:0，仅nominal Spatial3完成；raw在首动作前因首观察不一致停止，是基础设施失败，不能记成AEGIS科学失败。CPU观察探针8179539定位224图仅一像素/三通道差一个灰度级；感知前后图像相同，完整已记录物理、控制器、队列、RNG相同。这不能解释历史600行全部11个首chunk警示。

机制实验只对完整物理、力、相机、渲染模型和非RGB观测签名相同的状态共享首次真实RGB采样；保留原生差异、NPZ与统计。每次native infer仍执行并推进原RNG，但同输入/RNG执行共同首次输出。初始物理/观察/首chunk和中途整个sim/controller/deque/RNG/AEGIS状态都必须精确配对；不删除不一致重复。

参考规划可读取特权状态，但仅通过原7D OSC/gripper执行，平移.05m、旋转.5rad/action，无额外能力、障碍/对象重置或直接qpos写入。几何条件用碰撞网格顶点与保守基本形状拟合且验证同一AEGIS椭球包含性，不是完整机器人安全模型。身份和几何分开配对。

本轮冻结历史repeat0 GLM文字，0新增API调用，不读取key。六诊断身份均已正确；ObjectII正常对照原回答误认yellow rectangular book。

## 后续必须完成

1. 核验initial304次完整、视频及全部配对门；只读BRANCH_GATE确定资格，不能凭本文推断。
2. 资格为独立新验证有安全见证且identity_geometry至少一次未安全完成（包括不安全完成）。只有这类诊断状态执行同根归档launch.py branch_<状态>，顺序单GPU；不覆盖源码。
3. 使用identity_geometry前缀的未碰撞检查点0/27/77/152/252。27等非5倍数检验非空动作缓存。未到达/已碰撞/已安全完成明确列跳过原因，不当失败。已到达却缺checkpoint停根。
4. 比较感知受控AEGIS预算对照、固定特权参考、原策略、释放过滤5步、提升10步后原策略。有限普通候选与特权参考分开分析。原总300步与每检查点300步suffix分开；早晚只比较共同幸存重复，不能仅由预算解释。
5. export并实际查看REPORT.md、CSV、JSON、感知图、安全曲线、执行图与符合资格的续接图；对四路线作具体判断。重复只是固定状态/确定性物理的新策略种子，不由10/10或4/5认证高概率，更不能由参考失败推出真实不可行。
6. 全部实际研究与图文表交付完成后才mark goal complete；排队与运行不是完成，也不是阻塞。

## Quest边界与本地只读证据

每次Quest操作先按QUEST_WORKFLOW.md、setup/README.md并quest_sync.sh check核验。唯一checkout为$HOME/crash_bench，源提交必须clean、tested、published后归档提交。GPU p33100/gengpu，CPU p33100/short；登录节点只做轻量检查/Git，不能仿真/推理/绘图。

GPU活跃时自动审批两次拒绝在Quest checkout建立results symlink；不能重试。可只读SSH/base64明确结果路径保存本地副本并比SHA，不变更Quest checkout或源档案。
已取回旧证据`results/feasibility/prior_20261001/`与像素探针`results/feasibility/observation_audit_20261001/`，都不是本轮完整新结果。全部失败根、旧源码与历史raw结果保留可恢复。

## 子步接触与安全语义审计（2026-10-01后续）

- 主initial8189966最后实际核验RUNNING/qgpu2004，已73/304；依赖控制器8192029 PENDING/Dependency。不是完成。
- 子步回放**8193808 COMPLETED/0:0，2m20s**，根
  `20261001T222534Z_substep_contacts_5966839c2245`。三Spatial/rim及Spatial9/15联合几何AEGIS。
  qpos/qvel/ctrl在0与全部已存检查点逐字节匹配，最大位移和任务结果精确复现。
  Spatial9/15 rim虽然官方代理安全，仍有link5酒瓶接触，分别在4、5个动作内出现。
  Spatial3 rim的1.067mm不安全完成也有5个接触动作。不能声称这些是无受保护接触的见证。
  原596审计把桌面支撑标为other_dynamic；原始JSON保留，报告只解释robot/target。
  后续afdebc改按祖先DOF排除静态支撑，不改机器人/目标计数，不重写旧结果。
- 单一固定合法放置补试**8194623 COMPLETED/0:0，2m19s**，根
  `20261001T223433Z_south25_afdebc06fb29`。只改本地规划waypoint负Y25mm，未改真实对象/任务。
  三Spatial都未完成，且有link5酒瓶接触；没有再搜索偏移或评分阈值。失败不能证不可行。
- Object接触回放**8196196 COMPLETED/0:0，2m07s**，根
  `20261001T224312Z_object_contacts_afdebc06fb29`。Object0/2/5 center参考全部完成，所有子步无机器人/目标-酒瓶接触，仍是单次筛查，新重复验证在主initial中。
  Object0联合几何AEGIS在动作263的子步中有link5接触，动作端点旧回放漏检；位移1.373mm。
  支撑酒瓶的box_small_base接触不算机器人/目标接触。每个保存检查点物理数组完全复现。
- 这些接触与末端单椭球代理覆盖不足的机制一致，但没有单独排除离散控制、跟踪误差等因素。
  官方1mm评分保持不变；需要区分代理下安全与无受保护机器人/目标接触。Spat9/15主分叉资格仍按预先冻结官方代理协议，不声称更强安全可行性或真实可行性衰退。

最终增强分析准备根：
`/projects/p33100/siosio/crashbench_safelibero/feasibility/20261001T225559Z_analysis_865399bbbbb5/`。
源865399bbbbb5，ANALYSIS_READY=prepared_not_submitted，输入supplements.json包含两个完整接触审计和south25失败筛查。
analysis/protocol/runtime/reference/serve/adapter/geometry/observation_control/execute与主327源码逐字节一致。
只在主根COMPLETE且所有批准分叉完整、Slurm成功终止后，提交该archive的interpret.sbatch：CB_ANALYSIS_ROOT为分析根、CB_ANALYSIS_TARGET为主根；生成四类结果图、oracle库/最佳固定候选/预算图、接触见证图及中文判断证据表。
旧`20261001T221937Z_analysis_5324285a96aa`未提交，已标SUPERSEDED保留原source，不使用。
15项机制/分析/衔接测试通过；新图表必须在Quest生成并实际QA后取回。

SSH master曾短暂断开，旧命令最终返回Slurm/审计结果，随后socket消失；通过非交互BatchMode重新建立同一/tmp/quest.sock，quest_sync check通过。不重启任何作业、不删结果，不需用户密码或Duo。
Goal保持active：完整initial、批准分叉、最终图文表与研究建议仍未完成。
