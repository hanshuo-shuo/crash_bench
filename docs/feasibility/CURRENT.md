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
`/projects/p33100/siosio/crashbench_safelibero/feasibility/20261001T230540Z_analysis_c53449c82618/`。
源c53449c82618，ANALYSIS_READY=prepared_not_submitted，输入supplements.json包含两个完整接触审计和south25失败筛查。
analysis/protocol/runtime/reference/serve/adapter/geometry/observation_control/execute与主327源码逐字节一致。
只在主根COMPLETE且所有批准分叉完整、Slurm成功终止后，提交该archive的interpret.sbatch：CB_ANALYSIS_ROOT为分析根、CB_ANALYSIS_TARGET为主根；生成四类结果图、oracle库/最佳固定候选/预算图、接触见证图及中文判断证据表。
旧`20261001T221937Z_analysis_5324285a96aa`未提交，已标SUPERSEDED保留原source，不使用。
旧865分析根亦未提交并已SUPERSEDED保留；接触图重排布局，实际数据不变。15项机制/分析/衔接测试通过；新图表必须在Quest生成并实际QA后取回。

SSH master曾短暂断开，旧命令最终返回Slurm/审计结果，随后socket消失；通过非交互BatchMode重新建立同一/tmp/quest.sock，quest_sync check通过。不重启任何作业、不删结果，不需用户密码或Duo。
Goal保持active：完整initial、批准分叉、最终图文表与研究建议仍未完成。

阶段性接触图报告：8197299 COMPLETED/0:0/22s，旧图轴标签重叠不交付；已提交修正布局8197680，根20261001T230540Z_contact_report_c53449c82618。读取实际Slurm/REPORT_COMPLETE后，查看PNG并只读取回本地，不能将阶段性报告当最终研究完成。

8197680实际COMPLETED/0:0，18s；修正接触图已实际查看，标签清楚。图/中文报告/CSV/JSON及manifest仅只读取回results/feasibility/contact_screen_20261001，四个导出文件SHA256逐一匹配；PNG为22c6d67b7f1fadad7340312e66f6bc36069f186d0e604f1bc32a3198db4441f5。该报告只含单次筛查，不是最终目标完成。最终分析准备根现为20261001T230540Z_analysis_c53449c82618；旧865未提交且SUPERSEDED保留。

## 分叉前实际检查点覆盖（仍需等待新参考验证）

主initial的六诊断状态各五个策略种子的五条件已完成；六诊断全部条件的安全完成均为0/5。身份为no-op；几何修正带来部分不安全完成，没有恢复安全完成。两正常对照的条件已在前224次中完成；参考十次新验证仍在继续，不把这部分描述成全304已完成。
主根PRE_BRANCH_CHECKPOINT_AUDIT.json与PRE_BRANCH_CHECKPOINT_AUDIT_OBJECT.json记录实际前缀覆盖。
Spatial9/15潜在各125个分叉案例，只存在未触发官方碰撞代理的0/27/77步；152/252有的未到达，有的到达时已经碰撞，不执行、不计失败。
Object0潜在205个案例：repeat1的152已碰撞、252未到达；Object2/5各225个案例且五个检查点完整。早晚比较Object0晚期用四个共同幸存重复，不把缺少的一次当失败。
六诊断的所有30个corrected基线均未恰好在计划检查点终止，原schedule的终止边界不影响这批数据；实际branch_schedule校验通过，没有“已到达却缺失”的检查点。
这些是覆盖核验，未构成分叉资格授权；必须等INITIAL_COMPLETE和BRANCH_GATE。按当前可复现筛查预期五资格状态，但不得以预期替代实际门。独立参考十次验证未完成前不自行提交。

## 最终预算解释的额外核验

源8b0b7d5ed879新增trace_audit.py；18项机制/分析/审计测试通过。最终分析必须核对每对原剩余预算/300步续接预算在共同部分的全部记录字段相同；未干预AEGIS也须与原corrected轨迹共同部分相同，不能提前停止。违背即不导出预算解释，不删除不一致重复。
新增matched_time_budget_changes.png只用同一幸存重复比较早/晚，另列额外预算增益/损失；原时间曲线保留为条件描述图。
最新准备分析根为`/projects/p33100/siosio/crashbench_safelibero/feasibility/20261002T010021Z_analysis_8b0b7d5ed879/`，ANALYSIS_READY=prepared_not_submitted；输入supplements同前。9项实验模块与主327字节相同，仅最终后处理增加审计与图。旧c534分析根保留SUPERSEDED，不提交旧版。
只等全部批准branch完成、主根COMPLETE/全部实际Slurm成功，再提交此analysis archive的interpret.sbatch；全部新图表必须实际查看并校验副本SHA。
参考验证阶段有完整动作NPY；已核验Spat3十次0/10、Spat9十次10/10，各自只有一个动作tape hash，因此是确定性同状态的fresh执行重复，不是十个独立物理扰动或高概率认证。Spatial9/15无受保护接触安全仍未知，官方代理见证的接触问题不能被验证次数遮盖。

## Initial完整与首个实际分叉（2026-10-01 Chicago / 10-02 UTC）

Initial8189966实际COMPLETED/0:0，3h18m09s。304/304、120个配对组、视频/首观察/首chunk/完整初始执行状态门通过，原生同输入/RNG推理差异0，原生RGB变体206条保留。正式BRANCH_GATE为Spatial9/15与Object0/2/5；Spat3固定center新验证0/10，其他七状态选定参考10/10，仍是确定性重复不是概率认证。
已保存主根initial_report_snapshot，7个文件逐一SHA验证；只读SCP取回本地results/feasibility/initial_304_20261001，7文件SHA一致，三个图已实际查看。整图上的safe字样都是官方L1代理，Spatial9/15参考仍有实际link5接触，须同时看contact_screen报告，不可解释成更强安全。
旧控制器8192029 FAILED/1:0/6s，计算节点没有git，失败在launcher检查前，未提交任何分叉；主根无STOP，304结果完好。日志与旧source保留。
修复仅控制器SBATCH加载git/2.37.2，源f9d43272d624，独立根
`/projects/p33100/siosio/crashbench_safelibero/feasibility/20261002T012402Z_controller_f9d43272d624/`。
控制器8211864实际COMPLETED/0:0/21s，提交**branch_spatial_09 GPU8211895**及继任CPU**8211896**（afterany:8211895），receipt8211864.json完整。新控制器根替代旧root，别等旧8192029。
8211895已实际RUNNING/qgpu0403，第一t0/aegis控制与t0/lift_then_nominal已完成且前缀核验通过；源仍为主327冻结archive。Spatial9 schedule实际125，缺失/碰撞10条分别记录理由，不能把它们计作候选失败。
首个未干预AEGIS全轨迹复现审计FIRST_UNCHANGED_AEGIS_CONTROL_AUDIT.json在主根；不是只验证初始哈希。后续仍须全部5批准状态分叉、全部共同预算轨迹审计与最终8b分析图文表，再作研究建议与mark goal complete。

Spatial9首个重复的0/27步完整候选预算对比已形成；主根EARLY_COMMON_TRAJECTORY_AUDIT.json核验5个两预算共同轨迹对、3个未干预AEGIS控制，全记录字段完全一致。仅这批局部核验，不代替全905个计划分叉或最终预算审计。初始REPORT七文件已只读SCP取回本地并逐一SHA匹配；不用在Quest checkout建结果symlink，直接读输出SCP已通过自动审批。

## Spatial9分叉完整与后继实际提交（2026-10-02 UTC）

8211895实际COMPLETED/0:0，1h09m48s；主根branch_spatial_09_COMPLETE.json与rows.json确认125个分叉，累计429，STOP不存在。五个重复中，reference在t0/t27均官方代理安全完成5/5，在t77为4/5；原剩余预算与300步续接预算计数一致。普通候选只有lift_then_nominal在t77为2/5，两预算相同；其余普通候选及未干预AEGIS在全部可用检查点均0/5。2/5不足以称稳定有效或高概率认证；完整共同轨迹审计及逐条物理接触审计尚未完成，不能由官方代理结果推出更强无接触安全。

继任控制器8211896实际COMPLETED/0:0，12s，receipt确认提交branch_spatial_15 GPU **8217077**及其afterany控制器 **8217078**。最新实际队列8217077 PENDING/Priority，8217078 PENDING/Dependency。不要等待已完成的Spatial9句柄，也不要以静态文档猜后继进度。仍须Spatial15与Object0/2/5、完整预算轨迹审计、最新8b分析图文表与最终研究判断。Goal仍active；不改变主327科学源或Quest live checkout。

## 完整前缀接触核验与Spatial9续接安全语义（2026-10-02 UTC）

8217077随后实际RUNNING/qgpu0402，最新已439条；8217078仍Dependency。额外三项只读动作回放CPU均实际COMPLETED/0:0，源8b0b7d5ed879，所有qpos/qvel/ctrl保存检查点精确、位移和任务结果复现：
- 8217537，4m44s，`20261002T024159Z_spatial09_fork_contacts_8b0b7d5ed879`：Spatial9全部五corrected前缀、t77五参考及两个代理安全lift_then_nominal。
- 8217703，4m33s，`20261002T024412Z_prefix_contacts_spatial15_object00`：Spatial15/Object0全部十corrected前缀。
- 8217706，4m39s，`20261002T024414Z_prefix_contacts_object02_object05`：Object2/5全部十corrected前缀。

全部25条corrected基线现有完整子步机器人／目标-障碍接触证据。Spatial9首接触动作各为76、77、116、73、116，因此t77五个官方代理可分叉中只有repeat2/4此前无上述接触。两个lift代理成功repeat1/3早在前缀中有目标-障碍接触，整段都不是无接触安全恢复；不能凭2/5推进候选选择。t77 reference的repeat2是完整合法前缀+特权参考后缀无上述接触完成，动作回放精确复现；repeat4未完成。它是具体复合轨迹见证，不是从初始独立rim参考的高概率认证。

Spatial15五条首接触109..114，0/27/77此前均无上述接触。Object0 repeat1首接触140（152已被官方代理门排除），repeat3首接触249但252官方代理仍合格，因此252四个代理可分叉只有三个无先前接触。Object2五条整段无上述接触。Object5 repeat1首接触204，252代理五个可分叉只有四个无先前接触。

全部原矩阵和原分叉保留；最终补列严格前缀资格和未知续接接触数，不删除重复来声称观察/RNG混杂已排除。contact_conditions.py与新增四项语义测试，连同既有18项共22项通过，新增严格前缀CSV/图及续接接触计数。主实验九模块与327保持不变；仅后处理更新。旧8b分析根尚未提交，需使用后续新发布分析archive并加入这三完整contact roots；不能再提交缺少严格前缀区分的旧8b最终分析。所有新数值分析与绘图仍在Quest CPU作业执行。

最新分析准备根已实际创建：`/projects/p33100/siosio/crashbench_safelibero/feasibility/20261002T025810Z_analysis_36e1f2dca1e8`，源36e1f2dca1e834f425296be1f30a62b4da3f2b4c，22项测试在Quest轻量预检再通过，九科学模块逐字节等于主327。supplements已加入上述三完整接触根，SHA256=e5cee5c75d04146b49c886a30e3bc8b675e1c873216cc2eadd7f171c1f92b952。ANALYSIS_READY仍prepared_not_submitted；旧8b准备根已SUPERSEDED，原source/input保留。
末次实际8217077 RUNNING/qgpu0402/20m55s，累计459条；8217078 PENDING/Dependency。最终在所有批准分叉完成且Slurm成功后，优先核验无先前接触前缀中的代理安全续接（CPU保存动作回放，无新增推理/策略/API），再把完整接触证据加入准备分析输入并提交interpret.sbatch。未知续接接触不能代替研究决定所需的稳定无接触续接证据。Goal仍active，完整Object三状态、全部预算轨迹审计、实际新图QA、最终判断表未完成。

## 初始参考80次新验证的完整接触回放（2026-10-02 UTC）

已用源36e1f2dca1e8提交全部80条保存动作回放，两个CPU串行链（每个任务2核、4GB、10分钟），无GPU/推理/API。每个根有SOURCE_COMMIT、source.sha256、cases.json、SUBMITTED.json；同一已完成原row的SHA固定，不改变主327实验源码。

|状态|新接触根（共同feasibility目录）|作业|afterany前驱|
|---|---|---:|---:|
|Spatial3|20261002T030622Z_initial_ref_contacts_spatial_03|8221116|无|
|Spatial9|20261002T030623Z_initial_ref_contacts_spatial_09|8221117|无|
|Spatial15|20261002T030625Z_initial_ref_contacts_spatial_15|8221192|8221116|
|Object0|20261002T030628Z_initial_ref_contacts_object_00|8221197|8221117|
|Object2|20261002T030630Z_initial_ref_contacts_object_02|8221198|8221192|
|Object5|20261002T030633Z_initial_ref_contacts_object_05|8221203|8221197|
|对照Spatial1|20261002T030636Z_initial_ref_contacts_control_spatial_01|8221207|8221198|
|对照ObjectII2|20261002T030638Z_initial_ref_contacts_control_objectII_02|8221208|8221203|

末次8221117已COMPLETED/0:0/3m38s，其余只按sacct真实状态判断；不把afterany队列当完成。8221116 RUNNING/5m17s，8221197 RUNNING/1m31s。主8217077 RUNNING/qgpu0402/34m25s，累计478，继任8217078仍Dependency。
分析增加initial_reference_contacts逐条要求全部80审计，不由单条筛查外推；官方代理与同时无受保护机器人／目标接触的完成计数分开，确定性重复不认证概率。新增初始接触缺失/不一致门测试，共23项本地通过；仍不更改九个实验核心模块。最终报告补reference_contact_validation.csv/png并嵌入结果、预算及接触图。36e准备分析根尚未提交，将由下一新发布后处理archive替换；最终必须加入本表八根并验证全部COMPLETED/接触receipt，不能提交缺少80逐条核验门的旧分析源码。

## Spatial15完整、Object0实际启动与统一初始化回放

8217077实际COMPLETED/0:0，1h21m；Spatial15全部125分叉完成，主累计554。0/27/77每点的参考均官方代理安全完成5/5；普通三候选与未干预AEGIS全部0/5，两预算计数相同，全部方法退出0。不能将特权参考成功当作普通库选择学习证据。继任8217078实际COMPLETED/0:0/19s，receipt确认branch_object_00 GPU **8226877**及CPU继任 **8226880**。末次8226877 RUNNING/qgpu0403/2m41s，8226880 PENDING/Dependency；主无STOP。

旧初始参考接触链七状态COMPLETED，六诊断计数：Spatial3无完成；Spatial9/15各代理安全10/10但全部有机器人或目标与障碍接触；Object0/2/5各10/10同时通过代理和无上述接触。它们仍是固定初始状态确定性参考重复，不能认证概率。
ObjectII对照旧接触job8221208 FAILED/1:0/42s，初始化qpos不相同，未生成安全失败标签；诊断8223887 FAILED/1:0/36s保留PHYSICS_MISMATCH.json，最大差1.2281842209915794e-15。原robosuite离屏renderer构造在base._reset_internal中额外做一次sim.forward，旧headless遗漏。源6cd022f87faa394d3859d9c1cf0ce0801fc11a74增加可选reset_forward.py，复现原调用位置，无积分/状态写入或门放宽；25项测试通过，九科学模块仍与327字节相同。
单例8225458 COMPLETED/0:0/51s，根20261002T034402Z_controlII_reset_forward_6cd022f87faa：0/27/77/152的qpos/qvel/ctrl逐字节复现，完成且无机器人／目标-障碍接触。初始化forward计数2（构造和硬reset各一次）。

因此统一重核验全部80参考+25原前缀+7个Spatial9续接，共112案例，两个CPU串行链。实际batch receipt：
`/projects/p33100/siosio/crashbench_safelibero/feasibility/20261002T035124Z_renderer_forward_batch_6cd022f87faa/SUBMITTED.json`。
11作业：8226198、8226200、8226202、8226245、8226248、8226251、8226252、8226254、8226255、8226257、8226258。每根独立SOURCE_COMMIT/source.sha256/cases/SUBMITTED，replay_renderer_reset_forward=true，旧70完成回放和全部失败记录保留，不只重跑或删除不一致重复。
实际8226198 COMPLETED/0:0/3m26s，ControlII十次全部严格检查点复现、代理安全完成且无上述接触；8226200 COMPLETED/0:0/5m13s，Spatial9十二案例全部严格复现，无上述接触完成仍只有t77/repeat2参考一例。其他统一job按真实Slurm/receipt判断，尚未全部完成。

最新最终分析准备根 **20261002T031841Z_analysis_0ca3e13f0c32**，源0ca3e13f0c32d2f7acfef3fff84e18cd1bac5cf7（Quest23项通过，九科学模块等于327）。36e准备根已SUPERSEDED。0ca准备根仍未提交，输入已用11个统一新根替换旧112接触输入；旧supplements原文保存在supplements_before_renderer_forward.json，变更映射保存在SUPPLEMENT_UPDATE_renderer_forward.json，新SHA7493ee2ba67ec753fbbb726941239fd365bc82fbda7df14eafb3c2af30a8c646。其SOURCE不变，不另归档分析源。
仍须等待全部三Object分叉；全部共同预算/未干预轨迹审计；11个统一接触job成功与全部112receipt；无先前接触前缀中代理安全候选／参考的保存动作接触回放（用源6cd、reset_forward=true，新根）。把新续接contact roots加入未提交分析输入，保留前版manifest并更新SHA。最后提交0ca的interpret.sbatch、实际查看全部新图、只读取回并验SHA、写具体判断表。未知接触不能充当稳定恢复证据，不能凭现有半程结果mark goal complete。Goal仍active，无需用户输入。

## 统一112回放完整与Spatial剩余成功续接核验

统一11个作业已实际全部COMPLETED/0:0，112/112接触receipt和原保存检查点qpos/qvel/ctrl逐字节核验、任务/位移复现全部通过；每例renderer_reset_forward_replayed=true。batch/AUDIT_COMPLETE.json存全部实际Slurm状态与11个CONTACT_SUBSTEPS.json SHA256。不是只查看SUBMITTED或以missing句柄推断完成。
末次主Object0 GPU8226877实际RUNNING/qgpu0403/16m23s，累计573；8226880 PENDING/Dependency，主无STOP。后续只按最新Slurm与progress/receipt判断，不以此静态记录猜进度。

已根据完整两个Spatial分叉和统一25条前缀接触记录，提交所有剩余25条“无先前机器人／目标-障碍接触前缀、官方代理安全完成、同300步续接预算”的Spatial参考／普通候选轨迹回放。此前已精确审计的t77/repeat2参考不重复计数；此前已接触的分叉保留原结果、不进入强安全恢复子集。实际25条全是reference，因为所有合格普通候选没有代理安全成功。
batch：`/projects/p33100/siosio/crashbench_safelibero/feasibility/20261002T043023Z_spatial_suffix_contacts_6cd022f87faa`。
三个源6cd/reset_forward=true独立根：20261002T043033Z_spatial_suffix_contacts_chunk0 **8229645**（10）；20261002T043034Z_spatial_suffix_contacts_chunk1 **8229646**（10）；20261002T043037Z_spatial_suffix_contacts_chunk2 **8229649**（5，afterany8229645）。两个CPU链，主GPU与其源码不变。以上仅SUBMITTED，仍须读实际Slurm/完整接触receipt。
完成后把这三个新root加入0ca未提交分析supplements，保存之前manifest和变更SHA；Object三个状态完成后，同样覆盖每个合格且代理成功的同300步续接，未知接触不能当作成功或失败。预算/原AEGIS全轨迹审计与最终图文表仍未完成，Goal active。

## Spatial完整接触续接证据已并入

8229645/8229646/8229649实际全部COMPLETED/0:0，3m48s/3m23s/2m15s，25条均严格检查点复现且reset_forward=true；batch/AUDIT_COMPLETE.json保存实际状态与raw文件SHA。0ca准备分析supplements已加入三root，旧输入保留为supplements_before_spatial_suffix.json，SUPPLEMENT_UPDATE_spatial_suffix.json存映射和hash，新SHA **9739ca0c0e17fba28e17cdf86bc027fc85e20082924a41dafd955cae398e9641**。ANALYSIS_READY仍未提交。
完整同300步续接、此前无机器人／目标-障碍接触子集：Spatial9参考在0/27步分别0/5、0/5同时通过代理和无上述接触；77步为1/2（五个官方代理分叉只有两个此前无接触）。Spatial15参考在0/27步各0/5，77步为4/5；五个前缀此前均无上述接触。全部成功续接已审计，未知接触数0。所有合格普通候选的代理安全完成都是0，因此没有普通库强安全恢复见证。4/5只描述本诊断重复，不认证高成功概率。
从原初始状态出发的完整合法“受控AEGIS前缀+特权参考后缀”已形成无上述接触完成轨迹（Spatial9一例、Spatial15四例），并由独立保存动作回放复现；它们有别于初始直接固定rim参考（新验证10/10均有接触），不得混称为同一个初始参考方法的成功率。也不能由特权参考成功推出普通候选选择学习有效。Spatial证据没有显示无上述接触的早期参考机会随时间下降；未到达/已碰撞的更晚检查点不记失败。
统一112最新读取八状态的逐条参考接触计数：Spatial3代理及强接触条件均0/10；Spatial9/15代理各10/10、强条件各0/10；Object0/2/5及两个正常对照均同时通过代理与无上述接触10/10。仍是确定性固定状态重复，不是物理扰动或概率认证。
末次主Object0 8226877 RUNNING/qgpu0403/39m42s，累计599，继任8226880 Dependency。仍待Object三状态全部完成及其合格成功续接接触审计；最终全部共同轨迹/预算审计；图文表实际渲染QA和研究判断。Goal active。

## 接触条件下的配对预算分析补全

新增contact_qualified_view仅生成派生分析视图，原rows/评分不变；按同一无先前接触重复比较早晚同300步续接。预算短轨迹只有在已通过完整共同记录审计、完整动作数相同、完整初始执行指纹相同且长轨迹有接触核验时，才转移接触结果；不是新的独立试验。缺少任何合格代理成功（包括未干预AEGIS预算对照）的接触核验即停止最终导出。三项新语义测试后共28项通过；九科学模块仍逐字节等于327。
最终新增physical_time_budget.json/csv、contact_time_budget.png，分别列固定参考、普通候选库oracle与未干预AEGIS的预算效果；早晚参考/普通库仅同一幸存且此前无机器人／目标-障碍接触的重复。physical_libraries同时报告最佳单一候选与库存在性上界，不把privileged reference当普通候选。源0ca准备分析尚未提交，将由下一新发布后处理archive替换，必须保留其最新supplements（SHA9739ca0c0e17fba28e17cdf86bc027fc85e20082924a41dafd955cae398e9641）全部16个contact roots及alternative输入。
剩余Object成功续接接触回放须覆盖所有BRANCH_CONDITIONS，包括aegis（同300步续接预算），不限于reference/三个普通候选；仍只在此前无上述接触的前缀且代理安全完成中核验，原不合格行保留。原剩余预算成功可用上述完整同轨迹审计转移，否则需单独回放，不能放宽门。Goal仍active，不改变活跃Quest科学源。

最新最终分析准备根实际已创建：**20261002T050841Z_analysis_c35fe9e57764**，源c35fe9e5776498aeb8c27b8355ae5a0688151166。Quest28项轻量测试通过，九科学模块逐字节等于主327；完整16 contact roots和alternative输入从0ca复制，SHA仍9739ca0c0e17fba28e17cdf86bc027fc85e20082924a41dafd955cae398e9641。0ca未提交根已SUPERSEDED并保留原source/inputs；不要再提交旧0ca分析。C35 ANALYSIS_READY仍prepared_not_submitted。
末次实际主Object0 8226877 RUNNING/qgpu0403/1h09m58s，累计644；8226880 PENDING/Dependency。其后两Object尚未开始。后续接触补充与manifest SHA更新都在新c35准备分析根，原输入版须保留。主全部完成、全部成功续接接触核验与common-trace通过后才提交c35/interpret.sbatch；图形仍未实际生成QA，具体研究判断仍未完成。Goal active。

## Object0已完整前三个重复的成功续接接触证据

完整repeat0/1/2的分叉分别45/25/45；按预定规则选所有此前无机器人／目标-障碍接触且代理安全完成的同300步续接，含未干预AEGIS如有成功，实际15条（reference13、nominal2）。不据三重复作全状态判断。两nominal均来自repeat2的152/252步，不是两个独立重复。
batch `20261002T060522Z_object00_first3_suffix_contacts_6cd022f87faa`：根20261002T060532Z_object00_first3_suffix_chunk0 job **8239088 COMPLETED/0:0/5m17s**，10条；根20261002T060535Z_object00_first3_suffix_chunk1 job **8239090 COMPLETED/0:0/2m58s**，5条。源6cd/reset_forward=true，15条保存检查点严格复现且任务/位移复现，原row SHA逐一匹配；batch/AUDIT_COMPLETE保存Slurm及raw文件SHA。
13个reference均同时通过代理及无机器人／目标-障碍接触；两个nominal代理成功都接触了障碍（152步案例77个子步样本、252步案例10个，不能当77/10次独立碰撞）。所以前三个完整重复里尚无普通候选强接触条件下成功见证；剩余两个重复仍需完整核验。
c35未提交分析supplements已加入两根，contact roots现在18个，SHA **74be3a96f995d6dac3452a2a0dcd0a7525e986fe662be2996d19e92d22271751**。旧输入保留supplements_before_object00_first3.json，变更及hash存SUPPLEMENT_UPDATE_object00_first3.json，ANALYSIS_READY仍prepared_not_submitted。
末次主8226877 RUNNING/qgpu0403/2h15m23s，累计721（进入repeat4），继任8226880 Dependency。等Object0 COMPLETE/Slurm完成后，补audit剩余repeat3/4的所有合格代理成功（所有BRANCH_CONDITIONS）；通过新的18-root配置或两新root查已审计run_ids，避免重复计数。仍须Object2/5全阶段、全部common-trace/物理预算审计、实际图QA和具体判断表。Goal active。

## Object0完整与Object2实际启动

主GPU8226877实际COMPLETED/0:0/2h48m05s，Object0全部205分叉，主累计759，branch_object_00_COMPLETE存在，STOP不存在。控制器8226880实际COMPLETED/0:0/21s，receipt提交branch_object_02 GPU **8243879**与继任CPU **8243880**；末次8243879 RUNNING/qgpu0403/2m53s，8243880 Dependency。不要再等已完成的Object0句柄。
Object0全部五重复官方代理计数：参考0/27/77步5/5，两预算相同；152/252步原剩余预算0/4，同300续接预算4/4。普通候选只有release5在t0为1/5（repeat3），nominal在152/252同300续接各1/4（均repeat2，已核验有接触）；其余普通候选以及未干预AEGIS全检查点全预算0。后期参考的短预算失败由增加时间恢复，不能据此说机会随时间下降；最终仍需强接触条件同组预算全轨迹审计。
剩余合格同300续接代理成功共10条（reference9、release5一条），全来自repeat3/4；repeat3/252此前实际已接触，不列入强恢复子集，原行保留。源6cd/reset_forward=true已提交新唯一根 **20261002T065302Z_object00_remaining_suffix_contacts_6cd022f87faa**，CPU **8244439**（10案例），SUBMITTED.json。目前仅提交，须实际Slurm/CONTACT receipt后并入c35最新18-root manifest并保留旧版本；不要把一条release5代理成功当无接触成功或高成功概率。
仍待Object2/5完整、所有合格代理成功（含aegis预算对照）的接触核验、全common-trace预算审计、c35实际图QA、最终具体判断表；Goal active。

## Object0完整成功续接接触审计

8244439实际COMPLETED/0:0/4m25s，剩余10条source row SHA/保存检查点qpos/qvel/ctrl/任务/最大位移全部核验，renderer reset-forward=true。根20261002T065302Z_object00_remaining_suffix_contacts_6cd022f87faa/AUDIT_COMPLETE.json存Slurm与raw SHA。
c35准备分析已并入该根，contact roots现19个，supplements SHA **8f24e7d9a021f57da69cd940f6bf7b5301f543fe8f62b9fb71ae248996e8a36f**，旧输入保留supplements_before_object00_complete.json，变更hash存SUPPLEMENT_UPDATE_object00_complete.json；仍未提交报告。
Object0所有合格同300续接的代理成功接触未知数0。固定参考同时通过代理和无机器人／目标-障碍接触：0/27/77各5/5、152步4/4、252步3/3（排除repeat3已有接触，原行保留）。没有固定参考同预算的机会下降证据；原剩余预算晚期0/4的失败不能当作真实不可行或窗口衰退。普通库只有release5/t0/repeat3一例满足上述接触条件，1/5，不是稳定候选或概率认证；nominal/152、252的两代理成功均有接触，不能混入强安全续接。未干预AEGIS所有预算仍0。
末次实际Object2 GPU8243879 RUNNING/qgpu0403/11m23s，累计770；CPU继任8243880 Dependency。仍需Object2/5全部分叉与成功接触审计，完整共用轨迹/严格前缀配对预算核验，最终图QA/图文判断表。Goal active，主327冻结。

## Object2 前两个重复的接触证据（仍在运行）

Object2 GPU **8243879** 最后只读核验 RUNNING/qgpu0403/1h56m06s，主累计 **892/1209**，第三重复尚未完整；继任 **8243880** Dependency。不要用本条静态进度替代实际 progress/receipt，也不要重启作业。

- repeat0 的45条分叉完整；所有合格同300续接代理成功只有 reference 的0/27/77/152/252五条。
  接触根 `20261002T075840Z_object02_repeat0_suffix_contacts_6cd022f87faa`，CPU **8250639 COMPLETED/0:0/2m50s**。
- repeat1同样45条完整、五条reference成功。接触根 `20261002T081327Z_object_02_repeat1_suffix_contacts_6cd022f87faa`，CPU **8251820 COMPLETED/0:0/2m17s**。
- 两根均源6cd022f87faa，reset-forward=true；每条row SHA、保存检查点物理数组、任务与位移核验通过，十条均无机器人／目标-保护障碍接触。两重复不构成高概率认证，也不能将特权参考当成普通候选库的能力。
- c35准备分析已并入两根，contact roots现 **21**，最新supplements SHA `268a7ab7c0397461d83aa219138ece993b48b1f80491a111bdadfc5af964e1d3`。旧manifest及每次更新hash保留，分析仍未提交。

仍须Object2剩余重复、Object5全阶段、全部合格成功轨迹（包含aegis预算对照）的接触核验、全共同轨迹审计、Quest绘图与实际视觉QA、最终具体研究判断；Goal active。科学主327与Quest live d220源码保持冻结。

## Object2 完整及 Object5 实际启动

**8243879 COMPLETED/0:0/3h16m38s**，Object2全部225条分叉完整，主累计984。控制器 **8243880 COMPLETED/0:0/20s**；其receipt提交 **Object5 GPU8259539** 与 afterany继任 **8259540**。末次8259539实际RUNNING/qgpu0204/13m01s，累计998/1209，8259540 Dependency；以后须读取实际进度，不等旧Object2句柄。

Object2官方代理计数：0/27/77的reference两预算均5/5；152/252原剩余预算0/5、同300步续接5/5。所有普通候选与未干预AEGIS全部0。reference晚期失败可被时间预算解释，尚无同预算机会下降证据。

新增已完成接触根（各五条，均reference 0/27/77/152/252，源6cd/reset-forward=true、全部保存检查点与row SHA通过、无机器人／目标-障碍接触）：

- `20261002T084620Z_object_02_repeat2_suffix_contacts_6cd022f87faa`，**8254481 COMPLETED/0:0/2m52s**。
- `20261002T092458Z_object_02_repeat3_suffix_contacts_6cd022f87faa`，**8257220 COMPLETED/0:0/3m11s**。
- `20261002T100635Z_object_02_repeat4_suffix_contacts_6cd022f87faa`，**8259632 COMPLETED/0:0/2m31s**。

Object2全部25条合格同300步成功续接接触审计齐全、未知成功接触数0；各检查点同时通过代理及无上述接触5/5。这不是高概率认证，特权reference成功也不是普通库选择能力。完整common-trajectory与budget-alias审计仍待最终CPU分析。

c35准备分析contact roots现 **24**，supplements SHA `d25d74e68a99b27b14cc11f498054a6fdab943100aebf3b48c535ccf69fe9aae`；旧输入及每次hash更新均保留。分析仍未提交。尚须Object5、其所有合格代理成功接触审计、最终全轨迹/配对预算门、Quest实际图QA和具体研究决策表。Goal active。
