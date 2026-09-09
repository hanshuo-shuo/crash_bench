# 固定 Detour 收益开发：执行冻结

用户于2026-09-09明确要求执行完整开发方案。资产审计见asset_audit.md。本计划只授权开发16来源，不自动提交条件性8新来源验证。旧数据、D8及历史解释保持不变。

机器协议：configs/detour_benefit/development_v1.json；运行入口setup/detour_benefit.sbatch。
48条前缀（16来源×3条件），每个触发候选Base/Detour各4次，最多384 scored branches。
来源按已有风险模型使用边界选择：旧train及calibration中实际参与模型/阈值的12来源全部留fitting，其余来源按固定身份哈希选择4 calibration。
每来源取清单路径/placement ID字典序第一项，不读恢复结局来选择。历史已有hazard screening，不能称新无偏来源样本。
源身份与D8治理清单的confirmatory来源哈希无交集；不读D8结局。

候选为固定风险分数首次严格超过0.3072859857738742，最多在第0至219个任务动作前触发。
旧风险阈值仅用现归fitting侧的历史7来源选择，不使用新calibration/B结局；该阈值预先固定，不因触发稀少下调。
当前动作及hidden在原本的OpenVLA调用后产生，无额外VLA补特征。没有触发时持续Base直到终局/440，仍在分母。

同bundle所有分支共享环境、控制器、策略continuation、缓存、RNG和pending action。
Base先执行已生成proposal；Detour丢弃proposal，运行原DetourComplete legs，完成后以当前观测调用Base，无第二次候选。
DetourComplete本体动作代码未改变；支路已达到成功/事故时不再恢复Base。事故优先；失败和提前停止不能计为加速。
记录prefix、每个branch的逐步输入/动作/状态、调用和控制器动作数量、墙钟耗时、bundle及权重哈希。

H=220、2H=440都从episode任务动作起点计时，prefix和Detour动作均计入；settle10另记。
环境运行时上限必须支持至少450步；不在H重置或推理读出。完整成功累计曲线0..440全部报告。
A repeats0/1同bundle同RNG，B repeats2/3恢复同bundle后仅torch策略RNG分别设20270902/20270903，同repeat两臂配对。
OpenVLA greedy不改：不同种子不自动代表非退化随机策略分布。顺序按episode_index+repeat奇偶交替B/R。

模型PCA固定4维；训练当前帧数>=5且中心化rank>=4，否则两个学习gate全Base并报告支持不足，不改维度寻找成绩。
Scaler/PCA/Ridge只用12fitting来源A结果；来源总权重等同。Ridge alpha=1，分别预测H/2H的成功差和事故差。
Full gate以H成功预测作score，并要求两预算事故差预测<=0；risk-only benefit同配方。
StrongRisk用同候选风险分数，阈值只在4calibration来源A选择。所有阈值最大化H成功；约束两预算事故差<=0、正常控制逐候选边际成功无损。
允许全Base；平局更少事故、更少干预、更高阈值。预算H预先固定为主目标；2H独立报告，不选有利截止。
DirectQ同配方严格代数等价，合并sanity；不是新算法。A参考分别按H及2H正成功差选择，仅诊断。
所有模型、选择和阈值写freeze并校验哈希之后，才执行任何B分支。

主评价B：全部16来源等权，源内三条件等权；no-trigger与早终局作为共享实际结果保留，不能伪造重复分支。
报告相对Base、强Risk及DirectQ的成功/事故配对差，原成功损失、新增事故、正常保持、无触发、干预率和实际成本。
来源共享bootstrap5000次给描述性区间，完整逐来源表、A/B兑现和运行时波动。小样本、零事故不称安全保证。
实现和测试通过只表示可执行，不是科学结果。完成开发后依据用户去留规则撰写中文结论；不会自动扩展网络、来源或预算。
