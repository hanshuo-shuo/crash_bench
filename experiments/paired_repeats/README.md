# π0.5 / full AEGIS 配对重复实验

**当前状态（2026-09-28）：原复现3200集已完成；自检7647815因容器结果目录只读，
启动后84秒失败，实际完成0集。三项自检均尚未执行，正式600次未启动。**
本地已修正 `/gpfs/projects/...` 根目录的读写绑定，并加入模型加载前的实际写入检查；
通过12项配对测试和Shell语法检查。用户已授权重跑三项自检；使用新的发布提交和结果根目录，
容器写入与运行检查通过与否以新作业日志为准；正式600次仍未启动。
本目录是独立流水线；现有 `scripts/`、`setup/`、上游源码、复现作业及其结果均不修改。
最新执行状态只读新实验根目录的 `plan.json`、`self_checks.json`、`STOP.json`、`progress.json`。

设计固定为 Spatial/I/task1、Object/I/task2、Object/II/task1，episode 0–19，
每状态每方法 5 次。按状态、重复号交错两方法，并交替先跑哪一方。
种子为 `SHA256("paired-v1|suite/level/taskN|episode|repeat")` 前 4 字节大端 uint32；
每 run 开头服务端确认 `jax.random.key(seed)`，后续保留 OpenPI 原生每请求 split。
每次重新建环境，Python/NumPy/env seed=7；1024 图像、20 步空转、5 步 action chunk、
300 步上限、pi05_libero、完整六轴 QP 加原 gripper 均固定。两方法仅开关感知与过滤器。

自检按序执行，任一失败即写 STOP，后续不运行：① 60 状态各 300 次官方零位移 dummy
action（gripper=-1），记录相对 20 步空转结束时障碍物最大 L1 位移，任何 >.001 m 即失败；
② 每状态 3 个全新环境，空转后的 qpos float64 字节哈希完全一致；③ Spatial/I/task1
episode 0、1，各方法一次，核对 RNG 回执、qpos 和首个策略 chunk 完全相同，完整解码视频，
报告每次耗时。还必须目检视频并查看 `steps.jsonl` 的六轴动作差 L∞ 分布，才能写 `review.json`
冻结小阈值；不根据成功率挑阈值。冒烟不混入正式结果。

复跑：先运行 `python3 -m unittest discover -s experiments/paired_repeats -p 'test_*.py' -v`
及既有 `tests/`（当前共 34 项通过），发布 clean commit 到 `codex/paired-repeats`。
Quest 先 `scripts/quest_sync.sh check`。通过 `scripts/quest_sync.sh exec` 只 fetch 发布提交、
`git archive <commit>` 到新的 `/projects/p33100/siosio/crashbench_safelibero/paired_repeats/<唯一ID>/source`，
根目录写 `SOURCE_COMMIT`；**不 merge/sync 活跃 checkout**。
从归档用 OpenPI Python 执行 `experiments/paired_repeats/launch.py checks <新根>`。
脚本校验归档每个文件等于发布提交，自检作业依赖原复现数组结束，使用 p33100/gengpu 单 GPU。
三项通过、目检视频、冻结阈值后用同一脚本 `launch.py full <根>`；它再次检查原数组终止、
最新复现 COMPLETE、自检证据哈希和 review。新旧 API 消费合计仍限 $5，每请求预留 $.10，
未知消费保留；302 次新 VLM 调用按已观测首笔 $.0018084 估约 $.54614，不保证一定够用。
密钥已私下配置，只在网络 worker 读取；每次 AEGIS run 新调用 GLM-4.5V/Z.AI，不复用回答。

每 run 保存 `row.json`、逐步原动作/QP 输出/位移 `steps.jsonl`、qpos、RNG 回执、视频、
适配后的源码、commit/配置/Slurm provenance。`checks/raw.csv` 与 `full/raw.csv` 原子更新。
步数统一为 **1 起算的已执行 action 步**；最大位移为激活障碍相对空转后位置的 L1 最大值。
QP 无解触发未定义 `a` 的原行为保留，记录退出；空点云关闭过滤器记“未启用”。服务/API/初始化
失败停止实验，不伪造一行科学结果。退出按已观测碰撞/成功归类并另计数。

完整 600 行通过覆盖率、种子和 qpos 检查后才输出 `summary/raw.csv`、`tables.md`、
`cross_tables_1mm.csv` / `1cm.csv`、`state_differences_1mm.csv` / `1cm.csv`、`summary.json`。
每组 5 次至少 4 次同类才稳定；两边稳定填 3×3，否则计“碰巧”。安全完成→安全完成且 AEGIS
至少 3 个 run 改过动作计“白干预”。包含逐场景、合并、按障碍资产种类的表；碰撞仍成功同时
给出占全部 nominal 和占碰撞 nominal 两种分母。VLM canonical/alias 可自动核对，其余标记
待目检，绝不把未知当正确或错误；最终交付前须完成这些标注。

附核验：现有 nominal Spatial 400 集 SS 字段均等于 `success and not collision`，零处不符；
未碰撞 53 集，全部成功，所以 SS 与未碰撞率均为 **53/400=13.25%**；撞了仍成功 188/400。
`audit_ss.py` 可只读重算并输出原文件 SHA256。异常与运行自检结果待新任务产生后据实补记。
