# CrashBench 运行入口

本目录只保留环境、验证和 Slurm 入口。论文主线与实验优先级以
[当前状态](../docs/CURRENT.md)和[论文计划](../docs/PAPER_PLAN.md)为准；旧的按时间顺序
执行手册已归档为 [SETUP_README_20260812.md](../docs/archive/SETUP_README_20260812.md)。

## 环境

三个模型栈必须分开，避免依赖互相覆盖：

| 后端 | 安装入口 | 验证入口 |
|---|---|---|
| OpenVLA base | `install_openvla_env.sh` | `verify_openvla.sbatch` |
| OpenVLA-OFT | `install_openvla_oft_env.sh` | `verify_openvla_oft.sbatch` |
| pi0/openpi | `install_openpi_env.sh` | `verify_openpi.sbatch` |

Quest 计算节点无外网；下载安装在登录节点完成，GPU 节点只做验证和评测。完整旧版本
pin、sanity 记录和踩坑列表在归档 setup 文档中。

## 当前已有结果的规范入口

| 论文部分 | 入口 |
|---|---|
| wall causal sweep | `run_ood_control_v5.sbatch` |
| Base/OFT/pi0 matched behavior | `run_pilot_base_matched.sbatch`, `run_pilot_oft.sbatch`, `run_pilot_openpi.sbatch` |
| Base/OFT representation capture | `probe_selfreport.sbatch`, `probe_selfreport_oft.sbatch` |
| wall-directed behavior | `submit_wall_directed_braking.sh` |
| probe → `RetreatHold` | `phase3_intervention.sbatch` |
| steering ablation | `phase3_steering.sbatch`, `phase3_steering_diag.sbatch` |
| glass dose response/probe | `phase2_glass_prototype.sbatch`, `probe_glass.sbatch` |
| prompt safety–utility baseline | `submit_careful_prompt.sh` |

这些入口对应冻结结果；重跑必须使用新输出目录，不能覆盖 `results/` 中已冻结的 JSON。
提交前后需记录 commit、checkpoint revision、scenario fingerprints、seed/repeat、阈值选择
单位和 Slurm provenance，具体见[可复现性说明](../docs/REPRODUCIBILITY.md)。

## 下一轮实验

优先级是：

1. 五折 held-out wall online guard；
2. glass-specific learned detector → structured `DetourComplete`；
3. OFT-specific online `RetreatHold` replication。

当前仓库还不能直接提交前两项：五折重训缺 wall/OFT raw hidden/meta；glass 只有 probe
summary，没有可部署 checkpoint。先从 Quest 找回 capture 或重采，再冻结新协议和提交脚本。

## Legacy glass recovery

`glass_core_realign*`、`glass_recovery_pilot_b*` 和 `glass_recovery_smoke*` 为路径稳定的
E14/E15 历史流水线，不是当前 run queue。Broad B 已 no-go；当前 checkpoint 的 learned gate
没有及时触发点，D/F 不应正式运行。

为防止误烧 GPU，三个历史 submit wrapper 默认 fail closed；只有显式设置

```bash
export CB_ENABLE_LEGACY_GLASS_RECOVERY=1
```

才会继续。这个开关仅用于有明确 provenance 目的的历史诊断，不表示推荐补跑 D/F。完整旧
协议和执行日志在 [glass recovery archive](../docs/archive/glass_recovery_20260812/README.md)。

旧的多作业 paper-round submitter 和自动 commit/push helper 已移到
[legacy/setup](../legacy/README.md)，避免与当前入口混用。

## 零 GPU 检查

从仓库根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests -q
PYTHONDONTWRITEBYTECODE=1 python scripts/audit_repo.py
git diff --check
```
