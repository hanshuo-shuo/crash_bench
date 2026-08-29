# CrashBench 运行入口

本目录只保留环境、验证和 Slurm 入口。论文主线与实验优先级以
[当前状态](../docs/CURRENT.md)和[论文计划](../docs/PAPER_PLAN.md)为准；旧的按时间顺序
执行手册已归档为 [SETUP_README_20260812.md](../docs/archive/SETUP_README_20260812.md)。

## 从本机直接提交到 Quest

这是 CrashBench 唯一推荐的本地提交路径。用户只负责在本机完成密码/Duo 认证；后续的
项目核验、精确 commit 同步和 `sbatch` 提交都从本机仓库根目录执行，不需要先登录 Quest
再手动 `cd`。

### 1. 每个连接周期只做一次认证

在本机终端运行：

```bash
ssh -M -S /tmp/quest.sock -o ControlPersist=8h \
  -fN quest.northwestern.edu
```

连接闲置 8 小时后自动结束。项目隔离和 socket 故障处理见
[Quest 工作约定](../QUEST_WORKFLOW.md)。

### 2. 从本机核验连接

```bash
cd /Users/hanshuo/Desktop/crash_bench
scripts/quest_sync.sh check
```

`check` 会验证：

- socket 可以复用；
- 本机仓库是 `hanshuo-shuo/crash_bench`；
- Quest 目录是 `$HOME/crash_bench`；
- Quest 的 origin 指向同一 GitHub 仓库（HTTPS 和 SSH URL 都接受）；
- Quest 上存在 `sbatch`。

任一身份不匹配都会直接停止。不要根据 socket 文件名猜测当前项目。

### 3. 一条命令提交单个 sbatch job

```bash
scripts/quest_sync.sh submit setup/<job>.sbatch
```

例如：

```bash
scripts/quest_sync.sh submit setup/run_pilot_base_matched.sbatch
```

这条命令会自动：

1. 检查 sbatch 文件的 Bash 语法、`account=p33100` 和合法 partition；
2. 要求本地工作区 clean，且本地 HEAD 已发布到 upstream；
3. 要求 Quest 工作区 clean，且 Quest commit 是本地 HEAD 的祖先；
4. 让 Quest checkout 通过 Git fast-forward 到完全相同的 commit；
5. 从 Quest 的 `$HOME/crash_bench` 执行 `sbatch --parsable`；
6. 在本机打印 Slurm job ID。

因此“直接提交”仍然保持精确 provenance：未提交或尚未推送的本地代码不会被偷偷复制到
Quest。如果 Codex 为任务修改了代码，应先完成检查、commit 和 push，再运行上述提交命令。

### 多作业流水线和运行参数

`setup/submit_*.sh` 会建立依赖链、生成唯一输出目录或注入 commit；这类入口先同步，再从
本机让 Quest 执行 wrapper：

```bash
scripts/quest_sync.sh push
scripts/quest_sync.sh exec 'bash setup/submit_dynamic_first_crossing_calibration.sh'
```

需要环境变量覆盖时，也在远端命令中明确写出。例如：

```bash
scripts/quest_sync.sh push
scripts/quest_sync.sh exec \
  'CB_REPEATS=5 bash setup/submit_wall_directed_braking.sh'
```

不要在本机先 `export CB_...` 后假设变量会自动穿过 SSH。

### 查看作业和取回结果

```bash
scripts/quest_sync.sh queue
scripts/quest_sync.sh exec 'scontrol show job <job-id>'
scripts/quest_sync.sh exec 'sacct -j <job-id> --format=JobID,JobName,State,Elapsed,ExitCode'
scripts/quest_sync.sh pull-result results/<path>
```

大多数 job 的日志由 `#SBATCH --output=%x_%j.log` 写在 Quest 仓库根目录。可用
`scontrol show job <job-id>` 的 `StdOut=` 字段确认准确路径。`pull-result` 只接受
`results/` 下的明确路径，并且不会删除本地文件。

### 新 sbatch 文件的最小模板

GPU job 放在 `setup/`，至少应包含：

```bash
#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=crashbench_example
#SBATCH --output=%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"

printf 'commit=%s\n' "$(git rev-parse HEAD)"
printf 'host=%s job=%s\n' "$(hostname)" "$SLURM_JOB_ID"

module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"

python scripts/<entrypoint>.py
```

纯 CPU job 使用 `#SBATCH --partition=short` 并删除 `--gres`。仓库当前所有
`setup/*.sbatch` 都已经声明 `account` 和 `partition`；GPU job 使用 `gengpu`，少数纯 CPU
分析/训练 job 使用 `short`。不要在脚本里 `module load cuda`，也不要在计算节点下载依赖。

### 常见拒绝原因

| 报错 | 含义和处理 |
|---|---|
| `SSH socket not found` | 先完成上面的 Duo 连接 |
| `local worktree is dirty` | 检查改动，完成测试后 commit |
| `HEAD is not published` | 将当前分支 push 到 upstream |
| `Quest worktree is dirty` | 停止提交，先保留并核对 Quest 改动 |
| `Quest history is not an ancestor` | 本地和 Quest 已分叉，先人工核对 Git 历史 |
| `must declare #SBATCH ...` | 修正 account/partition，不能临时绕过保护 |

不要用 `git reset --hard`、强制 pull 或 `rsync` 覆盖来消除这些报错。

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
| **P2 动态 first-crossing Router** | 先运行 `submit_dynamic_first_crossing_calibration.sh` 冻结 trajectory-level boundary，再运行 `submit_dynamic_first_crossing_router.sh`；协议和指标见 `docs/P2_DYNAMIC_FIRST_CROSSING.md` |
| **ICLR Phase 2.5B source-cross-fit** | `iclr27_support_crossfit.sbatch`；Quest Job 5137872 已完成并记录为 `INCONCLUSIVE`，不得重跑覆盖，未授权 Screen A |
| **fresh counterfactual router 主结果** | `submit_fresh_counterfactual_router.sh`, `submit_fresh_counterfactual_router_supplement.sh`；结果已冻结，不要覆盖重跑 |
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

现有 20-source corpus 上的 CPU-only Phase 2.5B source-cross-fitted support rescue 已完成，
正式门为 `INCONCLUSIVE` 且 fail closed。不得在现有 8/13-source cohort 上搜索更深
sequence model、Transformer、ensemble world model 或 outcome-led threshold；当前也不得
启动 Screen A 或任何新 outcome-bearing rollout。后续动作需要先明确修订科学计划，不能把
`INCONCLUSIVE` 自动解释为 GO。

Phase 2.5B 以后只有对应 gate 明确授权，才考虑新实验；原有后备顺序为：

1. 在新 task family 上原样复制冻结的 counterfactual-router 方法和 frontier；
2. 五折 held-out wall online guard；
3. OFT-specific online `RetreatHold` replication。

新 task-family replication 需要新的 source-disjoint cohort 和 task-compatible structured
options。五折 wall/OFT 重训仍缺 raw hidden/meta，需要先从 Quest 找回 capture 或重采。

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
