# CrashBench ↔ Quest 工作约定

这份文件是以后在 Codex 中继续 CrashBench 工作时的固定交接说明。SSH socket 只是连接
Quest 的通道，不代表某个项目；真正的项目隔离由 **Quest 目录、Git 仓库和 tmux 会话名**
共同保证。

## 固定身份卡

| 项目项 | CrashBench 固定值 |
|---|---|
| 本机仓库 | `/Users/hanshuo/Desktop/crash_bench` |
| Quest 主机 | `quest.northwestern.edu` |
| SSH socket | `/tmp/quest.sock` |
| Quest 仓库 | `$HOME/crash_bench` |
| Git 仓库身份 | `github.com/hanshuo-shuo/crash_bench`（HTTPS/SSH 均可） |
| tmux 会话 | `crashbench` |
| Slurm account | `p33100` |
| GPU partition | `gengpu` |
| 大文件根目录 | `/projects/p33100/siosio` |

`Mice-first-person` 或任何其他仓库都不是本项目目录。处理 CrashBench 时，不得进入、同步、
修改或提交其他项目的目录。

## 每次开始时，你只需要做一件事

在本机终端完成密码/Duo 认证并建立统一连接：

```bash
ssh -M -S /tmp/quest.sock -o ControlPersist=8h \
  -fN quest.northwestern.edu
```

然后可以检查连接：

```bash
ssh -S /tmp/quest.sock -O check quest.northwestern.edu
```

看到 `Master running` 后，在 Codex 中说：

> Quest 已连接，请按 `QUEST_WORKFLOW.md` 检查后继续。

其余路径核验、代码同步、tmux 和 Slurm 操作由 Codex 根据你的具体任务完成。不要把密码、
Duo 验证码或私钥发给 Codex。

如果提示 socket 已存在，先运行上面的 `-O check`；连接仍在时直接复用。如果 socket 文件
失效，才删除这个精确文件并重新连接：

```bash
rm -f /tmp/quest.sock
ssh -M -S /tmp/quest.sock -o ControlPersist=8h \
  -fN quest.northwestern.edu
```

## Codex 每次远程操作前必须核验

Codex 必须先做只读检查，不得根据 socket 名称猜测当前项目。首选仓库内置检查：

```bash
scripts/quest_sync.sh check
```

它包含以下 SSH 核验逻辑：

```bash
ssh -S /tmp/quest.sock -o BatchMode=yes quest.northwestern.edu \
  'cd "$HOME/crash_bench" && pwd && git remote -v && git branch --show-current && git status --short'
```

必须同时满足：

1. `pwd` 位于 `$HOME/crash_bench`；
2. `origin` 指向 `hanshuo-shuo/crash_bench`（HTTPS 或 SSH URL 均可）；
3. 本地和 Quest 的目标 commit/分支关系已经核对；
4. Quest 工作区没有未保存改动，或这些改动已经明确交代并妥善保留。

任一项不符时立即停止，不得 `git reset --hard`、覆盖目录、强制拉取或猜测应该保留哪一份
改动。先向你报告实际路径、remote、分支和工作区状态。

## 项目隔离

CrashBench 使用以下边界：

```text
SSH 通道      /tmp/quest.sock
Quest 代码    $HOME/crash_bench
tmux 会话     crashbench
项目存储      /projects/p33100/siosio
Slurm         account=p33100, partition=gengpu
```

需要交互式 shell 时使用专属会话：

```bash
ssh -t -S /tmp/quest.sock quest.northwestern.edu \
  'tmux new -A -s crashbench -c "$HOME/crash_bench"'
```

其他项目必须使用不同的 Quest 目录和 tmux 会话名。所有 GPU/CPU 重计算都通过 Slurm；登录
节点只做 Git、查看文件、下载预编译依赖和轻量检查。

## 同步和提交作业的固定顺序

本项目使用精确 Git commit 同步，不用 `rsync` 覆盖 Quest 源码：

```bash
# 本机仓库根目录
scripts/quest_sync.sh status
scripts/quest_sync.sh dry-run
scripts/quest_sync.sh push
scripts/quest_sync.sh submit setup/<job>.sbatch
scripts/quest_sync.sh queue
```

`push` 前必须满足：本地工作区 clean、HEAD 已推送到 upstream、Quest 工作区 clean。需要取回
结果时，只拉明确的 `results/` 路径：

```bash
scripts/quest_sync.sh pull-result results/<path>
```

提交脚本应使用：

```bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
```

显卡型号、显存、时间和内存按具体实验决定。环境、缓存和 checkpoint 的现行安排见
[`cluster.md`](cluster.md)，当前实验入口见 [`setup/README.md`](setup/README.md)。不得覆盖
`results/` 中已经冻结的结果；重跑必须使用新的输出目录。

## 结束连接（可选）

连接会在闲置 8 小时后自动退出。需要提前结束时：

```bash
ssh -S /tmp/quest.sock -O exit quest.northwestern.edu
```
