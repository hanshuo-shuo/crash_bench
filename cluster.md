# Quest 上搭 PyTorch 环境备忘

> 维护人:shv7753 · 最近清理:2026-06-19

## 核心认知

- conda/pip 装的 PyTorch **自带 CUDA 运行时**,**不需要 `module load cuda`**。
  GPU 节点有 NVIDIA 驱动就够了。Quest module 里的 cuda(最高 12.1)是给编译型软件用的,
  load 了反而会和 torch 自带的运行时冲突 —— **别 load**。
- 现有环境:
  - `~/pytorch-cuda-12-4`(Python 3.12,torch 2.10.0+cu128 / transformers 5.7.0 / mujoco 3.3.7)。
    通用新栈;**注意 transformers 5.x 跑不了 OpenVLA**,别拿它评 OpenVLA。名字写 12-4 实际 cu128。
  - `~/crash_bench/envs/openvla`(Python 3.10,torch 2.2.0+cu121 / transformers 4.40.1 /
    flash-attn 2.5.5 + LIBERO)。CrashBench 评 OpenVLA 专用,官方老栈。
    安装脚本 `crash_bench/setup/install_openvla_env.sh`,sbatch `install_openvla.sbatch`。
- 权重/缓存全在 **p33100**:`HF_HOME=/projects/p33100/siosio/huggingface_cache`,
  `TORCH_HOME=.../torch_cache`,pip 缓存 `.../pip_cache`(已在 ~/.bashrc / 安装脚本里设好)。
  account 提交作业用 `p33100`。

## 三条铁律

1. **永远不要 `pip install --user`**。激活环境后直接 `pip install`。
   (`--user` 会把包丢进 `~/.local`,污染所有 conda 环境 —— 之前 16G 的祸根就是这个。)
2. **一个项目一个环境**,别堆一起。删起来干净:`mamba env remove -p ./envs/xxx`。
3. **`gengpu` 批处理计算节点无外网**(实测 2026-06:`git/pip/huggingface` 全连不上)。
   所以分两步:**联网的下载/装包(git clone、pip 装 wheel、HF 下 checkpoint)在登录节点做**
   —— 只要是预编译 wheel、没有源码编译,登录节点纯下载不吃 CPU,可放心装;
   **真正需要 GPU 的源码编译 / 验证**才用计算节点,且必须先在登录节点把包/源码备好再离线跑。
   (OpenVLA 这套全是预编译 wheel,所以整套在登录节点装完,GPU 节点只做验证/eval。)

已设 `export PYTHONNOUSERSITE=1`(在 `~/.bashrc`),防止 `~/.local` 的包再串进环境。

## Quest GPU 约定(官方,2026-06 现行)

- **GPU 分区**:`gengpu`(General Access,单作业最长 48 小时)。
- **account 格式**:`eXXXX` / `pXXXX` / `bXXXX`(填你自己的;不知道就 `sacctmgr show user $USER` 或问 quest-help)。
- **选卡**:`--gres=gpu:a100:1`(A100) / `--gres=gpu:h100:1`(H100) / `--gres=gpu:1`(任意,排队最快)。
- **选 80GB 显存**:加 `--constraint=sxm`(所有 H100 + 部分 A100 是 80GB)。
- **上限**:单用户同时最多 8 块 GPU,超了作业会卡在 `(QOSMaxGRESPerUser)`。
- GPU 显存不用单独申请,跟着卡走。

## 新建 PyTorch 环境流程

```bash
# 1. 进交互式 GPU 节点(<account> 换成你的 eXXXX/pXXXX/bXXXX)
salloc -A <account> -p gengpu --gres=gpu:a100:1 -N 1 -n 8 --mem=32G -t 2:00:00

# 2. 干净地初始化 mamba
module purge
module load mamba/24.3.0

# 3. 建环境(-p 放项目目录里;或 -n 名字 放 ~/.conda/envs)
mamba create -p ./envs/myproject python=3.12 -y
source activate ./envs/myproject     # Quest 上用 source activate,不是 conda activate

# 4a. 装 PyTorch —— 官方推荐做法(conda-forge,CONDA_OVERRIDE_CUDA 处理 CUDA):
CONDA_OVERRIDE_CUDA=12.6 mamba install 'pytorch[channel=conda-forge,subdir=linux-64,build=*cuda*]>=2.5.1' torchvision

# 4b. 或者直接用官方 pip wheel(最新版,我现有环境就是这么来的):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 5. 其它依赖照常 pip install(环境已激活,绝不加 --user)
pip install numpy pandas ...

# 6. 验证 GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## sbatch 提交脚本模板

```bash
#!/bin/bash
#SBATCH --account=<account>      # eXXXX / pXXXX / bXXXX
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1        # 或 gpu:h100:1 / gpu:1
#SBATCH --constraint=sxm         # 需要 80GB 显存时加;否则可删
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=myjob

module purge
module load mamba/24.3.0
source activate ./envs/myproject

python train.py
```

## 常用维护命令

```bash
mamba env list                       # 列出所有环境
du -sh ~/.conda/envs/* ~/*-cuda*      # 看各环境占用
mamba env remove -p ./envs/xxx        # 删环境(prefix 形式)
mamba env remove -n name              # 删环境(命名形式)
mamba clean -a -y                     # 清 conda 包缓存
rm -rf ~/.cache/pip                   # 清 pip 缓存
```

## 本地编辑、Quest 运行

VS Code 和 Codex 都在本地仓库运行，只通过一个持久 SSH 会话增量上传代码。先在本机终端
完成一次密码/Duo 认证：

```bash
ssh -M -S /tmp/quest.sock -o ControlPersist=8h \
  -fN quest.northwestern.edu
```

完整的项目隔离、连接核验、tmux 和交接约定见
[`QUEST_WORKFLOW.md`](QUEST_WORKFLOW.md)。

然后在本地项目根目录使用：

```bash
scripts/quest_sync.sh check
scripts/quest_sync.sh status
scripts/quest_sync.sh dry-run
scripts/quest_sync.sh push
scripts/quest_sync.sh submit setup/run_pilot.sbatch
scripts/quest_sync.sh queue
scripts/quest_sync.sh pull-result results/example.json
```

`push` 不再用 rsync 覆盖源码。它要求本地工作区 clean、本地 HEAD 已发布到 upstream、
Quest 工作区也 clean，然后让 Quest 通过 Git fast-forward 到完全相同的 commit。
因此 provenance 中的 commit 与实际执行代码一致；云端被忽略的 `envs/`、
`third_party/`、模型缓存、大视频和 activation dump 不会被 Git 操作触碰。

## 排查清单

- **GPU 用不了 / `cuda.is_available()` 是 False**:确认在 GPU 节点上(`nvidia-smi` 能看到卡);
  确认没 `module load cuda` 干扰;确认装的是 cu 版 wheel 不是 cpu 版。
- **import 到了奇怪的包版本**:检查 `~/.local`(应已清空),确认 `echo $PYTHONNOUSERSITE` 是 1;
  `python -c "import torch; print(torch.__file__)"` 看是不是指向当前环境。
- **`module avail` 报 lua 语法错误**:删损坏缓存 `rm -rf ~/.cache/lmod`,下次自动重建。

## Quest 配额

家目录默认配额约 80GB。空间紧张时优先清:多余 conda 环境、`~/.cache/pip`、`~/.local`。

## 官方文档链接(优先参考这些)

> 官方技术文档 2025 年起迁到了 **rcdsdocs.it.northwestern.edu**,旧的 services.northwestern.edu KB 文章仍可看但可能过时。

- **GPUs on Quest**(分区 / salloc / sbatch / 选卡):https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/gpu/gpu.html
- **GPUs in Python / PyTorch 环境**(官方 mamba 命令、验证):https://rcdsdocs.it.northwestern.edu/tutorials/python/python-gpus.html
- **Hugging Face on Quest**:https://rcdsdocs.it.northwestern.edu/tutorials/python/python-llm-huggingface.html
- **Quest 规格(节点 / GPU / 存储)**:https://rcdsdocs.it.northwestern.edu/systems/quest/specs/quest-specs.html
- 文档站首页:https://rcdsdocs.it.northwestern.edu
- GPU 资源总览(IT 官网):https://www.it.northwestern.edu/departments/it-services-support/research/computing/gpu/
- 旧 KB —— Anaconda 虚拟环境:https://services.northwestern.edu/TDClient/30/Portal/KB/ArticleDet?ID=2064
- 旧 KB —— Using Python on Quest:https://services.northwestern.edu/TDClient/30/Portal/KB/ArticleDet?ID=1672
- 旧 KB —— Quest User Guide:https://services.northwestern.edu/TDClient/30/Portal/KB/ArticleDet?ID=505
- 求助邮箱:quest-help@northwestern.edu

## 硬件速览(General Access GPU)

- H100 SXM 80GB:24 节点 × 4 卡
- A100 SXM 80GB:18 节点 × 4 卡
- A100 PCIe 40GB:16 节点 × 2 卡
- 共约 100 个 GPU 节点 / 308 块卡(含 Priority Access)
