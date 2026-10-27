# V5 交接（2026-10-27：检测器基线完成，`training_needed`）

读者：接手的人（用户本人，或下一个 Claude/Codex 会话）。先读 [AGENTS.md](../../AGENTS.md) 和 [CURRENT.md](../CURRENT.md)，再读本文的"接手指南"。后面按日期排的各节是历史记录。

## 接手指南（2026-10-27 更新，先读这一节）

### 现在在哪

V5 这一轮的故事线已经走完一遍，只用了 object/II/task1 一个任务组合：

| 步 | 结论 | 报告 |
|---|---|---|
| S3 | 直接判断（A）和先写"在哪一侧"（B）在反事实孪生验证上都是 50%（`both_shortcut`） | [V5_S3_REPORT.md](V5_S3_REPORT.md) |
| 拟合诊断 | B 能背下训练图，但不迁移（`inconclusive`） | [V5_FIT_REPORT.md](V5_FIT_REPORT.md) |
| S4 | P 先写四个点的位置、固定代码判定：验证 0.940，A 0.501（`positions_ground`） | [V5_S4_REPORT.md](V5_S4_REPORT.md) |
| 加 seed | 3 个 seed 上 P 0.933–0.947，A 0.50（`stable`） | [V5_SEEDS_REPORT.md](V5_SEEDS_REPORT.md) |
| 测试评估 | 4 条没见过的测试源上 P 0.924–0.942，A 0.50；P − A +0.432 [+0.416, +0.449]（`test_confirms`） | [V5_TEST_REPORT.md](V5_TEST_REPORT.md) |
| 检测器基线 | 不训练，AEGIS 的 GroundingDINO 找四个物体再跑同一段程序：验证 0.521，P 0.940；P − 检测器 +0.419 [+0.395, +0.443]（`training_needed`）；布丁几乎找不到 | [V5_DET_REPORT.md](V5_DET_REPORT.md) |

- **一句话**：在自然分布上训练、在挡住捷径的反事实孪生上测试，直接判断只有 50%；模型先写出看得见的位置、再由固定代码判定，验证 94%、测试 93%。模型负责"东西在哪"，代码负责"隔没隔开"；P 自己写的结论是随机水平。"东西在哪"不能换成现成检测器：AEGIS 自带的 GroundingDINO 加同一段程序只有 52%。
- **状态**：
  - 没有正在运行的 CrashBench 作业；队列里 p31777 账号的作业是别的项目，不要动；
  - 工作区干净，所有工作都在 `main` 上并已推送（10-27 从 `v5-det` 分支快进；见 CURRENT.md 的分支一节）；
  - 主报告页 <https://claude.ai/artifact/Ya7FQbZC9yEuNNyYhPBHHT> 是第 23 版（加了 `#v5-det`），与仓库里的 `PHASE2_REPORT.html` 相同。
- **10-27 方向**：用户决定主线回到原来的想法（模型自己推理、自己下结论）。P 的"代码判定"相当于把生成标签的规则直接写给了它，P 对 A 的差距衡量的主要是定位，所以 P 只作诊断，不当作方法。用 P 做闸门或系统（下表选项 1）保留为备选；当前主线是选项 2。
- **V5 测试集已经用掉**：`test`、`test_natural` 以后不再评估，也不拿来选模或调参。新方法如果要一个没碰过的测试，需要新的留出源（要另批数据）。V4 的锁定测试集同样不再评估。

### 接下来可以做什么（每一项都要用户先点头）

| 选项 | 要做什么 | 需要的批准 | 大概的量 | 从哪开始 |
|---|---|---|---|---|
| 1. AEGIS 闸门（10-27 起为备选） | 机器人动手前，用 P 判断新规则下任务还做不做得成，做不成就拦下。检测器基线说明感知要用 P，不能用 AEGIS 自带的检测器 | 控制器改动（AGENTS.md 的 Phase 4 规则）；如果要跑完整 AEGIS，还要付费 API 的预算批准 | 先写设计稿（10-26 建议：把规则连线作为 CBF-QP 的半空间约束；不加闸门时做不成的任务会停在线前，变成"安全但没完成"）：任务中途在线渲染同样的两张斜视图、规则何时加入、P 判"做不成"后控制器怎么做 | [REPRODUCTION.md](../REPRODUCTION.md)、[reproduction/BATCH_RUN.md](../reproduction/BATCH_RUN.md)、`setup/submit_safelibero_full.sh`；P 的输入构造在 `v5_pos_run.p_item` |
| 2. 教模型自己算隔开（10-27 起为主线） | 新的训练目标，让模型从自己写的四个点推出 `Separated`，不再靠代码 | GPU 训练；只看验证即可，要测试就需要新的留出源 | P 每个 seed 训练约 4 小时，3 个 seed 约 12 GPU 小时 | `v5_pos_contract.py`（目标、解析、程序）、`v5_pos_run.train_p` |
| 3. 扩到其他任务组合 | goal、spatial 任务；S0 显示这些任务上捷径还在，严格配平后孪生不够 | 补源（C1–C5、nominal 重新执行，之前暂停）需要数据批准 | CPU 回放和布局 + GPU 渲染，量取决于补多少源 | [V5_S0_REPORT.md](V5_S0_REPORT.md)、`verifier_dataset/v5_launch.py` |
| 4. 整理论文主线 | V4 Phase 2 加 V5 的验证、加 seed、测试三步写成一条线 | 不需要算力 | 写作 | [PRELIMINARY_STORY.md](PRELIMINARY_STORY.md)、[PAPER_READING.html](PAPER_READING.html)、本目录的 `V5_*_REPORT.md` |
| 5. 更强的现成检测器（可选） | 补 OWLv2 或 Florence-2 加同一段程序，查询词只在训练集上选 | GPU 推理；权重要先在登录节点下载放好 | 约 1 GPU 小时 | `v5_det_contract.py`、`v5_det_run.py`、`v5_det_detect.py` |

- 这次没有在测试集上评估 S3 的 B 和零样本模型，测试集用掉以后也不再补。

### 每一步的做法（V5 一直这样做）

1. 先写协议 `docs/generative_verifier/V5_<步>_PROTOCOL.md`：问题、设计、读数、事先写定的读法和门槛、资源估计。
2. 写代码和单元测试（`verifier_training/v5_<步>_contract.py`、`v5_<步>_run.py`，启动器 `launch_v5.py` 加一个动作，测试放在 `verifier_training/tests/`）。
3. 在草稿目录里跑 smoke，只用开发数据；把 smoke 的结果写进协议的冻结记录，冻结协议后提交、推送。
4. 新工作先从 `main` 开分支；用 `launch_v5` 从干净、已推送的 commit 启动；每次新建运行目录（`<UTC>_v5-<步>_<commit>`），源码归档、收据和哈希都在里面。
5. 结束后写 `V5_<步>_REPORT.md`，同步 CURRENT、本文、V5_PLAN 第 6 节、主报告页（中英文都改，用 `url` 发布仓库里的文件），再提交、推送，并快进 `main`。

### 要记住的规矩和坑

- 提交身份：`git -c user.name=hanshuo-shuo -c user.email=80494218+hanshuo-shuo@users.noreply.github.com commit`。
- Quest：账号 p33100，GPU 分区 gengpu，CPU 分区 short，同时最多 2 张 GPU；登录节点上不跑重活；计算节点离线，资产要先放好。有 CrashBench 作业在跑时不要改源码目录。
- 不调付费 API，除非用户另批；OpenRouter 的 key 不打印、不提交、不进日志。
- 评估已训练的 adapter 时要打开 TF32（`torch.backends.cuda.matmul.allow_tf32=True`），否则复现不了训练时的输出。
- 跑测试：`PYTHONPATH=/projects/p33100/siosio/crashbench_safelibero/verifier_training/assets/20261010_d94c245/packages:. /projects/p33100/siosio/envs/openpi/bin/python -m unittest <verifier_training/tests 下的模块>`（`unittest discover` 不能用）；共 174 个。
- 更多的坑见下面"坑"和"规矩"两小节。

### 关键路径

| 内容 | 位置 |
|---|---|
| S2 数据集 | `/projects/p33100/siosio/crashbench_safelibero/verifier_dataset/20261025T013358Z_v5-s2_d1d8d4129223/dataset.jsonl`（测试冻结在同目录的 `TEST_FREEZE.json`） |
| 运行目录 | `/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs/` 下：S3 `20261025T055227Z_v5-s3_bdcf277164b6`，拟合诊断 `20261025T153621Z_v5-fit_ea76dabed506`，S4 `20261025T180525Z_v5-s4_20913c71c3f3`，加 seed `20261026T001735Z_v5-seeds_ee20b2033465`，测试 `20261026T173919Z_v5-test_5594cebeb616`，检测器基线 `20261027T031559Z_v5-det_380d63210941` |
| 选定的 adapter | S3 的 `A_seed713`、S4 的 `P_seed713`、加 seed 目录里的 `A_seed714`、`A_seed715`、`P_seed714`、`P_seed715`，各自的 `selected_adapter/`；哈希记在测试目录的 `FREEZE.json` |
| 测试读数 | 测试目录的 `TEST_READING.json` |
| 检测器基线读数 | 检测器基线目录的 `DET_READING.json`；原始框在 `detect/detections.jsonl` |
| 主报告页 | <https://claude.ai/artifact/Ya7FQbZC9yEuNNyYhPBHHT>，仓库副本 `docs/generative_verifier/PHASE2_REPORT.html` |

## 2026-10-27 S5 运行中

- 用户定了方向（主线回到"模型自己下结论"）后说"我觉得很好 帮我开始吧"，又说"弄完后自动帮我交gpu的训练 我明早起来看"。
- **问题**：P 能找准四个点，但自己写的结论是随机水平。S 在 P 的四行之后写出位移、两个叉积（两个乘法、一个减法）、正负号、`Separated` 和 `Verdict`，读数看模型自己写的 `Verdict`。
- 协议 [V5_S5_PROTOCOL.md](V5_S5_PROTOCOL.md) 冻结于 `f20304e`（实现 `e62ce81`，hooks 合并修正 `017232a`）；代码 `v5_self_contract.py`、`v5_self_run.py`，`v5_pos_run.train_p` 加了可选的 `hooks`（不传时仍是 P），`v5_gpu_long.sbatch`（20 小时），启动器动作 `launch_v5 s5 --s3 --s4`。
- 两次 smoke：第一次查出 hooks 合并时键重复的报错，第二次全部 COMPLETED 0:0（见协议第 8 节）。
- 运行目录 `/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs/20261027T050745Z_v5-s5_f20304e222b9`；作业 5554027 → 5554028 → 5554029。读法事先写定：`self_computes`（自己写的结论 ≥ 0.85）、`partial`、`no_self_compute`（≤ 0.60）、`unclear`。
- 跑完后写 `V5_S5_REPORT.md`，同步 CURRENT、本文、V5_PLAN 第 6 节和报告页。

## 2026-10-27 检测器基线完成

- 用户 10-26 先和 Claude 对了一遍故事线和报告页，然后说"先做1"：不训练，直接用 AEGIS 自带的 GroundingDINO 定位，再跑 P 的同一段程序，看能不能判对孪生题。
- **结果**见 [V5_DET_REPORT.md](V5_DET_REPORT.md)，预注册读法 **`training_needed`**：
  - 验证集 1,538 行上检测器 0.521 [0.499, 0.546]，P（3 个 seed 平均）0.940；P − 检测器 +0.419，图像区间 [+0.395, +0.443]；
  - 布丁几乎找不到：392 张图里最高分框命中 3 张，前 5 个框里有布丁的 74 张；框多半落在其他杂货、字母汤罐、酒瓶上；
  - 锚点命中 47%：红杯子、酒瓶、橙汁盒约九成，字母汤罐、沙拉酱瓶、黄色书 6–21%；篮子 95%；
  - 训练集上 48 组框到点的比例都在 0.515–0.536（选中框中心）；AEGIS 原规则（第一个超过 0.35 的框）0.500。
- 协议 [V5_DET_PROTOCOL.md](V5_DET_PROTOCOL.md) 冻结于 `380d632`（实现 `988f9fe`，sbatch 修正 `293a299`）；代码 `v5_det_contract.py`、`v5_det_run.py`、`v5_det_detect.py`（在 AEGIS 容器里独立运行，Python 3.8）、`v5_det_gpu.sbatch`，启动器动作 `launch_v5 det --s3 --s4 --seeds`。
- 运行目录 `/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs/20261027T031559Z_v5-det_380d63210941`；作业 5520821（preflight）、5520823（检测，A100 PCIe，14 分 37 秒，7,190 次查询）、5520824（report）全部 COMPLETED 0:0。不调 API、不训练、不读测试集。
- **坑**：
  - GPU 阶段要在 AEGIS 容器里跑（`module load singularityce/4.3.1-gcc-8.5.0`，bind `/projects,/home,/gpfs`，`python -I`）；登录节点的非交互 shell 里要先 `module purge` 再 load 才找得到 singularity；
  - transformers 4.21 要从当前目录下名为 `bert-base-uncased` 的链接加载 BERT（同 AEGIS 的运行器）；
  - 第一次 smoke 因为 sbatch 里 `--bind` 参数和镜像路径之间漏了空格而失败，已加测试。
- 主报告页第 23 版加了 `#v5-det` 一节。下一步（AEGIS 闸门设计稿，或补更强的现成检测器）要用户决定。

## 2026-10-26 测试评估完成

- 用户 10-26 说"都改一下 然后做测试级"：先改了 3 处文档（`d964dfe`），再做测试评估。
- **结果**见 [V5_TEST_REPORT.md](V5_TEST_REPORT.md)，预注册读法 **`test_confirms`**：
  - 测试集（1,154 行反事实孪生，378 张图，4 条没见过的源）上，P 的程序判定 0.932 / 0.942 / 0.924（seed 713 / 714 / 715），A 0.503 / 0.501 / 0.500；
  - 3 个 seed 按行平均后 P − A = +0.432，图像 bootstrap 95% 区间 [+0.416, +0.449]；
  - 从验证到测试 P 只降 0.005–0.009；A 仍塌成常数；P 自己写的结论仍是随机水平（0.500 / 0.500 / 0.492）；
  - 测试自然（80 行）：P 0.975 / 0.950 / 0.950，A 0.913 / 0.813 / 0.413。
- 8 个作业全部 COMPLETED 0:0，没有重跑；6 个 adapter 的重新加载核对都是 32/32 逐字相同。没有正在运行的 CrashBench 作业。
- **测试集已经用过这一次，不要再评估、重跑或拿它选模。** 下一步（见下面"下一步"）都要用户另行决定。
- 主报告页第 22 版加了 `#v5-test` 一节和图 c13。
- 协议 [V5_TEST_PROTOCOL.md](V5_TEST_PROTOCOL.md)，冻结于 `5594ceb`（实现 `72d825b`，TF32 修正 `f071896`）；代码 `v5_test_contract.py`、`v5_test_run.py`，启动器动作 `launch_v5 test`。
- 6 个选定的 adapter 各评估一次（`test` 1,154 行，`test_natural` 80 行）。每个作业读测试行之前，要先在 32 行验证行上逐字复现训练时的输出。
- **坑**：评估时要打开 TF32（`allow_tf32=True`），和训练时一样。否则 P 的 fp32 LoRA 会让坐标差几个像素，第一次 smoke 就是因为这个没复现。打开后，在 PCIe 和 SXM 两种 A100 之间也能逐字复现。
- 运行目录 `/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs/20261026T173919Z_v5-test_5594cebeb616`；作业 5433550（preflight）、5433758–5433761、5433763、5433766（评估）、5433770（report）。
- 读数文件：运行目录下的 `TEST_READING.json`（report 作业写出）；每个 adapter 的记录在 `evaluations/<目标>/test.jsonl`、`test_natural.jsonl`，核对结果在 `RELOAD.json`、`COMPLETE.json`。

## 2026-10-26 加 seed 完成

用户 10-25 说"收尾下 然后让下一个agent开始干下一步"。接手的 agent 做完了已获批的加 seed：A、P 各补 714、715，只看验证。

- **预注册读法 `stable`**，见 [V5_SEEDS_REPORT.md](V5_SEEDS_REPORT.md)（协议 [V5_SEEDS_PROTOCOL.md](V5_SEEDS_PROTOCOL.md)，冻结 `ee20b20`，实现 `1a056d6`）：
  - P 的程序判定 0.940 / 0.947 / 0.933（seed 713 / 714 / 715），A 0.501 / 0.499 / 0.500；
  - 3 个 seed 按行平均后 P − A = +0.440，图像 bootstrap 95% 区间 [+0.427, +0.452]。
  - A 每个 seed 都塌成一个常数（713、714 几乎全判"不可行"，715 全判"可行"）；P 自己写的结论在 3 个 seed 上都是随机水平。
- 当时的下一步是测试集评估一次（已于 10-26 完成，见上一节）。
- 运行目录 `verifier_training/runs/20261026T001735Z_v5-seeds_ee20b2033465`，作业 5343157–5343162 全部 COMPLETED 0:0。没有正在运行的 CrashBench 作业。
- 主报告页已同步到第 18 版（新增 `#v5-seeds` 一节和图 c12）。
- **10-26 报告页白话版（第 19、20 版）**：用户说当前任务"好抽象"，又问"这个 p 程序判定是啥意思啊 我训练了 vlm 吗"，最后说"写清楚吧 然后这个收尾"。
  - 页首新增 `#now`"现在在做什么（白话版）"：用验证图 i1291 的两条规则从头讲一遍。内容有原图、标注图、为什么挪东西、A/B/P 的真实输出、结果和下一步。
  - 第 5 小节 `#now-program` 讲清楚：三组都是训练过的 Qwen2.5-VL-3B + LoRA；P 的"程序判定"是 VLM 写出四个坐标后，一段固定的几何代码判断是否隔开，不训练；P 自己写的结论在 3 个 seed 上都是随机水平（713、714 全写"不隔开、可行"，715 有 87% 写"隔开、不可行"），判定来自代码；P 比 A 多拿了位置监督。
  - 以后向用户解释 P 时用这套说法：模型负责"东西在哪"，代码负责"隔没隔开"。不要说模型自己学会了判断。
- 当时到此收尾；之后的测试评估见上一节。

## 2026-10-25 收尾

用户说："换吧 继续 然后这一步干完先收尾 反正要尽力跑通故事线"。S4 做完，本轮收尾。

**2026-10-25 用户："收尾下 然后让下一个agent开始干下一步"**：
- 下一步（加 seed）已获批，交给下一个 agent 去做；
- 测试集评估不在这次授权内，3 个 seed 都跑完后仍要用户确认。

### 故事线现状

| 步 | 做了什么 | 反事实孪生验证（1,538 行） | 报告 |
|---|---|---:|---|
| S3 | A 直接判断；B 先写"布丁、篮子在线靠机器人的一侧吗"再判断 | A 0.501，B 0.503（`both_shortcut`） | [V5_S3_REPORT.md](V5_S3_REPORT.md) |
| 拟合诊断 | B 在 16 / 64 行训练图上反复练，再看没练过的训练图 | 能背下来，但不迁移（`inconclusive`） | [V5_FIT_REPORT.md](V5_FIT_REPORT.md) |
| S4 | P 先写两个锚点、布丁、篮子在第 1 张图上的位置，程序判断是否隔开 | **P 0.940**，P − A +0.438 [+0.416, +0.461]（`positions_ground`） | [V5_S4_REPORT.md](V5_S4_REPORT.md) |
| 加 seed | A、P 各补 714、715，和 713 一起按行平均 | **P 0.940**（各 seed 0.933–0.947），A 0.500；P − A +0.440 [+0.427, +0.452]（`stable`） | [V5_SEEDS_REPORT.md](V5_SEEDS_REPORT.md) |

- **一句话**：
  - 在自然分布上训练、在挡住捷径的反事实孪生上测试，直接判断只有 50%；
  - 让模型先写出看得出来的事实（位置），再交给程序判断，是 94%；
  - 写"在哪一侧"这种关系事实不行（50%），模型学不会读那条看不见的线。
- **要如实说明的局限**：
  - 3 个 seed，只看了验证集；1 个任务组合、4 条验证源；
  - P 自己写的 `Separated` / `Verdict` 在 3 个 seed 上都是随机水平，结论来自程序；
  - 位置真值来自仿真几何；
  - 测试集没有评估。

### 本轮的运行目录（`/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs/`）

| 步 | 运行目录 | 冻结 commit |
|---|---|---|
| S3 | `20261025T055227Z_v5-s3_bdcf277164b6` | `bdcf277`（实现 `904ce55`） |
| 拟合诊断 | `20261025T153621Z_v5-fit_ea76dabed506` | `ea76dab`（实现 `1d9d099`） |
| S4 | `20261025T180525Z_v5-s4_20913c71c3f3` | `20913c7`（实现 `e651c3d`） |
| 加 seed | `20261026T001735Z_v5-seeds_ee20b2033465` | `ee20b20`（实现 `1a056d6`） |
| 测试评估 | `20261026T173919Z_v5-test_5594cebeb616` | `5594ceb`（实现 `72d825b`、`f071896`） |

### 代码入口（`verifier_training/`）

- **S3**：
  - `v5_contract.py`：配置、`validate()`（把测试行与 `TEST_FREEZE.json` 逐行比对后丢弃）、子集、查表、零样本提示、读法；
  - `v5_metrics.py`、`v5_run.py`、`v5_zero_shot_32b.py`。
- **拟合诊断**：`v5_fit_contract.py`、`v5_fit_run.py`。
- **S4**：
  - `v5_pos_contract.py`：投影、P 的目标、解析、程序、读法；
  - `v5_pos_run.py`：preflight、train、report；
  - `v5_pos_zero_shot.py`：零样本定位，3B 和 32B 都在 `qwen-vlm` 环境里跑。
- **加 seed**：
  - `v5_seeds_contract.py`：配置（原样嵌入 S3、S4 的配置）、新 seed 的样本顺序、3 个 seed 的行平均和图像 bootstrap（`seed_average`）、读法 `seeds_reading`；
  - `v5_seeds_run.py`：preflight、train、report；训练直接调用 `v5_run.train_arm`（A）和 `v5_pos_run.train_p`（P），这两个函数是 S3、S4 的 `train` 原样抽出的函数体。
- **测试评估**：
  - `v5_test_contract.py`：配置、6 个目标和两条线、旧文件清单、adapter 冻结、重新加载核对、读法 `test_reading`；
  - `v5_test_run.py`：preflight、evaluate、report；只有这里的 `context` 会读测试行。
- **启动器** `launch_v5.py`：动作 `create`、`pilot`、`stage`（S3），`fit`，`s4`，`seeds --s3 --s4`，`test --s3 --s4 --seeds`；GPU 作业时限 8 小时。
- **测试**：`verifier_training/tests/test_*.py` 共 164 个（V5 测试评估新增 11 个）。跑法见下面的"规矩"。
- **选定的 adapter**（测试评估要用）：S3 的 `A_seed713/selected_adapter`，S4 的 `P_seed713/selected_adapter`，加 seed 目录里的 `A_seed714`、`A_seed715`、`P_seed714`、`P_seed715` 下的 `selected_adapter`（选的分别是第 448、150、448、448、448、300 次更新：A 713/714/715，P 713/714/715）。

### 下一步（每项都要用户先确认）

- ~~测试评估一次~~：2026-10-26 完成，`test_confirms`，见 [V5_TEST_REPORT.md](V5_TEST_REPORT.md)。
1. 把 P 接到 AEGIS 前面当闸门，需要单独批准控制器改动。
2. P 自己写的结论在 3 个 seed 上都是随机水平，可以试着教模型自己算出 `Separated`。
3. 扩到其他任务组合（goal、spatial），之前暂停的补源也需要另批，要新建验证和测试源。
4. 整理成论文主线（V5 验证、加 seed、测试三步结论一致）。
- S3 的 B 和零样本模型没有在测试集上评估；要评估需要另批并事先写定。

### 坑

- **投影**：用每张图 `units/<图像编号>/CAMERA.json` 里的第一个视角。学生图是左右翻转过的，公式是 x = (1 − nx)/2·448，y = (1 − ny)/2·448。`RENDER_GEOMETRY.json` 里没有相机参数。
- **`qwen-vlm` 环境**：要设 `LD_LIBRARY_PATH=/projects/p33100/siosio/envs/qwen-vlm/lib`，否则 Pillow 加载失败。
- **坐标约定**：Qwen3-VL 用 0–1000 的相对坐标，Qwen2.5-VL 用输入图的绝对像素。
- **字段顺序**：S2 的行按键名排序写出，事实字段的顺序是字母序。
- **token 合并**：`]` 和换行会合并成一个 token，所以坐标值不能用 `encode` 的事实值区间来标记。
- **32B 的输出格式**：写事实时每行末尾带两个空格；零样本定位时常回答 "There are none."。

### 规矩

- 提交身份用 `git -c user.name=hanshuo-shuo -c user.email=80494218+hanshuo-shuo@users.noreply.github.com commit`。本机配置是另一个邮箱，不要用。
- 测试集（`test`、`test_natural`）已在 2026-10-26 评估过一次，以后不再评估，也不用于选模或调参。
- 付费 API、AEGIS 闸门、新数据，都要用户另行确认。
- 跑测试：`PYTHONPATH=/projects/p33100/siosio/crashbench_safelibero/verifier_training/assets/20261010_d94c245/packages:. /projects/p33100/siosio/envs/openpi/bin/python -m unittest <verifier_training/tests 下的模块>`。

## 用户的决定（按时间）

| 日期 | 用户原话 | 含义 |
|---|---|---|
| 10-24 | "推送+开始第一步 故事是最重要的" | V4 收口批不做；转向 V5，一步步来，故事优先 |
| 10-24 | "做吧 然后采用v5计划" | 做第 2 步之后的设计修改，采用 V5 计划；付费 API 仍需另行批准 |
| 10-24 | "3吧" | 用反事实布局（挪动锚点）构造测试 |
| 10-24 | "限定一下吧 然后暂时不要扩展太多了 就是故事都还没走通呢" | 只保留能产出孪生的任务组合；不补源、不扩规模 |
| 10-24 | "我想做1"，"这一轮先收尾" | 先用 object/II/task1 一个任务走通故事；本轮到此为止，没有正在运行的作业 |
| 10-24 | "训练集只用 object 一个任务" | S2 训练只用 object/II/task1 训练源，不加其他 5 个组合的自然行 |
| 10-24 | "哦哦有道理 继续吧 ……都加" | 三项都采用：训练加入训练源的反事实布局；B 沿用 V4 写法（复述连线、物体靠机器人一侧、篮子靠机器人一侧、Separated、Verdict）；S1 渲染加正俯视图。S1 协议：[V5_S1_PROTOCOL.md](V5_S1_PROTOCOL.md) |
| 10-25 | "要改" | 把用错身份的提交 c732956 改写为 d38420c（树相同）并强制推送 |
| 10-25 | "继续做s2 一起更新" | 做 S2；主报告页等 S2 完成后一起更新（第 14 版） |
| 10-25 | "可以 开始写然后 这个对话框很长了 我再递交给下一个你" | 开始 S3：写协议和代码；本会话写完协议后交接 |
| 10-25 | "接着做 V5 的 S3" | 新会话实现 S3 代码、跑 smoke、冻结协议并运行 pilot |
| 10-25 | "做一下吧" | 做 S3 报告建议的拟合诊断（B 在 16 / 64 行训练图上，只读训练集） |
| 10-25 | "换吧 继续 然后这一步干完先收尾 反正要尽力跑通故事线" | 换成"先写位置、再用程序判定"（S4）；做完后本轮收尾 |
| 10-25 | "收尾下 然后让下一个agent开始干下一步" | 本轮收尾；下一步（A 和 P 补 seed 714、715，只看验证）交给下一个 agent；测试评估仍需确认 |
| 10-26 | "都改一下 然后做测试级" | 修正 3 处文档；写测试协议和代码、冻结后在测试集上评估 6 个选定的 adapter 各一次 |
| 10-26 | "可以" | 不指定 GPU 型号（smoke 已证明打开 TF32 后不同型号逐字复现） |
| 10-26 | "现在收尾下 我接下来接手知道怎么继续" | 测试评估后收尾；本文开头写接手指南，之后每一步都等用户决定 |
| 10-26 | "先做1" | 做检测器基线：AEGIS 的 GroundingDINO 加 P 的程序，只看验证，只做推理（读法 `training_needed`） |
| 10-27 | "我还是觉得应该是先继续原来的想法 但是2可以保留下" | 主线回到生成式 verifier（模型自己下结论）；P 只作诊断；闸门或系统保留为备选。此前用户问"p似乎有点作弊"：代码就是生成标签的规则，P 的 94% 主要衡量定位 |

## 故事（当前版本）

在自然场景里，约束是一句指向真实物体的话，在机器人做到一半时加入（"别让任何东西越过 A 和 B 的连线"）。

- 训练数据用自然布局。这里只看文字或"规则 × 进度"就能猜对大部分标签。
- 测试用反事实布局：挪动机器人碰不到的锚点，同样的动作带回放仍是合法见证；同一张图配两条规则，一条分隔、一条不分隔。
- 预期：直接判断（A）学到的是捷径，在反事实孪生上失败；先写关系事实的模型（B）必须看图，能够推广。
- **实际结果（10-25 – 10-26）**：A 如预期是 50%；但 B 也是 50%，写"在哪一侧"没有让它去看图。换成先写位置、再由固定代码判定（P）后，验证 94%、测试 93%。见上面的接手指南。

这延续了 V4 的主线（关系事实监督 vs 直接判断），也对应 HazardArena 的"孪生几乎一样时，拒绝训练退化成表面匹配"。走偏检查见主报告 <https://claude.ai/artifact/Ya7FQbZC9yEuNNyYhPBHHT> 的"下一阶段：V5"一节（第 13 版）。

## 已完成（结果只读）

| 步 | 报告 | 运行目录（`/projects/p33100/siosio/crashbench_safelibero/verifier_dataset/`） |
|---|---|---|
| 第 1 步 CPU 几何摸底 | [V5_STEP1_REPORT.md](V5_STEP1_REPORT.md) | `20261024T201429Z_v5-scan_1d8a7cd84182` |
| 第 2 步 渲染与盲看 | [V5_STEP2_REPORT.md](V5_STEP2_REPORT.md) | `20261024T205732Z_v5-render_35fc6c5029bf` |
| C0 修正后重新回放 | [V5_C0_REPORT.md](V5_C0_REPORT.md) | `20261024T212428Z_v5-trace_0b0573ded5e0` |
| 反事实布局 smoke v1–v3 | [V5_LAYOUT_PROTOCOL.md](V5_LAYOUT_PROTOCOL.md) | `…220018Z…3229b5377671`、`…221104Z…5fac839b7681`、`…221853Z…7c9958eb72a7` |
| S0 筛查 | [V5_S0_REPORT.md](V5_S0_REPORT.md) | `20261024T224456Z_v5-screen_31bbc56ab8e2` |

S0 的关键数字：
- 38 条源、6 个任务组合可用（训练 21、验证 9、测试 8 条源）。
- 在"任务 × 规则 × 进度段"组内配平后，能挡住捷径的测试几乎全是 object/II/task1：验证、测试各约 87 个孪生单元、各 4 条源。

## 下一步：最小故事闭环（每步开始前向用户确认）

1. **S1：补布局，再盲看一次**（2026-10-25 完成，见 [V5_S1_REPORT.md](V5_S1_REPORT.md)）
   - 144 个布局全部合法；验证、测试各 392 / 384 个严格孪生单元，读法 `s1_supply_ok`。
   - 两轮盲看在判定、物体侧、篮子侧三项上都是 40/40；读法 `relation_visible`、`facts_readable`。
   - 俯视图没有带来提升，而且 13/20 个单元的布丁被夹爪挡住。S2 用两张斜视图。
   - 运行目录：`20261025T002929Z_v5-s1-layout_c7329566dbbb`（`c732956` 后来改写为 `d38420c`，内容相同）、`20261025T005258Z_v5-s1-render_d38420c96208`。
2. **S2：建数据**（用户 10-25 "继续做s2 一起更新"；协议 [V5_S2_PROTOCOL.md](V5_S2_PROTOCOL.md)；2026-10-25 完成，见 [V5_S2_REPORT.md](V5_S2_REPORT.md)）
   - `s2_dataset_ready`：训练 1,792 行，验证 1,538 行，测试 1,154 行，另有自然诊断各约 80 行；丢弃 0 行。
   - "规则 × 进度"查表：训练 0.760；验证、测试 0.500；自然诊断 0.747 / 0.912。
   - 数据：`/projects/p33100/siosio/crashbench_safelibero/verifier_dataset/20261025T013358Z_v5-s2_d1d8d4129223/dataset.jsonl`。`TEST_FREEZE.json` 冻结了测试，只在 S3 最后评估一次。
   - **训练**：只用 object/II/task1 的 10 条训练源，自然布局加上它们的反事实布局（用户 10-24 决定）。
   - **验证和测试**：对应切分源的反事实静置孪生，在组内配平。在任何模型看到之前冻结测试集。
   - **B 的中间事实**（用户 10-24 采用）：复述连线，"物体在连线靠机器人底座的一侧吗""篮子呢"，Separated = 两者不同，Verdict。机器人底座离连线 < 10 cm 的行排除。S1 的盲看检验这两个事实看不看得出，写进 S2 协议后再生成目标。
   - **10-24 只读摸底**（C0 回放和 S0 布局，不跑作业；脚本未入库，正式采用时写进 S2 协议）：
     - object/II/task1 的 18 条源自然布局完全相同：杂货位置的标准差 0.0 cm，篮子 0.6–1.0 cm；只有障碍物种类不同（5 种，都在约 (0, 0.10)）。所以"只用自然布局训练"等于只见过一个布局。
     - 以机器人为参照的两个事实（物体侧 T、篮子侧 G），在自然训练行上按"规则 × 进度"查表能拟合 0.985 和 1.000，到反事实验证行上是 0.517 和 0.462。和判定一样都是捷径，只用自然布局训练时 B 也没有理由看图。
     - 加入训练源的反事实布局后，查表上限：判定 0.767、T 0.698、G 0.674（只用自然布局时 0.985 / 0.985 / 1.000）。
     - 机器人底座（约 (-0.60, 0)）离连线不到 10 cm 的只有 `alphabet_soup|orange_juice` 一条规则（5.7 cm，自然行 536/3216）；反事实布局里有 456/6404 行。
     - "图上标位置 + 程序判定"这一候选：用第 2 步渲染的相机参数，按物体可见中心的投影在图上判断哪一侧，余量 ≥ 5 cm 时只有 60/74 与真实标签一致；改用地面接触点的投影是 73/74。斜视图里的视差很大，这种事实要换俯视图或改标接触点。
   - 渲染两张对向图（gengpu A100，排除 qgpu2014）。
3. **S3：A 和 B 的 pilot**（用户 10-25 批准开始；协议 [V5_S3_PROTOCOL.md](V5_S3_PROTOCOL.md)；2026-10-25 完成，见 [V5_S3_REPORT.md](V5_S3_REPORT.md)，读法 `both_shortcut`）
   - 各 1 个 seed，只看验证集，沿用 `verifier_training/` 的管线和损失设置；要把 `fd1000_contract` 里 V4 的事实键和强制前缀换成 V5 格式（`Line:`、`Target on robot side`、`Goal on robot side`、`Separated`）。
   - 加零样本 VLM 对照：本地 Qwen，不花 API。
   - 看到方向后再定 3 个 seed 和测试评估。测试集只评估一次。

**暂缓**：补源（C1–C5、nominal 重新执行）、挪动固定件、AEGIS 闭环、付费 API。

## 代码入口（`verifier_dataset/`）

| 文件 | 作用 |
|---|---|
| `v5_geometry.py` | 锚点规则（排除终点所属物体、相像物体）、连线标签与余量（5 cm + 0.1 × 抬起高度）、配平与孪生统计、选样 |
| `v5_scan.py` | 第 1 步：初始状态的物体几何 |
| `v5_trace.py` | 带物理哈希核验的 CPU 回放，记录每个 checkpoint 时刻的几何（自然布局行的来源） |
| `v5_layouts.py` | 反事实布局：叠放物体整体挪动、按设计摆放、回放见证核验、筛查与捷径迁移读数 |
| `v5_render.py` | 两张对向俯视图加诊断图（第 2 步用；S1 要接上布局） |
| `v5_launch.py` | 启动器，阶段为 `scan`、`render <scan>`、`trace`、`layout <trace>`、`screen <trace>` |

测试：`/projects/p33100/siosio/envs/openpi/bin/python -m unittest verifier_dataset.tests.test_v5_geometry verifier_dataset.tests.test_v5_trace verifier_dataset.tests.test_v5_layouts verifier_dataset.tests.test_v5_render`，共 31 个。

## 容易踩的坑

- **启动器要求工作区完全干净，代码已发布。** 用 `/projects/p33100/siosio/envs/openpi/bin/python -m verifier_dataset.v5_launch …` 启动，提交时用 hanshuo-shuo 的身份。
- **Quest 会复用作业号。** `sacct -j` 可能列出别人的旧作业；启动器里的 `completed()` 已按 p33100 账号过滤，手动查询时加 `-u $USER` 并看 Account 列。
- **容器里是 Python 3.8**（`aegis-py38.sif` 加 `envs/aegis_sim`），不要用 3.9 以后的语法。
- **盲看纪律**：先只读 `REVIEW_SHEET.json` 和学生图，判断写进 `verifier_dataset/reviews/` 并提交，之后才读 UNITS.json、RENDER_GEOMETRY.json 和诊断图。
- **报告页**：
  - **主报告**是 <https://claude.ai/artifact/Ya7FQbZC9yEuNNyYhPBHHT>，源文件 `docs/generative_verifier/PHASE2_REPORT.html`，公开链接，改动立刻可见。第 13 版新增 `#v5`"下一阶段：V5"一节，和上周的结论、后续工作逐条对照；第 18 版（10-26）在 `#v5-s4` 后加了 `#v5-seeds`（图 c12：3 个 seed 的 A、P）。中英双语用 `data-l="zh"` / `data-l="en"`，两种语言都要改。
  - 精读页 DJPGiauxhPAC8BYE4L14Y5（`PAPER_READING.html`）第 5 版也有一节"本周进展"。那是用户一开始给错了链接时加的，是否保留由用户决定。
