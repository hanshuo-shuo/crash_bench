# 外部恢复机制：从论文名单收敛到可运行对照

检查日期：2026-09-13。**SPR 的外部学习恢复接口已在 Quest 跑通。**
普通任务指令解出 7×7、学习回退指令解出 8×7 的有限动作矩阵；完整动作中的
未知 token 数均为零。两次应用层模型查询、零环境动作、零新来源。
[完成记录](evidence/spr_preflight/complete.json)。这推进了工程可用性；恢复
净收益仍需单独验证，当前不计入论文的恢复性能结果。

## 为什么现在选择 SPR

[官方项目](https://tingjundai.github.io/SPRVLA/)链接到
[代码仓库](https://github.com/TingjunDai/SPRVLA)和
[Spatial 模型](https://huggingface.co/SPRVLA/libero_spatial)。固定代码版本
`d57e4b81ebdcacea574b68be29d61ba04cdc7051`，模型版本
`13e02cf4d1e8cfc77c5fffbd8502289d853fcf07`；四个权重文件共
16,238,922,592 字节。完整文件清单与散列在 `spr_contract.json`。

其恢复能力通过训练得到：用构造的逆向示范学习回退，再在进度异常时短暂切换指令。
这提供了一个独立于手写 Detour 的机制。需要同时保留尺度：原论文的完整 SPR 相对
去掉 Rewind 的平均增益约 1 个百分点；Spatial 的对应值是 92.4% 与 92.6%，并非
该子集上已知有很大的回退增益。不能把完整模型对 MolmoAct 的提升全部归给回退。
[原论文第 III-C、IV-E 节和表 VI](https://arxiv.org/html/2603.09292v2)。

我们对已发布实现的代码核查还发现：当前 `PromptManager` 使用子任务计数异常，
源码没有同时实现论文描述的八步轨迹停滞触发。后续应明确采用发布代码的哪一个
接口。优先检验固定的学习回退操作，再单独讨论触发器；不能把一个移植后的模块
实验写成整套 SPR 的官方 benchmark 复现。
[冻结的官方执行代码](https://github.com/TingjunDai/SPRVLA/blob/d57e4b81ebdcacea574b68be29d61ba04cdc7051/experiments/libero/run_libero_eval_vllm.py)。

## 其余候选的实际状态

| 候选 | 本轮核实的可用项 | 当前缺口 / 判断 |
|---|---|---|
| LIBERO-RECOVER | 公开评测代码、场景/专家数据下载入口 | 启动脚本中的恢复模型仍是作者本地路径，尚未核实对应公开权重；适合作为后续场景资产，不等同于已有可运行恢复器 |
| B2FF | 原始论文、在线触发与受控触发说明 | 未找到经核实的官方完整代码/checkpoint；其未来图像条件接口不能直接套在普通 OpenVLA/pi0 上 |
| CoRe | 原始论文、推理时重对齐与物理恢复描述 | 本轮未找到经核实的完整运行包；操作权限和物理代价仍需与本项目对齐 |
| FAR | 公开 LeRobot 实现、LIBERO/真实机械臂例程 | 主要针对远程部署中的抓取/滑落与重试；可作为备选，但当前优先接通有公开学习回退权重的 SPR |

LIBERO-RECOVER 核查记录固定在
[README](https://github.com/liulin815/LIBERO-Recovery/blob/4027d2bb4bef3902575d53fad1cfa399bbb580bb/README.md)及
[启动脚本](https://github.com/liulin815/LIBERO-Recovery/blob/4027d2bb4bef3902575d53fad1cfa399bbb580bb/examples/LIBERO/eval_files/eval_libero_custom_scene.sh)，
散列保存在 `evidence/manifest.json`。其原生评测还包含初始物体扰动和夹爪条件，
直接复用时必须保留来源分组，不能将同一失败轨迹上的多个状态作为独立来源。

其他核对入口：[B2FF](https://arxiv.org/html/2606.09258v1)、
[CoRe](https://arxiv.org/html/2608.14822v1)、
[FAR 官方仓库](https://github.com/HyuanTan/lerobot_far)。
“未找到经核实的运行包”描述此次检索结果，不是断言公开实现不存在。

## 已完成的最小工程检查

只使用历史 fitting 锚点 e18 的已保存原始双相机观测。采用 SPR 的官方图像处理、
提示构造、模型实现和动作解码，对正常任务指令与回退指令各执行一次模型查询，
检查是否产生有限的七维动作序列。最多两次查询、零环境动作、零新物理来源。
这不是恢复率实验，也不评价该选中案例是否成功。

CPU 准备阶段核验模型散列并建立独立 Python 3.11 环境。GPU 原申请一张
A100、35 分钟；开始前为缩短排队调整为同分区兼容单卡、15 分钟上限，
实际使用 H100 80GB，仍在合约最多 35 分钟之内。使用官方 README 指定的 vLLM 0.8.5 / Transformers 4.52.1；为节省
工程初始化资源，使用 eager 模式及 0.85 显存比例，这些运行设置不应被当作原论文
性能复现合约。所有文件和安装日志保存在新的 commit/run 目录中。

这个检查已经通过。下一项科学工作是固定 SPR 的 Base/回退操作、预算和来源面板，
应用相同 A 选/B 估/C 验协议。必须同时报告学习回退是否完成、Base 是否本来就会
完成、回退代价及正常任务保持。新来源 Detour 结果不会被用来挑选对 SPR 有利的
未来终局。当前不预先认定外部结果会支持测量高估，也不把工程接通算作论文主结果。

## 补充核查：强恢复能力与接入成本应分开判断

[VoLoAgent 官方代码](https://github.com/NVlabs/VoLoAgent)公开了 VLM 编排、
VLA 与抓取／放置工具的组合，也包含 LIBERO 客户端。其原生 RoboVoLo 四组
消融中，完整系统为 41.80%，只保留 VLA 的编排版本为 34.97%；这些数值
不能移作 CrashBench 对照。接入还需感知、深度与运动规划服务及 VLM，
比当前 SPR 检查更重，但可作为更强物理恢复能力的合作候选。
[原论文表 2、附录 H](https://arxiv.org/html/2606.07723v1)。

另外核对了三项资产边界：

- [ReCoVLA 官方项目](https://hhd000exe.github.io/recovla/)给出学习残差恢复
  的仿真与实机结果，但该资源页仅链接论文／补充材料，本轮未核实公开的
  完整残差策略运行包。原生 Fetch／Behavior-1K 接口也需单独对齐。
- [HELM 原论文](https://arxiv.org/html/2604.18791v1)包含记忆、动作验证、
  回退与前向恢复变体。本轮未核实官方可运行权重，不能仅依据提示语描述
  就将其当作原有 OpenVLA 的即插即用恢复器。
- [ViFailback 官方仓库](https://github.com/x1nyuzhou/ViFailback)公开了失败
  诊断及纠正符号的推理／绘制接口；这本身不等于已经接通低层恢复动作。

这轮没有安装或运行这些额外系统。优先级仍是完成 SPR 接口检查，并在
选择下一项性能实验时要求原生场景的正能力对照。仅有漂亮 demo、可下载
模型或一个总成功率提升，都不自动给出该恢复模块的可重复净收益。

## 本次接口结果及执行注意点

CPU 作业 6246660 用时 6 分 42 秒，GPU 作业 6246661 用时 1 分 38 秒，
均正常退出。执行代码提交为 `6b91cee1d337e0bc28e2f792b8ccac44fe44c868`；
Torch 2.6.0、Transformers 4.52.1、vLLM 0.8.5、NumPy 1.26.4。
模型、tokenizer、旧 e18 输入 bundle 和图像处理代码均经过固定版本核验。
[版本与设备](evidence/spr_preflight/inference_provenance.json)。

普通查询返回七个完整动作，尾部还有一个未闭合的动作片段；回退查询返回
八个完整动作。我们保留原始生成文本，没有补齐或捏造第八个普通动作。
官方执行循环本身按实际返回的 N 个动作迭代；后续执行器应记录截断和真实
动作数，不能假定每个 chunk 总是八步，也不能把三次回退查询等同于三步
物理动作。本次通过的判据只涉及完整解码动作，不保证整段生成文本完整。
[普通输出](evidence/spr_preflight/normal.json)、
[回退输出](evidence/spr_preflight/rewind.json)。

原始权重、独立环境与完整安装日志保存在 Quest 的本项目资产目录，未修改
现有 OpenVLA 或 pi0 环境。此次没有调整生成上限去追求更好输出，也没有
将接口结果扩写成已复现 SPR 的任务完成率。
