# Legacy — superseded edge-bowl pilot (2026-06-21)

第一版 Phase 1 pilot 的产物。方法:把**目标黑碗推到桌沿**(unsafe-terminal),
期望"正常抓取"会把它碰下桌。结果 **crash_rate = 0%,全 safe_abort**。

诊断(见 [`../../crashbench/PHASE1.md`](../../crashbench/PHASE1.md)):把目标碗挪走后
OpenVLA 进入 OOD,根本不去碰那个碗 —— **危险物不在 VLA 的 nominal 行为路径上**,
所以测不出 crash。这是个真实(诚实)的负结果,故归档而非删除。

- `pilot_edge.json` — 4 个 edge 场景的闭环结果(0% crash)。
- `pilot_edge_videos/` — rollout 视频(可见 OpenVLA 去够中间盘子、不碰边缘碗)。
- `phase1_build_edge/` — 桌沿扫描的中间渲染图(gitignored)。
- 对应场景文件:[`../../scenarios/legacy/`](../../scenarios/legacy/)。

**取而代之的方向**:env-collision(README §4.3 cat-1)—— 让危险落在 VLA 必经路径上
(gripper 下扎桌面 / 撞障碍),崩溃判据用 `contact_force`。
