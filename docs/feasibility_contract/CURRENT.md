# Feasibility evidence pilot — 2026-10-03

The explicitly authorized 2026-10-02 design/implementation/Quest pilot is complete
as an **infrastructure and witness pilot**. It does **not** complete the intended
feasible/infeasible reachability benchmark: no nontrivial unreachable state with
an individually legal goal was certified. No follow-on run, training, controller
change, threshold search or sample expansion is queued.

Protocol: [experiments/feasibility_contract/README.md](../../experiments/feasibility_contract/README.md).
The original SafeLIBERO reproduction and 1209 diagnostic executions remain intact.

## Results and evidence scope

Core: seven states in six layout groups. Five layouts were preselected by index
before execution and held out from this pilot's development: milk I/task2/init6,
7,8 and chocolate pudding I/task1/init6,7. All five have actual legal, contact-safe
native successes with synchronized-predicate agreement. They are not claimed
unseen by pi0.5 pretraining or the old full benchmark.

The two external Object5 r0/t252 and r3/t252 checkpoints share one exposed layout.
Both fixed references failed within 300 actions from each checkpoint while
remaining contact-safe. Both labels are **unknown**, not infeasible. Native
success is point containment and does not imply stable placement or release.

There are 14 core reference executions, not 14 independent states. Eight native
task successes, 14/14 contact-safe executions and 14/14 original displacement-
proxy-safe executions are reported separately. Evidence labels: feasible 5,
infeasible 0, unknown 2; coverage 5/7 (71.4%), unknown 28.6%. No confidence or
generalization claim is made from these selected states.

The center reference alone certifies 3/7 at a 300-action query budget. Center
fails on both chocolate layouts; unchanged side reference succeeds at 217 and
221 actions. Thus those two center failures have concrete counter-witnesses to
any infeasibility interpretation. The two-reference library certifies 5/7 at a
600-action budget (300 each). Its equal-total-300 tier (150 each) certifies 0/7;
these costs are explicit and are not presented as equal-budget method gains.
Zero determinate errors is evidence conformance, not independent predictor accuracy.

Development smoke only: one exposed layout, four cap settings, 1 feasible / 2
definition-based infeasible sanity controls / 1 unknown near-boundary candidate.
The negative controls have disjoint goal and safety sets and are **excluded from
the core research pilot**. They are not physical reachability results.

All 17 completed executions (14 core, two smoke references, one exact smoke
replay) passed persisted contact/predicate/hash checks and final pose audits.
Final audit reads saved physics and performs forward calculations only, no actions.
It exports all body/site positions and orientations from the same state. The new
Object5 r3/t252 center final target center exceeds the native site's upper-z bound
by 6.524894 mm; side exceeds it by 4.577342 mm. These describe these newly executed
continuations only, not a retrospective submillimeter inference from old logs,
and do not prove recovery impossible.

## Restore and action admissibility

Original AEGIS prefixes contain small gripper-only overages (max absolute raw
command 1.014058 for r0; 1.008402 for r3). The first pilot stopped at this gate.
The installed PandaGripper source SHA is
`a3760a8fc4599fa6f79914304b82466f3c1bda1a945943ca693bb57f00dc59eb`;
its format_action uses `np.sign(action)`. Recovery retains raw/applied command
pairs and clips only gripper overages to a sign-identical legal value. It rejects
any arm overage or changed gripper implementation.

All four replayed prefixes exactly match saved qpos/qvel/ctrl, warmstart, applied
forces, mocap, act and original arm-controller fields at 0/27/77/152/252. Original
policy RNG/queue metadata is retained but a deterministic reference replaces that
policy. We do not claim to resume the learned policy stream or reproduce its RGB.
Ten completed held-out executions were hash-inherited unchanged, not rerun.

## Jobs, repairs and immutable outputs

All jobs used p33100/short, 4 CPUs, 16 GiB, no GPU. No API calls or key reads,
no training. Smoke cap 15 min; core pilot cap 30 min; external recovery cap 15
min; each forward audit cap 5 min. Actual sum of job wall times is 12m39s, or
0.8433 allocated CPU-hours. Completed runs used 5288 command steps including
prefixes and exact replay, excluding settling; the aborted prefix added two.

| Job | Source | Outcome | Elapsed | Role / reason |
|---|---|---|---|---|
|8350705|62e7bd255966|FAILED/1:0|19s|OSMesa binding import failure, before actions|
|8350760|dd6324068d56|FAILED/1:0|27s|native registration decorator returned None; before actions|
|8350819|0cbf26b9a554|COMPLETED/0:0|2m11s|smoke, exact replay and independent evidence audit|
|8350901|0cbf26b9a554|FAILED/1:0|4m43s|ten held-out runs complete; stopped at historical raw-action gate|
|8351274|98689091bbae|COMPLETED/0:0|4m12s|only four missing natural continuations; inherited ten runs|
|8351621|43f4345f0865|FAILED/1:0|20s|forward audit reader lost empty array shape; no new actions|
|8351701|279264b51e6f|COMPLETED/0:0|27s|all 17 final synchronized pose audits and report|

All roots below `/projects/p33100/siosio/crashbench_safelibero/feasibility_contract/`:

* `20261003T000834Z_smoke_62e7bd255966` — preserved first failure.
* `20261003T001033Z_smoke_dd6324068d56` — preserved second failure.
* `20261003T001208Z_smoke_0cbf26b9a554` — accepted smoke.
* `20261003T001508Z_pilot_0cbf26b9a554` — ten accepted held-out runs and failed prefix.
* `20261003T002218Z_external_recovery_98689091bbae` — composite core decisions and four natural runs.
* `20261003T002748Z_final_audit_43f4345f0865` — preserved reader failure.
* `20261003T002937Z_final_audit_279264b51e6f` — **final report and final-pose artifacts**.

Each root has receipt.json, SOURCE_COMMIT, frozen source/source.sha256 and
`slurm_<job>.log`. Scientific source, seed, BDDL, initial state, XML, actions,
physics/controller/RNG snapshots, all integration-step contact samples and
per-run hashes remain on Quest. Local deliverables:
`results/feasibility_contract/final_20261003/`, 145 downloaded files (1,552,221
bytes), every file SHA256 matched the remote bytes; DOWNLOAD_MANIFEST.json
records exact paths/hashes, and sacct.psv records complete job/step accounting.

Implementation commits: 953d263 (first implementation), 62e7bd2 (pre-submission
scope correction), dd63240 (GLX headless imports), 0cbf26b (module source audit),
9868909 (audited sign-equivalent prefix recovery), 43f4345 (final pose audit),
279264b (empty-array shape restoration). Original reference/geometry/reset-forward
sources were unchanged. 57 local unit checks pass (7 new, 50 existing), along
with Python syntax compilation, Bash syntax and diff checks.

## Remaining research requirement and next decision

Fixed visual barriers, apparent out-of-reach positions and short horizons are not
certificates here. Movable receivers/obstacles, unrestricted three-dimensional
routes, dynamic object motion and soft-contact penetration require additional
proof obligations. OSC clipping is not an object-speed bound.

A possible next explicitly synthetic slice: target and a receiver constrained to
remain on opposite sides of a full separating keep-out curtain; legal endpoints
on both sides; signed-side transition monitoring rejects discrete tunneling;
the paired positive opens a gate and must have a real native-action witness.
This changes the safety contract and receiver constraints, so it is not a claim
of impossibility under unchanged public SafeLIBERO. No such slice was implemented
or submitted. Decide whether that synthetic topological contract fits the paper;
otherwise keep native negatives unknown and do not claim a balanced certified
feasibility benchmark. A separate input-restricted judgment method and held-out
evaluation would still be needed beyond the privileged evidence verifier.
