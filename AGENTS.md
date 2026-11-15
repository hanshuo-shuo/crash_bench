# CrashBench / SafeLIBERO agent instructions

## Start here

1. Read docs/CURRENT.md (one page: where the project is, what is next, and a short glossary).
2. Read docs/generative_verifier/HIL_METHOD_PLAN.md, section 0 first. It is the active plan and the hand-off.
3. Before any Quest work read QUEST_WORKFLOW.md and setup/README.md.

This file was shortened on 2026-11-12. It keeps the standing rules, what is approved now, and one line per completed
study. You do not need the archives to start work. The full dated record of every earlier authorization, in the user's own
words, is docs/AUTHORIZATION_HISTORY_until_20261112.md (a verbatim copy of this file before shortening) and
docs/STATUS_HISTORY_until_20261016.md; consult them only when the exact scope of a past authorization matters, and in that
case the archived wording governs. docs/REPRODUCTION.md covers the SafeLIBERO baseline.

## Standing rules

**Approvals.** Work only inside a dated user authorization. Each of these needs the user's go-ahead unless an entry under
"Approved now" covers it: a new GPU stage or batch, a new runner or controller, new data collection or layouts, any
training, a paid API call, a checkpoint download, extra seeds, a second model, an AEGIS controller change or gate.
The 2026-09-26 boundary still holds outside later authorizations: no new learned intervention, controller, benchmark
task, confirmatory study or threshold search. Record every new authorization in this file with the date, the user's words
and the scope. The old experiment plans on codex/archive-crashbench-20260926 are historical, not a run queue.

**Completed work is read-only.** Do not rerun a completed batch. Keep historical raw results recoverable. A dated
correction note may be added at the top of a completed report; its numbers and frozen readings are not changed.

**Locked data.**
- The V4 locked test split was scored once (Phase 2). Do not score it again.
- The V5 test split was scored once (2026-10-26). Do not score it again.
- SafeLIBERO test sources episode0/8/9/15 stay untouched.
- The 896-row V4 set may be used as development data (user decision 2026-10-16); its failed shortcut gate stays
  disclosed and is reported as a baseline.

**Protocols.** Readings are frozen in the protocol before the formal batch is submitted (after the smoke where the
protocol says so) and are not changed afterwards. Stop and ask the user if a smoke or gate fails for a non-code reason.

**Quest.** Before any Quest SSH use read QUEST_WORKFLOW.md and setup/README.md, then run scripts/quest_sync.sh check.
Only $HOME/crash_bench / hanshuo-shuo/crash_bench is the valid project checkout. Do not touch another project's files or
jobs. Submit only a clean, tested, published commit through scripts/quest_sync.sh. Use account p33100, GPU partition
gengpu, CPU partition short. No heavy work on login nodes; only downloads, Git and lightweight checks there. Compute nodes
are offline: stage assets first. Never force pull, hard reset or rsync source. Every new run needs a unique output root,
code/upstream commit, configuration, seed and Slurm provenance. Never change or sync the Quest source checkout while any
CrashBench job depends on it; check the queue first. At most two GPUs at a time.

**OpenRouter.** The user has an OpenRouter API key and wants to be notified when it is needed. Never print, commit or
place that key in a job log. Do not silently substitute another VLM. Budget: $100/week in total; baseline reproduction
within $5/week, the other $95 is for the user's method. Before any bulk API worker, inspect usage metadata, estimate the
batch and add a spend guard. No provider-side key cap has been configured. No paid API call without separate approval.

## Approved now (as of 2026-11-15)

The order of work and the details are in docs/generative_verifier/HIL_METHOD_PLAN.md section 0.

**1. Census and dose reports: done (2026-11-14).** Written and published under the dose + census authorization of
2026-11-09 ("1，2 都开始干吧 然后给我结果就可以"). Both batches are complete; do not rerun them.

**1b. Direction and foundation work (2026-11-14).**
- What led to it: Claude reviewed the plan after reading Causeway and ReSteer in full, reported that the planned step 2 is
  nearly ReSteer's recipe, and offered three directions: (a) the plan as written, (b) one post-training fix for several
  kinds of mid-task instruction (switch task as the headline; stop, return home, put down as coverage), (c) the analysis
  paper. It offered to finish step 0, revise the plan for b and check the remaining papers.
- The user's words: "我觉得B比较好 你觉得呢", then "开始吧 其实我们都是在解决相似的问题 只要有一点不一样 都能发吧 方法只要更好就可以
  但是先把地基打稳".
- Scope: direction b as the storyline; step 0 (census report, the two correction notes, the report page, commit and push);
  the plan revision; the literature check; read-only CPU post-hoc analysis of stored runs.
- This is not an approval to train, to collect data, or to add scripted motions beyond step 1's `home` and `release`
  (a "lower, then release" script for "put down" needs the user's go-ahead). It did not mention GPU runs: confirm once
  before the first GPU submission of step 1.
- Later on 2026-11-14 Claude listed two things to decide: (1) step 1 can start, and the first GPU submission needs the
  user to say GPU runs are fine (formal batch about 25-30 GPU hours); (2) the single-pass side arm, which Claude advised
  putting on hold. The user: "我都批准". Read as: GPU runs for step 1 (smoke, gate, formal batch) are approved, and the
  single-pass side arm is on hold until the user says otherwise. Training and new scripted motions remain unapproved.

**2. Method plan, step 1 (2026-11-12 late).**
- What led to it: the user said the problem is found and they now want a good model-level method that can be published,
  and asked for a look at recent work. Claude reported that mid-task switching already has fixes (Causeway 2609.30913:
  pi0.5 26.0 -> 61.6% on 71 cross-object LIBERO-Goal pairs; ReSteer; SwitchVLA) while amp2 reads 13.1% vs 8.0% on the
  census failure cells, and recommended step 1 (a scripted release-and-retreat to the start pose, then native pi0.5 with
  B, as upper bound, trivial baseline and teacher) and step 2 (fine-tuning). Claude said step 1 needs a new scripted
  retreat and step 2 needs training, neither approved before, and offered to write step 1 as a protocol on the user's nod.
- The user's words: "开始吧 然后看看我这地基稳不稳 然后开始进行plan的修正和下一个agent的说明".
- Scope: the read-only foundation audit (HIL_FOUNDATION_AUDIT.md), the plan and hand-off (HIL_METHOD_PLAN.md), and step 1
  (HIL_RETREAT_PROTOCOL.md): the scripted segment in the branch runner, code and tests, a CPU dry run of the scripted
  segment, the smoke, then (constants and readings frozen) the gate -> formal batch on the census states, and the report.
  pi0.5 alone, no training, no paid API.
- The next agent acts on the user's instruction; if it does not mention GPU runs, confirm once before the first
  submission.
- Stop and ask if a gate fails for a non-code reason or step 1 reads U1 low.
- Later the same day the user agreed that a clever post-training fix is the storyline ("可以") and asked for the hand-off
  and plan to be tidied and the past content compressed. That is not an approval to train.
- 2026-11-14: reading U3 of the draft protocol (does the mark recover after the retreat) was demoted to report-only before
  any freeze; the reason is in the protocol.
- 2026-11-15: step 1 is done (HIL_RETREAT_REPORT.md). The batch is complete; do not rerun it. This authorization is used
  up: it does not cover further GPU runs. The next decision (training, new scripted motions such as "lower, then release"
  or "close the drawer") is the user's.

**3. Single-pass repair (2026-11-12), now a side arm.** The user called the two-pass repair impractical at evaluation and
approved a single-pass batch ("可以 帮我看看怎么弄好 然后写交接文档 我让下一个agent去行动"): the design in
docs/generative_verifier/HIL_MECHANISM_SINGLEPASS_HANDOFF.md, a protocol with readings frozen before submission, code and
tests, direction fitting and an open-loop screen on stored inputs, a closed-loop smoke -> gate -> formal batch and the
report. It starts after step 1's formal batch has finished. Arms outside that design and a second test bed on census cells
need the user's go-ahead. On 2026-11-14 Claude advised putting it on hold and the user agreed ("我都批准"): do not start it
until the user says so.

**Not approved.** Any training (step 2 of the method plan is a design sketch only); a second model; new data collection
beyond step 1's reruns; scripted motions beyond step 1's two arms; the two extra mechanism controls listed in section 3 of the method plan (other layer windows;
replacement by the instruction mean); anything listed as "needs go-ahead" below.

## Completed work (one line each; reports under docs/generative_verifier/ unless a path is given)

| Date | Study | Result | Still needs the user's go-ahead |
|---|---|---|---|
| 09-26 to 09-28 | SafeLIBERO reproduction, nominal pi0.5 and full AEGIS (docs/REPRODUCTION.md) | 64/64 cells, 3,200 episodes, $3.80 of the $5 cap | |
| 10-16 to 10-19 | Verifier on V4, Phases 0-2 (PHASE1_REPORT.md, PHASE2_REPORT.md) | Luna baseline spent $0.4669 of $3; locked test scored once | Phase 3 items; the AEGIS controller change; new paid image requests; new data |
| 10-24 to 10-27 | V5 line rules (V5_S4_REPORT.md, V5_TEST_REPORT.md) | S4 0.940 vs A 0.501; test_confirms | Any AEGIS gate |
| 10-27 to 10-29 | Mid-task rule, return and detour (V5_HIL_REPORT.md, V5_RETURN_REPORT.md, V5_DETOUR_REPORT.md) | ignores_rule, ignores_instruction, ignores_detour | More seeds; an online gate or rule executor; physical interventions |
| 10-29 to 10-31 | HIL suite (HIL_SUITE_REPORT.md) | pi0.5 and OpenVLA-OFT ignore stop / return / cancel; pi0.5 recovers from pushes; pi0 read env_suspect, branches not run | pi0 branches; OFT stages 3/5; an online gate with an outside executor |
| 11-03 to 11-04 | Language ablation (LANGUAGE_ABLATION_REPORT.md) | original / empty / random 96.75 / 50.25 / 53.75%; Goal 96 / 9 / 8% | Further seeds, policies or conditions |
| 11-04 | Mechanism pilot, open loop (HIL_MECHANISM_REPORT.md) | P1 tool_ok, P2 interface_partial, R2 mixed, R3 gap_does_not_predict | P2b closed loop |
| 11-04 | Closed-loop causal test (HIL_MECHANISM_CLOSED_LOOP_REPORT.md) | lam 2 restores_switch and selective (64/90 vs 17/90, rand2 22/90); lam 4 damages | |
| 11-05 | Repair batch (HIL_MECHANISM_REPAIR_REPORT.md) | R6 direction_specific; R7 partial (P5 pass condition not met); R8 selective; R9 amp_better | A held-out dose |
| 11-05 | P4 neurons and blind test (HIL_MECHANISM_NEURONS_REPORT.md) | N0 mlp_route, N1 moderate, N2 mixed, N3 nonspecific, N4 selective, N5 weakly_necessary, N6 interpretable 17/23 | Larger k or dose; other units |
| 11-05 | Held-out open-loop localization (HIL_MECHANISM_HELDOUT_OPEN_REPORT.md) | H1 differs, H2 image_route_fades, H3 proxy_fails, H4 nothing_to_push | A closed-loop window or dose on held-out pairs |
| 11-07 | Unit study (HIL_MECHANISM_UNITS_REPORT.md) | R1 moderate (about 10,000 units), R2 moderate (about 300) | A closed-loop necessity test with such units |
| 11-10 | Reverse dose (HIL_MECHANISM_DOSE_REPORT.md) | D1 nonspecific, D2 necessary (230 -> 6 of 232), D3 commits_like_natural | |
| 11-11 | Task-pair census (HIL_MECHANISM_CENSUS_REPORT.md, written 11-14) | S1 common, S2 weakly_restores (13.1% vs 8.0%), S3 unclear, S4a fade_tracks, S4b beyond_physics; post hoc: amplification mostly interrupts the old task | A second model |
| 11-14 to 11-15 | Retreat and redo, step 1 of the method plan (HIL_RETREAT_REPORT.md) | U1 high (home 88.3% vs native 22.8%, amp2 26.7%), U2 pose_matters (failure cells: home 87.7%, release 18.8%), U4 unclear; post hoc: old task 3 (top drawer left open) 29.6% vs 94.3% for the other old tasks | Training (step 2); new scripted motions; a second model |

What these results can and cannot carry is in docs/generative_verifier/HIL_FOUNDATION_AUDIT.md.
