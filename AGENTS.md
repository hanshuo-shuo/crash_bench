# CrashBench / SafeLIBERO agent instructions

Read docs/CURRENT.md first; it links the active plan,
docs/generative_verifier/PLAN.md. docs/REPRODUCTION.md covers the SafeLIBERO baseline.

On 2026-09-26 the user authorized keeping the eight results in retained/,
archiving other CrashBench work, and starting reproduction of nominal pi0.5
and full AEGIS on the unchanged public SafeLIBERO benchmark. The user has an
OpenRouter API key and wants to be notified when it is needed. Never print,
commit or place that key in a job log. Do not silently substitute another VLM.

The old experiment plans live on codex/archive-crashbench-20260926 and are
historical, not an active run queue. No new learned intervention, controller,
benchmark task, confirmatory study or threshold search is authorized. (This is
the 2026-09-26 boundary; later dated user authorizations below and in
docs/STATUS_HISTORY_until_20261016.md override it only for their stated scope.)

Before any Quest SSH use read QUEST_WORKFLOW.md and setup/README.md, then run
scripts/quest_sync.sh check. Only $HOME/crash_bench / hanshuo-shuo/crash_bench is
the valid project checkout. Do not touch another project's files or jobs.
Submit only a clean, tested, published commit through scripts/quest_sync.sh.
Use account p33100, GPU partition gengpu, CPU partition short. No heavy work on
login nodes; only downloads, Git and lightweight checks there. Compute nodes
are offline: stage assets first. Never force pull, hard reset or rsync source.
Keep historical raw results recoverable. Every new run needs a unique output
root, code/upstream commit, configuration, seed and Slurm provenance.

Budget update (2026-09-26): the user has $100/week for OpenRouter alone. Plan
baseline reproduction within $5/week and preserve $95 for their method. This
smoke batch has used two calls ($0.00279368); no more paid calls in this batch.
Before any bulk API worker, inspect usage metadata, estimate the batch and add
a spend guard. No provider-side key cap has been configured by this task.

Full-run authorization (2026-09-26): the user now authorized starting the full
nominal pi0.5 / full AEGIS official matrix, subject to $5 OpenRouter spending for
this reproduction (including earlier smoke calls). Use setup/submit_safelibero_full.sh
only after publishing and syncing. See docs/reproduction/BATCH_RUN.md. The batch
uses at most two GPUs and stops its own arrays on budget or infrastructure failure.
That batch is complete: 64/64 cells and 3,200 episodes, $3.80 reported plus
reserved (results/safelibero_batches/20260927T204911Z_d22035816647/COMPLETE.json).
Standing rule: never change or sync the Quest source checkout while any CrashBench
job depends on it; check the queue first.

Verifier decision (2026-10-16): the user decided the V4 review need not be strict
("as long as it works"). The 896-row V4 set may be used as development data for
verifier training and evaluation; its failed shortcut gate stays disclosed and is
reported as a baseline, not a training precondition. The locked test split is
scored once per selected checkpoint. Follow docs/generative_verifier/PLAN.md.
GPU phases still follow the Quest rules above. The AEGIS controller change in
Phase 4, new paid API image requests and new data collection each need separate
user approval.

Phase 0-1 (2026-10-17): the user asked to start Phase 0 and, if it went well, Phase 1,
naming "gpt luna 6" (openai/gpt-6-luna) for the Phase 1 API baseline. Both phases are
complete (docs/generative_verifier/PHASE1_REPORT.md); the baseline spent $0.4669 of its $3 cap
on validation only. On 2026-10-17 the user said "开始吧" to Phase 2; it is complete and the locked test
was scored once (docs/generative_verifier/PHASE2_REPORT.md). Do not score that test split again.
Phase 3 items each need the user's go-ahead.

V5 (2026-10-24): the user moved the verifier line to V5 (natural-language line rules
on real objects, added mid-task; counterfactual layouts by moving untouched anchors),
then restricted it to twin-capable task compositions with no source expansion and
chose object/II/task1 for the first story pilot. Read docs/generative_verifier/V5_HANDOFF.md.
Each S1-S3 step still needs the user's go-ahead; no paid API without separate approval.
S3, the fit diagnostic and S4 ran on 2026-10-25; S4 (positions plus a program) reached 0.940 against
A's 0.501 on the counterfactual validation twins and the round was wrapped up (V5_S4_REPORT.md). More seeds,
the single test evaluation and any AEGIS gate each need the user's go-ahead.
Seeds 714/715 and the single V5 test evaluation ran on 2026-10-26 with the user's go-ahead (reading
test_confirms, V5_TEST_REPORT.md). Do not score the V5 test split again; any AEGIS gate still needs approval.

HIL (2026-10-27): the user approved the design of the mid-task intervention
robustness experiment (docs/generative_verifier/V5_HIL_PROTOCOL.md): pi0.5 alone,
policy twins on object/II/task1, no verifier, no paid API. Implementation and unit
tests need no further approval; the GPU smoke and the formal batch are new pi0.5
executions and need the user's go-ahead before launch. Test sources episode0/8/9/15
stay untouched.

Detour HIL (2026-10-28): the user chose the next step: test whether pi0.5 follows a
feasible mid-task rule that requires a detour (keep-out zone rule, policy twins on
object/II/task1); brief in docs/generative_verifier/V5_HANDOFF.md. The protocol draft and
CPU checks on existing HIL records need no further approval; show the user the protocol.
New layouts are new data, and the GPU smoke and formal batch are new pi0.5 executions;
each needs the user's go-ahead. No paid API; test sources episode0/8/9/15 stay untouched.

Return HIL (2026-10-29): the user asked to add the simplest mid-task intervention: part-way
through the task the instruction changes to "move the arm back to its starting position"
(docs/generative_verifier/V5_RETURN_PROTOCOL.md): pi0.5 alone on the 160 existing HIL states,
no new layouts, no verifier, no paid API. The protocol draft, implementation and unit tests need
no further approval; the GPU smoke and formal batch are new pi0.5 executions and need the user's
go-ahead. Test sources episode0/8/9/15 stay untouched. On 2026-10-29 the user approved both
("gpu都可以直接用 开始吧"): 12-run smoke, freeze, formal 480 runs, report. All ran on 2026-10-29
(reading ignores_instruction for R and RA, V5_RETURN_REPORT.md).

Detour approval (2026-10-29): after the return report the user approved the detour HIL ("可以 绕路吧 你看着办
这个应该很好实现 也没啥歧义"): new on-route layouts (CPU), the 12-run GPU smoke and the formal batch, then the
report (V5_DETOUR_PROTOCOL.md section 12). pi0.5 alone, no verifier, no paid API, test sources episode0/8/9/15
untouched. Stop and ask if the layout stage reads layouts_thin or layouts_insufficient. All ran on 2026-10-29
(layouts_ready; reading ignores_detour, V5_DETOUR_REPORT.md). Any further step (more seeds, an online gate or
rule executor, physical interventions) needs the user's go-ahead.

HIL suite (2026-10-29 late): the user asked for a more concrete analysis of how to show that VLA robustness to
human-in-the-loop interventions is poor, with more scenarios and tasks. The draft is
docs/generative_verifier/HIL_SUITE_PLAN.md (standard LIBERO four suites, LIBERO-Goal mid-task switch with t = 0 ceilings,
untrained commands, physical interventions, pi0 and OpenVLA-OFT). It is a plan only: the design, each GPU stage and the
OFT checkpoint download each need the user's go-ahead. No paid API; SafeLIBERO test sources episode0/8/9/15 untouched.
On 2026-10-29 late the user approved all of it ("都可以 同意 干吧"): the design and readings, every stage (0-5) including its
GPU smoke and formal batch, pi0 and OpenVLA-OFT, and the OFT checkpoint download. Still pi0.5/pi0/OFT alone, no verifier,
no paid API, at most two GPUs; freeze each stage's readings after its smoke and before its formal batch.
All ran by 2026-10-31 (docs/generative_verifier/HIL_SUITE_REPORT.md): pi0.5 stages 0-3 and 5, OpenVLA-OFT stages 0-2
(H1 holds for both; pi0.5 weak form, OFT strong form; pi0.5 recovers from all four physical interventions). pi0 read
env_suspect at stage 0 (LIBERO-10 0.84 < 0.85), so by the frozen rule its branches were not run; running them, OFT
stages 3/5, or an online gate with an outside executor each need the user's go-ahead.

Language ablation (2026-11-03): the user requested testing whether task language matters by replacing existing-task
instructions with randomly generated A/B/C/D/E/F/+ strings. Execute the current pi0.5 on unchanged standard LIBERO's
40 tasks, from t=0: original / empty / random prompts, 10 initial states per task, plus 40 original-prompt repeats.
The 16-run pipeline smoke, protocol freeze, formal 1,240 runs and report are within this request's scope
(docs/generative_verifier/LANGUAGE_ABLATION_PROTOCOL.md). No paid API, verifier, training, new layouts or locked-test
access. At most two GPUs; existing Quest publishing and source-dependency rules still apply.
Completed by 2026-11-04 (LANGUAGE_ABLATION_REPORT.md): 1,240 formal runs, original/empty/random success
96.75%/50.25%/53.75%; Goal 96%/9%/8%. All integrity checks passed, no API or locked-test access.
Further seeds, policies or conditions need the user's go-ahead; do not silently rerun this completed batch.

Mechanism pilot (2026-11-04): after discussing with their advisor, the user chose to deepen the HIL line into a mechanism
study of mid-task switching in pi0.5, and after a literature check said "开始改吧". That approves turning
docs/generative_verifier/HIL_MECHANISM_IDEA.md into the pilot plan, the P0 state selection (read-only, stage 1 records)
and the P1 patching code and unit tests (experiments/mechanism/). Each GPU stage (P1 smoke, formal open-loop batch,
closed-loop runs), any new runner, and the P5 repair intervention each need the user's go-ahead. pi0.5 alone, inference
only on stored HIL suite inputs. No training, no paid API, no new data, SafeLIBERO test sources untouched, at most two GPUs.
On 2026-11-04 the user approved the P1 smoke ("跑一下吧"): 24 states, one GPU, open-loop inference on stored inputs via
experiments/mechanism/launch.py smoke. The formal 240-state batch still needs a separate go-ahead after section 7 is frozen.
After the smoke read tool_ok and section 7 was frozen (3fb6a44), the user approved the formal open-loop batch on
2026-11-04 ("开始跑吧"): 240 states, two GPUs, launch.py formal with the smoke root. P2b closed-loop, further P3 tools,
P4 neurons and P5 still each need a separate go-ahead.
After the formal open-loop report the user chose next step 1 on 2026-11-04 ("先做第一项吧"): the closed-loop causal
test (docs/generative_verifier/HIL_MECHANISM_CLOSED_LOOP.md). It reruns stage 1 switch branches with the layer 12-14
image-token conditioning amplified or randomly moved (serve_steer.py), and covers the protocol, the new server and
runner hand-off, the 86-run smoke, then (smoke tool_ok, section 5 frozen) the formal 774 runs on two GPUs and the report.
Stop and ask if the smoke fails for a non-code reason. P2b, P4 neurons and P5 repair still each need a separate go-ahead.
The user explicitly allowed recording this entry here ("允许的").
Completed on 2026-11-04 (HIL_MECHANISM_CLOSED_LOOP_REPORT.md): lam 2 reads restores_switch and selective (failure cells
amp2 64/90 vs native 17/90, rand2 22/90); lam 4 reads no_restore and damages. Do not rerun this completed batch; an
action-matched control, P5, P4 and P2b each still need the user's go-ahead.

Repair batch (2026-11-04): after the closed-loop report the user chose to run the action-matched random control and P5
repair on held-out stage 1 pairs as one batch, P4 later ("把'动作上匹配的随机对照'和'P5 修复（在别的任务对上检验）'合成一个批次先做，
P4 神经元放在后面"), agreed to online action matching, a fixed lam 2 and an output-level B - A extrapolation control, and said
to start the formal batch directly ("可以 这个直接开始正式的吧 反正gpu够用"). Scope: docs/generative_verifier/HIL_MECHANISM_REPAIR.md,
the new server arms and tests, then smoke -> integrity gate -> formal 2,024 runs -> report, chained without a stop (readings
frozen before submission). pi0.5 alone, reruns of stage 1 switch branches, no training, no paid API, no new layouts,
SafeLIBERO untouched, at most two GPUs. Stop and ask if the gate fails for a non-code reason. P4 and P2b still each need
a separate go-ahead.
Completed on 2026-11-05 (HIL_MECHANISM_REPAIR_REPORT.md): R6 direction_specific (amp2 65/90 vs actrand 21/90), R7 commits
nonspecific and derails restores_switch (both predictions wrong; the P5 pass condition was not met), R8 selective, R9 amp_better
for actdiff, rephrase and ampall. Do not rerun this completed batch; P4, held-out open-loop localization, a held-out dose and P2b
each still need the user's go-ahead.

P4 neurons (2026-11-05): after the repair report Claude listed three next steps with P4 neurons first; the user replied "可以 开始下一步吧
点头了". Scope (docs/generative_verifier/HIL_MECHANISM_NEURONS.md): the protocol, the neuron hooks and tests, the open-loop smoke,
then (section 6 frozen) the formal open-loop batch on stored stage 1 inputs, the closed-loop smoke, gate and formal batch (reruns of
stage 1 switch branches with neuron-level arms), the blind explanation test with a fresh subagent, and the report. pi0.5 alone, no
training, no paid API, no new layouts, SafeLIBERO untouched, at most two GPUs. Stop and ask if a smoke or gate fails for a non-code
reason. Held-out open-loop localization, a held-out dose and P2b still each need a separate go-ahead.
Completed on 2026-11-05 (HIL_MECHANISM_NEURONS_REPORT.md): N0 mlp_route, N1 moderate (k* 1,000-3,000), N2 mixed, N3 nonspecific
(nrn2 31/90 vs native 20, random direction 26; all-MLP 59, amp2 67), N4 selective, N5 weakly_necessary (one cell), N6 interpretable
(17/23). Do not rerun these batches or the blind test; any further step (larger k or dose, other units, held-out localization or dose,
P2b) needs the user's go-ahead.

Held-out open-loop localization (2026-11-05): after the P4 report Claude recommended this as the next step (inference only on
stored inputs, about 1-2 GPU hours) and the user replied "可以 先继续吧". Scope (docs/generative_verifier/HIL_MECHANISM_HELDOUT_OPEN.md):
the protocol with readings frozen before submission, the code and tests, then smoke (70 states) -> integrity gate -> formal (614
states: the eight held-out pairs of the repair batch with the pilot's open-loop measures, plus the open-loop push of the repair on
the five pilot failure cells) on at most two GPUs, and the report. pi0.5 alone, inference only on stored HIL suite stage 1 inputs,
no simulator, no training, no paid API, no new layouts, SafeLIBERO untouched. Stop and ask if the gate fails for a non-code reason.
A per-pair window or a held-out dose in closed loop, and P2b, still each need the user's go-ahead.
Completed on 2026-11-05 (HIL_MECHANISM_HELDOUT_OPEN_REPORT.md): H1 differs (every readable held-out pair reads B at layers 14-16;
the two weak pairs are image_window, the effective 8->7 and LIBERO-10 0->1 elsewhere), H2 image_route_fades (failure 0.55, success
1.13), H3 proxy_fails (rho -0.13), H4 nothing_to_push for both weak pairs (but the effective LIBERO-10 0->1 is equally unreadable).
Do not rerun this batch; a closed-loop window or dose on held-out pairs, and P2b, each still need the user's go-ahead.

Unit study (2026-11-07): after comparing P4 with Goodfire's "Neural Networks Are Already Clean" essay (it claims single neurons are
legible, not that behaviours rest on a few neurons), Claude proposed correcting the P4 wording and an open-loop follow-up that ranks
neuron x image-row units by their attributed effect on the action. The user replied "可以 改 然后做第二个". Scope
(docs/generative_verifier/HIL_MECHANISM_UNITS.md): the wording corrections, the protocol with readings frozen before submission, the
code and tests, then smoke (24 states) -> integrity gate -> formal (216 pilot open-loop states) on at most two GPUs, and the report.
pi0.5 alone, inference only on stored HIL suite stage 1 inputs, no simulator, no training, no paid API, no new data, SafeLIBERO
untouched. Stop and ask if the gate fails for a non-code reason. Using such unit sets in closed loop needs a separate go-ahead.
Completed on 2026-11-07 (HIL_MECHANISM_UNITS_REPORT.md): C1 tool_ok; R1 moderate (half the sufficiency effect needs a median of
10,000 neuron x image-row units, on the boundary; P4's ranking needs ~5 million), R2 moderate (half the necessity effect needs a
median of 300 units). Formal shard 1 first ran out of memory on a 40 GB card and was resubmitted on 80 GB (RESUBMIT.json in the root).
Do not rerun this batch; a closed-loop necessity test with such units needs the user's go-ahead.
