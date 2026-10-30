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
