# Robot policies that take language corrections, interjections or instruction changes during execution (2021 – Oct 2026)

Scope note for the report writer. Every arXiv ID below was checked against the arXiv API or arXiv abstract/HTML page on 2026-10-31 (title, first authors, first-version date). Conference venues are NOT verified in this pass unless stated; where a venue is given "(venue from memory)" treat it as unconfirmed. Overlap ratings are relative to our finding (pi0.5 and OpenVLA-OFT on LIBERO, policy twins, untrained stop/return/cancel ignored ~0/235, trained LIBERO-Goal switch only partial and decaying with progress, mid-task spatial rules ignored 160/160 and 168/168). Rating key: high = same models and/or same benchmark and same phenomenon; medium = same phenomenon in a different setting, or a direct claim about flat VLAs and mid-task language; low = related problem, different mechanism or setting.

## Q1. What do the classic mid-task language-interaction systems show, and what do they need?

### Takeaway
Every system that reliably acts on mid-task language either (a) puts a separate high-level LLM/VLM, planner or shared-autonomy layer between the human and the low-level controller (Inner Monologue, Sharma et al., LILAC, OLAF, DROC, YAY Robot, RACER, Hi Robot), or (b) trains the low-level policy on data built for re-instruction: hundreds of thousands of densely language-annotated clips (Interactive Language), language-motion labels (RT-H), or step-by-step verbal-coaching data (pi0.5, pi0.7). None of them shows that an off-the-shelf, benchmark-fine-tuned flat VLA follows a new instruction it was not trained to receive mid-episode. Work on chunked-policy latency (RTC, BID and later papers) is about reacting to state changes, not language. It only explains why a new instruction can take effect no earlier than the next inference call.

### Cited Findings

**Interactive Language: Talking to Robots in Real Time.** Lynch, Wahid, Tompson, Ding et al. (Google). arXiv 2210.06407, 2022-10-12 (venue from memory: IEEE RA-L 2023).
- Input: free-form real-time natural language at any time, used as the *only* way to specify behaviour. A human guides the robot through long-horizon rearrangements ("make a smiley face out of blocks") over about 5 minutes — [arXiv HTML](https://arxiv.org/html/2210.06407)
- Needs: a flat policy trained by behaviour cloning on a dataset of "hundreds of thousands of language-annotated trajectories" (nearly 600,000 released, Language-Table) — [arXiv 2210.06407](https://arxiv.org/abs/2210.06407). It runs "continuous 5Hz visuolinguomotor control", and the authors contrast it with "blocking parameterized skills" that prevent live interaction — [arXiv HTML](https://arxiv.org/html/2210.06407)
- Results: estimated 93.5% success on 87,000 unique natural-language strings; one human can guide several real-time robots at once — [arXiv 2210.06407](https://arxiv.org/abs/2210.06407); [arXiv HTML](https://arxiv.org/html/2210.06407). Setting: real (Language-Table) plus simulation.
- Overlap: **medium**. This is the strongest prior evidence that a *flat* policy can follow mid-task instruction changes, but only because it was trained for it on about 600K language-annotated clips; it does not test off-the-shelf VLAs.

**Inner Monologue: Embodied Reasoning through Planning with Language Models.** Huang, Xia, Xiao, Chan et al. (Google). arXiv 2207.05608, 2022-07-12 (venue from memory: CoRL 2022).
- Input: closed-loop language feedback that includes "human interaction", in addition to success detection and scene description. Needs: an LLM planner over pretrained skills, "without any additional training". Setting: simulated and real tabletop, real kitchen mobile manipulation — [arXiv 2207.05608](https://arxiv.org/abs/2207.05608)
- Overlap: **low**. The human feedback goes to an LLM planner, not to the low-level policy.

**Correcting Robot Plans with Natural Language Feedback.** Sharma, Sundaralingam, Blukis, Paxton et al. (MIT/NVIDIA). arXiv 2204.05186, 2022-04-11 (venue from memory: RSS 2022).
- Input: corrections such as new goals, new constraints ("avoid specific objects") or waypoints. Needs: a mapping from sentences to *cost-function transformations* for a planner, not a learned visuomotor policy. Results: 81% and 93% success on tasks where the original planner failed, after one or two corrections, in sim and real — [arXiv 2204.05186](https://arxiv.org/abs/2204.05186)
- Overlap: **low**. It handles constraint-style language (close to our spatial rules), but the constraint is executed in a planner's cost function, which supports "outside the policy".

**"No, to the Right": Online Language Corrections for Robotic Manipulation via Shared Autonomy (LILAC).** Cui, Karamcheti, Palleti, Shivakumar et al. (Stanford). arXiv 2301.02555, 2023-01-06 (venue from memory: HRI 2023). LILAC and "No, to the right" are the same paper.
- Input: online corrections ("to the right", "no, towards the book") during execution. Needs: language is mapped to a learned *low-dimensional control space* that the human then teleoperates (shared autonomy), learned from "a handful of demonstrations". Setting: Franka user study. Beats open-loop instruction-following baselines on completion and user preference — [arXiv 2301.02555](https://arxiv.org/abs/2301.02555)
- Overlap: **low**. The human stays in the control loop; this is not autonomous instruction following.

**OLAF: Interactive Robot Learning from Verbal Correction.** Huihan Liu, Alice Chen, Yuke Zhu, Adith Swaminathan et al. (UT Austin RPL). arXiv 2310.17555, 2023-10-26.
- Input: verbal corrections when the robot errs, e.g. "Stop what you're doing. You should move closer to the cup." Needs: an LLM-based system that *updates the visuomotor policy* from the feedback so the mistake is not repeated. Results: 20.0% average improvement in policy success, sim and real — [arXiv 2310.17555](https://arxiv.org/abs/2310.17555)
- Overlap: **low**. "Stop" is handled by the LLM system (Hi Robot describes OLAF as using "an LLM to modify robot trajectories" — [arXiv HTML](https://arxiv.org/html/2502.19417)) and a later policy update, not by the policy reading the utterance.

**DROC: Distilling and Retrieving Generalizable Knowledge for Robot Manipulation via Language Corrections.** Zha, Cui, Lin, Kwon et al. (Stanford). arXiv 2311.10678, 2023-11-17.
- Input: arbitrary online language corrections that address high-level plans and low-level skill parameters. Needs: an LLM-based system with a knowledge base and retrieval. Results: needs half as many corrections in the first round as LLM code-generation baselines — [arXiv 2311.10678](https://arxiv.org/abs/2311.10678)
- Overlap: **low**. Uses LLM code and primitives, not a learned VLA.

**CLAIRify: Errors are Useful Prompts: Instruction Guided Task Programming with Verifier-Assisted Iterative Prompting.** Skreta, Yoshikawa, Arellano-Rubach, Ji et al. (Toronto). arXiv 2303.14100, 2023-03-24.
- Iterative LLM prompting with a program verifier for a chemistry DSL; plans are executed through a task-and-motion planner. The loop is over *plan generation*, not mid-execution human language — [arXiv 2303.14100](https://arxiv.org/abs/2303.14100)
- Overlap: **low**. Not about mid-execution interaction.

**LaTTe / Reshaping Robot Trajectories Using Natural Language Commands.** Bucker, Figueredo, Haddadin, Kapoor et al. arXiv 2203.13411 (2022-03-25) and 2208.02918 (2022-08-04).
- Language commands modify an existing trajectory through BERT/CLIP and a transformer trained on trajectories paired with language edits — [arXiv 2203.13411](https://arxiv.org/abs/2203.13411); [arXiv 2208.02918](https://arxiv.org/abs/2208.02918)
- Overlap: **low**. Offline trajectory reshaping, not interjection into a running VLA.

**RT-H: Action Hierarchies Using Language.** Belkhale, Ding, Xiao, Sermanet et al. (Google DeepMind). arXiv 2403.01823, 2024-03-04 (venue from memory: RSS 2024).
- Input: human "language motion" corrections such as "move arm forward" during execution. Needs: one model trained to first predict language motions and then actions conditioned on them, i.e. an *in-model hierarchy* with language-motion labels extracted for the training data. Results: +15% over RT-2 on diverse tasks; fine-tuning on language-motion interventions beats IWR by 50% — [arXiv 2403.01823](https://arxiv.org/abs/2403.01823); [arXiv HTML](https://arxiv.org/html/2403.01823)
- Overlap: **low–medium**. Corrections work because the model was trained with a language-motion interface; this says nothing about off-the-shelf flat VLAs.

**YAY Robot: Yell At Your Robot: Improving On-the-Fly from Language Corrections.** Lucy X. Shi, Zheyuan Hu, Tony Z. Zhao, Archit Sharma et al. (Stanford). arXiv 2403.12910, 2024-03-19 (venue from memory: RSS 2024).
- Input: free-form real-time verbal corrections, from "move the sponge a little lower" to "I want more M&M's in my trail mix bag" — [arXiv HTML](https://arxiv.org/html/2403.12910)
- Needs: a hierarchy. A high-level policy issues language instructions to a "sufficiently rich and expressive low-level language-conditioned" skill policy, and human corrections replace or supervise the high-level output and are then used to fine-tune the high-level policy. Setting: real hardware, long-horizon dexterous tasks (bag packing, trail mix). Reports "significant performance improvement" with no extra teleoperation — [arXiv 2403.12910](https://arxiv.org/abs/2403.12910). Exact success numbers were not extracted (see Gaps).
- Overlap: **low–medium**. Fine-grained corrections reach the low level only because the low level was trained on densely annotated fine-grained commands.

**RACER: Rich Language-Guided Failure Recovery Policies for Imitation Learning.** Dai, Lee, Fazeli, Chai (U. Michigan). arXiv 2409.14674, 2024-09-23.
- Needs: a VLM supervisor that produces rich language guidance online, a language-conditioned visuomotor actor, and automatically generated failure-recovery data with fine-grained language — [arXiv 2409.14674](https://arxiv.org/abs/2409.14674)
- Includes "dynamic goal-change tasks" on RLBench: RACER succeeds in 60 of 100 trials and "significantly" outperforms RVT. In the RACER+H variant, humans can override the VLM's instructions with their own simple ones — [arXiv HTML](https://arxiv.org/html/2409.14674)
- Overlap: **medium–low**. It has a mid-task goal-change evaluation against a flat language-conditioned baseline (RVT), but on RLBench with a 3D policy, not VLAs. The RVT number was not extracted.

**Hi Robot: Open-Ended Instruction Following with Hierarchical VLA Models.** Lucy X. Shi, Ichter, Equi, Ke et al. (Physical Intelligence, Stanford). arXiv 2502.19417, 2025-02-26 (venue from memory: ICML 2025).
- Input: complex prompts and situated interjections during execution ("that's not trash", "I don't like that one", "leave it alone"). The user "can intervene at any point during policy execution ... or even change the task entirely" — [arXiv 2502.19417](https://arxiv.org/abs/2502.19417); [arXiv HTML](https://arxiv.org/html/2502.19417)
- Needs: a high-level VLM that reads prompts and feedback and emits atomic commands, which a low-level VLA (pi0) executes. Training uses *synthetically generated hypothetical prompts and human interjections*. The high level is invoked "after a set time or upon receiving new language feedback" — [arXiv HTML](https://arxiv.org/html/2502.19417)
- Results: evaluated on three platforms (single-arm, dual-arm, dual-arm mobile) for table bussing, sandwich making and grocery shopping. Hi Robot averages "over 40% higher instruction accuracy than GPT-4o" — [arXiv HTML](https://arxiv.org/html/2502.19417)
- Flat-VLA baseline, a direct statement about flat VLAs: "The flat baseline, on the other hand, does not react to real-time feedback", and "the flat baseline and GPT-4o often revert to default behaviors ... when the prompt changes mid-episode". A flat policy trained on the same synthetic data is also beaten by the hierarchy — [arXiv HTML](https://arxiv.org/html/2502.19417)
- Overlap: **medium**. This is the closest prior *qualitative* statement of our conclusion (a flat pi0 ignores mid-episode feedback), from PI itself, on real robots. It is not quantified per command type, does not use policy twins, and does not test pi0.5/OFT on LIBERO.

**RTC: Real-Time Execution of Action Chunking Flow Policies.** Kevin Black, Manuel Y. Galliker, Sergey Levine (Physical Intelligence). arXiv 2506.07339, 2025-06-09.
- Inference-time asynchronous chunking for any diffusion or flow VLA with no retraining. It generates the next chunk while executing the current one, "freezing" actions that are guaranteed to execute and "inpainting" the rest. Evaluated on 12 dynamic Kinetix tasks and 6 real bimanual tasks; it removes pauses and jerks at chunk boundaries and is robust to inference delay — [arXiv 2506.07339](https://arxiv.org/abs/2506.07339)
- Overlap: **low**. It addresses latency, not language. The relevant implication is that a new instruction can only affect actions after the frozen prefix.

**Bidirectional Decoding (BID).** Yuejiang Liu, Hamid, Xie, Lee et al. (Stanford). arXiv 2408.17355, 2024-08-30.
- Finds that action chunking captures temporal dependencies "at the cost of reduced reactivity to unexpected states", and proposes test-time sampling for backward coherence and forward contrast — [arXiv 2408.17355](https://arxiv.org/abs/2408.17355)
- Overlap: **low**. This is a mechanism for chunk-level inertia, not a study of language.

**Later chunk-responsiveness work (2026).**
- Legato: a training-time continuation method that reduces "spurious multimodal switching" relative to RTC — [arXiv 2602.12978](https://arxiv.org/abs/2602.12978)
- Event-triggered asynchronous inference: "VLA models commonly predict action chunks, limiting their ability to react to environmental changes during execution" — [arXiv 2609.22587](https://arxiv.org/abs/2609.22587)
- VLA-Feedback: chunks "are typically executed open-loop after inference", which limits responsiveness — [arXiv 2609.21022](https://arxiv.org/abs/2609.21022)
- Overlap of all three: **low**. They react to scene changes, not new language.

**Speech, full-duplex and interruptible VLA systems (2025–2026).**
- **VITA-E** (Xiaoyu Liu, Chaoyou Fu, Chi Yan, Chu Wu et al.; arXiv 2510.21817, 2025-10-21) states that current VLAs lack the ability to "handle real-time user interruptions dynamically" and that SOTA models "assume that user instructions are provided once at the beginning of a task and remain static". It adds a *dual-model architecture*: an "Active" and a "Standby" VLA instance, plus a fine-tuned VLM that emits *special tokens acting as system-level commands*, generated as the first token for emergency stops. It reports "an extremely high success rate on emergency stops and speech interruptions" on a real humanoid — [arXiv 2510.21817](https://arxiv.org/abs/2510.21817); [arXiv HTML](https://arxiv.org/html/2510.21817). Overlap: **medium**. Same diagnosis, and its fix for "stop" is executed at system level, outside the action policy. Numbers not extracted.
- **ELLSA: End-to-end Listen, Look, Speak and Act** (Siyin Wang, Wenyi Yu et al., Tsinghua; arXiv 2510.16756, 2025-10-19). A full-duplex end-to-end model (SA-MoE) that supports "action barge-ins", i.e. "instantly stopping an action upon hearing an interruptive spoken command", and also rejection of defective instructions — [arXiv 2510.16756](https://arxiv.org/abs/2510.16756); [arXiv HTML](https://arxiv.org/html/2510.16756). Overlap: **medium–low**. It needs a purpose-built architecture and training; barge-in numbers not extracted.
- **RoboOmni** (arXiv 2510.23763) infers intent from speech, sounds and visual cues using a 140k-episode OmniAction dataset — [arXiv 2510.23763](https://arxiv.org/abs/2510.23763)
- **VLAS** (arXiv 2502.13508) adds speech instructions end to end — [arXiv 2502.13508](https://arxiv.org/abs/2502.13508)
- **MIRA** (arXiv 2609.24547) handles full-duplex "barge-in preemption" for companion-robot gestures, "bounding physical commitment to a short, interruptible prefix" — [arXiv 2609.24547](https://arxiv.org/abs/2609.24547)
- Overlap of RoboOmni, VLAS and MIRA: **low**.
- **OneTwoVLA** (Fanqi Lin et al.; arXiv 2505.11917) is a single model that switches between reasoning and acting and claims "natural human-robot interaction". It needs synthesized embodied-reasoning data for co-training — [arXiv 2505.11917](https://arxiv.org/abs/2505.11917). Overlap: **low**.

### Inferences
- The requirements fall into three families: (1) an external high-level reasoner (LLM/VLM/planner) translates the human utterance into something the low level already knows (Inner Monologue, Sharma, OLAF, DROC, YAY, RACER, Hi Robot); (2) the low level is *trained* to be re-instructable through dense language annotation, language motions or coaching data (Interactive Language, RT-H, pi0.5/pi0.7 below); (3) a system-level interrupt path (VITA-E special tokens, ELLSA barge-in). Our finding fits this pattern: off-the-shelf benchmark-fine-tuned VLAs have none of the three.
- Hi Robot's flat-pi0 baseline remark ("does not react to real-time feedback") is the most citable prior support for our claim and should be cited as qualitative precedent. Our contribution there is quantification under a controlled twin design and an extension to pi0.5/OFT and to stop/return/cancel commands.

### Gaps
- Exact success numbers were not extracted for: YAY Robot, Hi Robot's flat baselines (IA/TP per task), RACER's RVT goal-change baseline, and VITA-E's and ELLSA's stop and barge-in rates.
- Conference venues were not verified in this pass.
- Gemini Robotics: the DeepMind blog says the model "continuously monitors its surroundings, detects changes to its environment or instructions, and adjusts its actions accordingly" and describes it as a VLA — [DeepMind blog](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/); [arXiv 2503.20020](https://arxiv.org/abs/2503.20020). A keyword scan of the arXiv HTML found no quantitative mid-task instruction-change evaluation. Treat this as a marketing-level claim with no measured number found. Gemini Robotics 1.5 "interleaves actions with a multi-level internal reasoning process in natural language" — [arXiv 2510.03342](https://arxiv.org/abs/2510.03342). It was not checked for interjection experiments.

## Q2. Is there a benchmark or paper (2024–2026) that changes or interrupts the instruction mid-episode for flat VLAs and measures whether they follow?

### Takeaway
Yes, and three of them overlap heavily with our trained-instruction-switch result. SwitchVLA (2025-06; pi0 and OpenVLA-OFT on LIBERO-Goal), ReSteer (2026-03; pi0.5, OpenVLA-OFT and MolmoAct on LIBERO-Goal) and Causeway (2026-09; pi0.5, OpenVLA-OFT and GR00T N1.5 on LIBERO-Goal and LIBERO-Object) all switch to another *trained* LIBERO-Goal instruction mid-execution. All three find that off-the-shelf VLAs follow poorly and worse as execution progresses. We found no paper that measures off-the-shelf flat VLAs on *untrained* meta-commands (stop / return arm to start / cancel) issued mid-task, or on mid-task spatial constraints ("don't cross this line", "keep out of this zone"). Those parts of our finding appear unscooped as of this search.

### Cited Findings

**SwitchVLA: Execution-Aware Task Switching for Vision-Language-Action Models.** Meng Li, Zhen Zhao, Zhengping Che, Fei Liao et al. (X-Humanoid). arXiv 2506.03574, 2025-06-04.
- Problem statement: VLAs "typically assume static task intent, failing to respond when new instructions arrive during ongoing execution" — [arXiv 2506.03574](https://arxiv.org/abs/2506.03574)
- Protocol: Task A is interrupted by Task B's instruction at early (pre-contact), mid (in-contact) or late (post-action) phase on LIBERO-Goal. Success is strict: Task A must behave as specified after the switch (e.g. roll back or put down the held object) *and then* Task B must be completed — [arXiv HTML](https://arxiv.org/html/2506.03574)
- LIBERO-Goal Table 1 (No switch / Early / Mid / Late, % success) — [arXiv HTML](https://arxiv.org/html/2506.03574)
  - pi0: 92.3 / 40.7 / 8.3 / 10.2
  - OpenVLA-OFT: 98.0 / 40.6 / 11.1 / 13.0
  - SwitchVLA: 93.0 / 93.5 / 50.9 / 68.7
  - The baseline labelled [9] renders as pi0 in the HTML math.
- Real robot, workstation 1, pi0: 97.9 / 50.1 / **0.0** / 32.0. Workstation 2, pi0: 100 / 75.0 / **0.0** / 64.6 — [arXiv HTML](https://arxiv.org/html/2506.03574)
- Conventional VLAs show "erratic behavior like oscillation or dropping items" when the user changes their mind — [arXiv HTML](https://arxiv.org/html/2506.03574)
- Needs: a Florence-2-based policy trained with contact-phase segmentation and behaviour-mode labels (forward / rollback / advance) derived from existing demos, with "no external planners or additional switch-specific data" — [arXiv 2506.03574](https://arxiv.org/abs/2506.03574); [arXiv HTML](https://arxiv.org/html/2506.03574)
- Overlap: **high**. Same benchmark (LIBERO-Goal), OpenVLA-OFT in common (pi0 instead of pi0.5), same early > mid/late decline.

**ReSteer: Quantifying and Refining the Steerability of Multitask Robot Policies.** Zhenyang Chen, Alan Tian, Liquan Wang, Benjamin Joffe et al., advised by Karamcheti and Danfei Xu (Georgia Tech). arXiv 2603.17300, 2026-03-18.
- Defines "task steerability", e.g. a robot "may fail to respond to a new instruction 'put the bowl in the sink' when moving towards the oven, executing 'close the oven', even though it can complete both tasks when executed separately" — [arXiv 2603.17300](https://arxiv.org/abs/2603.17300)
- Evaluates pi0.5, OpenVLA-OFT and MolmoAct on LIBERO-Goal: "Although all three VLAs achieve saturated single-task performance on LIBERO-Goal ... none exhibits reliable cross-task steering under prompt switches". Steerability scores in Table I are 0.252 / 0.295 / 0.403 in column order [pi0.5], OpenVLA-OFT, MolmoAct. Caveat: the first column header is a math symbol rendered as pi0.5 in the HTML, so verify the mapping in the PDF — [arXiv HTML](https://arxiv.org/html/2603.17300)
- "Baseline policies exhibit limited steerability, with task switching feasible only early in execution and substantial performance degradation as the rollout progresses" — [arXiv HTML](https://arxiv.org/html/2603.17300)
- Diagnosis: steerability is tied to limited overlap among training task trajectory distributions, and "most existing datasets exhibit pathologies that encourage learned models to ignore task conditioning". Fix: generate stage-aware steering data where "the task instruction changes mid-execution", use CMI-guided sampling and self-refining BC. This gives "~10–11%" improvement in simulation over 18k rollouts, plus real DROID experiments — [arXiv 2603.17300](https://arxiv.org/abs/2603.17300); [arXiv HTML](https://arxiv.org/html/2603.17300)
- Overlap: **high**. Same models (pi0.5, OFT), same benchmark (LIBERO-Goal), same "decays with progress" finding, and an evaluation from intermediate states that is essentially a policy-twin design.

**Causeway: Restoring Task Accessibility for Instruction Switching in VLA Policies.** Qingzi Wang, Kaixi Feng, Guangyao Shi, Xiyang Wu et al. arXiv 2609.30913, 2026-09-25.
- Studies "instruction switching, where a new task is issued during or after the execution of a different one". It names states from which a reliably completable target becomes inaccessible "task islands" — [arXiv 2609.30913](https://arxiv.org/abs/2609.30913)
- Setup: benchmark-fine-tuned pi0.5, OpenVLA-OFT and GR00T N1.5 on LIBERO-Goal, over 71 cross-object pairs and three switch timings (step 30, step 60, after completion). Bare-switch success averages **26.0% (pi0.5), 3.4% (OpenVLA-OFT), 9.9% (GR00T)** "despite near-perfect from-scratch performance" — [arXiv HTML](https://arxiv.org/html/2609.30913)
- Per-timing bare-switch values from Table I. These are read as step 30 / step 60 / after completion; column headers were inferred from the text — [arXiv HTML](https://arxiv.org/html/2609.30913)
  - pi0.5: 29.9 / 22.9 / 25.1
  - OFT: 5.1 / 2.9 / 2.2
  - GR00T: 14.6 / 11.8 / 3.4
- On OpenVLA-OFT, "instruction response attenuates most strongly along the approach" — [arXiv HTML](https://arxiv.org/html/2609.30913)
- Prompt-level contrast (CFG-style contrastive prompt guidance) "provides no consistent recovery" — [arXiv HTML](https://arxiv.org/html/2609.30913)
- Needs: a training-free, inference-time *representation write* that back-propagates through the frozen decoder toward a *re-entry pose for the target task*. This raises success to 47–65%. Extra experiments on LIBERO-Object and a real xArm — [arXiv 2609.30913](https://arxiv.org/abs/2609.30913)
- Overlap: **high**. Same two policies, same benchmark, same type of switch. Its absolute numbers for cross-object pairs are much lower than our ceiling-normalized ones.

**Learning What to Say to Your VLA: Mostly Harmless VLA Model Steering.** Hyun Joe Jeong, Gokul Swamy, Andrea Bajcsy. arXiv 2606.12299, 2026-06-10.
- Finds the language-to-behaviour mapping of VLAs "brittle and unintuitive". It searches for *language sequences* that improve closed-loop performance of *frozen* VLAs and distills them into a test-time language feedback policy (+24.7% sim, +65.0% hardware). It reports "recovery behaviors not observed with open-loop prompting" — [arXiv 2606.12299](https://arxiv.org/abs/2606.12299)
- Overlap: **medium**. It shows that changing language mid-rollout *can* steer frozen VLAs when the words are searched or learned, a counterpoint to "VLAs ignore mid-task language". Which VLAs were used was not extracted.

**Atomic Motion Coordinate for Language-Steerable and Force-Responsive Manipulation.** Jiaqi Zhai et al. arXiv 2609.15012, 2026-09-14.
- Asks "Can changing only the language instruction redirect a VLA policy's end effector, or does the visually driven motion prior dominate?" Over 7,520 offline horizon interventions, its method reaches 92.5/83.1% opposite-atom separation (single/dual arm), against 39.1/24.0% for an LA4VLA-style baseline — [arXiv 2609.15012](https://arxiv.org/abs/2609.15012)
- Overlap: **medium–low**. Same question at the motion level, but offline and with a trained interface.

**Hi Robot (see Q1).**
- Real-robot, qualitative: the flat pi0 baseline "does not react to real-time feedback" and reverts to default behaviours "when the prompt changes mid-episode" — [arXiv HTML](https://arxiv.org/html/2502.19417)

**RACER (see Q1).**
- RLBench "dynamic goal-change tasks": RACER 60/100, outperforming RVT — [arXiv HTML](https://arxiv.org/html/2409.14674)

**Methodologically adjacent: MAIL-Bench (Learning to Act under Visual Interruptions with VLA Models).** Mingle Jiang, Rui Xu, Yunke Wang, Chang Xu. arXiv 2609.35003, 2026-09-28.
- Interrupts *cameras* "at multiple stages of each policy's successful reference trajectory" for pi0.5 and GR00T N1.5. This is the same replay-to-checkpoint idea as our policy twins, applied to visual interruptions — [arXiv 2609.35003](https://arxiv.org/abs/2609.35003)
- Overlap: **low** on content, **medium** on method.

**Constraint-style ("what not to do") language, the closest to our spatial rules.**
- **NegaAlign / NegaBench** (Fazeng Li et al.; arXiv 2610.11952, 2026-10-08): makes GR00T, pi0 and pi0.5 follow *negated* instructions, i.e. "accomplish task goals while respecting explicit exclusions". Uses inserted layers trained with image-language supervision and a 10-scenario simulation benchmark — [arXiv 2610.11952](https://arxiv.org/abs/2610.11952). Overlap: **medium–low**. It covers constraints, but the abstract presents them as part of the initial instruction, not as mid-task additions.
- **SafeStage** (arXiv 2609.21223): a lifecycle safety benchmark with an "Execution-Time Safety" stage that scores "unsafe contacts, trajectories, region entries". It finds "nominal task completion frequently coexists with safety violations" — [arXiv 2609.21223](https://arxiv.org/abs/2609.21223). Overlap: **low–medium**. Region entries are measured, but rules are not added mid-task.
- **ReStruct** (arXiv 2606.26588): inference-time preference steering through a VLM-compiled temporal-logic automaton over a neural automaton policy. It beats "the prompt-steered vision-language-action model π0.5" (70.9% sim / 92.5% real preference following) — [arXiv 2606.26588](https://arxiv.org/abs/2606.26588). Overlap: **low–medium**. It is evidence that prompting pi0.5 alone is a weak way to impose preferences or constraints, and it solves the problem outside the VLA.

**Steerability-by-training papers (trained interfaces, not off-the-shelf evaluation).**
- Steerable Policies (William Chen, Bhatia, Glossop et al.): VLAs trained on synthetic commands at many abstraction levels, driven by a VLM — [arXiv 2602.13193](https://arxiv.org/abs/2602.13193)
- FineVLA: fine-grained "how" instructions — [arXiv 2605.27284](https://arxiv.org/abs/2605.27284)
- 2AM: an agent holds task memory and steers an "episodically stateless" action model with subtask language and hints; 76.3% on LIBERO-Mem — [arXiv 2609.11308](https://arxiv.org/abs/2609.11308)
- TempoBridge: tempo cues on LIBERO — [arXiv 2610.09451](https://arxiv.org/abs/2610.09451)
- Overlap of all four: **low**.

### Inferences
- **The trained-switch half of our finding is scooped or precedented.** SwitchVLA (2025) already showed pi0 and OpenVLA-OFT on LIBERO-Goal dropping from ~40% early-switch success to ~8–13% at mid/late switches. ReSteer (2026-03) and Causeway (2026-09) repeat this with pi0.5 and OpenVLA-OFT specifically. Any paper must cite all three and frame our LIBERO-Goal switch numbers as a replication or extension, not a discovery.
- **Our numbers are notably *higher* than theirs, and the paper must reconcile them.** Ours: pi0.5 .84/.78/.70 and OFT .50/.50/.20 of the t = 0 ceiling. Causeway's absolute bare-switch success: pi0.5 ~26%, OFT ~3%. SwitchVLA's mid/late: pi0 8–10%, OFT 11–13%. Likely causes:
  - (a) normalization by the t = 0 ceiling;
  - (b) pair selection, since Causeway uses 71 *cross-object* pairs only;
  - (c) timing defined by progress fraction rather than absolute steps or contact phase;
  - (d) success criteria, since SwitchVLA also requires correct handling of Task A, e.g. putting down the held object.
  
  A reviewer will ask for this reconciliation. Reporting raw (unnormalized) switch success by pair type would make the numbers comparable.
- **Still apparently open, judging by this search:**
  - (1) untrained meta-commands (stop / return arm to start / cancel) issued mid-task to off-the-shelf flat VLAs, measured against a twin control;
  - (2) mid-task addition of spatial constraints (line / keep-out) to off-the-shelf VLAs;
  - (3) breadth across all four standard LIBERO suites;
  - (4) a two-family comparison (pi0.5 vs OFT) on the same twins.
  
  VITA-E and ELLSA handle "stop" or barge-in only by building dedicated mechanisms. They do not measure what an off-the-shelf VLA does with "stop".
- **Other work agrees with "execute the response outside the policy" but scopes it.** VITA-E uses system-level stop tokens, ReStruct an automaton outside the VLA, Hi Robot a high-level VLM. On the other side, SwitchVLA and ReSteer fix switching *inside* the policy through training, Causeway fixes it at inference time with a representation write (it needs a target re-entry pose), and Learning-What-to-Say steers frozen VLAs with learned language. So our conclusion is safest as "*off-the-shelf, benchmark-fine-tuned* flat VLAs do not act on mid-task language". The claim that it *cannot* be done inside a policy is contradicted by these papers.

### Gaps
- The Causeway affiliations and venue were not extracted. The ReSteer column mapping (pi0.5 = 0.252?) and the Causeway timing-column labels were inferred from the HTML text and should be checked in the PDFs.
- The VLAs used in "Learning What to Say to Your VLA" were not extracted.
- Search method: arXiv API keyword queries on abstracts ("interrupt*", "task switching", "instruction switching", "mid-task/mid-execution/mid-episode", "language correction", "full-duplex", "steerab*", "stop command", "intent change", "lock-in/commitment"), plus targeted fetches. OpenReview (ICLR 2026 / CoRL 2026 submissions) and Google Scholar were not searched. A paper that uses none of these words, e.g. "re-instruction" or "goal re-specification", could be missed.
- No paper was found that issues "stop" / "return to start" / "cancel" to pi0, pi0.5, OpenVLA, Octo, RDT or GR00T without special training and reports the response rate. This is absence of evidence from a bounded search, not proof of absence.

## Q3. Do PI's pi0.5 / Hi Robot / pi0.6 / pi0.7 claim that the flat low-level policy follows mid-task interjections, or is it the high-level policy?

### Takeaway
In PI's own descriptions, interjections and open-ended feedback are handled by a high-level language step, either a separate VLM (Hi Robot) or the same model's text-decoding pass (pi0.5, pi0.7). The low-level action expert then follows *short subtask commands* that it was trained on, including step-by-step human "verbal coaching" commands that change mid-episode. PI never claims that a flat VLA given a raw interjection reacts correctly. Hi Robot says explicitly that the flat pi0 baseline does not react to real-time feedback. pi0.7's claim of following human coaching mid-episode is the strongest PI evidence that the low level follows *trained-format* instruction changes. Even so, it rests on dedicated training data and an inference loop that re-prompts when the subtask changes.

### Cited Findings
- **Hi Robot (2502.19417).** The high-level VLM processes prompts and feedback and outputs atomic commands to a low-level VLA. The high level runs "after a set time or upon receiving new language feedback". The flat baseline "does not react to real-time feedback" and "often revert[s] to default behaviors ... when the prompt changes mid-episode". The training data include synthetic hypothetical interjections — [arXiv HTML](https://arxiv.org/html/2502.19417)
- **pi0.5 (arXiv 2504.16054, 2025-04-22).**
  - Co-training includes "'verbal instruction' demonstrations, where a person coaches the robot through a complex task by telling it what to do step by step" — [PI blog, pi0.5](https://www.pi.website/blog/pi05)
  - At run time the model "first ... output[s] a 'high level' action expressed in text, and then ... follow[s] this high level action" with a 50-step (1-second) action chunk. This "follows our recently developed Hi Robot system, except that the same model is used for both the high-level decisions and low-level motor control" — [PI blog, pi0.5](https://www.pi.website/blog/pi05)
  - Paper abstract — [arXiv 2504.16054](https://arxiv.org/abs/2504.16054)
- **pi*0.6 / RECAP (arXiv 2511.14759, 2025-11-18).** The "corrections" are "expert teleoperated interventions provided during autonomous execution", i.e. action-level interventions, not language interjections — [arXiv 2511.14759](https://arxiv.org/abs/2511.14759). No language-interjection claim was found in the abstract.
- **pi0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities (arXiv 2604.15483, 2026-04-16).**
  - Conditioned on language plus "metadata about task performance and subgoal images" — [arXiv 2604.15483](https://arxiv.org/abs/2604.15483)
  - Claims pi0.7 can be "coached" with "step-by-step verbal instructions" on unseen appliance tasks. The subtask instruction "is provided either by a learned high-level language policy or by a human supervisor for coaching". Its test-time loop re-runs inference when the subtask has "changed or [a] timer elapsed", asynchronously with RTC — [arXiv HTML](https://arxiv.org/html/2604.15483)
  - PI states: "Prior models generally lack the language following ability needed to follow the coaching commands, and thus achieve very poor performance" — [arXiv HTML](https://arxiv.org/html/2604.15483)
  - The blog shows "interactively directing π0.7 with varied language commands" — [PI blog, pi0.7](https://www.pi.website/blog/pi07)
  - Secondary press coverage (TechCrunch via texxr) reports that pi0.7 cannot yet complete complex multi-step tasks from a single high-level command — [texxr / TechCrunch repost](https://texxr.com/1167182/physical-intelligence-introduces-0-7-robot-model). This is secondary and not checked against the paper.
- **RTC (PI).** Asynchronous chunk execution with frozen prefixes — [arXiv 2506.07339](https://arxiv.org/abs/2506.07339). pi0.7 uses it in its prompting loop — [arXiv HTML](https://arxiv.org/html/2604.15483)

### Inferences
- PI's line runs: Hi Robot uses a separate VLM, pi0.5 a self-generated text subtask, and pi0.7 a human or learned high-level policy issuing coaching subtasks. Across all three, the thing that reacts to an interjection is a language-level step. The low-level controller only sees commands of a trained type, so PI's claims are consistent with our finding rather than contradicting it.
- pi0.7's remark that prior models "achieve very poor performance" when coached is a PI-internal admission that earlier models do not follow step-by-step language changes well. The extract does not show which "prior models" were meant (possibly pi0.5 and pi0.6), so cite it carefully.
- Our pi0.5 result uses the openpi LIBERO fine-tuned checkpoint. The verbal-coaching co-training of the base pi0.5 may not survive LIBERO fine-tuning, which the "lock-in" and "instruction blindness" papers in Q4 predict. The paper should say that the tested artefact is the public LIBERO checkpoint, not PI's internal pi0.5 with high-level inference.

### Gaps
- The specific "prior models" in the pi0.7 coaching comparison and their numbers were not extracted.
- It is not verified whether the pi0.5 or pi0.7 papers report a quantitative mid-task interjection (as opposed to coaching) experiment for the low-level policy alone.
- No PI release after pi0.7 (Apr 2026) up to Oct 2026 was found by an arXiv author search on "Physical Intelligence". The search returned pi0.7, pi*0.6 and pi0.5 only.

## Q4. Is there evidence of commitment or inertia: once a policy starts, it continues regardless of new commands?

### Takeaway
Yes, from several independent angles:
- Switch studies show policies continue the original task: ReSteer's "close the oven" example, Hi Robot's "revert to default behaviors", SwitchVLA's oscillation and dropping, Causeway's "task islands".
- Robustness studies show VLAs ignore or barely use language even at t = 0: LIBERO-PRO, LIBERO-Plus, LangGap, instruction blindness, information collapse, lock-in.
- Chunking studies show a mechanical source of reduced reactivity (BID, RTC and follow-ups).

None of these frames inertia exactly as "the untrained command is ignored and the original task is still completed", which is our twin-controlled measure.

### Cited Findings
- **ReSteer.** The policy fails "to respond to a new instruction ... executing 'close the oven', even though it can complete both tasks when executed separately". Steerability is "feasible only early in execution" — [arXiv 2603.17300](https://arxiv.org/abs/2603.17300); [arXiv HTML](https://arxiv.org/html/2603.17300)
- **Causeway.** Task islands: after Task A, the target becomes inaccessible. Bare-switch success after completion ranges from 3% to 50% and is "strongly negatively associated with the mean target-relative approach-axis error at exit" — [arXiv HTML](https://arxiv.org/html/2609.30913)
- **SwitchVLA.** Conventional VLAs "cannot adjust [their] plan, leading to erratic behavior like oscillation or dropping items". Mid-phase switch success is 0.0% for pi0 on both real workstations — [arXiv HTML](https://arxiv.org/html/2506.03574)
- **Hi Robot.** The flat baseline "does not react to real-time feedback" and reverts to default behaviours when the prompt changes mid-episode — [arXiv HTML](https://arxiv.org/html/2502.19417)
- **LIBERO-PRO** (Xueyang Zhou et al.; arXiv 2510.03827, 2025-10-04). Models "persist in executing grasping actions when the target object is replaced with irrelevant items, and their outputs remain unchanged even when given corrupted instructions or even messy tokens". Performance collapses from >90% to 0.0% under generalized perturbations — [arXiv 2510.03827](https://arxiv.org/abs/2510.03827). Overlap: **medium**. Same benchmark family, but the perturbation is at t = 0.
- **LIBERO-Plus** (Senyu Fei et al.; arXiv 2510.13626, 2025-10-15). Models are "largely insensitive to language variations ... models tend to ignore language instructions completely" — [arXiv 2510.13626](https://arxiv.org/abs/2510.13626). Overlap: **medium**.
- **Breaking Lock-In** (Suning Huang et al.; arXiv 2604.23121, 2026-04-25). After low-data SFT a generalist VLA "stops responding to new instructions and is limited to behaviors observed during post-training". Concept and spatial lock-in are mitigated by DeLock, which preserves visual grounding and adds test-time contrastive prompt guidance — [arXiv 2604.23121](https://arxiv.org/abs/2604.23121). Overlap: **medium**. This is a likely mechanism for why LIBERO-fine-tuned checkpoints ignore new commands.
- **Flatness Preserves Instruction Following** (Haochen Zhang, Yonatan Bisk; arXiv 2606.23641). Limited-data fine-tuning yields policies "that ignore language instructions in favor of visual shortcuts", termed "instruction blindness". SAM fine-tuning improves instruction following by >60% — [arXiv 2606.23641](https://arxiv.org/abs/2606.23641). Overlap: **medium–low**.
- **LangForce** (arXiv 2601.15197). Goal-driven data make instructions predictable from vision, so the conditional mutual information between instruction and action "vanish[es]" ("Information Collapse") and models "degenerate into vision-only policies" — [arXiv 2601.15197](https://arxiv.org/abs/2601.15197). Overlap: **low–medium**.
- **LangGap** (arXiv 2603.00592). "Current state-of-the-art VLA models largely ignore language instructions", shown on pi0.5 with semantic perturbations under fixed layouts; LIBERO assigns "only one task per layout" — [arXiv 2603.00592](https://arxiv.org/abs/2603.00592). Overlap: **medium–low**.
- **Paraphrase Consensus Is Not Self-Consistency** (arXiv 2610.19438). On LIBERO, rewording the instruction costs base OpenVLA 38.0 points. A "wrong-instruction probe measures how much of a policy's success depends on the instruction" across three policies including OpenVLA-OFT — [arXiv 2610.19438](https://arxiv.org/abs/2610.19438). Overlap: **medium–low**. The wrong-instruction probe is a t = 0 analogue of our twin control.
- **Robust Skills, Brittle Grounding** (arXiv 2602.24143). For SmolVLA and pi0.5, "manipulation skill acquisition is decoupled from instruction following" — [arXiv 2602.24143](https://arxiv.org/abs/2602.24143). Overlap: **low–medium**.
- **Do VLAs Really Follow Instructions?** (arXiv 2610.21251). How-level instructions "do not help current VLAs and lower task success" (SIMPLER-IN) — [arXiv 2610.21251](https://arxiv.org/abs/2610.21251). Overlap: **low**.
- **HINT** (arXiv 2609.02653). In VLAs, "visual correlations can then dominate semantic intent, leading actions to follow visual shortcuts rather than human goals" — [arXiv 2609.02653](https://arxiv.org/abs/2609.02653). Overlap: **low**.
- **Chunk-level inertia.** Chunking costs "reduced reactivity to unexpected states" — [arXiv 2408.17355](https://arxiv.org/abs/2408.17355). Chunks are "executed open-loop after inference" — [arXiv 2609.21022](https://arxiv.org/abs/2609.21022). RTC freezes the actions that are guaranteed to execute — [arXiv 2506.07339](https://arxiv.org/abs/2506.07339). Overlap: **low**. This explains latency, not total non-response.

### Inferences
- Two separable causes of "inertia" appear in the literature:
  - (1) *Semantic*: fine-tuned VLAs rely little on the instruction (LIBERO-PRO, LIBERO-Plus, LangGap, lock-in, information collapse, instruction blindness). This predicts that an untrained command like "stop" is effectively invisible.
  - (2) *State and geometry*: once the arm is committed, the new task is out of the demonstrated state distribution (task islands, low trajectory overlap in ReSteer). This predicts that even trained switches decay with progress.
  
  Our data seem to show both: about 0/235 for untrained commands, which matches cause 1, and a ceiling ratio that falls with progress for trained switches, which matches cause 2. The writer can use this split to position the contribution.
- Our "task still completed after an untrained command" measure is a stronger, behavioural form of LIBERO-PRO's "outputs remain unchanged under corrupted instructions". It shows the same insensitivity mid-episode under a twin control.

### Gaps
- No paper was found that reports, as a primary metric, "original task still completed after a conflicting mid-task command" for off-the-shelf VLAs. SwitchVLA and Causeway report switch success, not completion of the abandoned task.
- Numbers for Breaking Lock-In and LIBERO-Plus per model were not extracted.
- Search terms "inertia", "momentum", "commitment" and "lock-in" turned up only Breaking Lock-In and unrelated items, so other framings may exist under different wording.
