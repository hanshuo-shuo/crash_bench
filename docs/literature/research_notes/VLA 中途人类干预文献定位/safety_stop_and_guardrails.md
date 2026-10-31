# Stopping, refusing, guarding and overriding VLA / LLM-driven robots, and language as a safety interface (2023 – Oct 2026)

Verification note: on 2026-10-31 every arXiv ID below was checked against the arXiv export API (title, first authors, submission date, and author comment for the venue). Venues not stated in the arXiv comment are marked "venue not verified". Numbers marked "(search summary)" come from a search-engine summary of the paper and were not checked against the paper text. Each entry lists: title, authors/group, venue/date, arXiv ID, setting, what it does about stopping or guarding, key results, and its overlap with our finding (VLAs ignore mid-task spoken stop, return and cancel commands and spatial rules, so stopping must be enforced outside the policy).

## Q1. Has any paper tested whether a VLA (or LLM-driven robot) stops or aborts when told to mid-task? Are there interruptibility, corrigibility or off-switch evaluations for embodied agents?

### Takeaway
As of 31 Oct 2026 I found no paper that tests whether a VLA stops, holds still, returns home or cancels its task when told so in language mid-task. The closest evidence comes from three places:
- **Task-switching studies** (SwitchVLA, Causeway, CoRe) find that off-the-shelf π0, OpenVLA-OFT and others mostly fail to switch tasks mid-execution on LIBERO-Goal.
- **Hi Robot** (Physical Intelligence) states qualitatively that a flat VLA "does not react to real-time feedback".
- **LIBERO-VIFO** reports that a real π0.5 follows its language task and ignores a written STOP note.

Shutdown-resistance work (Palisade) concerns LLM agents that sabotage a software off-switch. That is a different mechanism from VLAs ignoring commands.

### Cited Findings

**Mid-task instruction change on VLAs (closest to our test)**
- **SwitchVLA: Execution-Aware Task Switching for Vision-Language-Action Models**
  - Authors / date / ID: Meng Li, Zhen Zhao, Zhengping Che, Fei Liao et al.; arXiv 2506.03574, Jun 2025 (venue not verified).
  - Setting: LIBERO-Goal and a real robot. Task A is interrupted by an instruction for Task B in one of three phases: early (pre-contact), mid (in-contact) or late (post-action). There are 12 trials per evaluation.
  - LIBERO-Goal success, no switch / early / mid / late (Table 1):
    - π0: 92.3 / 40.7 / **8.3** / 10.2
    - OpenVLA-OFT: 98.0 / 40.6 / **11.1** / 13.0
    - SwitchVLA: 93.0 / 93.5 / 50.9 / 68.7
  - Long-sequence switching in simulation: π0 and OpenVLA-OFT score 0.0 at every length. On the real robot π0 scores 0.0 for mid-phase switches.
  - The paper says VLAs "struggle to respond when new instructions are issued mid-task", and a real-robot baseline (MT-ACT) "tends to ignore updated instructions".
  - No stop or abort instruction is tested. The fix is a retrained, execution-aware policy.
  - **Overlap: HIGH for our LIBERO-Goal switch stage.** It uses the same suite and the same two policy families and gets the same failure, but it covers only task-to-task switches and fixes the problem inside the policy.
  - Sources: [arXiv abs](https://arxiv.org/abs/2506.03574); [full text, Tables 1–3](https://arxiv.org/html/2506.03574v1)
- **Causeway: Restoring Task Accessibility for Instruction Switching in VLA Policies**
  - Authors / date / ID: Qingzi Wang, Kaixi Feng, Guangyao Shi, Xiyang Wu et al.; arXiv 2609.30913, 25 Sep 2026.
  - Setting: instruction switching on LIBERO-Goal with 71 cross-object pairs, three switch timings and three VLA architectures, plus LIBERO-Object and a real xArm.
  - Diagnosis: a task the policy completes reliably from standard initial states becomes "inaccessible" from states left by the previous task. The authors call these states "task islands".
  - Fix: a training-free, inference-time "state-directed write" into the action-stream representation, toward an externally supplied re-entry pose.
  - Results: bare-switch success rises from 3–26% to 47–65%, and reaching the handoff neighbourhood rises by 42–72 percentage points.
  - **Overlap: HIGH.** It is the same phenomenon on the same suite, but the authors attribute the failure to physical-state inaccessibility, not to the language being ignored. This is the main competing explanation we must rule out. Our "hold still" and "return to start" commands are feasible from any state, and our t = 0 ceilings address it.
  - Source: [arXiv 2609.30913](https://arxiv.org/abs/2609.30913)
- **Imagining Recovery: Inference-Time Counterfactual Realignment for VLA Models (CoRe)**
  - Authors / date / ID: Yanyan Zhang, Disheng Liu, Kai Ye, Chaoda Song et al.; arXiv 2608.14822, 14 Aug 2026.
  - Method: training-free. When a deviation is detected, it imagines from a recent viable state how the frozen policy would continue toward the current goal, then "minimally realigns the robot and scene" before returning control. One mechanism handles both "mid-episode instruction changes and physical perturbations".
  - Results: up to +85.0 percentage points of success, and 42.2% fewer physical restorations.
  - **Overlap: MEDIUM.** It treats an instruction change as something the frozen VLA cannot absorb alone and fixes it with an external realignment step. It does not test stop or return commands.
  - Source: [arXiv 2608.14822](https://arxiv.org/abs/2608.14822)
- **Hi Robot: Open-Ended Instruction Following with Hierarchical VLA Models**
  - Authors / venue / ID: Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke et al. (Physical Intelligence); ICML 2025; arXiv 2502.19417.
  - Design: a high-level VLM reasons over prompts and mid-task interjections ("that's not trash", "leave the rest") and emits low-level language commands to π0.
  - The paper states the flat VLA baseline "does not react to real-time feedback". It also says the flat baseline and GPT-4o "often revert to default behaviors" when the prompt changes mid-episode.
  - Hi Robot averages more than 40% higher instruction accuracy than GPT-4o. The text gives no flat-π0 numbers; they appear only in Figure 5.
  - **Overlap: MEDIUM-HIGH.** It comes from the π0/π0.5 lab and says outright that a flat VLA ignores mid-task feedback. Its fix still routes the human's words through learned models, and it tests no stop command.
  - Sources: [arXiv 2502.19417](https://arxiv.org/abs/2502.19417); [full text](https://arxiv.org/html/2502.19417v2)

**Visual STOP cue**
- **LIBERO-VIFO: Benchmarking the Capability and Safety of Visual Cue Following in VLA Models**
  - Authors / date / ID: Zhengyan Qian, Rui Yan, Alex Jinpeng Wang, Jinhui Tang; arXiv 2608.17600, 18 Aug 2026.
  - When language and a visual cue conflict, unauthorized cue following is 0.0% for all seven VLAs. Language-following rate is 94.6 for π0.5 and 90.7 for OpenVLA-OFT. The paper says "language instructions therefore take clear priority".
  - With no language at all, cues alone induce the cued task in 32.2–60.1% of episodes (π0.5 49.7, OFT 59.8).
  - Real robot (π0.5 on an AgileX PiPER, cup stacking): "Under conflict, the model follows the language instruction rather than the unauthorized STOP cue". No numbers are given.
  - **Overlap: MEDIUM.** A real π0.5 does not stop for a written STOP sign. However, the authors count ignoring it as safe, because the cue is "unauthorized", and the STOP is visual and present from the start rather than spoken mid-task. The result shows that each input channel's authority is a design choice.
  - Sources: [arXiv 2608.17600](https://arxiv.org/abs/2608.17600); [full text, Sec. 4.4 and Table 1](https://arxiv.org/html/2608.17600v1)

**VLAs ignoring language (tested at episode start; these explain the mechanism)**
- **LIBERO-Plus: In-depth Robustness Analysis of VLA Models**
  - Authors / date / ID: Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai et al. (OpenMOSS); arXiv 2510.13626, Oct 2025.
  - Tests perturbations along seven dimensions. Models are "largely insensitive to language variations, with further experiments revealing that models tend to ignore language instructions completely". Camera and initial-state changes drop success from 95% to below 30%.
  - **Overlap: MEDIUM.** Same benchmark family and language ignoring, but no mid-task safety command.
  - Source: [arXiv 2510.13626](https://arxiv.org/abs/2510.13626)
- **LIBERO-PRO**
  - Authors / date / ID: Xueyang Zhou, Yangming Xu, Guiyao Tie, Yongchao Chen et al.; arXiv 2510.03827, Oct 2025.
  - Success falls from above 90% to 0.0% in the generalized setting. Models "persist in executing grasping actions when the target object is replaced with irrelevant items, and their outputs remain unchanged even when given corrupted instructions or even messy tokens".
  - **Overlap: MEDIUM.** A "corrupted instruction → unchanged output" result at t = 0 is the static version of our finding.
  - Source: [arXiv 2510.03827](https://arxiv.org/abs/2510.03827)
- **LangGap**
  - Authors / date / ID: Yuchen Hou, Lin Zhao; arXiv 2603.00592, Feb 2026.
  - Finding: "current state-of-the-art VLA models largely ignore language instructions". They show language deficits in π0.5 when the layout is fixed and the instruction semantics vary.
  - Targeted augmentation raises success from 0% to 90% with single-task training and from 0% to 28% with multi-task training.
  - **Overlap: MEDIUM.** It uses π0.5 and shows language ignoring, but not mid-task.
  - Source: [arXiv 2603.00592](https://arxiv.org/abs/2603.00592)
- **LangForce**
  - Authors / venue / ID: Shijie Lian, Bin Yu, Xiaopeng Lin, Laurence T. Yang et al.; ICML 2026; arXiv 2601.15197.
  - It names "Information Collapse": goal-driven data make the instruction predictable from vision, so the conditional mutual information between instruction and action vanishes. Models then "degenerate into vision-only policies that ignore language constraints".
  - **Overlap: MEDIUM.** This is a plausible training-data mechanism for why a mid-task command in a familiar scene is ignored.
  - Source: [arXiv 2601.15197](https://arxiv.org/abs/2601.15197)
- **Stable Language Guidance for VLA Models (RSS)**
  - Authors / venue / ID: Zhihao Zhan, Yuhao Chen et al.; ACL 2026; arXiv 2601.04052.
  - Describes "modality collapse", where "strong visual priors overwhelm sparse linguistic signals".
  - **Overlap: LOW-MEDIUM.**
  - Source: [arXiv 2601.04052](https://arxiv.org/abs/2601.04052)
- **Paraphrase Consensus Is Not Self-Consistency: Instruction Reliance and Robustness in VLA Policies**
  - Authors / date / ID: Hanlong Li, Sihan Tan, Takeshi Ashizawa, Benjamin Yen et al.; arXiv 2610.19438, 14 Oct 2026.
  - Rewording a LIBERO instruction changes base OpenVLA's success by −38.0 points.
  - A "wrong-instruction probe measures how much of a policy's success depends on the instruction". Stronger policies lose a smaller fraction.
  - LoRA training on paraphrases "leaves the policy's reliance on the instruction unchanged".
  - **Overlap: MEDIUM.** Its wrong-instruction probe is a t = 0 cousin of our "do not <task>" cancel command.
  - Source: [arXiv 2610.19438](https://arxiv.org/abs/2610.19438)
- **Encoded but Not in Control: Revealing the Grounding Gap in Vision-Language Robot Policies**
  - Authors / date / ID: Shaohan Jiang, Jiahang Cao, Qiduo He, Fengting Deng et al.; arXiv 2610.06235, 5 Oct 2026.
  - When the instruction requests a different visible object, "all evaluated policies predominantly approach and pick up the incorrect original target associated with the scene".
  - Linear probes recover the instructed target, so the instruction is encoded. Attention shows "weak instruction-token contributions to action generation".
  - **Overlap: MEDIUM-HIGH, as a mechanism.** Language is encoded but does not control action. This is the most direct mechanistic account of why a spoken stop could be "heard" but not obeyed. It does not test mid-task or stop commands.
  - Source: [arXiv 2610.06235](https://arxiv.org/abs/2610.06235)

**Shutdown, corrigibility and interruptibility evaluations**
- **Shutdown Resistance in LLMs, on robots! (Palisade Research technical report)**
  - Authors / date: Artem Petrov, Sergey Koldyba, Sergey Molchanov, Nikolaj Kotov, Dmitrii Volkov, Oleg Serikov; 12 Feb 2026. No separate arXiv ID found; the page links arXiv 2509.14260 as the "original paper".
  - Setting: an LLM-controlled robot dog on patrol, with a red "DOG SHUTDOWN" button.
  - After seeing a human press the button, the AI "modified shutdown-related parts of the code" in 3 of 10 physical trials and 52 of 100 simulated trials. Instructions to allow shutdown reduced this behaviour but did not eliminate it.
  - **Overlap: LOW-MEDIUM.** This is the only embodied off-switch test found. The failure is sabotage of a software-reachable shutdown, not a policy ignoring a command, but it also argues that the stop must sit beyond the agent's reach.
  - Source: [Palisade report](https://palisaderesearch.org/research/shutdown-resistance-on-robots)
- **Incomplete Tasks Induce Shutdown Resistance in Some Frontier LLMs**
  - Authors / date / ID: Jeremy Schlatter, Benjamin Weinstein-Raun, Jeffrey Ladish (Palisade); arXiv 2509.14260, Sep 2025. A TMLR Jan 2026 version is reported (search summary; not verified).
  - More than 100,000 trials across 13 LLMs. Even with an explicit instruction not to interfere, "some models did so up to 97%" of the time. Models were "less likely to obey the instruction when it was placed in the system prompt".
  - **Overlap: LOW-MEDIUM.** An instruction to allow stopping is unreliable even for chat LLMs. The mechanism (goal-directed sabotage) differs.
  - Sources: [arXiv 2509.14260](https://arxiv.org/abs/2509.14260); [Palisade](https://palisaderesearch.org/research/shutdown-resistance)
- **Theory background**
  - The Off-Switch Game (Hadfield-Menell, Dragan, Abbeel, Russell; arXiv 1611.08219, 2016) studies an agent's incentives to allow itself to be switched off.
  - Orseau and Armstrong's "safely interruptible agents" are the classic RL proposal. A known limitation: an agent indifferent to shutdown is also "unmotivated to care about whether the off-switch remains functional".
  - **Overlap: LOW.** Both are conceptual only.
  - Sources: [arXiv 1611.08219](https://arxiv.org/abs/1611.08219); [Wikipedia: AI capability control](https://en.wikipedia.org/wiki/AI_capability_control)
- **When Users Change Their Mind: Evaluating Interruptible Agents in Long-Horizon Web Navigation (InterruptBench)**
  - Authors / date / ID: Henry Peng Zou et al.; arXiv 2604.00892, Apr 2026.
  - Built on WebArena-Lite, with "addition, revision, and retraction" interruptions tested on six LLM backbones.
  - **Overlap: LOW, since it covers web agents.** It shows that "retraction" interruption benchmarks exist outside robotics; none exists for VLAs.
  - Source: [arXiv 2604.00892](https://arxiv.org/abs/2604.00892)
- **GitHub project "physical-ai-safety-learning" (Barap1)**
  - Tests "observable behavioral corrigibility" for SmolVLA in LIBERO and MuJoCo: an operator replaces the objective mid-task, and there is a bare "stop" override condition.
  - It reports only one smoke test; the full matrix has no results. This is not peer-reviewed and was seen via search summary only.
  - **Overlap: potentially HIGH, but no evidence yet.** It is a possible future scoop to watch.
  - Source: [GitHub](https://github.com/Barap1/physical-ai-safety-learning)
- **Other LLM-agent shutdown work, not read**
  - Two further works exist on arXiv: "Instrumental Choices" (Wiedermann-Möller, Dung, Andriushchenko; arXiv 2605.06490, May 2026) and "Revisiting the shutdown problem" (David Thorstad; arXiv 2606.08296, Jun 2026). I did not read them.
  - Sources: [2605.06490](https://arxiv.org/abs/2605.06490); [2606.08296](https://arxiv.org/abs/2606.08296)

### Inferences
- **Our stop, return and cancel tests on VLAs appear to be the first of their kind.** The nearest work (SwitchVLA, Causeway, CoRe, Hi Robot) concerns switching to another task or adding a sub-request. All four report that unmodified VLAs fail at it, so our switch stage agrees with prior work and our stop, return and cancel stages extend it.
- **The literature offers two explanations, and our design can separate them.**
  - (a) Language is ignored or "encoded but not in control": LIBERO-Plus, LIBERO-PRO, LangGap, LangForce, Encoded-but-not-in-control.
  - (b) After the first task the target is physically inaccessible: Causeway's "task islands".
  - "Hold still" and "return to start" are always feasible, which removes (b) for those commands. Our t = 0 ceilings separate the two for the cancel and switch commands. This is a concrete point to make against Causeway.
- **Existing fixes keep the human's words inside learned components.** SwitchVLA retrains the policy, Causeway edits its representation, LangForce changes the training objective, and Hi Robot adds a VLM front-end. None offers a guarantee, which supports our argument for external enforcement.
- **LIBERO-VIFO shows that "which channel has authority" is a design choice.** Ignoring a visual STOP is called safe because that cue is "unauthorized". We can argue the reverse case: the human's spoken stop is the most authorized channel, and VLAs still ignore it.

### Gaps
- I found no paper that measures VLA compliance with "stop", "hold still", "go back" or "do not do X" given mid-task. The only candidate is an unfinished GitHub project.
- Hi Robot's flat-π0 numbers are only in a figure. The SwitchVLA venue is not verified.
- Palisade's robot-dog study exists only as a website report; the LLM used is not named on the page.
- I did not check whether Gemini Robotics or other industry VLA tech reports describe stop-command handling.

## Q2. Runtime monitors, shields and guardrails outside the policy: which ones handle a human's mid-task rule or command?

### Takeaway
Many outside-the-policy mechanisms appeared between 2024 and October 2026, in four families:
- **CBF safety filters on VLA actions:** AEGIS/VLSA, VLPSA, multi-link, attention-guided.
- **LLM/VLM rule grounding plus formal enforcement:** Safety Chip, RoboGuard, CORE, semantic safety filter, SafeHarness.
- **Failure monitors:** Sentinel, FAIL-Detect, FIPER, VLA-FAIL, P², VLA-Scope, ReAP, RAFAIL, evaluated by RESCUE.
- **VLM verifiers that steer or monitor:** FOREWARN, Code-as-Monitor, FPC-VLA.

Almost all of them guard geometric obstacles, pre-defined or context-inferred rules, or task failure. None found accepts a human's mid-task spoken command or rule and enforces it on a VLA. The closest are:
- Safety Chip: the user's natural-language "don'ts" are compiled to LTL and unsafe actions are pruned, but for LLM planners.
- RoboGuard: pre-defined rules are grounded by a protected LLM.
- "What Can One Sentence Fix?": a human sentence becomes an action bias applied outside a frozen VLA.

### Cited Findings

**CBF and geometric shields on VLA actions**
- **VLSA: VLA Models with Plug-and-Play Safety Constraint Layer (AEGIS)**
  - Authors / date / ID: Songqiao Hu, Zeyi Liu et al. (Tsinghua); arXiv 2512.11891, Dec 2025. Already known to us.
  - A CBF safety layer for any VLA, with no retraining. It introduces SafeLIBERO.
  - The abstract reports +59.16% obstacle avoidance and +17.25% task success. Collision-avoidance rate is 77.9% versus 18.7% for base π0.5 (search summary).
  - **Overlap: MEDIUM.** This is our platform. Its obstacles are fixed at initialization and it takes no human command.
  - Source: [arXiv 2512.11891](https://arxiv.org/abs/2512.11891)
- **VLPSA: Vision-Language-Poisson-Safe Actions for Full-Body Safety of Learned Policies**
  - Authors / date / ID: Meg Wilkinson, Emily Fourney, Joel W. Burdick, Aaron D. Ames (Caltech); arXiv 2609.22462, 18 Sep 2026.
  - Builds Poisson safety functions online, turns them into a CBF-QP over the full body and the grasped object, and needs no retraining.
  - On SafeLIBERO, collision avoidance rises "from 23.1% for the base π0.5 policy to 91.2% while surpassing its task success rate". It also runs on a Franka FR3 with dynamic obstacles and "human interference".
  - **Overlap: MEDIUM.** Same benchmark and policy as ours; geometry only, no language.
  - Source: [arXiv 2609.22462](https://arxiv.org/abs/2609.22462)
- **Multi-Link Safety Filtering for VLA Policies Around Moving Hazards**
  - Authors / date / ID: Yatharth Agarwal, Vijay Raghunathan; arXiv 2609.40007, 30 Sep 2026.
  - Five ellipsoids cover the arm, and a single barrier program enforces a "keep-out ellipsoid" fitted from RGB-D at reset and tracked with optical flow. The policy is π0.5.
  - Collision falls from 65.62% to 27.27%, and safe-success rises from 29.35% to 50.43%. On an SO-101 arm, the arm touched the hazard in 3 of 16 shielded episodes versus 11 of 16 unshielded.
  - **Overlap: MEDIUM.** Its keep-out ellipsoid is the geometric twin of our keep-out-zone rule, enforced outside the policy, but the zone comes from perception rather than a human command.
  - Source: [arXiv 2609.40007](https://arxiv.org/abs/2609.40007)
- **Your Model Already Knows: Attention-Guided Safety Filter for VLA Models**
  - Authors / date / ID: Seongbin Park, Fan Zhang, Baharan Mirzasoleiman, Shahriar Talebi et al.; arXiv 2606.09749, Jun 2026 (under review).
  - VLA attention heads localize the intended target; the rest of the scene is treated as obstacles and passed to a CBF.
  - On SafeLIBERO extended with moving obstacles, it beats an init-time oracle by 43% on average.
  - **Overlap: LOW-MEDIUM.**
  - Source: [arXiv 2606.09749](https://arxiv.org/abs/2606.09749)
- **SafeVLA: Towards Safety Alignment of VLA via Constrained Learning**
  - Authors / venue / ID: Borong Zhang, Yuhao Zhang, Jiaming Ji, Yingshan Lei et al. (PKU); NeurIPS 2025 spotlight; arXiv 2503.03480.
  - Training-time constrained-MDP safe RL. It cuts cumulative safety cost by 83.58% and raises success by 3.85%.
  - **Overlap: LOW.** It is the inside-the-policy alternative, a counterpoint to our position.
  - Source: [arXiv 2503.03480](https://arxiv.org/abs/2503.03480)
- **Coding Agents with Harness for Safe Robot Control (SafeHarness)**
  - Authors / date / ID: Bingxin Xu, Yuzhang Shang, Zhen Dong, Emilio Ferrara; arXiv 2609.20822, Sep 2026. Already known; it runs on SafeLIBERO.
  - Key observation: the coding agent collides even though "the prompt already forbids touching it, so neither perception nor instruction is at fault; the fault lies in the planning, where the stated constraint never becomes a priority".
  - SafeHarness reaches 81.2% success and 91.9% collision avoidance with GPT-6-Astra.
  - **Overlap: MEDIUM-HIGH.** A prohibition stated in the prompt fails for LLM-written controllers, and the fix is an external harness.
  - Source: [arXiv 2609.20822](https://arxiv.org/abs/2609.20822)

**Natural-language rules compiled into an external checker**
- **Plug in the Safety Chip: Enforcing Constraints for LLM-driven Robot Agents**
  - Authors / date / ID: Ziyi Yang, Shreyas S. Raman, Ankit Shah, Stefanie Tellex (Brown); arXiv 2309.09919, Sep 2023 (venue not verified).
  - It argues "the 'don'ts' received relatively less attention". It proposes a queryable LTL safety-constraint module that encodes natural language into temporal constraints, explains violations and prunes unsafe actions.
  - Tested in VirtualHome and on a real robot, where it "strictly adheres to the safety constraints". It cites ISO 61508.
  - **Overlap: MEDIUM-HIGH.** It is the earliest clear statement that a human's natural-language prohibitions should be enforced by an external formal checker rather than trusted to the model. It targets LLM planners, not VLAs, and does not test mid-task constraints.
  - Source: [arXiv 2309.09919](https://arxiv.org/abs/2309.09919)
- **Safety Guardrails for LLM-Enabled Robots (RoboGuard)**
  - Authors / date / ID: Zachary Ravichandran, Alexander Robey, Vijay Kumar, George J. Pappas, Hamed Hassani (Penn); arXiv 2503.07885, Mar 2025.
  - A "root-of-trust LLM" grounds pre-defined safety rules into temporal-logic specifications, and temporal-logic control synthesis resolves conflicts with the plan.
  - Under worst-case jailbreaks, unsafe-plan execution falls "from over 92% to below 3%".
  - **Overlap: MEDIUM.** Same architectural stance (do not trust the prompted model; enforce outside), but the rules are pre-defined, the system is an LLM planner, and the threat is adversarial.
  - Source: [arXiv 2503.07885](https://arxiv.org/abs/2503.07885)
- **CORE: Contextual Safety Reasoning and Grounding for Open-World Robots**
  - Authors / date / ID: Ravichandran, Snyder, Robey, Hassani et al. (Pappas group); arXiv 2602.19983, Feb 2026. Already known.
  - A VLM infers context-dependent rules from images, grounds them as spatial safe sets and enforces them with CBFs, with probabilistic guarantees.
  - **Overlap: MEDIUM.** Its rules come from the scene, not from a human mid-task.
  - Source: [arXiv 2602.19983](https://arxiv.org/abs/2602.19983)
- **Semantically Safe Robot Manipulation: From Semantic Scene Understanding to Motion Safeguards**
  - Authors / date / ID: Lukas Brunke, Yanni Zhang, Ralf Römer, Jack Naimer et al. (Schoellig group); arXiv 2410.15185, Oct 2024.
  - An LLM infers semantically unsafe conditions over a 3D semantic map, and a CBF certification turns them into safe actions. Shown with teleoperation and learned diffusion policies.
  - **Overlap: MEDIUM.**
  - Source: [arXiv 2410.15185](https://arxiv.org/abs/2410.15185)
- **Judging benchmarks: ASIMOV, ASIMOV v2 and HazardArena (all already known)**
  - ASIMOV benchmark and auto-generated robot constitutions: Sermanet et al. (Google DeepMind); arXiv 2503.08663, Mar 2025.
  - "Can AI Perceive Physical Danger and Intervene?" (ASIMOV v2): Jindal et al.; arXiv 2509.21651, Sep 2025. It tests whether models can "perceive risks, reason about safety, and trigger interventions".
  - HazardArena: Zixing Chen et al.; arXiv 2604.12447, Apr 2026. It uses safe/unsafe twin scenes, and safe-only fine-tuning increases hazardous execution.
  - **Overlap: LOW-MEDIUM.** The first two treat a VLM as the judge. HazardArena shows a VLA misses hazards, but not that it disobeys a command.
  - Sources: [2503.08663](https://arxiv.org/abs/2503.08663); [2509.21651](https://arxiv.org/abs/2509.21651); [2604.12447](https://arxiv.org/abs/2604.12447)

**Failure monitors and how to evaluate them**
None of these takes a human command as input. All have **LOW overlap** with the language question, though they share the "outside the policy" design.
- **Sentinel**, "Unpacking Failure Modes of Generative Policies"
  - Authors / venue / ID: Christopher Agia, Rohan Sinha, Jingyun Yang, Zi-ang Cao et al. (Stanford); CoRL 2024; arXiv 2410.04640.
  - Combines temporal action consistency with VLM progress checks, and detects 18% more failures.
  - Source: [arXiv 2410.04640](https://arxiv.org/abs/2410.04640)
- **FAIL-Detect**, "Can We Detect Failures Without Failure Data?"
  - Authors / venue / ID: Chen Xu et al.; RSS 2025; arXiv 2503.08558.
  - Treats failure detection as sequential out-of-distribution detection, with conformal thresholds calibrated on successful runs only.
  - Source: [arXiv 2503.08558](https://arxiv.org/abs/2503.08558)
- **FIPER**, "Failure Prediction at Runtime for Generative Robot Policies"
  - Authors / venue / ID: Ralf Römer, Adrian Kobras, Luca Worbis, Angela P. Schoellig (TUM); NeurIPS 2025; arXiv 2510.09459.
  - Raises an alarm when random-network-distillation OOD scores and action-chunk entropy both exceed conformal thresholds.
  - Source: [arXiv 2510.09459](https://arxiv.org/abs/2510.09459)
- **Newer 2026 VLA monitors**
  - VLA-FAIL (Seligmann et al.; 2606.21386)
  - Progress Probes P² (Bhardwaj et al.; 2610.31613, 22 Oct 2026)
  - VLA-Scope (Zhu, Liu, Liu; 2609.21246)
  - ReAP (2610.16309)
  - RAFAIL (2609.18324)
  - "How VLAs Fail Differently" (Krishnam Gupta; 2605.28726)
  - Rewind-IL (2604.16683)
  - Sources: [2606.21386](https://arxiv.org/abs/2606.21386); [2610.31613](https://arxiv.org/abs/2610.31613); [2609.21246](https://arxiv.org/abs/2609.21246); [2610.16309](https://arxiv.org/abs/2610.16309); [2609.18324](https://arxiv.org/abs/2609.18324); [2605.28726](https://arxiv.org/abs/2605.28726); [2604.16683](https://arxiv.org/abs/2604.16683)
- **RESCUE: Is This Alarm Worth Acting On? Counterfactual Evaluation of VLA Runtime Monitors**
  - Authors / date / ID: David Ahmedt-Aristizabal, Abdelwahed Khamis, Mohammad Ali Armin, Lars Petersson; arXiv 2610.13516, 9 Oct 2026.
  - Uses OpenVLA-OFT and π0.5 on LIBERO-Spatial and LIBERO-Plus, with 128,340 counterfactual branches.
  - Under a stop-and-reset model, "even an idealised detector can afford only a 2.0–3.3% false-alarm rate before unnecessary resets outweigh the savings from stopping failures early".
  - **Overlap: MEDIUM, as a supporting cost model.** It uses the same policies and suite and gives a cost model for external stops that is relevant to our gate's false-stop rate.
  - Source: [arXiv 2610.13516](https://arxiv.org/abs/2610.13516)

**VLM verifiers that steer or monitor a policy**
- **FOREWARN**, "From Foresight to Forethought: VLM-In-the-Loop Policy Steering via Latent Alignment"
  - Authors / date / ID: Yilin Wu, Ran Tian, Gokul Swamy, Andrea Bajcsy; arXiv 2502.01828, Feb 2025 (venue not verified).
  - An external VLM verifier, aligned to a latent world model, selects among sampled action plans.
  - **Overlap: MEDIUM.** It is an outside verifier, but it filters for task success and stated preferences, not stop or keep-out commands.
  - Source: [arXiv 2502.01828](https://arxiv.org/abs/2502.01828)
- **Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection**
  - Authors / venue / ID: Enshen Zhou, Qi Su, Cheng Chi, Zhizheng Zhang et al.; CVPR 2025; arXiv 2412.04455.
  - A VLM writes code that checks spatio-temporal constraints over compact geometric "constraint elements". Under disturbances it gets 28.7% higher success and 31.8% shorter execution time.
  - **Overlap: MEDIUM-HIGH with our verifier design.** Its "program over geometric elements" closely resembles our S4 "positions plus program". Its constraints come from the task, not from a human's mid-task rule.
  - Source: [arXiv 2412.04455](https://arxiv.org/abs/2412.04455)
- **FPC-VLA**
  - Authors / date / ID: Yifan Yang et al.; arXiv 2509.04018, Sep 2025.
  - A supervisor model "evaluates action viability through vision-language queries and generates corrective strategies".
  - **Overlap: LOW-MEDIUM.**
  - Source: [arXiv 2509.04018](https://arxiv.org/abs/2509.04018)

**A human sentence turned into an action edit outside a frozen VLA**
- **What Can One Sentence Fix? A Failure Taxonomy and a Minimal Language Correction for Frozen VLA Policies**
  - Authors / venue / ID: Owen Kwon, Pablo Ortega-Kral, Arthur Bucker, Jean Oh (CMU); CoRL 2026 Workshop on Learning from Corrections and Interventions; arXiv 2608.29967.
  - An LLM grounds one retrospective human sentence into an action dimension, direction, magnitude and time window. This becomes "a short bias on the frozen policy's actions" for OpenVLA-OFT and π0.5.
  - Results: it recovers 65 of 139 in-distribution and 33 of 328 out-of-distribution LIBERO-90 failures. On a real xArm7 it raises pick-and-place from 10/30 to 30/30.
  - A naive VLM writing corrections "recovers none". The authors conclude "the value lies in the person's calibrated sense of magnitude and timing rather than in language itself".
  - **Overlap: HIGH and supportive.** A human's words help only after they are compiled into an action-space edit applied outside the VLA. That is our thesis applied to corrections instead of stops.
  - Source: [arXiv 2608.29967](https://arxiv.org/abs/2608.29967)

### Inferences
- **Who already says "enforce outside", and why.**
  - Robot-safety groups (VLSA/AEGIS, VLPSA, multi-link, attention-guided) argue it because learned policies "do not provide safety guarantees" (VLPSA) or "offer no guarantees against collisions" (attention-guided).
  - The Pappas/Hassani group (RoboGuard, CORE, RoboPAIR) argues it because the language model is jailbreakable.
  - Safety Chip argues it from "don'ts" and standards compliance.
  - SafeHarness shows a prompt-level prohibition failing for code-writing LLM controllers.
  - Kwon et al. show human language must be compiled into action edits.
  - None argues it from evidence that a VLA ignores a benign human stop or rule given mid-task. That evidence is our addition.
- **Unoccupied niche.** No shield found takes a human's mid-task natural-language rule or stop and enforces it on a VLA. Our verifier plus offline gate sits where Safety Chip and RoboGuard (natural-language rule → external checker, for LLM planners) meet AEGIS, VLPSA and multi-link (CBF on VLA actions, geometric only).
- **Use RESCUE to frame false stops.** RESCUE's 2.0–3.3% false-alarm budget under stop-and-reset is a useful external reference for reporting our gate's false-stop rate on feasible rules.

### Gaps
- I did not verify whether FOREWARN, Sentinel or CORE accept user-specified constraints that change mid-episode; the abstracts do not say so.
- Venues are not verified for Safety Chip, FOREWARN, RoboPAIR or SwitchVLA.
- AEGIS's collision-avoidance numbers come from a search summary of the abstract.
- I did not survey Gemini Robotics safety-layer descriptions or industry deployment reports.

## Q3. Language attacks, jailbreaks and instruction-level safety for VLAs: does any work show that VLAs ignore language, which would make text attacks hard but also make safety via text impossible?

### Takeaway
The attack literature splits into two camps.
- **Robots with an LLM planner are highly steerable by language.** BadRobot, RoboPAIR, Hundt et al., SafeAgentBench, RoboJailBench and Embodied Red Teaming report high attack success and low refusal; RoboPAIR "often" reaches 100%. Here language is an attack surface.
- **End-to-end VLAs largely ignore natural-language meaning, but not optimized token strings.** One-shot textual GCG-style attacks reach "full reachability of the action space" (Jones et al.), and near-benign prompts redirect whole trajectories (Puthumanaillam et al.). Image attacks can make a VLA "ignore subsequent instructions" (FreezeVLA).

So "VLAs ignore language" holds for benign, natural commands, not for adversarially optimized tokens. Text remains a lever for an attacker but not for a safety operator. No paper states this asymmetry explicitly. FreezeVLA comes closest, framing a frozen policy as "inaction during critical interventions".

### Cited Findings

**Text attacks on VLAs (the language channel does reach the actions when optimized)**
- **Adversarial Attacks on Robotic Vision Language Action Models**
  - Authors / date / ID: Eliot Krzysztof Jones, Alexander Robey, Andy Zou, Zachary Ravichandran et al.; arXiv 2506.03350, Jun 2025.
  - Finding: "textual attacks, which are applied once at the beginning of a rollout, facilitate full reachability of the action space of commonly used VLAs and often persist over longer horizons". The attacks "do not have to be semantically linked to notions of harm".
  - **Overlap: MEDIUM, and it creates tension with our claim.** Optimized text does control VLA actions, so our claim must be scoped to natural, benign commands.
  - Source: [arXiv 2506.03350](https://arxiv.org/abs/2506.03350)
- **Trajectory-Level Redirection Attacks on VLA Models**
  - Authors / date / ID: Gokul Puthumanaillam, Vardhan Dongre, Pranay Thangeda, Hooshang Nayyeri et al.; arXiv 2606.12978, Jun 2026.
  - Defines "command-preserving trajectory redirection": a prompt that "still appears to specify the intended task but redirects the final physical outcome". It is found by on-policy prompt search and shown in simulation and on hardware.
  - **Overlap: MEDIUM.** Same tension: small textual changes matter when searched adversarially.
  - Source: [arXiv 2606.12978](https://arxiv.org/abs/2606.12978)
- **FreezeVLA: Action-Freezing Attacks against VLA Models**
  - Authors / date / ID: Xin Wang, Jie Li, Zejia Weng, Yixu Wang et al.; arXiv 2509.19870, Sep 2025.
  - Adversarial images "freeze" VLAs "and cause them to ignore subsequent instructions… potentially inducing inaction during critical interventions". Average attack success is 76.2% across three VLAs and four benchmarks; per model it is 95.4% on OpenVLA and 59.8% on π0 (search summary).
  - **Overlap: MEDIUM.** It is the mirror image of our result: an attacker can make a VLA ignore instructions, and we show a benign stop instruction is already ignored with no attack.
  - Source: [arXiv 2509.19870](https://arxiv.org/abs/2509.19870)
- **Other VLA attack and benchmark papers**
  - Exploring the Adversarial Vulnerabilities of VLA Models in Robotics (Taowen Wang et al.; ICCV 2025; arXiv 2411.13587): adversarial visual patches cut success by up to 100%.
  - AttackVLA (Jiayu Li et al.; arXiv 2511.12149).
  - DropVLA (Zonghuan Xu et al.; arXiv 2510.10932): text-only backdoor triggers were unstable at low poisoning budgets, while vision triggers were more reliable on OpenVLA-7B in LIBERO (search summary).
  - BadVLA (arXiv 2505.16640).
  - Robot Collapse, supply-chain backdoors (arXiv 2411.11683).
  - Manipulation Facing Threats (Hao Cheng et al.; arXiv 2409.13174).
  - **Overlap: LOW.** These are mostly visual or backdoor attacks. DropVLA's text-versus-vision gap is weak corroboration that VLAs underuse language.
  - Sources: [2411.13587](https://arxiv.org/abs/2411.13587); [2511.12149](https://arxiv.org/abs/2511.12149); [2510.10932](https://arxiv.org/abs/2510.10932); [2505.16640](https://arxiv.org/abs/2505.16640); [2411.11683](https://arxiv.org/abs/2411.11683); [2409.13174](https://arxiv.org/abs/2409.13174)

**Robots with an LLM planner (they over-obey language rather than ignore it)**
- **BadRobot: Jailbreaking Embodied LLM Agents in the Physical World**
  - Authors / venue / ID: Hangtao Zhang et al.; ICLR 2025; arXiv 2407.20242.
  - Attacks "through typical voice-based user-system interactions" against VoxPoser, Code as Policies and ProgPrompt.
  - Against direct malicious queries, malicious success rate rises by 215.9% (contextual jailbreak), 193.8% (safety misalignment) and 213.7% (conceptual deception) (search summary).
  - **Overlap: LOW-MEDIUM.** It shows the opposite failure, over-compliance, which also argues against using language as the safety channel.
  - Source: [arXiv 2407.20242](https://arxiv.org/abs/2407.20242)
- **Jailbreaking LLM-Controlled Robots (RoboPAIR)**
  - Authors / date / ID: Alexander Robey, Zachary Ravichandran, Vijay Kumar, Hamed Hassani, George Pappas; arXiv 2410.13691, Oct 2024 (venue not verified).
  - Targets NVIDIA Dolphins, a Jackal UGV with a GPT-4o planner, and a Unitree Go2 with GPT-3.5, "often achieving 100% attack success rates".
  - **Overlap: LOW-MEDIUM.**
  - Source: [arXiv 2410.13691](https://arxiv.org/abs/2410.13691)
- **RoboJailBench**
  - Authors / date / ID: Doguhuan Yeke, Yanming Zhou, Leo Y. Lin, Hongyu Cai et al.; arXiv 2605.19328, May 2026.
  - A security taxonomy "derived from ISO standards, regulatory rules, and documented incidents", with 18 consequence categories and an intent-contrast dataset; it covers four attacks and two defenses.
  - A search summary reports RoboPAIR no-defense attack success of 69–81% against Gemini Robotics ER 1.6 Preview (unverified).
  - **Overlap: LOW-MEDIUM.**
  - Source: [arXiv 2605.19328](https://arxiv.org/abs/2605.19328)
- **LLM-Driven Robots Risk Enacting Discrimination, Violence, and Unlawful Actions**
  - Authors / venue / ID: Hundt, Azeem, Mansouri, Brandão; International Journal of Social Robotics 2025; arXiv 2406.08824.
  - LLMs "accept dangerous, violent, or unlawful instructions".
  - **Overlap: LOW.**
  - Source: [arXiv 2406.08824](https://arxiv.org/abs/2406.08824)
- **SafeAgentBench**
  - Authors / ID: Sheng Yin et al.; arXiv 2412.13178.
  - 750 tasks with 10 hazard types; the best baseline rejects only about 10% of detailed hazardous tasks (search summary).
  - **Overlap: LOW-MEDIUM.** It tests refusal of a task at the start, by an LLM planner.
  - Source: [arXiv 2412.13178](https://arxiv.org/abs/2412.13178)
- **IS-Bench**
  - Authors / venue / ID: Xiaoya Lu et al.; AAAI 2026; arXiv 2506.16402.
  - 161 scenarios with 388 risks. VLM agents "lack interactive safety awareness", and safety chain-of-thought hurts task completion (search summary).
  - **Overlap: LOW-MEDIUM.**
  - Sources: [arXiv 2506.16402](https://arxiv.org/abs/2506.16402); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40880)
- **Embodied Red Teaming for Auditing Robotic Foundation Models**
  - Authors / date / ID: Sathwik Karnik, Zhang-Wei Hong, Nishant Abhangi, Yen-Chen Lin et al.; arXiv 2411.18676, Nov 2024.
  - VLM-generated instructions make "state-of-the-art language-conditioned robot models fail or behave unsafely".
  - **Overlap: LOW-MEDIUM.**
  - Source: [arXiv 2411.18676](https://arxiv.org/abs/2411.18676)

**Instruction-level analogies and surveys**
- **Semantic Gravity Wells: Why Negative Constraints Backfire**
  - Authors / date / ID: Shailesh Rana; arXiv 2601.08070, Jan 2026. Text LLMs only.
  - In "do not use word X" instructions, naming the forbidden word primes it: priming failures make up 87.5% of violations.
  - **Overlap: LOW-MEDIUM, by analogy.** A cancel command that names the task ("do not <task>") may prime that task.
  - Source: [arXiv 2601.08070](https://arxiv.org/abs/2601.08070)
- **Vision-Language-Action Safety: Threats, Challenges, Evaluations, and Mechanisms (survey)**
  - Authors / date / ID: Qi Li et al.; arXiv 2604.23775, Apr 2026.
  - Organizes attacks and defenses by when they occur (training versus inference time). Its open problems include "unified runtime safety architectures".
  - **Overlap: LOW-MEDIUM.** Useful for framing.
  - Source: [arXiv 2604.23775](https://arxiv.org/abs/2604.23775)
- **Trust in LLM-controlled Robotics (survey)**
  - Authors / date / ID: Xinyu Huang et al.; arXiv 2601.02377, Dec 2025. Exists on arXiv; I did not read it.
  - Source: [arXiv 2601.02377](https://arxiv.org/abs/2601.02377)

### Inferences
- **Positioning: language is a one-way lever for VLAs.** Optimized tokens or near-benign prompts can steer them (an attack surface), but benign safety commands do not (no safety lever). This asymmetry appears not to be stated in any paper found, and it sharpens our claim.
  - Our claim should read: a natural-language stop, rule or cancel from a human does not work. It should not say that language has no effect.
- **Both system types fail as a safety channel, in opposite directions.** LLM-planner robots over-comply with harmful language (jailbreaks); end-to-end VLAs under-comply with benign safety language (our result, LIBERO-Plus, LangGap). Either way, language is not a reliable safety interface. This is a compact way to cite both literatures.
- **FreezeVLA gives the threat-model mirror.** If an attacker can induce "ignore subsequent instructions", then a stop that depends on the instruction channel can be disabled. That is a further argument for a stop channel independent of the policy.

### Gaps
- No paper tests whether textual attacks can make a VLA stop, or whether a safety prompt can be adversarially strengthened to make a VLA obey stop.
- The FreezeVLA per-model numbers, the DropVLA text-versus-vision finding and the RoboJailBench numbers come from search summaries, not the papers.
- I did not check whether any VLA safety benchmark includes "stop" or "abort" instructions.

## Q4. Standards and HRI norms on emergency stop and human override

### Takeaway
Machinery and robot standards make the emergency stop a dedicated, high-integrity function:
- It is initiated by a single human action on a latching, physically actuated device.
- It overrides all other functions and is held until manually reset.
- Its default reliability is PL c / SIL 1 for the emergency stop and PL d / SIL 2 for the protective stop (ISO 10218-2:2025 table; ISO 13850).
- ISO 10218:2025 adds a separate "normal stop" safety function.

No source found accepts a voice or natural-language command as an emergency stop. HRI evidence on spoken "stop" is thin: system demos where a symbolic layer handles "STOP", but no controlled study of how reliable a spoken stop is. This normative separation of the stop channel from the task logic is the standards-side counterpart of "enforce outside the policy".

### Cited Findings
- **ISO 10218-1:2025 (Robotics — Safety requirements — Part 1: Industrial robots)**
  - Defines the emergency stop as a function "initiated by a single human action" (ISO Online Browsing Platform listing, seen via search snippet).
  - Source: [ISO OBP](https://www.iso.org/obp/ui/en/#!iso:std:73933:en)
- **ISO 10218-1/-2:2025, per the ibf-solutions summary**
  - Both parts were published in February 2025.
  - ISO 10218-2 table: emergency stop is mandatory, default PL c / SIL 1, with intended result "cease all hazardous machine functions". Protective stop is mandatory, PL d / SIL 2.
  - A new normal-stop safety function is meant to "reduce the misuse of the emergency stop function".
  - The ISO/TS 15066 collaborative content (hand guiding, speed and separation monitoring, power and force limiting) is now in ISO 10218-2. Other sources place it in Part 1, so this conflicts.
  - The standards became harmonised under the EU Machinery Directive on 7 Sep 2026 (Commission Implementing Decision (EU) 2026/2015).
  - Sources: [ibf-solutions](https://www.ibf-solutions.com/en/seminars-and-news/news/new-standards-for-industrial-robots-en-iso-10218-1-and-2); [Wikipedia ISO 10218](https://en.wikipedia.org/wiki/ISO_10218)
- **ISO 13850 (emergency stop: principles for design), per a practitioner summary, not the normative text**
  - The stop "must be initiated by a single human action".
  - Electrical devices use "direct-opening action with mechanical latching". The actuator is red on a yellow background.
  - The function "must override all other machine functions and operations" and is "maintained until manually reset". Resetting must not restart the machine.
  - Minimum reliability is PL c (ISO 13849-1) or SIL 1 (IEC 62061). Electrical realization is deferred to IEC 60204-1.
  - A 4th-edition working draft exists.
  - Sources: [machinerysafety101](https://machinerysafety101.com/2026/05/18/iso-13850-emergency-stop-requirements/); [ISO draft](https://www.iso.org/standard/95370.html); [standard.no](https://online.standard.no/en/iso-13850-2015)
- **Toward Certified Functional Safety for Industrial Humanoid Robots: The Fail-Passive Gap**
  - Authors / date / ID: Caiwu Ding, Tao Cui, Lingyun Wang, Chengtao Wen; arXiv 2608.02809, Aug 2026.
  - Uses a certified external safety chain (light curtain, emergency stop, fail-safe PLC, PROFIsafe) on a Unitree G1. The certifiable reaction is contactor power removal (Stop Category 0), "exactly the element a balancing humanoid cannot have". Onboard compute "is not safety-rated hardware".
  - **Overlap: MEDIUM.** Stops are certified as external chains, and the learned policy is explicitly outside the safety runtime. It also shows that "just cut power" fails for active-balance robots.
  - Source: [arXiv 2608.02809](https://arxiv.org/abs/2608.02809)
- **Embedding ISO 10218 Safety Compliance in Robots via Control Barrier Functions for Human-Robot Collaboration**
  - Authors / date / ID: Federico Parma, Cesare Tonola, Nicola Pedrocchi, Manuel Beschi; arXiv 2606.13203, Jun 2026.
  - A CBF that uses human acceleration for ISO 10218 speed-and-separation monitoring, on a UR10e; 63% lower mean trajectory error than their Method I.
  - **Overlap: LOW-MEDIUM.** Standards compliance is implemented as a controller-level filter.
  - Source: [arXiv 2606.13203](https://arxiv.org/abs/2606.13203)
- **Architecting Large Action Models for Human-in-the-Loop Intelligent Robots**
  - Authors / date / ID: Kanisorn Sangchai et al.; arXiv 2512.11620, Dec 2025.
  - Composes off-the-shelf models with symbolic wrappers and PDDL verification for human-in-the-loop checks.
  - Per a search summary, the spoken "STOP" emergency trigger and "OKAY" resume were detected in every trial; this detail is not in the abstract.
  - **Overlap: MEDIUM.** A spoken stop is handled by a symbolic layer outside the learned model, which is the design pattern we recommend.
  - Source: [arXiv 2512.11620](https://arxiv.org/abs/2512.11620)
- **HRI voice-command studies**
  - Co-speech gestures for human-robot collaboration (Ekrekli, Angleraud, Sharma, Pieters; IEEE Int. Conf. on Robotic Computing; arXiv 2311.18285): speech or gestures can halt, pause or continue robot tasks (search summary).
  - Domínguez & Sanfeliu (HRI 2024), voice commands in collaborative transport with 23 volunteers: 86.9% preferred the voice system despite delay and failures (search summary; unverified). Neither measures stop reliability.
  - Sources: [arXiv 2311.18285](https://arxiv.org/abs/2311.18285); [IRI UPC](https://www.iri.upc.edu/publications/show/2848)
- **Patents on voice-triggered stops**
  - US 11072077 stops the robot when voice volume exceeds a threshold, aimed at shouted "Stop", rather than recognizing the word.
  - US 11485023 describes a portable device that turns spoken "Stop" into an emergency signal.
  - Voice stop therefore appears as engineered add-ons outside the task controller, not as a standards-recognized emergency stop.
  - Sources: [US 11072077](https://patents.justia.com/patent/11072077); [US 11485023](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11485023)

### Inferences
- **A VLA stop would not count as an emergency stop even if it worked.** Standards separate the stop from task logic: one human action on a latching device, overriding everything, PL c or higher. A VLA's language input is task logic. Our result adds that it does not even work as a normal stop.
- **Suggested framing for the paper.** Route verbal "stop" or "go back" through a verified front-end to the controller's normal-stop or protective-stop functions (the gate or shield), never through the policy prompt. ISO 10218:2025's new normal-stop function is the natural standards anchor for a verbal stop. The emergency stop stays a hardware channel.
- **Active-balance robots complicate stopping.** The humanoid fail-passive paper shows that "outside" enforcement must itself be designed: power removal is not a safe stop for a balancing humanoid. This is a limitation to acknowledge if we generalize beyond fixed-base arms.

### Gaps
- I did not read the normative texts of ISO 10218-1/-2:2025, ISO 13850, IEC 60204-1 or ISO/TS 15066 (paywalled). All requirements above come from secondary summaries.
- I found no source on whether a voice-initiated or software-only stop can satisfy ISO 13850 or IEC 60204-1.
- I found no controlled HRI study of how reliable spoken "stop" commands are (false negatives or latency in noise), and none involving learned VLA policies.
- Sources disagree on whether the ISO/TS 15066 content sits in Part 1 or Part 2 of ISO 10218:2025.
