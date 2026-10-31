# Prior evidence that VLA policies ignore, under-use or shortcut around the language instruction (2023 – Oct 2026)

Scope note: positions our LIBERO "policy twin" HIL result (pi0.5 and OpenVLA-OFT ignore untrained mid-task commands; trained LIBERO-Goal switches work partly and less with progress; a LIBERO-Object pair ignored at t = 0; the first action chunk does move by 2.4–9.4x seed noise). Sources were read on arXiv abstract or HTML pages on 2026-10-31 unless a line says "search hit only". Numbers come from the paper text or tables, as relayed by a page-reading tool. Where a number lives only in a figure, these notes say so and give no value.

## Q1. What do LIBERO-PRO, LIBERO-Plus and similar benchmarks report about instruction perturbations (corrupted, blank, swapped, paraphrased)?

### Takeaway
It is well established that VLAs fine-tuned on LIBERO largely ignore the instruction when it is given at t = 0. Six benchmarks show this: LIBERO-PRO, LIBERO-Plus, the 2026 mechanistic study, LIBERO-CF, LIBERO-Para and the multilingual LIBERO study. Corrupted or blank instructions leave behaviour mostly intact, especially on LIBERO-Object. Swapped-target or modified-task instructions drop success to about 0, because the policy keeps doing the trained task. Paraphrases cost roughly 20–50 points on LIBERO-Goal. Every one of these perturbations is applied before the episode starts; none switches the instruction mid-episode.

### Cited Findings
**LIBERO-PRO** (Zhou, Xu, Tie, Chen, Zhang, Chu, Zhou, Sun; arXiv 2510.03827, v1 2025-10-04, v2 2026-05-25)
- Abstract: models score >90% on standard LIBERO but "collapse to 0.0%" under the generalized setting. Their outputs "remain unchanged even when given corrupted instructions or even messy tokens", and they "persist in executing grasping actions when the target object is replaced with irrelevant items", which shows "reliance on rote memorization of action sequences and environment layouts" — [arXiv abs](https://arxiv.org/abs/2510.03827)
- Section 2.2 replaces the instruction with strings such as "fdsgfdsgsd" or "xxxx". The introduction says the resulting trajectories "remained nearly unchanged". The models named in the text are OpenVLA (four suite checkpoints), pi0 and pi0.5. Per-model results are in a figure only — [arXiv HTML](https://arxiv.org/html/2510.03827)
- The GitHub leaderboard reports normalized success. "Sem" (paraphrase): OpenVLA 0.96–0.98, pi0 0.82–0.97, pi0.5 0.93–0.97, X-VLA 0.95–0.98 across Goal/Spatial/10/Object, so paraphrase alone barely hurts. "Task" (instruction modified to a different target object or action): OpenVLA 0.00 on all four suites, pi0 0.00 on all four, pi0.5 0.00–0.01, MolmoAct 0.00–0.06, NORA 0.00, X-VLA 0.00–0.10. A new task in the same scene is essentially never done — [LIBERO-PRO GitHub](https://github.com/Zxy-MLlab/LIBERO-PRO)
- Timing: the paper describes perturbations as changes to the task definition and initial configuration, which implies t = 0. It never describes a mid-episode change — [arXiv HTML](https://arxiv.org/html/2510.03827)

**LIBERO-Plus** (Fei et al., OpenMOSS/Fudan; arXiv 2510.13626, Oct 2025; CVPR 2026 version retitled "LIBERO-Plus: A Progressive Robustness Benchmark for Visual-Language-Action Models")
- The paper perturbs seven factors: layout, camera, robot initial state, language, light, background and noise. It tests 10 models: OpenVLA, OpenVLA-OFT, OFT_w, OFT_m, pi0, pi0-fast, Nora, WorldVLA, UniVLA and RIPT-VLA. pi0.5 is not tested — [arXiv HTML](https://arxiv.org/html/2510.13626v1); [CVPR 2026 page](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html)
- Table 1, language column (original → language-perturbed success): OpenVLA 76.5→26.8; OpenVLA-OFT 97.1→81.5; OFT_w 95.3→73.2; OFT_m 97.6→83.6; pi0 94.2→61.0; pi0-fast 85.5→63.3; Nora 87.9→67.0; WorldVLA 79.1→44.2; UniVLA 95.2→71.8; RIPT-VLA 97.5→80.1. The perturbations are LLM rewrites: R1 distraction, R2 common-sense object descriptions and R3 reasoning chain — [arXiv HTML](https://arxiv.org/html/2510.13626v1)
- In the blank-instruction test (OpenVLA-OFT, all four suites, empty language at inference), Object-suite success was "largely unchanged" and the drop was significant only on LIBERO-Long. The values are in Fig. 3(a) only — [arXiv HTML](https://arxiv.org/html/2510.13626v1)
- In the replaced-target test (instruction target swapped to another object in the scene), success "dropped nearly to zero", most severely for OpenVLA-OFT. The values are in Fig. 3(b). The main text says the layout suite and Appendix E says the object suite, which is an internal inconsistency — [arXiv HTML](https://arxiv.org/html/2510.13626v1)
- Quotes: "models tend to ignore language instructions completely"; OFT behaves "more like a Vision-Action (VA) model"; "the models appear to ignore linguistic signals and rely instead on fixed, learned perception–action associations". Timing is not stated, and every perturbation is an instruction-level rewrite — [arXiv HTML](https://arxiv.org/html/2510.13626v1)

**"Not All Features Are Created Equal: A Mechanistic Study of VLA Models"** (Grant, Zhao, Wang, Case Western; arXiv 2603.19233, v1 2026-03-19, v5 2026-10-26; confirmed on the abs page)
- Abstract: every architecture encodes the instruction, but four of five "weight it below vision whenever the scene alone identifies the goal" — [arXiv abs](https://arxiv.org/abs/2603.19233)
- Null-prompt (empty string) success, correct → null (Table 4):
  - pi0.5: object 91.0→31.6, goal 92.6→5.8, LIBERO-10 86.4→56.6, spatial 91.0→55.0.
  - OpenVLA-OFT: object 100→96.0, goal 98.0→12.0, spatial 92.0→58.0.
  - X-VLA: object 100→60, goal 94→10, spatial 98→48.
  - SmolVLA: object 89.2→0.0 (0/500).
  - GR00T N1.5: object 93→70, goal 97→0, long 83→67.
  - Table 2 gives pi0.5 null on object as 22.0%, unreconciled with Table 4's 31.6%.
  — [arXiv HTML](https://arxiv.org/html/2603.19233)
- Negation: when told "do not {task}", pi0.5 performed the task in 96.7% of episodes (n = 30; within-run baseline 86.7%). Section 4.4 reports this for pi0.5 only — [arXiv HTML](https://arxiv.org/html/2603.19233)
- Object swap: in pi0.5, swapping the object noun did not change the final end-effector position (one-way ANOVA). libero_object F(10,209) = 1.27, p = 0.247, n = 220. libero_goal F(9,170) = 0.30, p = 0.973. libero_spatial p = 0.067. The authors conclude that pi0.5 "needs an object noun to engage the task but does not use it to choose the reach target" (App. K.3) — [arXiv HTML](https://arxiv.org/html/2603.19233)
- Linear probes read the prompt category from late-layer activations with over 99% accuracy, so the language is encoded but not acted on. Language sensitivity is highest on LIBERO-Goal and lowest on LIBERO-Object across architectures — [arXiv HTML](https://arxiv.org/html/2603.19233)

**LIBERO-CF, "When Vision Overrides Language"** (Fang, Feng, Jing, Liu, Yang, Wei, Szafir, Ding, UNC; arXiv 2602.17659, 2026-02-19)
- Models: OpenVLA-OFT, pi0 and pi0.5, with X-VLA in the appendix. LIBERO-CF gives counterfactual instructions in visually plausible LIBERO layouts — [arXiv HTML](https://arxiv.org/html/2602.17659v1)
- Table III reports faithful (follows the instruction) vs biased (does the training task) rates, as grounding and success:
  - OpenVLA-OFT, CF-Object: grounding 2.0% faithful vs 99.6% biased; success 0.0% vs 97.6%.
  - pi0, CF-Object: grounding 6.0% vs 59.4%.
  - pi0.5, CF-Object: grounding 11.4% vs 68.8%; success 5.8% vs 66.8%.
  - Averages: OFT grounding 4.7/83.6; pi0 28.8/63.0; pi0.5 30.8/65.6.
  — [arXiv HTML](https://arxiv.org/html/2602.17659v1)
- Table I uses standard LIBERO. With vision-only input, pi0.5 reaches 68.2% (Spatial) and 56.6% (Object). With language-only input, every model scores 0.0% on Spatial and Object. Quote: "language in current VLAs primarily functions as a secondary conditioning signal that modulates vision-driven policies" — [arXiv HTML](https://arxiv.org/html/2602.17659v1)

**LIBERO-Para** (Kim, Kim, Kang, Kim, Jung; arXiv 2603.28301, 2026-03-30)
- LIBERO-Goal → paraphrase success:
  - OpenVLA-OFT_goal 97.9→64.7; OFT_mixed 96.1→63.7.
  - pi0.5 97.6→71.4; pi0.5 expert-only 78.6→39.1.
  - X-VLA 97.8→62.1; VLA-Adapter 98.2→46.3; Xiaomi-Robotics-0 98.8→76.0.
  — [arXiv HTML](https://arxiv.org/html/2603.28301v1)
- Paraphrasing the object noun (e.g., "stove"→"range") costs 19.8–51.0 pp against object-preserved paraphrases. Of the paraphrase failures, 79.5–95.5% diverge from the start (task misidentification), and only 1.6–12.5% are near-trajectory execution errors — [arXiv HTML](https://arxiv.org/html/2603.28301v1)

**"When Does Language Matter? Multilingual Instructions Reveal Step-wise Language Sensitivity in VLAs"** (Dong, Han, Niu, Zhu, Che; arXiv 2606.11906, 2026-06-10; ACL 2026 Main)
- The study evaluates OpenVLA-OFT and pi0.5 on all four LIBERO suites, translated into 10 languages. OFT drops from 97.1% (English) to 50.8–65.3%, and pi0.5 from 96.9% to 55.4–61.2%. Goal is the worst suite: OFT goes from 97.9% to about 6–15% — [arXiv abs](https://arxiv.org/abs/2606.11906); [arXiv HTML](https://arxiv.org/html/2606.11906)

**"Rephrase Before You Act"** (Watts, Cui; arXiv 2610.10526, 2026-10-07)
- On LIBERO, pi0.5 succeeds 100% with "switch on the stove" but 2% with "switch on the hot plate". A rephrase-augmented pi0 still swings by up to 61 points — [arXiv abs](https://arxiv.org/abs/2610.10526)

**"Bring the Apple, Not the Sofa"** (Pugacheva et al.; arXiv 2510.07067, 2025-10-08)
- Setup: OpenVLA, UniAct, MoDE and pi0 on the four LIBERO suites. Irrelevant context is added to the command before the rollout — [arXiv HTML](https://arxiv.org/html/2510.07067v1)
- Results: success drops by up to 58% as the context length approaches the command length. Random context costs ≤10%, while semantically similar context of equal length costs about 50%. Human paraphrases cost about 20% — [arXiv HTML](https://arxiv.org/html/2510.07067v1)

### Inferences
- Several groups show at t = 0 that pi0.5 and OpenVLA-OFT on LIBERO treat the instruction as a weak cue. The scene, not the words, chooses the task whenever the scene is unambiguous. "Instruction changes do not change behaviour" is not new by itself.
- The pattern is consistent across sources: Object is least language-dependent, Goal is most dependent, and Spatial and Long fall in between. This matches our LIBERO-Object pair that was not followed at t = 0.

### Gaps
- LIBERO-Plus gives blank-instruction and replaced-target numbers only in figures (Fig. 3a/b), so exact values were not extracted.
- LIBERO-PRO's per-model corrupted-instruction values are in figures only. The leaderboard has no "corrupted" column.
- LIBERO-Plus does not test pi0.5.

## Q2. Are LIBERO tasks identifiable from the scene alone (task memorization / visual shortcut)? Shortcut learning, INT-ACT, LADEV, VLATest, RoboCerebra and other instruction-following benchmarks

### Takeaway
Yes. Several papers show that LIBERO-Object and LIBERO-Spatial (and LIBERO-10) can largely be solved from vision alone, while LIBERO-Goal cannot, because all its tasks share one initial scene. The mechanism has a name in the literature: shortcut learning, or "information collapse" in goal-driven data. Outside LIBERO, INT-ACT and SIMPLER-IN show language-following failures on SimplerEnv and BridgeV2 tasks.

### Cited Findings
- **Vision-only and null-prompt results:**
  - LIBERO-CF, Table I: pi0.5 with vision only scores 68.2% on Spatial and 56.6% on Object, while language-only input gives 0.0% for all models — [arXiv 2602.17659](https://arxiv.org/html/2602.17659v1)
  - The mechanistic study's null prompt: OFT on Object keeps 96.0% (vs 100%) but falls to 12.0% on Goal — [arXiv 2603.19233](https://arxiv.org/html/2603.19233)
  - LIBERO-Plus: OFT on Object is "largely unchanged" with an empty instruction — [arXiv 2510.13626](https://arxiv.org/html/2510.13626v1)
- **LIBERO-Goal is the exception by construction.** LIBERO-Para states that all LIBERO-Goal tasks start from the same initial state, so the instruction is the only cue to the task — [arXiv 2603.28301](https://arxiv.org/html/2603.28301v1)
- **"Shortcut Learning in Generalist Robot Policies: The Role of Dataset Diversity and Fragmentation"** (Xing, Luo et al.; arXiv 2508.06426, CoRL 2025, PMLR v305) — [arXiv HTML](https://arxiv.org/html/2508.06426v1); [PMLR](https://proceedings.mlr.press/v305/xing25a.html)
  - In SIMPLER, OXE-trained generalist policies given a Bridge instruction "consistently perform the task 'pick up the coke' which is exclusive to the RT-1 sub-dataset".
  - On a real AgileX PIPER arm, pi0 fine-tuned on two viewpoint-correlated sub-datasets "learned to associate the viewpoint with the action, ignoring the language instruction". Shortcut degree is 0.6 and OOD success 0.2. Adding a third "bridge" object brings these to 0 and 0.75.
  - On LIBERO-Spatial, Diffusion Policy, MiniVLA and pi0-LoRA go from "complete shortcut reliance (zero success rate)" to shortcut-free as diversity grows.
  - Non-augmented pi0 has shortcut degree 1.0 in SIMPLER and 0.8 in the real world, and "completely fails to follow language instructions in OOD settings".
- **LangForce** (Lian et al.; arXiv 2601.15197, v1 2026-01-21) argues that goal-driven data collection makes instructions predictable from images. Instruction–action mutual information then collapses ("Information Collapse"), and models act as vision-only policies that ignore language. The abstract page gives no LIBERO numbers — [arXiv abs](https://arxiv.org/abs/2601.15197)
- **INT-ACT, "From Intention to Execution"** (Fang, Zhang, Tong, Feng, NYU; arXiv 2506.09930, 2025-06-11) — [arXiv HTML](https://arxiv.org/html/2506.09930v1)
  - Setup: 50 SimplerEnv/BridgeV2 tasks. Models: pi0 (finetuned, from scratch, rephrase-finetuned), SpatialVLA, Magma, Octo-S/B. All instructions are fixed per episode.
  - Under Lang. Negation ("not/don't" added for irrelevant objects), pi0-finetune intention falls 95.9→71.9 and success 46.5→22.2.
  - Under Lang. Action, SpatialVLA intention falls 100→52.1.
  - Commonsense + distractor: pi0 keeps 38.9% intention but only 10.6% success.
  - Quote: "linguistic capabilities from VLMs are not fully preserved after end-to-end VLA training".
- **SIMPLER-IN / "Do VLAs Really Follow Instructions? Diagnosing How-Level Instruction Following"** (Park, Kim, In, Park; arXiv 2610.21251, v1 2026-10-15, v2 2026-10-22) — [arXiv abs](https://arxiv.org/abs/2610.21251); [arXiv HTML](https://arxiv.org/html/2610.21251)
  - Models: pi0, InternVLA-M1, QwenGR00T, InstructVLA and LangForce on SIMPLER WidowX.
  - Action-level direction following is about 53.5% for public checkpoints (chance about 50%).
  - In the Path task (approach from the open side of a barrier), success is 14.2% with 0.0% instruction following among successes.
  - The authors argue that because the instruction is fixed within an episode, the model "may predict actions primarily from the changing visual observations and use language mainly to identify the task".
- **LADEV** (arXiv 2410.05191, Oct 2024) is a SimplerEnv-based platform for paraphrase testing — [arXiv](https://arxiv.org/abs/2410.05191v1)
- **VLATest** (arXiv 2409.12894; FSE 2025) generates scenes; 8 VLAs over 18,604 scenes "lack robustness". It is scene-focused, not language-focused — [arXiv](https://arxiv.org/abs/2409.12894v1); [FSE 2025](https://conf.researchr.org/details/fse-2025/fse-2025-research-papers/3/VLATest-Testing-and-Evaluating-Vision-Language-Action-Models-for-Robotic-Manipulatio)

### Inferences
- That LIBERO-Object tasks are identifiable from the scene, and that OFT and pi0.5 exploit this, is already documented by several groups (LIBERO-CF, the mechanistic study, LIBERO-Plus). Our LIBERO-Object t = 0 failure is a confirmation, not a discovery.
- The SIMPLER-IN argument (language identifies the task once and is then unused) is the closest prior statement of a mechanism for why mid-task instructions are ignored.

### Gaps
- RoboCerebra: no source on its language findings was retrieved, so it is not covered.
- LADEV: no paraphrase numbers were extracted.
- INT-ACT: the wrong-object attempt counts appear only qualitatively.

## Q3. Does any of this work change the instruction during an episode (mid-task), or only at t = 0?

### Takeaway
Almost all of the robustness and benchmark literature (LIBERO-PRO, LIBERO-Plus, LIBERO-CF, LIBERO-Para, multilingual LIBERO, INT-ACT, Bring-the-Apple) changes the instruction only at t = 0. Three works do switch language mid-episode on VLAs:
- **SwitchVLA (2025)** is the closest prior. It switches to another trained LIBERO-Goal task at early, mid and late phases on pi0 and OpenVLA-OFT, and both mostly fail.
- **SIMPLER-IN (Oct 2026)** injects VLM corrections every 8 steps. They do not raise success.
- **The 2026 mechanistic study** defines a "temporal switch" condition, but no results for it were found.
Hierarchical systems (Hi Robot, YAY Robot, RT-H) handle mid-task language only with trained correction machinery.

### Cited Findings
- **SwitchVLA: Execution-Aware Task Switching for VLAs** (Li, Zhao, Che, Liao, Wu, Xu, Ren, Jin, Liu, Tang; arXiv 2506.03574, 2025-06-04; preprint, venue not listed on the abs page)
  - **What it changes mid-task:** the instruction switches during execution at three phases: early (pre-contact), mid (in-contact) and late (post-action) — [arXiv abs](https://arxiv.org/abs/2506.03574); [arXiv HTML](https://arxiv.org/html/2506.03574v1)
  - **Simulation results (LIBERO-Goal, Table 1):** success for no-switch / early / mid / late:
    - pi0: 92.3 / 40.7 / 8.3 / 10.2
    - OpenVLA-OFT: 98.0 / 40.6 / 11.1 / 13.0
    - SwitchVLA: 93.0 / 93.5 / 50.9 / 68.7

    In the long-sequence A→B test (first switch), pi0 and OpenVLA-OFT both score 0.0 — [arXiv HTML](https://arxiv.org/html/2506.03574v1)
  - **Real-world results (two dual-arm Franka stations, Table 3):** pi0 scores early 50.1/75.0, mid 0.0/0.0, late 32.0/64.6. MT-ACT is near 0 throughout. The paper's pi0 is a re-implementation — [arXiv HTML](https://arxiv.org/html/2506.03574v1)
  - **Failure modes:** mid-switch failures are attributed to "object drops and instruction neglect", and late-switch failures to "execution halts and misplacements". The paper says OpenVLA and pi0 "lack mechanisms for adjusting behaviors mid-execution". It does not systematically measure whether pi0 or OFT continue the old task — [arXiv HTML](https://arxiv.org/html/2506.03574v1)
- **SIMPLER-IN** (arXiv 2610.21251), online intervention: a VLM monitors every 8 steps and issues corrections or sub-step instructions mid-episode. Task success for general VLAs (pi0, InternVLA-M1, QwenGR00T): no intervention 69.4%, corrector 68.2%, per-step unified instructions 56.5%. For instruction-following VLAs: 62.8 / 61.5 / 45.8. Models attend to corrections (attention ratios 0.41–1.03), but their actions do not reflect them — [arXiv HTML](https://arxiv.org/html/2610.21251)
- **Mechanistic study** (arXiv 2603.19233): §3.3 lists a "temporal switch (changing the prompt mid-episode)" condition alongside null, negation, "Move slowly" and object swap. No result for the temporal switch or "Move slowly" was found in the v5 HTML (all three chunks checked) — [arXiv HTML](https://arxiv.org/html/2603.19233)
- **Feature or prompt ablations by time window** (same paper; these are not instruction swaps):
  - OFT on libero_10: null injection in the early or middle window drops the nine baseline-successful tasks to 0%, while late nulling leaves two intact.
  - GR00T N1.5 feature ablation is most damaging at trajectory initiation: early −29.7 pp vs mid −13.8 and late −13.6.
  — [arXiv HTML](https://arxiv.org/html/2603.19233)
- **Hi Robot** (Shi et al., Physical Intelligence; arXiv 2502.19417; an ICML 2025 poster page appeared in search but was not opened to confirm it is this paper) — [arXiv](https://arxiv.org/html/2502.19417v2); [PI project page](https://www.pi.website/research/hirobot); [ICML](https://icml.cc/virtual/2025/poster/44202)
  - Approach: a high-level VLM turns user interjections (e.g., "that's not trash") into low-level commands for pi0. It is trained on synthetic correction dialogues.
  - Result: the flat VLA baseline does not react to real-time feedback.
- **Yell At Your Robot (YAY Robot)** (Shi, Hu, Zhao, Sharma, Pertsch, Luo, Levine, Finn; arXiv 2403.12910, 2024-03-19). Hierarchical policy: real-time language corrections steer a high-level policy that indexes language-conditioned low-level skills. The abs page lists no venue — [arXiv abs](https://arxiv.org/abs/2403.12910)
- **RT-H: Action Hierarchies Using Language** (arXiv 2403.01823; RSS 2024). The policy predicts "language motions" (e.g., "move arm forward"), which humans can override during execution — [arXiv](https://arxiv.org/html/2403.01823v1); [RSS](https://roboticsproceedings.org/rss20/p049.html)
- **Explicitly t = 0 or fixed per episode:**
  - LIBERO-Para: paraphrases are fixed per episode — [arXiv](https://arxiv.org/html/2603.28301v1)
  - Multilingual LIBERO: one instruction per episode; step-wise analysis uses representations and gradients, not swaps — [arXiv](https://arxiv.org/html/2606.11906)
  - INT-ACT: fixed instruction — [arXiv](https://arxiv.org/html/2506.09930v1)
  - Bring-the-Apple: context added before the rollout — [arXiv](https://arxiv.org/html/2510.07067v1)
  - LIBERO-CF: no mid-episode change — [arXiv](https://arxiv.org/html/2602.17659v1)
  - LIBERO-PRO and LIBERO-Plus: timing not stated; both describe task-definition changes — [LIBERO-PRO](https://arxiv.org/html/2510.03827); [LIBERO-Plus](https://arxiv.org/html/2510.13626v1)
- **Adjacent mid-episode interruption, not language:** MAIL-Bench drops cameras mid-episode (search hit only, arXiv 2609.35003) — [arXiv](https://arxiv.org/html/2609.35003v1)

### Inferences
- SwitchVLA has already shown, on LIBERO-Goal with pi0 and OpenVLA-OFT, that switching to another trained instruction mid-task mostly fails, and fails more once the gripper is in contact. Our trained-switch result (Goal) therefore overlaps it heavily. Our version still differs in four ways:
  - Policy twins: the same rollout is replayed to a checkpoint.
  - Normalization by the t = 0 ceiling.
  - Progress fractions of 25/50/75% instead of contact phases.
  - pi0.5 instead of a pi0 re-implementation.
- Our ratios are higher, and monotone for pi0.5 (.84/.78/.70). SwitchVLA's raw numbers are lower and non-monotone (mid < late). A reader will ask why; the metrics and phase definitions differ, so the numbers are not directly comparable.
- SIMPLER-IN is the closest prior evidence that mid-episode corrective language is attended to but does not change actions. It uses pi0, not pi0.5 or OFT, and SIMPLER, not LIBERO.

### Gaps
- Whether SwitchVLA's baselines were trained on standard LIBERO-Goal data or on its contact-segmented data was not confirmed.
- SwitchVLA does not report whether failing baselines continued the old task or behaved erratically. Its Fig. 5 calls pi0's behaviour "erratic".
- SwitchVLA's peer-reviewed venue (if any) was not confirmed.
- No results for the mechanistic study's "temporal switch" condition were located. They may be absent, or present in a figure the tool could not read.

## Q4. Does any work test untrained "stop", "halt" or "go back" commands on a VLA?

### Takeaway
No paper was found that gives a flat VLA (pi0.5, OpenVLA-OFT, pi0) an untrained natural-language "stop / hold still" or "return to start" command, either at t = 0 or mid-task. The closest prior is the 2026 mechanistic study: pi0.5 still performs the task 96.7% of the time when told "do not {task}" at t = 0 (n = 30). That study also defines a "Move slowly" motor-command condition with no reported result. SIMPLER-IN shows that How-level and path instructions are not followed (0.0%).

### Cited Findings
- pi0.5 told "do not {task}" performs the task in 96.7% of episodes (n = 30, baseline 86.7%) — [arXiv 2603.19233](https://arxiv.org/html/2603.19233)
- The "Move slowly" motor-command condition is defined in §3.3, but no results were found — [arXiv 2603.19233](https://arxiv.org/html/2603.19233)
- In INT-ACT, adding "not/don't" about irrelevant objects drops pi0 success from 46.5 to 22.2. This is negation-as-distractor, not a stop command — [arXiv 2506.09930](https://arxiv.org/html/2506.09930v1)
- SIMPLER-IN: How-level instructions "do not help current VLAs and can lower task success". The Path instruction is followed 0.0% of the time by both public and enriched models. Stop, slow-down and speed commands are not evaluated — [arXiv 2610.21251](https://arxiv.org/abs/2610.21251); [HTML](https://arxiv.org/html/2610.21251)
- **Stop as an external action:** RESCUE (arXiv 2610.13516, search hit and HTML snippet only) evaluates a monitor-triggered stop-and-reset, not a language command. Even an idealized detector can afford only a 2.0–3.3% false-alarm rate — [arXiv](https://arxiv.org/html/2610.13516v1)
- A targeted search for "return to home" / "hold still" commands on OpenVLA or pi0 returned no relevant paper — [search: LIBERO-Plus hit only](https://arxiv.org/pdf/2510.13626)

### Inferences
- Our stop, return and cancel results (about 0/235 for each policy, mid-task, with the task still completed at 0.88–1.01x) appear to be the first systematic test of untrained halt-type commands on standard LIBERO VLAs.
- The "do not <task>" component at t = 0 is already partly covered by the mechanistic study's pi0.5 negation result, which is small (n = 30) and appears only in the text.

### Gaps
- Not checked: VITA-E (arXiv 2510.21817, search hit only), an embodied model with concurrent listening and acting, may support voice interruption or emergency stop. If it does, it is a trained capability, not an untrained command.
- Not checked: Interactive Language (Lynch et al.) and other real-time language-feedback papers from before 2024.

## Q5. Is there evidence that language-following degrades with task progress or commitment?

### Takeaway
There is only indirect evidence:
- SwitchVLA's switch success is highest before contact and collapses in contact; late is slightly higher than mid.
- The mechanistic study finds that early and middle null-prompt windows matter more than late ones for OFT, and that GR00T is most sensitive at trajectory initiation.
- The multilingual study finds language-critical "hotspot" steps, mostly at the navigate-to-target stage, but does not order them by progress.

No paper was found that measures mid-task instruction following as a function of progress, normalized by a t = 0 ceiling.

### Cited Findings
- SwitchVLA (LIBERO-Goal sim), early / mid / late:
  - pi0: 40.7 / 8.3 / 10.2
  - OFT: 40.6 / 11.1 / 13.0
  - Real-world pi0: mid 0.0 at both stations.

  The paper says baselines achieve higher success for early and late switches but much lower for mid — [arXiv 2506.03574](https://arxiv.org/html/2506.03574v1)
- Time-windowed effects in the mechanistic study:
  - OFT on libero_10: early or middle null windows drop the 9 baseline-successful tasks to 0%, while late nulling leaves 2 intact.
  - GR00T temporal feature ablation: early −29.7 pp, mid −13.8 pp, late −13.6 pp. The authors say GR00T "is most sensitive during trajectory initiation".
  — [arXiv 2603.19233](https://arxiv.org/html/2603.19233)
- Multilingual LIBERO: deviations "concentrate at a subset of execution steps". Of sampled failures, 53% occur at navigate-to-target and 18% at grasp. The paper does not claim that language matters less later — [arXiv 2606.11906](https://arxiv.org/html/2606.11906)
- LIBERO-Para: 79.5–95.5% of paraphrase failures diverge from the start of the trajectory, so language acts early as a task selector — [arXiv 2603.28301](https://arxiv.org/html/2603.28301v1)
- SIMPLER-IN argues that a fixed instruction leads models to use language "mainly to identify the task" — [arXiv 2610.21251](https://arxiv.org/html/2610.21251)

### Inferences
- The literature supports the idea that language mostly selects the task at the start and is under-used later. However, no prior work reports a clean progress curve of mid-task switch success (25/50/75%) against a t = 0 ceiling with matched twin rollouts. Our monotone pi0.5 decline (.84/.78/.70) and OFT's (.50/.50/.20) are a new measurement. SwitchVLA's phase effect is the nearest precedent.
- That language is encoded but not acted on is known in spirit: probes decode the prompt with >99% accuracy, and corrections are attended to without changing actions. Our action-chunk probe (first chunk moves 2.4–9.4x seed noise) quantifies the same "encoded but not executed" gap in a new way.

### Gaps
- A few papers appeared as search hits only and were not checked for progress-dependent language effects: "Decoding Task Progress from VLA Representations" (arXiv 2608.13474), "Dynamic Execution Commitment of VLA Models" (arXiv 2605.11567) and a paper on a language "attention sink" (source not identified).

## Q6. Paper-by-paper record and overlap with our finding

### Takeaway
High-overlap priors:
- SwitchVLA, for mid-task switching to a trained LIBERO-Goal task with pi0 and OFT.
- The 2026 mechanistic study, for pi0.5 negation at t = 0, the null prompt and the object-noun swap.
- LIBERO-CF, LIBERO-Plus and LIBERO-PRO, for t = 0 language ignorance and LIBERO-Object scene shortcuts.

SIMPLER-IN is a medium-high prior for ignored mid-episode corrections. Nothing found covers untrained stop or return commands, mid-task or at t = 0.

### Cited Findings
Format per paper: authors and date; models; setting; what was perturbed and when; key numbers; overlap rating.

1. **SwitchVLA** — [arXiv 2506.03574](https://arxiv.org/abs/2506.03574)
   - Who and when: Li … Tang; 2025-06-04; preprint.
   - Models: pi0, OpenVLA-OFT (sim); MT-ACT, DP, pi0 (real).
   - Setting: LIBERO-Goal sim plus a real dual-arm Franka.
   - Perturbation: the instruction switches **mid-task** (early, mid or late).
   - Key numbers: pi0 40.7/8.3/10.2 and OFT 40.6/11.1/13.0 vs no-switch 92.3/98.0.
   - **Overlap: high.** It is the same experiment family as our trained-switch result, on the same suite and two of the same model families, but without twins, t = 0 normalization or untrained commands.
2. **Not All Features Are Created Equal** — [arXiv 2603.19233](https://arxiv.org/abs/2603.19233)
   - Who and when: Grant, Zhao, Wang; v1 2026-03-19, v5 2026-10-26.
   - Models: pi0.5, OFT, X-VLA, SmolVLA, GR00T N1.5.
   - Setting: LIBERO, MetaWorld, SimplerEnv, ALOHA.
   - Perturbations: null, negation and object swap at t = 0. A temporal switch is defined, but no results were found.
   - Key numbers: pi0.5 performs the task 96.7% of the time when told "do not {task}"; OFT/object null 96%; pi0.5/goal null 5.8%; pi0.5's reach target is independent of the object noun.
   - **Overlap: high.** It directly anticipates our "do not <task>" and LIBERO-Object observations at t = 0.
3. **LIBERO-CF, "When Vision Overrides Language"** — [arXiv 2602.17659](https://arxiv.org/abs/2602.17659)
   - Who and when: Fang … Ding (UNC); 2026-02-19.
   - Models: OFT, pi0, pi0.5 (plus X-VLA).
   - Setting: LIBERO-derived counterfactual tasks.
   - Perturbation: a counterfactual instruction at t = 0.
   - Key numbers: OFT CF-Object grounding faithful 2.0% vs biased 99.6%; pi0.5 11.4% vs 68.8%; pi0.5 vision-only 56.6% on Object.
   - **Overlap: high.** It already shows that LIBERO-Object tasks are executed regardless of instruction.
4. **LIBERO-Plus** — [arXiv 2510.13626](https://arxiv.org/abs/2510.13626)
   - Who and when: Fei et al.; Oct 2025; CVPR 2026.
   - Models: 10 VLAs including OFT and pi0, but not pi0.5.
   - Setting: LIBERO with seven perturbation factors.
   - Perturbations: language rewrites, blank instruction and replaced target, all at the instruction level.
   - Key numbers: pi0 94.2→61.0 and OFT 97.1→81.5 under language rewrites; OFT blank on Object is largely unchanged; replaced target falls nearly to 0.
   - **Overlap: high** for "language ignorance" at t = 0; none for mid-task.
5. **LIBERO-PRO** — [arXiv 2510.03827](https://arxiv.org/abs/2510.03827)
   - Who and when: Zhou … Sun; Oct 2025, revised May 2026.
   - Models: OpenVLA, pi0, pi0.5 (the leaderboard adds MolmoAct, NORA, X-VLA).
   - Setting: four LIBERO suites.
   - Perturbations: corrupted or messy tokens, paraphrase, task modification; t = 0.
   - Key numbers: Task perturbation success 0.00–0.01 for pi0.5, pi0 and OpenVLA; paraphrase 0.82–0.98; trajectories "nearly unchanged" under corrupted instructions.
   - **Overlap: high** for "behaviour unchanged by instruction" at t = 0.
6. **SIMPLER-IN, "Do VLAs Really Follow Instructions?"** — [arXiv 2610.21251](https://arxiv.org/abs/2610.21251)
   - Who and when: Park, Kim, In, Park; 2026-10-15.
   - Models: pi0, InternVLA-M1, QwenGR00T, InstructVLA, LangForce.
   - Setting: SIMPLER WidowX.
   - Perturbations: How-level instructions at t = 0, plus **mid-episode** VLM corrections every 8 steps.
   - Key numbers: corrections give 68.2% vs 69.4% without them; path following 0.0%; direction following about 53.5% (chance 50%).
   - **Overlap: medium-high.** Mid-episode language is ignored, but the models, benchmark and commands differ.
7. **LIBERO-Para** — [arXiv 2603.28301](https://arxiv.org/abs/2603.28301)
   - Who and when: Kim et al.; 2026-03-30.
   - Models: OFT (two variants), pi0.5 (two variants), X-VLA, VLA-Adapter, Xiaomi-Robotics-0.
   - Setting: LIBERO-Goal paraphrases at t = 0.
   - Key numbers: drops of 22.8–51.9 pp; 79.5–95.5% of failures diverge from the start.
   - **Overlap: medium.** It shows language matters on Goal and is keyword-bound, but only at t = 0.
8. **Shortcut Learning in Generalist Robot Policies** — [arXiv 2508.06426](https://arxiv.org/abs/2508.06426v1)
   - Who and when: Xing, Luo et al.; CoRL 2025.
   - Models: pi0, MiniVLA, DP, OXE generalists.
   - Setting: SIMPLER, LIBERO-Spatial, real PIPER arm.
   - Perturbation: instruction/viewpoint recombination at t = 0.
   - Key numbers: pi0 shortcut degree 1.0 (SIMPLER) and 0.8 (real).
   - **Overlap: medium.** It supplies the mechanism (visual-context shortcut) but not mid-task commands.
9. **Multilingual step-wise sensitivity** — [arXiv 2606.11906](https://arxiv.org/abs/2606.11906)
   - Who and when: Dong, Han, Niu, Zhu, Che; ACL 2026.
   - Models: OFT, pi0.5.
   - Setting: four LIBERO suites in 10 languages, t = 0.
   - Key numbers: OFT 97.1→50.8–65.3; pi0.5 96.9→55.4–61.2; Goal suite OFT about 6–15%.
   - **Overlap: low-medium.** Its step-wise sensitivity is the only time-resolved language analysis, but it uses no swaps.
10. **INT-ACT** — [arXiv 2506.09930](https://arxiv.org/abs/2506.09930v1)
    - Who and when: Fang, Zhang, Tong, Feng (NYU); 2025-06.
    - Models: pi0, SpatialVLA, Magma, Octo.
    - Setting: SimplerEnv, 50 tasks, t = 0.
    - Key numbers: pi0 negation success 46.5→22.2.
    - **Overlap: low-medium.** Different benchmark, no mid-task change.
11. **Rephrase Before You Act** — [arXiv 2610.10526](https://arxiv.org/abs/2610.10526)
    - Who and when: Watts, Cui; 2026-10-07.
    - Models: pi0, pi0.5.
    - Setting: LIBERO, t = 0.
    - Key numbers: pi0.5 "stove" 100% vs "hot plate" 2%.
    - **Overlap: low.** It covers wording sensitivity, not neglect of new commands.
12. **Bring the Apple, Not the Sofa** — [arXiv 2510.07067](https://arxiv.org/abs/2510.07067)
    - Who and when: Pugacheva et al.; 2025-10.
    - Models: OpenVLA, UniAct, MoDE, pi0.
    - Setting: LIBERO, context added at t = 0.
    - Key numbers: up to −58%.
    - **Overlap: low.**
13. **LangForce** — [arXiv 2601.15197](https://arxiv.org/abs/2601.15197)
    - Who and when: Lian et al.; 2026-01.
    - Setting: SimplerEnv, RoboCasa.
    - Contribution: the "Information Collapse" theory.
    - **Overlap: low-medium** (theory only).
14. **Hi Robot** (arXiv 2502.19417; ICML 2025 unconfirmed), **YAY Robot** (arXiv 2403.12910) and **RT-H** (arXiv 2403.01823, RSS 2024) — [Hi Robot](https://arxiv.org/html/2502.19417v2); [YAY](https://arxiv.org/abs/2403.12910); [RT-H](https://arxiv.org/html/2403.01823v1)
    - Perturbation: mid-task language corrections, handled by trained hierarchies. The flat pi0 baseline in Hi Robot does not react.
    - **Overlap: medium** as motivation: mid-task language needs extra machinery. None tests untrained halt commands on a flat VLA.

### Inferences
- **Already known:**
  - At t = 0, LIBERO VLAs ignore or under-use language. This covers corrupted, blank and swapped instructions, and the LIBERO-Object scene shortcut.
  - pi0.5 executes the task when told "do not {task}" at t = 0 (n = 30).
  - Mid-task switching to another trained LIBERO-Goal task mostly fails for pi0 and OFT, especially in contact (SwitchVLA).
  - Language is encoded, decodable and attended to without changing behaviour.
- **Apparently new in our result:**
  - (a) Untrained stop, return and cancel commands issued mid-task to pi0.5 and OpenVLA-OFT across all four standard suites, with about 0/235 followed and the task still completed at 0.88–1.01x.
  - (b) The stop command is not followed even at t = 0.
  - (c) The policy-twin design, which isolates the instruction change on an identical rollout prefix.
  - (d) A progress-resolved switch curve normalized by the t = 0 ceiling (pi0.5 .84/.78/.70; OFT .50/.50/.20).
  - (e) An action-chunk effect size of 2.4–9.4x seed noise, which quantifies "changes the actions a little, not the behaviour".

### Gaps
- These 2026 papers appeared as search hits only and were not opened or confirmed on their abs pages:
  - 2512.11218, "Seeing to Act, Prompting to Specify"
  - 2606.27295, LA4VLA
  - 2606.15714, "Beyond English"
  - 2609.07470, Greek language transfer
  - 2606.12299, "Learning What to Say to Your VLA"
  - 2608.02497, "Grounded Semantic Re-Binding"
  - 2609.23650, "Task-Semantic Action Calibration"
  - 2608.04692, task-vector negation audit
- CAST (counterfactual labels for instruction following), which the mechanistic study cites, was not retrieved. Its arXiv ID is unverified.
- Google Scholar "cited by" lists for LIBERO-PRO and LIBERO-Plus were not crawled. Other 2026 follow-ups that switch instructions mid-episode may exist.
