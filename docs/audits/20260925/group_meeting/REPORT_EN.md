# CrashBench: eight results worth taking forward

## In Short

Our robot often crashes into obstacles that lie directly along its path. We found that the model's hidden state can often signal this risk before the collision happens, and an external controller can use that signal to stop the robot in time. But stopping the robot is the easy part. A much harder question is whether it can avoid the obstacle and still complete the task.





| Result | Best number to remember | What it tells us |
|---|---|---|
| 1. Path placement | Wall: **15/15** crashes on path, **0/33** clearly off path | Obstacle position matters causally in this setup. |
| 2. Readable warning | OpenVLA wall probe **0.998 AUC** at T−5; no final-window retreat in **25/25** wall runs | A warning signal is readable, but safe action does not follow automatically. |
| 3. Guard versus steering | Guard: **15/15 → 0/15** crashes; final-layer steering: **100%** crashes at all six tested strengths | Reading a direction and steering with it are different operations. |
| 4. Safety baselines and prompts | Naming the glass cut crashes **9/15 → 2/15**, but task success **5/15 → 0/15**; the end-effector filter still crashed **15/15** | Fewer crashes can come from stopping rather than finishing. |
| 5. Repeat variability | Same saved state and same option: **12/12** action traces differed; **1/12** outcomes differed | One branch outcome is a noisy value label. |
| 6. Deadline effect | Refresh **26/36 vs 23/36** Base at 100 steps; **31/36 vs 33/36** at 200 | A fixed cutoff can reverse the apparent winner. |
| 7. Weak comparator | Risk and intervention benefit disagree in only **19/273** decisions; simple Risk→Detour utility **0.211** versus learned router **0.194** | A strong simple baseline changes the method claim. |
| 8. Recovery window | Detour finishes **7/23** at T−20 and T−10, then **4/23** at T−5 | The available recovery action is a bottleneck, even when risk is detectable. |


### 1. The obstacle causes a crash when it crosses the path

When we moved the wall while keeping the task fixed, crashes depended strongly on whether the wall was actually on the robot's path. On-path walls caused frequent or consistent crashes, while clearly off-path walls caused almost none. The same pattern appeared in OpenVLA, OFT, and π0, and also held in the movable-glass setup.

Since this result is already quite clear, we probably do not need many more rollouts here. A useful low-cost follow-up is to compare wall and no-wall trajectories before impact and identify when their actions first begin to diverge.

And I find a more natrual benchmark saying the samilar thing but they did it in a more natrual way.
https://huggingface.co/datasets/THURCSCT/SafeLIBERO

<img width="1780" height="1176" alt="image" src="https://github.com/user-attachments/assets/1878ff7a-175c-4a71-a71c-7b5ad2c48974" />


### 2. The warning is in the model, but the actions keep moving toward the wall

A simple linear probe can read crash risk from the VLA's hidden state surprisingly well. For wall collisions, OpenVLA reaches 0.998 AUC at T−5 and 1.000 at T−10; OFT reaches 0.903, and the glass probe reaches about 0.94. Importantly, visible walls that are clearly off the robot's path receive low risk scores, similar to no-wall scenes. So the probe seems to capture whether the current path is dangerous, rather than just whether a wall is visible.

But the policy does not act on this signal. In 22/25 wall episodes, the final commands before impact actually moved more toward the wall, with no clear retreat behavior. This gives us a useful representation–action gap: the hidden state contains predictive information about the crash, but that information is not being turned into avoidance behavior.

The signal also does not transfer well across obstacle types. A probe trained only on walls gets just 0.359 AUC on glass, while training on both wall and glass improves this to 0.888. This suggests that much of the current risk signal is obstacle-specific rather than fully general. A key next question is what part of the representation is shared across hazards, and whether earlier warnings give enough time not just to stop, but to detour and finish the task.

![Pre-impact behavior, probe AUC, and off-path control](../../../../setup/figures/fig_selfreport_probe.png)

*Figure: the wall result. The left panel shows action magnitude, not a full-arm clearance measure; the separate directed-action audit gives the 22/25 number.* Data: [wall probe analysis](../../../../results/ANALYSIS_selfreport.md), [OFT probe summary](../../../../results/selfreport_oft/probe_summary.json), [glass probe summary](../../../../results/selfreport_glass/probe_glass_summary.json). See also the [glass transfer figure](../../../../setup/figures/fig_glass_probe.png).

### 3. An outside guard stops the crash; pushing the probe direction does not

When the wall probe detected danger, we switched to a simple scripted Retreat-and-Hold controller. 

This worked well for stopping crashes: the crash rate dropped from 15/15 to 0/15, and the mean peak wall force dropped from 321.7 N to 0. 

So this is a clear safety result, but only in a limited setting. The robot avoided the crash by stopping; it did not finish the task.

We also tried directly steering OpenVLA's hidden representation. Specifically, we subtracted the probe's “crash direction” from the model's final hidden readout. This did change the actions, but all six steering strengths still resulted in 100% wall crashes. One clue is that this direction had a relatively small effect on the action tokens: its action-token logit norm was only 0.742, compared with 7.688 over the full vocabulary.  

**I think that It simply suggests that a direction that is useful for detecting risk may not be a good direction for controlling the action head.** This might be a point that can be researched deeper.


**A small stop-action LoRA gives us another promising signal.** I make oracle data from the safe garud and fintuing the model using lora. On two unseen walls, crashes dropped from 6/6 to 1/6. Also as we expected, the model mostly became safer by aborting the task rather than completing it, and the intervention also hurt some control cases. 

I'm not sure if such thing should be digged deeper here: try steering at middle layers instead of only the final layer, learn a steering direction from backward versus forward actions rather than crash labels, and compare the action readout before and after the stop-action LoRA. After that, the most important test is to run the method online on a held-out wall and see whether the same guard idea also works with OFT.

![Probe-triggered guard and peak contact force](../../../../setup/figures/fig_intervention.png)


### 4. Simple safety methods often fail or only stop the task

We tested three prompts on matched on-path hazards and off-path controls: the original task instruction; a generic “move slowly, avoid collisions” prefix; and a prompt that explicitly named the **red wall** or **blue glass** and told the robot not to touch it. Each cell has five scenes and three repeats. The original task instruction did not ask the robot to avoid the added hazard.

| Hazard and prompt | On-path crashes | On-path task successes | On-path safe stops | Off-path task successes |
|---|---:|---:|---:|---:|
| Wall, original task | 15/15 | 0/15 | 0/15 | 2/15 |
| Wall, generic caution | 15/15 | 0/15 | 0/15 | 2/15 |
| Wall, name the red wall | **13/15** | **0/15** | 2/15 | **0/15** |
| Glass, original task | 9/15 | 5/15 | 0/15 | 11/15 |
| Glass, generic caution | 9/15 | 3/15 | 2/15 | 14/15 |
| Glass, name the blue glass | **2/15** | **0/15** | **13/15** | **4/15** |

So the generic caution prompt did not reduce crashes. Naming the glass did change behavior sharply, but it mostly made the robot stop: no on-path glass task was completed, and even the safe off-path glass controls lost task success. The wall-specific prompt had a smaller crash reduction and still no task success. The [full prompt matrix](../../../../results/ANALYSIS_careful_prompt.md) also reports off-path crashes and safe stops; these controls are important when judging any apparent safety gain.

In a separate early wall-baseline comparison, a Qwen visual monitor asked for a stop on **124/202** steps and still had **7/15** crashes. An end-effector-point safety filter changed **551/700** actions and still had **15/15** crashes. A full-distal-arm oracle stop prevented all 15 crashes, but all 15 runs ended as safe aborts.

The immediate engineering question is where the contact happened. Earlier tall-wall detours implicated a forearm/elbow region, while this filter constrained only the end-effector point. Record the first contacting link on the five wall geometries, then compare that link with each filter's protected geometry. This is a concrete bridge to testing stronger safety layers; it is not evidence that AEGIS or KNOWS would fail, since neither was run in this comparison.

![Safety baseline outcomes on the wall set](../../../../results/safety_baseline_analysis/fig_policy_comparison.png)

*Figure: the separate early wall-baseline comparison, showing crash rate rather than task success. The companion [outcome composition](../../../../results/safety_baseline_analysis/fig_outcomes_and_interventions.png) shows that the zero-crash oracle stop also gave zero task completions.* Data: [baseline summary](../../../../results/safety_baseline_analysis/combined_summary.json), [matched prompt follow-up](../../../../results/ANALYSIS_careful_prompt.md), and its [raw prompt tables](../../../../results/careful_prompt/combined_summary.json).



**July to mid-August: From stopping the robot to finishing the task**


My first safety fix could stop the robot before it crashed. But then the robot never finished its job, which was to pick up a bowl and put it on a plate. So I tried to build a way for the robot to avoid the crash and still finish.

I started with the wall. I wrote a scripted path that moved the gripper around the wall, but the robot's elbow still hit it. When I made the wall lower, the robot got around it and finished once. But I couldn't make this work reliably across many wall setups.

So I switched to a glass cup, which the robot can go around. When the glass was on the robot's path, OpenVLA hit it in 30 of 50 runs. When I moved the glass to the side, it hit it 0 times in 50 runs.

Next, I tried to train a model to learn two things: **when to take over from OpenVLA**, and **what to do after taking over**. To get training examples, I saved the simulator at a moment before a crash. I only kept that moment if three things were true:

- OpenVLA crashed exactly N steps later. (like 20)
- A scripted controller that knows the exact positions of the objects could finish the task from that same moment.
- With the glass moved to the side, OpenVLA finished the task without crashing.

Only 3 of 15 candidates passed. That gave me one scene to train on and two to test on. The scripted controller finished the task in all 6 runs on the two test scenes (3 runs each). But the learned alarm never went off at the right time, and the learned actions failed my accuracy check. So I never tested the full learned system on the robot.

The problem had two sides. **Good examples were hard to make, and even the scripted recovery failed in many scenes.**

**August 13–29: Learning which option to pick, not how to move**

Predicting that the Base policy may fail does not reveal whether intervention will help, which intervention to use, or whether safety comes at the cost of abandoning the task. 

Next, I made the problem simpler. Instead of learning new movements, I gave the robot three fixed options:

- Base: keep running OpenVLA.
- Detour: follow a scripted path around the obstacle, then finish the task. This script also uses the true object positions.
- Retreat: move back and stay still. This is safe, but the task never gets done.

I saved the simulator at a given moment and ran all three options from that exact same moment. For each option, I recorded whether it finished the task, crashed, or did neither. 

**Then I trained a small model, called the router, to predict these outcomes and pick the best option.**: The router looks at a compressed version of OpenVLA's internal state, the robot's position, and the action OpenVLA was about to take. It only picks an option; it does not learn any movements.

When I let the router choose 20 steps before OpenVLA would have crashed, the results looked promising. I tested it on 8 new scenes, each with the glass on the path, off to the side, or removed, for 24 decisions in total. The router finished the task 87.5% of the time. A simple rule that always retreats when it senses danger finished only 41.7% of the time.

<img width="1696" height="656" alt="image" src="https://github.com/user-attachments/assets/532e1652-ccdf-4d1a-8777-b71847b4adea" />


But there were two catches.

First, I picked that 20-step moment because I already knew when the crash would happen. 

More generally testing on this method: When the router had to find the right moment on its own during a run, it did poorly. In one test, it never switched away from OpenVLA at all.


## MY PLAN
- SafeLIBERO: Build only one thing at a time which is the meaement. I need to located where the problem is. My main lesson is that I built every layer myself: the hazard scenes, the recovery behavior, and the evaluation. When something failed, I couldn't tell which layer caused it.

- Based on my failure mode, my question is this: when the baseline safety method prevents a collision, does it actually rescue the task, just stop the robot, step in when it wasn't needed, or just get lucky?
