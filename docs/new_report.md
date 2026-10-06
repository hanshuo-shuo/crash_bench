### Prev's results: 

At first, I wanted to show that the robot knows when it is about to crash. In my wall experiments, I found that OpenVLA's internal features can predict a crash several steps before it happens. However, even if the model seems to know the danger, it does not stop.

I then added a simple safety intervention to stop the robot. This prevented crashes, but the robot also stopped completing the task. I also tried removing the "dangerous" direction from the internal features, but this did not solve the problem.



**Direction 1: Task solvability and state recoverability**

I need to tell apart three situations: (a) the task was infeasible from the start, (b) the task is still recoverable but the current policy does not know how to recover, and (c) the robot gradually loses its chance to recover during execution. First, I will build independent evidence of feasibility under clearly defined robot capabilities, safety requirements, and time budgets. Then I will study whether recoverability can be predicted from runtime observations. A reliable state classification can also help with building benchmarks. However, I will not treat the failure of a current recovery method as proof that the task is unsolvable.

**Direction 2: Intervention methods on a fixed framework**

I will stop expanding the tasks, designing a recovery controller, and changing the evaluation all at the same time. First, I will reproduce existing benchmarks and public baselines, and find specific, repeatable failure modes. Only after I confirm that the benefit of intervention differs in a predictable way across states, I will develop one module that targets either the timing of intervention or the choice of recovery action. I will compare it with strong simple rules. Intervention benefit is the quantity to measure or optimize. It is not the method itself.

### Moving to a public benchmark made the question stricter：

I then moved to SafeLIBERO, which keeps the original tasks and action space, and reproduced both nominal π0.5 and the full AEGIS (their safety policy) pipeline.
I evaluated each method on 1,600 episodes. These experiments were completed on September 28 and are separate from the 600 diagnostic reruns discussed later.


| **Full Evaluation — 1,600 Episodes per Method** | **Nominal π0.5** | **Full AEGIS** |
|---|---:|---:|
| **Collision Avoidance Rate (CAR)** | 255 / 1600 (**15.94%**) | 1045 / 1600 (**65.31%**) |
| **Task Success Rate (TSR)** | 932 / 1600 (**58.25%**) | 1088 / 1600 (**68.00%**) |
| **Safe Success** *(task completed without collision)* | 238 / 1600 (**14.88%**) | 783 / 1600 (**48.94%**) |



> **AEGIS improves safety and task success, but it still does not tell me which failure states are recoverable.**

I want to know what  happens after AEGIS prevents a crash?

- Spatial I / task-index 1：black bowl on the ramekin → plate
- Object I / task-index 2：milk → basket
- Object II / task-index 1：chocolate pudding → basket

I compared nominal π0.5 and AEGIS from the same 60 initial states. These tasks are in the appendix, where the author mentioned the limitation of this method.
Each method was run 5 times per state.
A state was considered stable if at least 4 out of 5 runs had the same outcome.
Among the 44 stable paired states, I compared how the outcome changed after turning on AEGIS.

In this setting, AEGIS behaves more like a brake than a steering wheel. Similar to what I found earlier.

| Same initial state | # states |
|---|---:|
| Crash → Crash | **16** |
| Crash → Safe but incomplete | **12** |
| Crash → Safe success | **2** |
| Safe success → Safe success | **14** |

Among 30 stable states that crashed under the nominal policy, AEGIS turned 12 into safe-but-incomplete outcomes, but only 2 into safe successes.


### Perception explains some failures

wrong box → polluted geometry example:

<img width="2088" height="712" alt="image" src="https://github.com/user-attachments/assets/50a2c8e0-8419-485f-8906-8d34220df21d" />

- Object II / task-index 1：chocolate pudding → basket


| VLM obstacle selection | Runs | Crashes |
|---|---:|---:|
| **Correct** | 43 | **0** |
| **Mismatch** | 57 | **28** |

But perception is not the whole story.

In Object I, 57 safe-but-incomplete runs still had the correct obstacle identity and localization.

Even if the robot understands the obstacle correctly, is there actually a safe way to finish the task?


Of Course, A failed recovery attempt does not mean the state is infeasible. The next problem was how to define recoverability.

### strong reference controller

A natural idea is to use a strong reference controller: if it can recover, the state is feasible; if it fails, maybe the state is infeasible. 

However, building a strong reference controller is not that easy. This left for next week.



The second plan is inspired by my previous experiments this summer; it is very easy to build a privileged feasibility oracle + construct certified infeasible cases.

Feasible：Codex hard-code controller、motion planner, paper's methods


| Label | Evidence |
|---|---|
| **Feasible** | A valid safe-success trajectory exists |
| **Conditionally infeasible** | An explicit geometric/task certificate rules out all admissible solutions under stated assumptions |
| **Unknown** | Neither has been established |

<img width="1840" height="466" alt="image" src="https://github.com/user-attachments/assets/f885421b-a6d4-4f8a-ad0a-9115bc811cd2" />


- Next, I wanted to test whether the VLA representation actually encodes this notion of feasibility.

For positive examples, I required a safe-success witness: an actual trajectory that completes the task without violating the safety constraint.
For negative examples, I used explicit geometric certificates that rule out any admissible solution under the stated task and safety assumptions.

| **Readout / Baseline** | **Role** | **Held-out AUROC ↑** | **Balanced Acc. ↑** |
|---|---|---:|---:|
| **Pre-selected VLA probe (Layer-2 MLP)** | **Main test** | **0.500** | **0.500** |
| Final VLA feature | Diagnostic | 0.708 | 0.646 |
| VLA visual tower | Diagnostic | 0.802 | 0.500 |
| DINOv2 visual baseline | Visual baseline | 0.833 | 0.708 |
| RGB + red-cue baseline | Shortcut baseline | **0.979** | 0.625 |
| Privileged geometry | Oracle baseline | **1.000** | **1.000** |

RGB + red-cue baseline： This is a shortcut baseline. It does not use VLA features at all. It only uses simple statistics from the raw RGB images and red-obstacle cues. Simple visual cues almost perfectly rank the current feasible and infeasible examples, but they do not form a well-calibrated feasibility classifier at the fixed threshold.

Next: 

state feasibility? action-conditioned recoverability?

Recoverability is relational. It depends not only on the state, but also on the task, safety constraints, available capabilities, and remaining time. So the VLA has to know more than just 
some probe fitting. Does the model know its own action possibilities well enough to act on that knowledge? I feel like humans have such ability but not vlm/vlas.
