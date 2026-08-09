# CrashBench Motivation

> **Historical framing.** This early benchmark motivation is retained for
> provenance. See [CURRENT.md](../CURRENT.md) for the scoped causal-diagnosis
> framing used now.

## From Failure Modes to a Benchmark

Our initial failure-mode analysis suggests that the apparent safety capability of vision-language models in control settings can be misleading. In early experiments, the model often appeared to choose safe actions, but the prompt contained explicit physical information such as clearance, safety labels, velocity, progress-to-goal, or other hand-engineered abstractions. Once these quantities were exposed, the task became much closer to selecting from a pre-labeled safety structure than performing genuine visual physical reasoning.

This creates an important ambiguity. When a VLM selects an action that avoids a wall, a hazard, or a bad terminal state, it is unclear whether the model actually inferred the future physical consequence from visual trajectory information, or whether it simply used safety-relevant quantities that had already been computed and inserted into the prompt. In other words, high success under richly annotated prompts does not necessarily imply that the model understands physical risk.

Ablations further support this concern. When the model is given only predicted trajectory shapes, it often fails to choose safe or useful actions. Its behavior improves only when additional task-relevant physical quantities are added, such as predicted velocity or progress toward the goal. This suggests that the model does not reliably extract safety-critical physical state from visual input alone. Instead, it depends heavily on explicit abstractions supplied by the system designer.

This is especially problematic for vision-language-action models. VLAs are typically trained on successful demonstrations, where the robot remains inside the feasible region of the task. These demonstrations rarely contain the states that occur immediately before failure: the gripper drifting into a wall, the object sliding toward the table edge, the peg beginning to bind, the arm entering a bad contact configuration, or the grasp becoming unstable. As a result, the policy may learn how to complete nominal tasks, but it may never learn what to do when execution leaves the demonstration manifold.

Existing VLA benchmarks mainly measure nominal task completion. They evaluate whether a model can solve a manipulation task from a reasonable initial state under mostly successful execution. This misses a failure mode that practitioners frequently observe: the model may perform well until a small error accumulates, after which it drifts into an unsafe or unrecoverable state. The important question is therefore not only whether a VLA can complete a task when everything goes well, but whether it can recognize and recover when a crash is imminent.

CrashBench is motivated by this gap.

## Core Hypothesis

VLAs do not currently have a reliable policy for pre-crash states, because those states are underrepresented or absent in successful demonstration data. When placed near failure, a model may fail for three different reasons:

1. **Perception failure**: the model does not recognize the wall, object edge, unstable grasp, contact risk, or constraint violation.
2. **Prediction failure**: the model sees the relevant scene elements but does not correctly anticipate the physical consequence of its next actions.
3. **Policy failure**: the model sees the danger and may even be able to describe it, but still chooses an unsafe action.

CrashBench is designed to separate these cases. Instead of evaluating only task completion from nominal starts, it drops a VLA into carefully constructed pre-crash states and asks whether the model can recover. The benchmark therefore measures a capability that is central to real deployment but largely invisible in standard evaluation: crash recovery.

## Why Pre-Crash States Matter

A pre-crash state is not necessarily an impossible state. In many cases, a safe recovery still exists: the robot can stop, back away, release, regrasp, move around an obstacle, reduce force, or return to a known-good configuration. These states are precisely where a robust policy should intervene. If a model crashes from a recoverable pre-crash state, the failure is not merely that the task was difficult; it indicates that the policy lacks the ability to reason about imminent physical failure and choose a corrective action.

This makes pre-crash states a useful diagnostic lens. They are close enough to the nominal task distribution to be realistic, but far enough from successful demonstrations to expose whether the model has learned recovery behavior. They also allow us to test different time horizons. At `T-1`, the model must react immediately. At `T-5`, it must make a short-horizon correction. At `T-20`, it must anticipate that continuing the current behavior will eventually lead to failure. If models fail similarly across all horizons, then they are not using the extra time to anticipate danger.

## What CrashBench Measures

CrashBench evaluates VLAs in a small, hand-curated set of recoverable pre-crash scenarios. Each scenario starts from a simulator state where a crash will occur if the current behavior continues, but where at least one verified recovery trajectory exists. This ensures that failures are meaningful: the model was not placed in an impossible state; it failed to find a safe recovery.

The benchmark focuses on four outcomes:

* whether the model crashes;
* whether it recovers and completes the original task;
* whether it safely aborts without crashing;
* how severe the impact is when a crash occurs.

This makes CrashBench more than a task-completion benchmark. It is a safety and recovery benchmark. It asks whether a VLA can remain useful after execution has already started to go wrong.

## Relation to Our Prior Findings

Our failure-mode analysis showed that VLM-based control improves when explicit physical information is inserted into the prompt. However, this also reveals a weakness: if a model needs clearance, velocity, progress, stability, or safety labels to be provided manually, then it is not acting as a reliable low-level physical policy. The system designer is doing much of the physical reasoning for the model.

CrashBench turns this observation into an evaluation target. Instead of asking whether richer prompts can make the model look safe, it asks whether the model can handle states where safety actually matters. If a VLA can recover only when given explicit safety annotations, then the benchmark will expose that dependence. If a VLA can identify danger but still fails to act safely, the benchmark will reveal a policy gap. If a model cannot even predict that a crash is coming, the benchmark will reveal an anticipation gap.

This also connects naturally to methods that route behavior through known-good states. If successful demonstrations define a feasible region, then pre-crash states are states near or outside the boundary of that region. A good recovery strategy should move the robot back toward a known-good state before continuing the task. CrashBench provides exactly the scenarios needed to test whether such recovery-oriented planning works.
