# Group meeting: what the router and options actually do

## One-sentence idea

A risk monitor only predicts whether Base may fail.  Our router predicts what
will happen under each available option, and then chooses whether and how to
intervene.

## What is learned and what is scripted

The router is learned.  The recovery options are not learned.

| Option | Implementation | Goal |
|---|---|---|
| Base | OpenVLA continues producing actions | Finish the original task |
| Detour | Scripted Cartesian state machine with simulator geometry | Go around the glass and finish the task |
| RetreatHold | Scripted move in world `-x` and `+z`, then hold | Abort safely |
| FailSafeHold | Zero Cartesian motion with the gripper open | Stop when directional retreat is unsafe |

The paper claim is therefore:

> We learn to route among structured options.  We do not learn the recovery
> controllers themselves.

## How Detour works

`DetourComplete` directly receives the glass, bowl, and plate geometry.  It is
a hand-written state machine:

```text
move away from the glass
→ lift the end effector
→ move into a side lane
→ move above the bowl
→ descend and close the gripper
→ lift the bowl
→ move above the plate
→ place and release
```

At every step it outputs a seven-dimensional LIBERO action:

```text
[delta x, delta y, delta z,
 delta roll, delta pitch, delta yaw,
 gripper]
```

The simulator's operational-space controller converts this Cartesian command
to robot control.  Detour uses privileged simulator geometry, so it is a
structured oracle controller, not an end-to-end visual recovery policy.

## How Retreat and Hold work

The historical `RetreatHold` sets one target when it is triggered:

```text
target x = current x - 0.14 m
target y = current y
target z = current z + 0.10 m
```

It uses proportional control to reach that target and then naturally holds.
The direction is fixed in world coordinates.  It is not relative to the glass.
In the P1 smoke, this fixed direction caused a collision at T-5.

`FailSafeHold` is the simpler late fail-safe:

```text
[0, 0, 0, 0, 0, 0, gripper open]
```

It does not try to finish the task or move away.  It only stops the current
motion.  P1 keeps historical RetreatHold for comparison but uses FailSafeHold
for timing isolation.

## How the counterfactual data are collected

We restore exactly the same simulator and controller state and run all options:

```text
same state x
├── Base       → success / catastrophe / safe noncompletion
├── Detour     → success / catastrophe / safe noncompletion
└── Retreat    → success / catastrophe / safe noncompletion
```

This gives option-conditioned supervision.  The result is not confounded by
starting the three options from different states.

## How the router is implemented

The input for one decision is:

- a 4096-dimensional OpenVLA hidden state;
- train-only PCA to 16 dimensions;
- 8 robot-state dimensions;
- 7 nominal-action dimensions.

The final feature has 31 dimensions.  Three linear softmax heads predict:

```text
P(success | state, option)
P(catastrophe | state, option)
P(safe noncompletion | state, option)
```

for Base, Detour, and Retreat.

For each option, the router computes

```text
utility = P(success) - lambda * P(catastrophe)
```

It keeps Base unless the best intervention has a calibration-frozen advantage
larger than `delta`.  This is why the method is conservative: a small predicted
benefit is not enough to override Base.

## What the baselines really are

Most baselines do not require a new VLA model.  They are different selection
rules over the same exact-state option outcomes.

| Baseline | Rule |
|---|---|
| Base | Always keep Base |
| Always Detour | Always select the Detour branch |
| Always Retreat | Always select the Retreat branch |
| Risk to Retreat | If predicted Base risk is high, select Retreat |
| Risk to Detour | If predicted Base risk is high, select Detour |
| Two-threshold Risk | Low risk: Base; medium: Detour; high: Retreat |
| Counterfactual Oracle | Use the realized outcomes of all branches and select the best one |
| Geometry Gate | Use the true nominal path and true glass geometry to trigger Detour |

The risk baselines use the same features as the router, but train only one
binary Base-catastrophe predictor.  Geometry Gate is an oracle-style diagnostic
because it uses simulator geometry.  It shows that the old benchmark had an
easy shortcut; it is not the main P1 research question.

## Why P1 changes the benchmark

In the old T-20 benchmark, the router selected Base 10 times, Detour 14 times,
and Retreat zero times.  The result was mostly a Base-versus-Detour gate.

P1 samples the same Base trajectory at T-30, T-20, T-10, and T-5.  The target
classes are:

| State type | Correct option |
|---|---|
| Early and recoverable | Detour |
| Late and only safe stopping remains | FailSafeHold |
| Intervention is unnecessary | Base |
| No option can finish | FailSafeHold |

This benchmark tests option value and intervention timing, rather than only
testing whether the glass is on the path.

## Current evidence and honest status

- The exact-state multi-anchor pipeline runs on Quest.
- Three source-disjoint smoke trajectories produced 36 matched decisions and
  108 option rollouts with raw features.
- One trajectory is a clean library-unsolved example.
- Another trajectory is recoverable at T-20/T-10, but historical directional
  Retreat fails at T-5.
- We have not yet shown a fresh Detour-to-FailSafeHold timing pair.

The next experiment is only one glass-only trajectory with `FailSafeHold`.
We should not expand the cohort before this option contract works.

## A short script for the meeting

> Our main point is that failure risk is not enough to choose an intervention.
> We restore the exact same robot state and run Base, a scripted task-completing
> Detour, and a scripted fail-safe option.  A small router predicts the outcome
> of every option and overrides Base only when the predicted advantage is large
> enough.  The recovery controllers are hand-written; the learned component is
> the option selection.  Our old benchmark was too close to glass detection, so
> P1 now samples T-30 to T-5 on the same trajectory.  This creates states where
> Base, Detour, and safe stopping should be selected for different reasons.

