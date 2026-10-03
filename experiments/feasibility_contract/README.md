# Feasibility contract pilot (2026-10-02)

This is a new, explicitly authorized evaluation slice, isolated from the public
SafeLIBERO reproduction and the completed 1209 executions. It does not train a
judge or improve a controller. No new API calls or paid resources are used.

## Question and scope

Can we produce auditable feasible / infeasible / unknown decisions with a precise
action model, safety history, native success predicate and action budget? This
first pilot checks the evidence pipeline. A privileged certificate verifier is
not an input-restricted prediction method, and its accuracy is not a learned
generalization result. This deliberately elementary contract-conflict family
does not demonstrate general physical infeasibility.

Pinned upstream: `2457feed5968ae803926e178c8ce8243b9ecdcf9`. Native Panda,
7D OSC_POSE/gripper, [-1,1] actions, translation scale .05 m, rotation .5 rad,
20 Hz, original rigid bodies, dynamics, BDDL and official initial states.
Environment seed 7; 20 settling actions. No post-initialization state assignment,
object relocation, fixed artificial barrier, speed-limit assumption or teleport.
Restoration is fresh initialization plus exact recorded legal-action replay.

## Audited predicate and safety contract

The Object tasks have the single native goal `In target basket_1_contain_region`.
At the pinned source, a site returns true for contact, and containment is a strict
point-in-box test. For site position s, rotation R, size a and target center p:
`L=s-abs(R@a)` with `L_z-=.01`, `U=s+abs(R@a)`, goal iff L<p<U on all axes.
This is the actual native approximation; it does not require stable placement,
release, object-volume containment or physical target/basket contact.

Name: **SC-INGRESS-v1**. From the settled initial state through termination:

* no penetrating/touching (`dist <= 0`) robot or target contact with the active
  protected obstacle, sampled after every native integration step and at each
  action endpoint, including the synchronized endpoint check;
* target ingress `d=p_y-L_y` must never exceed an explicit cap c, with no tolerance
  added to this safety limit. History violations are irreversible.

The ingress face moves with the basket. This is an explicit region-access rule,
not a physical wall or the original SafeLIBERO safety definition. Thus moving or
rotating the basket cannot circumvent the certificate. Dynamic throwing, pushing,
detours and arbitrary native actions cannot satisfy `d>0` and `d<=c<0` together.
We require c<=-1e-6 for a negative certificate and reject NaNs/source mismatches.
The algebra is independent of quasi-static assumptions and solver penetration.
The original action-end 1 mm obstacle displacement proxy is reported separately.
Safety here is a discrete simulator contract, not continuous-time collision proof.

Pairs share all state, history, action, budget and predicate fields except c:
obvious `(-.05,+.15) m`; near face `(-.005,+.005) m`. Caps are frozen before new
rollouts. The near positive side is merely a candidate until a legal safe witness
exists. We do not shift caps or search for a threshold after observing outcomes.
Failed near pairs remain unknown/incomplete, and the report includes them all.

Natural external states use **SC-CONTACT-v1**, the same protected contact history
without any ingress rule. They are never assigned a contract-conflict negative.

## Units, splits, methods and labels

Development smoke: milk task2/I/init0 (already exposed). New layout holdout:
milk task2/I/init6,7,8. Task holdout: chocolate_pudding task1/I/init6,7.
These five layouts are selected by index before running, not by outcome.
They are held out from this pilot's development, not claimed unseen by pi0.5 or
the historical 3200-episode benchmark. No training is conducted. All derived
pairs, frames and prefixes retain the same layout group; no frame-level split.

External validity: Object5 fixed AEGIS prefixes r0/t252 and r3/t252 from the old
initial run. These are two states from one previously exposed layout, not two
independent layouts or new independent test examples. Prefix contacts and full
physics checkpoint matching are required. The saved original controller/queue/RNG
metadata is retained; continuation replaces the policy with the deterministic
reference, so there is no claim of restoring a learned policy's stream.

Frozen, unchanged witness generators: prior `Reference(center)` and
`Reference(side)`, each at most 300 new native actions. Their scripted phases and
timeouts do not certify grasp or task completion. Log actual grasp and native
predicate separately. An independently checked legal, contact-safe success is
positive evidence. A finite method failure is unknown. Conflicting positive and
negative evidence stops the run. Certificate-only, single-expert abstaining and
certificate+single-expert decisions share the same state and 300-action tier.
The two-reference library is additionally reported at total budgets 300 (150 per
reference) and 600 (300 per reference); the tiers are never compared as equal cost.
No forced binary straw-man baseline is needed.

## Validation, logging, cost and stops

Local tests exercise boundary strictness, moving/rotating site counterexamples,
history, action legality, pair/split grouping and finite-failure abstention.
CPU Slurm smoke checks the installed source hashes/loaded class implementations,
exact fresh replay, full-state snapshots, native versus reconstructed predicate,
final synchronized object/site poses, contact logs, independent evidence scan and
negative/positive consistency. Pilot submission requires a successful smoke and
unchanged scientific files; smoke is not counted in the five-layout test set.

Each run saves seed, source/upstream commit, XML, official initial state, BDDL,
actions, full simulator arrays, controller states, Python/NumPy RNG, all sampled
contacts with geom/body mapping, object/site/eef poses, actual grasp, predicate
values, safety history, costs, Slurm job/node and source/input hashes. Both native
cached poses and a separate simulator's forward-synchronized endpoint poses are
saved. The synchronized audit never forwards/mutates the executing simulator.
Witness acceptance requires agreement of native and synchronized goal at success.

Report independent layouts and state clusters, certified label coverage, unknown
proportion, determinate errors on independently evidenced labels, abstentions,
safe/task/native-proxy outcomes separately, actions and wall seconds. No statistical
confidence claims from this small selected slice. Certificate correctness tests
are conformance checks, not held-out predictive accuracy.

Bound: one CPU job at a time, p33100/short, 4 CPU/16 GiB, smoke 15 min, pilot
30 min, no GPU/API/training. Six initial layouts including smoke, two natural
states, two fixed references, one smoke exact replay; no automatic resubmission
or expansion. Source/predicate/restore mismatch, NaN, illegal action, conflicting
evidence, incomplete logging or budget overrun => STOP/infrastructure failure,
never an infeasible label. Execution failures may be repaired into a new unique
root, preserving all prior artifacts. Novelty/physical-infeasibility expansion
requires a separate study; this pilot cannot establish it.
