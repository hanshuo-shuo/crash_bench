# Red enclosure calibration stopped at the witness gate, 2026-10-03

The user authorized a bounded enclosure pair with the unchanged Reference,
followed by nominal pi0.5 only after a safe open witness and a closed certificate.
The final allowed fixture correction still touched a wall. No additional geometry
search, new policy run or training is queued. See experiments/red_enclosure/README.md.

Final scientific source: `7d30812884f8382db71c6af8d418a2262701419b`.
Quest root: `/projects/p33100/siosio/crashbench_safelibero/red_enclosure/20261003T020342Z_7d30812884f8`.
Slurm **8356882** is FAILED/1:0, 78 seconds, because the explicit positive-witness
gate was not met. The simulation outputs and audits are complete and preserved.

- One underlying exposed layout, two final paired contract states; no holdout.
- Open Reference: 85 legal actions, no native task success; palm contacted the
  opposite `red_y_low` wall. Evidence label **unknown**, not infeasible.
- Sealed Reference: 70 legal actions, no task success; finger contacted the lid.
  All six RE-1 certificate assumptions passed. Conditional **infeasible** label
  comes from the checked enclosure invariant, not this failed rollout.
- Shared initial simulator/controller state passed exact hashes after restoring
  the same canonical snapshot. Pre-restore milk-x delta 16.1658 micrometres is
  retained. Model XML differs only by the lid; native task and frozen goal match.
- The five-action open prefix replay passed exact state hashes. Independent
  persisted-record audit passed 4,000 integrations / 8,166 samples across these
  three executions. The two full reference attempts have forbidden red contact;
  none has robot/target contact with the original protected wine bottle.
- **No pi0.5 server/inference, no AEGIS, no training and zero paid/API calls.**
  The GPU allocation supplied rendering and simulation only. The required
  feasible/infeasible pair was not completed.

Earlier roots remain immutable: job 8356324 / `4959b50` failed a geometry-name
selector before actions (39 seconds); job 8356568 / `4f00d37` observed open palm
contact at action 74 with the original 22 x 22 cm opening, and then stopped on
the strict initial-state pairing gate (66 seconds). The final 22 x 24 cm opening
moved the shared front wall 2 cm; no controller or roof height changed. All three
one-A100 allocations total 183 seconds, with 12.2 allocated CPU-minutes. There
were five completed traces including one short replay, 304 post-settling legal
actions total; these are not five independent layouts.

Raw local evidence: `results/red_enclosure/final_20261003/`. It includes real
native agentview PNGs, complete initial/final state, actions, contacts, synchronized
object/site poses, hashes and Slurm logs. The old 17-execution headless pilot has
no delivered scene images; its PDF QA PNGs are not experimental frames.

RE-1 requires native instantaneous target-root In plus a common frozen-world
goal AABB, and irreversible contact safety plus a geometry-derived adjacent-state
point crossing guard. This is a conditional digital-safety certificate, not a
contact-only continuous-physics proof or full swept-volume robot checker. It does
not establish release/stable placement or model infeasibility recognition.

Suggested next discussion, not executed: derive the full hand/arm collision-volume
envelope along an existing successful Reference trajectory before choosing an
opening, or choose a fixture compatible with that unchanged trajectory. The current
failure identifies inadequate fixture clearance for the Reference; it provides no
pi0.5 capability result and no proof that the open instance is impossible. Broader
research novelty and held-out generalization remain unestablished.
