# Limitations and ethics/safety scope

## Limitations

1. **One mechanical case study.** The claim-bearing evidence uses one OpenVLA
   checkpoint, one LIBERO pick-and-place task, one embodiment, and one
   `glass_recovery` design. `glass`, `offpath`, and `noglass` are not three
   independent hazard families.

2. **Exposed development evidence.** The 20 sources retain historical 5/7/8
   split labels for provenance, but all were examined across the frozen audit
   sequence. The paper is descriptive and diagnostic, not confirmatory.

3. **Small effective support.** Risk-benefit disagreement spans 12 sources,
   but the exhaustive tight witness set spans only two, and its exact
   Detour/Retreat witness has one-source support. All 23 strict Retreat states
   and 51/60 strict Detour-or-Retreat states are glass.

4. **Finite engineered options.** Detour has privileged obstacle geometry and
   is not an end-to-end learned recovery. Retreat is usually a safe abort. The
   realized Oracle is only the best of these three options under the frozen
   utility.

5. **Utility omits option cost.** The `+1/0/-1` utility does not charge option
   duration, path length, control effort, latency, or geometry privilege.

6. **Restoration boundary.** Five branch-start identities are hash-verified:
   simulator, controller/dynamics, full numeric continuation, observation, and
   model XML. Global RNG bytes were not separately serialized and hashed per
   decision; fixed seed and deterministic decoding do not replace that missing
   audit.

7. **Artifact availability.** Reviewed aggregate tables and figures are tracked
   and hashed. Raw capture files are locally present and hash-pinned but live in
   a Git-ignored directory. Raw witness pixels were not retained.

8. **No learned-method success.** OutcomeRouter gains only `+0.0178` over the
   risk reference and fails the frozen method gate. Full tiny ADR utility is
   `0.2311`, below the `0.2544` reference. Oracle hybrids are unavailable at
   deployment.

9. **Sequential null.** Fresh direct first crossing chooses Base `24/24`,
   recovers `0/8` glass episodes, and misses `2/2` known opportunities. This
   closes the frozen protocol, not every possible temporal method.

10. **Non-glass v1 is not evidence.** The optional engineering preflight opened
    zero option outcomes and is absent from every scientific denominator.

## Ethics and safety

The claim-bearing study is simulated robotic manipulation and uses no human
subjects or personal data. Its artifacts expose both the opportunity and the
failure of runtime selection, which can be useful for auditing unsafe
assumptions. They must not be presented as a certification, deployment policy,
or guarantee of collision avoidance.

The privileged controller, simulator restoration, and realized Oracle are
evaluation instruments. A real system would require task-specific hazard
analysis, independent hardware fail-safes, uncertainty-aware calibration,
latency and actuation audits, human oversight where appropriate, and validation
on the target robot and environment. Preserving the method and sequential nulls
is itself a safety measure: it prevents a large offline Oracle gap from being
mistaken for operational protection.
