# Non-glass unstable-placement v1 preflight audit

**Audit date:** 2026-08-30
**Scientific status:** `INVALID_PREFLIGHT_ZERO_OPTION_OUTCOMES`
**Slurm job:** `5165648` (`p33100`, `gengpu`, `COMPLETED`, `00:07:04`, exit `0:0`)
**Execution commit:** `53f62a64ae5b184b70676491e4d5353aa0d072ea`
**Protocol fingerprint:** `c0bb9015a0a3dfb4aa9d70672e2c7a046c56f688842d880e46b8d46a68bc9413`
**Result root:** `results/iclr27/non_glass_option_ambiguity_screen_c0bb9015a0a3_20260829T170210Z/`

This is an additive audit of the first unstable-final-placement implementation.
It does not alter the immutable result root or reinterpret a branch outcome.
There were no option branch outcomes to interpret.

## 1. What the job did and did not test

The Slurm job and software environment completed normally. The runner evaluated
the complete eight-entry nominal source-candidate list, then stopped in
freeze-only source/geometry preflight. It did not open the physical option grid.

Machine accounting:

| Quantity | Value |
|---|---:|
| Predeclared source candidates attempted | 8 |
| Sources marked selected before location resolution | 2 |
| Resolved physical blocks | 0 |
| Attempted linked option decisions | 0 |
| Restoration audits | 0 |
| Ordinary option-outcome rows | 0 |

Therefore v1 provides no evidence about:

- whether the unstable patch makes Base catastrophize;
- whether `stable_offset_place` completes the task;
- whether `safe_setdown` avoids object loss;
- exact-state restoration under the injected patch;
- strict option support;
- same-risk action ambiguity;
- the viability of the `unstable_final_placement_v1` family.

The recorded `SCREEN_BLOCKED_ENVIRONMENT` label is a pipeline terminal label.
It must not be cited as a scientific screen failure or as evidence against the
paper's diagnostic idea.

## 2. Candidate accounting

The authoritative `source_registry.json` contains eight attempted candidates,
not two. The two values reported as selected are default states 6 and 7.

| State | Nominal task success | Release anchor | v1 disposition |
|---:|---|---|---|
| 0 | not recorded after exception | not recorded | geometry derivation exception |
| 1 | yes | no | held-to-release detector did not fire |
| 2 | yes | T-20/T-10/T-5 available | false `task_geometry_drift` rejection |
| 3 | yes | T-20/T-10/T-5 available | false `task_geometry_drift` rejection |
| 4 | yes | T-20/T-10/T-5 available | false `task_geometry_drift` rejection |
| 5 | yes | T-20/T-10/T-5 available | false `task_geometry_drift` rejection |
| 6 | yes | T-20/T-10/T-5 available | selected, then location resolution failed |
| 7 | yes | T-20/T-10/T-5 available | selected, then location resolution failed |

All eight candidate source hashes and seeds remain registered as exposed screen
sources and excluded from a later claim-bearing benchmark.

## 3. Root causes

### 3.1 Whole-object world AABB was used as a bottom support footprint

The runner unions every bowl geom in world coordinates and sets
`bowl_footprint_radius_m` to half the smaller whole-body AABB extent. The live
value was approximately `0.07844 m`; the plate world-AABB radius was
approximately `0.08181 m`. The support margin was approximately `0.00784 m`, so

```text
plate radius - bowl radius - margin
  = 0.08181 - 0.07844 - 0.00784
  < 0
```

The resulting "strict support region is empty" exception is a definition error.
The whole bowl rim is not its bottom contact footprint. The nominal LIBERO task
successes themselves show that the object has a valid support configuration on
the plate.

### 3.2 Pose-dependent world AABBs were treated as invariant asset identity

Across default source poses, the measured bowl and plate radii varied by roughly
`0.27--1.88 mm`. The implementation compared the complete world-geometry
signature with an absolute tolerance of `0.2 mm`. A changed yaw or initial pose
changes a world AABB without changing the model asset, task, or support
mechanics. States 2--5 were consequently rejected as `task_geometry_drift`
despite nominal stable task success and valid release anchors.

### 3.3 A failed first derivation poisoned later candidates

For state 0, the runner assigned `raw_geometry` before calling
`derive_geometry`. Derivation raised, the candidate-level exception handler
caught the error, but `raw_geometry` stayed non-null while resolved `geometry`
stayed null. Later candidates therefore skipped derivation and were compared to
the failed first candidate's pose-dependent signature.

### 3.4 Selected sources could not resolve physical locations

At the end of the candidate loop, unresolved geometry was replaced with
`{"parameters": [], "blocked": true}`. Location resolution for states 6 and 7
then raised `KeyError: 'plate_support_radius_m'`. No source/parameter physical
block was frozen.

### 3.5 Final accounting hid the full failure chain

The analyzer wrote `attempted_source_count = 2` using the selected-source set,
although the registry contains eight source attempts. The headline blocker
reported only the four-source selection shortfall and did not expose the later
location-resolution errors.

## 4. Scientific interpretation

The v1 result establishes only that this implementation could not freeze a
valid geometry/source grid. It does not establish that:

- the protocol is too difficult for the simulator;
- unstable final placement cannot generate a catastrophe;
- offset placement and safe setdown lack distinct utility;
- non-glass option ambiguity is absent;
- `Risk Is Not Regret` is false or unpublishable.

The correct paper-facing treatment is to exclude v1 from scientific outcome
counts and mention it only as an engineering preflight if needed for
reproducibility.

## 5. Corrective contract if v2 is explicitly authorized

Any v2 must be additive and prospective:

1. preserve v1 and its result root;
2. create a new config, commit, fingerprint, result root, and audit;
3. compute invariant model-local asset geometry;
4. identify the bowl bottom contact footprint/support polygon separately from
   the whole-body render/collision envelope;
5. derive source-specific poses only after invariant dimensions pass;
6. commit reference geometry transactionally after successful derivation;
7. prove four selected sources, 12 physical source/parameter blocks, and 108
   linked decisions before opening an intervention;
8. separate attempted, nominal-valid, selected, location-valid, restoration-
   valid, and outcome-eligible source counts;
9. use the revised feasibility gate in
   [`PUBLICATION_FIRST_RESOLUTION.md`](PUBLICATION_FIRST_RESOLUTION.md).

This corrective contract is optional. It is not part of the active
publication-first Definition of Done and does not authorize a rerun by itself.
