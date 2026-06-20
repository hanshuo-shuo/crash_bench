# crashbench/ — package layout & usage

The Python package implementing CrashBench (PLAN.md §5). Substrate = LIBERO/robosuite,
reusing OpenVLA's verified obs/action bridge.

## Modules

| file | what |
|---|---|
| `scenario.py` | `Scenario` dataclass (PLAN §2) + `PredicateSpec`; JSON+npy save/load |
| `predicates.py` | crash/success predicates from specs (PLAN §3): `contact_force`, `object_fell`, `grasp_dropped`, `libero_task_success`. Query a `SimView`, never vision |
| `envs/libero_adapter.py` | `LiberoEnv` wraps the exact `run_libero_eval.py` API (reset / set_init_state / step / render); `LiberoSimView` reads mujoco for predicates |
| `policies/openvla_policy.py` | `OpenVLAPolicy` wraps OpenVLA `get_model`/`get_action` (+ gripper normalize/invert). `prompt_prefix` = README §7 prompted-careful baseline |
| `eval.py` | `run_episode` closed loop → `Outcome` ∈ {crash, recovery_success, safe_abort, timeout} + `EpisodeResult` |
| `metrics.py` | the 4 README §5 metrics + by-horizon / by-category breakdowns + headline crash@T-5 |

Entry points: `scripts/run_pilot.py` (the go/no-go number), `scripts/author_scenario.py` (snapshot a pre-crash state).

## Status (2026-06-19)

- ✅ Core logic done & unit-tested (`tests/test_core.py`, no GPU needed): serialization, predicates, metrics.
- ⚠️ **Marked `TODO(verify)`** — needs confirming on a real LIBERO env before crash numbers are trustworthy:
  1. `libero_adapter.py`: mujoco body/geom names for contact force, object z, and the real `_check_grasp` wiring.
  2. `author_scenario.py`: the flattened `set_init_state` vector layout — which indices move the eef / an object.
     The `perturb_*` functions are **stubs that return the state unchanged**; fill them in after inspecting a live env.

So the harness runs end-to-end, but **authoring real pre-crash states is the remaining Phase-1 work**
(PLAN §4 Phase 1 step 1). Verify the bridge first by reproducing the nominal LIBERO success rate through
`LiberoEnv` (should match the 66.7% sanity / full-run number), then perturb states and confirm by re-rendering.

## Quick test (any node, no GPU)

```bash
source activate ~/crash_bench/envs/openvla
pip install -e .            # makes `crashbench` importable
python tests/test_core.py   # -> all core tests passed ✓
```

## Run the pilot (GPU node)

```bash
MUJOCO_GL=egl python scripts/run_pilot.py \
  --scenarios scenarios --checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --unnorm_key libero_spatial --out results/pilot.json
```
