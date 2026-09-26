# pi0.5 and full AEGIS reproduction

## Fixed scientific target

Upstream THU-RCSCT/vlsa-aegis commit 2457feed5968ae803926e178c8ce8243b9ecdcf9;
pi05_libero checkpoint; full six-axis CBF/QP correction plus unchanged gripper.
Keep official BDDL/init states, observation preprocessing, action scale, 20 settling
steps, 5-action chunks, seed 7 and horizons 300/550. Keep obstacle L1 displacement
>0.001 as the benchmark collision proxy. Collisions do not end the episode.
Report CAR, TSR, ETS and safe success; infrastructure exceptions are failed jobs,
not scientific noncompletion labels.

First smoke is Spatial/I/task0/episode0. It is exposed engineering data. No full
1600-episode sweep is launched automatically before the nominal and AEGIS smoke
are checked. No learned selector or new controller is part of this reproduction.

## Quest execution

Code: $HOME/crash_bench. Assets and immutable run roots:
/projects/p33100/siosio/crashbench_safelibero/. GPU jobs use p33100/gengpu; image
conversion and metadata checks use p33100/short. Networking/downloading happens
on the login node; compilation, rendering and inference happen in Slurm.

Existing OpenPI dependency environment is reused read-only for the model server,
with source imports explicitly pointing to the pinned upstream openpi fork.
Record import locations and package versions; this is not a claim that the old
pi0 checkpoint reproduces pi0.5. pi05_libero is downloaded separately.

Simulation uses a digest-pinned NVIDIA CUDA/OpenGL Ubuntu 20.04 container,
a separate Python 3.8.20 runtime, official torch 1.11/CUDA 11.3 wheels and
Open3D 0.19 (Quest host glibc 2.28 cannot use the latter's manylinux_2_31 wheel).
OCI layers and Python distributions are downloaded and hash-verified before
compute-side construction. GPU compilation uses the container's CUDA toolchain,
never a host CUDA module. The old PyTorch 1.11 image was rejected during preparation
because its base is Ubuntu 18.04. Dependency changes needed for portability are recorded.

## GLM via OpenRouter

The user supplies an OpenRouter key when needed. Use z-ai/glm-4.5v, the original
prompt, temperature=0.1, top_p=0.1 and reasoning enabled. Pin the z-ai provider and
disable fallbacks so a provider change is not silent. Endpoint/provider routing
is an explicit deviation from the original Zhipu SDK; identical model naming
alone does not prove identical server weights or outputs.

The compute node exports the exact PNG input and request metadata. An authorized
networked process sends only this simulator image and instruction to OpenRouter.
Cache the complete response with model/provider, request hash and timestamp.
Re-evaluation refuses cache misses rather than inventing an obstacle or replacing
the VLM. No API call is made without a configured key. Keys never enter manifests.

## Minimal adapters

The pinned upstream file remains unmodified. A checked in-memory adapter disables
only obstacle perception and the CBF path for the nominal arm. Both arms retain
identical environment initialization, preprocessing, horizons and scoring.
The AEGIS adapter replaces only the network perception call with its exact-input
cache and makes swallowed infrastructure exceptions fail loudly. It records every
completed episode in JSON. These changes are auditable against pinned source hashes.

Before later statewise benefit work, continuation capture must include simulator,
controller, client action queue, server JAX RNG and AEGIS internal state. No claim
of exact-state branching is made by these reproduction scripts.

## Acceptance

1. Verify upstream revision, all 32 init files, checkpoint/norm stats/tokenizer,
   GroundingDINO weights/BERT assets and isolated LIBERO_CONFIG_PATH.
2. Construct environment offline; verify imports and GPU/EGL rendering.
3. Run nominal smoke and cache the initial GLM input. Configure API only then.
4. Fill the matching perception response cache and run full AEGIS smoke.
5. Review videos, failures, action scaling, metric records and resource use before
   preparing a fixed baseline evaluation matrix. Preserve all failed attempts.
