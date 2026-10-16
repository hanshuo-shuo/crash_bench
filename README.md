# CrashBench → SafeLIBERO feasibility verifier

We retain [eight findings from CrashBench](retained/README.md), reproduced nominal
**pi0.5-LIBERO** and **full AEGIS** on the fixed public SafeLIBERO benchmark, and now
train a generative feasibility verifier: a VLM that judges, before execution, whether
a task admits a constraint-satisfying solution.

- [Current state](docs/CURRENT.md)
- [Generative verifier plan](docs/generative_verifier/PLAN.md)
- [Verifier dataset (FD1000 / V4)](verifier_dataset/CURRENT.md) and [training code](verifier_training/README.md)
- [Reproduction protocol and deviations](docs/REPRODUCTION.md)
- [Quest commands](setup/README.md)
- [Historical archive](https://github.com/hanshuo-shuo/crash_bench/tree/codex/archive-crashbench-20260926)
- [Upstream](https://github.com/THU-RCSCT/vlsa-aegis/tree/2457feed5968ae803926e178c8ce8243b9ecdcf9)

Historical result bytes are unchanged; model caches and other Quest projects are
outside every cleanup ([2026-10-16 record](docs/CLEANUP_20261016.md)). API
credentials must never enter Git or logs.
