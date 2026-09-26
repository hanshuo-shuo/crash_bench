# CrashBench → SafeLIBERO reproduction

We retain [eight findings from CrashBench](retained/README.md) and now reproduce
nominal **pi0.5-LIBERO** and **full AEGIS** on the fixed public SafeLIBERO benchmark.
The first deliverable is a verified baseline, followed by repeat-aware failure
diagnosis. Intervention benefit is a quantity to measure, not a method claim.

- [Current state](docs/CURRENT.md)
- [Reproduction protocol and deviations](docs/REPRODUCTION.md)
- [Quest commands](setup/README.md)
- [Historical archive](https://github.com/hanshuo-shuo/crash_bench/tree/codex/archive-crashbench-20260926)
- [Upstream](https://github.com/THU-RCSCT/vlsa-aegis/tree/2457feed5968ae803926e178c8ce8243b9ecdcf9)

The active tree contains only selected historical evidence and reproduction tooling.
Historical result bytes are unchanged; model caches and other Quest projects are
outside this cleanup. API credentials must never enter Git or logs.
