# Fixed higher-rank SFT engineering comparison

Authorized by the user on 2026-10-11: continue SFT and compare higher-rank LoRA.
This runs only the generative B implementation. The comparator is the corrected
binary rank16 model, not the earlier word-label model with asymmetric decoding.

The only training configuration changes are rank16 -> rank64 and alpha32 ->
alpha128, preserving PEFT's ordinary alpha/r scaling of 2. Target coverage stays
at all252 language attention/MLP projection modules. Trainable parameter count
increases from29,933,568 to119,734,272. Vision/projector, backbone, input images,
task/constraints, output format, loss, optimizer, seed, data order and decoding
remain fixed. Full fit uses fresh base/adapter, LR1e-4, dropout.05, effective
batch4, ten epochs/300 updates. It never continues the rank16 or smoke checkpoint.

The same eight balanced smoke examples use their original LR2e-4/dropout0,
240-update cap, generation every20 steps and strict8/8 exact-output gate plus
75% loss reductions. CPU tests and all120 encoding comparisons must pass first;
full training is submitted only after this rank64 smoke passes. No parameter
search, automatic extension or second full fit is part of this protocol.

Use `launch_rank_comparison create`, followed by explicitly gated `smoke`,
`full`, `conditioning`, and `report` commands through quest_sync.sh on published
source. One A100 at a time, p33100/gengpu; CPU p33100/short. Slurm time limits
allow30min smoke and75min full for added rank64 overhead, without changing update
budgets. All outputs use a new root; baseline files are hash-checked and read-only.

Record component/decision/format/EOS losses, free-generated class accuracy,
grounded visibility statements, exact full outputs, all120 gold/own-prefix
probabilities, zero-command transitions and the hidden-milk stratum. Produce
overlaid rank16/rank64 curves and paired per-case output changes. Correctness of
the rationale is visibility/caveat correctness, not a physical feasibility proof.

This is a single-seed fixed-recipe engineering comparison. Larger adapter shapes
change initialization and RNG consumption even with the same seed. Keeping
alpha/r fixed does not equalize effective optimization dynamics. A gain would
support this higher-rank recipe, not isolate pure representational capacity or
establish generalization. All120 descendants still share one source scene;
missing physical evidence and absent matched constraint changes remain data
limitations regardless of fit quality. No new simulator data or scientific
pipeline changes are included.
