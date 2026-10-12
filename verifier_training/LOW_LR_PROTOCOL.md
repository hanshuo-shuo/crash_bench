# One fixed lower-learning-rate rank64 SFT comparison

The user authorized a single lower-LR comparison on Quest, fixing rank64 and all
other settings, with balanced8 overfit before a conditional120 fit, and no search.
Predeclared choice: reduce each stage's original LR by four. Smoke2e-4 ->5e-5;
full1e-4 ->2.5e-5. This is one prescribed recipe, not a tested-best selection.

Rank64/alpha128, all252 language projection modules, fresh base/adapter, model
revision, binary0/1 targets, normalized .5/.5 component losses, two RGB inputs,
frozen vision/projector, seed713 and original stage-specific settings stay fixed.
Smoke: same4+4 examples, dropout0, batch4, cap240, generate every20, strict8/8 full
outputs and75% loss reductions. Full: fresh initialization, dropout.05, batch4,
ten epochs/300 updates, same full-data order, no extension or early stopping.
Neither stage resumes the earlier failed checkpoint or the new smoke adapter.

CPU preflight runs the tests,240 paired encoding checks and exact configuration
comparison. Only learning_rate may differ from the frozen rank64 reference.
If smoke fails, no full job is submitted and a CPU smoke report closes the run.
If it passes, submit one full job, then read-only gold/generated-rationale prefix
scoring and a CPU report. Maintain separate evidence-grounded rationale grading,
class-specific decision-token/EOS losses, full-output accuracy and zero-command
prediction transitions. All stages use unique output roots and frozen published
source under p33100; GPUgengpu/A100, CPUshort, offline staged assets, no paid API.

The smoke directly compares learning rate at fixed rank/initialization procedure.
Record equality or differences in initial full-eight losses and the first real
gradient probe as an additional control. Report all checkpoints, including any
instability or repeated text. A failed240-step gate is a failed bounded recipe,
not proof that longer optimization could never succeed.

There is no previous high-LR rank64 full120 fit because its smoke failed. If the
new full fit runs, comparison to rank16 changes rank and LR jointly; it cannot
isolate the learning-rate effect. Do not launch an otherwise unauthorized high-LR
full run to fill this cell. All120 labels have one original source scene; no
generalization, scientific-model selection, or new simulator collection follows.
