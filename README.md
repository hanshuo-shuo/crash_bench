# CrashBench: Intervention and Recovery Research

CrashBench studies intervention choice and task recovery from exact robot states.
Current work: Refresh evaluation diagnosis closed (2026-09-08).
See the [integrated results chapter](docs/audits/20260908/refresh_closeout/RESULTS_CHAPTER_ZH.md)
and [fixed-option benefit-gate plan](docs/audits/20260908/refresh_closeout/BENEFIT_GATE_PLAN_ZH.md).
The next plan is drafted, not executed; no automatic experiment or architecture search.
The [current state](docs/CURRENT.md) supersedes the historical work summaries below.

The user authorized [paper-oriented investigation](docs/audits/20260906/paper_review/PLAN.md),
including a bounded training-source neutral-control probe and a comprehensive
research report. The [comprehensive report and figure](docs/audits/20260906/paper_review/COMPREHENSIVE_REPORT_ZH.md)
are complete. Completed development rounds below remain historical evidence.

The user authorized [round two](docs/audits/20260906/round2/PLAN.md): diagnose
Refresh gain learning on real train states and compare paired targets, optimization
budget, weighting and train-only standardization. Round-one results below remain
historical development evidence. Round two is complete: see the
[results and diagnostic figure](docs/audits/20260906/round2/RESULTS_ZH.md).

本轮依据[三分支审计](docs/audits/20260906/PROJECT_REVIEW_ZH.md)，修正源支持统计、
option-conditioned outcome prediction 的状态×选项交互，以及开发评估与校准。
[当前状态](docs/CURRENT.md)和[第一轮计划](docs/audits/20260906/ROUND1_PLAN.md)
是新的研究入口。第一轮已完成，见[修复与实验结果](docs/audits/20260906/ROUND1_RESULTS_ZH.md)。

The Option-Conditioned Distributional Utility Router (ODUR) is compared with its
original additive architecture, independent option heads and DirectQ on the same
104-D pooled camera/proprio features. All learned models fit train sources only;
development remains an exposed development readout. No successful, superior, or
deployable ODUR method is claimed from an implementation fix.

Historical release: `SCOPED_TEST_SCOPE_FAILURE_RELEASE`. Its frozen machine
artifacts remain unchanged. The old "only 1 source" statistic counted entirely-B0
sources, not sources containing B0 states. The exposed D8 corpus contains B0 states
in all 32 sources; 31 also contain B1 states. This correction is retrospective and
does not turn the exposed test into new confirmatory evidence.

The [original ODUR README](docs/audits/20260906/README_ODUR_20260831.txt) is preserved
byte-for-byte, including its original negative interpretation. The glass work,
*Knowing When to Intervene*, remains historical evidence; see the
[publication-first resolution](docs/iclr27/PUBLICATION_FIRST_RESOLUTION.md).

Run local checks with `python -m pytest tests -q` and `python scripts/audit_repo.py`.
Quest execution uses [QUEST_WORKFLOW.md](QUEST_WORKFLOW.md) and
`setup/audit_repair_round1.sbatch`, a bounded CPU job over existing D5 data.
