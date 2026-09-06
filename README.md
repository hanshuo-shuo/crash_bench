# CrashBench: Audit Repair Round 1

CrashBench studies intervention choice and task recovery from exact robot states.
Current work: `AUDIT_REPAIR_ROUND1_DEVELOPMENT` (2026-09-06).

本轮依据[三分支审计](docs/audits/20260906/PROJECT_REVIEW_ZH.md)，修正源支持统计、
option-conditioned outcome prediction 的状态×选项交互，以及开发评估与校准。
[当前状态](docs/CURRENT.md)和[第一轮计划](docs/audits/20260906/ROUND1_PLAN.md)
是新的研究入口。

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
