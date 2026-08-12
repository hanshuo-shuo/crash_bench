# Glass-recovery execution archive — 2026-08-12

This directory preserves the superseded E14/E15 learned-recovery roadmap and its
execution record. It is an audit trail, not the current paper narrative or a GPU
run queue.

## Contents

- `GLASS_PAPER_EXECUTION.md`: former P0-A through Pilot F execution plan.
- `GLASS_RECOVERY_V1.md`: frozen v1/v2 protocol and command contracts.
- `E15_EXPERIMENT_LOG_20260811.md`: chronological Pilot A/B execution log.
- `PILOT_B_REPAIR_20260811.md`, `PILOT_B_FRESH_H20_20260812.md`, and
  `PILOT_B_CONTROLLER_COMPATIBLE_H20_20260812.md`: authoring and scoped repair
  notes.
- `GLASS_RECOVERY_PROGRESS_REPORT_20260812.md`: illustrated audit of scoped B/C.
- `report_assets/`: media used by that audit.

## Frozen interpretation

- Broad Pilot B is a completed feasibility no-go.
- Scoped B certifies 3/15 accidents (3/12 conditional on Base catastrophe).
- Pilot C is an Oracle upper bound on two development source states, not learned
  recovery or held-out generalization.
- The frozen learned gate has no timely trigger at its calibrated threshold; the
  action head has not passed its Oracle-timing prerequisite.
- Pilots D/F are not current run targets.

The compact, hash-pinned decision record is
[`glass_recovery_checkpoint_readiness_audit_20260812.json`](../../../results/glass_recovery_checkpoint_readiness_audit_20260812.json).

For the paper-facing glass boundary, use
[GLASS_SAFETY_UTILITY.md](../../appendix/GLASS_SAFETY_UTILITY.md). For current
project decisions, use [CURRENT.md](../../CURRENT.md).

Historical documents may mention their old `docs/...` locations. Those strings
are preserved where they form part of the execution record.
