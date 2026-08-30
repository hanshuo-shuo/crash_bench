# D2 mechanical preflight audit

Status: `D2_MECHANICAL_PREFLIGHT_GO`

Quest array `5204996_[0-7]` derived all mechanism parameters from the tracked
nominal paths without executing a runtime option. All eight cells and all 64
source attempts were accounted for; every source was mechanically valid across
all three frozen severities.

Fragile-path cells used live static support-surface resolution. Narrow-clearance
cells used the same support resolver plus the live distal robot geom envelope.
Observation-staleness and action-drift cells used their frozen delay and bias
grids. No world-AABB support proxy, option outcome, or router score was used.

The machine gate permits the first non-unstable D2 outcome screen. The ordered
first mechanism is `fragile_path_collision_v2`. Unstable-placement v2 remains
unauthorized and absent from all executable arrays.
