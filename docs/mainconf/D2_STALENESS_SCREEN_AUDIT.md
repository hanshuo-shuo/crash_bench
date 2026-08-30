# D2 observation-staleness screen audit

Status: `GO__FORMAL_SHORTLIST_ELIGIBLE`

Quest array `5205375_[0-15]` completed all sixteen exposed source shards at
commit `0100901d1cc664073b4bb12c18f00b272811ffcb`. All 144 planned
severity-condition blocks had complete Base, observation-refresh, and safe-stop
outcomes with exact environment, policy, RNG, and sensor-queue restoration.

Matched-control catastrophe was 2.083%. Under stale observations, Base
catastrophe was 25% and Base task success was 25%, compared with 89.58% control
success. Ten sources contained a beneficial intervention opportunity and six did
not; both tasks contributed benefit sources. Two sources exhibited two distinct
strict intervention winners. Every frozen screen criterion passed.

Observation staleness is therefore the first mechanism eligible for the formal
benchmark shortlist. This result does not open a test split and does not
authorize unstable-placement v2.
