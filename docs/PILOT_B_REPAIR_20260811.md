# Pilot B repair and re-entry plan

## Status

Pilot B remains a scientific no-go.  The repair below changes the continuation
contract and oracle controller, but it does not retroactively turn r7 into a
pass.  The r7 placements may be used only for the diagnostic; any renewed
frontier must use newly authored, outcome-blind placements.

## Repaired continuation contract

The old branch snapshot contained `[time, qpos, qvel]`, three MuJoCo arrays,
and a partial OSC/interpolator snapshot.  The repaired snapshot additionally
contains:

- all available MuJoCo integration inputs (`qacc_warmstart`, actuator/control,
  applied forces, mocap, equality, userdata, and plugin state);
- OSC goals, gains, initial joint reference, torques, and interpolators;
- robosuite time/episode counters and robot temporal buffers;
- observable values, timers, sampled flags, and the observation cache;
- the exact injected model XML used at capture time.

Same-scene branches must now match simulator, controller, full continuation,
observation, and model hashes against the values captured in the live Base
rollout.  A reset-derived hash is no longer allowed to define its own expected
value.  Cross-scene off-path controls restore dynamics/controller state but
intentionally regenerate observations after moving the glass.

## Oracle repair

The path-aligned detour now first moves radially away from the glass at the
current height and lifts from that safe point before entering the lateral
lane.  This removes the old diagonal lift-and-sidestep segment that could cut
through the glass at late trigger states.  Carry and placement stage indices
are now derived from the waypoint list instead of being hard-coded.

## Required diagnostic

Run the exposed r7 placement manifest through the diagnostic-only 2x2 at one
candidate H (start with H=20):

```bash
export CB_PILOT_B_STAGE=diagnose
export CB_PILOT_B_ROOT="$HOME/results/glass_recovery_v2/pilot_b_repair_diag_20260811"
export CB_PILOT_B_DIAGNOSTIC_PLACEMENTS=/absolute/path/to/r7/frontier_placements/placements.json
export CB_PILOT_B_DIAGNOSTIC_H=20
export CB_PILOT_B_DIAGNOSTIC_CANDIDATES=6
setup/submit_glass_recovery_pilot_b.sh diagnose
```

The output has reset-method rows (`in_memory`, `serialized`) and branch columns
(`base_suffix`, `oracle_recovery`).  Interpret it as follows:

| Result | Interpretation |
|---|---|
| in-memory Base exact, serialized Base not exact | model serialization/rebuild is still defective |
| neither Base cell exact | dynamic continuation snapshot or event determinism is still defective |
| both Base cells exact, oracle fails in both | remaining bottleneck is controller/geometry recoverability |
| in-memory oracle succeeds, serialized oracle fails | rebuild still changes recovery conditions |
| all four cells pass broadly | repair is eligible for a fresh outcome-blind frontier |

Do not rerun the full six-H frontier until every observed live Base catastrophe
in the diagnostic has exact replay in both reset rows.  Oracle success should
be read separately from replay success; the diagnostic reports both instead of
using accepted three-branch pairs as a proxy for oracle recoverability.

## Re-entry rule

After the diagnostic passes, capture or reserve fresh source states, author a
new placement manifest without observing Base/oracle outcomes, and run the
frontier from scratch.  Do not promote any r7 placement or the diagnostic's
successful cells.  Pilot C remains blocked until that new frontier freezes one
common H and the fixed-H collection reaches its declared split counts.
