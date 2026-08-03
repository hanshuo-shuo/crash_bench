"""Online recovery controllers (Path 1-1a, PLAN.md §11 Stage 1).

The Phase-2 witness (`scripts/phase2_witness.py`) proved that a retreat+hold from the
pre-crash state avoids every treatment wall at 0 N (5/5 safe-abort). There it ran as an
OFFLINE scripted trajectory. Here we expose the SAME behaviour as an ONLINE controller
that computes its action each step from the live observation alone (no env access, no
model) — so a guarded policy can hand control to it the moment the probe fires.

`RetreatHold.engage(obs)` latches a retreat target relative to the eef at trigger time
(back off in -x, lift in +z, exactly as run_safe_abort); `step(obs)` P-controls toward it
and naturally decays to a near-zero hold once reached. The action is the 7-DoF vector
LiberoEnv.step expects (gripper already in [-1, 1]).
"""

from __future__ import annotations

import numpy as np

GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0


def _eef_from_obs(obs: dict) -> np.ndarray:
    """eef xyz from either a raw env obs (robot0_eef_pos) or a policy observation
    whose 'state' is concat(eef_pos[3], eef_axisangle[3], gripper_qpos[2])."""
    if "robot0_eef_pos" in obs:
        return np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    return np.asarray(obs["state"][:3], dtype=np.float32)


class WitnessReplay:
    """Replay a fixed witness action sequence open-loop from its recorded start state.

    It is not a general online recovery controller: a late or different trigger state invalidates
    the replay assumption. The only tracked task-completion witness is the separately identified
    low-wall d62 existence demo.
    """

    def __init__(self, actions):
        self.actions = np.asarray(actions, dtype=np.float32)
        self.k = 0

    def engage(self, obs: dict) -> None:
        self.k = 0

    def step(self, obs: dict) -> np.ndarray:
        i = min(self.k, len(self.actions) - 1)
        self.k += 1
        a = self.actions[i]
        if self.k > len(self.actions):          # past the end: hold last pose, gripper open
            return np.array([0, 0, 0, 0, 0, 0, GRIP_OPEN], dtype=np.float32)
        return np.asarray(a, dtype=np.float32)


class DetourComplete:
    """Experimental online tool-handoff recovery: route the gripper AROUND the on-path wall, grasp the bowl,
    and place it on the plate — the SAME full-arm collision-free detour proven offline by the
    task-completion witness (scripts/phase2_task_witness.py), recomputed online as a leg state
    machine so a guarded policy can hand off to it when the probe fires.

    Unlike RetreatHold (which only safe-aborts), this controller attempts task completion. It is
    not stable enough to claim general use; tall-wall d70/d78/d85 detours are negative results.
    The controller only sees the eef (from obs); the wall/bowl/plate geometry is
    INJECTED at construction (they are static pre-grasp, so their episode-start world positions —
    which the witness also used — are correct). Uses pure POSITION control at neutral orientation
    (orientation control destabilises OSC; the elbow is kept off the wall by the lowered geometry).
    """

    def __init__(self, wall: dict, target_pos, plate_pos, *, side: float = -1.0,
                 lane_margin: float = 0.22, transit_z: float | None = None, k: float = 12.0,
                 tol: float = 0.02, leg_cap: int = 80, grasp_steps: int = 18, release_steps: int = 30,
                 descend_off: float = 0.04, place_off: float = 0.015,
                 target_name: str | None = None):
        self.wall, self.k, self.tol, self.leg_cap = wall, k, tol, leg_cap
        self.side, self.lane_margin = side, lane_margin
        self.bowl = np.asarray(target_pos, dtype=np.float32)
        self.plate = np.asarray(plate_pos, dtype=np.float32)
        self.transit_z = float(self.bowl[2] + 0.40) if transit_z is None else float(transit_z)
        self.grasp_steps, self.release_steps = grasp_steps, release_steps
        self.descend_off, self.place_off = descend_off, place_off
        self.target_name = target_name
        self.legs: list | None = None
        self.i = 0
        self._in_leg = 0
        self._carry_adjusted = False

    def engage(self, obs: dict) -> None:
        eef = _eef_from_obs(obs)
        wx, wy = self.wall["pos"][0], self.wall["pos"][1]
        whx, why = self.wall["size"][0], self.wall["size"][1]
        b, p, ez = self.bowl, self.plate, self.transit_z
        dy = wy + self.side * (why + self.lane_margin)          # detour lane past the wall y-edge
        sx = max(float(b[0]), wx + whx) + 0.13                  # staging x: past wall/bowl (+x, open)
        cx = float(eef[0])
        O, C = GRIP_OPEN, GRIP_CLOSE
        # (kind, ...): move -> (target xyz, grip); hold -> (grip, n_steps). Mirrors run_detour legs.
        self.legs = [
            ("move", [cx, dy, ez], O),                          # 1. sidestep into detour lane
            ("move", [sx, dy, ez], O),                          # 2. advance past wall/bowl (+x)
            ("move", [sx, float(b[1]), ez], O),                 # 3. come to bowl y (open, +x)
            ("move", [float(b[0]), float(b[1]), ez], O),        # 4. approach bowl FROM +x
            ("move", [float(b[0]), float(b[1]), float(b[2]) + self.descend_off], O),  # 5. descend
            ("hold", C, self.grasp_steps),                      # 6. grasp
            ("move", [float(b[0]), float(b[1]), ez], C),        # 7. lift
            ("move", [float(p[0]), float(p[1]), ez], C),        # 8. carry above plate
            ("move", [float(p[0]), float(p[1]), float(p[2]) + self.place_off], C),    # 9. set on plate
            ("hold", O, self.release_steps),                    # 10. release + settle
        ]
        self.i = 0
        self._in_leg = 0
        self._carry_adjusted = False

    def _act(self, dxyz, grip):
        return np.array([float(np.clip(self.k * dxyz[0], -1, 1)),
                         float(np.clip(self.k * dxyz[1], -1, 1)),
                         float(np.clip(self.k * dxyz[2], -1, 1)),
                         0.0, 0.0, 0.0, float(grip)], dtype=np.float32)

    def step(self, obs: dict) -> np.ndarray:
        if self.legs is None:
            self.engage(obs)
        eef = _eef_from_obs(obs)
        if self.i >= len(self.legs):                            # done: hold in place, gripper open
            return self._act([0.0, 0.0, 0.0], GRIP_OPEN)
        # Once the bowl is grasped and lifted, compensate for the measured
        # bowl--EEF xy offset before carrying/lowering.  Aiming the EEF itself at
        # plate center is not enough: the held bowl hangs off-center and misses
        # LIBERO's placement region.  This mirrors the verified offline witness.
        if self.i == 7 and not self._carry_adjusted and self.target_name:
            key = f"{self.target_name}_pos"
            if key in obs:
                target_xy = np.asarray(obs[key], dtype=np.float32)[:2]
                offset = target_xy - eef[:2]
                px, py = float(self.plate[0] - offset[0]), float(self.plate[1] - offset[1])
                self.legs[7] = ("move", [px, py, self.transit_z], GRIP_CLOSE)
                self.legs[8] = ("move", [px, py, float(self.plate[2]) + self.place_off], GRIP_CLOSE)
                self._carry_adjusted = True
        leg = self.legs[self.i]
        self._in_leg += 1
        if leg[0] == "move":
            target, grip = np.asarray(leg[1], dtype=np.float32), leg[2]
            d = target - eef
            reached = float(np.linalg.norm(d)) < self.tol
            if reached or self._in_leg >= self.leg_cap:         # advance on reach or safety cap
                self.i += 1; self._in_leg = 0
            return self._act(d, grip)
        else:                                                   # hold: command grip for n steps
            grip, n = leg[1], leg[2]
            if self._in_leg >= n:
                self.i += 1; self._in_leg = 0
            return self._act([0.0, 0.0, 0.0], grip)


class RetreatHold:
    """Online closed-loop safe-abort controller: retreat from the wall and hold.

    This is the recovery used by the scoped probe-gated intervention; it intentionally does not
    claim original-task completion.
    """

    def __init__(self, back: float = 0.16, up: float = 0.10, k: float = 12.0,
                 grip: float = GRIP_OPEN):
        self.back, self.up, self.k, self.grip = back, up, k, grip
        self.target: np.ndarray | None = None

    def engage(self, obs: dict) -> None:
        eef = _eef_from_obs(obs)
        self.target = np.array([eef[0] - self.back, eef[1], eef[2] + self.up], dtype=np.float32)

    def step(self, obs: dict) -> np.ndarray:
        if self.target is None:
            self.engage(obs)
        eef = _eef_from_obs(obs)
        d = self.target - eef
        return np.array([
            float(np.clip(self.k * d[0], -1, 1)),
            float(np.clip(self.k * d[1], -1, 1)),
            float(np.clip(self.k * d[2], -1, 1)),
            0.0, 0.0, 0.0, float(self.grip),
        ], dtype=np.float32)
