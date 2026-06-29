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


class RetreatHold:
    """Retreat away from the wall (-x, slightly up) and hold, gripper clear. Mirrors
    scripts/phase2_witness.py:run_safe_abort, recomputed online from obs."""

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
