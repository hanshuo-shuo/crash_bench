"""Policy interface. A policy maps (observation, instruction) -> env-ready action."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    @property
    def resize_size(self) -> int:
        """Image size (px) the policy wants its `full_image` rendered at."""
        ...

    def act(self, observation: dict, instruction: str) -> np.ndarray:
        """Return a 7-DoF action ready to pass to LiberoEnv.step (gripper already
        normalized/inverted as the env expects)."""
        ...
