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

    @property
    def supports_exact_branching(self) -> bool:
        """Whether continuation snapshot/restore is complete for this backend."""
        ...

    def snapshot_continuation(self):
        """Return a versioned policy-continuation snapshot."""
        ...

    def restore_continuation(self, snapshot) -> None:
        """Restore a snapshot without querying the model or advancing a queue."""
        ...
