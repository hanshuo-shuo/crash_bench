"""Probe-guarded policy for the scoped base-wall intervention.

Wraps a base VLA (OpenVLA, capture_hidden=True) with the frozen self-report probe. On
every step it runs the base policy, reads the base's last hidden state, and scores the
crash-imminence logit. While the logit stays below threshold it passes the base action
through unchanged. The first time the logit crosses threshold ("I am about to crash") it
LATCHES into abort: it hands control to an online RetreatHold controller for the rest of
the episode (retreat from the wall + hold, the witnessed 0 N safe recovery).

The validated scope is OpenVLA base, a wall-trained probe, the on-path wall crash mode, and
a structured RetreatHold controller. It is not general collision avoidance or general recovery.
The wrapper is model-agnostic at the interface level, but that does not expand the evidence scope.
"""

from __future__ import annotations

import numpy as np

from crashbench.probe import Probe
from crashbench.recovery import RetreatHold


class GuardedPolicy:
    def __init__(self, base, probe: Probe, thr: float | None = None, recovery=None):
        """base: a hidden-state-capable policy; probe: a loaded wall probe.
        thr: override the probe's stored threshold (else uses probe.thr)."""
        self.base = base
        self.probe = probe
        self.thr = probe.thr if thr is None else float(thr)
        self.recovery = recovery if recovery is not None else RetreatHold()
        self.reset()

    def reset(self) -> None:
        self.aborting = False
        self.trigger_step = -1
        self.logits: list[float] = []
        self._t = -1

    @property
    def resize_size(self) -> int:
        return self.base.resize_size

    def act(self, observation: dict, instruction: str) -> np.ndarray:
        self._t += 1
        if self.aborting:                                  # latched: stay in recovery
            return self.recovery.step(observation)

        action = self.base.act(observation, instruction)
        h = self.base.last_hidden
        logit = float("nan") if h is None else self.probe.logit(h)
        self.logits.append(logit)

        if h is not None and logit > self.thr:             # "I will crash" -> intervene
            self.aborting = True
            self.trigger_step = self._t
            self.recovery.engage(observation)
            return self.recovery.step(observation)
        return action
