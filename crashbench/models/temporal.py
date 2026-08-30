"""Causal wait-vs-stop labels and temporal option-value model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = object


@dataclass(frozen=True)
class StoppingLabels:
    q_wait: np.ndarray
    q_stop: np.ndarray
    optimal_action: tuple[str, ...]
    value: np.ndarray


def optimal_stopping_labels(
    stop_utility: np.ndarray,
    *,
    option_ids: Sequence[str],
    terminal_wait_value: float,
    delay_cost: float,
    latency_steps: int = 0,
) -> StoppingLabels:
    """Backward DP on a fixed causal monitor grid.

    ``stop_utility[t,o]`` is realized utility if option ``o`` is invoked at t.
    Waiting advances one grid point and pays ``delay_cost``. A fixed runtime
    latency shifts each stop value to the later reachable row, fail-closing to
    the terminal wait value beyond the grid.
    """

    utility = np.asarray(stop_utility, dtype=np.float64)
    options = tuple(map(str, option_ids))
    if utility.ndim != 2 or utility.shape[1] != len(options) or len(options) < 1:
        raise ValueError("stop utility must be [time,option] with matching option IDs")
    if not np.all(np.isfinite(utility)) or delay_cost < 0 or latency_steps < 0:
        raise ValueError("stopping inputs must be finite with nonnegative costs/latency")
    time_steps = len(utility)
    q_stop = np.full_like(utility, float(terminal_wait_value))
    for time in range(time_steps):
        reachable = time + int(latency_steps)
        if reachable < time_steps:
            q_stop[time] = utility[reachable]
    q_wait = np.empty(time_steps, dtype=np.float64)
    value = np.empty(time_steps, dtype=np.float64)
    actions = [""] * time_steps
    next_value = float(terminal_wait_value)
    for time in range(time_steps - 1, -1, -1):
        q_wait[time] = next_value - float(delay_cost)
        best_option_index = int(np.argmax(q_stop[time]))
        best_stop = float(q_stop[time, best_option_index])
        if q_wait[time] >= best_stop:
            value[time] = q_wait[time]
            actions[time] = "wait"
        else:
            value[time] = best_stop
            actions[time] = options[best_option_index]
        next_value = value[time]
    return StoppingLabels(q_wait, q_stop, tuple(actions), value)


def causal_prefix_batch(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build zero-padded prefixes and lengths; never expose t+1 to row t."""

    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or len(values) < 1:
        raise ValueError("temporal sequence must be non-empty [time,features]")
    time_steps, width = values.shape
    batch = np.zeros((time_steps, time_steps, width), dtype=np.float32)
    lengths = np.arange(1, time_steps + 1, dtype=np.int64)
    for time in range(time_steps):
        batch[time, : time + 1] = values[: time + 1]
    return batch, lengths


if torch is not None:

    class TemporalOptionModel(nn.Module):
        def __init__(self, *, input_dim: int, option_ids: Sequence[str], hidden_dim: int = 128):
            super().__init__()
            self.option_ids = tuple(map(str, option_ids))
            if input_dim < 1 or not self.option_ids:
                raise ValueError("temporal model needs input features and options")
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.wait_head = nn.Linear(hidden_dim, 1)
            self.stop_head = nn.Linear(hidden_dim, len(self.option_ids))

        def forward(self, prefixes: "torch.Tensor", lengths: "torch.Tensor"):
            if prefixes.ndim != 3 or lengths.ndim != 1 or len(prefixes) != len(lengths):
                raise ValueError("temporal model expects [batch,time,features] plus lengths")
            packed = nn.utils.rnn.pack_padded_sequence(
                prefixes,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
            state = hidden[-1]
            return {
                "q_wait": self.wait_head(state).squeeze(-1),
                "q_stop": self.stop_head(state),
            }

else:

    class TemporalOptionModel:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("TemporalOptionModel requires PyTorch")
