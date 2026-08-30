from __future__ import annotations

import numpy as np
import pytest

from crashbench.models.temporal import (
    TemporalOptionModel,
    causal_prefix_batch,
    optimal_stopping_labels,
)


def test_backward_dp_selects_wait_then_best_stop():
    utility = np.array([[0.0, -1.0], [0.2, 0.1], [0.1, 1.0]])
    labels = optimal_stopping_labels(
        utility,
        option_ids=("refresh", "safe"),
        terminal_wait_value=-0.5,
        delay_cost=0.01,
    )
    assert labels.optimal_action[-1] == "safe"
    assert labels.optimal_action[0] == "wait"
    assert labels.value[0] > utility[0].max()


def test_latency_shifts_stop_reachability():
    utility = np.array([[1.0], [-1.0]])
    no_latency = optimal_stopping_labels(
        utility, option_ids=("refresh",), terminal_wait_value=0, delay_cost=0, latency_steps=0
    )
    delayed = optimal_stopping_labels(
        utility, option_ids=("refresh",), terminal_wait_value=0, delay_cost=0, latency_steps=1
    )
    assert no_latency.q_stop[0, 0] == 1
    assert delayed.q_stop[0, 0] == -1


def test_causal_prefixes_do_not_expose_future_rows():
    sequence = np.arange(12, dtype=np.float32).reshape(4, 3)
    batch, lengths = causal_prefix_batch(sequence)
    assert lengths.tolist() == [1, 2, 3, 4]
    np.testing.assert_array_equal(batch[1, :2], sequence[:2])
    assert np.all(batch[1, 2:] == 0)


def test_temporal_model_output_shapes():
    torch = pytest.importorskip("torch")
    model = TemporalOptionModel(input_dim=5, option_ids=("refresh", "safe"), hidden_dim=8)
    outputs = model(torch.randn(3, 4, 5), torch.tensor([4, 2, 3]))
    assert outputs["q_wait"].shape == (3,)
    assert outputs["q_stop"].shape == (3, 2)
