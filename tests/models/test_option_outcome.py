from __future__ import annotations

import numpy as np
import pytest

from crashbench.models.option_outcome import (
    COST_TARGETS,
    OUTCOME_CLASSES,
    OptionOutcomeModel,
    pooled_anchor_features,
    source_balanced_weights,
)


def test_pooled_anchor_features_use_both_cameras_and_state():
    mapping = {
        "full_image": np.full((8, 8, 3), 255, dtype=np.uint8),
        "wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "state": np.arange(8, dtype=np.float32),
    }
    features = pooled_anchor_features(mapping, grid=4)
    assert features.shape == (4 * 4 * 3 * 2 + 8,)
    assert np.all(features[: 4 * 4 * 3] == 1)


def test_source_balanced_weights_give_each_source_equal_total_mass():
    ids = ["a", "a", "a", "b"]
    weights = source_balanced_weights(ids)
    assert weights[:3].sum() == pytest.approx(weights[3])
    assert weights.mean() == pytest.approx(1)


def test_variable_option_model_forward_and_source_weighted_loss():
    torch = pytest.importorskip("torch")
    model = OptionOutcomeModel(
        input_dim=10,
        option_ids=("base_continue", "observation_refresh", "safe_stop"),
        hidden_dim=16,
    )
    features = torch.randn(6, 10)
    indices = model.option_indices(
        ["base_continue", "observation_refresh", "safe_stop"] * 2
    )
    outputs = model(features, indices)
    assert outputs["outcome_logits"].shape == (6, len(OUTCOME_CLASSES))
    assert outputs["cost_prediction"].shape == (6, len(COST_TARGETS))
    losses = model.loss(
        outputs,
        outcome_targets=torch.tensor([0, 1, 2, 0, 1, 2]),
        cost_targets=torch.rand(6, len(COST_TARGETS)),
        row_weights=torch.ones(6),
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


def test_unknown_option_fails_closed():
    pytest.importorskip("torch")
    model = OptionOutcomeModel(input_dim=2, option_ids=("base", "safe"))
    with pytest.raises(KeyError, match="unknown option"):
        model.option_indices(["oracle"])
