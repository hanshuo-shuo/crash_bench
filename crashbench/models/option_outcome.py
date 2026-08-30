"""Option-conditioned distributional outcome and cost predictor for ODUR."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - lightweight non-training installs
    torch = None
    nn = object
    F = None


OUTCOME_CLASSES = ("task_success", "catastrophe", "safe_noncompletion")
COST_TARGETS = (
    "option_duration_norm",
    "path_length_norm",
    "force_exposure_norm",
    "latency_norm",
)


def _pooled_image(image: np.ndarray, grid: int) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] not in (1, 3, 4):
        raise ValueError(f"policy image must be HxWxC, got {value.shape}")
    value = value[..., :3].astype(np.float32)
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        value /= 255.0
    rows = []
    for y_indices in np.array_split(np.arange(value.shape[0]), grid):
        for x_indices in np.array_split(np.arange(value.shape[1]), grid):
            patch = value[np.ix_(y_indices, x_indices)]
            rows.extend(np.mean(patch, axis=(0, 1)).tolist())
    return np.asarray(rows, dtype=np.float32)


def pooled_anchor_features(
    mapping: Mapping[str, Any], *, grid: int = 4
) -> np.ndarray:
    """Fixed deployable feature view: pooled cameras plus policy proprio state."""

    if grid < 1:
        raise ValueError("feature pooling grid must be positive")
    parts = []
    image_keys = sorted(key for key, value in mapping.items() if np.asarray(value).ndim == 3)
    if not image_keys:
        raise ValueError("anchor feature mapping contains no image")
    for key in image_keys:
        parts.append(_pooled_image(np.asarray(mapping[key]), grid))
    state = np.asarray(mapping.get("state"), dtype=np.float32)
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        raise ValueError("anchor feature mapping needs a finite one-dimensional state")
    parts.append(state)
    result = np.concatenate(parts).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("pooled anchor feature contains NaN/Inf")
    return result


def source_balanced_weights(source_ids: Sequence[str]) -> np.ndarray:
    values = np.asarray(list(map(str, source_ids)))
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("source-balanced weights need at least one row")
    unique, counts = np.unique(values, return_counts=True)
    inverse = {source: 1.0 / count for source, count in zip(unique, counts)}
    weights = np.asarray([inverse[source] for source in values], dtype=np.float32)
    return weights / np.mean(weights)


if torch is not None:

    class OptionOutcomeModel(nn.Module):
        def __init__(
            self,
            *,
            input_dim: int,
            option_ids: Sequence[str],
            hidden_dim: int = 128,
            option_embedding_dim: int = 16,
        ):
            super().__init__()
            self.option_ids = tuple(map(str, option_ids))
            if input_dim < 1 or len(self.option_ids) < 2 or len(set(self.option_ids)) != len(self.option_ids):
                raise ValueError("ODUR needs positive input_dim and unique variable option IDs")
            self.option_to_index = {option: index for index, option in enumerate(self.option_ids)}
            self.option_embedding = nn.Embedding(len(self.option_ids), option_embedding_dim)
            self.trunk = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            joint_dim = hidden_dim + option_embedding_dim
            self.outcome_head = nn.Linear(joint_dim, len(OUTCOME_CLASSES))
            self.cost_head = nn.Linear(joint_dim, len(COST_TARGETS))

        def option_indices(self, option_ids: Sequence[str], *, device=None) -> "torch.Tensor":
            try:
                values = [self.option_to_index[str(option)] for option in option_ids]
            except KeyError as exc:
                raise KeyError(f"unknown option ID for ODUR: {exc.args[0]}") from exc
            return torch.tensor(values, dtype=torch.long, device=device)

        def forward(self, features: "torch.Tensor", option_indices: "torch.Tensor"):
            if features.ndim != 2 or option_indices.ndim != 1 or len(features) != len(option_indices):
                raise ValueError("ODUR expects [batch,features] and [batch] option indices")
            hidden = self.trunk(features)
            option = self.option_embedding(option_indices)
            joint = torch.cat([hidden, option], dim=-1)
            return {
                "outcome_logits": self.outcome_head(joint),
                "cost_prediction": torch.sigmoid(self.cost_head(joint)),
            }

        def loss(
            self,
            outputs: Mapping[str, "torch.Tensor"],
            *,
            outcome_targets: "torch.Tensor",
            cost_targets: "torch.Tensor",
            row_weights: "torch.Tensor",
            cost_scale: float = 1.0,
        ) -> Mapping[str, "torch.Tensor"]:
            classification = F.cross_entropy(
                outputs["outcome_logits"], outcome_targets, reduction="none"
            )
            costs = F.mse_loss(
                outputs["cost_prediction"], cost_targets, reduction="none"
            ).mean(dim=-1)
            weights = row_weights / torch.clamp(row_weights.mean(), min=1e-12)
            classification_loss = torch.mean(classification * weights)
            cost_loss = torch.mean(costs * weights)
            total = classification_loss + float(cost_scale) * cost_loss
            return {
                "loss": total,
                "classification_loss": classification_loss,
                "cost_loss": cost_loss,
            }

else:

    class OptionOutcomeModel:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("OptionOutcomeModel requires PyTorch")
