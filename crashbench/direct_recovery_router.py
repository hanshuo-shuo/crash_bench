"""Frozen direct recovery-window head and its strict first-crossing latch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DIRECT_OPTIONS = ("Base", "Detour", "FailSafeHold")
DIRECT_INPUT_DIMENSION = 10


def router_output_feature(prediction: Mapping[str, Any]) -> np.ndarray:
    """Return the frozen 3x3 outcome probabilities plus Base catastrophe risk."""

    probabilities = np.asarray(
        prediction["option_outcome_probabilities"], dtype=np.float64
    )
    if probabilities.shape != (3, 3):
        raise ValueError("frozen outcome Router probabilities must have shape (3, 3)")
    feature = np.concatenate((
        probabilities.reshape(-1),
        [float(prediction["base_catastrophe_probability"])],
    ))
    if not np.all(np.isfinite(feature)):
        raise ValueError("direct Router input contains non-finite values")
    return feature


@dataclass(frozen=True)
class DirectRecoveryWindowHead:
    """Standardized three-class linear head frozen by P3.1."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    head_weight: np.ndarray
    head_bias: np.ndarray
    classes: tuple[str, ...] = DIRECT_OPTIONS

    @classmethod
    def load(cls, path: str | Path) -> "DirectRecoveryWindowHead":
        with np.load(Path(path), allow_pickle=False) as archive:
            classes = tuple(str(value) for value in archive["classes"].tolist())
            input_dimension = int(archive["input_dimension"])
            input_field = str(archive["input_feature_field"])
            arrays = {
                name: np.asarray(archive[name], dtype=np.float64)
                for name in (
                    "feature_mean", "feature_scale", "head_weight", "head_bias"
                )
            }
        if classes != DIRECT_OPTIONS:
            raise ValueError(f"direct option mapping differs from {DIRECT_OPTIONS}")
        if input_dimension != DIRECT_INPUT_DIMENSION:
            raise ValueError("direct head input dimension is not frozen at 10")
        if input_field != "router_output_feature":
            raise ValueError("direct head input field is not router_output_feature")
        if arrays["feature_mean"].shape != (DIRECT_INPUT_DIMENSION,):
            raise ValueError("direct head feature mean has the wrong shape")
        if arrays["feature_scale"].shape != (DIRECT_INPUT_DIMENSION,):
            raise ValueError("direct head feature scale has the wrong shape")
        if arrays["head_weight"].shape != (
            DIRECT_INPUT_DIMENSION, len(DIRECT_OPTIONS)
        ):
            raise ValueError("direct head weight has the wrong shape")
        if arrays["head_bias"].shape != (len(DIRECT_OPTIONS),):
            raise ValueError("direct head bias has the wrong shape")
        if np.any(arrays["feature_scale"] <= 0.0):
            raise ValueError("direct head feature scales must be positive")
        return cls(classes=classes, **arrays)

    def logits(self, feature: np.ndarray) -> np.ndarray:
        values = np.asarray(feature, dtype=np.float64)
        if values.shape[-1:] != (DIRECT_INPUT_DIMENSION,):
            raise ValueError("direct head requires a 10-D final feature axis")
        standardized = (values - self.feature_mean) / self.feature_scale
        return standardized @ self.head_weight + self.head_bias

    def probabilities(self, feature: np.ndarray) -> np.ndarray:
        logits = self.logits(feature)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


@dataclass
class DirectFirstCrossingRouter:
    """Strict source-calibrated first crossing with a frozen option latch."""

    head: DirectRecoveryWindowHead
    sequential_boundary: float
    selected_option_index: int | None = None
    trigger_action_index: int | None = None

    @property
    def latched(self) -> bool:
        return self.selected_option_index is not None

    def reset(self) -> None:
        self.selected_option_index = None
        self.trigger_action_index = None

    def observe(self, feature: np.ndarray, *, action_index: int) -> dict[str, Any]:
        logits = np.asarray(self.head.logits(feature), dtype=np.float64)
        probabilities = np.asarray(self.head.probabilities(feature), dtype=np.float64)
        candidate = 1 + int(np.argmax(logits[1:]))
        margin = float(logits[candidate] - logits[0])
        first_crossing = (
            not self.latched and margin > float(self.sequential_boundary)
        )
        if first_crossing:
            self.selected_option_index = candidate
            self.trigger_action_index = int(action_index)
        return {
            "action_index": int(action_index),
            "logits": logits,
            "probabilities": probabilities,
            "candidate_option_index": candidate,
            "margin": margin,
            "sequential_boundary": float(self.sequential_boundary),
            "first_crossing": bool(first_crossing),
            "latched_option_index": self.selected_option_index,
        }
