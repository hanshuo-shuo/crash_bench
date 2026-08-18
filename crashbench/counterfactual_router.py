"""Data contract for counterfactual option-routing supervision.

The collector stores raw frozen-VLA features.  PCA is deliberately deferred to
training so it can be fitted on source-disjoint training states only.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


OPTIONS = ("base_continue", "detour_complete", "retreat_hold")
OUTCOMES = ("task_success", "catastrophe", "safe_noncompletion")
DEFAULT_HORIZONS = (40, 30, 20, 10, 5)


def outcome_probabilities(logits: np.ndarray) -> np.ndarray:
    """Convert option/outcome logits to probabilities along the outcome axis."""

    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def option_utilities(
    probabilities: np.ndarray, catastrophe_cost: float
) -> np.ndarray:
    """Compute ``p(success) - lambda * p(catastrophe)`` for each option."""

    values = np.asarray(probabilities, dtype=np.float64)
    return values[..., 0] - float(catastrophe_cost) * values[..., 1]


def conservative_option_choice(
    probabilities: np.ndarray,
    *,
    catastrophe_cost: float,
    intervention_margin: float,
) -> np.ndarray:
    """Choose the best intervention only when its Base advantage clears delta."""

    utility = option_utilities(probabilities, catastrophe_cost)
    intervention = 1 + np.argmax(utility[..., 1:], axis=-1)
    advantage = np.take_along_axis(
        utility, np.expand_dims(intervention, axis=-1), axis=-1
    )[..., 0] - utility[..., 0]
    return np.where(advantage > float(intervention_margin), intervention, 0).astype(
        np.int64
    )


def calibrated_advantage_lcb(
    probabilities: np.ndarray,
    *,
    catastrophe_cost: float,
    intervention_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return utility, Base-relative advantage, and calibrated lower score.

    The frozen router has a calibration-set Base-favoring margin rather than a
    bootstrap posterior.  Its deployable conservative score is therefore
    ``LCB_cal(Delta(o)) = Delta_hat(o) - margin`` for non-Base options.  Base is
    the reference and keeps score zero.
    """

    utility = option_utilities(probabilities, catastrophe_cost)
    advantage = utility - utility[..., [0]]
    lcb = advantage - float(intervention_margin)
    lcb[..., 0] = 0.0
    return utility, advantage, lcb


@dataclass
class FirstCrossingRouter:
    """Episode-local first-crossing state machine with an option latch."""

    catastrophe_cost: float
    intervention_margin: float
    selected_option_index: int | None = None
    trigger_action_index: int | None = None

    @property
    def latched(self) -> bool:
        return self.selected_option_index is not None

    def reset(self) -> None:
        self.selected_option_index = None
        self.trigger_action_index = None

    def observe(
        self, probabilities: np.ndarray, *, action_index: int
    ) -> dict[str, Any]:
        """Score one causal decision state and latch at the first positive LCB."""

        utility, advantage, lcb = calibrated_advantage_lcb(
            probabilities,
            catastrophe_cost=self.catastrophe_cost,
            intervention_margin=self.intervention_margin,
        )
        candidate = 1 + int(np.argmax(lcb[1:]))
        first_crossing = not self.latched and float(lcb[candidate]) > 0.0
        if first_crossing:
            self.selected_option_index = candidate
            self.trigger_action_index = int(action_index)
        return {
            "action_index": int(action_index),
            "utility": np.asarray(utility, dtype=np.float64),
            "advantage_vs_base": np.asarray(advantage, dtype=np.float64),
            "calibrated_lcb": np.asarray(lcb, dtype=np.float64),
            "candidate_option_index": int(candidate),
            "first_crossing": bool(first_crossing),
            "latched_option_index": self.selected_option_index,
        }


@dataclass(frozen=True)
class FrozenOutcomeRouter:
    """Small deployable single-frame counterfactual outcome router."""

    manifest: Mapping[str, Any]
    pca_mean: np.ndarray
    pca_components: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    outcome_coef: np.ndarray
    risk_coef: np.ndarray

    @classmethod
    def load(cls, manifest_path: str | Path) -> "FrozenOutcomeRouter":
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text())
        artifact = path.parent / str(manifest["artifact_npz"])
        with np.load(artifact) as archive:
            arrays = {name: archive[name] for name in archive.files}
        return cls(manifest=manifest, **arrays)

    def features(
        self, hidden: np.ndarray, robot_state: np.ndarray, nominal_action: np.ndarray
    ) -> np.ndarray:
        hidden = np.asarray(hidden, dtype=np.float64)
        projected = (hidden - self.pca_mean) @ self.pca_components
        return np.concatenate((
            projected,
            np.asarray(robot_state, dtype=np.float64),
            np.asarray(nominal_action, dtype=np.float64),
        ))

    def predict(
        self, hidden: np.ndarray, robot_state: np.ndarray, nominal_action: np.ndarray
    ) -> dict[str, Any]:
        feature = self.features(hidden, robot_state, nominal_action)
        standardized = (feature - self.feature_mean) / self.feature_scale
        design = np.concatenate((standardized, [1.0]))
        probabilities = outcome_probabilities(
            np.einsum("d,dok->ok", design, self.outcome_coef)
        )
        risk_logit = float(design @ self.risk_coef)
        risk_probability = float(1.0 / (1.0 + np.exp(-np.clip(risk_logit, -40, 40))))
        return {
            "option_outcome_probabilities": probabilities,
            "base_catastrophe_probability": risk_probability,
        }

    def choose(
        self,
        prediction: Mapping[str, Any],
        *,
        catastrophe_cost: float,
        intervention_margin: float,
    ) -> int:
        return int(conservative_option_choice(
            np.asarray(prediction["option_outcome_probabilities"]),
            catastrophe_cost=catastrophe_cost,
            intervention_margin=intervention_margin,
        ))


def classify_option_outcome(*, crashed: bool, succeeded: bool) -> str:
    """Map a terminal rollout to one of the router's exhaustive outcomes."""

    if crashed and succeeded:
        raise ValueError("an option rollout cannot both crash and succeed")
    if crashed:
        return "catastrophe"
    if succeeded:
        return "task_success"
    return "safe_noncompletion"


def temporal_window(
    values: Sequence[np.ndarray], anchor_index: int, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a zero-left-padded causal window and its validity mask."""

    if length < 1:
        raise ValueError("temporal window length must be positive")
    if not 0 <= int(anchor_index) < len(values):
        raise ValueError("anchor index lies outside the captured trajectory")
    anchor_index = int(anchor_index)
    arrays = [np.asarray(value) for value in values]
    shape = arrays[anchor_index].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("temporal features must have a constant shape")
    dtype = np.result_type(*(array.dtype for array in arrays))
    window = np.zeros((length, *shape), dtype=dtype)
    mask = np.zeros(length, dtype=np.float32)
    start = max(0, anchor_index - length + 1)
    captured = np.stack(arrays[start:anchor_index + 1]).astype(dtype, copy=False)
    window[-len(captured):] = captured
    mask[-len(captured):] = 1.0
    return window, mask


def validate_decision_rows(
    rows: Iterable[Mapping],
    *,
    options: Sequence[str] = OPTIONS,
    outcomes: Sequence[str] = OUTCOMES,
) -> dict:
    """Validate a source-grouped option table and return compact accounting."""

    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError("counterfactual option table is empty")
    expected_options = tuple(options)
    allowed_outcomes = set(outcomes)
    seen: set[tuple[str, str, int, str, str]] = set()
    decisions: dict[tuple[str, str, int, str], set[str]] = defaultdict(set)
    source_splits: dict[str, str] = {}
    outcome_counts: Counter[str] = Counter()
    for row in rows:
        required = {
            "source_state_sha256", "placement_id", "horizon_actions",
            "condition", "option", "outcome", "split",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(f"option row is missing {sorted(missing)}")
        source = str(row["source_state_sha256"])
        split = str(row["split"])
        previous = source_splits.setdefault(source, split)
        if previous != split:
            raise ValueError(f"source state {source} leaks across {previous}/{split}")
        option = str(row["option"])
        outcome = str(row["outcome"])
        if option not in expected_options:
            raise ValueError(f"unsupported option {option!r}")
        if outcome not in allowed_outcomes:
            raise ValueError(f"unsupported outcome {outcome!r}")
        decision = (
            source,
            str(row["placement_id"]),
            int(row["horizon_actions"]),
            str(row["condition"]),
        )
        key = (*decision, option)
        if key in seen:
            raise ValueError(f"duplicate option rollout {key}")
        seen.add(key)
        decisions[decision].add(option)
        outcome_counts[outcome] += 1
    incomplete = {
        decision: sorted(set(expected_options) - present)
        for decision, present in decisions.items()
        if present != set(expected_options)
    }
    if incomplete:
        first, missing = next(iter(incomplete.items()))
        raise ValueError(f"decision {first} is missing options {missing}")
    return {
        "decision_states": len(decisions),
        "option_rollouts": len(rows),
        "source_states": len(source_splits),
        "source_states_by_split": dict(sorted(Counter(source_splits.values()).items())),
        "outcomes": {name: int(outcome_counts[name]) for name in outcomes},
    }
