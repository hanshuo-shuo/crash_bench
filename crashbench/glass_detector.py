"""Deployable glass-risk detector and episode-level operating-point utilities.

This module is intentionally independent of the legacy learned recovery action
head.  It fits a small linear readout over frozen policy features and stores a
self-contained artifact that can later be connected to a structured controller.

The D0 contract uses one score per independent episode:

* treatment: the score at the exact H-action pre-catastrophe anchor;
* control: the maximum score anywhere in an off-path or no-glass episode.

Thresholds are selected on a source-disjoint calibration split.  Development
rows can be evaluated at the frozen threshold but never influence it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


DETECTOR_SCHEMA_VERSION = 1
DETECTOR_HORIZON_ACTIONS = 20
DETECTOR_VARIANTS = (
    "hidden_robot_action",
    "hidden_only",
    "robot_action_only",
)
DETECTOR_SPLITS = ("train", "calibration", "development")
CONTROL_CONDITIONS = ("offpath", "noglass")
TREATMENT_CONDITION = "glass"


def _condition(row: Mapping) -> str:
    value = str(row.get("condition", row.get("cond", ""))).lower()
    aliases = {
        "on_path": TREATMENT_CONDITION,
        "onpath": TREATMENT_CONDITION,
        "no_glass": "noglass",
        "off_path": "offpath",
    }
    value = aliases.get(value, value)
    if value not in (TREATMENT_CONDITION, *CONTROL_CONDITIONS):
        raise ValueError(f"unsupported detector condition {value!r}")
    return value


def _episode_id(row: Mapping) -> str:
    value = row.get("episode_id")
    if not value:
        required = ("scenario_id", "rep")
        if any(key not in row for key in required):
            raise ValueError("detector rows require episode_id or scenario_id + rep")
        value = f"{row['scenario_id']}::{_condition(row)}::rep{row['rep']}"
    return str(value)


def _source_id(row: Mapping) -> str:
    value = row.get("source_state_sha256")
    if not value:
        raise ValueError("detector rows require source_state_sha256")
    return str(value)


def validate_capture_arrays(
    hidden: np.ndarray,
    robot_state: np.ndarray,
    nominal_action: np.ndarray,
    rows: Sequence[Mapping],
) -> None:
    """Validate the deployable D0 capture schema without loading a simulator."""

    hidden = np.asarray(hidden)
    robot_state = np.asarray(robot_state)
    nominal_action = np.asarray(nominal_action)
    n_rows = len(rows)
    expected = {
        "hidden": (hidden, 2),
        "robot_state": (robot_state, 2),
        "nominal_action": (nominal_action, 2),
    }
    for name, (array, ndim) in expected.items():
        if array.ndim != ndim or array.shape[0] != n_rows:
            raise ValueError(
                f"{name} must have ndim={ndim}, first dim={n_rows}; got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    if robot_state.shape[1] != 8:
        raise ValueError("robot_state must be [N, 8]")
    if nominal_action.shape[1] != 7:
        raise ValueError("nominal_action must be [N, 7]")

    episode_contracts: dict[str, tuple[str, str, bool]] = {}
    for row in rows:
        episode = _episode_id(row)
        contract = (
            _source_id(row),
            _condition(row),
            bool(row.get("crashed_episode")),
        )
        previous = episode_contracts.setdefault(episode, contract)
        if previous != contract:
            raise ValueError(
                f"episode {episode!r} mixes source states, conditions, or outcomes"
            )
        if contract[1] in CONTROL_CONDITIONS and contract[2]:
            raise ValueError("off-path and no-glass detector controls must be crash-free")
        if contract[1] == TREATMENT_CONDITION and contract[2]:
            tte = row.get("time_to_catastrophe_actions")
            if not isinstance(tte, (int, np.integer)) or int(tte) < 1:
                raise ValueError(
                    "crashed on-path rows require positive time_to_catastrophe_actions"
                )


def validate_source_split(
    rows: Sequence[Mapping],
    source_splits: Mapping[str, str],
) -> dict[str, list[str]]:
    """Require every source state to belong to exactly one frozen D0 split."""

    normalized = {str(source): str(split) for source, split in source_splits.items()}
    bad = sorted({split for split in normalized.values() if split not in DETECTOR_SPLITS})
    if bad:
        raise ValueError(f"unsupported detector splits {bad}; expected {DETECTOR_SPLITS}")
    observed = {_source_id(row) for row in rows}
    missing = sorted(observed - set(normalized))
    if missing:
        raise ValueError(f"source split manifest is missing {len(missing)} capture sources")
    unused = sorted(set(normalized) - observed)
    if unused:
        raise ValueError(f"source split manifest contains {len(unused)} unknown sources")
    by_split = {
        split: sorted(source for source in observed if normalized[source] == split)
        for split in DETECTOR_SPLITS
    }
    empty = [split for split, sources in by_split.items() if not sources]
    if empty:
        raise ValueError(f"source split manifest has empty splits: {empty}")
    return by_split


def split_indices(
    rows: Sequence[Mapping],
    source_splits: Mapping[str, str],
    split: str,
) -> np.ndarray:
    if split not in DETECTOR_SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return np.asarray(
        [source_splits[_source_id(row)] == split for row in rows], dtype=bool
    )


def risk_frame_targets(
    rows: Sequence[Mapping],
    *,
    horizon_actions: int = DETECTOR_HORIZON_ACTIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return binary H-step targets and a censoring-aware usable-frame mask."""

    if int(horizon_actions) < 1:
        raise ValueError("horizon_actions must be positive")
    targets = np.zeros(len(rows), dtype=np.float32)
    usable = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        condition = _condition(row)
        if condition in CONTROL_CONDITIONS:
            usable[index] = True
            continue
        if not bool(row.get("crashed_episode")):
            # Right-censored on-path runs are not clean controls and do not have
            # a verified negative suffix, so D0 excludes them from supervision.
            continue
        tte = int(row["time_to_catastrophe_actions"])
        usable[index] = True
        targets[index] = float(tte <= int(horizon_actions))
    return targets, usable


def _randomized_pca(
    values: np.ndarray,
    components: int,
    *,
    seed: int,
    power_iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    centered = values - mean
    rank = min(int(components), centered.shape[0] - 1, centered.shape[1])
    if rank < 1:
        raise ValueError("PCA needs at least two rows and one feature")
    sketch_rank = min(centered.shape[1], rank + 10)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((centered.shape[1], sketch_rank))
    basis, _ = np.linalg.qr(centered @ omega, mode="reduced")
    for _ in range(max(0, int(power_iterations))):
        basis, _ = np.linalg.qr(centered @ (centered.T @ basis), mode="reduced")
    small = basis.T @ centered
    _, _, right = np.linalg.svd(small, full_matrices=False)
    return mean.astype(np.float32), right[:rank].T.astype(np.float32)


def _balanced_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=bool)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if not positive or not negative:
        raise ValueError("detector fitting needs positive and negative training frames")
    weights = np.where(labels, 0.5 / positive, 0.5 / negative)
    return weights * len(labels)


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
    iterations: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = (features - mean) / scale
    design = np.column_stack((standardized, np.ones(len(standardized))))
    weights = _balanced_weights(labels)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(int(iterations)):
        logits = np.clip(design @ coefficients, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ ((probabilities - labels) * weights) / weights.sum()
        gradient[:-1] += float(l2) * coefficients[:-1] / len(labels)
        coefficients -= float(learning_rate) * gradient
    return (
        mean.astype(np.float32),
        scale.astype(np.float32),
        coefficients.astype(np.float32),
    )


@dataclass(frozen=True)
class GlassRiskDetector:
    """Frozen linear glass-risk readout with optional hidden-state PCA."""

    variant: str
    hidden_mean: np.ndarray
    hidden_components: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    threshold: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.variant not in DETECTOR_VARIANTS:
            raise ValueError(f"unsupported detector variant {self.variant!r}")
        if not np.isfinite(float(self.threshold)):
            raise ValueError("detector threshold must be finite")
        if self.coefficients.shape != (self.feature_mean.size + 1,):
            raise ValueError("detector coefficient shape does not match feature transform")
        if self.feature_scale.shape != self.feature_mean.shape:
            raise ValueError("detector feature scale shape mismatch")

    def _features(
        self,
        hidden: np.ndarray,
        robot_state: np.ndarray,
        nominal_action: np.ndarray,
    ) -> np.ndarray:
        hidden = np.asarray(hidden, dtype=np.float32)
        robot_state = np.asarray(robot_state, dtype=np.float32)
        nominal_action = np.asarray(nominal_action, dtype=np.float32)
        single = hidden.ndim == 1
        if single:
            hidden = hidden[None, :]
            robot_state = robot_state[None, :]
            nominal_action = nominal_action[None, :]
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_mean.size:
            raise ValueError("runtime hidden dimension disagrees with detector checkpoint")
        if robot_state.shape != (len(hidden), 8):
            raise ValueError("runtime robot_state must be [N, 8]")
        if nominal_action.shape != (len(hidden), 7):
            raise ValueError("runtime nominal_action must be [N, 7]")
        pieces = []
        if self.variant != "robot_action_only":
            pieces.append((hidden - self.hidden_mean) @ self.hidden_components)
        if self.variant != "hidden_only":
            pieces.extend((robot_state, nominal_action))
        features = np.concatenate(pieces, axis=1)
        return features[0] if single else features

    def score(
        self,
        hidden: np.ndarray,
        robot_state: np.ndarray,
        nominal_action: np.ndarray,
    ) -> float | np.ndarray:
        features = self._features(hidden, robot_state, nominal_action)
        standardized = (features - self.feature_mean) / self.feature_scale
        scores = standardized @ self.coefficients[:-1] + self.coefficients[-1]
        return float(scores) if np.ndim(scores) == 0 else np.asarray(scores)

    def fires(
        self,
        hidden: np.ndarray,
        robot_state: np.ndarray,
        nominal_action: np.ndarray,
    ) -> bool | np.ndarray:
        return self.score(hidden, robot_state, nominal_action) >= self.threshold

    def with_threshold(self, threshold: float, calibration: Mapping) -> "GlassRiskDetector":
        metadata = dict(self.metadata)
        metadata["calibration"] = dict(calibration)
        return GlassRiskDetector(
            variant=self.variant,
            hidden_mean=self.hidden_mean,
            hidden_components=self.hidden_components,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            coefficients=self.coefficients,
            threshold=float(threshold),
            metadata=metadata,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": DETECTOR_SCHEMA_VERSION,
            "kind": "glass_risk_detector",
            "variant": self.variant,
            **dict(self.metadata),
        }
        np.savez_compressed(
            path,
            hidden_mean=np.asarray(self.hidden_mean, dtype=np.float32),
            hidden_components=np.asarray(self.hidden_components, dtype=np.float32),
            feature_mean=np.asarray(self.feature_mean, dtype=np.float32),
            feature_scale=np.asarray(self.feature_scale, dtype=np.float32),
            coefficients=np.asarray(self.coefficients, dtype=np.float32),
            threshold=np.asarray(self.threshold, dtype=np.float64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GlassRiskDetector":
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            if metadata.get("schema_version") != DETECTOR_SCHEMA_VERSION:
                raise ValueError("unsupported glass detector checkpoint schema")
            if metadata.get("kind") != "glass_risk_detector":
                raise ValueError("checkpoint is not a glass risk detector")
            return cls(
                variant=str(metadata["variant"]),
                hidden_mean=np.asarray(archive["hidden_mean"], dtype=np.float32),
                hidden_components=np.asarray(
                    archive["hidden_components"], dtype=np.float32
                ),
                feature_mean=np.asarray(archive["feature_mean"], dtype=np.float32),
                feature_scale=np.asarray(archive["feature_scale"], dtype=np.float32),
                coefficients=np.asarray(archive["coefficients"], dtype=np.float32),
                threshold=float(archive["threshold"]),
                metadata={
                    key: value
                    for key, value in metadata.items()
                    if key not in ("schema_version", "kind", "variant")
                },
            )


def fit_glass_detector(
    hidden: np.ndarray,
    robot_state: np.ndarray,
    nominal_action: np.ndarray,
    targets: np.ndarray,
    *,
    variant: str,
    pca_components: int = 50,
    l2: float = 2.0,
    iterations: int = 800,
    learning_rate: float = 0.2,
    seed: int = 0,
    metadata: Mapping | None = None,
) -> GlassRiskDetector:
    if variant not in DETECTOR_VARIANTS:
        raise ValueError(f"unsupported detector variant {variant!r}")
    hidden = np.asarray(hidden, dtype=np.float32)
    robot_state = np.asarray(robot_state, dtype=np.float32)
    nominal_action = np.asarray(nominal_action, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    validate_capture_arrays(
        hidden,
        robot_state,
        nominal_action,
        [
            {
                "episode_id": f"fit_{index}",
                "source_state_sha256": f"fit_{index}",
                "condition": "noglass",
            }
            for index in range(len(hidden))
        ],
    )
    if targets.shape != (len(hidden),):
        raise ValueError("detector targets must be one-dimensional and frame-aligned")
    if np.any((targets != 0) & (targets != 1)):
        raise ValueError("detector targets must be binary")

    if variant == "robot_action_only":
        hidden_mean = np.zeros(hidden.shape[1], dtype=np.float32)
        hidden_components = np.empty((hidden.shape[1], 0), dtype=np.float32)
        features = np.concatenate((robot_state, nominal_action), axis=1)
    else:
        hidden_mean, hidden_components = _randomized_pca(
            hidden, pca_components, seed=seed
        )
        projected = (hidden - hidden_mean) @ hidden_components
        if variant == "hidden_only":
            features = projected
        else:
            features = np.concatenate((projected, robot_state, nominal_action), axis=1)
    feature_mean, feature_scale, coefficients = _fit_logistic(
        features,
        targets,
        l2=l2,
        iterations=iterations,
        learning_rate=learning_rate,
    )
    return GlassRiskDetector(
        variant=variant,
        hidden_mean=hidden_mean,
        hidden_components=hidden_components,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        threshold=0.0,
        metadata=dict(metadata or {}),
    )


def _group_episode_scores(
    scores: np.ndarray,
    rows: Sequence[Mapping],
    selected: np.ndarray,
    *,
    horizon_actions: int,
) -> tuple[list[dict], list[dict], int]:
    scores = np.asarray(scores, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    if scores.shape != (len(rows),) or selected.shape != (len(rows),):
        raise ValueError("episode scoring inputs must be frame-aligned")
    episodes: dict[str, list[int]] = {}
    for index in np.flatnonzero(selected):
        episodes.setdefault(_episode_id(rows[index]), []).append(int(index))
    treatment, controls = [], []
    excluded_onpath = 0
    for episode, indices in sorted(episodes.items()):
        conditions = {_condition(rows[index]) for index in indices}
        sources = {_source_id(rows[index]) for index in indices}
        if len(conditions) != 1 or len(sources) != 1:
            raise ValueError(f"episode {episode!r} mixes contracts")
        condition = next(iter(conditions))
        source = next(iter(sources))
        if condition in CONTROL_CONDITIONS:
            controls.append({
                "episode_id": episode,
                "source_state_sha256": source,
                "condition": condition,
                "score": float(scores[indices].max()),
            })
            continue
        if not any(bool(rows[index].get("crashed_episode")) for index in indices):
            excluded_onpath += 1
            continue
        anchors = [
            index for index in indices
            if int(rows[index]["time_to_catastrophe_actions"]) == int(horizon_actions)
        ]
        if len(anchors) != 1:
            raise ValueError(
                f"crashed treatment episode {episode!r} needs exactly one T-{horizon_actions} anchor"
            )
        treatment.append({
            "episode_id": episode,
            "source_state_sha256": source,
            "condition": condition,
            "score": float(scores[anchors[0]]),
        })
    if not treatment or not controls:
        raise ValueError("operating-point evaluation needs treatment and control episodes")
    return treatment, controls, excluded_onpath


def select_operating_point(
    treatment_scores: Sequence[float],
    control_scores: Sequence[float],
    *,
    max_control_episode_fpr: float = 0.10,
) -> dict:
    """Select a finite threshold without using a magic probability sentinel."""

    treatment = np.asarray(treatment_scores, dtype=float)
    controls = np.asarray(control_scores, dtype=float)
    if not len(treatment) or not len(controls):
        raise ValueError("threshold selection needs treatment and control episodes")
    if not np.isfinite(treatment).all() or not np.isfinite(controls).all():
        raise ValueError("threshold selection scores must be finite")
    if not 0.0 <= float(max_control_episode_fpr) <= 1.0:
        raise ValueError("max_control_episode_fpr must lie in [0, 1]")
    upper = np.nextafter(float(max(treatment.max(), controls.max())), np.inf)
    candidates = np.unique(np.concatenate((treatment, controls, [upper])))
    best = None
    for threshold in candidates:
        fpr = float(np.mean(controls >= threshold))
        timely = float(np.mean(treatment >= threshold))
        if fpr <= float(max_control_episode_fpr) + 1e-12:
            key = (timely, -fpr, float(threshold))
            if best is None or key > best[0]:
                best = (key, float(threshold), fpr, timely)
    assert best is not None
    return {
        "threshold": best[1],
        "timely_trigger_rate": best[3],
        "control_episode_fpr": best[2],
        "max_control_episode_fpr": float(max_control_episode_fpr),
        "treatment_episodes": int(len(treatment)),
        "control_episodes": int(len(controls)),
        "control_fpr_resolution": float(1.0 / len(controls)),
        "has_positive_operating_point": bool(best[3] > 0.0),
        "objective": (
            "maximize_exact_T20_trigger_rate_subject_to_"
            "offpath_and_noglass_episode_fpr"
        ),
    }


def episode_operating_metrics(
    scores: np.ndarray,
    rows: Sequence[Mapping],
    selected: np.ndarray,
    *,
    threshold: float | None = None,
    horizon_actions: int = DETECTOR_HORIZON_ACTIONS,
    max_control_episode_fpr: float = 0.10,
) -> dict:
    """Calibrate or evaluate one frozen episode-level detector operating point."""

    treatment, controls, excluded = _group_episode_scores(
        scores, rows, selected, horizon_actions=horizon_actions
    )
    if threshold is None:
        operating = select_operating_point(
            [row["score"] for row in treatment],
            [row["score"] for row in controls],
            max_control_episode_fpr=max_control_episode_fpr,
        )
        threshold = float(operating["threshold"])
    else:
        threshold = float(threshold)
        operating = {
            "threshold": threshold,
            "timely_trigger_rate": float(np.mean([
                row["score"] >= threshold for row in treatment
            ])),
            "control_episode_fpr": float(np.mean([
                row["score"] >= threshold for row in controls
            ])),
            "max_control_episode_fpr": float(max_control_episode_fpr),
            "treatment_episodes": len(treatment),
            "control_episodes": len(controls),
            "control_fpr_resolution": float(1.0 / len(controls)),
            "has_positive_operating_point": bool(any(
                row["score"] >= threshold for row in treatment
            )),
            "objective": "evaluate_frozen_exact_T20_episode_operating_point",
        }
    operating.update({
        "horizon_actions": int(horizon_actions),
        "calibration_unit": (
            "exact_T20_treatment_anchor_vs_full_control_episode_maximum"
        ),
        "treatment_source_states": len({
            row["source_state_sha256"] for row in treatment
        }),
        "control_source_states": len({
            row["source_state_sha256"] for row in controls
        }),
        "excluded_right_censored_onpath_episodes": int(excluded),
        "control_episode_fpr_by_condition": {
            condition: float(np.mean([
                row["score"] >= threshold
                for row in controls if row["condition"] == condition
            ])) if any(row["condition"] == condition for row in controls) else None
            for condition in CONTROL_CONDITIONS
        },
    })
    return operating


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    return float(np.mean(
        (positive[:, None] > negative[None, :])
        + 0.5 * (positive[:, None] == negative[None, :])
    ))
