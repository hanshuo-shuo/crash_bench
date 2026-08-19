#!/usr/bin/env python3
"""Train and audit the P3.1 direct recovery-window routing head.

The head is intentionally small: the frozen Router's ten prediction outputs are
standardized and passed to one source-balanced multinomial linear softmax.  All
reported development predictions are leave-one-source_state_sha256-out (LOSO);
the separately saved candidate is fitted once on all P3.0 records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize


CLASSES = ("Base", "Detour", "FailSafeHold")
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}
FEATURE_FIELD = "router_output_feature"
FEATURE_DIMENSION = 10
FIXED_L2 = 0.01
CANDIDATE_STATUS = "development candidate; not yet dynamically confirmed"
TARGET_RECOVERY_PLACEMENTS = (
    "glass_recovery_heldout_0004",
    "glass_recovery_heldout_0009",
    "glass_recovery_heldout_0010",
)
TARGET_HOLD_PLACEMENTS = (
    "glass_recovery_heldout_0003",
    "glass_recovery_heldout_0019",
)
FORBIDDEN_MODEL_INPUTS = (
    "horizon_actions",
    "actions_to_window_closure",
    "window_closure_horizon_actions",
    "collision-relative timing",
    "action_index",
    "trajectory_id",
    "source_state_sha256",
    "source ID",
    "base_outcome",
    "detour_outcome",
    "hold_outcome",
    "true option outcomes",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def load_records(
    path: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Load the P3 records and extract the only permitted model input."""

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("P3 recovery-window record file is empty")
    features = np.asarray([row[FEATURE_FIELD] for row in records], dtype=np.float64)
    if features.shape != (len(records), FEATURE_DIMENSION):
        raise ValueError(
            f"{FEATURE_FIELD} must have shape (records, {FEATURE_DIMENSION}); "
            f"got {features.shape}"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError(f"{FEATURE_FIELD} contains non-finite values")
    unknown = sorted({str(row["preferred_option"]) for row in records} - set(CLASSES))
    if unknown:
        raise ValueError(f"unsupported preferred options: {unknown}")
    labels = np.asarray(
        [CLASS_INDEX[str(row["preferred_option"])] for row in records],
        dtype=np.int64,
    )
    sources = np.asarray([str(row["source_state_sha256"]) for row in records])
    if any(not source for source in sources):
        raise ValueError("source_state_sha256 must be non-empty")
    decision_ids = [str(row["decision_id"]) for row in records]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("decision_id values must be unique")
    return records, features, labels, sources


def source_balanced_weights(sources: np.ndarray) -> np.ndarray:
    """Give every source equal total weight and keep mean row weight at one."""

    values = np.asarray(sources).astype(str)
    if not len(values):
        raise ValueError("source-balanced weighting requires at least one row")
    counts = Counter(values.tolist())
    weights = np.asarray([1.0 / counts[source] for source in values], dtype=np.float64)
    return weights * (len(weights) / weights.sum())


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def fit_multinomial_head(
    features: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    *,
    l2: float = FIXED_L2,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit one standardized source-balanced three-way linear softmax head."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    source_values = np.asarray(sources).astype(str)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(source_values):
        raise ValueError("features, labels, and sources must have matching row counts")
    if x.shape[1] != FEATURE_DIMENSION:
        raise ValueError(f"direct head requires exactly {FEATURE_DIMENSION} features")
    if set(y.tolist()) - set(range(len(CLASSES))):
        raise ValueError("labels lie outside the declared preferred-option classes")
    if set(y.tolist()) != set(range(len(CLASSES))):
        raise ValueError("every training fit must contain all three preferred-option classes")

    sample_weight = source_balanced_weights(source_values)
    mean = np.average(x, axis=0, weights=sample_weight)
    variance = np.average((x - mean) ** 2, axis=0, weights=sample_weight)
    scale = np.sqrt(np.maximum(variance, 0.0)) + 1e-6
    standardized = (x - mean) / scale
    design = np.column_stack((standardized, np.ones(len(standardized))))
    normalizer = float(sample_weight.sum())
    rows = np.arange(len(y))

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coefficient = flat.reshape(design.shape[1], len(CLASSES))
        logits = design @ coefficient
        probabilities = _softmax(logits)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_normalizer = np.log(np.exp(shifted).sum(axis=1))
        negative_log_likelihood = log_normalizer - shifted[rows, y]
        loss = float(np.sum(sample_weight * negative_log_likelihood) / normalizer)
        loss += 0.5 * float(l2) * float(np.sum(coefficient[:-1] ** 2))

        residual = probabilities
        residual[rows, y] -= 1.0
        residual *= sample_weight[:, None] / normalizer
        gradient = design.T @ residual
        gradient[:-1] += float(l2) * coefficient[:-1]
        return loss, gradient.ravel()

    initial = np.zeros(design.shape[1] * len(CLASSES), dtype=np.float64)
    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fitted.success:
        raise RuntimeError(f"multinomial head did not converge: {fitted.message}")
    coefficient = fitted.x.reshape(design.shape[1], len(CLASSES))
    # The zero initialization and softmax gradient keep this centered already;
    # explicit centering makes the saved representation canonical.
    coefficient -= coefficient.mean(axis=1, keepdims=True)
    model = {
        "feature_mean": mean,
        "feature_scale": scale,
        "head_weight": coefficient[:-1],
        "head_bias": coefficient[-1],
    }
    fit_summary = {
        "converged": True,
        "optimizer": "L-BFGS-B",
        "iterations": int(fitted.nit),
        "objective_weighted_cross_entropy_plus_l2": float(fitted.fun),
        "training_records": int(len(y)),
        "training_sources": int(len(set(source_values.tolist()))),
        "training_class_counts": {
            name: int(np.sum(y == index)) for index, name in enumerate(CLASSES)
        },
    }
    return model, fit_summary


def predict_multinomial_head(
    model: Mapping[str, np.ndarray], features: np.ndarray
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    standardized = (x - model["feature_mean"]) / model["feature_scale"]
    logits = standardized @ model["head_weight"] + model["head_bias"]
    return _softmax(logits)


def leave_one_source_out_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    *,
    l2: float = FIXED_L2,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Return strict LOSO probabilities, fold indices, and fit provenance."""

    source_values = np.asarray(sources).astype(str)
    unique_sources = sorted(set(source_values.tolist()))
    if len(unique_sources) < 2:
        raise ValueError("LOSO evaluation requires at least two sources")
    probabilities = np.full((len(labels), len(CLASSES)), np.nan, dtype=np.float64)
    fold_index = np.full(len(labels), -1, dtype=np.int64)
    folds = []
    for index, held_out_source in enumerate(unique_sources):
        test = source_values == held_out_source
        train = ~test
        model, fit_summary = fit_multinomial_head(
            features[train], labels[train], source_values[train], l2=l2
        )
        probabilities[test] = predict_multinomial_head(model, features[test])
        fold_index[test] = index
        train_sources = sorted(set(source_values[train].tolist()))
        if held_out_source in train_sources:
            raise AssertionError("held-out source leaked into LOSO training")
        folds.append({
            "fold_index": index,
            "held_out_source_state_sha256": held_out_source,
            "held_out_records": int(test.sum()),
            "training_records": int(train.sum()),
            "training_source_count": len(train_sources),
            "training_sources": train_sources,
            "fit": fit_summary,
        })
    if np.any(fold_index < 0) or not np.all(np.isfinite(probabilities)):
        raise AssertionError("LOSO predictions are incomplete")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
        raise AssertionError("LOSO softmax probabilities do not sum to one")
    return probabilities, fold_index, folds


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for truth, prediction in zip(labels, predictions):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, sources: np.ndarray
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    source_values = np.asarray(sources).astype(str)
    matrix = confusion_matrix(y, predicted)
    per_class = {}
    f1_values = []
    for index, name in enumerate(CLASSES):
        true_positive = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted_count = int(matrix[:, index].sum())
        recall = None if support == 0 else true_positive / support
        precision = None if predicted_count == 0 else true_positive / predicted_count
        denominator = support + predicted_count
        f1 = 0.0 if denominator == 0 else (2.0 * true_positive / denominator)
        f1_values.append(f1)
        per_class[name] = {
            "support": support,
            "predicted_count": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    source_accuracy = {
        source: float(np.mean(predicted[source_values == source] == y[source_values == source]))
        for source in sorted(set(source_values.tolist()))
    }
    return {
        "record_accuracy": float(np.mean(predicted == y)),
        "source_macro_accuracy": float(np.mean(list(source_accuracy.values()))),
        "macro_f1_fixed_three_class": float(np.mean(f1_values)),
        "class_metrics": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_orientation": "rows=true preferred option; columns=predicted option",
        "source_accuracy": source_accuracy,
    }


def binary_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    """Compute ordinary record-level ROC AUC with half credit for score ties."""

    y = np.asarray(target, dtype=bool)
    values = np.asarray(score, dtype=np.float64)
    positive = values[y]
    negative = values[~y]
    if not len(positive) or not len(negative):
        return None
    comparison = positive[:, None] - negative[None, :]
    return float(np.mean((comparison > 0.0) + 0.5 * (comparison == 0.0)))


def old_operational_predictions(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Reconstruct the frozen Router's original pointwise operational choice."""

    choices = []
    for row in records:
        if float(row["old_router_pointwise_lcb"]) > 0.0:
            choices.append(CLASS_INDEX[str(row["old_router_candidate_option"])])
        else:
            choices.append(CLASS_INDEX["Base"])
    return np.asarray(choices, dtype=np.int64)


def old_scalar_advantage_predictions(
    records: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Turn the raw scalar axis into its natural untuned sign-based option choice."""

    choices = []
    for row in records:
        if float(row["old_router_advantage"]) > 0.0:
            choices.append(CLASS_INDEX[str(row["old_router_candidate_option"])])
        else:
            choices.append(CLASS_INDEX["Base"])
    return np.asarray(choices, dtype=np.int64)


def _ranking_metrics(
    records: Sequence[Mapping[str, Any]],
    direct_probabilities: np.ndarray,
    old_predictions: np.ndarray,
) -> dict[str, dict[str, Any]]:
    recovery = np.asarray([bool(row["recovery_open"]) for row in records])
    intervention = np.asarray([bool(row["intervention_needed"]) for row in records])
    hard = np.asarray([bool(row["hard_negative"]) for row in records])
    old_advantage = np.asarray(
        [float(row["old_router_advantage"]) for row in records], dtype=np.float64
    )

    recovery_comparison = recovery | hard
    intervention_comparison = intervention | hard
    base_index = CLASS_INDEX["Base"]
    detour_index = CLASS_INDEX["Detour"]
    return {
        "old_router_scalar_advantage": {
            "score": "old Router max non-Base scalar advantage",
            "recovery_open_vs_hard_negative_auc": binary_auc(
                recovery[recovery_comparison], old_advantage[recovery_comparison]
            ),
            "intervention_needed_vs_hard_negative_auc": binary_auc(
                intervention[intervention_comparison], old_advantage[intervention_comparison]
            ),
        },
        "old_router_operational_decision": {
            "score": "discrete original operational choice (diagnostic AUC only)",
            "recovery_open_vs_hard_negative_auc": binary_auc(
                recovery[recovery_comparison],
                (old_predictions[recovery_comparison] == detour_index).astype(float),
            ),
            "intervention_needed_vs_hard_negative_auc": binary_auc(
                intervention[intervention_comparison],
                (old_predictions[intervention_comparison] != base_index).astype(float),
            ),
        },
        "direct_recovery_window_head": {
            "score": "OOF P(Detour) for recovery; OOF 1-P(Base) for intervention",
            "recovery_open_vs_hard_negative_auc": binary_auc(
                recovery[recovery_comparison],
                direct_probabilities[recovery_comparison, detour_index],
            ),
            "intervention_needed_vs_hard_negative_auc": binary_auc(
                intervention[intervention_comparison],
                1.0 - direct_probabilities[intervention_comparison, base_index],
            ),
        },
    }


def _recall_for_mask(
    labels: np.ndarray, predictions: np.ndarray, mask: np.ndarray, class_name: str
) -> dict[str, Any]:
    class_index = CLASS_INDEX[class_name]
    selected = np.asarray(mask, dtype=bool) & (labels == class_index)
    return {
        "support": int(selected.sum()),
        "recall": (
            None
            if not selected.any()
            else float(np.mean(predictions[selected] == class_index))
        ),
    }


def targeted_diagnostics(
    records: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    method_predictions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    placements = np.asarray([str(row["placement_id"]) for row in records])
    recovery = np.asarray([bool(row["recovery_open"]) for row in records])
    hard = np.asarray([bool(row["hard_negative"]) for row in records])
    result: dict[str, Any] = {
        "recovery_open_recall_by_placement": {},
        "hold_recall_by_placement": {},
        "hard_control_base_retention": {},
    }
    for placement in TARGET_RECOVERY_PLACEMENTS:
        mask = (placements == placement) & recovery
        result["recovery_open_recall_by_placement"][placement] = {
            method: _recall_for_mask(labels, prediction, mask, "Detour")
            for method, prediction in method_predictions.items()
        }
    for placement in TARGET_HOLD_PLACEMENTS:
        mask = placements == placement
        result["hold_recall_by_placement"][placement] = {
            method: _recall_for_mask(labels, prediction, mask, "FailSafeHold")
            for method, prediction in method_predictions.items()
        }
    for method, prediction in method_predictions.items():
        result["hard_control_base_retention"][method] = {
            "support": int(hard.sum()),
            "retention": (
                None
                if not hard.any()
                else float(np.mean(prediction[hard] == CLASS_INDEX["Base"]))
            ),
        }
    return result


def per_source_results(
    records: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    predictions: np.ndarray,
    sources: np.ndarray,
    folds: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source_values = np.asarray(sources).astype(str)
    fold_by_source = {
        str(fold["held_out_source_state_sha256"]): int(fold["fold_index"])
        for fold in folds
    }
    result = []
    for source in sorted(set(source_values.tolist())):
        mask = source_values == source
        subset_records = [row for row, selected in zip(records, mask) if selected]
        metrics = classification_metrics(labels[mask], predictions[mask], source_values[mask])
        result.append({
            "fold_index": fold_by_source[source],
            "held_out_source_state_sha256": source,
            "placements": sorted({str(row["placement_id"]) for row in subset_records}),
            "record_types": dict(sorted(Counter(
                str(row["record_type"]) for row in subset_records
            ).items())),
            "records": int(mask.sum()),
            "true_class_counts": {
                name: int(np.sum(labels[mask] == index))
                for index, name in enumerate(CLASSES)
            },
            "predicted_class_counts": {
                name: int(np.sum(predictions[mask] == index))
                for index, name in enumerate(CLASSES)
            },
            "accuracy": metrics["record_accuracy"],
            "macro_f1_fixed_three_class": metrics["macro_f1_fixed_three_class"],
            "class_recall": {
                name: metrics["class_metrics"][name]["recall"] for name in CLASSES
            },
            "confusion_matrix": metrics["confusion_matrix"],
        })
    return result


def _write_oof_csv(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    predictions: np.ndarray,
    scalar_predictions: np.ndarray,
    old_predictions: np.ndarray,
    fold_index: np.ndarray,
) -> None:
    fields = [
        "row_index",
        "fold_index",
        "held_out_source_state_sha256",
        "decision_id",
        "placement_id",
        "record_type",
        "condition",
        "preferred_option",
        "direct_predicted_option",
        "direct_correct",
        "probability_Base",
        "probability_Detour",
        "probability_FailSafeHold",
        "recovery_open",
        "intervention_needed",
        "hard_negative",
        "loss_control",
        "old_router_advantage",
        "old_router_scalar_option",
        "old_router_scalar_correct",
        "old_router_pointwise_lcb",
        "old_router_candidate_option",
        "old_router_operational_option",
        "old_router_operational_correct",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(records):
            writer.writerow({
                "row_index": index,
                "fold_index": int(fold_index[index]),
                "held_out_source_state_sha256": row["source_state_sha256"],
                "decision_id": row["decision_id"],
                "placement_id": row["placement_id"],
                "record_type": row["record_type"],
                "condition": row["condition"],
                "preferred_option": row["preferred_option"],
                "direct_predicted_option": CLASSES[int(predictions[index])],
                "direct_correct": bool(
                    CLASSES[int(predictions[index])] == str(row["preferred_option"])
                ),
                "probability_Base": f"{probabilities[index, 0]:.17g}",
                "probability_Detour": f"{probabilities[index, 1]:.17g}",
                "probability_FailSafeHold": f"{probabilities[index, 2]:.17g}",
                "recovery_open": bool(row["recovery_open"]),
                "intervention_needed": bool(row["intervention_needed"]),
                "hard_negative": bool(row["hard_negative"]),
                "loss_control": bool(row["loss_control"]),
                "old_router_advantage": row["old_router_advantage"],
                "old_router_scalar_option": CLASSES[int(scalar_predictions[index])],
                "old_router_scalar_correct": bool(
                    CLASSES[int(scalar_predictions[index])]
                    == str(row["preferred_option"])
                ),
                "old_router_pointwise_lcb": row["old_router_pointwise_lcb"],
                "old_router_candidate_option": row["old_router_candidate_option"],
                "old_router_operational_option": CLASSES[int(old_predictions[index])],
                "old_router_operational_correct": bool(
                    CLASSES[int(old_predictions[index])] == str(row["preferred_option"])
                ),
            })


def _load_font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    names = (
        ("DejaVuSans-Bold.ttf", "Arial Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "Arial.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _write_confusion_png(path: Path, matrix: np.ndarray) -> None:
    """Render a dependency-light, publication-readable OOF confusion matrix."""

    from PIL import Image, ImageDraw

    width, height = 1180, 1010
    image = Image.new("RGB", (width, height), "#FAFAF8")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(38, bold=True)
    label_font = _load_font(27, bold=True)
    cell_font = _load_font(34, bold=True)
    detail_font = _load_font(22)
    note_font = _load_font(20)
    draw.text(
        (width / 2, 45),
        "P3.1 direct recovery-window head",
        fill="#17212B",
        font=title_font,
        anchor="ma",
    )
    draw.text(
        (width / 2, 95),
        "Leave-one-source_state_sha256-out predictions",
        fill="#4C5967",
        font=detail_font,
        anchor="ma",
    )

    left, top, cell = 260, 220, 210
    maximum = max(int(matrix.max()), 1)
    for column, name in enumerate(CLASSES):
        draw.text(
            (left + column * cell + cell / 2, top - 38),
            name,
            fill="#243444",
            font=label_font,
            anchor="ms",
        )
    for row, name in enumerate(CLASSES):
        draw.text(
            (left - 28, top + row * cell + cell / 2),
            name,
            fill="#243444",
            font=label_font,
            anchor="rm",
        )
        row_total = int(matrix[row].sum())
        for column in range(len(CLASSES)):
            value = int(matrix[row, column])
            strength = value / maximum
            if row == column:
                color = (
                    int(224 - 96 * strength),
                    int(241 - 48 * strength),
                    int(232 - 62 * strength),
                )
            else:
                color = (
                    int(247 - 50 * strength),
                    int(231 - 74 * strength),
                    int(228 - 72 * strength),
                )
            x0 = left + column * cell
            y0 = top + row * cell
            draw.rounded_rectangle(
                (x0 + 4, y0 + 4, x0 + cell - 4, y0 + cell - 4),
                radius=16,
                fill=color,
                outline="#FFFFFF",
                width=4,
            )
            draw.text(
                (x0 + cell / 2, y0 + cell / 2 - 16),
                str(value),
                fill="#17212B",
                font=cell_font,
                anchor="mm",
            )
            percentage = 0.0 if row_total == 0 else 100.0 * value / row_total
            draw.text(
                (x0 + cell / 2, y0 + cell / 2 + 34),
                f"{percentage:.1f}% of true class",
                fill="#37495A",
                font=detail_font,
                anchor="mm",
            )
    draw.text(
        (left + 1.5 * cell, top + 3 * cell + 54),
        "Predicted preferred option",
        fill="#243444",
        font=label_font,
        anchor="ma",
    )
    draw.text(
        (left - 28, top - 38),
        "TRUE",
        fill="#5E6B76",
        font=note_font,
        anchor="rs",
    )
    draw.text(
        (width - 35, height - 28),
        "Cells show count and row-normalized recall contribution.",
        fill="#5E6B76",
        font=note_font,
        anchor="rd",
    )
    image.save(path, format="PNG", optimize=True)


def _save_candidate_model(
    path: Path,
    model: Mapping[str, np.ndarray],
    fit_summary: Mapping[str, Any],
    *,
    input_path: Path,
    records: int,
    sources: int,
) -> None:
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        kind=np.asarray("direct_recovery_window_multinomial_linear_softmax"),
        status=np.asarray(CANDIDATE_STATUS),
        classes=np.asarray(CLASSES),
        input_feature_field=np.asarray(FEATURE_FIELD),
        input_dimension=np.asarray(FEATURE_DIMENSION, dtype=np.int64),
        feature_mean=np.asarray(model["feature_mean"], dtype=np.float64),
        feature_scale=np.asarray(model["feature_scale"], dtype=np.float64),
        head_weight=np.asarray(model["head_weight"], dtype=np.float64),
        head_bias=np.asarray(model["head_bias"], dtype=np.float64),
        l2=np.asarray(FIXED_L2, dtype=np.float64),
        source_weighting=np.asarray("equal total sample weight per source"),
        training_records=np.asarray(records, dtype=np.int64),
        training_sources=np.asarray(sources, dtype=np.int64),
        training_records_sha256=np.asarray(_sha256(input_path)),
        optimizer=np.asarray(str(fit_summary["optimizer"])),
        optimizer_converged=np.asarray(bool(fit_summary["converged"])),
        optimizer_iterations=np.asarray(int(fit_summary["iterations"]), dtype=np.int64),
    )


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _short_placement(value: str) -> str:
    return value.removeprefix("glass_recovery_heldout_")


def _dynamic_closeout_assessment(
    direct_metrics: Mapping[str, Any],
    ranking: Mapping[str, Mapping[str, Any]],
    targeted: Mapping[str, Any],
) -> dict[str, Any]:
    collapsed = [
        name
        for name in CLASSES
        if int(direct_metrics["class_metrics"][name]["predicted_count"]) == 0
    ]
    scalar = ranking["old_router_scalar_advantage"]
    direct = ranking["direct_recovery_window_head"]
    improvements = {
        "recovery_open_vs_hard_negative_auc": (
            float(direct["recovery_open_vs_hard_negative_auc"])
            - float(scalar["recovery_open_vs_hard_negative_auc"])
        ),
        "intervention_needed_vs_hard_negative_auc": (
            float(direct["intervention_needed_vs_hard_negative_auc"])
            - float(scalar["intervention_needed_vs_hard_negative_auc"])
        ),
    }
    hard_retention = targeted["hard_control_base_retention"][
        "direct_recovery_window_head"
    ]["retention"]
    targeted_recovery = [
        values["direct_recovery_window_head"]["recall"]
        for values in targeted["recovery_open_recall_by_placement"].values()
        if values["direct_recovery_window_head"]["support"] > 0
    ]
    targeted_hold = [
        values["direct_recovery_window_head"]["recall"]
        for values in targeted["hold_recall_by_placement"].values()
        if values["direct_recovery_window_head"]["support"] > 0
    ]
    suitable = bool(
        not collapsed
        and all(value > 0.0 for value in improvements.values())
        and hard_retention is not None
        and float(hard_retention) >= 0.9
        and targeted_recovery
        and all(value is not None and float(value) > 0.0 for value in targeted_recovery)
        and targeted_hold
        and all(value is not None and float(value) > 0.0 for value in targeted_hold)
    )
    return {
        "suitable_for_one_frozen_dynamic_closeout": suitable,
        "basis": (
            "descriptive development gate: no globally collapsed predicted class, "
            "both requested OOF ranking AUCs improve over the frozen scalar axis, "
            "at least 90% hard-control Base retention, and nonzero direct OOF recall "
            "on every pre-specified recovery-open and Hold trajectory"
        ),
        "collapsed_predicted_classes": collapsed,
        "auc_improvement_over_old_scalar": improvements,
        "direct_hard_control_base_retention": hard_retention,
        "caveat": (
            "LOSO development evidence only; the all-record candidate has not been "
            "tested in a causal dynamic rollout"
        ),
    }


def _write_report(result: Mapping[str, Any], path: Path) -> None:
    methods = result["methods"]
    ranking = result["ranking"]
    direct = methods["direct_recovery_window_head"]
    scalar = methods["old_router_scalar_advantage"]
    old = methods["old_router_operational_decision"]
    scalar_ranking = ranking["old_router_scalar_advantage"]
    old_ranking = ranking["old_router_operational_decision"]
    direct_ranking = ranking["direct_recovery_window_head"]
    assessment = result["dynamic_closeout_assessment"]
    lines = [
        "# P3.1 Direct Recovery-Window Routing Head",
        "",
        f"Status: **{CANDIDATE_STATUS}**.",
        "",
        "This development experiment fits one standardized multinomial linear softmax "
        "head to the frozen Router's 10 prediction outputs. There is no temporal encoder, "
        "no rollout, no hyperparameter sweep, and no change to the frozen Router's lambda, "
        "margin, or alpha.",
        "",
        "## Protocol",
        "",
        f"- Records: **{result['counts']['records']}** across **{result['counts']['unique_sources']}** "
        "unique `source_state_sha256` values.",
        f"- Model input: only `{FEATURE_FIELD}` ({FEATURE_DIMENSION} values).",
        f"- Model: standardized three-class linear softmax with fixed `L2={FIXED_L2:g}`; "
        "each source has equal total training weight.",
        "- Evaluation: strict leave-one-`source_state_sha256`-out. All rows from a held-out "
        "source share one fold; standardization and fitting are repeated using training "
        "sources only.",
        "- Decision: class argmax; no threshold or margin was selected on OOF results.",
        "- Ranking AUCs are ordinary record-level ROC AUCs. Recovery uses OOF `P(Detour)`; "
        "intervention uses OOF `1-P(Base)`.",
        "",
        "## Out-of-fold core results",
        "",
        "| Method | Source-macro accuracy | Macro-F1 | Base recall | Detour recall | Hold recall | Recovery-open vs hard-negative AUC | Intervention-needed vs hard-negative AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Old Router scalar advantage (raw sign choice) | "
        f"{_fmt(scalar['source_macro_accuracy'])} | "
        f"{_fmt(scalar['macro_f1_fixed_three_class'])} | "
        f"{_fmt(scalar['class_metrics']['Base']['recall'])} | "
        f"{_fmt(scalar['class_metrics']['Detour']['recall'])} | "
        f"{_fmt(scalar['class_metrics']['FailSafeHold']['recall'])} | "
        f"{_fmt(scalar_ranking['recovery_open_vs_hard_negative_auc'])} | "
        f"{_fmt(scalar_ranking['intervention_needed_vs_hard_negative_auc'])} |",
        f"| Old Router original operational decision | {_fmt(old['source_macro_accuracy'])} | "
        f"{_fmt(old['macro_f1_fixed_three_class'])} | "
        f"{_fmt(old['class_metrics']['Base']['recall'])} | "
        f"{_fmt(old['class_metrics']['Detour']['recall'])} | "
        f"{_fmt(old['class_metrics']['FailSafeHold']['recall'])} | "
        f"{_fmt(old_ranking['recovery_open_vs_hard_negative_auc'])}* | "
        f"{_fmt(old_ranking['intervention_needed_vs_hard_negative_auc'])}* |",
        f"| Direct recovery-window head (strict OOF) | **{_fmt(direct['source_macro_accuracy'])}** | "
        f"**{_fmt(direct['macro_f1_fixed_three_class'])}** | "
        f"{_fmt(direct['class_metrics']['Base']['recall'])} | "
        f"{_fmt(direct['class_metrics']['Detour']['recall'])} | "
        f"{_fmt(direct['class_metrics']['FailSafeHold']['recall'])} | "
        f"**{_fmt(direct_ranking['recovery_open_vs_hard_negative_auc'])}** | "
        f"**{_fmt(direct_ranking['intervention_needed_vs_hard_negative_auc'])}** |",
        "",
        "Scalar classification uses its natural untuned sign rule: choose the old "
        "non-Base candidate only when max non-Base advantage is strictly positive; its "
        "AUC columns use the continuous advantage. No threshold was fitted.",
        "",
        "\\*The old operational row has only a discrete choice, so these two AUCs are "
        "tie-heavy decision-level diagnostics, not continuous ranking curves.",
        "",
        "Macro-F1 is the unweighted mean over the fixed class set Base, Detour, and "
        "FailSafeHold. Source-macro accuracy first computes accuracy within each source "
        "and then weights all sources equally.",
        "",
        "![OOF confusion matrix](p3_1_direct_recovery_head_confusion_20260819.png)",
        "",
        "## Held-out source results",
        "",
        "| Fold | Held-out source | Placement | Type | N | Accuracy | Macro-F1 | Base / Detour / Hold recall | Predicted Base / Detour / Hold |",
        "|---:|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in result["per_source"]:
        source = str(row["held_out_source_state_sha256"])
        placements = ", ".join(_short_placement(value) for value in row["placements"])
        record_types = ", ".join(
            f"{name}:{count}" for name, count in row["record_types"].items()
        )
        recall = row["class_recall"]
        predicted = row["predicted_class_counts"]
        lines.append(
            f"| {row['fold_index']} | `{source[:12]}...` | {placements} | {record_types} | "
            f"{row['records']} | {_fmt(row['accuracy'])} | "
            f"{_fmt(row['macro_f1_fixed_three_class'])} | "
            f"{_fmt(recall['Base'])} / {_fmt(recall['Detour'])} / "
            f"{_fmt(recall['FailSafeHold'])} | {predicted['Base']} / "
            f"{predicted['Detour']} / {predicted['FailSafeHold']} |"
        )

    lines.extend([
        "",
        "Per-source macro-F1 also uses the fixed three-class set; a correctly classified "
        "single-class hard-control source therefore has macro-F1 `0.333`, not `1.000`.",
    ])

    targeted = result["targeted_diagnostics"]
    lines.extend([
        "",
        "## Pre-specified trajectory diagnostics",
        "",
        "| Diagnostic | Support | Old scalar sign | Old operational | Direct OOF |",
        "|---|---:|---:|---:|---:|",
    ])
    for placement, values in targeted["recovery_open_recall_by_placement"].items():
        old_value = values["old_router_operational_decision"]
        scalar_value = values["old_router_scalar_advantage"]
        direct_value = values["direct_recovery_window_head"]
        lines.append(
            f"| {_short_placement(placement)} recovery-open recall | "
            f"{direct_value['support']} | {_fmt(scalar_value['recall'])} | "
            f"{_fmt(old_value['recall'])} | "
            f"{_fmt(direct_value['recall'])} |"
        )
    for placement, values in targeted["hold_recall_by_placement"].items():
        old_value = values["old_router_operational_decision"]
        scalar_value = values["old_router_scalar_advantage"]
        direct_value = values["direct_recovery_window_head"]
        lines.append(
            f"| {_short_placement(placement)} Hold recall | {direct_value['support']} | "
            f"{_fmt(scalar_value['recall'])} | {_fmt(old_value['recall'])} | "
            f"{_fmt(direct_value['recall'])} |"
        )
    scalar_retention = targeted["hard_control_base_retention"][
        "old_router_scalar_advantage"
    ]
    old_retention = targeted["hard_control_base_retention"][
        "old_router_operational_decision"
    ]
    direct_retention = targeted["hard_control_base_retention"][
        "direct_recovery_window_head"
    ]
    lines.append(
        f"| Hard controls retained as Base | {direct_retention['support']} | "
        f"{_fmt(scalar_retention['retention'])} | "
        f"{_fmt(old_retention['retention'])} | {_fmt(direct_retention['retention'])} |"
    )

    improvement = assessment["auc_improvement_over_old_scalar"]
    collapsed = assessment["collapsed_predicted_classes"]
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Relative to the old scalar axis, the direct head changes recovery-open vs "
        f"hard-negative ordering by **{improvement['recovery_open_vs_hard_negative_auc']:+.3f} AUC** "
        "and intervention-needed vs hard-negative ordering by "
        f"**{improvement['intervention_needed_vs_hard_negative_auc']:+.3f} AUC**.",
        "",
        (
            "No preferred-option class collapses globally under strict OOF argmax."
            if not collapsed
            else "Globally collapsed OOF predicted class(es): **"
            + ", ".join(collapsed)
            + "**."
        ),
        "",
        "The gain is not uniform option recovery: direct OOF Detour recall is "
        f"`{direct['class_metrics']['Detour']['recall']:.3f}`, and it is lower than the "
        "old operational choice on each of 0004/0009/0010. The strongest evidence is the "
        "recovery-v-hard ordering plus hard-control Base retention; the main residual error "
        "is Detour/FailSafeHold confusion.",
        "",
        (
            "The development evidence is sufficient to enter **one frozen dynamic closeout** "
            "as a decisive online test, not as a deployment claim."
            if assessment["suitable_for_one_frozen_dynamic_closeout"]
            else "The development evidence is **not sufficient** to enter a frozen dynamic closeout."
        ),
        "This is a development judgment only: the saved all-record head has not been "
        "dynamically confirmed, and no OOF result is an online outcome.",
        "",
        "## Saved candidate",
        "",
        f"`{result['outputs']['candidate_model']['path']}` was fitted once on all "
        f"{result['counts']['records']} P3.0 records using the identical fixed model and "
        f"regularization. It is explicitly marked **{CANDIDATE_STATUS}**.",
        "",
        "The candidate consumes only the 10-D frozen prediction-output vector. Horizon, "
        "collision-relative timing, action index, trajectory/source identifiers, and true "
        "option outcomes are absent from its feature matrix.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    input_path: Path,
    *,
    output_report: Path,
    output_oof: Path,
    output_analysis: Path,
    output_confusion: Path,
    output_model: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    outputs = (
        output_report,
        output_oof,
        output_analysis,
        output_confusion,
        output_model,
    )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")

    records, features, labels, sources = load_records(input_path)
    probabilities, fold_index, folds = leave_one_source_out_predictions(
        features, labels, sources, l2=FIXED_L2
    )
    direct_predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    scalar_predictions = old_scalar_advantage_predictions(records)
    old_predictions = old_operational_predictions(records)
    methods = {
        "old_router_scalar_advantage": classification_metrics(
            labels, scalar_predictions, sources
        ),
        "old_router_operational_decision": classification_metrics(
            labels, old_predictions, sources
        ),
        "direct_recovery_window_head": classification_metrics(
            labels, direct_predictions, sources
        ),
    }
    ranking = _ranking_metrics(records, probabilities, old_predictions)
    targeted = targeted_diagnostics(
        records,
        labels,
        {
            "old_router_scalar_advantage": scalar_predictions,
            "old_router_operational_decision": old_predictions,
            "direct_recovery_window_head": direct_predictions,
        },
    )
    per_source = per_source_results(
        records, labels, direct_predictions, sources, folds
    )

    full_model, full_fit = fit_multinomial_head(
        features, labels, sources, l2=FIXED_L2
    )
    _write_oof_csv(
        output_oof,
        records,
        probabilities,
        direct_predictions,
        scalar_predictions,
        old_predictions,
        fold_index,
    )
    _write_confusion_png(
        output_confusion,
        np.asarray(methods["direct_recovery_window_head"]["confusion_matrix"]),
    )
    _save_candidate_model(
        output_model,
        full_model,
        full_fit,
        input_path=input_path,
        records=len(records),
        sources=len(set(sources.tolist())),
    )

    counts = {
        "records": len(records),
        "unique_sources": len(set(sources.tolist())),
        "dense_anchors": sum(row["record_type"] == "dense_glass_anchor" for row in records),
        "hard_controls": sum(bool(row["hard_negative"]) for row in records),
        "preferred_options": dict(sorted(Counter(
            str(row["preferred_option"]) for row in records
        ).items())),
        "recovery_open": sum(bool(row["recovery_open"]) for row in records),
        "intervention_needed": sum(bool(row["intervention_needed"]) for row in records),
    }
    assessment = _dynamic_closeout_assessment(
        methods["direct_recovery_window_head"], ranking, targeted
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "p3_1_direct_recovery_window_head_development_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": CANDIDATE_STATUS,
        "input": _artifact(input_path),
        "protocol": {
            "model_input_fields": [FEATURE_FIELD],
            "input_dimension": FEATURE_DIMENSION,
            "forbidden_model_inputs_not_used": list(FORBIDDEN_MODEL_INPUTS),
            "model": "standardized multinomial linear softmax/logistic head",
            "classes": list(CLASSES),
            "fixed_l2": FIXED_L2,
            "source_weighting": "equal total sample weight per source within each fit",
            "cross_validation": "leave-one-source_state_sha256-out",
            "fold_standardization": "fit on training sources only",
            "random_split": False,
            "hyperparameter_sweep": False,
            "temporal_encoder": False,
            "decision_rule": "argmax of three OOF probabilities",
            "old_operational_decision_rule": (
                "old_router_candidate_option iff old_router_pointwise_lcb > 0; else Base"
            ),
            "old_scalar_classification_rule": (
                "old_router_candidate_option iff old_router_advantage > 0; else Base"
            ),
            "auc": "ordinary record-level ROC AUC with half credit for ties",
        },
        "counts": counts,
        "folds": folds,
        "fold_integrity": {
            "all_rows_from_each_source_in_exactly_one_test_fold": True,
            "each_test_source_absent_from_its_training_sources": True,
            "oof_records_predicted_exactly_once": True,
        },
        "methods": methods,
        "ranking": ranking,
        "targeted_diagnostics": targeted,
        "per_source": per_source,
        "dynamic_closeout_assessment": assessment,
        "candidate_model": {
            "status": CANDIDATE_STATUS,
            "training_scope": "all P3.0 records",
            "fit": full_fit,
            "array_shapes": {
                key: list(np.asarray(value).shape) for key, value in full_model.items()
            },
        },
        "outputs": {
            "oof_predictions": _artifact(output_oof),
            "confusion_figure": _artifact(output_confusion),
            "candidate_model": _artifact(output_model),
            "report": {"path": str(output_report)},
            "analysis": {"path": str(output_analysis)},
        },
    }
    # The report needs the candidate path but cannot contain its own hash.
    _write_report(result, output_report)
    result["outputs"]["report"] = _artifact(output_report)
    output_analysis.write_text(
        json.dumps(_json_value(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/p3_recovery_window_records_20260819.jsonl"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("results/P3_1_DIRECT_RECOVERY_HEAD_DEV_20260819.md"),
    )
    parser.add_argument(
        "--output-oof",
        type=Path,
        default=Path("results/p3_1_direct_recovery_head_oof_predictions_20260819.csv"),
    )
    parser.add_argument(
        "--output-analysis",
        type=Path,
        default=Path("results/p3_1_direct_recovery_head_analysis_20260819.json"),
    )
    parser.add_argument(
        "--output-confusion",
        type=Path,
        default=Path("results/p3_1_direct_recovery_head_confusion_20260819.png"),
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("results/p3_1_direct_recovery_head_model.npz"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_experiment(
        args.input,
        output_report=args.output_report,
        output_oof=args.output_oof,
        output_analysis=args.output_analysis,
        output_confusion=args.output_confusion,
        output_model=args.output_model,
        overwrite=args.overwrite,
    )
    direct = result["methods"]["direct_recovery_window_head"]
    ranking = result["ranking"]["direct_recovery_window_head"]
    print(json.dumps({
        "status": result["status"],
        "records": result["counts"]["records"],
        "unique_sources": result["counts"]["unique_sources"],
        "source_macro_accuracy": direct["source_macro_accuracy"],
        "macro_f1": direct["macro_f1_fixed_three_class"],
        "class_recall": {
            name: direct["class_metrics"][name]["recall"] for name in CLASSES
        },
        "recovery_open_vs_hard_negative_auc": ranking[
            "recovery_open_vs_hard_negative_auc"
        ],
        "intervention_needed_vs_hard_negative_auc": ranking[
            "intervention_needed_vs_hard_negative_auc"
        ],
        "suitable_for_one_frozen_dynamic_closeout": result[
            "dynamic_closeout_assessment"
        ]["suitable_for_one_frozen_dynamic_closeout"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
