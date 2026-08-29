#!/usr/bin/env python3
"""Audit strict option support in the exposed 20-source full capture.

Phase 2.5A is deliberately descriptive and CPU-only.  It consumes the frozen
full-capture metadata plus the sealed phase-2 prediction archive, rejects any
fresh/test split before reading outcomes, preserves every utility tie as a
separate label, and emits a machine-readable support gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[2]
OPTIONS = ("base_continue", "detour_complete", "retreat_hold")
STRICT_LABELS = ("strict_base", "strict_detour", "strict_retreat")
ALL_LABELS = (
    "strict_base",
    "strict_detour",
    "strict_retreat",
    "detour_retreat_tie",
    "base_tie",
    "no_good_option",
)
LABEL_FOR_OPTION = dict(zip(range(3), STRICT_LABELS))


@dataclass(frozen=True)
class AuditData:
    decision_id: np.ndarray
    source: np.ndarray
    split: np.ndarray
    condition: np.ndarray
    horizon: np.ndarray
    placement_id: np.ndarray
    placement_key: np.ndarray
    matched_scan_index: np.ndarray
    outcomes: np.ndarray
    utility: np.ndarray
    risk_score: np.ndarray
    frozen_risk_choice: np.ndarray
    baseline_manifest: Mapping[str, Any]
    input_hashes: Mapping[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--short"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _native(value.item())
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("option-support config must be a mapping")
    if config.get("kind") != "iclr27_option_support_audit":
        raise ValueError("unexpected option-support config kind")
    utility = config.get("utility", {})
    if float(utility.get("lambda", -1)) != 1.0:
        raise ValueError("Phase 2.5A freezes lambda=1")
    if float(utility.get("eta", -1)) != 0.0:
        raise ValueError("Phase 2.5A freezes eta=0")
    if tuple(utility.get("option_order", ())) != OPTIONS:
        raise ValueError("Phase 2.5A requires the frozen three-option order")
    return config


def validate_allowed_splits(splits: Sequence[str], config: Mapping[str, Any]) -> None:
    """Reject outcome-bearing test rows before the outcome array is accessed."""

    policy = config["data_policy"]
    observed = set(map(str, splits))
    allowed = set(map(str, policy["allowed_splits"]))
    forbidden = set(map(str, policy["forbidden_splits"]))
    bad = observed - allowed
    if bad or observed & forbidden:
        names = sorted(bad | (observed & forbidden))
        raise ValueError(
            "option-support audit refuses outcome-bearing split(s): "
            + ", ".join(names)
        )
    missing = allowed - observed
    if missing:
        raise ValueError("exposed corpus is missing split(s): " + ", ".join(sorted(missing)))


def _verify_input_hashes(
    capture: Path,
    baseline: Path,
    baseline_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    declared_capture = baseline_manifest.get("capture_sha256", {})
    for name in config["inputs"]["required_capture_files"]:
        path = capture / str(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        expected = declared_capture.get(str(name))
        if expected is None or actual != expected:
            raise ValueError(f"capture hash mismatch for {name}")
        hashes[f"capture/{name}"] = actual
    declared_artifacts = baseline_manifest.get("artifact_sha256", {})
    for name in config["inputs"]["required_baseline_files"]:
        path = baseline / str(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if name != "manifest.json":
            expected = declared_artifacts.get(str(name))
            if expected is None or actual != expected:
                raise ValueError(f"baseline artifact hash mismatch for {name}")
        hashes[f"baseline/{name}"] = actual
    return hashes


def load_audit_data(
    capture: Path, baseline: Path, config: Mapping[str, Any]
) -> AuditData:
    baseline_manifest = _load_json(baseline / "manifest.json")
    if baseline_manifest.get("kind") != "iclr27_router_baseline_suite":
        raise ValueError("input manifest is not the frozen baseline suite")
    input_hashes = _verify_input_hashes(
        capture, baseline, baseline_manifest, config
    )

    prediction_path = baseline / "all_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as archive:
        required_metadata = (
            "method_names",
            "choices",
            "choice_probabilities",
            "decision_id",
            "source",
            "split",
            "condition",
            "horizon",
        )
        missing = [name for name in required_metadata if name not in archive]
        if missing:
            raise ValueError("prediction archive is missing: " + ", ".join(missing))
        # Split validation deliberately precedes access to ``outcomes``.
        splits = np.asarray(archive["split"], dtype=str)
        validate_allowed_splits(splits, config)
        if "outcomes" not in archive:
            raise ValueError("prediction archive is missing: outcomes")

        method_names = tuple(map(str, np.asarray(archive["method_names"]).tolist()))
        risk_method = str(config["inputs"]["risk_score_method"])
        if risk_method not in method_names:
            raise ValueError(f"risk score method is unavailable: {risk_method}")
        method_index = method_names.index(risk_method)
        choices = np.asarray(archive["choices"], dtype=np.int64)
        choice_probabilities = np.asarray(
            archive["choice_probabilities"], dtype=np.float64
        )
        decision_id = np.asarray(archive["decision_id"], dtype=str)
        source = np.asarray(archive["source"], dtype=str)
        condition = np.asarray(archive["condition"], dtype=str)
        horizon = np.asarray(archive["horizon"], dtype=np.int64)
        outcomes = np.asarray(archive["outcomes"], dtype=str)

    n = len(decision_id)
    if len(set(decision_id.tolist())) != n:
        raise ValueError("decision IDs must be unique")
    if choices.shape != (len(method_names), n):
        raise ValueError("choices must have shape method x decision")
    if choice_probabilities.shape != (len(method_names), n, len(OPTIONS)):
        raise ValueError("choice_probabilities must have shape method x decision x option")
    if outcomes.shape != (n, len(OPTIONS)):
        raise ValueError("outcomes must have shape decision x option")
    source_count = len(set(source.tolist()))
    expected_source_count = int(config["data_policy"]["expected_source_count"])
    if source_count != expected_source_count:
        raise ValueError(
            f"expected {expected_source_count} exposed sources, found {source_count}"
        )
    expected_split_counts = config["data_policy"][
        "expected_historical_split_source_counts"
    ]
    observed_split_counts = {
        split: len(set(source[splits == split].tolist()))
        for split in sorted(set(splits.tolist()))
    }
    if observed_split_counts != {
        str(split): int(count) for split, count in expected_split_counts.items()
    }:
        raise ValueError(
            "historical split source counts disagree with the frozen 5/7/8 corpus"
        )
    risk_probabilities = choice_probabilities[method_index]
    if not np.all(np.isfinite(risk_probabilities)):
        raise ValueError("frozen risk probabilities contain non-finite values")
    if np.any(risk_probabilities < -1e-12) or np.any(risk_probabilities > 1 + 1e-12):
        raise ValueError("frozen risk probabilities fall outside [0, 1]")
    np.testing.assert_allclose(risk_probabilities.sum(axis=1), 1.0, atol=1e-8)
    risk_score = 1.0 - risk_probabilities[:, 0]

    metadata = _load_json(capture / "decision_metadata.json")
    metadata = sorted(metadata, key=lambda row: int(row["feature_index"]))
    if [int(row["feature_index"]) for row in metadata] != list(range(n)):
        raise ValueError("capture feature indices are not contiguous")
    comparisons = {
        "decision_id": decision_id,
        "source_state_sha256": source,
        "split": splits,
        "condition": condition,
        "horizon_actions": horizon,
    }
    for field, expected in comparisons.items():
        observed = np.asarray([row[field] for row in metadata], dtype=expected.dtype)
        if not np.array_equal(observed, expected):
            raise ValueError(f"capture metadata disagrees with predictions: {field}")

    outcome_values = {
        str(key): float(value)
        for key, value in config["utility"]["outcome_values"].items()
    }
    unknown = sorted(set(outcomes.reshape(-1).tolist()) - set(outcome_values))
    if unknown:
        raise ValueError("unknown outcome(s): " + ", ".join(unknown))
    utility = np.vectorize(outcome_values.__getitem__, otypes=[float])(outcomes)

    declared_options = tuple(
        map(str, baseline_manifest.get("protocol", {}).get("options", ()))
    )
    if declared_options != OPTIONS:
        raise ValueError("baseline manifest option order does not match the audit")
    return AuditData(
        decision_id=decision_id,
        source=source,
        split=splits,
        condition=condition,
        horizon=horizon,
        placement_id=np.asarray([str(row["placement_id"]) for row in metadata]),
        placement_key=np.asarray([str(row["placement_key"]) for row in metadata]),
        matched_scan_index=np.asarray(
            [int(row["matched_scan_index"]) for row in metadata], dtype=np.int64
        ),
        outcomes=outcomes,
        utility=np.asarray(utility, dtype=np.float64),
        risk_score=np.asarray(risk_score, dtype=np.float64),
        frozen_risk_choice=choices[method_index],
        baseline_manifest=baseline_manifest,
        input_hashes=input_hashes,
    )


def strict_option_labels(
    utility: np.ndarray, *, catastrophe_utility: float = -1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return tie-preserving labels and unique-winner margins.

    ``no_good_option`` is intentionally narrow: all three options must have the
    catastrophe utility.  A safe noncompletion that uniquely dominates crashes
    remains strict support for its option.
    """

    utility = np.asarray(utility, dtype=np.float64)
    if utility.ndim != 2 or utility.shape[1] != len(OPTIONS):
        raise ValueError("utility must have shape N x 3")
    labels: list[str] = []
    margins = np.zeros(len(utility), dtype=np.float64)
    for index, row in enumerate(utility):
        if np.all(np.isclose(row, catastrophe_utility, atol=1e-12, rtol=0.0)):
            labels.append("no_good_option")
            continue
        maximum = float(np.max(row))
        winners = np.flatnonzero(np.isclose(row, maximum, atol=1e-12, rtol=0.0))
        if len(winners) == 1:
            winner = int(winners[0])
            labels.append(LABEL_FOR_OPTION[winner])
            margins[index] = maximum - float(np.max(np.delete(row, winner)))
        elif 0 in winners:
            labels.append("base_tie")
        elif set(winners.tolist()) == {1, 2}:
            labels.append("detour_retreat_tie")
        else:  # Defensive: three options exhaust all possible winner sets.
            raise AssertionError(f"unhandled winner set: {winners.tolist()}")
    return np.asarray(labels, dtype=str), margins


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(str(name))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _native(row.get(name)) for name in fieldnames})


def _source_weights(sources: np.ndarray) -> np.ndarray:
    sources = np.asarray(sources, dtype=str)
    counts = Counter(sources.tolist())
    weights = np.asarray([1.0 / counts[source] for source in sources], dtype=float)
    return weights / weights.sum()


def _source_macro_mean(values: np.ndarray, sources: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    sources = np.asarray(sources, dtype=str)
    return float(
        np.mean([np.mean(values[sources == source]) for source in sorted(set(sources))])
    )


def decision_rows(
    data: AuditData, labels: np.ndarray, margins: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(len(labels)):
        rows.append({
            "decision_id": data.decision_id[index],
            "source": data.source[index],
            "split": data.split[index],
            "placement_id": data.placement_id[index],
            "placement_key": data.placement_key[index],
            "condition": data.condition[index],
            "horizon": int(data.horizon[index]),
            "matched_scan_index": int(data.matched_scan_index[index]),
            "strict_label": labels[index],
            "strict_advantage": float(margins[index]),
            "base_outcome": data.outcomes[index, 0],
            "detour_outcome": data.outcomes[index, 1],
            "retreat_outcome": data.outcomes[index, 2],
            "base_utility": float(data.utility[index, 0]),
            "detour_utility": float(data.utility[index, 1]),
            "retreat_utility": float(data.utility[index, 2]),
            "oracle_utility": float(np.max(data.utility[index])),
            "base_catastrophe_risk": float(data.risk_score[index]),
        })
    return rows


def count_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    splits = sorted({str(row["split"]) for row in rows})
    conditions = sorted({str(row["condition"]) for row in rows})
    horizons = sorted({int(row["horizon"]) for row in rows}, reverse=True)
    result = []
    for split in splits:
        for condition in conditions:
            for horizon in horizons:
                for label in ALL_LABELS:
                    selected = [
                        row for row in rows
                        if row["split"] == split
                        and row["condition"] == condition
                        and int(row["horizon"]) == horizon
                        and row["strict_label"] == label
                    ]
                    result.append({
                        "split": split,
                        "condition": condition,
                        "horizon": horizon,
                        "strict_label": label,
                        "decision_count": len(selected),
                        "distinct_source_count": len({row["source"] for row in selected}),
                    })
    return result


def support_rows(
    rows: Sequence[Mapping[str, Any]], *, gamma: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    label_rows = []
    for label in ALL_LABELS:
        selected = [row for row in rows if row["strict_label"] == label]
        gamma_selected = [
            row for row in selected if float(row["strict_advantage"]) >= gamma
        ] if label in STRICT_LABELS else []
        label_rows.append({
            "strict_label": label,
            "decision_count": len(selected),
            "distinct_source_count": len({row["source"] for row in selected}),
            "support_advantage_gamma": gamma,
            "gamma_qualified_decision_count": len(gamma_selected),
            "source_support": len({row["source"] for row in gamma_selected}),
        })

    per_source = []
    for source in sorted({str(row["source"]) for row in rows}):
        selected = [row for row in rows if row["source"] == source]
        counts = Counter(str(row["strict_label"]) for row in selected)
        qualified = {
            label: any(
                row["strict_label"] == label
                and float(row["strict_advantage"]) >= gamma
                for row in selected
            )
            for label in STRICT_LABELS
        }
        detour_retreat_flip = qualified["strict_detour"] and qualified["strict_retreat"]
        base_intervention_flip = qualified["strict_base"] and (
            qualified["strict_detour"] or qualified["strict_retreat"]
        )
        per_source.append({
            "source": source,
            "historical_split": sorted({str(row["split"]) for row in selected})[0],
            "decision_count": len(selected),
            **{f"{label}_count": counts[label] for label in ALL_LABELS},
            **{f"supports_{label}": qualified[label] for label in STRICT_LABELS},
            "has_detour_retreat_strict_flip": detour_retreat_flip,
            "has_base_intervention_strict_flip": base_intervention_flip,
            "qualifies_gate_flip": detour_retreat_flip or base_intervention_flip,
        })
    return label_rows, per_source


def transition_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build strict transitions over timing and matched condition changes."""

    result: list[dict[str, Any]] = []
    timing_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    condition_groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        timing_groups[(
            str(row["source"]), str(row["placement_id"]), str(row["condition"])
        )].append(row)
        condition_groups[(
            str(row["source"]), str(row["placement_id"]), int(row["horizon"])
        )].append(row)

    for (source, placement, condition), group in sorted(timing_groups.items()):
        ordered = sorted(
            group,
            key=lambda row: (-int(row["horizon"]), int(row["matched_scan_index"])),
        )
        for earlier, later in zip(ordered, ordered[1:]):
            if earlier["strict_label"] not in STRICT_LABELS or later["strict_label"] not in STRICT_LABELS:
                continue
            result.append({
                "transition_axis": "timing_earlier_to_later",
                "source": source,
                "placement_id": placement,
                "context": condition,
                "from_decision_id": earlier["decision_id"],
                "to_decision_id": later["decision_id"],
                "from_horizon": int(earlier["horizon"]),
                "to_horizon": int(later["horizon"]),
                "from_label": earlier["strict_label"],
                "to_label": later["strict_label"],
                "label_changed": earlier["strict_label"] != later["strict_label"],
            })

    condition_order = {name: index for index, name in enumerate(("glass", "offpath", "noglass"))}
    for (source, placement, horizon), group in sorted(condition_groups.items()):
        ordered = sorted(
            group,
            key=lambda row: (condition_order.get(str(row["condition"]), 99), str(row["condition"])),
        )
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1:]:
                if left["strict_label"] not in STRICT_LABELS or right["strict_label"] not in STRICT_LABELS:
                    continue
                result.append({
                    "transition_axis": "matched_condition_pair",
                    "source": source,
                    "placement_id": placement,
                    "context": f"{left['condition']}->{right['condition']}",
                    "from_decision_id": left["decision_id"],
                    "to_decision_id": right["decision_id"],
                    "from_horizon": horizon,
                    "to_horizon": horizon,
                    "from_label": left["strict_label"],
                    "to_label": right["strict_label"],
                    "label_changed": left["strict_label"] != right["strict_label"],
                })
    return result


def transition_matrix_rows(
    transitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    axes = sorted({str(row["transition_axis"]) for row in transitions})
    for axis in axes:
        for from_label in STRICT_LABELS:
            for to_label in STRICT_LABELS:
                selected = [
                    row for row in transitions
                    if row["transition_axis"] == axis
                    and row["from_label"] == from_label
                    and row["to_label"] == to_label
                ]
                result.append({
                    "transition_axis": axis,
                    "from_label": from_label,
                    "to_label": to_label,
                    "transition_count": len(selected),
                    "distinct_source_count": len({row["source"] for row in selected}),
                })
    return result


def advantage_rows(
    rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail = []
    for row in rows:
        label = str(row["strict_label"])
        if label not in {"strict_detour", "strict_retreat"}:
            continue
        option = 1 if label == "strict_detour" else 2
        other = 2 if option == 1 else 1
        utilities = (
            float(row["base_utility"]),
            float(row["detour_utility"]),
            float(row["retreat_utility"]),
        )
        detail.append({
            "decision_id": row["decision_id"],
            "source": row["source"],
            "split": row["split"],
            "condition": row["condition"],
            "horizon": row["horizon"],
            "strict_label": label,
            "strict_advantage": row["strict_advantage"],
            "advantage_over_base": utilities[option] - utilities[0],
            "advantage_over_other_intervention": utilities[option] - utilities[other],
        })
    summary = []
    for label in ("strict_detour", "strict_retreat"):
        selected = [row for row in detail if row["strict_label"] == label]
        values = np.asarray([row["strict_advantage"] for row in selected], dtype=float)
        summary.append({
            "strict_label": label,
            "decision_count": len(selected),
            "distinct_source_count": len({row["source"] for row in selected}),
            "mean": float(np.mean(values)) if len(values) else None,
            "minimum": float(np.min(values)) if len(values) else None,
            "q25": float(np.quantile(values, 0.25)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "q75": float(np.quantile(values, 0.75)) if len(values) else None,
            "maximum": float(np.max(values)) if len(values) else None,
        })
    return detail, summary


def _recovery_metrics(
    name: str,
    selector_type: str,
    selected: np.ndarray,
    base: np.ndarray,
    oracle: np.ndarray,
    sources: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=float)
    available_sum = float(np.sum(oracle - base))
    gain_sum = float(np.sum(selected - base))
    base_macro = _source_macro_mean(base, sources)
    oracle_macro = _source_macro_mean(oracle, sources)
    selected_macro = _source_macro_mean(selected, sources)
    macro_available = oracle_macro - base_macro
    return {
        "method": name,
        "selector_type": selector_type,
        "decision_count": len(selected),
        "distinct_source_count": len(set(map(str, sources))),
        "mean_utility": float(np.mean(selected)),
        "source_macro_mean_utility": selected_macro,
        "value_gain_over_base_sum": gain_sum,
        "oracle_value_available_sum": available_sum,
        "oracle_value_recovered": gain_sum / available_sum if available_sum > 1e-12 else None,
        "source_macro_oracle_value_recovered": (
            (selected_macro - base_macro) / macro_available
            if macro_available > 1e-12 else None
        ),
        "fixed_mapping_gap_mean": float(np.mean(oracle - selected)),
        "source_macro_fixed_mapping_gap": oracle_macro - selected_macro,
    }


def fixed_option_recovery_rows(data: AuditData) -> list[dict[str, Any]]:
    utility = data.utility
    base = utility[:, 0]
    oracle = np.max(utility, axis=1)
    indices = np.arange(len(utility))
    selections = (
        ("Base", "fixed_base", base),
        ("AlwaysDetour", "always_fixed_nonbase", utility[:, 1]),
        ("AlwaysRetreat", "always_fixed_nonbase", utility[:, 2]),
        ("OracleGate->Detour", "realized_oracle_base_vs_fixed", np.maximum(base, utility[:, 1])),
        ("OracleGate->Retreat", "realized_oracle_base_vs_fixed", np.maximum(base, utility[:, 2])),
        (
            "FrozenRisk->BestFixed",
            "sealed_learned_risk_gate_to_calibration_selected_fixed_option",
            utility[indices, data.frozen_risk_choice],
        ),
        ("Oracle", "full_three_option_oracle", oracle),
    )
    return [
        _recovery_metrics(name, selector, selected, base, oracle, data.source)
        for name, selector, selected in selections
    ]


def _balanced_mode(
    labels: np.ndarray, sources: np.ndarray, class_order: Sequence[str]
) -> str:
    weights = _source_weights(sources)
    scores = {
        label: float(np.sum(weights[labels == label])) for label in class_order
    }
    return max(class_order, key=lambda label: (scores[label], -class_order.index(label)))


def diagnostic_prediction_rows(
    rows: Sequence[Mapping[str, Any]], feature_sets: Sequence[str]
) -> list[dict[str, Any]]:
    result = []
    tasks = (
        ("strict_three_way", STRICT_LABELS),
        ("strict_intervention_two_way", ("strict_detour", "strict_retreat")),
    )
    for target_task, class_order in tasks:
        eligible = [row for row in rows if row["strict_label"] in class_order]
        for feature_set in feature_sets:
            fields = str(feature_set).split("+")
            for held_out in sorted({str(row["source"]) for row in eligible}):
                train = [row for row in eligible if row["source"] != held_out]
                test = [row for row in eligible if row["source"] == held_out]
                train_labels = np.asarray([row["strict_label"] for row in train], dtype=str)
                train_sources = np.asarray([row["source"] for row in train], dtype=str)
                fallback = _balanced_mode(train_labels, train_sources, class_order)
                lookup: dict[tuple[str, ...], str] = {}
                keys = sorted({tuple(str(row[field]) for field in fields) for row in train})
                for key in keys:
                    selected = [
                        row for row in train
                        if tuple(str(row[field]) for field in fields) == key
                    ]
                    lookup[key] = _balanced_mode(
                        np.asarray([row["strict_label"] for row in selected], dtype=str),
                        np.asarray([row["source"] for row in selected], dtype=str),
                        class_order,
                    )
                for row in test:
                    key = tuple(str(row[field]) for field in fields)
                    prediction = lookup.get(key, fallback)
                    result.append({
                        "diagnostic": feature_set,
                        "target_task": target_task,
                        "decision_id": row["decision_id"],
                        "source": held_out,
                        "feature_key": "|".join(key),
                        "target": row["strict_label"],
                        "prediction": prediction,
                        "correct": prediction == row["strict_label"],
                        "used_global_fallback": key not in lookup,
                    })
    return result


def _classification_metrics(
    rows: Sequence[Mapping[str, Any]], classes: Sequence[str]
) -> dict[str, Any]:
    target = np.asarray([row["target"] for row in rows], dtype=str)
    prediction = np.asarray([row["prediction"] for row in rows], dtype=str)
    sources = np.asarray([row["source"] for row in rows], dtype=str)
    recalls = []
    f1s = []
    for label in classes:
        true_positive = int(np.sum((target == label) & (prediction == label)))
        false_negative = int(np.sum((target == label) & (prediction != label)))
        false_positive = int(np.sum((target != label) & (prediction == label)))
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    correct = target == prediction
    return {
        "decision_count": len(rows),
        "distinct_source_count": len(set(sources.tolist())),
        "accuracy": float(np.mean(correct)) if len(rows) else None,
        "source_macro_accuracy": _source_macro_mean(correct.astype(float), sources) if len(rows) else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "fallback_count": int(sum(bool(row["used_global_fallback"]) for row in rows)),
    }


def diagnostic_summary_rows(
    predictions: Sequence[Mapping[str, Any]], feature_sets: Sequence[str]
) -> list[dict[str, Any]]:
    result = []
    for feature_set in feature_sets:
        for target_task, classes in (
            ("strict_three_way", STRICT_LABELS),
            ("strict_intervention_two_way", ("strict_detour", "strict_retreat")),
        ):
            selected = [
                row for row in predictions
                if row["diagnostic"] == feature_set
                and row["target_task"] == target_task
            ]
            result.append({
                "diagnostic": feature_set,
                "target_task": target_task,
                "evaluation": "leave_one_source_out",
                **_classification_metrics(selected, classes),
            })
    return result


def _entropy(probabilities: Iterable[float]) -> float:
    values = np.asarray(list(probabilities), dtype=float)
    values = values[values > 0]
    return float(-np.sum(values * np.log2(values))) if len(values) else 0.0


def risk_entropy_rows(
    rows: Sequence[Mapping[str, Any]], edges: Sequence[float]
) -> list[dict[str, Any]]:
    strict = [row for row in rows if row["strict_label"] in STRICT_LABELS]
    result = []
    for bin_index, (low, high) in enumerate(zip(edges, edges[1:])):
        selected = [
            row for row in strict
            if float(low) <= float(row["base_catastrophe_risk"]) < float(high)
        ]
        labels = np.asarray([row["strict_label"] for row in selected], dtype=str)
        sources = np.asarray([row["source"] for row in selected], dtype=str)
        if len(selected):
            weights = _source_weights(sources)
            balanced = {
                label: float(np.sum(weights[labels == label])) for label in STRICT_LABELS
            }
        else:
            balanced = {label: 0.0 for label in STRICT_LABELS}
        counts = {label: int(np.sum(labels == label)) for label in STRICT_LABELS}
        micro = {
            label: counts[label] / len(selected) if selected else 0.0
            for label in STRICT_LABELS
        }
        intervention_n = counts["strict_detour"] + counts["strict_retreat"]
        intervention_probabilities = (
            [
                counts["strict_detour"] / intervention_n,
                counts["strict_retreat"] / intervention_n,
            ] if intervention_n else []
        )
        result.append({
            "risk_bin_index": bin_index,
            "risk_lower_inclusive": float(low),
            "risk_upper_exclusive": float(high),
            "decision_count": len(selected),
            "distinct_source_count": len(set(sources.tolist())),
            **{f"{label}_count": counts[label] for label in STRICT_LABELS},
            **{f"{label}_proportion": micro[label] for label in STRICT_LABELS},
            **{f"source_balanced_{label}_proportion": balanced[label] for label in STRICT_LABELS},
            "strict_option_entropy_bits": _entropy(micro.values()),
            "source_balanced_strict_option_entropy_bits": _entropy(balanced.values()),
            "strict_intervention_decision_count": intervention_n,
            "detour_retreat_entropy_bits": _entropy(intervention_probabilities),
        })
    return result


def matched_risk_pair_rows(
    rows: Sequence[Mapping[str, Any]], *, maximum_difference: float
) -> list[dict[str, Any]]:
    detour = [row for row in rows if row["strict_label"] == "strict_detour"]
    retreat = [row for row in rows if row["strict_label"] == "strict_retreat"]
    if not detour or not retreat:
        return []
    n_detour, n_retreat = len(detour), len(retreat)
    size = n_detour + n_retreat
    forbidden = 1e6
    unmatched = 1.0
    cost = np.full((size, size), forbidden, dtype=float)
    for left, detour_row in enumerate(detour):
        for right, retreat_row in enumerate(retreat):
            difference = abs(
                float(detour_row["base_catastrophe_risk"])
                - float(retreat_row["base_catastrophe_risk"])
            )
            if difference <= maximum_difference + 1e-12:
                cost[left, right] = difference + 1e-12 * (left * n_retreat + right)
        cost[left, n_retreat + left] = unmatched
    for right in range(n_retreat):
        cost[n_detour + right, right] = unmatched
    cost[n_detour:, n_retreat:] = 0.0
    assigned_rows, assigned_columns = linear_sum_assignment(cost)
    result = []
    for left, right in zip(assigned_rows, assigned_columns):
        if left >= n_detour or right >= n_retreat or cost[left, right] >= unmatched:
            continue
        detour_row, retreat_row = detour[left], retreat[right]
        difference = abs(
            float(detour_row["base_catastrophe_risk"])
            - float(retreat_row["base_catastrophe_risk"])
        )
        result.append({
            "pair_id": "",
            "absolute_risk_difference": difference,
            "detour_decision_id": detour_row["decision_id"],
            "detour_source": detour_row["source"],
            "detour_split": detour_row["split"],
            "detour_condition": detour_row["condition"],
            "detour_horizon": detour_row["horizon"],
            "detour_base_risk": detour_row["base_catastrophe_risk"],
            "retreat_decision_id": retreat_row["decision_id"],
            "retreat_source": retreat_row["source"],
            "retreat_split": retreat_row["split"],
            "retreat_condition": retreat_row["condition"],
            "retreat_horizon": retreat_row["horizon"],
            "retreat_base_risk": retreat_row["base_catastrophe_risk"],
            "same_source": detour_row["source"] == retreat_row["source"],
            "same_condition": detour_row["condition"] == retreat_row["condition"],
            "same_horizon": detour_row["horizon"] == retreat_row["horizon"],
        })
    result.sort(key=lambda row: (
        float(row["absolute_risk_difference"]),
        str(row["detour_decision_id"]),
        str(row["retreat_decision_id"]),
    ))
    for index, row in enumerate(result, start=1):
        row["pair_id"] = f"risk_pair_{index:03d}"
    return result


def decide_gate(
    *,
    strict_base_support: int,
    strict_detour_support: int,
    strict_retreat_support: int,
    within_source_flip_sources: int,
    best_fixed_recovery: float,
    gate_config: Mapping[str, Any],
) -> dict[str, Any]:
    passed = gate_config["pass_support"]
    supplement = gate_config["supplement_support"]
    stopped = gate_config["stop_support"]
    actual = {
        "strict_base_source_support": int(strict_base_support),
        "strict_detour_source_support": int(strict_detour_support),
        "strict_retreat_source_support": int(strict_retreat_support),
        "within_source_flip_sources": int(within_source_flip_sources),
        "best_fixed_nonbase_oracle_value_recovered": float(best_fixed_recovery),
    }
    criteria = {
        "base_support": strict_base_support >= int(passed["strict_base_source_support_min"]),
        "detour_support": strict_detour_support >= int(passed["strict_detour_source_support_min"]),
        "retreat_support": strict_retreat_support >= int(passed["strict_retreat_source_support_min"]),
        "within_source_flips": within_source_flip_sources >= int(passed["within_source_flip_sources_min"]),
        "fixed_option_ambiguity": best_fixed_recovery <= float(passed["best_fixed_nonbase_oracle_value_recovered_max"]),
    }
    stop_reasons = []
    if strict_retreat_support < int(stopped["strict_retreat_source_support_below"]):
        stop_reasons.append("strict Retreat source support is below the STOP threshold")
    if best_fixed_recovery > float(stopped["best_fixed_nonbase_oracle_value_recovered_above"]):
        stop_reasons.append("best fixed non-Base option exceeds the STOP recovery threshold")
    if stop_reasons:
        decision = "STOP-SUPPORT"
    elif all(criteria.values()):
        decision = "PASS-SUPPORT"
    elif (
        int(supplement["strict_retreat_source_support_min"])
        <= strict_retreat_support
        <= int(supplement["strict_retreat_source_support_max"])
    ):
        decision = "SUPPLEMENT-SUPPORT"
    else:
        decision = "INCONCLUSIVE"
    return {
        "schema_version": 1,
        "phase": "2.5A",
        "decision": decision,
        "actual": actual,
        "pass_criteria": criteria,
        "stop_reasons": stop_reasons,
        "thresholds": _native(gate_config),
        "authorization": {
            "phase_2_5b_pooled_crossfit": decision == "PASS-SUPPORT",
            "training_only_source_supplement": decision == "SUPPLEMENT-SUPPORT",
            "screen_a_mechanical_authoring": False,
            "new_confirmatory_rollout": False,
        },
        "interpretation": (
            "Only PASS-SUPPORT authorizes Phase 2.5B. Screen A remains blocked "
            "until Phase 2.5B returns GO-SIGNAL or GO-VALUE-ONLY."
        ),
    }


def run_audit(
    *,
    config_path: Path,
    capture: Path,
    baseline: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = _load_config(config_path)
    data = load_audit_data(capture, baseline, config)
    output_dir.mkdir(parents=True, exist_ok=False)

    catastrophe_utility = float(config["utility"]["outcome_values"]["catastrophe"])
    labels, margins = strict_option_labels(
        data.utility, catastrophe_utility=catastrophe_utility
    )
    decisions = decision_rows(data, labels, margins)
    gamma = float(config["strict_labels"]["support_advantage_gamma"])
    labels_table, per_source = support_rows(decisions, gamma=gamma)
    transitions = transition_rows(decisions)
    transition_matrix = transition_matrix_rows(transitions)
    advantage_detail, advantage_summary = advantage_rows(decisions)
    recovery = fixed_option_recovery_rows(data)
    diagnostic_features = list(config["diagnostics"]["feature_sets"])
    diagnostic_predictions = diagnostic_prediction_rows(decisions, diagnostic_features)
    diagnostics = diagnostic_summary_rows(diagnostic_predictions, diagnostic_features)
    entropy = risk_entropy_rows(
        decisions, list(map(float, config["diagnostics"]["risk_bins"]["edges"]))
    )
    pairs = matched_risk_pair_rows(
        decisions,
        maximum_difference=float(
            config["diagnostics"]["matched_pairs"]["maximum_absolute_risk_difference"]
        ),
    )

    outputs: dict[str, Sequence[Mapping[str, Any]]] = {
        "decision_labels.csv": decisions,
        "decision_counts.csv": count_rows(decisions),
        "label_support.csv": labels_table,
        "per_source_support.csv": per_source,
        "within_source_transitions.csv": transitions,
        "transition_matrix.csv": transition_matrix,
        "intervention_advantages.csv": advantage_detail,
        "advantage_summary.csv": advantage_summary,
        "fixed_option_recovery.csv": recovery,
        "diagnostics.csv": diagnostics,
        "diagnostic_predictions.csv": diagnostic_predictions,
        "risk_entropy_bins.csv": entropy,
        "matched_risk_pairs.csv": pairs,
    }
    for name, table in outputs.items():
        _write_csv(output_dir / name, table)

    support_by_label = {
        row["strict_label"]: int(row["source_support"]) for row in labels_table
    }
    flip_sources = sum(bool(row["qualifies_gate_flip"]) for row in per_source)
    fixed_candidates = [
        row for row in recovery
        if row["method"] in {"OracleGate->Detour", "OracleGate->Retreat"}
    ]
    best_fixed = max(
        fixed_candidates, key=lambda row: float(row["oracle_value_recovered"])
    )
    gate = decide_gate(
        strict_base_support=support_by_label["strict_base"],
        strict_detour_support=support_by_label["strict_detour"],
        strict_retreat_support=support_by_label["strict_retreat"],
        within_source_flip_sources=flip_sources,
        best_fixed_recovery=float(best_fixed["oracle_value_recovered"]),
        gate_config=config["gate"],
    )
    gate["best_fixed_nonbase_method"] = best_fixed["method"]
    gate["sensitivity"] = {
        "source_macro_best_fixed_nonbase_oracle_value_recovered": float(
            best_fixed["source_macro_oracle_value_recovered"]
        ),
        "gate_uses_source_macro_sensitivity": False,
        "reason": (
            "The frozen gate retains the repository's existing decision-level "
            "Oracle value recovered formula; source-macro recovery is reported, "
            "not substituted post hoc."
        ),
    }
    (output_dir / "gate_decision.json").write_text(
        json.dumps(_native(gate), indent=2, sort_keys=True) + "\n"
    )

    config_snapshot = output_dir / "option_support_audit.yaml"
    config_snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
    artifact_hashes = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "kind": "iclr27_option_support_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_worktree_dirty": _git_dirty(),
        "evidence_level": "exposed-development-only",
        "new_rollouts": False,
        "fresh_test_outcomes_loaded": False,
        "capture_root": str(capture.resolve()),
        "baseline_audit_root": str(baseline.resolve()),
        "baseline_protocol_commit": data.baseline_manifest.get("git_commit"),
        "input_sha256": dict(data.input_hashes),
        "config_source": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "config_snapshot": config_snapshot.name,
        "decision_count": len(decisions),
        "source_count": len(set(data.source.tolist())),
        "historical_split_source_counts": {
            split: len(set(data.source[data.split == split].tolist()))
            for split in sorted(set(data.split.tolist()))
        },
        "strict_label_definition": {
            "labels": list(ALL_LABELS),
            "no_good_option": "all three utilities equal catastrophe utility",
            "ties_are_never_assigned_to_a_strict_class": True,
            "support_advantage_gamma": gamma,
        },
        "risk_score": {
            "method": config["inputs"]["risk_score_method"],
            "definition": "1 - choice_probability[base_continue]",
            "target": "Base-option catastrophe",
        },
        "gate_decision": gate["decision"],
        "artifact_sha256": artifact_hashes,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_native(manifest), indent=2, sort_keys=True) + "\n"
    )
    return {"manifest": manifest, "gate": gate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/iclr27/option_support_audit.yaml",
    )
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--baseline-audit", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = _load_config(args.config.resolve())
    capture = (
        args.capture.resolve()
        if args.capture else (ROOT / config["inputs"]["capture_path"]).resolve()
    )
    baseline = (
        args.baseline_audit.resolve()
        if args.baseline_audit
        else (ROOT / config["inputs"]["baseline_audit_path"]).resolve()
    )
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (
            ROOT / "results/iclr27"
            / f"option_support_audit_{_git_commit()[:12]}_{stamp}"
        )
    result = run_audit(
        config_path=args.config.resolve(),
        capture=capture,
        baseline=baseline,
        output_dir=output_dir,
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "gate_decision": result["gate"]["decision"],
        "actual": result["gate"]["actual"],
        "authorization": result["gate"]["authorization"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
