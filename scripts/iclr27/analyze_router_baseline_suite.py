#!/usr/bin/env python3
"""Analyze the ICLR'27 router-baseline suite on development sources only.

The trainer deliberately exports one rectangular NPZ so every method is scored
against exactly the same potential outcomes.  This script is intentionally
NumPy-only: it writes paper-facing flat tables, grouped paired inference, and a
machine-readable GO/NO-GO audit into the trainer output directory.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CANONICAL_OUTCOMES = ("task_success", "catastrophe", "safe_noncompletion")
DEFAULT_OPTIONS = ("Base", "Detour", "Retreat")
INFERENCE_METRICS = (
    "mean_utility",
    "task_success_rate",
    "catastrophe_rate",
    "intervention_rate",
    "harmful_intervention_rate",
    "missed_beneficial_intervention_rate",
    "mean_oracle_regret",
)
HIGHER_IS_BETTER = {
    "mean_utility": True,
    "task_success_rate": True,
    "catastrophe_rate": False,
    "intervention_rate": False,
    "harmful_intervention_rate": False,
    "missed_beneficial_intervention_rate": False,
    "mean_oracle_regret": False,
    "oracle_value_recovered": True,
}


@dataclass(frozen=True)
class SuiteData:
    method_names: tuple[str, ...]
    option_names: tuple[str, ...]
    outcome_names: tuple[str, ...]
    choices: np.ndarray
    predicted_values: np.ndarray
    outcome_probabilities: np.ndarray
    choice_probabilities: np.ndarray
    decision_id: np.ndarray
    source: np.ndarray
    split: np.ndarray
    condition: np.ndarray
    horizon: np.ndarray
    outcomes: np.ndarray
    parameter_count: np.ndarray
    latency_us_per_decision: np.ndarray
    outcome_utility: np.ndarray
    option_cost: np.ndarray
    base_index: int


def _native(value: Any) -> Any:
    """Convert NumPy values into strict-JSON-compatible Python values."""
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


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _slug(value: Any) -> str:
    return _text(value).strip().lower().replace("-", "_").replace(" ", "_")


def _canonical_outcome(value: Any, declared: Sequence[str]) -> str:
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        index = int(value)
        if 0 <= index < len(declared):
            value = declared[index]
    elif isinstance(value, (np.floating, float)) and float(value).is_integer():
        index = int(value)
        if 0 <= index < len(declared):
            value = declared[index]
    key = _slug(value)
    aliases = {
        "success": "task_success",
        "task_success": "task_success",
        "completed": "task_success",
        "catastrophe": "catastrophe",
        "crash": "catastrophe",
        "collision": "catastrophe",
        "safe_noncompletion": "safe_noncompletion",
        "safe_non_completion": "safe_noncompletion",
        "noncompletion": "safe_noncompletion",
        "safe_abort": "safe_noncompletion",
    }
    if key not in aliases:
        raise ValueError(f"unknown outcome label: {value!r}")
    return aliases[key]


def _mapping_at(root: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any]:
    current: Any = root
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return {}
        current = current[key]
    return current if isinstance(current, Mapping) else {}


def _first(root: Mapping[str, Any], paths: Iterable[Sequence[str]], default: Any) -> Any:
    for path in paths:
        current: Any = root
        found = True
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                found = False
                break
            current = current[key]
        if found:
            return current
    return default


def _declared_names(manifest: Mapping[str, Any], key: str, default: Sequence[str]) -> tuple[str, ...]:
    candidates = [
        manifest.get(key),
        manifest.get(f"{key[:-1]}_names") if key.endswith("s") else None,
        _mapping_at(manifest, ("protocol",)).get(key),
    ]
    value = next((item for item in candidates if item is not None), None)
    if isinstance(value, Mapping):
        if "names" in value:
            value = value["names"]
        else:
            try:
                value = [name for _, name in sorted(value.items(), key=lambda item: int(item[0]))]
            except (TypeError, ValueError):
                value = list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        names = []
        for item in value:
            if isinstance(item, Mapping):
                names.append(_text(item.get("name", item.get("label", len(names)))))
            else:
                names.append(_text(item))
        if names:
            return tuple(names)
    return tuple(default)


def _primary_lambda(manifest: Mapping[str, Any]) -> float:
    raw = _first(
        manifest,
        (("primary_lambda",), ("protocol", "primary_lambda"), ("primary", "lambda")),
        5.0,
    )
    if isinstance(raw, Mapping):
        raw = raw.get("value", raw.get("catastrophe_cost", 5.0))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 5.0
    return value if math.isfinite(value) and value >= 0 else 5.0


def _utility_spec(
    manifest: Mapping[str, Any], option_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    lam = _primary_lambda(manifest)
    protocol = _mapping_at(manifest, ("protocol",))
    spec = manifest.get("utility", {})
    if not isinstance(spec, Mapping):
        spec = {}
    values = spec.get("outcome_values", spec.get("outcomes", spec))
    if not isinstance(values, Mapping):
        values = {}
    utility = {
        "task_success": float(values.get("task_success", values.get("success", 1.0))),
        "catastrophe": float(values.get("catastrophe", values.get("crash", -lam))),
        "safe_noncompletion": float(
            values.get(
                "safe_noncompletion",
                values.get(
                    "safe_non_completion",
                    -float(protocol.get("noncompletion_cost", 0.0)),
                ),
            )
        ),
    }
    costs: Any = spec.get(
        "option_costs",
        spec.get(
            "intervention_costs",
            manifest.get(
                "option_costs",
                manifest.get("intervention_costs", protocol.get("option_costs", {})),
            ),
        ),
    )
    option_cost = np.zeros(len(option_names), dtype=np.float64)
    if isinstance(costs, Mapping):
        for index, name in enumerate(option_names):
            for candidate in (name, _slug(name), str(index), index):
                if candidate in costs:
                    option_cost[index] = float(costs[candidate])
                    break
    elif isinstance(costs, Sequence) and not isinstance(costs, (str, bytes)):
        for index, value in enumerate(costs[: len(option_names)]):
            option_cost[index] = float(value)
    raw_options = manifest.get("options")
    if isinstance(raw_options, Sequence) and not isinstance(raw_options, (str, bytes)):
        for index, item in enumerate(raw_options[: len(option_names)]):
            if isinstance(item, Mapping):
                option_cost[index] = float(
                    item.get("cost", item.get("intervention_cost", option_cost[index]))
                )
    return (
        np.asarray([utility[name] for name in CANONICAL_OUTCOMES], dtype=np.float64),
        option_cost,
    )


def _optional_array(
    archive: Mapping[str, np.ndarray], key: str, shape: tuple[int, ...]
) -> np.ndarray:
    if key not in archive:
        return np.full(shape, np.nan, dtype=np.float64)
    value = np.asarray(archive[key], dtype=np.float64)
    if value.shape != shape:
        # A missing calibration head is represented as blank, not a fatal suite error.
        return np.full(shape, np.nan, dtype=np.float64)
    return value


def load_suite(input_dir: Path, manifest: Mapping[str, Any]) -> SuiteData:
    path = input_dir / "all_predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        required = (
            "method_names", "choices", "decision_id", "source", "split",
            "condition", "horizon", "outcomes", "parameter_count",
            "latency_us_per_decision",
        )
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"all_predictions.npz is missing: {', '.join(missing)}")
        method_names = tuple(_text(item) for item in np.asarray(archive["method_names"]).tolist())
        if not method_names or len(set(method_names)) != len(method_names):
            raise ValueError("method_names must be non-empty and unique")
        choices = np.asarray(archive["choices"])
        decision_id = np.asarray([_text(item) for item in np.asarray(archive["decision_id"]).tolist()])
        n_methods, n_decisions = len(method_names), len(decision_id)
        option_names = _declared_names(manifest, "options", DEFAULT_OPTIONS)
        if len(option_names) != 3:
            raise ValueError("the baseline suite must declare exactly three options")
        declared_outcomes = _declared_names(manifest, "outcomes", CANONICAL_OUTCOMES)
        if len(declared_outcomes) != 3:
            declared_outcomes = CANONICAL_OUTCOMES
        expected_choice_shape = (n_methods, n_decisions)
        if choices.shape != expected_choice_shape:
            raise ValueError(
                f"choices has shape {choices.shape}, expected {expected_choice_shape}"
            )
        if not np.all(np.isfinite(choices.astype(np.float64))):
            raise ValueError("choices must be finite")
        integer_choices = choices.astype(np.int64)
        if not np.array_equal(integer_choices, choices) or np.any(integer_choices < 0) or np.any(integer_choices >= 3):
            raise ValueError("choices must contain option indices 0, 1, or 2")
        outcomes_raw = np.asarray(archive["outcomes"])
        if outcomes_raw.shape != (n_decisions, 3):
            raise ValueError(f"outcomes has shape {outcomes_raw.shape}, expected {(n_decisions, 3)}")
        outcomes = np.asarray(
            [[_canonical_outcome(value, declared_outcomes) for value in row] for row in outcomes_raw],
            dtype="U32",
        )
        vectors: dict[str, np.ndarray] = {}
        for key in ("source", "split", "condition"):
            vector = np.asarray([_text(item) for item in np.asarray(archive[key]).tolist()])
            if vector.shape != (n_decisions,):
                raise ValueError(f"{key} must have shape {(n_decisions,)}")
            vectors[key] = vector
        horizon = np.asarray(archive["horizon"])
        if horizon.shape != (n_decisions,):
            raise ValueError(f"horizon must have shape {(n_decisions,)}")
        parameter_count = np.asarray(archive["parameter_count"], dtype=np.float64)
        latency = np.asarray(archive["latency_us_per_decision"], dtype=np.float64)
        if parameter_count.shape != (n_methods,) or latency.shape != (n_methods,):
            raise ValueError("parameter_count and latency_us_per_decision must have shape (M,)")
        predicted_values = _optional_array(archive, "predicted_values", (n_methods, n_decisions, 3))
        outcome_probabilities = _optional_array(
            archive, "outcome_probabilities", (n_methods, n_decisions, 3, 3)
        )
        choice_probabilities = _optional_array(
            archive, "choice_probabilities", (n_methods, n_decisions, 3)
        )

    outcome_utility, option_cost = _utility_spec(manifest, option_names)
    base_candidates = [index for index, name in enumerate(option_names) if _slug(name) in {"base", "base_continue", "continue"}]
    base_index = base_candidates[0] if base_candidates else 0
    return SuiteData(
        method_names=method_names,
        option_names=option_names,
        outcome_names=tuple(_canonical_outcome(name, declared_outcomes) for name in declared_outcomes),
        choices=integer_choices,
        predicted_values=predicted_values,
        outcome_probabilities=outcome_probabilities,
        choice_probabilities=choice_probabilities,
        decision_id=decision_id,
        source=vectors["source"],
        split=vectors["split"],
        condition=vectors["condition"],
        horizon=horizon,
        outcomes=outcomes,
        parameter_count=parameter_count,
        latency_us_per_decision=latency,
        outcome_utility=outcome_utility,
        option_cost=option_cost,
        base_index=base_index,
    )


def _utility_matrix(data: SuiteData, indices: np.ndarray) -> np.ndarray:
    lookup = {name: data.outcome_utility[index] for index, name in enumerate(CANONICAL_OUTCOMES)}
    result = np.asarray(
        [[lookup[_text(value)] for value in row] for row in data.outcomes[indices]],
        dtype=np.float64,
    )
    # Option costs are the preregistered tie-break ordering, not an unreported
    # additive utility penalty.  Switching penalties are already frozen into
    # each method's operating point by the trainer.
    return result


def _oracle_choice(utilities: np.ndarray, base_index: int, option_cost: np.ndarray) -> np.ndarray:
    # np.argmax supplies fixed option order.  The explicit order below adds the
    # preregistered Base-first, then lower-cost tie break.
    order = [base_index] + sorted(
        (index for index in range(utilities.shape[1]) if index != base_index),
        key=lambda index: (float(option_cost[index]), index),
    )
    result = np.empty(len(utilities), dtype=np.int64)
    for row_index, row in enumerate(utilities):
        best = float(np.max(row))
        result[row_index] = next(index for index in order if abs(float(row[index]) - best) <= 1e-12)
    return result


def metric_values(data: SuiteData, method_index: int, indices: np.ndarray) -> dict[str, Any]:
    choices = data.choices[method_index, indices]
    outcomes = data.outcomes[indices]
    utilities = _utility_matrix(data, indices)
    rows = np.arange(len(indices))
    selected_outcome = outcomes[rows, choices]
    selected_utility = utilities[rows, choices]
    base_utility = utilities[:, data.base_index]
    oracle_utility = utilities.max(axis=1)
    intervened = choices != data.base_index
    unnecessary = intervened & (outcomes[:, data.base_index] == "task_success")
    harmful = intervened & (selected_utility < base_utility - 1e-12)
    beneficial = intervened & (selected_utility > base_utility + 1e-12)
    opportunity = oracle_utility > base_utility + 1e-12
    missed = (~intervened) & opportunity
    n = len(indices)

    def rate_fields(name: str, mask: np.ndarray, denominator_mask: np.ndarray | None = None) -> dict[str, Any]:
        denominator = n if denominator_mask is None else int(np.sum(denominator_mask))
        numerator = int(np.sum(mask))
        return {
            f"{name}_count": numerator,
            f"{name}_numerator": numerator,
            f"{name}_denominator": denominator,
            f"{name}_rate": float(numerator / denominator) if denominator else None,
        }

    base_success = outcomes[:, data.base_index] == "task_success"
    fields: dict[str, Any] = {
        "n_decisions": n,
        "n_sources": len(set(map(_text, data.source[indices]))),
        **rate_fields("task_success", selected_outcome == "task_success"),
        **rate_fields("catastrophe", selected_outcome == "catastrophe"),
        **rate_fields("safe_noncompletion", selected_outcome == "safe_noncompletion"),
        **rate_fields("intervention", intervened),
        **rate_fields("unnecessary_intervention", unnecessary),
        **rate_fields("harmful_intervention", harmful),
        **rate_fields("beneficial_intervention", beneficial),
        **rate_fields("missed_beneficial_intervention", missed),
        "unnecessary_given_base_success_count": int(np.sum(unnecessary)),
        "unnecessary_given_base_success_denominator": int(np.sum(base_success)),
        "unnecessary_given_base_success_rate": (
            float(np.sum(unnecessary) / np.sum(base_success)) if np.any(base_success) else None
        ),
        "missed_given_beneficial_opportunity_count": int(np.sum(missed)),
        "missed_given_beneficial_opportunity_denominator": int(np.sum(opportunity)),
        "missed_given_beneficial_opportunity_rate": (
            float(np.sum(missed) / np.sum(opportunity)) if np.any(opportunity) else None
        ),
        "utility_sum": float(np.sum(selected_utility)),
        "utility_denominator": n,
        "mean_utility": float(np.mean(selected_utility)) if n else None,
        "oracle_regret_sum": float(np.sum(oracle_utility - selected_utility)),
        "oracle_regret_denominator": n,
        "mean_oracle_regret": float(np.mean(oracle_utility - selected_utility)) if n else None,
        "value_gain_over_base_sum": float(np.sum(selected_utility - base_utility)),
        "oracle_value_available_sum": float(np.sum(oracle_utility - base_utility)),
    }
    denominator = fields["oracle_value_available_sum"]
    fields["oracle_value_recovered"] = (
        float(fields["value_gain_over_base_sum"] / denominator)
        if denominator > 1e-12 else None
    )
    # A compact alias is useful to plotting/reporting code while the task-facing
    # name remains task_success_rate.
    fields["success_rate"] = fields["task_success_rate"]
    return fields


def _metric_number(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    return float(value) if value is not None else float("nan")


def _group_rows(
    data: SuiteData,
    development: np.ndarray,
    group_name: str,
    values: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dev_values = values[development]
    for value in sorted(set(dev_values.tolist()), key=lambda item: _text(item)):
        member = development[np.asarray([item == value for item in dev_values], dtype=bool)]
        for method_index, method in enumerate(data.method_names):
            rows.append({"method": method, group_name: _native(value), **metric_values(data, method_index, member)})
    return rows


def _bootstrap_settings(manifest: Mapping[str, Any]) -> tuple[int, int]:
    config = _mapping_at(manifest, ("statistics", "bootstrap"))
    if not config:
        config = _mapping_at(manifest, ("audit", "bootstrap"))
    replicates = int(config.get("replicates", 5000))
    seed = int(config.get("seed", 2027))
    return max(0, replicates), seed


def _source_metric_matrix(
    data: SuiteData,
    development: np.ndarray,
    sources: Sequence[str],
) -> dict[str, np.ndarray]:
    matrix = {
        metric: np.full((len(data.method_names), len(sources)), np.nan, dtype=np.float64)
        for metric in INFERENCE_METRICS
    }
    for source_index, source in enumerate(sources):
        indices = development[data.source[development] == source]
        for method_index in range(len(data.method_names)):
            row = metric_values(data, method_index, indices)
            for metric in INFERENCE_METRICS:
                matrix[metric][method_index, source_index] = _metric_number(row, metric)
    return matrix


def _percentile_interval(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def _sign_flip(
    differences: np.ndarray,
    seed: int,
    max_exact_sources: int = 16,
    monte_carlo_replicates: int = 100000,
) -> dict[str, Any]:
    differences = differences[np.isfinite(differences)]
    nonzero = differences[np.abs(differences) > 1e-15]
    observed = float(np.mean(differences)) if len(differences) else None
    if not len(nonzero):
        return {
            "mode": "degenerate",
            "n_sources": int(len(differences)),
            "n_nonzero_sources": 0,
            "observed_mean_difference": observed,
            "two_sided_p": 1.0,
            "one_sided_greater_p": 1.0,
            "patterns_or_replicates": 1,
            "seed": seed,
        }
    if len(differences) <= max_exact_sources:
        signs = np.asarray(
            list(itertools.product((-1.0, 1.0), repeat=len(differences))), dtype=np.float64
        )
        null = np.mean(signs * differences[None, :], axis=1)
        two_sided = float(np.mean(np.abs(null) >= abs(float(observed)) - 1e-15))
        greater = float(np.mean(null >= float(observed) - 1e-15))
        mode = "exact_enumeration"
        count = len(null)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(monte_carlo_replicates, len(differences)),
        )
        null = np.mean(signs * differences[None, :], axis=1)
        two_sided = float((np.sum(np.abs(null) >= abs(float(observed)) - 1e-15) + 1) / (len(null) + 1))
        greater = float((np.sum(null >= float(observed) - 1e-15) + 1) / (len(null) + 1))
        mode = "fixed_seed_monte_carlo"
        count = len(null)
    return {
        "mode": mode,
        "n_sources": int(len(differences)),
        "n_nonzero_sources": int(len(nonzero)),
        "observed_mean_difference": observed,
        "two_sided_p": two_sided,
        "one_sided_greater_p": greater,
        "patterns_or_replicates": int(count),
        "seed": seed,
    }


def _primary_method(manifest: Mapping[str, Any], methods: Sequence[str]) -> str:
    raw = _first(
        manifest,
        (("primary_method",), ("protocol", "primary_method"), ("primary", "method")),
        None,
    )
    if isinstance(raw, Mapping):
        raw = raw.get("name")
    if raw is not None and _text(raw) in methods:
        return _text(raw)
    outcome = [name for name in methods if "outcome" in _slug(name) and "without" not in _slug(name) and "hidden_only" not in _slug(name)]
    return outcome[0] if outcome else methods[0]


def _inference(
    data: SuiteData,
    development: np.ndarray,
    manifest: Mapping[str, Any],
    overall_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sources = tuple(sorted(set(map(_text, data.source[development]))))
    replicates, seed = _bootstrap_settings(manifest)
    rng = np.random.default_rng(seed)
    draws = (
        rng.integers(0, len(sources), size=(replicates, len(sources)), endpoint=False)
        if replicates and sources else np.empty((0, len(sources)), dtype=np.int64)
    )
    matrices = _source_metric_matrix(data, development, sources)
    primary = _primary_method(manifest, data.method_names)
    primary_index = data.method_names.index(primary)

    method_bootstrap: dict[str, Any] = {}
    by_method_row = {row["method"]: row for row in overall_rows}
    for method_index, method in enumerate(data.method_names):
        metric_result: dict[str, Any] = {}
        for metric in INFERENCE_METRICS:
            source_values = matrices[metric][method_index]
            bootstrap = (
                np.nanmean(source_values[draws], axis=1) if len(draws) else np.asarray([])
            )
            interval = _percentile_interval(bootstrap)
            metric_result[metric] = {
                "source_macro_estimate": float(np.nanmean(source_values)),
                "source_bootstrap_95_ci": interval,
            }
            by_method_row[method][f"source_macro_{metric}"] = float(np.nanmean(source_values))
            by_method_row[method][f"{metric}_source_bootstrap_ci_low"] = interval[0] if interval else None
            by_method_row[method][f"{metric}_source_bootstrap_ci_high"] = interval[1] if interval else None
        method_bootstrap[method] = metric_result

    comparisons: dict[str, Any] = {}
    for other_index, other in enumerate(data.method_names):
        if other == primary:
            continue
        metric_result = {}
        for offset, metric in enumerate(INFERENCE_METRICS):
            differences = matrices[metric][primary_index] - matrices[metric][other_index]
            bootstrap = (
                np.nanmean(differences[draws], axis=1) if len(draws) else np.asarray([])
            )
            oriented = differences if HIGHER_IS_BETTER[metric] else -differences
            exact = _sign_flip(oriented, seed + 1009 * (other_index + 1) + offset)
            metric_result[metric] = {
                "difference_definition": f"{primary} minus {other}",
                "higher_value_is_better": HIGHER_IS_BETTER[metric],
                "source_differences": {
                    source: float(value) for source, value in zip(sources, differences)
                },
                "observed_source_macro_difference": float(np.nanmean(differences)),
                "source_bootstrap_95_ci": _percentile_interval(bootstrap),
                "discordant_sources_favoring_primary": int(np.sum(oriented > 1e-15)),
                "discordant_sources_favoring_comparator": int(np.sum(oriented < -1e-15)),
                "tied_sources": int(np.sum(np.abs(oriented) <= 1e-15)),
                "oriented_sign_flip_test": exact,
            }
        comparisons[other] = {
            "left_method": primary,
            "right_method": other,
            "metrics": metric_result,
        }
    return {
        "schema_version": 1,
        "evaluation_split": "development",
        "independent_unit": "source",
        "sources": list(sources),
        "primary_method": primary,
        "shared_source_bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "same_resampled_source_indices_for_all_methods_and_metrics": True,
        },
        "method_bootstrap": method_bootstrap,
        "comparisons": comparisons,
    }


def _valid_probabilities(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.all(np.isfinite(probabilities), axis=-1)
    nonnegative = np.all(probabilities >= 0.0, axis=-1)
    total = np.sum(probabilities, axis=-1)
    valid = finite & nonnegative & (total > 0.0)
    normalized = np.full_like(probabilities, np.nan, dtype=np.float64)
    normalized[valid] = probabilities[valid] / total[valid, None]
    return normalized, valid


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> dict[str, Any]:
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    valid = np.isfinite(confidence) & np.isfinite(correct)
    confidence, correct = confidence[valid], correct[valid]
    rows = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        member = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if not np.any(member):
            continue
        mean_confidence = float(np.mean(confidence[member]))
        empirical = float(np.mean(correct[member]))
        ece += float(np.mean(member)) * abs(mean_confidence - empirical)
        rows.append({
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": int(np.sum(member)),
            "mean_confidence": mean_confidence,
            "empirical_frequency": empirical,
        })
    return {"ece": float(ece) if len(confidence) else None, "bins": rows}


def _multiclass_calibration(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    normalized, valid = _valid_probabilities(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    valid &= (labels >= 0) & (labels < probabilities.shape[-1])
    count = int(np.sum(valid))
    if not count:
        return {
            "available": False,
            "n_valid": 0,
            "n_missing_or_invalid": int(len(labels)),
            "nll": None,
            "multiclass_brier": None,
            "ece": None,
            "reliability_bins": [],
        }
    p = normalized[valid]
    y = labels[valid]
    row = np.arange(count)
    one_hot = np.eye(probabilities.shape[-1], dtype=np.float64)[y]
    predicted = np.argmax(p, axis=1)
    reliability = _ece(np.max(p, axis=1), predicted == y)
    return {
        "available": True,
        "n_valid": count,
        "n_missing_or_invalid": int(len(labels) - count),
        "nll": float(-np.mean(np.log(np.clip(p[row, y], 1e-12, 1.0)))),
        "multiclass_brier": float(np.mean(np.sum(np.square(p - one_hot), axis=1))),
        "ece": reliability["ece"],
        "reliability_bins": reliability["bins"],
    }


def _binary_calibration(probability: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = (
        np.isfinite(probability) & np.isfinite(target)
        & (probability >= 0.0) & (probability <= 1.0)
    )
    if not np.any(valid):
        return {
            "available": False,
            "n_valid": 0,
            "n_missing_or_invalid": int(len(probability)),
            "nll": None,
            "brier": None,
            "ece": None,
            "reliability_bins": [],
        }
    p, y = probability[valid], target[valid]
    reliability = _ece(p, y)
    return {
        "available": True,
        "n_valid": int(len(p)),
        "n_missing_or_invalid": int(np.sum(~valid)),
        "nll": float(-np.mean(y * np.log(np.clip(p, 1e-12, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-12, 1.0)))),
        "brier": float(np.mean(np.square(p - y))),
        "ece": reliability["ece"],
        "reliability_bins": reliability["bins"],
    }


def _source_weights(sources: np.ndarray) -> np.ndarray:
    sources = np.asarray(sources)
    counts = {source: int(np.sum(sources == source)) for source in set(sources.tolist())}
    return np.asarray([1.0 / counts[source] for source in sources], dtype=np.float64)


def _source_balanced_mean(values: np.ndarray, sources: np.ndarray) -> float:
    return float(np.average(np.asarray(values, dtype=np.float64), weights=_source_weights(sources)))


def _rate_threshold(scores: np.ndarray, sources: np.ndarray, target: float) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    if not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("preference-transfer calibration scores must be finite and non-empty")
    values = sorted(set(map(float, scores.tolist())))
    candidates = [
        float("-inf"),
        *(float((left + right) / 2.0) for left, right in zip(values, values[1:])),
        float("inf"),
    ]
    weights = _source_weights(sources)
    records = []
    for threshold in sorted(set(candidates)):
        rate = float(np.average(scores > threshold, weights=weights))
        records.append((-abs(rate - target), -rate, threshold, rate))
    best = max(records, key=lambda row: row[:3])
    return {"threshold": float(best[2]), "source_balanced_rate": float(best[3]), "target_rate": float(target)}


def _expected_values_for_lambda(
    probabilities: np.ndarray,
    outcome_axis: Sequence[str],
    catastrophe_cost: float,
    noncompletion_cost: float,
) -> np.ndarray | None:
    normalized, valid = _valid_probabilities(probabilities)
    if not np.all(valid):
        return None
    value_by_outcome = {
        "task_success": 1.0,
        "catastrophe": -float(catastrophe_cost),
        "safe_noncompletion": -float(noncompletion_cost),
    }
    weights = np.asarray([value_by_outcome[name] for name in outcome_axis], dtype=np.float64)
    return normalized @ weights


def _choice_from_values(
    values: np.ndarray, threshold: float, base_index: int, option_cost: np.ndarray
) -> np.ndarray:
    intervention_options = [index for index in range(values.shape[1]) if index != base_index]
    order = sorted(intervention_options, key=lambda index: (float(option_cost[index]), index))
    intervention = np.empty(len(values), dtype=np.int64)
    for row_index, row in enumerate(values):
        best = max(float(row[index]) for index in intervention_options)
        intervention[row_index] = next(
            index for index in order if abs(float(row[index]) - best) <= 1e-12
        )
    score = values[np.arange(len(values)), intervention] - values[:, base_index]
    return np.where(score > threshold, intervention, base_index).astype(np.int64)


def _realized_utility_for_lambda(
    data: SuiteData,
    indices: np.ndarray,
    catastrophe_cost: float,
    noncompletion_cost: float,
) -> np.ndarray:
    values = {
        "task_success": 1.0,
        "catastrophe": -float(catastrophe_cost),
        "safe_noncompletion": -float(noncompletion_cost),
    }
    return np.asarray(
        [[values[_text(outcome)] for outcome in row] for row in data.outcomes[indices]],
        dtype=np.float64,
    )


def _choice_operating_metrics(
    data: SuiteData,
    indices: np.ndarray,
    choices: np.ndarray,
    utility: np.ndarray,
) -> dict[str, float]:
    rows = np.arange(len(indices))
    selected_outcome = data.outcomes[indices][rows, choices]
    sources = data.source[indices]
    return {
        "source_balanced_mean_utility": _source_balanced_mean(utility[rows, choices], sources),
        "source_balanced_catastrophe_rate": _source_balanced_mean(selected_outcome == "catastrophe", sources),
        "source_balanced_task_success_rate": _source_balanced_mean(selected_outcome == "task_success", sources),
        "source_balanced_intervention_rate": _source_balanced_mean(choices != data.base_index, sources),
    }


def _preference_transfer(
    data: SuiteData,
    manifest: Mapping[str, Any],
    outcome_method: str,
) -> dict[str, Any]:
    protocol = _mapping_at(manifest, ("protocol",))
    lambdas_raw = protocol.get("lambda_grid", [])
    if not isinstance(lambdas_raw, Sequence) or isinstance(lambdas_raw, (str, bytes)):
        lambdas_raw = []
    lambdas = [float(value) for value in lambdas_raw]
    primary_lambda = _primary_lambda(manifest)
    noncompletion_cost = float(protocol.get("noncompletion_cost", 0.0))
    target_rate = float(protocol.get("primary_target_intervention_rate", 0.0))
    thresholds = _thresholds(manifest)
    explicit = bool(
        _first(
            manifest,
            (
                ("protocol", "preference_transfer_evidence"),
                ("evidence", "preference_transfer"),
                ("method_metadata", outcome_method, "preference_reweighting_evidence"),
            ),
            False,
        )
    )
    if outcome_method not in data.method_names or not lambdas or thresholds is None:
        return {
            "available": False,
            "outcome_method": outcome_method,
            "reason": "outcome method, lambda_grid, or GO tolerances unavailable",
            "explicit_manifest_evidence": explicit,
            "reweighting_value": explicit,
            "per_lambda": [],
        }
    calibration_indices = np.flatnonzero(np.asarray([_slug(value) == "calibration" for value in data.split]))
    development_indices = np.flatnonzero(np.asarray([_slug(value) == "development" for value in data.split]))
    if not len(calibration_indices) or not len(development_indices):
        return {
            "available": False,
            "outcome_method": outcome_method,
            "reason": "calibration or development split is empty",
            "explicit_manifest_evidence": explicit,
            "reweighting_value": explicit,
            "per_lambda": [],
        }
    method_index = data.method_names.index(outcome_method)
    all_probabilities = data.outcome_probabilities[method_index]
    per_lambda = []
    passes = []
    for catastrophe_cost in lambdas:
        comparator = (
            "DirectQ" if np.isclose(catastrophe_cost, primary_lambda)
            else f"DirectQ-lambda{catastrophe_cost:g}"
        )
        entry: dict[str, Any] = {
            "lambda": catastrophe_cost,
            "outcome_method": outcome_method,
            "separately_trained_direct_q_comparator": comparator,
        }
        if comparator not in data.method_names:
            entry.update({"available": False, "reason": "matching Direct-Q comparator unavailable", "passes": False})
            per_lambda.append(entry)
            continue
        expected = _expected_values_for_lambda(
            all_probabilities,
            data.outcome_names,
            catastrophe_cost,
            noncompletion_cost,
        )
        if expected is None:
            entry.update({"available": False, "reason": "outcome probabilities missing or invalid", "passes": False})
            per_lambda.append(entry)
            continue
        intervention_options = [index for index in range(3) if index != data.base_index]
        score = np.max(expected[:, intervention_options], axis=1) - expected[:, data.base_index]
        calibration_record = _rate_threshold(
            score[calibration_indices], data.source[calibration_indices], target_rate
        )
        threshold = calibration_record["threshold"]
        outcome_choice = _choice_from_values(
            expected[development_indices], threshold, data.base_index, data.option_cost
        )
        direct_index = data.method_names.index(comparator)
        direct_choice = data.choices[direct_index, development_indices]
        realized = _realized_utility_for_lambda(
            data, development_indices, catastrophe_cost, noncompletion_cost
        )
        outcome_metrics = _choice_operating_metrics(
            data, development_indices, outcome_choice, realized
        )
        direct_metrics = _choice_operating_metrics(
            data, development_indices, direct_choice, realized
        )
        utility_difference = (
            outcome_metrics["source_balanced_mean_utility"]
            - direct_metrics["source_balanced_mean_utility"]
        )
        catastrophe_difference = (
            outcome_metrics["source_balanced_catastrophe_rate"]
            - direct_metrics["source_balanced_catastrophe_rate"]
        )
        intervention_difference = (
            outcome_metrics["source_balanced_intervention_rate"]
            - direct_metrics["source_balanced_intervention_rate"]
        )
        passes_lambda = bool(
            utility_difference >= -thresholds["utility_absolute"]
            and catastrophe_difference <= thresholds["catastrophe_rate_absolute"]
            and abs(intervention_difference)
            <= thresholds["intervention_rate_absolute"]
        )
        passes.append(catastrophe_cost) if passes_lambda else None
        entry.update({
            "available": True,
            "calibration": calibration_record,
            "development_outcome_reweighted": outcome_metrics,
            "development_direct_q_separately_trained": direct_metrics,
            "outcome_minus_direct_q_utility": utility_difference,
            "outcome_minus_direct_q_catastrophe_rate": catastrophe_difference,
            "outcome_minus_direct_q_intervention_rate": intervention_difference,
            "absolute_utility_difference_within_tolerance": abs(utility_difference) <= thresholds["utility_absolute"],
            "absolute_catastrophe_difference_within_tolerance": abs(catastrophe_difference) <= thresholds["catastrophe_rate_absolute"],
            "absolute_intervention_difference_within_tolerance": abs(intervention_difference) <= thresholds["intervention_rate_absolute"],
            "noninferior_within_frozen_tolerances": passes_lambda,
            "passes": passes_lambda,
        })
        per_lambda.append(entry)
    minimum = min(3, len(lambdas))
    measured_value = len(passes) >= minimum and minimum >= 3
    return {
        "available": any(entry.get("available", False) for entry in per_lambda),
        "outcome_method": outcome_method,
        "outcome_model_refit_across_lambdas": False,
        "comparison": "one frozen outcome distribution reweighted versus a separately trained Direct-Q at each lambda",
        "target_intervention_rate": target_rate,
        "minimum_noninferior_lambdas_required": 3,
        "passing_lambdas": passes,
        "measured_reweighting_value": measured_value,
        "explicit_manifest_evidence": explicit,
        "reweighting_value": bool(measured_value or explicit),
        "tolerances_from_manifest": thresholds,
        "per_lambda": per_lambda,
    }


def _advantage_calibration(
    predicted_values: np.ndarray,
    utilities: np.ndarray,
    base_index: int,
    bins: int = 10,
) -> dict[str, Any]:
    intervention_options = [index for index in range(utilities.shape[1]) if index != base_index]
    valid = np.all(np.isfinite(predicted_values), axis=1)
    if not np.any(valid):
        return {
            "available": False,
            "n_valid": 0,
            "n_missing_or_invalid": int(len(predicted_values)),
            "mae": None,
            "rmse": None,
            "pearson_correlation": None,
            "calibration_error": None,
            "bins": [],
        }
    predicted_values = predicted_values[valid]
    utilities = utilities[valid]
    local_best = np.argmax(predicted_values[:, intervention_options], axis=1)
    option = np.asarray(intervention_options, dtype=np.int64)[local_best]
    row = np.arange(len(option))
    predicted = predicted_values[row, option] - predicted_values[:, base_index]
    realized = utilities[row, option] - utilities[:, base_index]
    order = np.argsort(predicted, kind="stable")
    groups = [group for group in np.array_split(order, min(bins, len(order))) if len(group)]
    calibration_rows = []
    weighted_error = 0.0
    for group in groups:
        predicted_mean = float(np.mean(predicted[group]))
        realized_mean = float(np.mean(realized[group]))
        weighted_error += len(group) / len(order) * abs(predicted_mean - realized_mean)
        calibration_rows.append({
            "count": int(len(group)),
            "predicted_advantage_min": float(np.min(predicted[group])),
            "predicted_advantage_max": float(np.max(predicted[group])),
            "predicted_advantage_mean": predicted_mean,
            "realized_advantage_mean": realized_mean,
            "beneficial_fraction": float(np.mean(realized[group] > 1e-12)),
        })
    correlation = None
    if len(predicted) > 1 and np.std(predicted) > 0 and np.std(realized) > 0:
        correlation = float(np.corrcoef(predicted, realized)[0, 1])
    residual = predicted - realized
    return {
        "available": True,
        "n_valid": int(len(predicted)),
        "n_missing_or_invalid": int(np.sum(~valid)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "pearson_correlation": correlation,
        "calibration_error": float(weighted_error),
        "bins": calibration_rows,
    }


def _calibration(
    data: SuiteData,
    development: np.ndarray,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    utilities = _utility_matrix(data, development)
    oracle = _oracle_choice(utilities, data.base_index, data.option_cost)
    metadata = _method_metadata(manifest, data.method_names)
    class_index = {name: index for index, name in enumerate(data.outcome_names)}
    # data.outcome_names records the probability-axis order, whereas outcomes
    # are canonicalized strings.
    labels = np.asarray(
        [[class_index[_text(value)] for value in row] for row in data.outcomes[development]],
        dtype=np.int64,
    )
    methods: dict[str, Any] = {}
    for method_index, method in enumerate(data.method_names):
        outcome_probabilities = data.outcome_probabilities[method_index, development]
        per_option = {}
        flat_result = _multiclass_calibration(
            outcome_probabilities.reshape(-1, 3), labels.reshape(-1)
        )
        for option_index, option in enumerate(data.option_names):
            per_option[option] = _multiclass_calibration(
                outcome_probabilities[:, option_index, :], labels[:, option_index]
            )
        outcome_result = {**flat_result, "per_option": per_option}
        choice_result = _multiclass_calibration(
            data.choice_probabilities[method_index, development], oracle
        )
        choice_result["target"] = "realized_oracle_choice_with_frozen_tie_break"
        family = _slug(metadata[method].get("family", ""))
        if family in {"scalar_risk", "two_stage"} or "risk" in _slug(method):
            base_probability = data.choice_probabilities[
                method_index, development, data.base_index
            ]
            risk_result = _binary_calibration(
                1.0 - base_probability,
                data.outcomes[development, data.base_index] == "catastrophe",
            )
            risk_result["target"] = "base_option_catastrophe"
            risk_result["probability_definition"] = "1 - choice_probability[Base]"
        else:
            risk_result = {
                "available": False,
                "reason": "not a scalar_risk method",
                "target": "base_option_catastrophe",
                "n_valid": 0,
                "n_missing_or_invalid": int(len(development)),
                "nll": None,
                "brier": None,
                "ece": None,
                "reliability_bins": [],
            }
        advantage_result = _advantage_calibration(
            data.predicted_values[method_index, development], utilities, data.base_index
        )
        methods[method] = {
            "outcome_probabilities": outcome_result,
            "choice_probabilities": choice_result,
            "base_catastrophe_binary": risk_result,
            "predicted_advantage": advantage_result,
        }
    primary = _primary_method(manifest, data.method_names)
    return {
        "schema_version": 1,
        "evaluation_split": "development",
        "outcome_probability_axis": list(data.outcome_names),
        "choice_probability_axis": list(data.option_names),
        "missing_predictions_are_reported_as_unavailable": True,
        "methods": methods,
        "preference_transfer": _preference_transfer(data, manifest, primary),
    }


def _method_metadata(manifest: Mapping[str, Any], methods: Sequence[str]) -> dict[str, dict[str, Any]]:
    result = {method: {} for method in methods}
    raw = manifest.get("method_metadata", manifest.get("methods", {}))
    if isinstance(raw, Mapping):
        for method in methods:
            item = raw.get(method, {})
            if isinstance(item, Mapping):
                result[method] = dict(item)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Mapping):
                name = _text(item.get("name", item.get("method", "")))
                if name in result:
                    result[name] = dict(item)
    return result


def _method_tags(name: str, metadata: Mapping[str, Any]) -> set[str]:
    values = [name]
    for key in (
        "role", "family", "kind", "feature_contract", "features",
        "representation", "geometry", "fixed_option",
    ):
        if key in metadata:
            values.append(metadata[key])
    tags: set[str] = set()
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            tags.update(_slug(item) for item in value)
        else:
            tags.add(_slug(value))
    return tags


def _contains(tags: set[str], *needles: str) -> bool:
    joined = " ".join(sorted(tags))
    return any(needle in joined for needle in needles)


def _frontier_rows(
    data: SuiteData,
    manifest: Mapping[str, Any],
    overall_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata = _method_metadata(manifest, data.method_names)
    primary = _primary_method(manifest, data.method_names)
    for index, row in enumerate(overall_rows):
        row["parameter_count"] = _native(data.parameter_count[index])
        row["latency_us_per_decision"] = _native(data.latency_us_per_decision[index])
        row["is_primary_method"] = row["method"] == primary
        row["method_family"] = metadata[row["method"]].get("family", "")
        row["feature_contract"] = metadata[row["method"]].get("feature_contract", "")
    frontier = []
    for row in overall_rows:
        utility_dominated = any(
            other["method"] != row["method"]
            and _metric_number(other, "mean_utility") >= _metric_number(row, "mean_utility") - 1e-12
            and _metric_number(other, "intervention_rate") <= _metric_number(row, "intervention_rate") + 1e-12
            and (
                _metric_number(other, "mean_utility") > _metric_number(row, "mean_utility") + 1e-12
                or _metric_number(other, "intervention_rate") < _metric_number(row, "intervention_rate") - 1e-12
            )
            for other in overall_rows
        )
        success_dominated = any(
            other["method"] != row["method"]
            and _metric_number(other, "task_success_rate") >= _metric_number(row, "task_success_rate") - 1e-12
            and _metric_number(other, "catastrophe_rate") <= _metric_number(row, "catastrophe_rate") + 1e-12
            and (
                _metric_number(other, "task_success_rate") > _metric_number(row, "task_success_rate") + 1e-12
                or _metric_number(other, "catastrophe_rate") < _metric_number(row, "catastrophe_rate") - 1e-12
            )
            for other in overall_rows
        )
        frontier.append({
            **row,
            "is_utility_intervention_frontier": not utility_dominated,
            "is_success_catastrophe_frontier": not success_dominated,
        })
    return frontier


def _thresholds(manifest: Mapping[str, Any]) -> dict[str, float] | None:
    raw = _mapping_at(manifest, ("protocol", "go_thresholds"))
    required = (
        "intervention_rate_absolute", "utility_absolute", "catastrophe_rate_absolute"
    )
    if not raw or any(key not in raw for key in required):
        return None
    try:
        result = {key: float(raw[key]) for key in required}
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        return None
    return result


def _gate_decision(
    data: SuiteData,
    manifest: Mapping[str, Any],
    frontier: list[dict[str, Any]],
    inference: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = _thresholds(manifest)
    primary = _primary_method(manifest, data.method_names)
    metadata = _method_metadata(manifest, data.method_names)
    rows = {row["method"]: row for row in frontier}
    if thresholds is None:
        return {
            "schema_version": 1,
            "decision": "INCONCLUSIVE",
            "primary_method": primary,
            "reason": "manifest.protocol.go_thresholds is missing or incomplete",
            "required_thresholds": [
                "intervention_rate_absolute", "utility_absolute", "catastrophe_rate_absolute"
            ],
            "oracle_geometry_can_trigger_stop_d": False,
        }
    rate_tol = thresholds["intervention_rate_absolute"]
    utility_tol = thresholds["utility_absolute"]
    catastrophe_tol = thresholds["catastrophe_rate_absolute"]
    declared_primary = _mapping_at(manifest, ("protocol",)).get("primary_methods")
    if isinstance(declared_primary, Sequence) and not isinstance(declared_primary, (str, bytes)):
        gate_methods = [name for name in map(_text, declared_primary) if name in data.method_names]
    else:
        gate_methods = list(data.method_names)
    tags = {name: _method_tags(name, metadata[name]) for name in data.method_names}
    primary_row = rows[primary]

    comparator_spec = _mapping_at(manifest, ("protocol", "go_comparators"))
    risk_reference = _text(comparator_spec.get("risk_reference", ""))

    def declared_methods(key: str) -> list[str]:
        raw = comparator_spec.get(key, [])
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return []
        return [name for name in map(_text, raw) if name in gate_methods]

    risk = declared_methods("pivot_risk_methods")
    direct = declared_methods("go_a_direct_cover_methods")
    go_b_direct = declared_methods("go_b_value_methods")
    required_comparators = {
        risk_reference,
        *risk,
        *direct,
        *go_b_direct,
    }
    if (
        not comparator_spec
        or not risk_reference
        or risk_reference not in gate_methods
        or len(risk) != len(comparator_spec.get("pivot_risk_methods", []))
        or len(direct) != len(comparator_spec.get("go_a_direct_cover_methods", []))
        or len(go_b_direct) != len(comparator_spec.get("go_b_value_methods", []))
        or not required_comparators <= set(gate_methods)
    ):
        return {
            "schema_version": 1,
            "decision": "INCONCLUSIVE",
            "primary_method": primary,
            "reason": "manifest.protocol.go_comparators is missing or names unavailable methods",
            "oracle_geometry_can_trigger_stop_d": False,
        }
    simple = []
    oracle_geometry = []
    stop_eligible_raw = _mapping_at(manifest, ("geometry_availability",)).get(
        "stop_d_eligible"
    )
    stop_eligible = (
        set(map(_text, stop_eligible_raw))
        if isinstance(stop_eligible_raw, Sequence) and not isinstance(stop_eligible_raw, (str, bytes))
        else None
    )
    for name in gate_methods:
        if name == primary:
            continue
        candidate = _contains(
            tags[name], "robot_action", "robot_state", "without_vla", "without_hidden",
            "state_action_only", "simple_geometry", "observable_geometry", "deployable_geometry",
        )
        geometry = _contains(tags[name], "geometry")
        is_oracle = bool(
            metadata[name].get("oracle_geometry", False)
            or metadata[name].get("oracle_diagnostic_only", False)
            or (geometry and metadata[name].get("deployable") is False)
            or (geometry and _contains(tags[name], "oracle"))
            or _contains(tags[name], "oracle_geometry", "privileged_geometry", "future_geometry")
        )
        if geometry and is_oracle:
            oracle_geometry.append(name)
        elif candidate and (stop_eligible is None or name in stop_eligible):
            simple.append(name)

    def gate_value(row: Mapping[str, Any], metric: str) -> float:
        source_macro = row.get(f"source_macro_{metric}")
        return float(source_macro) if source_macro is not None else _metric_number(row, metric)

    def comparable(comparator: str) -> bool:
        row = rows[comparator]
        return bool(
            gate_value(row, "mean_utility") >= gate_value(primary_row, "mean_utility") - utility_tol
            and gate_value(row, "catastrophe_rate") <= gate_value(primary_row, "catastrophe_rate") + catastrophe_tol
            and gate_value(row, "task_success_rate") >= gate_value(primary_row, "task_success_rate") - utility_tol
            and gate_value(row, "intervention_rate") <= gate_value(primary_row, "intervention_rate") + rate_tol
        )

    def same_rate_tracks(comparator: str) -> bool:
        row = rows[comparator]
        return bool(
            abs(gate_value(row, "intervention_rate") - gate_value(primary_row, "intervention_rate")) <= rate_tol
            and gate_value(row, "mean_utility") >= gate_value(primary_row, "mean_utility") - utility_tol
            and gate_value(row, "catastrophe_rate") <= gate_value(primary_row, "catastrophe_rate") + catastrophe_tol
        )

    stop_matches = [name for name in simple if comparable(name)]
    pivot_matches = [name for name in risk if same_rate_tracks(name)]
    excluded_oracle_matches = [name for name in oracle_geometry if comparable(name)]

    preference_transfer = calibration.get("preference_transfer", {})
    calibration_or_reweighting = bool(
        preference_transfer.get("reweighting_value", False)
        or metadata[primary].get("preference_reweighting_evidence", False)
    )
    if risk_reference:
        risk_row = rows[risk_reference]
        outcome_beats_risk = bool(
            gate_value(primary_row, "mean_utility")
            >= gate_value(risk_row, "mean_utility") + utility_tol
            and gate_value(primary_row, "catastrophe_rate")
            <= gate_value(risk_row, "catastrophe_rate") + catastrophe_tol
        )
        not_rate_only = bool(
            gate_value(primary_row, "intervention_rate")
            <= gate_value(risk_row, "intervention_rate") + rate_tol
        )
        comparison = inference.get("comparisons", {}).get(risk_reference, {})
        utility_comparison = comparison.get("metrics", {}).get("mean_utility", {})
        discordant_stable = bool(
            utility_comparison.get("discordant_sources_favoring_primary", 0)
            >= utility_comparison.get("discordant_sources_favoring_comparator", 0)
        )
        same_rate_stable = not_rate_only and discordant_stable
    else:
        outcome_beats_risk = False
        not_rate_only = False
        same_rate_stable = False
    direct_cover = [name for name in direct if comparable(name)]
    outcome_not_covered = not direct_cover
    go_a_criteria = {
        "outcome_router_beats_risk_best_fixed": outcome_beats_risk,
        "not_fully_covered_by_direct_methods": outcome_not_covered,
        "calibration_or_preference_reweighting_value_available": calibration_or_reweighting,
        "improvement_not_explained_by_higher_intervention_rate": not_rate_only,
        "same_rate_source_direction_stable": same_rate_stable,
    }

    def similar_to_outcome(name: str) -> bool:
        row = rows[name]
        return bool(
            abs(gate_value(primary_row, "mean_utility") - gate_value(row, "mean_utility")) <= utility_tol
            and abs(gate_value(primary_row, "catastrophe_rate") - gate_value(row, "catastrophe_rate")) <= catastrophe_tol
            and abs(gate_value(primary_row, "intervention_rate") - gate_value(row, "intervention_rate")) <= rate_tol
        )

    risk_row = rows[risk_reference]

    def clearly_beats_risk(name: str) -> bool:
        row = rows[name]
        return bool(
            gate_value(row, "mean_utility")
            >= gate_value(risk_row, "mean_utility") + utility_tol
            and gate_value(row, "catastrophe_rate")
            <= gate_value(risk_row, "catastrophe_rate") + catastrophe_tol
            and abs(
                gate_value(row, "intervention_rate")
                - gate_value(risk_row, "intervention_rate")
            ) <= rate_tol
        )

    direct_similarity = {
        name: similar_to_outcome(name) for name in go_b_direct
    }
    beats_risk = {
        name: clearly_beats_risk(name) for name in [primary, *go_b_direct]
    }
    outcome_direct_similar = bool(go_b_direct) and all(direct_similarity.values())
    both_beat_risk = bool(go_b_direct) and all(beats_risk.values())
    go_b_criteria = {
        "outcome_and_direct_value_methods_similar": outcome_direct_similar,
        "both_clearly_beat_scalar_risk": both_beat_risk,
        "similarity_by_predeclared_direct_method": direct_similarity,
        "matched_rate_safe_improvement_over_risk": beats_risk,
    }
    go_b_all_met = outcome_direct_similar and both_beat_risk

    if stop_matches:
        decision = "STOP-D"
        reason = "A deployable robot-state/simple-geometry method matches the primary router within frozen tolerances."
    elif pivot_matches:
        decision = "PIVOT-C"
        reason = "Risk-to-Detour/Best-Fixed tracks the primary router at the same intervention rate."
    elif all(go_a_criteria.values()):
        decision = "GO-A"
        reason = "Outcome decomposition clears all five frozen GO-A checks on development sources."
    elif go_b_all_met:
        decision = "GO-B"
        reason = "Multi-option value learning clears scalar risk, without a distinct outcome-head advantage."
    else:
        decision = "INCONCLUSIVE"
        reason = "No frozen GO/NO-GO branch is fully supported by the development evidence."
    return {
        "schema_version": 1,
        "evaluation_split": "development",
        "decision": decision,
        "reason": reason,
        "primary_method": primary,
        "thresholds_from_manifest": thresholds,
        "candidate_methods": {
            "manifest_primary_methods": gate_methods,
            "risk_to_best_fixed_or_detour": risk,
            "direct_choice_q_pairwise": direct,
            "go_b_value_methods": go_b_direct,
            "risk_reference": risk_reference,
            "deployable_simple_or_robot_only": simple,
            "oracle_geometry_diagnostic_only": oracle_geometry,
        },
        "go_a": {"criteria": go_a_criteria, "all_met": all(go_a_criteria.values())},
        "go_b": {"criteria": go_b_criteria, "all_met": go_b_all_met},
        "pivot_c": {"matching_methods": pivot_matches, "triggered": bool(pivot_matches)},
        "stop_d": {"matching_methods": stop_matches, "triggered": bool(stop_matches)},
        "oracle_geometry": {
            "matching_methods_excluded_from_stop_d": excluded_oracle_matches,
            "can_trigger_stop_d_alone": False,
        },
    }


def _choice_rows(data: SuiteData, development: np.ndarray) -> list[dict[str, Any]]:
    utilities = _utility_matrix(data, development)
    oracle = _oracle_choice(utilities, data.base_index, data.option_cost)
    rows = np.arange(len(development))
    result = []
    for method_index, method in enumerate(data.method_names):
        choice = data.choices[method_index, development]
        selected_utility = utilities[rows, choice]
        base_utility = utilities[:, data.base_index]
        oracle_utility = utilities[rows, oracle]
        predicted_values = data.predicted_values[method_index, development]
        for local_index, data_index in enumerate(development):
            chosen = int(choice[local_index])
            oracle_index = int(oracle[local_index])
            intervened = chosen != data.base_index
            result.append({
                "decision_id": data.decision_id[data_index],
                "source": data.source[data_index],
                "split": "development",
                "condition": data.condition[data_index],
                "horizon": _native(data.horizon[data_index]),
                "method": method,
                "choice_index": chosen,
                "choice": data.option_names[chosen],
                "selected_outcome": data.outcomes[data_index, chosen],
                "base_outcome": data.outcomes[data_index, data.base_index],
                "oracle_choice_index": oracle_index,
                "oracle_choice": data.option_names[oracle_index],
                "oracle_outcome": data.outcomes[data_index, oracle_index],
                "intervention": intervened,
                "unnecessary_intervention": bool(
                    intervened and data.outcomes[data_index, data.base_index] == "task_success"
                ),
                "harmful_intervention": bool(
                    intervened and selected_utility[local_index] < base_utility[local_index] - 1e-12
                ),
                "missed_beneficial_intervention": bool(
                    not intervened and oracle_utility[local_index] > base_utility[local_index] + 1e-12
                ),
                "selected_utility": float(selected_utility[local_index]),
                "base_utility": float(base_utility[local_index]),
                "oracle_utility": float(oracle_utility[local_index]),
                "predicted_selected_value": _native(predicted_values[local_index, chosen]),
                "predicted_base_value": _native(predicted_values[local_index, data.base_index]),
            })
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_native(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def analyze(input_dir: Path) -> dict[str, Path]:
    input_dir = input_dir.resolve()
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest.json must contain a JSON object")
    data = load_suite(input_dir, manifest)
    development = np.flatnonzero(np.asarray([_slug(value) == "development" for value in data.split]))
    if not len(development):
        raise ValueError("all_predictions.npz contains no development decisions")

    overall_rows = [
        {"method": method, **metric_values(data, method_index, development)}
        for method_index, method in enumerate(data.method_names)
    ]
    inference = _inference(data, development, manifest, overall_rows)
    per_source = _group_rows(data, development, "source", data.source)
    per_condition = _group_rows(data, development, "condition", data.condition)
    per_horizon = _group_rows(data, development, "horizon", data.horizon)
    calibration = _calibration(data, development, manifest)
    frontier = _frontier_rows(data, manifest, overall_rows)
    gate = _gate_decision(data, manifest, frontier, inference, calibration)
    choices = _choice_rows(data, development)

    outputs = {
        "overall_metrics": input_dir / "overall_metrics.csv",
        "per_source_metrics": input_dir / "per_source_metrics.csv",
        "per_condition_metrics": input_dir / "per_condition_metrics.csv",
        "per_horizon_metrics": input_dir / "per_horizon_metrics.csv",
        "all_choices": input_dir / "all_choices.csv",
        "exact_tests": input_dir / "exact_tests.json",
        "calibration": input_dir / "calibration.json",
        "frontier": input_dir / "frontier.csv",
        "gate_decision": input_dir / "gate_decision.json",
    }
    _write_csv(outputs["overall_metrics"], overall_rows)
    _write_csv(outputs["per_source_metrics"], per_source)
    _write_csv(outputs["per_condition_metrics"], per_condition)
    _write_csv(outputs["per_horizon_metrics"], per_horizon)
    _write_csv(outputs["all_choices"], choices)
    _write_json(outputs["exact_tests"], inference)
    _write_json(outputs["calibration"], calibration)
    _write_csv(outputs["frontier"], frontier)
    _write_json(outputs["gate_decision"], gate)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help="Directory containing manifest.json and all_predictions.npz; outputs are written in place.",
    )
    args = parser.parse_args()
    outputs = analyze(args.input_dir)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
