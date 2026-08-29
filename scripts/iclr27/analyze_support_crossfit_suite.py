#!/usr/bin/env python3
"""Analyze Phase 2.5B OOF predictions and emit the fail-closed rescue gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.counterfactual_router import OPTIONS
from scripts.iclr27.analyze_router_baseline_suite import _sign_flip
from scripts.iclr27.train_router_baseline_suite import realized_utility
from scripts.train_minimal_counterfactual_router import _source_weights


STRICT_LABEL_TO_OPTION = {
    "strict_base": 0,
    "strict_detour": 1,
    "strict_retreat": 2,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_native(rows))


def _source_macro_mean(values: np.ndarray, sources: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return float("nan")
    return float(np.average(values, weights=_source_weights(np.asarray(sources))))


def _source_macro_recall(
    choice: np.ndarray, labels: np.ndarray, sources: np.ndarray, label: str
) -> float:
    mask = labels == label
    if not np.any(mask):
        return float("nan")
    return _source_macro_mean(
        choice[mask] == STRICT_LABEL_TO_OPTION[label], sources[mask]
    )


def _strict_macro_f1(
    choice: np.ndarray, labels: np.ndarray, sources: np.ndarray
) -> float:
    mask = np.isin(labels, list(STRICT_LABEL_TO_OPTION))
    target = np.asarray([STRICT_LABEL_TO_OPTION[str(label)] for label in labels[mask]])
    predicted = choice[mask]
    weights = _source_weights(sources[mask])
    scores = []
    for option in range(len(OPTIONS)):
        tp = float(weights[(target == option) & (predicted == option)].sum())
        fp = float(weights[(target != option) & (predicted == option)].sum())
        fn = float(weights[(target == option) & (predicted != option)].sum())
        denominator = 2.0 * tp + fp + fn
        scores.append(0.0 if denominator <= 0.0 else 2.0 * tp / denominator)
    return float(np.mean(scores))


def metric_row(
    *,
    method: str,
    choice: np.ndarray,
    outcomes: np.ndarray,
    utility: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    conditions: np.ndarray,
) -> dict[str, Any]:
    rows = np.arange(len(choice))
    selected_outcome = outcomes[rows, choice]
    selected_utility = utility[rows, choice]
    base_utility = utility[:, 0]
    oracle_utility = utility.max(axis=1)
    source_utility = _source_macro_mean(selected_utility, sources)
    source_base = _source_macro_mean(base_utility, sources)
    source_oracle = _source_macro_mean(oracle_utility, sources)
    available = source_oracle - source_base
    recovery = (
        (source_utility - source_base) / available
        if available > 1e-12 else float("nan")
    )
    detour_recall = _source_macro_recall(
        choice, labels, sources, "strict_detour"
    )
    retreat_recall = _source_macro_recall(
        choice, labels, sources, "strict_retreat"
    )
    base_recall = _source_macro_recall(choice, labels, sources, "strict_base")
    strict_intervention = np.isin(labels, ["strict_detour", "strict_retreat"])
    strict_target = np.asarray([
        STRICT_LABEL_TO_OPTION.get(str(label), -1) for label in labels
    ])
    wrong_catastrophe = (
        (choice != 0) & (choice != strict_target)
        & (selected_outcome == "catastrophe")
    )
    intervention_regret = np.max(utility[:, 1:], axis=1) - selected_utility
    base_optimal = utility[:, 0] >= utility.max(axis=1) - 1e-12
    base_success = outcomes[:, 0] == "task_success"
    controls = conditions != "glass"
    harmful = (choice != 0) & (selected_utility < base_utility - 1e-12)
    row = {
        "method": method,
        "source_macro_utility": source_utility,
        "task_success_rate": _source_macro_mean(
            selected_outcome == "task_success", sources
        ),
        "catastrophe_rate": _source_macro_mean(
            selected_outcome == "catastrophe", sources
        ),
        "intervention_rate": _source_macro_mean(choice != 0, sources),
        "harmful_intervention_rate": _source_macro_mean(harmful, sources),
        "oracle_value_recovered": float(recovery),
        "strict_base_recall": base_recall,
        "strict_detour_recall": detour_recall,
        "strict_retreat_recall": retreat_recall,
        "detour_retreat_balanced_accuracy": float(
            np.nanmean([detour_recall, retreat_recall])
        ),
        "strict_choice_macro_f1": _strict_macro_f1(
            choice, labels, sources
        ),
        "wrong_intervention_catastrophe_rate": _source_macro_mean(
            wrong_catastrophe[strict_intervention], sources[strict_intervention]
        ),
        "intervention_choice_regret": _source_macro_mean(
            intervention_regret[strict_intervention], sources[strict_intervention]
        ),
        "control_task_success_rate": _source_macro_mean(
            selected_outcome[controls] == "task_success", sources[controls]
        ),
        "unnecessary_intervention_rate": _source_macro_mean(
            choice[base_optimal] != 0, sources[base_optimal]
        ),
        "base_success_overridden_rate": _source_macro_mean(
            choice[base_success] != 0, sources[base_success]
        ),
        "raw_decisions": int(len(choice)),
        "raw_task_successes": int(np.sum(selected_outcome == "task_success")),
        "raw_catastrophes": int(np.sum(selected_outcome == "catastrophe")),
        "raw_interventions": int(np.sum(choice != 0)),
        "raw_harmful_interventions": int(np.sum(harmful)),
        "raw_strict_base_correct": int(np.sum((labels == "strict_base") & (choice == 0))),
        "raw_strict_base_total": int(np.sum(labels == "strict_base")),
        "raw_strict_detour_correct": int(np.sum((labels == "strict_detour") & (choice == 1))),
        "raw_strict_detour_total": int(np.sum(labels == "strict_detour")),
        "raw_strict_retreat_correct": int(np.sum((labels == "strict_retreat") & (choice == 2))),
        "raw_strict_retreat_total": int(np.sum(labels == "strict_retreat")),
    }
    return row


def _per_source_rows(
    methods: Sequence[str],
    choices: np.ndarray,
    outcomes: np.ndarray,
    utility: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(methods):
        choice = choices[method_index]
        for source in sorted(set(sources.tolist())):
            mask = sources == source
            index = np.flatnonzero(mask)
            selected_outcome = outcomes[index, choice[mask]]
            selected_utility = utility[index, choice[mask]]
            source_labels = labels[mask]
            row = {
                "method": method,
                "source": source,
                "decisions": int(mask.sum()),
                "utility": float(np.mean(selected_utility)),
                "task_success_rate": float(np.mean(selected_outcome == "task_success")),
                "catastrophe_rate": float(np.mean(selected_outcome == "catastrophe")),
                "intervention_rate": float(np.mean(choice[mask] != 0)),
            }
            for label, option in STRICT_LABEL_TO_OPTION.items():
                class_mask = source_labels == label
                row[f"{label}_recall"] = (
                    float(np.mean(choice[mask][class_mask] == option))
                    if np.any(class_mask) else None
                )
            rows.append(row)
    return rows


def _paired_rows(
    per_source: Sequence[Mapping[str, Any]], comparators: Sequence[str]
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["method"]), str(row["source"])): row for row in per_source
    }
    methods = sorted({str(row["method"]) for row in per_source})
    sources = sorted({str(row["source"]) for row in per_source})
    rows = []
    for comparator in comparators:
        if comparator not in methods:
            continue
        for method in methods:
            if method == comparator:
                continue
            for source in sources:
                left = lookup[(method, source)]
                right = lookup[(comparator, source)]
                rows.append({
                    "method": method,
                    "comparator": comparator,
                    "source": source,
                    "utility_difference": float(left["utility"]) - float(right["utility"]),
                    "task_success_rate_difference": float(left["task_success_rate"]) - float(right["task_success_rate"]),
                    "catastrophe_rate_difference": float(left["catastrophe_rate"]) - float(right["catastrophe_rate"]),
                    "intervention_rate_difference": float(left["intervention_rate"]) - float(right["intervention_rate"]),
                })
    return rows


def _interval(values: np.ndarray) -> list[float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def inference_summary(
    *,
    per_source: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    comparator: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    bootstrap = config["statistics"]["bootstrap"]
    replicates = int(bootstrap["replicates"])
    seed = int(bootstrap["seed"])
    sources = sorted({str(row["source"]) for row in per_source})
    lookup = {
        (str(row["method"]), str(row["source"])): row for row in per_source
    }
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sources), size=(replicates, len(sources)))
    metrics = ("utility", "task_success_rate", "catastrophe_rate", "intervention_rate")
    result: dict[str, Any] = {}
    exact_config = config["statistics"]["exact_test"]
    for method_index, method in enumerate(methods):
        if method == comparator:
            continue
        metric_results = {}
        for metric_index, metric in enumerate(metrics):
            differences = np.asarray([
                float(lookup[(method, source)][metric])
                - float(lookup[(comparator, source)][metric])
                for source in sources
            ])
            boot = np.mean(differences[draws], axis=1)
            oriented = differences if metric in ("utility", "task_success_rate") else -differences
            exact = _sign_flip(
                oriented,
                seed + 1009 * (method_index + 1) + metric_index,
                max_exact_sources=int(exact_config["maximum_exact_sources"]),
                monte_carlo_replicates=int(exact_config["monte_carlo_replicates"]),
            )
            metric_results[metric] = {
                "difference_definition": f"{method} minus {comparator}",
                "source_macro_difference": float(np.mean(differences)),
                "source_bootstrap_95_ci": _interval(boot),
                "nonnegative_sources": int(np.sum(differences >= -1e-15)),
                "positive_sources": int(np.sum(differences > 1e-15)),
                "negative_sources": int(np.sum(differences < -1e-15)),
                "oriented_paired_sign_flip": exact,
            }
        result[method] = metric_results
    return {
        "schema_version": 1,
        "independent_unit": "source",
        "sources": sources,
        "comparator": comparator,
        "shared_source_bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "shared_draw_indices": True,
        },
        "comparisons": result,
    }


def decide_gate(
    *,
    metrics: Sequence[Mapping[str, Any]],
    per_source: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["gate"]
    by_method = {str(row["method"]): row for row in metrics}
    risk_name = str(gate["risk_reference"])
    risk = by_method[risk_name]
    source_lookup = {
        (str(row["method"]), str(row["source"])): row for row in per_source
    }
    sources = sorted({str(row["source"]) for row in per_source})
    go = gate["go_signal"]

    def core(method: str) -> dict[str, Any]:
        row = by_method[method]
        nonnegative = sum(
            float(source_lookup[(method, source)]["utility"])
            - float(source_lookup[(risk_name, source)]["utility"]) >= -1e-15
            for source in sources
        )
        criteria = {
            "strict_detour_recall": float(row["strict_detour_recall"]) >= float(go["strict_detour_recall_min"]),
            "strict_retreat_recall": float(row["strict_retreat_recall"]) >= float(go["strict_retreat_recall_min"]),
            "strict_base_recall": float(row["strict_base_recall"]) >= float(go["strict_base_recall_min"]),
            "utility_gain_over_risk": float(row["source_macro_utility"]) - float(risk["source_macro_utility"]) >= float(go["utility_gain_over_risk_min"]),
            "intervention_rate_vs_risk": float(row["intervention_rate"]) - float(risk["intervention_rate"]) <= float(go["intervention_rate_increase_over_risk_max"]),
            "catastrophe_rate_vs_risk": float(row["catastrophe_rate"]) - float(risk["catastrophe_rate"]) <= float(go["catastrophe_rate_increase_over_risk_max"]),
            "nonnegative_paired_sources": nonnegative >= int(go["nonnegative_paired_sources_min"]),
        }
        return {
            "method": method,
            "criteria": criteria,
            "passes_core": all(criteria.values()),
            "actual": {
                "strict_detour_recall": row["strict_detour_recall"],
                "strict_retreat_recall": row["strict_retreat_recall"],
                "strict_base_recall": row["strict_base_recall"],
                "utility_gain_over_risk": float(row["source_macro_utility"]) - float(risk["source_macro_utility"]),
                "intervention_rate_increase_over_risk": float(row["intervention_rate"]) - float(risk["intervention_rate"]),
                "catastrophe_rate_increase_over_risk": float(row["catastrophe_rate"]) - float(risk["catastrophe_rate"]),
                "nonnegative_paired_sources": nonnegative,
            },
        }

    outcome_checks = []
    for method in gate["outcome_candidates"]:
        check = core(str(method))
        comparator_checks = []
        for comparator in gate["support_balanced_comparators"]:
            left = by_method[str(method)]
            right = by_method[str(comparator)]
            utility_gain = float(left["source_macro_utility"]) - float(right["source_macro_utility"])
            f1_gain = float(left["strict_choice_macro_f1"]) - float(right["strict_choice_macro_f1"])
            passes = (
                utility_gain >= float(go["support_comparator_utility_gain_min"])
                or (
                    f1_gain >= float(go["support_comparator_macro_f1_gain_min"])
                    and utility_gain >= -float(go["support_comparator_utility_drop_max"])
                )
            )
            comparator_checks.append({
                "comparator": comparator,
                "utility_gain": utility_gain,
                "strict_choice_macro_f1_gain": f1_gain,
                "passes": passes,
            })
        check["support_comparator_checks"] = comparator_checks
        check["passes"] = bool(check["passes_core"] and any(
            row["passes"] for row in comparator_checks
        ))
        outcome_checks.append(check)

    value_checks = []
    for method in gate["value_candidates"]:
        check = core(str(method))
        check["passes"] = bool(check["passes_core"])
        value_checks.append(check)

    stop = gate["stop_rescue"]
    learned = [str(name) for name in gate["learned_selectors"]]
    stop_reasons = []
    if float(risk["oracle_value_recovered"]) >= float(
        stop["risk_best_fixed_oracle_value_recovered_min"]
    ):
        stop_reasons.append("Risk->BestFixed recovers at least 0.85 of source-macro Oracle value")
    best_learned_retreat = max(
        float(by_method[name]["strict_retreat_recall"]) for name in learned
    )
    if best_learned_retreat <= float(stop["strict_retreat_random_level"]):
        stop_reasons.append("no learned selector exceeds the frozen strict-Retreat random level")

    selected_method = None
    if stop_reasons:
        decision = "STOP-RESCUE"
    else:
        passing_outcomes = [row for row in outcome_checks if row["passes"]]
        passing_values = [row for row in value_checks if row["passes"]]
        if passing_outcomes:
            selected_method = max(
                passing_outcomes,
                key=lambda item: (
                    float(by_method[item["method"]]["source_macro_utility"]),
                    float(by_method[item["method"]]["strict_choice_macro_f1"]),
                ),
            )["method"]
            decision = "GO-SIGNAL"
        elif passing_values:
            selected_method = max(
                passing_values,
                key=lambda item: (
                    float(by_method[item["method"]]["source_macro_utility"]),
                    float(by_method[item["method"]]["strict_choice_macro_f1"]),
                ),
            )["method"]
            decision = "GO-VALUE-ONLY"
        else:
            pivot = gate["benchmark_pivot"]
            learned_threshold = float(pivot["learned_detour_retreat_recall_min"])
            diagnostic_threshold = float(pivot["diagnostic_detour_retreat_recall_min"])
            any_learned_joint = any(
                float(by_method[name]["strict_detour_recall"]) >= learned_threshold
                and float(by_method[name]["strict_retreat_recall"]) >= learned_threshold
                for name in learned
            )
            distinguishing_diagnostics = [
                str(name) for name in gate["diagnostics"]
                if float(by_method[str(name)]["strict_detour_recall"]) >= diagnostic_threshold
                and float(by_method[str(name)]["strict_retreat_recall"]) >= diagnostic_threshold
            ]
            decision = (
                "BENCHMARK-PIVOT"
                if (not any_learned_joint and distinguishing_diagnostics)
                else "INCONCLUSIVE"
            )

    screen_authorized = decision in {
        "GO-SIGNAL", "GO-VALUE-ONLY", "BENCHMARK-PIVOT"
    }
    return {
        "schema_version": 1,
        "phase": "2.5B",
        "decision": decision,
        "selected_method": selected_method,
        "risk_reference": risk_name,
        "risk_reference_oracle_value_recovered": risk["oracle_value_recovered"],
        "best_learned_strict_retreat_recall": best_learned_retreat,
        "stop_reasons": stop_reasons,
        "outcome_candidate_checks": outcome_checks,
        "value_candidate_checks": value_checks,
        "thresholds": gate,
        "authorization": {
            "screen_a_mechanical_authoring": screen_authorized,
            "screen_a_scope": (
                "benchmark_authoring_only" if decision == "BENCHMARK-PIVOT"
                else ("method_and_benchmark_authoring" if screen_authorized else "none")
            ),
            "freeze_candidate_architecture": decision in {"GO-SIGNAL", "GO-VALUE-ONLY"},
            "new_confirmatory_rollout": False,
            "fresh_test_outcomes": False,
        },
        "interpretation": (
            "INCONCLUSIVE is fail-closed and does not authorize Screen A. "
            "No Phase 2.5B branch authorizes confirmatory rollout."
        ),
    }


def analyze(input_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "iclr27_support_crossfit_suite":
        raise ValueError("not a Phase 2.5B support-crossfit result")
    prediction_path = input_dir / "all_predictions.npz"
    if manifest["artifact_sha256"].get("all_predictions.npz") != _sha256(prediction_path):
        raise ValueError("all_predictions.npz hash mismatch")
    config = yaml.safe_load((input_dir / str(manifest["config_snapshot"])).read_text())
    with np.load(prediction_path) as archive:
        methods = [str(value) for value in archive["method_names"]]
        choices = np.asarray(archive["choices"], dtype=np.int64)
        decision_id = np.asarray(archive["decision_id"], dtype=str)
        sources = np.asarray(archive["source"], dtype=str)
        split = np.asarray(archive["historical_split"], dtype=str)
        conditions = np.asarray(archive["condition"], dtype=str)
        horizon = np.asarray(archive["horizon"], dtype=np.int64)
        outcomes = np.asarray(archive["outcomes"], dtype=str)
        labels = np.asarray(archive["strict_label"], dtype=str)
    if choices.shape != (len(methods), len(sources)):
        raise ValueError("OOF prediction dimensions disagree")
    if np.any((choices < 0) | (choices >= len(OPTIONS))):
        raise ValueError("invalid or incomplete OOF choices")
    utility = realized_utility(
        outcomes,
        float(config["utility"]["lambda"]),
        float(config["utility"]["eta"]),
    )

    overall = [
        metric_row(
            method=method, choice=choices[index], outcomes=outcomes,
            utility=utility, labels=labels, sources=sources, conditions=conditions,
        )
        for index, method in enumerate(methods)
    ]
    per_source = _per_source_rows(
        methods, choices, outcomes, utility, labels, sources
    )
    comparators = [
        str(config["gate"]["risk_reference"]),
        *map(str, config["gate"]["support_balanced_comparators"]),
    ]
    paired = _paired_rows(per_source, comparators)
    inference = inference_summary(
        per_source=per_source,
        methods=methods,
        comparator=str(config["gate"]["risk_reference"]),
        config=config,
    )
    gate = decide_gate(
        metrics=overall, per_source=per_source, config=config
    )
    choice_rows = []
    for method_index, method in enumerate(methods):
        choice = choices[method_index]
        for index in range(len(sources)):
            choice_rows.append({
                "decision_id": decision_id[index],
                "source": sources[index],
                "historical_split": split[index],
                "condition": conditions[index],
                "horizon": int(horizon[index]),
                "strict_label": labels[index],
                "method": method,
                "choice": OPTIONS[int(choice[index])],
                "selected_outcome": outcomes[index, choice[index]],
                "selected_utility": utility[index, choice[index]],
            })

    _write_csv(input_dir / "overall_metrics.csv", overall)
    _write_csv(input_dir / "per_source_metrics.csv", per_source)
    _write_csv(input_dir / "paired_differences.csv", paired)
    _write_csv(input_dir / "all_choices.csv", choice_rows)
    (input_dir / "inference.json").write_text(
        json.dumps(_native(inference), indent=2, sort_keys=True) + "\n"
    )
    (input_dir / "gate_decision.json").write_text(
        json.dumps(_native(gate), indent=2, sort_keys=True) + "\n"
    )
    analysis_files = (
        "overall_metrics.csv", "per_source_metrics.csv", "paired_differences.csv",
        "all_choices.csv", "inference.json", "gate_decision.json",
    )
    manifest["gate_decision"] = gate["decision"]
    manifest["selected_method"] = gate["selected_method"]
    manifest["artifact_sha256"].update({
        name: _sha256(input_dir / name) for name in analysis_files
    })
    provenance_path = input_dir / "slurm_provenance.txt"
    if provenance_path.is_file():
        manifest["artifact_sha256"][provenance_path.name] = _sha256(provenance_path)
    manifest_path.write_text(
        json.dumps(_native(manifest), indent=2, sort_keys=True) + "\n"
    )
    return {"gate": gate, "overall": overall}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.input_dir)
    print(json.dumps({
        "input_dir": str(args.input_dir.resolve()),
        "gate_decision": result["gate"]["decision"],
        "selected_method": result["gate"]["selected_method"],
        "screen_a_authorized": result["gate"]["authorization"]["screen_a_mechanical_authoring"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
