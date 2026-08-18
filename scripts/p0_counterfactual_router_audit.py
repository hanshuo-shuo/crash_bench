#!/usr/bin/env python3
"""P0 survival audit for the fresh counterfactual-router result.

This is deliberately a CPU-only analysis of already-collected exact-state
branches.  It answers the immediate paper question: do simple geometry/risk
gates explain the router result before we spend more GPU time?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from crashbench.counterfactual_router import (
    OPTIONS,
    OUTCOMES,
    FrozenOutcomeRouter,
    conservative_option_choice,
    option_utilities,
)
from scripts.analyze_fresh_counterfactual_router import (
    combine_fresh_captures,
    oracle_choice,
)
from scripts.train_minimal_counterfactual_router import build_features, load_capture


PAPER_LAMBDA = 1.0
PAPER_TARGET_RATE = 0.6
PREVALENCES = (0.01, 0.05, 0.10, 0.25, 0.50)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utility(outcomes: np.ndarray, catastrophe_cost: float) -> np.ndarray:
    return np.where(
        outcomes == "task_success", 1.0,
        np.where(outcomes == "catastrophe", -float(catastrophe_cost), 0.0),
    ).astype(np.float64)


def _source_weights(sources: np.ndarray) -> np.ndarray:
    counts = Counter(map(str, sources))
    weights = np.asarray([1.0 / counts[str(source)] for source in sources])
    return weights * (len(weights) / weights.sum())


def evaluate_choices(
    choice: np.ndarray,
    outcomes: np.ndarray,
    sources: np.ndarray,
    catastrophe_cost: float,
) -> dict[str, Any]:
    choice = np.asarray(choice, dtype=np.int64)
    rows = np.arange(len(choice))
    utility = _utility(outcomes, catastrophe_cost)
    selected = outcomes[rows, choice]
    selected_utility = utility[rows, choice]
    oracle_utility = utility.max(axis=1)
    base_utility = utility[:, 0]
    denominator = float(oracle_utility.mean() - base_utility.mean())
    outcome_counts = Counter(map(str, selected))
    source_values = [
        float(selected_utility[sources == source].mean())
        for source in sorted(set(map(str, sources)))
    ]
    return {
        "n_decisions": int(len(choice)),
        "n_source_states": int(len(set(map(str, sources)))),
        "choice_counts": {
            option: int(np.sum(choice == index))
            for index, option in enumerate(OPTIONS)
        },
        "task_success_rate": outcome_counts["task_success"] / len(choice),
        "catastrophe_rate": outcome_counts["catastrophe"] / len(choice),
        "safe_noncompletion_rate": outcome_counts["safe_noncompletion"] / len(choice),
        "intervention_rate": float(np.mean(choice != 0)),
        "mean_utility": float(selected_utility.mean()),
        "source_macro_mean_utility": float(np.mean(source_values)),
        "mean_oracle_regret": float(np.mean(oracle_utility - selected_utility)),
        "oracle_value_recovered": (
            float((selected_utility.mean() - base_utility.mean()) / denominator)
            if denominator > 1e-12 else 0.0
        ),
    }


def candidate_thresholds(scores: np.ndarray) -> list[float]:
    values = sorted(set(map(float, scores)))
    return [float("-inf"), *(
        (left + right) / 2.0 for left, right in zip(values, values[1:])
    ), float("inf")]


def calibrate_fixed_fallback(
    scores: np.ndarray,
    outcomes: np.ndarray,
    sources: np.ndarray,
    catastrophe_cost: float,
    fallback_options: tuple[int, ...] = (1, 2),
) -> dict[str, Any]:
    utility = _utility(outcomes, catastrophe_cost)
    weights = _source_weights(sources)
    rows = np.arange(len(scores))
    candidates = []
    for option in fallback_options:
        for threshold in candidate_thresholds(scores):
            choice = np.where(scores > threshold, option, 0)
            value = float(np.average(utility[rows, choice], weights=weights))
            rate = float(np.average(choice != 0, weights=weights))
            candidates.append((value, -rate, -option, threshold, option, rate))
    best = max(candidates, key=lambda row: row[:3])
    return {
        "threshold": float(best[3]),
        "fallback_option": OPTIONS[int(best[4])],
        "fallback_option_index": int(best[4]),
        "calibration_source_balanced_utility": float(best[0]),
        "calibration_intervention_rate": float(best[5]),
    }


def calibrate_two_threshold(
    scores: np.ndarray,
    outcomes: np.ndarray,
    sources: np.ndarray,
    catastrophe_cost: float,
) -> dict[str, Any]:
    """Calibrate Base/Detour/Retreat thresholds on source-balanced utility."""

    utility = _utility(outcomes, catastrophe_cost)
    weights = _source_weights(sources)
    rows = np.arange(len(scores))
    thresholds = candidate_thresholds(scores)
    candidates = []
    for low_index, low in enumerate(thresholds):
        for high in thresholds[low_index:]:
            choice = np.where(scores > high, 2, np.where(scores > low, 1, 0))
            value = float(np.average(utility[rows, choice], weights=weights))
            rate = float(np.average(choice != 0, weights=weights))
            retreat_rate = float(np.average(choice == 2, weights=weights))
            candidates.append((value, -rate, -retreat_rate, low, high, rate, retreat_rate))
    best = max(candidates, key=lambda row: row[:3])
    return {
        "base_to_detour_threshold": float(best[3]),
        "detour_to_retreat_threshold": float(best[4]),
        "calibration_source_balanced_utility": float(best[0]),
        "calibration_intervention_rate": float(best[5]),
        "calibration_retreat_rate": float(best[6]),
    }


def two_threshold_choice(scores: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    low = float(calibration["base_to_detour_threshold"])
    high = float(calibration["detour_to_retreat_threshold"])
    return np.where(scores > high, 2, np.where(scores > low, 1, 0)).astype(np.int64)


def _router_training_predictions(
    router: FrozenOutcomeRouter,
    training_capture: Path,
    catastrophe_cost: float,
) -> dict[str, Any]:
    data = load_capture(training_capture, catastrophe_cost)
    features = build_features(
        data["arrays"], (router.pca_mean, router.pca_components),
        history=False, include_robot_action=True,
    )
    standardized = (features - router.feature_mean) / router.feature_scale
    design = np.column_stack((standardized, np.ones(len(features))))
    logits = np.einsum("nd,dok->nok", design, router.outcome_coef)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    risk_logits = design @ router.risk_coef
    risk_scores = 1.0 / (1.0 + np.exp(-np.clip(risk_logits, -40, 40)))
    data["probabilities"] = probabilities
    data["risk_scores"] = risk_scores
    return data


def _load_geometry_choices(
    decisions: list[dict[str, Any]],
    placements_path: Path | None,
    source_traces_path: Path | None,
    surface_margin_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if placements_path is None or source_traces_path is None:
        choice = np.asarray([
            1 if row["condition"] == "glass" else 0 for row in decisions
        ], dtype=np.int64)
        return choice, {
            "mode": "condition_label_proxy",
            "rule": "Detour iff condition == glass",
            "surface_margin_m": None,
            "rows": [],
        }

    placement_payload = json.loads(placements_path.read_text())
    placements = {row["placement_id"]: row for row in placement_payload["placements"]}
    trace_payload = json.loads(source_traces_path.read_text())
    traces = {int(row["source_state_index"]): row for row in trace_payload["traces"]}
    rows = []
    choices = []
    for decision in decisions:
        condition = str(decision["condition"])
        if condition == "noglass":
            choices.append(0)
            rows.append({
                "decision_id": decision["decision_id"], "condition": condition,
                "surface_clearance_m": None, "intersects": False,
            })
            continue
        placement = placements[str(decision["placement_id"])]
        trace = traces[int(placement["metadata"]["source_state_index"])]
        eef = np.load(source_traces_path.parent / trace["eef_xyz_path"], allow_pickle=False)
        glass = placement[
            "on_path_glass" if condition == "glass" else "off_path_glass"
        ]
        center = np.asarray(glass["pos"][:2], dtype=np.float64)
        center_distance = float(np.linalg.norm(eef[:, :2] - center, axis=1).min())
        surface_clearance = center_distance - float(glass["size"][0])
        intersects = surface_clearance <= float(surface_margin_m)
        choices.append(1 if intersects else 0)
        rows.append({
            "decision_id": decision["decision_id"], "condition": condition,
            "centerline_distance_m": center_distance,
            "glass_radius_m": float(glass["size"][0]),
            "surface_clearance_m": surface_clearance,
            "intersects": bool(intersects),
        })
    return np.asarray(choices, dtype=np.int64), {
        "mode": "nominal_eef_swept_path_surface_clearance",
        "rule": "Detour iff min nominal EEF-to-glass surface clearance <= margin",
        "surface_margin_m": float(surface_margin_m),
        "placements": str(placements_path),
        "source_traces": str(source_traces_path),
        "rows": rows,
    }


def _reliability(probability: np.ndarray, target: np.ndarray, bins: int = 5) -> dict[str, Any]:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        member = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1
            else probability < edges[index + 1]
        )
        if not np.any(member):
            continue
        confidence = float(probability[member].mean())
        frequency = float(target[member].mean())
        weight = float(member.mean())
        ece += weight * abs(confidence - frequency)
        rows.append({
            "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "count": int(member.sum()), "mean_probability": confidence,
            "empirical_frequency": frequency,
        })
    return {"ece": float(ece), "bins": rows}


def _probability_metrics(probabilities: np.ndarray, outcomes: np.ndarray) -> dict[str, Any]:
    label_index = {name: index for index, name in enumerate(OUTCOMES)}
    labels = np.asarray([label_index[str(value)] for value in outcomes], dtype=np.int64)
    one_hot = np.eye(len(OUTCOMES))[labels]
    rows = np.arange(len(labels))
    return {
        "n": int(len(labels)),
        "nll": float(-np.log(probabilities[rows, labels] + 1e-12).mean()),
        "multiclass_brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "catastrophe_reliability": _reliability(
            probabilities[:, 1], labels == 1,
        ),
    }


def calibration_audit(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    router_choice: np.ndarray,
    catastrophe_cost: float,
) -> dict[str, Any]:
    per_option = {
        option: _probability_metrics(probabilities[:, index], outcomes[:, index])
        for index, option in enumerate(OPTIONS)
    }
    rows = np.arange(len(router_choice))
    selected = _probability_metrics(
        probabilities[rows, router_choice], outcomes[rows, router_choice]
    )
    flat_labels = outcomes.reshape(-1) == "catastrophe"
    catastrophe = _reliability(probabilities[:, :, 1].reshape(-1), flat_labels)

    predicted_utility = option_utilities(probabilities, catastrophe_cost)
    intervention = 1 + np.argmax(predicted_utility[:, 1:], axis=1)
    margin = (
        predicted_utility[rows, intervention] - predicted_utility[:, 0]
    )
    realized = _utility(outcomes, catastrophe_cost)
    realized_advantage = realized[rows, intervention] - realized[:, 0]
    correlation = None
    if np.std(margin) > 0 and np.std(realized_advantage) > 0:
        correlation = float(np.corrcoef(margin, realized_advantage)[0, 1])
    order = np.argsort(margin)
    margin_bins = []
    for indices in np.array_split(order, min(4, len(order))):
        margin_bins.append({
            "n": int(len(indices)),
            "predicted_margin_mean": float(margin[indices].mean()),
            "realized_advantage_mean": float(realized_advantage[indices].mean()),
            "beneficial_fraction": float(np.mean(realized_advantage[indices] > 0)),
        })
    return {
        "per_option": per_option,
        "selected_option": selected,
        "all_option_catastrophe_reliability": catastrophe,
        "predicted_margin_vs_realized_advantage": {
            "pearson_correlation": correlation,
            "bins": margin_bins,
        },
    }


def option_choice_audit(
    choice: np.ndarray,
    oracle: np.ndarray,
    data: dict[str, Any],
    catastrophe_cost: float,
) -> dict[str, Any]:
    confusion = {
        oracle_name: {
            router_name: int(np.sum((oracle == oi) & (choice == ri)))
            for ri, router_name in enumerate(OPTIONS)
        }
        for oi, oracle_name in enumerate(OPTIONS)
    }
    by_condition = {}
    for condition in ("glass", "offpath", "noglass"):
        member = data["conditions"] == condition
        by_condition[condition] = {
            option: int(np.sum(choice[member] == index))
            for index, option in enumerate(OPTIONS)
        }

    utility = _utility(data["outcomes"], catastrophe_cost)
    signatures: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(data["outcomes"]):
        signatures[" / ".join(map(str, row))].append(index)
    signature_rows = []
    for signature, indices_list in sorted(signatures.items()):
        indices = np.asarray(indices_list, dtype=np.int64)
        selected_regret = utility[indices].max(axis=1) - utility[indices, choice[indices]]
        signature_rows.append({
            "signature": signature,
            "n": int(len(indices)),
            "router_choice_counts": {
                option: int(np.sum(choice[indices] == oi))
                for oi, option in enumerate(OPTIONS)
            },
            "oracle_choice_counts": {
                option: int(np.sum(oracle[indices] == oi))
                for oi, option in enumerate(OPTIONS)
            },
            "router_mean_regret": float(selected_regret.mean()),
        })
    return {
        "overall_choice_counts": {
            option: int(np.sum(choice == index))
            for index, option in enumerate(OPTIONS)
        },
        "by_condition": by_condition,
        "router_choice_vs_realized_oracle_choice": confusion,
        "by_realized_outcome_signature": signature_rows,
    }


def router_choice_grid(
    probabilities: np.ndarray,
    conditions: np.ndarray,
    router_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    calibration = router_manifest["calibration"]
    for catastrophe_cost in calibration["lambdas"]:
        for target_rate in calibration["target_intervention_rates"]:
            point = calibration["router_frontier"][
                f"lambda_{float(catastrophe_cost):g}"
            ][f"target_{float(target_rate):.1f}"]
            choice = conservative_option_choice(
                probabilities, catastrophe_cost=float(catastrophe_cost),
                intervention_margin=float(point["delta"]),
            )
            rows.append({
                "lambda": float(catastrophe_cost),
                "target_intervention_rate": float(target_rate),
                "delta": float(point["delta"]),
                "choice_counts": {
                    option: int(np.sum(choice == index))
                    for index, option in enumerate(OPTIONS)
                },
                "choice_counts_by_condition": {
                    condition: {
                        option: int(np.sum(choice[conditions == condition] == index))
                        for index, option in enumerate(OPTIONS)
                    }
                    for condition in ("glass", "offpath", "noglass")
                },
            })
    return rows


def prevalence_reweighting(
    methods: dict[str, np.ndarray],
    data: dict[str, Any],
    catastrophe_cost: float,
) -> dict[str, Any]:
    hazard = data["conditions"] == "glass"
    control = ~hazard
    result = {}
    for name, choice in methods.items():
        hazard_metrics = evaluate_choices(
            choice[hazard], data["outcomes"][hazard], data["sources"][hazard],
            catastrophe_cost,
        )
        control_metrics = evaluate_choices(
            choice[control], data["outcomes"][control], data["sources"][control],
            catastrophe_cost,
        )
        rows = []
        for prevalence in PREVALENCES:
            row = {"hazard_prevalence": prevalence}
            for metric in (
                "task_success_rate", "catastrophe_rate",
                "safe_noncompletion_rate", "intervention_rate",
            ):
                row[metric] = float(
                    prevalence * hazard_metrics[metric]
                    + (1.0 - prevalence) * control_metrics[metric]
                )
            rows.append(row)
        result[name] = {
            "hazard": hazard_metrics,
            "control": control_metrics,
            "reweighted": rows,
        }
    return result


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    weak = (
        left["task_success_rate"] >= right["task_success_rate"] - 1e-12
        and left["catastrophe_rate"] <= right["catastrophe_rate"] + 1e-12
        and left["intervention_rate"] <= right["intervention_rate"] + 1e-12
    )
    strict = (
        left["task_success_rate"] > right["task_success_rate"] + 1e-12
        or left["catastrophe_rate"] < right["catastrophe_rate"] - 1e-12
        or left["intervention_rate"] < right["intervention_rate"] - 1e-12
    )
    return bool(weak and strict)


def _write_report(result: dict[str, Any], output: Path) -> None:
    rows = result["main_table"]
    verdict = result["decision"]["verdict"]
    lines = [
        "# P0 counterfactual-router survival audit",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"Display point: `lambda={result['protocol']['catastrophe_cost']:g}`, "
        f"`target={result['protocol']['target_intervention_rate']:.1f}`; "
        f"{result['capture']['n_source_states']} source states / "
        f"{result['capture']['n_decisions']} matched decisions.",
        "",
        "| Method | Success | Catastrophe | Safe noncompletion | Intervention | Choices B/D/R |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in rows.items():
        counts = metrics["choice_counts"]
        lines.append(
            f"| {name} | {metrics['task_success_rate']:.2%} | "
            f"{metrics['catastrophe_rate']:.2%} | "
            f"{metrics['safe_noncompletion_rate']:.2%} | "
            f"{metrics['intervention_rate']:.2%} | "
            f"{counts['base_continue']}/{counts['detour_complete']}/{counts['retreat_hold']} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        result["decision"]["plain_language"],
        "",
        "Geometry evidence mode: `" + result["geometry_gate"]["mode"] + "`.",
        "",
        "## Evidence files",
        "",
        "- `audit.json`: full choice matrices, calibration, thresholds, and prevalence audit.",
        "- `main_table.csv`: compact method comparison.",
        "",
        "Direct classifier/advantage-regression fresh evaluation is not identifiable from "
        "the saved fresh artifact because raw fresh features were not retained. The geometry "
        "and risk baselines above require no new rollout.",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    capture_roots = [args.capture.resolve(), *(
        path.resolve() for path in args.additional_capture
    )]
    data = combine_fresh_captures(capture_roots)
    router_path = args.router_model.resolve()
    router_manifest = json.loads(router_path.read_text())
    router = FrozenOutcomeRouter.load(router_path)
    catastrophe_cost = float(args.catastrophe_cost)
    target_rate = float(args.target_rate)
    n = len(data["outcomes"])
    rows = np.arange(n)

    point = router_manifest["calibration"]["router_frontier"][
        f"lambda_{catastrophe_cost:g}"
    ][f"target_{target_rate:.1f}"]
    router_choice = conservative_option_choice(
        data["probabilities"], catastrophe_cost=catastrophe_cost,
        intervention_margin=float(point["delta"]),
    )
    realized_oracle = oracle_choice(data["outcomes"], catastrophe_cost)
    risk_point = router_manifest["calibration"]["binary_risk_retreat_frontier"][
        f"target_{target_rate:.1f}"
    ]
    rate_threshold = float(risk_point["threshold"])

    training = _router_training_predictions(
        router, args.training_capture.resolve(), catastrophe_cost,
    )
    calibration = training["splits"] == "calibration"
    calibration_scores = training["risk_scores"][calibration]
    calibration_outcomes = training["outcomes"][calibration]
    calibration_sources = training["sources"][calibration]
    best_fixed = calibrate_fixed_fallback(
        calibration_scores, calibration_outcomes, calibration_sources,
        catastrophe_cost,
    )
    two_threshold = calibrate_two_threshold(
        calibration_scores, calibration_outcomes, calibration_sources,
        catastrophe_cost,
    )
    geometry_choice, geometry_evidence = _load_geometry_choices(
        data["decisions"], args.placements, args.source_traces,
        args.geometry_margin,
    )

    methods = {
        "Base": np.zeros(n, dtype=np.int64),
        "Always Detour": np.ones(n, dtype=np.int64),
        "Always Retreat": np.full(n, 2, dtype=np.int64),
        "Risk -> Retreat (rate-matched)": np.where(
            data["risk_scores"] > rate_threshold, 2, 0,
        ),
        "Risk -> Detour (rate-matched)": np.where(
            data["risk_scores"] > rate_threshold, 1, 0,
        ),
        "Risk -> best fixed option": np.where(
            data["risk_scores"] > float(best_fixed["threshold"]),
            int(best_fixed["fallback_option_index"]), 0,
        ),
        "Two-threshold Risk": two_threshold_choice(
            data["risk_scores"], two_threshold,
        ),
        "Geometry Gate": geometry_choice,
        "Counterfactual Router": router_choice,
        "Counterfactual Oracle": realized_oracle,
    }
    main_table = {
        name: evaluate_choices(
            choice, data["outcomes"], data["sources"], catastrophe_cost,
        )
        for name, choice in methods.items()
    }
    geometry_dominates = _dominates(
        main_table["Geometry Gate"], main_table["Counterfactual Router"]
    )
    verdict = (
        "NO-GO for expanding the current cohort"
        if geometry_dominates else "Geometry Gate does not dominate; continue remaining P0 baselines"
    )
    result = {
        "schema_version": 1,
        "kind": "p0_counterfactual_router_survival_audit",
        "protocol": {
            "catastrophe_cost": catastrophe_cost,
            "target_intervention_rate": target_rate,
            "geometry_surface_margin_m": float(args.geometry_margin),
            "geometry_margin_source": (
                "frozen Detour departure_clearance; not tuned on fresh outcomes"
            ),
        },
        "capture": {
            "roots": [str(path) for path in capture_roots],
            "n_source_states": int(len(set(map(str, data["sources"])))),
            "n_decisions": int(n),
            "artifact_sha256": {
                str(root / name): _sha256(root / name)
                for root in capture_roots
                for name in ("capture_manifest.json", "fresh_decisions.json", "option_rollouts.jsonl")
            },
        },
        "router": {
            "manifest": str(router_path),
            "manifest_sha256": _sha256(router_path),
            "delta": float(point["delta"]),
        },
        "main_table": main_table,
        "option_choice": option_choice_audit(
            router_choice, realized_oracle, data, catastrophe_cost,
        ),
        "option_choice_grid_by_lambda_target_condition": router_choice_grid(
            data["probabilities"], data["conditions"], router_manifest,
        ),
        "risk_baselines": {
            "rate_matched_threshold": rate_threshold,
            "best_fixed_calibration": best_fixed,
            "two_threshold_calibration": two_threshold,
        },
        "geometry_gate": geometry_evidence,
        "calibration": calibration_audit(
            data["probabilities"], data["outcomes"], router_choice,
            catastrophe_cost,
        ),
        "prevalence_reweighting": prevalence_reweighting(
            {
                "Geometry Gate": methods["Geometry Gate"],
                "Two-threshold Risk": methods["Two-threshold Risk"],
                "Counterfactual Router": methods["Counterfactual Router"],
                "Always Detour": methods["Always Detour"],
            },
            data, catastrophe_cost,
        ),
        "decision": {
            "geometry_gate_dominates_router": geometry_dominates,
            "verdict": verdict,
            "plain_language": (
                "The fixed nominal-path Geometry Gate is at least as good on success, "
                "catastrophe, and intervention, and strictly better on at least one metric. "
                "Do not enlarge this cohort yet; the present router advantage can be explained "
                "by a simpler geometry-triggered Detour policy."
                if geometry_dominates else
                "The fixed nominal-path Geometry Gate does not dominate the Router on this cohort."
            ),
        },
        "limitations": [
            "fresh raw hidden/robot/action features were not saved, so direct classifier and direct advantage regression cannot be evaluated fresh without recollection",
            "the dynamic first-crossing intervention time remains outside this exact T-20 branch audit",
        ],
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    with (output / "main_table.csv").open("w", newline="") as handle:
        fieldnames = [
            "method", "task_success_rate", "catastrophe_rate",
            "safe_noncompletion_rate", "intervention_rate", "base_choices",
            "detour_choices", "retreat_choices", "mean_oracle_regret",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for name, metrics in main_table.items():
            writer.writerow({
                "method": name,
                **{key: metrics[key] for key in fieldnames[1:5]},
                "base_choices": metrics["choice_counts"]["base_continue"],
                "detour_choices": metrics["choice_counts"]["detour_complete"],
                "retreat_choices": metrics["choice_counts"]["retreat_hold"],
                "mean_oracle_regret": metrics["mean_oracle_regret"],
            })
    _write_report(result, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--additional-capture", action="append", type=Path, default=[])
    parser.add_argument("--router-model", required=True, type=Path)
    parser.add_argument("--training-capture", required=True, type=Path)
    parser.add_argument("--placements", type=Path)
    parser.add_argument("--source-traces", type=Path)
    parser.add_argument("--geometry-margin", type=float, default=0.06)
    parser.add_argument("--catastrophe-cost", type=float, default=PAPER_LAMBDA)
    parser.add_argument("--target-rate", type=float, default=PAPER_TARGET_RATE)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "verdict": result["decision"]["verdict"],
        "geometry_gate": result["main_table"]["Geometry Gate"],
        "counterfactual_router": result["main_table"]["Counterfactual Router"],
        "two_threshold_risk": result["main_table"]["Two-threshold Risk"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
