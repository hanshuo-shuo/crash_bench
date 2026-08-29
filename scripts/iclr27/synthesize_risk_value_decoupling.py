#!/usr/bin/env python3
"""Build the frozen, model-free PIVOT-0 Risk Is Not Regret evidence package.

This script only reads reviewed artifacts.  It does not fit a model, launch a
simulator, select a new threshold, or reinterpret historical splits.  Every
input is pinned by SHA-256 and all scientific gates are checked before an
output directory is created.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.counterfactual_router import OPTIONS
from scripts.train_minimal_counterfactual_router import load_capture


OPTION_NAMES = ("Base", "Detour", "Retreat")
OPTION_INDEX = {name: index for index, name in enumerate(OPTIONS)}
STRICT_TO_INDEX = {
    "strict_base": 0,
    "strict_detour": 1,
    "strict_retreat": 2,
}
OUTCOMES = ("task_success", "catastrophe", "safe_noncompletion")
STORY_STATUSES = (
    "BROAD_BENCHMARK_PIVOT",
    "GLASS_SCOPED_BENCHMARK_PIVOT",
    "INSUFFICIENT_FOR_PIVOT",
)
REQUIRED_OUTPUTS = (
    "manifest.json",
    "risk_benefit_crosstab.csv",
    "source_support.csv",
    "family_support.csv",
    "risk_only_policy_regret.csv",
    "oracle_hybrid_waterfall.csv",
    "same_risk_different_decision_witnesses.csv",
    "sequential_realization_gap.csv",
    "story_decision.json",
    "figure_risk_benefit_flow.pdf",
    "figure_oracle_gap_waterfall.pdf",
    "figure_source_support.pdf",
)


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
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    materialized = [_native(dict(row)) for row in rows]
    if not materialized:
        raise ValueError(f"refusing to write an empty required table: {path.name}")
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("kind") != "iclr27_risk_value_decoupling":
        raise ValueError("unexpected risk-value synthesis config")
    labels = config["canonical_labels"]
    if float(labels["epsilon"]) != 0.0:
        raise ValueError("PIVOT-0 freezes the strict benefit epsilon at 0.0")
    if float(labels["equality_tolerance"]) != 1e-12:
        raise ValueError("only the frozen 1e-12 equality tolerance is allowed")
    if tuple(labels["option_order"]) != tuple(OPTIONS):
        raise ValueError("the frozen Base/Detour/Retreat option order changed")
    if tuple(config["outputs"]["required"]) != REQUIRED_OUTPUTS:
        raise ValueError("the exact twelve-artifact output contract changed")
    contract = config["data_contract"]
    if bool(contract["condition_labels_are_mechanical_families"]):
        raise ValueError("condition labels may not be promoted to mechanical families")
    if int(contract["condition_family_count"]) != 3:
        raise ValueError("the frozen condition-label count changed")
    if int(contract["independent_mechanical_family_count"]) != 1:
        raise ValueError("the current corpus contains one mechanical hazard design")
    if config["policies"]["oracle_benefit_best_fixed_option"] != "detour_complete":
        raise ValueError("Phase 2.5A globally freezes Detour as Best Fixed")
    return config


def canonical_labels(
    utility: np.ndarray,
    base_outcomes: Sequence[str],
    *,
    epsilon: float = 0.0,
    equality_tolerance: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Construct canonical R, B, O, A_int, A_DR and strict support labels."""

    utility = np.asarray(utility, dtype=np.float64)
    base_outcomes = np.asarray(base_outcomes, dtype=str)
    if utility.ndim != 2 or utility.shape[1] != 3 or len(base_outcomes) != len(utility):
        raise ValueError("canonical labels require aligned N x 3 utilities and outcomes")
    if float(epsilon) != 0.0:
        raise ValueError("benefit epsilon must remain exactly 0.0")
    if float(equality_tolerance) != 1e-12:
        raise ValueError("equality tolerance must remain exactly 1e-12")
    risk = base_outcomes == "catastrophe"
    intervention_advantage = np.max(utility[:, 1:], axis=1) - utility[:, 0]
    benefit = np.max(utility[:, 1:], axis=1) > utility[:, 0] + float(epsilon)
    detour_retreat_advantage = utility[:, 1] - utility[:, 2]
    optimal = np.empty(len(utility), dtype=np.int64)
    strict_labels: list[str] = []
    for index, row in enumerate(utility):
        tied = np.flatnonzero(np.isclose(
            row, row.max(), atol=float(equality_tolerance), rtol=0.0
        ))
        optimal[index] = int(tied[0])  # frozen Base > Detour > Retreat order
        if np.all(np.isclose(
            row, -1.0, atol=float(equality_tolerance), rtol=0.0
        )):
            strict_labels.append("no_good_option")
        elif len(tied) == 1:
            strict_labels.append(("strict_base", "strict_detour", "strict_retreat")[int(tied[0])])
        elif 0 in tied:
            strict_labels.append("base_tie")
        else:
            strict_labels.append("detour_retreat_tie")
    return {
        "risk": risk.astype(np.int64),
        "benefit": benefit.astype(np.int64),
        "optimal": optimal,
        "intervention_advantage": intervention_advantage,
        "detour_retreat_advantage": detour_retreat_advantage,
        "strict_label": np.asarray(strict_labels, dtype=str),
    }


def source_macro_mean(
    values: Sequence[float] | np.ndarray,
    sources: Sequence[str] | np.ndarray,
    denominator: Sequence[bool] | np.ndarray | None = None,
) -> tuple[float, int]:
    """Equal mean over source-level means, optionally within a denominator."""

    values = np.asarray(values, dtype=np.float64)
    sources = np.asarray(sources, dtype=str)
    if values.shape != (len(sources),):
        raise ValueError("source-macro values and sources must align")
    selected = np.ones(len(values), dtype=bool) if denominator is None else np.asarray(
        denominator, dtype=bool
    )
    if selected.shape != (len(values),):
        raise ValueError("source-macro denominator must align")
    means = [
        float(values[selected & (sources == source)].mean())
        for source in sorted(set(sources[selected].tolist()))
        if np.any(selected & (sources == source))
    ]
    return (float(np.mean(means)), len(means)) if means else (float("nan"), 0)


def recovered_fraction(value: float, risk_reference: float, oracle: float) -> float:
    opportunity = float(oracle) - float(risk_reference)
    if opportunity <= 0.0:
        raise ValueError("Oracle must strictly exceed Risk->BestFixed")
    return (float(value) - float(risk_reference)) / opportunity


def policy_metrics(
    choice: np.ndarray,
    outcomes: np.ndarray,
    utility: np.ndarray,
    sources: np.ndarray,
) -> dict[str, float | int]:
    choice = np.asarray(choice, dtype=np.int64)
    outcomes = np.asarray(outcomes)
    utility = np.asarray(utility, dtype=np.float64)
    sources = np.asarray(sources, dtype=str)
    if (
        choice.shape != (len(sources),)
        or outcomes.shape != utility.shape
        or utility.shape != (len(sources), 3)
    ):
        raise ValueError("policy evaluation arrays are not aligned")
    row = np.arange(len(choice))
    selected_outcome = outcomes[row, choice].astype(str)
    selected_utility = utility[row, choice]
    result: dict[str, float | int] = {"decisions": int(len(choice))}
    raw_values = {
        "success": selected_outcome == "task_success",
        "catastrophe": selected_outcome == "catastrophe",
        "safe_noncompletion": selected_outcome == "safe_noncompletion",
        "intervention": choice != 0,
        "utility": selected_utility,
    }
    for name, values in raw_values.items():
        result[f"raw_{name}_rate" if name != "utility" else "raw_utility"] = float(
            np.asarray(values, dtype=np.float64).mean()
        )
        macro, count = source_macro_mean(values, sources)
        result[
            f"source_macro_{name}_rate" if name != "utility" else "source_macro_utility"
        ] = macro
        result["source_macro_sources"] = count
    return result


def _verify_declared_files(
    root: Path, files: Mapping[str, str], prefix: str
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, expected in files.items():
        path = root / str(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != str(expected):
            raise ValueError(f"frozen input hash mismatch: {prefix}/{name}")
        hashes[f"{prefix}/{name}"] = actual
    return hashes


def _verify_folds(
    rows: Sequence[Mapping[str, str]], sources: Sequence[str], *, held_role: str
) -> dict[int, dict[str, Any]]:
    expected_sources = set(map(str, sources))
    grouped: dict[int, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["outer_fold"])].append(row)
    if sorted(grouped) != list(range(len(expected_sources))):
        raise ValueError("outer folds are not the frozen 20-fold enumeration")
    result: dict[int, dict[str, Any]] = {}
    for fold, local in sorted(grouped.items()):
        if len(local) != len(expected_sources):
            raise ValueError("outer fold does not contain every source exactly once")
        if {row["source"] for row in local} != expected_sources:
            raise ValueError("outer fold source set changed")
        held_names = {row["source"] for row in local if row["role"] == held_role}
        fit_names = {row["source"] for row in local if row["role"] == "fit"}
        calibration_names = {row["source"] for row in local if row["role"] == "calibration"}
        held_declared = {row["held_source"] for row in local}
        if len(held_names) != 1 or held_names != held_declared:
            raise ValueError("held source or held-source declaration changed")
        if len(fit_names) != 14 or len(calibration_names) != 5:
            raise ValueError("frozen 14-fit/5-calibration partition changed")
        if held_names & (fit_names | calibration_names) or fit_names & calibration_names:
            raise ValueError("outer fold roles overlap")
        result[fold] = {
            "held": next(iter(held_names)),
            "fit": tuple(sorted(fit_names)),
            "calibration": tuple(sorted(calibration_names)),
        }
    return result


def _validate_inputs(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    inputs = config["inputs"]
    hashes: dict[str, str] = {}
    roots: dict[str, Path] = {}
    for key in ("capture", "phase_2_5a", "phase_2_5b", "phase_2_5b_r"):
        spec = inputs[key]
        root = (ROOT / spec["path"]).resolve()
        roots[key] = root
        hashes.update(_verify_declared_files(root, spec["files"], key))
    sequential_paths: dict[str, Path] = {}
    for key, spec in inputs["sequential"].items():
        path = (ROOT / spec["path"]).resolve()
        if not path.is_file() or _sha256(path) != str(spec["sha256"]):
            raise ValueError(f"frozen sequential input hash mismatch: {key}")
        sequential_paths[key] = path
        hashes[f"sequential/{key}"] = str(spec["sha256"])

    capture_manifest = json.loads((roots["capture"] / "capture_manifest.json").read_text())
    if capture_manifest.get("kind") != inputs["capture"]["expected_kind"]:
        raise ValueError("unexpected frozen capture kind")
    # Reject forbidden outcome-bearing splits from metadata before opening the
    # option-outcome rows. The bytes are already hash-pinned, but this ordering
    # keeps the no-fresh/test contract explicit and fail-closed.
    preflight_metadata = json.loads(
        (roots["capture"] / "decision_metadata.json").read_text()
    )
    allowed_splits = set(map(str, config["data_contract"]["allowed_splits"]))
    observed_splits = {str(row["split"]) for row in preflight_metadata}
    if observed_splits != allowed_splits:
        raise ValueError(
            "capture metadata contains a forbidden split or misses a frozen split"
        )
    data = load_capture(roots["capture"], catastrophe_cost=1.0)
    metadata = data["metadata"]
    n = int(config["data_contract"]["expected_decisions"])
    if len(metadata) != n:
        raise ValueError("the frozen decision count changed")
    decision_ids = np.asarray([str(row["decision_id"]) for row in metadata], dtype=str)
    if len(set(decision_ids.tolist())) != n:
        raise ValueError("capture decision IDs are not unique")
    sources = np.asarray(data["sources"], dtype=str)
    unique_sources = sorted(set(sources.tolist()))
    if len(unique_sources) != int(config["data_contract"]["expected_sources"]):
        raise ValueError("the frozen source count changed")
    splits = np.asarray(data["splits"], dtype=str)
    allowed = set(map(str, config["data_contract"]["allowed_splits"]))
    if set(splits.tolist()) != allowed:
        raise ValueError("historical split labels changed or a forbidden split entered")
    split_source_counts: dict[str, int] = {}
    for source in unique_sources:
        local = set(splits[sources == source].tolist())
        if len(local) != 1:
            raise ValueError("a source crosses historical split labels")
    for split in sorted(allowed):
        split_source_counts[split] = len(set(sources[splits == split].tolist()))
    expected_split_counts = {
        str(key): int(value)
        for key, value in config["data_contract"]["expected_split_source_counts"].items()
    }
    if split_source_counts != expected_split_counts:
        raise ValueError("the frozen 5/7/8 source split changed")
    conditions = np.asarray(data["conditions"], dtype=str)
    if set(conditions.tolist()) != set(config["data_contract"]["expected_condition_labels"]):
        raise ValueError("the frozen condition labels changed")

    labels = canonical_labels(
        data["utility"], data["outcomes"][:, 0],
        epsilon=float(config["canonical_labels"]["epsilon"]),
        equality_tolerance=float(config["canonical_labels"]["equality_tolerance"]),
    )
    label_rows = _read_csv(roots["phase_2_5a"] / "decision_labels.csv")
    if len(label_rows) != n or {row["decision_id"] for row in label_rows} != set(decision_ids):
        raise ValueError("Phase 2.5A labels do not cover the frozen decisions")
    index = {decision_id: i for i, decision_id in enumerate(decision_ids.tolist())}
    labels_by_id = {row["decision_id"]: row for row in label_rows}
    for decision_id, row in labels_by_id.items():
        i = index[decision_id]
        expected_meta = metadata[i]
        if (
            row["source"] != str(expected_meta["source_state_sha256"])
            or row["split"] != str(expected_meta["split"])
            or row["condition"] != str(expected_meta["condition"])
            or int(row["horizon"]) != int(expected_meta["horizon_actions"])
        ):
            raise ValueError("Phase 2.5A label metadata changed")
        expected_outcomes = tuple(map(str, data["outcomes"][i]))
        observed_outcomes = (row["base_outcome"], row["detour_outcome"], row["retreat_outcome"])
        if observed_outcomes != expected_outcomes:
            raise ValueError("Phase 2.5A outcome labels changed")
        observed_utility = np.asarray([
            float(row["base_utility"]), float(row["detour_utility"]),
            float(row["retreat_utility"]),
        ])
        if not np.allclose(observed_utility, data["utility"][i], atol=1e-12, rtol=0.0):
            raise ValueError("Phase 2.5A utility labels changed")
        if row["strict_label"] != labels["strict_label"][i]:
            raise ValueError("Phase 2.5A strict support label changed")

    a_manifest = json.loads((roots["phase_2_5a"] / "manifest.json").read_text())
    a_gate = json.loads((roots["phase_2_5a"] / "gate_decision.json").read_text())
    required_a_gate = inputs["phase_2_5a"]["expected_gate"]
    if (
        a_manifest.get("kind") != inputs["phase_2_5a"]["expected_kind"]
        or a_manifest.get("gate_decision") != required_a_gate
        or a_gate.get("decision") != required_a_gate
    ):
        raise ValueError("Phase 2.5A PASS-SUPPORT provenance changed")
    b_manifest = json.loads((roots["phase_2_5b"] / "manifest.json").read_text())
    b_gate = json.loads((roots["phase_2_5b"] / "gate_decision.json").read_text())
    required_b_gate = inputs["phase_2_5b"]["expected_gate"]
    if (
        b_manifest.get("kind") != inputs["phase_2_5b"]["expected_kind"]
        or b_manifest.get("gate_decision") != required_b_gate
        or b_gate.get("decision") != required_b_gate
    ):
        raise ValueError("Phase 2.5B must remain frozen INCONCLUSIVE")
    r_manifest = json.loads((roots["phase_2_5b_r"] / "manifest.json").read_text())
    r_decision = json.loads((roots["phase_2_5b_r"] / "decision.json").read_text())
    required_r_decision = inputs["phase_2_5b_r"]["expected_decision"]
    if (
        r_manifest.get("kind") != inputs["phase_2_5b_r"]["expected_kind"]
        or r_manifest.get("primary_decision") != required_r_decision
        or r_decision.get("primary_decision") != required_r_decision
    ):
        raise ValueError("Phase 2.5B-R machine decision changed")
    if any(bool(value) for value in r_decision.get("method_pass", {}).values()):
        raise ValueError("a deployable ADR unexpectedly passes method_pass")
    expected_choice_pass = {
        "ADR-linear-full": False,
        "ADR-linear-split": False,
        "ADR-tiny-nonlinear-choice": True,
    }
    if r_decision.get("choice_pass") != expected_choice_pass:
        raise ValueError("the frozen conditional-choice capacity diagnosis changed")
    for manifest, name in ((a_manifest, "Phase 2.5A"), (b_manifest, "Phase 2.5B"), (r_manifest, "Phase 2.5B-R")):
        if int(manifest.get("decision_count", -1)) != n or int(manifest.get("source_count", -1)) != len(unique_sources):
            raise ValueError(f"{name} count provenance changed")
    if r_manifest.get("data_fingerprint") != inputs["capture_data_fingerprint"]:
        raise ValueError("resolver capture fingerprint changed")

    b_folds = _verify_folds(
        _read_csv(roots["phase_2_5b"] / "fold_assignments.csv"), unique_sources,
        held_role="held_out",
    )
    r_folds = _verify_folds(
        _read_csv(roots["phase_2_5b_r"] / "fold_assignments.csv"), unique_sources,
        held_role="outer_test",
    )
    if b_folds != r_folds:
        raise ValueError("Phase 2.5B-R changed the frozen outer partitions")
    calibration = json.loads((roots["phase_2_5b"] / "calibration.json").read_text())
    calibration_methods = {
        key.split("/", 1)[1] for key in calibration
        if "/" in key
    }
    expected_calibrations = len(unique_sources) * 16
    if (
        len(calibration_methods) != 16
        or not calibration_methods <= set(map(str, b_manifest["method_order"]))
        or len(calibration) != expected_calibrations
    ):
        raise ValueError("Phase 2.5B calibration record count changed")
    for key, value in calibration.items():
        if int(key.split("/", 1)[0].split("_")[-1]) not in b_folds:
            raise ValueError("unknown Phase 2.5B calibration fold")
        if float(value["maximum_intervention_rate"]) != 0.60:
            raise ValueError("Phase 2.5B calibration cap changed")
        if float(value["source_macro_intervention_rate"]) > 0.60 + 1e-12:
            raise ValueError("Phase 2.5B calibration violates its frozen cap")

    choice_rows = _read_csv(roots["phase_2_5b"] / "all_choices.csv")
    required_choices = tuple(config["policies"]["frozen_phase_2_5b"])
    choice_maps: dict[str, dict[str, int]] = {}
    for method in required_choices:
        local = [row for row in choice_rows if row["method"] == method]
        if len(local) != n or len({row["decision_id"] for row in local}) != n:
            raise ValueError(f"frozen OOF choices incomplete for {method}")
        mapped: dict[str, int] = {}
        for row in local:
            i = index.get(row["decision_id"])
            if i is None or row["source"] != sources[i] or row["condition"] != conditions[i]:
                raise ValueError("Phase 2.5B OOF choice metadata changed")
            option = OPTION_INDEX.get(row["choice"])
            if option is None:
                raise ValueError("unknown Phase 2.5B option")
            if (
                row["selected_outcome"] != str(data["outcomes"][i, option])
                or not math.isclose(
                    float(row["selected_utility"]), float(data["utility"][i, option]),
                    abs_tol=1e-12, rel_tol=0.0,
                )
            ):
                raise ValueError("Phase 2.5B selected outcome/utility changed")
            mapped[row["decision_id"]] = option
        choice_maps[method] = mapped

    oof_rows = _read_csv(roots["phase_2_5b_r"] / "all_oof_predictions.csv")
    needed_resolver_methods = {
        "ADR-linear-split",
        "ADR-linear-split/LearnedGate+OracleChoice",
        "ADR-linear-split/OracleGate+LearnedChoice",
        "ADR-linear-split/OracleGate+OracleChoice",
        "ADR-tiny-nonlinear-choice/LearnedGate+LearnedChoice",
        "ADR-tiny-nonlinear-choice/LearnedGate+OracleChoice",
        "ADR-tiny-nonlinear-choice/OracleGate+LearnedChoice",
        "ADR-tiny-nonlinear-choice/OracleGate+OracleChoice",
    }
    resolver_maps: dict[str, dict[str, int]] = {}
    for method in sorted(needed_resolver_methods):
        local = [row for row in oof_rows if row["method"] == method]
        if len(local) != n or len({row["decision_id"] for row in local}) != n:
            raise ValueError(f"resolver OOF rows incomplete for {method}")
        mapped = {}
        for row in local:
            decision_id = row["decision_id"]
            i = index.get(decision_id)
            if i is None or row["source"] != sources[i] or row["family"] != conditions[i]:
                raise ValueError("resolver OOF metadata changed")
            option = OPTION_INDEX.get(row["choice"])
            if option is None:
                raise ValueError("unknown resolver OOF option")
            if (
                row["selected_outcome"] != str(data["outcomes"][i, option])
                or not math.isclose(
                    float(row["selected_utility"]), float(data["utility"][i, option]),
                    abs_tol=1e-12, rel_tol=0.0,
                )
            ):
                raise ValueError("resolver OOF selected outcome/utility changed")
            mapped[decision_id] = option
        resolver_maps[method] = mapped
    hybrid_rows = _read_csv(roots["phase_2_5b_r"] / "oracle_hybrid_metrics.csv")
    hybrid_lookup = {(row["method"], row["hybrid"]): row for row in hybrid_rows}
    for method in ("ADR-linear-split", "ADR-tiny-nonlinear-choice"):
        for hybrid in ("OracleGate+OracleChoice", "OracleGate+LearnedChoice", "LearnedGate+OracleChoice"):
            key = (method, hybrid)
            if key not in hybrid_lookup:
                raise ValueError("required frozen oracle-hybrid metric is missing")
            choices = np.asarray([resolver_maps[f"{method}/{hybrid}"][item] for item in decision_ids])
            metrics = policy_metrics(choices, data["outcomes"], data["utility"], sources)
            if not math.isclose(
                float(metrics["source_macro_utility"]),
                float(hybrid_lookup[key]["scalar_utility"]), abs_tol=1e-12, rel_tol=0.0,
            ):
                raise ValueError("resolver OOF rows disagree with oracle-hybrid metrics")

    sequential = {
        key: (Path(path).read_text() if key == "p2_report" else None)
        for key, path in sequential_paths.items()
    }
    sequential["p2_5_morphology"] = json.loads(sequential_paths["p2_5_morphology"].read_text())
    sequential["p3_1_analysis"] = json.loads(sequential_paths["p3_1_analysis"].read_text())
    sequential["p3_2_analysis"] = json.loads(sequential_paths["p3_2_analysis"].read_text())
    sequential["p3_2_trace_audit"] = json.loads(sequential_paths["p3_2_trace_audit"].read_text())
    sequential["p3_2_known_recovery"] = [
        json.loads(line) for line in sequential_paths["p3_2_known_recovery"].read_text().splitlines()
        if line.strip()
    ]
    if (
        sequential["p2_5_morphology"].get("kind") != "p2_dynamic_router_trace_morphology_audit"
        or sequential["p3_1_analysis"].get("kind") != "p3_1_direct_recovery_window_head_development_analysis"
        or sequential["p3_2_analysis"].get("kind") != "p3_2_frozen_dynamic_closeout_analysis"
        or sequential["p3_2_trace_audit"].get("kind") != "p3_2_frozen_dynamic_closeout_posthoc_trace_audit"
    ):
        raise ValueError("a frozen sequential artifact kind changed")

    return {
        "config_path": config_path,
        "input_sha256": hashes,
        "roots": roots,
        "data": data,
        "labels": labels,
        "decision_ids": decision_ids,
        "index": index,
        "label_rows": label_rows,
        "choice_maps": choice_maps,
        "resolver_maps": resolver_maps,
        "hybrid_lookup": hybrid_lookup,
        "manifests": {"phase_2_5a": a_manifest, "phase_2_5b": b_manifest, "phase_2_5b_r": r_manifest},
        "folds": b_folds,
        "sequential": sequential,
    }


def _crosstab_rows(labels: Mapping[str, np.ndarray], sources: np.ndarray) -> list[dict[str, Any]]:
    risk = np.asarray(labels["risk"], dtype=np.int64)
    benefit = np.asarray(labels["benefit"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for r in (0, 1):
        for b in (0, 1):
            cell = (risk == r) & (benefit == b)
            macro, macro_sources = source_macro_mean(cell, sources)
            rows.append({
                "row_type": "cell", "metric": f"R{r}_B{b}", "risk": r, "benefit": b,
                "raw_count": int(cell.sum()), "raw_rate": float(cell.mean()),
                "source_macro_rate": macro, "source_macro_denominator_sources": macro_sources,
                "independent_sources_with_event": len(set(sources[cell].tolist())),
            })
    summaries = (
        ("P(B=1|R=1)", benefit == 1, risk == 1),
        ("P(B=1|R=0)", benefit == 1, risk == 0),
        ("P(R=1|B=1)", risk == 1, benefit == 1),
        ("risk_positive_no_benefit", (risk == 1) & (benefit == 0), np.ones(len(risk), dtype=bool)),
        ("risk_negative_positive_benefit", (risk == 0) & (benefit == 1), np.ones(len(risk), dtype=bool)),
    )
    for metric, event, denominator in summaries:
        selected = np.asarray(denominator, dtype=bool)
        macro, macro_sources = source_macro_mean(event, sources, selected)
        rows.append({
            "row_type": "summary", "metric": metric, "risk": "", "benefit": "",
            "raw_count": int(np.sum(event & selected)), "raw_denominator": int(selected.sum()),
            "raw_rate": float(np.mean(np.asarray(event)[selected])) if np.any(selected) else float("nan"),
            "source_macro_rate": macro, "source_macro_denominator_sources": macro_sources,
            "independent_sources_with_event": len(set(sources[np.asarray(event) & selected].tolist())),
        })
    return rows


def _group_summary(
    name: str, value: str | int, mask: np.ndarray,
    labels: Mapping[str, np.ndarray], sources: np.ndarray,
    strict_labels: np.ndarray,
) -> dict[str, Any]:
    risk = np.asarray(labels["risk"], dtype=np.int64)
    benefit = np.asarray(labels["benefit"], dtype=np.int64)
    row: dict[str, Any] = {
        "breakdown": name, "value": value, "decisions": int(mask.sum()),
        "independent_sources": len(set(sources[mask].tolist())),
    }
    for r in (0, 1):
        for b in (0, 1):
            event = (risk == r) & (benefit == b)
            row[f"r{r}_b{b}_count"] = int(np.sum(mask & event))
            macro, _ = source_macro_mean(event[mask], sources[mask])
            row[f"r{r}_b{b}_source_macro_rate"] = macro
    disagreement = risk != benefit
    row["disagreement_count"] = int(np.sum(mask & disagreement))
    row["disagreement_sources"] = len(set(sources[mask & disagreement].tolist()))
    row["risk_positive_no_benefit_count"] = int(np.sum(mask & (risk == 1) & (benefit == 0)))
    row["risk_negative_positive_benefit_count"] = int(np.sum(mask & (risk == 0) & (benefit == 1)))
    for strict in STRICT_TO_INDEX:
        row[f"{strict}_count"] = int(np.sum(mask & (strict_labels == strict)))
        row[f"{strict}_sources"] = len(set(sources[mask & (strict_labels == strict)].tolist()))
    return row


def _source_support_rows(
    metadata: Sequence[Mapping[str, Any]], labels: Mapping[str, np.ndarray],
    sources: np.ndarray,
) -> list[dict[str, Any]]:
    strict = np.asarray(labels["strict_label"], dtype=str)
    risk = np.asarray(labels["risk"], dtype=np.int64)
    benefit = np.asarray(labels["benefit"], dtype=np.int64)
    conditions = np.asarray([row["condition"] for row in metadata], dtype=str)
    horizons = np.asarray([int(row["horizon_actions"]) for row in metadata])
    exact_flip_sources: set[str] = set()
    for source in sorted(set(sources.tolist())):
        local_indices = np.flatnonzero(sources == source)
        for left, right in itertools.combinations(local_indices.tolist(), 2):
            if (
                conditions[left] == conditions[right]
                and horizons[left] == horizons[right]
                and risk[left] == risk[right]
                and strict[left] in STRICT_TO_INDEX
                and strict[right] in STRICT_TO_INDEX
                and strict[left] != strict[right]
            ):
                exact_flip_sources.add(source)
    rows = []
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        present = set(strict[mask].tolist())
        rows.append({
            "source": source,
            "historical_split": str(np.asarray([row["split"] for row in metadata])[mask][0]),
            "decisions": int(mask.sum()),
            "family_labels": "|".join(sorted(set(conditions[mask].tolist()))),
            "horizons": "|".join(map(str, sorted(set(horizons[mask].tolist()), reverse=True))),
            "benefit_0_count": int(np.sum(mask & (benefit == 0))),
            "benefit_1_count": int(np.sum(mask & (benefit == 1))),
            "has_both_benefit_labels": bool({0, 1} <= set(benefit[mask].tolist())),
            "risk_positive_no_benefit_count": int(np.sum(mask & (risk == 1) & (benefit == 0))),
            "risk_negative_positive_benefit_count": int(np.sum(mask & (risk == 0) & (benefit == 1))),
            "has_risk_benefit_disagreement": bool(np.any(mask & (risk != benefit))),
            "strict_base_count": int(np.sum(mask & (strict == "strict_base"))),
            "strict_detour_count": int(np.sum(mask & (strict == "strict_detour"))),
            "strict_retreat_count": int(np.sum(mask & (strict == "strict_retreat"))),
            "has_strict_base_detour_flip": bool({"strict_base", "strict_detour"} <= present),
            "has_strict_detour_retreat_flip": bool({"strict_detour", "strict_retreat"} <= present),
            "has_exact_same_risk_option_flip": source in exact_flip_sources,
            "mechanical_family": "glass_recovery",
        })
    return rows


def select_witnesses(
    metadata: Sequence[Mapping[str, Any]], labels: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    outcomes: np.ndarray, utility: np.ndarray,
) -> list[dict[str, Any]]:
    """Deterministically select the nearest exact matched pair per contrast."""

    n = len(metadata)
    mask = np.asarray(arrays["history_mask"], dtype=np.float64)
    if mask.shape[0] != n:
        raise ValueError("witness history mask does not align")
    final_indices = np.asarray([
        int(np.flatnonzero(row > 0.0)[-1]) if np.any(row > 0.0) else -1 for row in mask
    ])
    if np.any(final_indices < 0):
        raise ValueError("a witness candidate has no valid deployable frame")
    robot = np.asarray([
        arrays["robot_state"][i, final_indices[i]] for i in range(n)
    ], dtype=np.float64)
    action = np.asarray([
        arrays["nominal_action"][i, final_indices[i]] for i in range(n)
    ], dtype=np.float64)
    feature = np.column_stack((robot, action))
    mean = feature.mean(axis=0)
    scale = feature.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    standardized = (feature - mean) / scale
    strict = np.asarray(labels["strict_label"], dtype=str)
    risk = np.asarray(labels["risk"], dtype=np.int64)
    benefit = np.asarray(labels["benefit"], dtype=np.int64)
    optimal = np.asarray(labels["optimal"], dtype=np.int64)
    groups: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for i, row in enumerate(metadata):
        if strict[i] in STRICT_TO_INDEX:
            groups[(
                str(row["source_state_sha256"]), str(row["condition"]),
                int(row["horizon_actions"]), int(risk[i]),
            )].append(i)
    candidates: dict[tuple[str, tuple[int, int]], list[tuple[Any, ...]]] = defaultdict(list)
    for (_, family, _, _), indices in groups.items():
        for left, right in itertools.combinations(indices, 2):
            if optimal[left] == optimal[right]:
                continue
            contrast = tuple(sorted((int(optimal[left]), int(optimal[right]))))
            distance = float(np.linalg.norm(standardized[left] - standardized[right]))
            left_id = str(metadata[left]["decision_id"])
            right_id = str(metadata[right]["decision_id"])
            candidates[(family, contrast)].append((
                distance, min(left_id, right_id), max(left_id, right_id), left, right
            ))
    selected: list[dict[str, Any]] = []
    for (family, contrast), options in sorted(candidates.items()):
        distance, _, _, left, right = min(options, key=lambda item: item[:3])
        if optimal[left] > optimal[right]:
            left, right = right, left
        left_hidden = np.ascontiguousarray(arrays["hidden"][left, final_indices[left]])
        right_hidden = np.ascontiguousarray(arrays["hidden"][right, final_indices[right]])
        left_float = left_hidden.astype(np.float64)
        right_float = right_hidden.astype(np.float64)
        denominator = float(np.linalg.norm(left_float) * np.linalg.norm(right_float))
        cosine_distance = (
            1.0 - float(np.dot(left_float, right_float) / denominator)
            if denominator > 0.0 else float("nan")
        )
        source = str(metadata[left]["source_state_sha256"])
        if source != str(metadata[right]["source_state_sha256"]):
            raise AssertionError("witness source match was lost")
        selected.append({
            "family": family,
            "mechanical_family": "glass_recovery",
            "horizon": int(metadata[left]["horizon_actions"]),
            "risk": int(risk[left]),
            "unordered_contrast": f"{OPTION_NAMES[contrast[0]]}_vs_{OPTION_NAMES[contrast[1]]}",
            "source": source,
            "state_a_id": str(metadata[left]["decision_id"]),
            "state_b_id": str(metadata[right]["decision_id"]),
            "state_a_placement_id": str(metadata[left]["placement_id"]),
            "state_b_placement_id": str(metadata[right]["placement_id"]),
            "state_a_observation_sha256": str(
                metadata[left].get("branch_start_hashes", {}).get(
                    "observation_sha256", "unavailable_in_unit_fixture"
                )
            ),
            "state_b_observation_sha256": str(
                metadata[right].get("branch_start_hashes", {}).get(
                    "observation_sha256", "unavailable_in_unit_fixture"
                )
            ),
            "state_a_benefit": int(benefit[left]),
            "state_b_benefit": int(benefit[right]),
            "state_a_optimal_option": OPTION_NAMES[int(optimal[left])],
            "state_b_optimal_option": OPTION_NAMES[int(optimal[right])],
            "state_a_robot_state_json": json.dumps(_native(robot[left]), separators=(",", ":")),
            "state_b_robot_state_json": json.dumps(_native(robot[right]), separators=(",", ":")),
            "state_a_nominal_action_json": json.dumps(_native(action[left]), separators=(",", ":")),
            "state_b_nominal_action_json": json.dumps(_native(action[right]), separators=(",", ":")),
            "zscored_robot_action_distance": distance,
            "state_a_hidden_sha256": hashlib.sha256(left_hidden.tobytes()).hexdigest(),
            "state_b_hidden_sha256": hashlib.sha256(right_hidden.tobytes()).hexdigest(),
            "final_hidden_cosine_distance": cosine_distance,
            "state_a_outcomes_json": json.dumps(dict(zip(OPTION_NAMES, map(str, outcomes[left]))), separators=(",", ":")),
            "state_b_outcomes_json": json.dumps(dict(zip(OPTION_NAMES, map(str, outcomes[right]))), separators=(",", ":")),
            "state_a_utilities_json": json.dumps(dict(zip(OPTION_NAMES, map(float, utility[left]))), separators=(",", ":")),
            "state_b_utilities_json": json.dumps(dict(zip(OPTION_NAMES, map(float, utility[right]))), separators=(",", ":")),
            "rationale": (
                "Same source, frozen condition family, horizon, and Base-catastrophe risk, "
                "but distinct strict optimal options; a scalar risk bit is identical for the pair."
            ),
        })
    return selected


def decide_story(
    counts: Mapping[str, int], rule: Mapping[str, Any]
) -> str:
    broad = rule["broad"]
    broad_pass = (
        int(counts["risk_benefit_disagreement_sources"]) >= int(broad["disagreement_sources_min"])
        and int(counts["exact_same_risk_option_flip_sources"]) >= int(broad["exact_same_risk_flip_sources_min"])
        and int(counts["independent_mechanical_family_count"]) >= int(broad["independent_mechanical_families_min"])
        and int(counts.get("exact_flip_mechanical_family_count", 0)) >= int(broad["independent_mechanical_families_min"])
    )
    if broad_pass:
        return "BROAD_BENCHMARK_PIVOT"
    scoped = rule["glass_scoped"]
    if (
        int(counts["glass_disagreement_sources"]) >= int(scoped["glass_disagreement_sources_min"])
        and int(counts["glass_source_profile_option_flip_sources"])
        >= int(scoped["glass_source_profile_option_flip_sources_min"])
        and int(counts["glass_exact_same_risk_option_flip_sources"])
        >= int(scoped["glass_exact_same_risk_option_flip_sources_min"])
    ):
        return "GLASS_SCOPED_BENCHMARK_PIVOT"
    return "INSUFFICIENT_FOR_PIVOT"


def _story_decision(
    config: Mapping[str, Any], source_rows: Sequence[Mapping[str, Any]],
    witnesses: Sequence[Mapping[str, Any]], labels: Mapping[str, np.ndarray],
    sources: np.ndarray, conditions: np.ndarray,
) -> dict[str, Any]:
    strict = np.asarray(labels["strict_label"], dtype=str)
    risk = np.asarray(labels["risk"], dtype=np.int64)
    benefit = np.asarray(labels["benefit"], dtype=np.int64)
    glass = conditions == "glass"
    glass_flip_sources = set()
    for row in source_rows:
        source = str(row["source"])
        local = (sources == source) & glass
        present = set(strict[local].tolist())
        if {"strict_detour", "strict_retreat"} <= present:
            glass_flip_sources.add(source)
    exact_sources = {str(row["source"]) for row in witnesses}
    glass_exact_sources = {
        str(row["source"]) for row in witnesses if str(row["family"]) == "glass"
    }
    strict_dr = np.isin(strict, ["strict_detour", "strict_retreat"])
    glass_strict_dr = strict_dr & glass
    non_glass = ~glass
    counts = {
        "total_decisions": len(sources),
        "total_sources": len(set(sources.tolist())),
        "risk_benefit_disagreement_states": int(np.sum(risk != benefit)),
        "sources_with_both_B0_and_B1": sum(bool(row["has_both_benefit_labels"]) for row in source_rows),
        "risk_benefit_disagreement_sources": len(set(sources[risk != benefit].tolist())),
        "risk_positive_no_benefit_states": int(np.sum((risk == 1) & (benefit == 0))),
        "risk_positive_no_benefit_sources": len(set(sources[(risk == 1) & (benefit == 0)].tolist())),
        "risk_negative_positive_benefit_states": int(np.sum((risk == 0) & (benefit == 1))),
        "risk_negative_positive_benefit_sources": len(set(sources[(risk == 0) & (benefit == 1)].tolist())),
        "strict_base_detour_flip_sources": sum(bool(row["has_strict_base_detour_flip"]) for row in source_rows),
        "strict_detour_retreat_flip_sources": sum(bool(row["has_strict_detour_retreat_flip"]) for row in source_rows),
        "exact_same_risk_option_flip_pairs": len(witnesses),
        "exact_same_risk_option_flip_sources": len(exact_sources),
        "exact_same_risk_option_flip_condition_families": len({str(row["family"]) for row in witnesses}),
        "condition_family_count": int(config["data_contract"]["condition_family_count"]),
        "independent_mechanical_family_count": int(config["data_contract"]["independent_mechanical_family_count"]),
        "exact_flip_mechanical_family_count": 1 if witnesses else 0,
        "glass_disagreement_sources": len(set(sources[glass & (risk != benefit)].tolist())),
        "glass_strict_detour_or_retreat_states": int(glass_strict_dr.sum()),
        "glass_strict_detour_or_retreat_sources": len(
            set(sources[glass_strict_dr].tolist())
        ),
        "all_strict_detour_or_retreat_states": int(strict_dr.sum()),
        "all_strict_detour_or_retreat_sources": len(
            set(sources[strict_dr].tolist())
        ),
        "glass_strict_retreat_states": int(np.sum(glass & (strict == "strict_retreat"))),
        "glass_strict_retreat_sources": len(set(sources[glass & (strict == "strict_retreat")].tolist())),
        "glass_source_profile_option_flip_sources": len(glass_flip_sources),
        "glass_exact_same_risk_option_flip_sources": len(glass_exact_sources),
        "non_glass_strict_retreat_states": int(np.sum(non_glass & (strict == "strict_retreat"))),
        "non_glass_strict_detour_sources": len(
            set(sources[non_glass & (strict == "strict_detour")].tolist())
        ),
        "oracle_hybrid_effective_sources": len(set(sources.tolist())),
    }
    status = decide_story(counts, config["story_rule"])
    action = str(config["story_rule"]["exactly_one_next_action"][status])
    supported = (
        "Within the frozen glass-recovery design, Base catastrophe risk, intervention "
        "benefit, best intervention, and sequential realizability are empirically distinct "
        "across multiple independent sources."
    )
    return {
        "schema_version": 1,
        "kind": "iclr27_risk_value_decoupling_story_decision",
        "story_status": status,
        "decision_role": "paper story scope; not a method GO gate",
        "capture_data_fingerprint": config["inputs"]["capture_data_fingerprint"],
        "exact_independent_source_counts": counts,
        "family_semantics": {
            "frozen_condition_labels": list(config["data_contract"]["expected_condition_labels"]),
            "condition_family_count": int(config["data_contract"]["condition_family_count"]),
            "independent_mechanical_family": config["data_contract"]["independent_mechanical_family_label"],
            "independent_mechanical_family_count": int(config["data_contract"]["independent_mechanical_family_count"]),
            "condition_labels_are_independent_mechanical_families": False,
        },
        "dominant_family": {
            "condition_label": "glass",
            "strict_detour_or_retreat_states": int(glass_strict_dr.sum()),
            "strict_detour_or_retreat_sources": len(
                set(sources[glass_strict_dr].tolist())
            ),
            "all_strict_detour_or_retreat_states": int(strict_dr.sum()),
            "all_strict_detour_or_retreat_sources": len(
                set(sources[strict_dr].tolist())
            ),
            "concentration": float(glass_strict_dr.sum() / strict_dr.sum()),
            "literal_concentration": f"{int(glass_strict_dr.sum())}/{int(strict_dr.sum())}",
        },
        "supported_headline": supported,
        "prohibited_headlines": [
            "A deployable ADR or router passes method_success or deployment_success.",
            "Risk-value decoupling is established across multiple mechanical hazard families.",
            "The tiny nonlinear choice head is the paper method.",
            "The result is confirmatory, foundation-model-general, or deployment-ready.",
            "The frozen statewise result establishes reliable sequential first-crossing intervention.",
            "Condition labels glass/offpath/noglass are independent hazard families.",
        ],
        "preserved_machine_decisions": {
            "phase_2_5a": "PASS-SUPPORT",
            "phase_2_5b": "INCONCLUSIVE",
            "phase_2_5b_r": "CHOICE_CAPACITY_BOTTLENECK",
        },
        "exactly_one_next_action": action,
    }


def _choices_in_capture_order(mapping: Mapping[str, int], decision_ids: np.ndarray) -> np.ndarray:
    if set(mapping) != set(decision_ids.tolist()):
        raise ValueError("policy choices do not exactly cover the frozen decisions")
    return np.asarray([int(mapping[item]) for item in decision_ids], dtype=np.int64)


def _policy_tables(validated: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]
]:
    data = validated["data"]
    labels = validated["labels"]
    ids = validated["decision_ids"]
    frozen = validated["choice_maps"]
    resolver = validated["resolver_maps"]
    policy_choices: dict[str, np.ndarray] = {
        "Always Base": _choices_in_capture_order(frozen["Base"], ids),
        "Always Detour": _choices_in_capture_order(frozen["Always Detour"], ids),
        "Always Retreat": _choices_in_capture_order(frozen["Always Retreat"], ids),
        "Risk -> Detour": _choices_in_capture_order(frozen["Risk->Detour"], ids),
        "Risk -> Retreat": _choices_in_capture_order(frozen["Risk->Retreat"], ids),
        "Risk -> Best Fixed": _choices_in_capture_order(frozen["Risk->BestFixed"], ids),
    }
    best_fixed = OPTION_INDEX[str(config["policies"]["oracle_benefit_best_fixed_option"])]
    policy_choices["Oracle Benefit Gate -> Best Fixed"] = np.where(
        np.asarray(labels["benefit"], dtype=bool), best_fixed, 0
    ).astype(np.int64)
    policy_choices["Oracle Benefit Gate -> learned tiny nonlinear choice"] = _choices_in_capture_order(
        resolver["ADR-tiny-nonlinear-choice/OracleGate+LearnedChoice"], ids
    )
    policy_choices["Learned Benefit Gate -> Oracle Choice"] = _choices_in_capture_order(
        resolver["ADR-tiny-nonlinear-choice/LearnedGate+OracleChoice"], ids
    )
    policy_choices["full learned ADR"] = _choices_in_capture_order(
        resolver["ADR-tiny-nonlinear-choice/LearnedGate+LearnedChoice"], ids
    )
    policy_choices["Oracle"] = _choices_in_capture_order(
        resolver["ADR-linear-split/OracleGate+OracleChoice"], ids
    )
    metrics = {
        name: policy_metrics(choice, data["outcomes"], data["utility"], data["sources"])
        for name, choice in policy_choices.items()
    }
    risk = metrics["Risk -> Best Fixed"]
    oracle = metrics["Oracle"]
    policy_rows = []
    for order, name in enumerate(policy_choices, start=1):
        local = dict(metrics[name])
        local.update({
            "order": order,
            "policy": name,
            "raw_regret_to_oracle": float(oracle["raw_utility"]) - float(local["raw_utility"]),
            "source_macro_regret_to_oracle": float(oracle["source_macro_utility"]) - float(local["source_macro_utility"]),
            "raw_oracle_improvement_vs_risk_recovered": recovered_fraction(
                float(local["raw_utility"]), float(risk["raw_utility"]), float(oracle["raw_utility"])
            ),
            "source_macro_oracle_improvement_vs_risk_recovered": recovered_fraction(
                float(local["source_macro_utility"]), float(risk["source_macro_utility"]),
                float(oracle["source_macro_utility"]),
            ),
            "choice_provenance": (
                "Phase 2.5B frozen foldwise OOF" if name.startswith("Risk ->")
                else "Phase 2.5A global Detour under canonical B" if name == "Oracle Benefit Gate -> Best Fixed"
                else "Phase 2.5B-R frozen resolver OOF" if name not in {"Always Base", "Always Detour", "Always Retreat"}
                else "Phase 2.5B frozen OOF fixed policy"
            ),
            "paper_role": (
                "capacity diagnostic only; not the paper method"
                if name in {
                    "Oracle Benefit Gate -> learned tiny nonlinear choice",
                    "Learned Benefit Gate -> Oracle Choice",
                    "full learned ADR",
                }
                else "oracle diagnostic only" if name == "Oracle"
                else "policy regret comparator"
            ),
        })
        policy_rows.append(local)

    split_oracle_learned = _choices_in_capture_order(
        resolver["ADR-linear-split/OracleGate+LearnedChoice"], ids
    )
    waterfall_choices = [
        ("Risk -> Best Fixed", policy_choices["Risk -> Best Fixed"], "Phase 2.5B foldwise OOF"),
        ("LearnedGate + OracleChoice", policy_choices["Learned Benefit Gate -> Oracle Choice"], "ADR-tiny-nonlinear-choice"),
        ("OracleGate + learned linear choice", split_oracle_learned, "ADR-linear-split"),
        ("OracleGate + learned tiny nonlinear choice", policy_choices["Oracle Benefit Gate -> learned tiny nonlinear choice"], "ADR-tiny-nonlinear-choice capacity diagnostic"),
        ("full learned ADR", policy_choices["full learned ADR"], "ADR-tiny-nonlinear-choice capacity diagnostic; not the paper method"),
        ("OracleGate + OracleChoice", policy_choices["Oracle"], "diagnostic option Oracle"),
    ]
    waterfall_metrics = {
        name: policy_metrics(choice, data["outcomes"], data["utility"], data["sources"])
        for name, choice, _ in waterfall_choices
    }
    risk_u = float(waterfall_metrics["Risk -> Best Fixed"]["source_macro_utility"])
    oracle_u = float(waterfall_metrics["OracleGate + OracleChoice"]["source_macro_utility"])
    opportunity = oracle_u - risk_u
    choice_fraction = recovered_fraction(
        float(waterfall_metrics["OracleGate + learned tiny nonlinear choice"]["source_macro_utility"]), risk_u, oracle_u
    )
    gate_fraction = recovered_fraction(
        float(waterfall_metrics["LearnedGate + OracleChoice"]["source_macro_utility"]), risk_u, oracle_u
    )
    full_fraction = recovered_fraction(
        float(waterfall_metrics["full learned ADR"]["source_macro_utility"]), risk_u, oracle_u
    )
    linear_fraction = recovered_fraction(
        float(waterfall_metrics["OracleGate + learned linear choice"]["source_macro_utility"]), risk_u, oracle_u
    )
    waterfall_rows = []
    previous = None
    for order, (name, _, provenance) in enumerate(waterfall_choices, start=1):
        local = waterfall_metrics[name]
        utility_value = float(local["source_macro_utility"])
        waterfall_rows.append({
            "order": order, "stage": name, "provenance": provenance,
            "source_macro_utility": utility_value,
            "raw_utility": float(local["raw_utility"]),
            "source_macro_regret_to_oracle": oracle_u - utility_value,
            "increment_vs_previous": "" if previous is None else utility_value - previous,
            "recovered_fraction_vs_risk_best_fixed": recovered_fraction(utility_value, risk_u, oracle_u),
            "oracle_opportunity": opportunity,
            "choice_recovered_fraction": choice_fraction,
            "gate_recovered_fraction": gate_fraction,
            "full_recovered_fraction": full_fraction,
            "linear_choice_recovered_fraction": linear_fraction,
            "oracle_to_tiny_choice_utility_loss": oracle_u - float(waterfall_metrics["OracleGate + learned tiny nonlinear choice"]["source_macro_utility"]),
            "oracle_to_learned_gate_utility_loss": oracle_u - float(waterfall_metrics["LearnedGate + OracleChoice"]["source_macro_utility"]),
        })
        previous = utility_value
    if not math.isclose(
        waterfall_rows[0]["oracle_to_tiny_choice_utility_loss"], 0.0725,
        abs_tol=1e-12, rel_tol=0.0,
    ):
        raise ValueError("frozen tiny-choice decomposition magnitude changed")
    if not math.isclose(
        waterfall_rows[0]["oracle_to_learned_gate_utility_loss"],
        0.29277777777777775, abs_tol=1e-12, rel_tol=0.0,
    ):
        raise ValueError("frozen learned-gate decomposition magnitude changed")
    return policy_rows, waterfall_rows, policy_choices


def _sequential_rows(validated: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    seq = validated["sequential"]
    p2 = config["inputs"]["sequential"]["p2_report"]["canonical"]
    p25 = seq["p2_5_morphology"]
    p31 = seq["p3_1_analysis"]
    p32 = seq["p3_2_analysis"]
    audit = seq["p3_2_trace_audit"]
    known = seq["p3_2_known_recovery"]
    if len(known) != int(p32["counts"]["known_recovery_diagnostics"]):
        raise ValueError("P3.2 known-recovery diagnostic row count changed")
    known_success = sum(row["outcome"] == "task_success" for row in known)
    rows: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        base = {
            "phase": "", "evidence_layer": "", "method_or_diagnostic": "",
            "sources": "", "units": "", "task_success_count": "", "task_success_rate": "",
            "catastrophe_count": "", "catastrophe_rate": "", "safe_noncompletion_count": "",
            "safe_noncompletion_rate": "", "intervention_count": "", "intervention_rate": "",
            "known_recovery_support": "", "missed_known_recovery": "",
            "false_interventions": "", "false_intervention_denominator": "",
            "separability_metric": "", "separability_value": "", "base_collapse": "",
            "interpretation": "", "source_artifact": "",
        }
        base.update(kwargs)
        rows.append(base)

    episodes = int(p2["episodes"])
    add(
        phase="P2", evidence_layer="matched_state_oracle_opportunity",
        method_or_diagnostic="Exact T-20 Base/Detour opportunity diagnostic",
        sources=int(p2["sources"]), units=int(p2["sources"]),
        known_recovery_support=int(p2["known_recovery_support"]),
        interpretation=(
            "Two of four glass sources contain a successful structured Detour from "
            "the identical T-20 state where Base catastrophizes; this is an option "
            "opportunity label, not a deployable method."
        ),
        source_artifact="results/P2_SEQUENTIAL_FIRST_CROSSING_DEV_20260819.md",
    )
    add(
        phase="P2", evidence_layer="matched_state_learned_performance",
        method_or_diagnostic="Fixed T-20 Router (oracle timing)", sources=int(p2["sources"]), units=episodes,
        task_success_count=int(p2["oracle_timing_success"]), task_success_rate=float(p2["oracle_timing_success"] / episodes),
        catastrophe_count=int(p2["oracle_timing_catastrophe"]), catastrophe_rate=float(p2["oracle_timing_catastrophe"] / episodes),
        safe_noncompletion_count=int(p2["oracle_timing_safe_noncompletion"]), safe_noncompletion_rate=float(p2["oracle_timing_safe_noncompletion"] / episodes),
        intervention_count=int(p2["oracle_timing_interventions"]), intervention_rate=float(p2["oracle_timing_interventions"] / episodes),
        known_recovery_support=int(p2["known_recovery_support"]),
        missed_known_recovery=int(p2["missed_known_recovery"]),
        base_collapse=True,
        interpretation=(
            "Privileged fixed T-20 timing for the learned Router, which intervenes "
            "0/12 and recovers 0/2 known opportunities; it is not the option Oracle."
        ),
        source_artifact="results/P2_SEQUENTIAL_FIRST_CROSSING_DEV_20260819.md",
    )
    add(
        phase="P2", evidence_layer="development_sequential_first_crossing",
        method_or_diagnostic="Sequential dynamic first crossing", sources=int(p2["sources"]), units=episodes,
        task_success_count=int(p2["learned_success"]), task_success_rate=float(p2["learned_success"] / episodes),
        catastrophe_count=int(p2["learned_catastrophe"]), catastrophe_rate=float(p2["learned_catastrophe"] / episodes),
        safe_noncompletion_count=int(p2["learned_safe_noncompletion"]), safe_noncompletion_rate=float(p2["learned_safe_noncompletion"] / episodes),
        intervention_count=int(p2["learned_interventions"]), intervention_rate=float(p2["learned_interventions"] / episodes),
        known_recovery_support=int(p2["known_recovery_support"]), missed_known_recovery=int(p2["missed_known_recovery"]),
        false_interventions=int(p2["false_interventions"]), false_intervention_denominator=int(p2["base_success_controls"]),
        interpretation="Small selective development Pareto point, while missing both known exact T-20 recoveries.",
        source_artifact="results/P2_SEQUENTIAL_FIRST_CROSSING_DEV_20260819.md",
    )
    add(
        phase="P2.5", evidence_layer="offline_recovery_window_separability",
        method_or_diagnostic="Frozen Router trajectory morphology", sources=int(p25["cohort"]["stable_source_count"]),
        units=int(p25["cohort"]["primary_comparison_episode_count"]),
        known_recovery_support=int(p25["group_summaries"]["missed_t20_recoverable_glass"]["n"]),
        separability_metric="raw max AUC; MA-3/5/8 AUC",
        separability_value=json.dumps({
            "raw_max_auc": p25["rankings"]["max_nonbase_raw_max"]["missed_vs_control_auc"],
            "moving_average_k3_auc": p25["rankings"]["max_nonbase_moving_average_max_k3"]["missed_vs_control_auc"],
            "moving_average_k5_auc": p25["rankings"]["max_nonbase_moving_average_max_k5"]["missed_vs_control_auc"],
            "moving_average_k8_auc": p25["rankings"]["max_nonbase_moving_average_max_k8"]["missed_vs_control_auc"],
        }, sort_keys=True),
        interpretation="Simple temporal aggregation does not repair recovery-versus-control ordering.",
        source_artifact="results/p2_trace_morphology_audit_20260819.json",
    )
    direct = p31["methods"]["direct_recovery_window_head"]
    add(
        phase="P3.1", evidence_layer="offline_recovery_window_separability",
        method_or_diagnostic="Direct recovery-window head, strict OOF", sources=int(p31["counts"]["unique_sources"]),
        units=int(p31["counts"]["records"]),
        separability_metric="recovery-open AUC; intervention-needed AUC; source-macro accuracy",
        separability_value=json.dumps({
            "recovery_open_auc": p31["ranking"]["direct_recovery_window_head"]["recovery_open_vs_hard_negative_auc"],
            "intervention_needed_auc": p31["ranking"]["direct_recovery_window_head"]["intervention_needed_vs_hard_negative_auc"],
            "source_macro_accuracy": direct["source_macro_accuracy"],
            "macro_f1": direct["macro_f1_fixed_three_class"],
            "detour_recall": direct["class_metrics"]["Detour"]["recall"],
            "hard_control_base_retention": p31["targeted_diagnostics"]["hard_control_base_retention"]["direct_recovery_window_head"]["retention"],
        }, sort_keys=True),
        false_interventions=1, false_intervention_denominator=int(p31["counts"]["hard_controls"]),
        interpretation="Offline source-held-out state separability only; no online outcome is implied.",
        source_artifact="results/p3_1_direct_recovery_head_analysis_20260819.json",
    )
    for method, display in (("P2SequentialRouter", "Frozen P2 sequential Router"), ("DirectRecoveryRouter", "Direct recovery Router")):
        record = p32["methods"][method]
        outcome = record["outcomes"]
        controls = record["base_success_controls"]
        direct_collapse = method == "DirectRecoveryRouter" and record["intervention"]["count"] == 0
        add(
            phase="P3.2", evidence_layer="fresh_sequential_first_crossing",
            method_or_diagnostic=display, sources=int(p32["counts"]["sources"]), units=int(record["episodes"]),
            task_success_count=int(outcome["task_success"]["count"]), task_success_rate=float(outcome["task_success"]["rate"]),
            catastrophe_count=int(outcome["catastrophe"]["count"]), catastrophe_rate=float(outcome["catastrophe"]["rate"]),
            safe_noncompletion_count=int(outcome["safe_noncompletion"]["count"]), safe_noncompletion_rate=float(outcome["safe_noncompletion"]["rate"]),
            intervention_count=int(record["intervention"]["count"]), intervention_rate=float(record["intervention"]["rate"]),
            known_recovery_support=int(record["known_recovery_t20"]["support"]), missed_known_recovery=int(record["known_recovery_t20"]["missed_count"]),
            false_interventions=int(controls["intervention_count"]), false_intervention_denominator=int(controls["support"]),
            base_collapse=direct_collapse,
            interpretation=(
                "Zero false interventions occurred because the Direct method never crossed and operationally collapsed to Base; it is not a selective-intervention success."
                if direct_collapse else "Fresh sequential comparator: one known recovery retained, with two unnecessary control interventions."
            ),
            source_artifact="results/p3_2_frozen_dynamic_closeout_analysis_20260819.json",
        )
    known_catastrophe = sum(row["outcome"] == "catastrophe" for row in known)
    known_safe_noncompletion = sum(
        row["outcome"] == "safe_noncompletion" for row in known
    )
    add(
        phase="P3.2", evidence_layer="option_oracle_diagnostic",
        method_or_diagnostic="Exact T-20 Detour branches", sources=len({row["source_state_sha256"] for row in known}),
        units=len(known), task_success_count=known_success, task_success_rate=known_success / len(known),
        catastrophe_count=known_catastrophe,
        catastrophe_rate=known_catastrophe / len(known),
        safe_noncompletion_count=known_safe_noncompletion,
        safe_noncompletion_rate=known_safe_noncompletion / len(known),
        known_recovery_support=known_success,
        interpretation="Existing option-bearing branches diagnose availability; they are not P2 oracle timing or a deployable policy.",
        source_artifact="results/p3_2_known_recovery_t20_diagnostics_20260819.jsonl",
    )
    add(
        phase="P3.2", evidence_layer="fresh_trace_ordering_audit",
        method_or_diagnostic="Direct trajectory maxima", sources=int(audit["known_recovery_t20"]["sources"]),
        units=int(p32["counts"]["source_condition_units"]),
        intervention_count=int(audit["fresh_trajectory_crossings"]), intervention_rate=float(audit["fresh_trajectory_crossings"] / p32["counts"]["source_condition_units"]),
        separability_metric="known-recovery vs Base-success-control trajectory-max AUC",
        separability_value=float(audit["known_recovery_t20"]["versus_base_success_control_trajectory_max_auc"]),
        base_collapse=True,
        interpretation="Known-recovery maxima rank below many controls; lowering the frozen boundary is not authorized.",
        source_artifact="results/p3_2_frozen_dynamic_closeout_trace_audit_20260819.json",
    )
    return rows


def analyze_frozen_inputs(config_path: Path) -> dict[str, Any]:
    """Validate all frozen inputs and return tables without writing or plotting."""

    config_path = config_path.resolve()
    config = _load_config(config_path)
    validated = _validate_inputs(config, config_path)
    data = validated["data"]
    labels = validated["labels"]
    sources = np.asarray(data["sources"], dtype=str)
    conditions = np.asarray(data["conditions"], dtype=str)
    strict = np.asarray(labels["strict_label"], dtype=str)
    crosstab = _crosstab_rows(labels, sources)
    source_support = _source_support_rows(data["metadata"], labels, sources)
    family_support = [
        _group_summary("condition", family, conditions == family, labels, sources, strict)
        for family in sorted(set(conditions.tolist()))
    ]
    family_support.extend([
        _group_summary(
            "mechanical_family", "glass_recovery", np.ones(len(conditions), dtype=bool),
            labels, sources, strict,
        ),
        _group_summary(
            "condition_scope", "glass_treatment", conditions == "glass",
            labels, sources, strict,
        ),
        _group_summary(
            "condition_scope", "non_glass_controls", conditions != "glass",
            labels, sources, strict,
        ),
    ])
    family_support.extend(
        _group_summary("horizon", int(horizon), np.asarray(data["horizons"]) == horizon, labels, sources, strict)
        for horizon in sorted(set(map(int, data["horizons"])), reverse=True)
    )
    witnesses = select_witnesses(
        data["metadata"], labels, data["arrays"], data["outcomes"], data["utility"]
    )
    if any(
        str(row[key]).startswith("unavailable")
        for row in witnesses
        for key in ("state_a_observation_sha256", "state_b_observation_sha256")
    ):
        raise ValueError("a frozen witness is missing its deployable observation hash")
    policy_rows, waterfall, policy_choices = _policy_tables(validated, config)
    story = _story_decision(
        config, source_support, witnesses, labels, sources, conditions
    )
    sequential = _sequential_rows(validated, config)
    config_sha = _sha256(config_path)
    fingerprint_payload = {
        "config_sha256": config_sha,
        "input_sha256": dict(sorted(validated["input_sha256"].items())),
    }
    fingerprint = hashlib.sha256(json.dumps(
        fingerprint_payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()[:12]
    story["evidence_package_fingerprint"] = fingerprint
    return {
        "config": config,
        "config_sha256": config_sha,
        "validated": validated,
        "evidence_package_fingerprint": fingerprint,
        "risk_benefit_crosstab": crosstab,
        "source_support": source_support,
        "family_support": family_support,
        "risk_only_policy_regret": policy_rows,
        "oracle_hybrid_waterfall": waterfall,
        "same_risk_different_decision_witnesses": witnesses,
        "sequential_realization_gap": sequential,
        "story_decision": story,
        "policy_choices": policy_choices,
    }


def _plot_risk_benefit(path: Path, analysis: Mapping[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    cells = {
        (int(row["risk"]), int(row["benefit"])): row
        for row in analysis["risk_benefit_crosstab"] if row["row_type"] == "cell"
    }
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    fig.patch.set_facecolor("#f7f4ed")
    ax.set_facecolor("#f7f4ed")
    colors = {(0, 0): "#d9e4df", (0, 1): "#f2c879", (1, 0): "#e9957c", (1, 1): "#6ca6a3"}
    for (r, b), row in cells.items():
        x, y = b, 1 - r
        box = FancyBboxPatch((x + 0.06, y + 0.08), 0.88, 0.80,
                             boxstyle="round,pad=0.025,rounding_size=0.035",
                             linewidth=0, facecolor=colors[(r, b)])
        ax.add_patch(box)
        ax.text(x + 0.5, y + 0.59, f"{row['raw_count']} states", ha="center", va="center",
                fontsize=19, weight="bold", color="#17323a")
        ax.text(x + 0.5, y + 0.34, f"{100 * row['source_macro_rate']:.1f}% source-macro",
                ha="center", va="center", fontsize=11, color="#28464e")
        ax.text(x + 0.5, y + 0.17, f"{row['independent_sources_with_event']} sources",
                ha="center", va="center", fontsize=10, color="#28464e")
    ax.text(1.0, 2.14, "Risk is not intervention benefit", ha="center", fontsize=23,
            weight="bold", color="#17323a")
    ax.text(1.0, 2.02, "Frozen exact-state outcomes | 273 decisions | 20 sources",
            ha="center", fontsize=11, color="#51666b")
    ax.text(-0.10, 1.5, "Base safe\nR=0", ha="right", va="center", fontsize=12, weight="bold")
    ax.text(-0.10, 0.5, "Base catastrophe\nR=1", ha="right", va="center", fontsize=12, weight="bold")
    ax.text(0.5, -0.06, "No intervention benefit\nB=0", ha="center", va="top", fontsize=12, weight="bold")
    ax.text(1.5, -0.06, "Positive intervention benefit\nB=1", ha="center", va="top", fontsize=12, weight="bold")
    ax.text(1.0, -0.40, "19 disagreements: 10 risk-positive/no-benefit + 9 risk-negative/positive-benefit",
            ha="center", fontsize=11, color="#9a3e2f", weight="bold")
    ax.set_xlim(-0.55, 2.03); ax.set_ylim(-0.52, 2.28); ax.axis("off")
    fig.savefig(path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_waterfall(path: Path, analysis: Mapping[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = analysis["oracle_hybrid_waterfall"]
    values = [float(row["source_macro_utility"]) for row in rows]
    labels = [str(row["stage"]).replace(" + ", "\n+ ") for row in rows]
    colors = ["#687b83", "#d68155", "#5ba39d", "#70b8ad", "#b86969", "#183f4a"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.patch.set_facecolor("#f7f4ed"); ax.set_facecolor("#f7f4ed")
    bars = ax.bar(range(len(rows)), values, color=colors, width=0.72)
    oracle = values[-1]
    ax.axhline(oracle, color="#183f4a", lw=1.4, ls="--", alpha=0.65)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.014, f"{value:.3f}",
                ha="center", va="bottom", fontsize=11, weight="bold")
    ax.set_xticks(range(len(rows)), labels, fontsize=9)
    ax.set_ylabel("Source-macro utility", fontsize=12)
    ax.set_ylim(min(-0.02, min(values) - 0.08), oracle + 0.11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", color="#cad1cf", alpha=0.6, lw=0.7)
    ax.set_title("The benefit gate dominates the frozen oracle-hybrid loss", loc="left",
                 fontsize=21, weight="bold", color="#17323a", pad=18)
    ax.text(0.0, 1.01,
            f"Tiny-choice loss = {rows[0]['oracle_to_tiny_choice_utility_loss']:.4f}   |   "
            f"learned-gate loss = {rows[0]['oracle_to_learned_gate_utility_loss']:.4f}",
            transform=ax.transAxes, fontsize=11, color="#51666b")
    fig.savefig(path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_source_support(path: Path, analysis: Mapping[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = [
        row for row in analysis["family_support"]
        if row["breakdown"] == "condition"
    ]
    families = [str(row["value"]) for row in rows]
    series = [
        ("disagreement_sources", "R/B disagreement", "#d68155"),
        ("strict_base_sources", "Strict Base", "#687b83"),
        ("strict_detour_sources", "Strict Detour", "#398783"),
        ("strict_retreat_sources", "Strict Retreat", "#b86969"),
    ]
    x = np.arange(len(rows), dtype=float)
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    fig.patch.set_facecolor("#f7f4ed"); ax.set_facecolor("#f7f4ed")
    for offset, (key, label, color) in enumerate(series):
        values = [int(row[key]) for row in rows]
        positions = x + (offset - 1.5) * width
        bars = ax.bar(positions, values, width=width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value + 0.28, str(value),
                ha="center", va="bottom", fontsize=9, color="#17323a",
            )
    ax.set_xticks(x, families, fontsize=11, weight="bold")
    ax.set_ylabel("Independent sources with support", fontsize=11)
    ax.set_ylim(0, max(21, max(int(row["independent_sources"]) for row in rows) + 3))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", color="#cad1cf", alpha=0.6, lw=0.7)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_title("Source support is real, but mechanically glass-scoped", loc="left",
                 fontsize=20, weight="bold", color="#17323a", pad=28)
    ax.text(
        0.0, 1.025,
        "All three condition labels share one glass-recovery design; all 23 strict Retreat states and 51/60 strict D/R states are glass",
        transform=ax.transAxes, fontsize=10.2, color="#51666b",
    )
    ax.text(
        0.0, -0.15,
        "Offpath and noglass are matched controls, not independent mechanical hazard families.",
        transform=ax.transAxes, fontsize=10, color="#9a3e2f", weight="bold",
    )
    fig.savefig(path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_synthesis(config_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    analysis = analyze_frozen_inputs(config_path)
    config = analysis["config"]
    fingerprint = analysis["evidence_package_fingerprint"]
    # Capture repository provenance before creating the untracked staging
    # directory; otherwise the synthesis would mark its own clean run dirty.
    git_commit_at_start = _git_commit()
    git_worktree_dirty_at_start = _git_dirty()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output_dir is None:
        output_dir = ROOT / config["outputs"]["root"] / (
            f"{config['outputs']['directory_prefix']}_{fingerprint}_{timestamp}"
        )
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".risk_value_decoupling_", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        _write_csv(staging / "risk_benefit_crosstab.csv", analysis["risk_benefit_crosstab"])
        _write_csv(staging / "source_support.csv", analysis["source_support"])
        _write_csv(staging / "family_support.csv", analysis["family_support"])
        _write_csv(staging / "risk_only_policy_regret.csv", analysis["risk_only_policy_regret"])
        _write_csv(staging / "oracle_hybrid_waterfall.csv", analysis["oracle_hybrid_waterfall"])
        _write_csv(staging / "same_risk_different_decision_witnesses.csv", analysis["same_risk_different_decision_witnesses"])
        _write_csv(staging / "sequential_realization_gap.csv", analysis["sequential_realization_gap"])
        (staging / "story_decision.json").write_text(
            json.dumps(_native(analysis["story_decision"]), indent=2, sort_keys=True) + "\n"
        )
        (staging / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        _plot_risk_benefit(staging / "figure_risk_benefit_flow.pdf", analysis)
        _plot_waterfall(staging / "figure_oracle_gap_waterfall.pdf", analysis)
        _plot_source_support(staging / "figure_source_support.pdf", analysis)
        artifact_names = [name for name in REQUIRED_OUTPUTS if name != "manifest.json"]
        artifact_names.append("config_resolved.yaml")
        if set(path.name for path in staging.iterdir()) != set(artifact_names):
            raise AssertionError("staging directory violates the exact artifact contract")
        manifest = {
            "schema_version": 1,
            "kind": "iclr27_risk_value_decoupling",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit_at_start,
            "git_worktree_dirty": git_worktree_dirty_at_start,
            "evidence_level": "frozen-evidence-synthesis",
            "new_training": False,
            "new_rollouts": False,
            "new_fresh_test_outcomes_collected": False,
            "exact_state_fresh_test_outcomes_loaded": False,
            "frozen_sequential_closeout_summaries_read": True,
            "capture_data_fingerprint": config["inputs"]["capture_data_fingerprint"],
            "evidence_package_fingerprint": fingerprint,
            "fingerprint_definition": "first 12 hex of canonical SHA-256 over sorted frozen input hashes plus config SHA-256",
            "config_sha256": analysis["config_sha256"],
            "input_sha256": dict(sorted(analysis["validated"]["input_sha256"].items())),
            "artifact_sha256": {name: _sha256(staging / name) for name in artifact_names},
            "decision_count": int(config["data_contract"]["expected_decisions"]),
            "source_count": int(config["data_contract"]["expected_sources"]),
            "condition_family_count": int(config["data_contract"]["condition_family_count"]),
            "independent_mechanical_family_count": int(config["data_contract"]["independent_mechanical_family_count"]),
            "story_status": analysis["story_decision"]["story_status"],
            "exactly_one_next_action": analysis["story_decision"]["exactly_one_next_action"],
            "preserved_machine_decisions": analysis["story_decision"]["preserved_machine_decisions"],
            "required_artifacts": list(REQUIRED_OUTPUTS),
            "optional_artifacts": ["config_resolved.yaml"],
        }
        (staging / "manifest.json").write_text(
            json.dumps(_native(manifest), indent=2, sort_keys=True) + "\n"
        )
        if set(path.name for path in staging.iterdir()) != set(REQUIRED_OUTPUTS) | {"config_resolved.yaml"}:
            raise AssertionError("completed package violates the exact artifact contract")
        staging.replace(output_dir)
    return {
        "output_dir": str(output_dir),
        "evidence_package_fingerprint": fingerprint,
        "story_status": analysis["story_decision"]["story_status"],
        "exactly_one_next_action": analysis["story_decision"]["exactly_one_next_action"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/iclr27/risk_value_decoupling.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run_synthesis(args.config, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
