"""Raw outcome vectors, frozen scalar utility, sensitivity, and Pareto tools."""

from __future__ import annotations

import itertools
import re
from dataclasses import asdict, dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PhysicalBudgets:
    option_duration_steps: float
    path_length_m: float
    force_exposure_ns: float
    latency_ms: float
    source: str

    def __post_init__(self) -> None:
        if min(
            self.option_duration_steps,
            self.path_length_m,
            self.force_exposure_ns,
            self.latency_ms,
        ) <= 0:
            raise ValueError("all physical normalization budgets must be positive")
        source = self.source.lower().replace("/", " ")
        mentions_test_quantile = re.search(r"\btest\s+quantiles?\b", source) is not None
        explicitly_negated = re.search(
            r"\b(no|not|without)\b[^.;]{0,48}\btest\s+quantiles?\b", source
        ) is not None
        derived_from_test = any(
            phrase in source
            for phrase in (
                "derived from test", "estimated from test", "fit from test",
                "test-derived", "test outcome quantile",
            )
        )
        if derived_from_test or (mentions_test_quantile and not explicitly_negated):
            raise ValueError("normalization budgets cannot be derived from test quantiles")


@dataclass(frozen=True)
class OutcomeVector:
    task_success: int
    catastrophe: int
    safe_noncompletion: int
    intervention_invoked: int
    human_help: int
    option_duration_steps: float
    path_length_m: float
    force_exposure_ns: float
    latency_ms: float
    catastrophe_type: str | None = None
    termination_reason: str | None = None
    handoff_mode: str | None = None
    return_to_base_success: bool | None = None

    def __post_init__(self) -> None:
        binary = (
            self.task_success,
            self.catastrophe,
            self.safe_noncompletion,
            self.intervention_invoked,
            self.human_help,
        )
        if any(value not in (0, 1) for value in binary):
            raise ValueError("outcome indicator fields must be binary")
        if self.task_success + self.catastrophe + self.safe_noncompletion != 1:
            raise ValueError("exactly one terminal outcome indicator must be one")
        if min(
            self.option_duration_steps,
            self.path_length_m,
            self.force_exposure_ns,
            self.latency_ms,
        ) < 0:
            raise ValueError("continuous outcome costs cannot be negative")


@dataclass(frozen=True)
class UtilityWeights:
    task_success: float = 1.0
    catastrophe: float = -2.0
    safe_noncompletion: float = -0.25
    intervention_invoked: float = -0.05
    human_help: float = -0.25
    option_duration_norm: float = -0.02
    path_length_norm: float = -0.02
    force_exposure_norm: float = -0.02
    latency_norm: float = -0.01


PRIMARY_WEIGHTS = UtilityWeights()


def normalized_costs(outcome: OutcomeVector, budgets: PhysicalBudgets) -> dict[str, float]:
    return {
        "option_duration_norm": float(np.clip(outcome.option_duration_steps / budgets.option_duration_steps, 0, 1)),
        "path_length_norm": float(np.clip(outcome.path_length_m / budgets.path_length_m, 0, 1)),
        "force_exposure_norm": float(np.clip(outcome.force_exposure_ns / budgets.force_exposure_ns, 0, 1)),
        "latency_norm": float(np.clip(outcome.latency_ms / budgets.latency_ms, 0, 1)),
    }


def scalar_utility(
    outcome: OutcomeVector,
    budgets: PhysicalBudgets,
    weights: UtilityWeights = PRIMARY_WEIGHTS,
) -> float:
    normalized = normalized_costs(outcome, budgets)
    return float(
        weights.task_success * outcome.task_success
        + weights.catastrophe * outcome.catastrophe
        + weights.safe_noncompletion * outcome.safe_noncompletion
        + weights.intervention_invoked * outcome.intervention_invoked
        + weights.human_help * outcome.human_help
        + sum(getattr(weights, key) * value for key, value in normalized.items())
    )


def sensitivity_weights() -> tuple[UtilityWeights, ...]:
    rows = []
    for catastrophe_cost, noncompletion_cost, intervention_cost, continuous_scale in itertools.product(
        (1.0, 2.0, 5.0),
        (0.0, 0.25, 0.5),
        (0.0, 0.02, 0.05, 0.10),
        (0.0, 1.0, 2.0),
    ):
        rows.append(
            UtilityWeights(
                catastrophe=-catastrophe_cost,
                safe_noncompletion=-noncompletion_cost,
                intervention_invoked=-intervention_cost,
                option_duration_norm=-0.02 * continuous_scale,
                path_length_norm=-0.02 * continuous_scale,
                force_exposure_norm=-0.02 * continuous_scale,
                latency_norm=-0.01 * continuous_scale,
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class OptionDecision:
    best_options: tuple[str, ...]
    strict_winner: str | None
    strict_gap: float
    benefit: bool
    best_deployable_advantage: float


def option_decision(
    mean_utility: Mapping[str, float],
    *,
    base_option_id: str,
    deployable_option_ids: Iterable[str],
    epsilon: float = 0.0,
    gamma: float = 0.10,
    tie_tolerance: float = 1e-12,
) -> OptionDecision:
    deployable = tuple(sorted(set(deployable_option_ids)))
    if base_option_id not in mean_utility:
        raise KeyError(f"base option missing from utilities: {base_option_id}")
    if not deployable or any(option not in mean_utility for option in deployable):
        raise ValueError("deployable option set is empty or lacks utility values")
    values = {key: float(mean_utility[key]) for key in deployable}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("option utilities must be finite")
    maximum = max(values.values())
    best = tuple(sorted(key for key, value in values.items() if abs(value - maximum) <= tie_tolerance))
    ordered = sorted(values.values(), reverse=True)
    gap = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float("inf")
    strict = best[0] if len(best) == 1 and gap >= gamma else None
    non_base = [value for key, value in values.items() if key != base_option_id]
    best_advantage = max(non_base) - values[base_option_id] if non_base else float("-inf")
    return OptionDecision(
        best_options=best,
        strict_winner=strict,
        strict_gap=gap,
        benefit=bool(best_advantage > epsilon),
        best_deployable_advantage=float(best_advantage),
    )


def pareto_frontier(rows: Mapping[str, Sequence[float]]) -> tuple[str, ...]:
    """Return non-dominated methods for all-maximized metric vectors."""

    names = sorted(rows)
    vectors = {name: np.asarray(rows[name], dtype=np.float64) for name in names}
    if not vectors or any(vector.ndim != 1 for vector in vectors.values()):
        raise ValueError("Pareto rows must be non-empty one-dimensional vectors")
    widths = {len(vector) for vector in vectors.values()}
    if len(widths) != 1 or not all(np.all(np.isfinite(vector)) for vector in vectors.values()):
        raise ValueError("Pareto rows must have equal width and finite values")
    frontier = []
    for name in names:
        value = vectors[name]
        dominated = any(
            other != name
            and np.all(vectors[other] >= value)
            and np.any(vectors[other] > value)
            for other in names
        )
        if not dominated:
            frontier.append(name)
    return tuple(frontier)
