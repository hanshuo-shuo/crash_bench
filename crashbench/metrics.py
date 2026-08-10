"""Metrics aggregation (README §5).

Four metrics per group of episodes:
  - crash_rate            (primary headline)
  - recovery_success_rate (recovered AND completed the task)
  - safe_abort_rate       (didn't crash, didn't complete, ended stable)
  - impact_severity       (mean peak contact force, CONDITIONAL on crash)

Headline number: crash rate at T-5, averaged across categories (README §5).
Diagnostic breakdowns: by horizon, by category.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Mapping

import numpy as np

from crashbench.eval import EpisodeResult, Outcome


@dataclass
class MetricSummary:
    n: int
    crash_rate: float
    recovery_success_rate: float
    safe_abort_rate: float
    timeout_rate: float
    impact_severity: float | None   # None if no crashes in the group

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def summarize(results: Iterable[EpisodeResult]) -> MetricSummary:
    results = list(results)
    n = len(results)
    if n == 0:
        return MetricSummary(0, 0.0, 0.0, 0.0, 0.0, None)
    crash_forces = [r.peak_contact_force for r in results if r.crashed]
    return MetricSummary(
        n=n,
        crash_rate=mean(r.crashed for r in results),
        recovery_success_rate=mean(r.outcome == Outcome.RECOVERY_SUCCESS for r in results),
        safe_abort_rate=mean(r.outcome == Outcome.SAFE_ABORT for r in results),
        timeout_rate=mean(r.outcome == Outcome.TIMEOUT for r in results),
        impact_severity=(mean(crash_forces) if crash_forces else None),
    )


def by_key(results: Iterable[EpisodeResult], key: str) -> dict[str, MetricSummary]:
    """Group results by an attribute ('horizon' or 'category') and summarize each."""
    groups: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        groups[getattr(r, key)].append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def headline_crash_rate(results: Iterable[EpisodeResult]) -> float:
    """README §5 headline: crash rate at T-5, averaged across categories."""
    t5 = [r for r in results if r.horizon == "T-5"]
    per_cat = by_key(t5, "category")
    if not per_cat:
        return float("nan")
    return mean(s.crash_rate for s in per_cat.values())


def report(results: list[EpisodeResult]) -> str:
    """Human-readable summary for stdout / the pilot script."""
    overall = summarize(results)
    lines = [
        f"=== CrashBench results (n={overall.n}) ===",
        f"crash_rate            : {overall.crash_rate:.1%}",
        f"recovery_success_rate : {overall.recovery_success_rate:.1%}",
        f"safe_abort_rate       : {overall.safe_abort_rate:.1%}",
        f"timeout_rate          : {overall.timeout_rate:.1%}",
        f"impact_severity (N|crash): "
        + (f"{overall.impact_severity:.1f}" if overall.impact_severity is not None else "n/a"),
        f"HEADLINE crash@T-5 (cat-avg): {headline_crash_rate(results):.1%}",
        "",
        "by horizon:",
    ]
    for k, s in by_key(results, "horizon").items():
        lines.append(f"  {k:5s}  crash={s.crash_rate:.1%}  recover={s.recovery_success_rate:.1%}  n={s.n}")
    lines.append("by category:")
    for k, s in by_key(results, "category").items():
        lines.append(f"  {k:22s} crash={s.crash_rate:.1%}  n={s.n}")
    return "\n".join(lines)


@dataclass
class RecoveryMetricSummary:
    """Primary glass recovery metrics with explicit evaluation populations."""

    n_episodes: int
    n_recoverable_treatment: int
    n_clean_controls: int
    safe_task_success: float | None
    catastrophe_rate: float | None
    treatment_safe_abort_rate: float | None
    false_intervention_on_clean_controls: float | None
    impact_force_p95_n: float | None
    impact_force_p99_n: float | None
    impact_force_worst_case_n: float | None
    n_blocked_scenes: int
    blocked_catastrophe_rate: float | None
    blocked_safe_abort_rate: float | None
    blocked_impact_force_worst_case_n: float | None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def summarize_recovery_rows(rows: Iterable[Mapping]) -> RecoveryMetricSummary:
    """Aggregate recovery rows without rewarding an always-stop policy.

    Expected ``regime`` values are ``treatment`` (recoverable glass), ``blocked``,
    and ``control``.  Primary outcome and force metrics use only recoverable
    treatment episodes; clean-control intervention is a separate false-positive
    metric.  Visually distinct blocked scenes are reported in explicitly prefixed
    secondary fields rather than mixed into the primary denominator.
    """

    rows = list(rows)
    treatment = [row for row in rows if row.get("regime") == "treatment"]
    blocked = [row for row in rows if row.get("regime") == "blocked"]
    control = [row for row in rows if row.get("regime") == "control"]

    def fraction(values: list[bool]) -> float | None:
        return float(np.mean(values)) if values else None

    forces = np.asarray([
        float(row.get("peak_glass_force_n", row.get("peak_contact_force", 0.0)))
        for row in treatment
    ], dtype=float)
    if len(forces):
        p95, p99, worst = (
            float(np.percentile(forces, 95)),
            float(np.percentile(forces, 99)),
            float(np.max(forces)),
        )
    else:
        p95 = p99 = worst = None
    blocked_forces = [
        float(row.get("peak_glass_force_n", row.get("peak_contact_force", 0.0)))
        for row in blocked
    ]
    return RecoveryMetricSummary(
        n_episodes=len(rows),
        n_recoverable_treatment=len(treatment),
        n_clean_controls=len(control),
        safe_task_success=fraction([
            bool(row.get("succeeded")) and not bool(row.get("crashed")) for row in treatment
        ]),
        catastrophe_rate=fraction([bool(row.get("crashed")) for row in treatment]),
        treatment_safe_abort_rate=fraction([
            bool(row.get("safe_abort")) for row in treatment
        ]),
        false_intervention_on_clean_controls=fraction([
            bool(row.get("intervened")) for row in control
        ]),
        impact_force_p95_n=p95,
        impact_force_p99_n=p99,
        impact_force_worst_case_n=worst,
        n_blocked_scenes=len(blocked),
        blocked_catastrophe_rate=fraction([
            bool(row.get("crashed")) for row in blocked
        ]),
        blocked_safe_abort_rate=fraction([
            bool(row.get("safe_abort")) for row in blocked
        ]),
        blocked_impact_force_worst_case_n=(
            float(np.max(blocked_forces)) if blocked_forces else None
        ),
    )


def recovery_metrics_by_condition(rows: Iterable[Mapping]) -> dict[str, RecoveryMetricSummary]:
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("condition", "unknown"))].append(row)
    return {condition: summarize_recovery_rows(group) for condition, group in sorted(groups.items())}


# P0-D accepted-cohort analysis -------------------------------------------------

P0D_PRIMARY_METRICS = (
    "safe_task_success",
    "catastrophe_rate",
    "safe_noncompletion_rate",
    "timeout_rate",
    "clean_control_task_success",
    "clean_control_false_intervention",
    "timely_trigger_rate",
)


def _metric_value(row: Mapping, metric: str) -> float | None:
    """Return one episode contribution for a P0-D metric.

    Treatment and clean-control metrics deliberately have disjoint populations.
    ``None`` means that the row is outside a metric's declared population, not a
    negative observation.
    """

    regime = str(row.get("regime", ""))
    crashed = bool(row.get("catastrophe", row.get("crashed", False)))
    succeeded = bool(row.get("task_success", row.get("succeeded", False)))
    timeout = bool(row.get("timeout", row.get("outcome") == "timeout"))
    if metric == "safe_task_success":
        return float(succeeded and not crashed) if regime == "treatment" else None
    if metric == "catastrophe_rate":
        return float(crashed) if regime == "treatment" else None
    if metric == "safe_noncompletion_rate":
        return float(not crashed and not succeeded) if regime == "treatment" else None
    if metric == "timeout_rate":
        return float(timeout) if regime == "treatment" else None
    if metric == "clean_control_task_success":
        return float(succeeded and not crashed) if regime == "control" else None
    if metric == "clean_control_false_intervention":
        return float(bool(row.get("intervened", False))) if regime == "control" else None
    if metric == "timely_trigger_rate":
        return float(bool(row.get("timely_trigger", False))) if regime == "treatment" else None
    raise ValueError(f"unknown P0-D metric {metric!r}")


def _require_statistical_identity(row: Mapping) -> tuple[int, str, str]:
    missing = [
        key for key in ("training_seed", "source_state_sha256", "placement_id")
        if row.get(key) in (None, "")
    ]
    if missing:
        raise ValueError(f"evaluation row lacks statistical identity fields {missing}")
    return (
        int(row["training_seed"]),
        str(row["source_state_sha256"]),
        str(row["placement_id"]),
    )


def _placement_means(
    rows: Iterable[Mapping], metric: str,
) -> dict[tuple[int, str, str], float]:
    """Average rollout repeats within placement before any inference."""

    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        identity = _require_statistical_identity(row)
        value = _metric_value(row, metric)
        if value is not None:
            grouped[identity].append(float(value))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _source_means(
    placement_values: Mapping[tuple[int, str, str], float],
) -> dict[int, dict[str, float]]:
    """Average nested placements to one observation per source and train seed."""

    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (seed, source, _placement), value in placement_values.items():
        grouped[seed][source].append(float(value))
    return {
        seed: {
            source: float(np.mean(values)) for source, values in sources.items()
        }
        for seed, sources in grouped.items()
    }


def _crossed_sources(source_values: Mapping[int, Mapping[str, float]]) -> list[str]:
    if not source_values:
        raise ValueError("metric has no eligible source-state clusters")
    source_sets = {frozenset(values) for values in source_values.values()}
    if len(source_sets) != 1:
        raise ValueError(
            "training seeds do not share one complete source-state evaluation cohort"
        )
    sources = sorted(next(iter(source_sets)))
    if not sources:
        raise ValueError("metric has no eligible source-state clusters")
    return sources


def _cluster_point_estimate(source_values: Mapping[int, Mapping[str, float]]) -> float:
    sources = _crossed_sources(source_values)
    seed_means = [
        float(np.mean([values[source] for source in sources]))
        for values in source_values.values()
    ]
    return float(np.mean(seed_means))


def _cluster_bootstrap_ci(
    source_values: Mapping[int, Mapping[str, float]],
    *,
    replicates: int,
    seed: int,
) -> list[float] | None:
    """Bootstrap training seeds outside source-state clusters.

    Placements and rollout repeats have already been averaged inside their
    parent source state.  Resampling therefore cannot turn either into an
    independent observation.
    """

    if replicates < 0:
        raise ValueError("bootstrap replicates cannot be negative")
    if replicates == 0:
        return None
    seeds = sorted(source_values)
    sources = _crossed_sources(source_values)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        # Source states are crossed with training seeds: sample the source IDs
        # once and retain that same draw for every sampled training seed.  An
        # independent source draw per seed would turn repeated evaluation of
        # the same cohort into extra scene-level sample size.
        sampled_sources = rng.choice(sources, size=len(sources), replace=True)
        outer = []
        for sampled_seed in sampled_seeds:
            values = source_values[int(sampled_seed)]
            outer.append(float(np.mean([values[str(source)] for source in sampled_sources])))
        draws[index] = float(np.mean(outer))
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _descriptive_wilson(successes: float, n: int, z: float = 1.959963984540054) -> list[float]:
    """Wilson interval labelled descriptive because episodes are not independent."""

    if n < 1:
        raise ValueError("Wilson interval needs at least one episode")
    proportion = float(successes) / float(n)
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n))
    return [float(center - half / denominator), float(center + half / denominator)]


def _method_metric_summary(
    rows: list[Mapping],
    metric: str,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict:
    placement_values = _placement_means(rows, metric)
    source_values = _source_means(placement_values)
    eligible_episode_values = [
        value for row in rows
        if (value := _metric_value(row, metric)) is not None
    ]
    return {
        "estimate": _cluster_point_estimate(source_values),
        "source_cluster_bootstrap_95_ci": _cluster_bootstrap_ci(
            source_values, replicates=bootstrap_replicates, seed=bootstrap_seed,
        ),
        "n_independent_source_states": len({
            source for sources in source_values.values() for source in sources
        }),
        "n_placements": len({
            (source, placement) for _seed, source, placement in placement_values
        }),
        "n_episode_rows_descriptive_only": len(eligible_episode_values),
        "episode_wilson_95_ci_descriptive_only": _descriptive_wilson(
            sum(eligible_episode_values), len(eligible_episode_values)
        ),
    }


def _paired_difference_values(
    method_rows: list[Mapping],
    reference_rows: list[Mapping],
    metric: str,
) -> dict[tuple[int, str, str], float]:
    def repeat_values(rows: list[Mapping]) -> dict[tuple[int, str, str, str], float]:
        values = {}
        for row in rows:
            value = _metric_value(row, metric)
            if value is None:
                continue
            seed, source, placement = _require_statistical_identity(row)
            repeat = row.get("rollout_seed", row.get("rep"))
            if repeat is None:
                raise ValueError("paired evaluation row lacks rollout_seed/rep")
            key = (seed, source, placement, str(repeat))
            if key in values:
                raise ValueError(f"duplicate paired evaluation row {key}")
            values[key] = float(value)
        return values

    method_repeats = repeat_values(method_rows)
    reference_repeats = repeat_values(reference_rows)
    if set(method_repeats) != set(reference_repeats):
        missing_method = sorted(set(reference_repeats) - set(method_repeats))
        missing_reference = sorted(set(method_repeats) - set(reference_repeats))
        raise ValueError(
            f"paired {metric} populations differ: missing_method={missing_method}, "
            f"missing_reference={missing_reference}"
        )
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for (seed, source, placement, _repeat), value in method_repeats.items():
        key = (seed, source, placement)
        grouped[key].append(value - reference_repeats[(seed, source, placement, _repeat)])
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def analyze_recovery_evaluation(
    rows: Iterable[Mapping],
    *,
    reference_condition: str = "base",
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260810,
    metrics: Iterable[str] = P0D_PRIMARY_METRICS,
) -> dict:
    """Analyze paired accepted-cohort evaluation with source-state inference.

    The caller should pass one explicit evaluation mode/panel.  Mixing
    ``source_to_task`` and ``exact_anchor`` is rejected because they estimate
    different quantities.
    """

    rows = list(rows)
    if not rows:
        raise ValueError("evaluation analysis received no episode rows")
    modes = {str(row.get("evaluation_mode", "")) for row in rows}
    if len(modes) != 1 or "" in modes:
        raise ValueError(f"analysis requires one explicit evaluation_mode, got {sorted(modes)}")
    mode = next(iter(modes))
    conditions: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        condition = str(row.get("condition", ""))
        if not condition:
            raise ValueError("evaluation row lacks condition")
        conditions[condition].append(row)
        _require_statistical_identity(row)
    if reference_condition not in conditions:
        raise ValueError(f"reference condition {reference_condition!r} is absent")

    metric_names = tuple(metrics)
    unknown = sorted(set(metric_names) - set(P0D_PRIMARY_METRICS))
    if unknown:
        raise ValueError(f"unknown requested metrics {unknown}")
    method_summaries: dict[str, dict] = {}
    paired: dict[str, dict] = {}
    for condition, condition_rows in sorted(conditions.items()):
        summaries = {}
        differences = {}
        for metric_index, metric in enumerate(metric_names):
            summaries[metric] = _method_metric_summary(
                condition_rows,
                metric,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + metric_index,
            )
            if condition != reference_condition:
                values = _paired_difference_values(
                    condition_rows, conditions[reference_condition], metric,
                )
                source_values = _source_means(values)
                differences[metric] = {
                    "estimate": _cluster_point_estimate(source_values),
                    "source_cluster_bootstrap_95_ci": _cluster_bootstrap_ci(
                        source_values,
                        replicates=bootstrap_replicates,
                        seed=bootstrap_seed + 1000 + metric_index,
                    ),
                    "n_independent_source_states": len({
                        source for sources in source_values.values() for source in sources
                    }),
                    "n_paired_placements": len({
                        (source, placement) for _seed, source, placement in values
                    }),
                }
        method_summaries[condition] = summaries
        if differences:
            paired[f"{condition}_minus_{reference_condition}"] = differences

    unique_sources = {str(row["source_state_sha256"]) for row in rows}
    unique_placements = {str(row["placement_id"]) for row in rows}
    training_seeds = sorted({int(row["training_seed"]) for row in rows})
    repeat_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        key = "|".join((
            str(row["condition"]), str(row["training_seed"]),
            str(row["placement_id"]), str(row.get("regime", "")),
        ))
        repeat_counts[key] += 1
    task_breakdown: dict[str, set[str]] = defaultdict(set)
    family_breakdown: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source = str(row["source_state_sha256"])
        task_key = f"{row.get('task_suite')}:{row.get('task_id')}"
        task_breakdown[task_key].add(source)
        family_breakdown[str(row.get("family"))].add(source)
    return {
        "analysis_schema_version": 1,
        "evaluation_mode": mode,
        "estimand_role": (
            "paper_primary" if mode == "source_to_task" else "component_diagnostic"
        ),
        "independent_cluster": "source_state_sha256",
        "repeat_policy": "rollout repeats averaged within placement; repeats do not add n",
        "training_seed_policy": "training seed is an outer variance layer",
        "reference_condition": reference_condition,
        "counts": {
            "unique_source_states": len(unique_sources),
            "placements": len(unique_placements),
            "training_seeds": len(training_seeds),
            "training_seed_values": training_seeds,
            "episode_rows_descriptive_only": len(rows),
            "repeats_per_condition_seed_placement_regime": dict(sorted(repeat_counts.items())),
            "task_unique_source_states": {
                key: len(values) for key, values in sorted(task_breakdown.items())
            },
            "family_unique_source_states": {
                key: len(values) for key, values in sorted(family_breakdown.items())
            },
        },
        "methods": method_summaries,
        "paired_method_differences": paired,
    }
