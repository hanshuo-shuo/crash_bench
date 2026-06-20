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
from typing import Iterable

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
