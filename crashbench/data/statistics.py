"""Physical-source-first paired inference for expansion estimands."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PairedEffect:
    source_ids: tuple[str, ...]
    differences: np.ndarray
    point_estimate: float


def source_macro_paired_effect(
    rows: Iterable[Mapping[str, object]],
    *,
    method_id: str,
    comparator_id: str,
    source_key: str = "physical_source_id",
    method_key: str = "method_id",
    value_key: str = "value",
) -> PairedEffect:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        source = str(row[source_key])
        method = str(row[method_key])
        value = float(row[value_key])
        if not np.isfinite(value):
            raise ValueError("paired estimand values must be finite")
        grouped.setdefault((source, method), []).append(value)
    sources = sorted({source for source, method in grouped if method in {method_id, comparator_id}})
    if not sources:
        raise ValueError("paired estimand contains no sources")
    missing = [
        source
        for source in sources
        if (source, method_id) not in grouped or (source, comparator_id) not in grouped
    ]
    if missing:
        raise ValueError(f"paired methods are missing physical sources: {missing}")
    differences = np.asarray(
        [
            np.mean(grouped[(source, method_id)])
            - np.mean(grouped[(source, comparator_id)])
            for source in sources
        ],
        dtype=np.float64,
    )
    return PairedEffect(tuple(sources), differences, float(np.mean(differences)))


def shared_cluster_bootstrap_indices(
    n_sources: int, *, draws: int = 10_000, seed: int = 0
) -> np.ndarray:
    if n_sources < 1 or draws < 1:
        raise ValueError("bootstrap requires positive source and draw counts")
    return np.random.default_rng(seed).integers(0, n_sources, size=(draws, n_sources))


def paired_cluster_bootstrap(
    effect: PairedEffect,
    *,
    indices: np.ndarray | None = None,
    draws: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    n = len(effect.source_ids)
    sampled = indices if indices is not None else shared_cluster_bootstrap_indices(n, draws=draws, seed=seed)
    sampled = np.asarray(sampled, dtype=np.int64)
    if sampled.ndim != 2 or sampled.shape[1] != n or np.any((sampled < 0) | (sampled >= n)):
        raise ValueError("bootstrap indices do not match the paired physical-source count")
    estimates = np.mean(effect.differences[sampled], axis=1)
    alpha = 1 - confidence
    return {
        "n_physical_sources": n,
        "point_estimate": effect.point_estimate,
        "ci_lower": float(np.quantile(estimates, alpha / 2)),
        "ci_upper": float(np.quantile(estimates, 1 - alpha / 2)),
        "bootstrap_draws": int(len(estimates)),
    }


def paired_sign_flip_pvalue(
    differences: Sequence[float], *, alternative: str = "greater", seed: int = 0, draws: int = 100_000
) -> dict[str, float | int | str]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("sign-flip differences must be a finite non-empty vector")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("unsupported sign-flip alternative")
    observed = float(np.mean(values))
    if len(values) <= 20:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
        mode = "exact"
    else:
        signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(draws, len(values)))
        mode = "monte_carlo"
    null = np.mean(signs * values[None, :], axis=1)
    tolerance = 1e-15
    if alternative == "greater":
        extreme = null >= observed - tolerance
    elif alternative == "less":
        extreme = null <= observed + tolerance
    else:
        extreme = np.abs(null) >= abs(observed) - tolerance
    # Add-one correction is conservative for Monte Carlo and harmless for exact enumeration.
    pvalue = (int(np.sum(extreme)) + (1 if mode == "monte_carlo" else 0)) / (
        len(null) + (1 if mode == "monte_carlo" else 0)
    )
    return {
        "n_physical_sources": len(values),
        "observed_mean": observed,
        "pvalue": float(pvalue),
        "alternative": alternative,
        "mode": mode,
        "draws": int(len(null)),
    }


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    return float(
        sum(
            math.comb(n, index)
            * probability**index
            * (1 - probability) ** (n - index)
            for index in range(k + 1)
        )
    )


def clopper_pearson_one_sided(
    successes: int, total: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Return exact one-sided lower and upper bounds for a binomial rate."""

    k, n = int(successes), int(total)
    if n < 1 or k < 0 or k > n or not 0 < confidence < 1:
        raise ValueError("invalid binomial count or confidence")
    alpha = 1 - confidence
    if k == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            upper_tail = 1 - _binomial_cdf(k - 1, n, mid)
            if upper_tail < alpha:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    if k == n:
        upper = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            cdf = _binomial_cdf(k, n, mid)
            if cdf > alpha:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return float(lower), float(upper)


def source_simultaneous_coverage(
    covered_by_source: Mapping[str, Sequence[bool]], *, confidence: float = 0.95
) -> dict[str, float | int]:
    if not covered_by_source or any(len(values) < 1 for values in covered_by_source.values()):
        raise ValueError("coverage needs at least one prediction for every physical source")
    source_success = {
        source: bool(all(bool(value) for value in values))
        for source, values in covered_by_source.items()
    }
    successes = sum(source_success.values())
    total = len(source_success)
    lower, upper = clopper_pearson_one_sided(successes, total, confidence=confidence)
    return {
        "n_physical_sources": total,
        "covered_sources": successes,
        "point_estimate": successes / total,
        "one_sided_lower": lower,
        "one_sided_upper": upper,
    }
