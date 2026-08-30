"""Source-conformal conservative option selection with Base abstention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Selection:
    option_id: str
    abstained_to_base: bool
    advantage_lcb: float
    catastrophe_margin: float
    reason: str


def source_conformal_quantile(values: Sequence[float], *, alpha: float) -> float:
    residuals = np.sort(np.asarray(values, dtype=np.float64))
    if residuals.ndim != 1 or len(residuals) < 1 or not np.all(np.isfinite(residuals)):
        raise ValueError("conformal residuals must be finite and non-empty")
    if not 0 < alpha < 1:
        raise ValueError("conformal alpha must lie in (0,1)")
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * (1 - alpha)))
    return float(residuals[rank - 1])


class SourceConformalSelector:
    def __init__(
        self,
        option_ids: Sequence[str],
        *,
        alpha: float = 0.10,
        catastrophe_delta: float = 0.02,
    ):
        self.option_ids = tuple(map(str, option_ids))
        self.alpha = float(alpha)
        self.catastrophe_delta = float(catastrophe_delta)
        if "base_continue" not in self.option_ids or len(set(self.option_ids)) != len(self.option_ids):
            raise ValueError("selector requires unique options including base_continue")

    def fit(
        self,
        *,
        source_ids: Sequence[str],
        option_ids: Sequence[str],
        predicted_utility: Sequence[float],
        actual_utility: Sequence[float],
        roles: Sequence[str],
    ) -> "SourceConformalSelector":
        sources = np.asarray(source_ids)
        options = np.asarray(option_ids)
        predicted = np.asarray(predicted_utility, dtype=np.float64)
        actual = np.asarray(actual_utility, dtype=np.float64)
        roles = np.asarray(roles)
        if not (
            len(sources) == len(options) == len(predicted) == len(actual) == len(roles)
        ):
            raise ValueError("conformal input length mismatch")
        if set(roles) != {"calibration"}:
            raise ValueError("conformal fitting may read calibration rows only")
        self.quantiles_ = {}
        for option in self.option_ids:
            mask = options == option
            if not np.any(mask):
                raise ValueError(f"empty calibration option stratum: {option}")
            source_max = []
            for source in sorted(set(sources[mask])):
                source_mask = mask & (sources == source)
                source_max.append(float(np.max(np.abs(actual[source_mask] - predicted[source_mask]))))
            self.quantiles_[option] = source_conformal_quantile(source_max, alpha=self.alpha)
        return self

    def interval(self, option_id: str, predicted_utility: float) -> tuple[float, float]:
        if not hasattr(self, "quantiles_"):
            raise RuntimeError("selector is not calibrated")
        q = self.quantiles_[option_id]
        return float(predicted_utility - q), float(predicted_utility + q)

    def select(
        self,
        *,
        predicted_utility: Mapping[str, float],
        catastrophe_probability: Mapping[str, float],
        minimum_advantage_lcb: float = 0.0,
    ) -> Selection:
        if set(predicted_utility) != set(self.option_ids):
            raise ValueError("selector utility options differ from frozen catalog")
        if set(catastrophe_probability) != set(self.option_ids):
            raise ValueError("selector catastrophe options differ from frozen catalog")
        base_lower, base_upper = self.interval(
            "base_continue", predicted_utility["base_continue"]
        )
        candidates = []
        for option in self.option_ids:
            if option == "base_continue":
                continue
            lower, _ = self.interval(option, predicted_utility[option])
            lcb = lower - base_upper
            catastrophe_margin = (
                float(catastrophe_probability[option])
                - float(catastrophe_probability["base_continue"])
            )
            if lcb > minimum_advantage_lcb and catastrophe_margin <= self.catastrophe_delta:
                candidates.append((lcb, -catastrophe_margin, option, catastrophe_margin))
        if not candidates:
            return Selection(
                "base_continue", True, float("-inf"), 0.0,
                "no option has positive conformal advantage under catastrophe constraint"
            )
        lcb, _, option, margin = max(candidates)
        return Selection(option, False, float(lcb), float(margin), "conservative intervention")


def ensemble_mean_std(predictions: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim < 2 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("ensemble requires at least two finite prediction arrays")
    return np.mean(values, axis=0), np.std(values, axis=0, ddof=1)
