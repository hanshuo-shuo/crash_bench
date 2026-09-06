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
    rank = math.ceil((len(residuals) + 1) * (1 - alpha))
    if rank > len(residuals):
        return float("inf")
    return float(residuals[rank - 1])


@dataclass(frozen=True)
class PairwiseSelection:
    option_id: str
    action: str
    minimum_pairwise_lcb: float
    catastrophe_difference_ucb: float
    catastrophe_absolute_ucb: float
    reason: str


class PairwiseSourceConformalSelector:
    """Frozen simultaneous pairwise selector with explicit safety UCBs.

    Quantiles are source-max residual quantiles computed by the calibration-only
    script.  A mechanism-specific quantile is always combined with the global
    quantile using max; missing or infinite strata therefore fail closed.
    """

    GLOBAL = "__global__"

    def __init__(
        self,
        option_ids: Sequence[str],
        *,
        utility_quantiles: Mapping[str, float],
        catastrophe_difference_quantiles: Mapping[str, float],
        catastrophe_absolute_quantiles: Mapping[str, float],
        catastrophe_delta: float = 0.02,
        catastrophe_absolute: float = 0.10,
    ):
        self.option_ids = tuple(map(str, option_ids))
        if "base_continue" not in self.option_ids or len(set(self.option_ids)) != len(self.option_ids):
            raise ValueError("pairwise selector requires unique options including base_continue")
        self.utility_quantiles = dict(utility_quantiles)
        self.catastrophe_difference_quantiles = dict(catastrophe_difference_quantiles)
        self.catastrophe_absolute_quantiles = dict(catastrophe_absolute_quantiles)
        for values in (
            self.utility_quantiles,
            self.catastrophe_difference_quantiles,
            self.catastrophe_absolute_quantiles,
        ):
            if self.GLOBAL not in values:
                raise ValueError("pairwise selector requires a global calibration quantile")
            if any(float(value) < 0 or math.isnan(float(value)) for value in values.values()):
                raise ValueError("calibration quantiles must be nonnegative or infinity")
        self.catastrophe_delta = float(catastrophe_delta)
        self.catastrophe_absolute = float(catastrophe_absolute)

    @staticmethod
    def _combined(mapping: Mapping[str, float], mechanism_id: str) -> float:
        return max(float(mapping[PairwiseSourceConformalSelector.GLOBAL]), float(mapping.get(mechanism_id, math.inf)))

    def certificate_diagnostic(self, mechanism_id: str) -> dict:
        """Check feasibility, not coverage or a conditional risk guarantee.

        Binary-outcome residual quantiles do not certify conditional event
        probabilities. Preserve the frozen selector behavior for comparison.
        """
        q = self._combined(self.catastrophe_absolute_quantiles, mechanism_id)
        impossible = q > self.catastrophe_absolute
        finite = all(math.isfinite(self._combined(mapping, mechanism_id)) for mapping in (
            self.utility_quantiles, self.catastrophe_difference_quantiles,
            self.catastrophe_absolute_quantiles))
        return {
            "status": "CALIBRATION_FAILURE" if impossible or not finite else "NOT_PROVEN_INFEASIBLE",
            "absolute_safety_feasible_for_any_probability": not impossible,
            "forced_safe_stop_for_all_legal_predictions": impossible and "safe_stop" in self.option_ids,
            "q_absolute": q,
            "catastrophe_absolute_limit": self.catastrophe_absolute,
            "conditional_risk_guarantee": False,
        }

    def select(
        self,
        *,
        mechanism_id: str,
        predicted_utility: Mapping[str, float],
        pairwise_scale: Mapping[tuple[str, str], float],
        predicted_catastrophe: Mapping[str, float],
        admissible_options: Sequence[str] | None = None,
    ) -> PairwiseSelection:
        options = tuple(self.option_ids if admissible_options is None else map(str, admissible_options))
        if "base_continue" not in options or set(options) - set(self.option_ids):
            raise ValueError("admissible options must be a frozen-catalog subset containing Base")
        if set(predicted_utility) != set(self.option_ids) or set(predicted_catastrophe) != set(self.option_ids):
            raise ValueError("prediction options differ from frozen catalog")
        if not all(math.isfinite(float(value)) for value in predicted_utility.values()):
            raise ValueError("utility predictions must be finite")
        if not all(math.isfinite(float(value)) and 0 <= float(value) <= 1
                   for value in predicted_catastrophe.values()):
            raise ValueError("catastrophe predictions must be legal probabilities")
        q_utility = self._combined(self.utility_quantiles, mechanism_id)
        q_difference = self._combined(self.catastrophe_difference_quantiles, mechanism_id)
        q_absolute = self._combined(self.catastrophe_absolute_quantiles, mechanism_id)
        candidates = []
        base_catastrophe = float(predicted_catastrophe["base_continue"])
        for option in options:
            if option == "base_continue":
                continue
            pair_lcbs = []
            for other in options:
                if other == option:
                    continue
                scale = float(pairwise_scale.get((option, other), math.nan))
                if not math.isfinite(scale) or scale < 0.05:
                    raise ValueError("pairwise scale must be finite and at least 0.05")
                pair_lcbs.append(
                    float(predicted_utility[option]) - float(predicted_utility[other])
                    - q_utility * scale
                )
            minimum_lcb = min(pair_lcbs)
            difference_ucb = (
                float(predicted_catastrophe[option]) - base_catastrophe + q_difference
            )
            absolute_ucb = float(predicted_catastrophe[option]) + q_absolute
            if (
                minimum_lcb > 0
                and difference_ucb <= self.catastrophe_delta
                and absolute_ucb <= self.catastrophe_absolute
            ):
                candidates.append(
                    (minimum_lcb, -absolute_ucb, option, difference_ucb, absolute_ucb)
                )
        if candidates:
            minimum_lcb, _, option, difference_ucb, absolute_ucb = max(candidates)
            return PairwiseSelection(
                option, "INTERVENE", float(minimum_lcb), float(difference_ucb),
                float(absolute_ucb), "strict simultaneous utility and safety contracts pass"
            )
        base_absolute_ucb = base_catastrophe + q_absolute
        if base_absolute_ucb > self.catastrophe_absolute and "safe_stop" in options:
            return PairwiseSelection(
                "safe_stop", "ABSTAIN_TO_SAFE_STOP", float("-inf"), float("inf"),
                float(base_absolute_ucb), "Base violates absolute safety and no recovery is certified"
            )
        return PairwiseSelection(
            "base_continue", "PRESERVE_BASE", float("-inf"), 0.0,
            float(base_absolute_ucb), "no non-Base option satisfies all simultaneous contracts"
        )


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
