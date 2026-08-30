"""Strong linear direct-Q and option-advantage baselines for the scoped pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _weighted_ridge(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> np.ndarray:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or w.shape != (len(x),):
        raise ValueError("weighted ridge shape mismatch")
    if ridge < 0 or np.any(w <= 0) or not np.all(np.isfinite(x)):
        raise ValueError("weighted ridge inputs are invalid")
    augmented = np.column_stack([np.ones(len(x)), x])
    gram = augmented.T @ (w[:, None] * augmented)
    penalty = np.eye(gram.shape[0]) * ridge
    penalty[0, 0] = 0
    rhs = augmented.T @ (w * y)
    return np.linalg.solve(gram + penalty, rhs)


def _predict_ridge(coefficients: np.ndarray, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    return coefficients[0] + x @ coefficients[1:]


@dataclass
class RidgeOptionValue:
    option_ids: tuple[str, ...]
    ridge: float = 1.0
    shared: bool = False

    def fit(
        self,
        features: np.ndarray,
        option_ids: Sequence[str],
        utility: np.ndarray,
        row_weights: np.ndarray,
    ) -> "RidgeOptionValue":
        options = np.asarray(option_ids)
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(utility, dtype=np.float64)
        w = np.asarray(row_weights, dtype=np.float64)
        if set(options) - set(self.option_ids):
            raise ValueError("ridge baseline received unknown option")
        if self.shared:
            one_hot = np.column_stack([options == option for option in self.option_ids]).astype(float)
            self.coefficients_ = _weighted_ridge(
                np.column_stack([x, one_hot]), y, w, self.ridge
            )
        else:
            self.coefficients_ = {}
            for option in self.option_ids:
                mask = options == option
                if not np.any(mask):
                    raise ValueError(f"ridge baseline option has no rows: {option}")
                self.coefficients_[option] = _weighted_ridge(
                    x[mask], y[mask], w[mask], self.ridge
                )
        return self

    def predict(self, features: np.ndarray, option_ids: Sequence[str]) -> np.ndarray:
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("ridge baseline is not fitted")
        options = np.asarray(option_ids)
        x = np.asarray(features, dtype=np.float64)
        if self.shared:
            one_hot = np.column_stack([options == option for option in self.option_ids]).astype(float)
            return _predict_ridge(self.coefficients_, np.column_stack([x, one_hot]))
        values = np.empty(len(x), dtype=np.float64)
        for option in self.option_ids:
            mask = options == option
            values[mask] = _predict_ridge(self.coefficients_[option], x[mask])
        if set(options) - set(self.option_ids):
            raise ValueError("ridge baseline prediction contains unknown option")
        return values


def best_fixed_option(
    source_ids: Sequence[str], option_ids: Sequence[str], utility: Sequence[float]
) -> tuple[str, Mapping[str, float]]:
    sources = np.asarray(source_ids)
    options = np.asarray(option_ids)
    values = np.asarray(utility, dtype=np.float64)
    option_values = {}
    for option in sorted(set(options)):
        source_means = [
            np.mean(values[(sources == source) & (options == option)])
            for source in sorted(set(sources))
            if np.any((sources == source) & (options == option))
        ]
        option_values[option] = float(np.mean(source_means))
    winner = max(sorted(option_values), key=lambda option: option_values[option])
    return winner, option_values


def source_macro_policy_value(
    rows: Sequence[Mapping], choices: Mapping[str, str]
) -> float:
    grouped = {}
    for row in rows:
        block = str(row["block_id"])
        option = str(row["option_id"])
        if choices.get(block) != option:
            continue
        grouped.setdefault(str(row["physical_source_id"]), []).append(float(row["u0"]))
    if set(grouped) != {str(row["physical_source_id"]) for row in rows}:
        raise ValueError("policy choices do not cover every source")
    return float(np.mean([np.mean(values) for values in grouped.values()]))
