"""Complete-case exact-state branching over a finite runtime option set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .state import ExactStateBundle, restore_exact_state


class BranchExecutionError(RuntimeError):
    """A planned branch failed, so the decision cannot enter a complete view."""


@dataclass(frozen=True)
class BranchResult:
    option_id: str
    repeat_index: int
    branch_seed: int | None
    branch_start_bundle_id: str
    branch_start_component_hashes: Mapping[str, str]
    outcome: Mapping[str, Any]
    terminal_signature: str


def terminal_signature(outcome: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(outcome, sort_keys=True, separators=(",", ":"), default=_json_default).encode()
    ).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": value.shape, "values": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"outcome is not JSON serializable: {type(value).__name__}")


class BranchEngine:
    """Restore an immutable anchor before every planned option execution."""

    def __init__(
        self,
        env: Any,
        policy: Any,
        *,
        numpy_generators: Mapping[str, np.random.Generator] | None = None,
        jax_key_targets: dict[str, Any] | None = None,
        restore_torch_rng: bool = True,
    ):
        self.env = env
        self.policy = policy
        self.numpy_generators = numpy_generators
        self.jax_key_targets = jax_key_targets
        self.restore_torch_rng = restore_torch_rng

    def execute_complete_lattice(
        self,
        bundle: ExactStateBundle,
        option_runners: Mapping[str, Callable[[Any, Any], Mapping[str, Any]]],
        *,
        repeats: int = 1,
        branch_seeds: Sequence[int | None] | None = None,
    ) -> list[BranchResult]:
        if not option_runners:
            raise ValueError("option lattice cannot be empty")
        if repeats < 1:
            raise ValueError("repeats must be positive")
        seeds = tuple(branch_seeds) if branch_seeds is not None else (bundle.rng.declared_branch_seed,) * repeats
        if len(seeds) != repeats:
            raise ValueError("branch_seeds length must equal repeats")

        results: list[BranchResult] = []
        for option_id, runner in option_runners.items():
            if not option_id:
                raise ValueError("option_id cannot be empty")
            for repeat_index, branch_seed in enumerate(seeds):
                # A branch-specific seed is permitted only when it is already
                # the seed captured in the immutable anchor. Per-branch seed
                # mutation would make branch starts incomparable.
                if branch_seed != bundle.rng.declared_branch_seed:
                    raise BranchExecutionError(
                        "branch seed differs from captured bundle; capture one bundle per declared seed"
                    )
                try:
                    restore_exact_state(
                        bundle,
                        self.env,
                        self.policy,
                        numpy_generators=self.numpy_generators,
                        jax_key_targets=self.jax_key_targets,
                        restore_torch_rng=self.restore_torch_rng,
                    )
                    outcome = runner(self.env, self.policy)
                except Exception as exc:
                    raise BranchExecutionError(
                        f"planned branch failed: option={option_id} repeat={repeat_index}"
                    ) from exc
                if not isinstance(outcome, Mapping) or not outcome:
                    raise BranchExecutionError(
                        f"branch returned no terminal outcome: option={option_id} repeat={repeat_index}"
                    )
                results.append(
                    BranchResult(
                        option_id=option_id,
                        repeat_index=repeat_index,
                        branch_seed=branch_seed,
                        branch_start_bundle_id=bundle.bundle_id,
                        branch_start_component_hashes=dict(bundle.component_hashes),
                        outcome=dict(outcome),
                        terminal_signature=terminal_signature(outcome),
                    )
                )
        expected = len(option_runners) * repeats
        if len(results) != expected:
            raise BranchExecutionError(f"incomplete branch lattice: {len(results)} != {expected}")
        return results


def assert_deterministic_repeats(results: Sequence[BranchResult]) -> None:
    signatures: dict[str, set[str]] = {}
    for row in results:
        signatures.setdefault(row.option_id, set()).add(row.terminal_signature)
    mismatched = sorted(option for option, values in signatures.items() if len(values) != 1)
    if mismatched:
        raise BranchExecutionError(f"deterministic repeat mismatch for options: {mismatched}")
