"""Portable policy-continuation contract used by the branch engine."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class PolicyContinuation:
    backend: str
    contract_version: int
    payload: Mapping[str, Any]
    stateless_certificate: bool = False

    def sha256(self) -> str:
        return hashlib.sha256(pickle.dumps(self, protocol=5)).hexdigest()


@runtime_checkable
class ExactBranchingPolicy(Protocol):
    @property
    def supports_exact_branching(self) -> bool: ...

    def snapshot_continuation(self) -> PolicyContinuation: ...

    def restore_continuation(self, snapshot: PolicyContinuation) -> None: ...


def capture_policy_continuation(policy: Any) -> PolicyContinuation:
    if not isinstance(policy, ExactBranchingPolicy) or not policy.supports_exact_branching:
        raise RuntimeError(f"policy {type(policy).__name__} does not support exact branching")
    snapshot = policy.snapshot_continuation()
    if not isinstance(snapshot, PolicyContinuation):
        raise TypeError("snapshot_continuation must return PolicyContinuation")
    return snapshot


def restore_policy_continuation(policy: Any, snapshot: PolicyContinuation) -> None:
    if not isinstance(policy, ExactBranchingPolicy) or not policy.supports_exact_branching:
        raise RuntimeError(f"policy {type(policy).__name__} does not support exact branching")
    policy.restore_continuation(snapshot)
