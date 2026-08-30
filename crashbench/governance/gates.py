"""Non-compensatory gates with prospective scoped-continuation semantics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable


class CriterionClass(str, Enum):
    HARD_VALIDITY = "hard_validity"
    CLAIM_SCOPE = "claim_scope"


class EvidenceState(str, Enum):
    OBSERVED = "observed"
    NOT_OBSERVED = "not_observed"


@dataclass(frozen=True)
class Criterion:
    criterion_id: str
    criterion_class: CriterionClass
    actual: float | int | bool | str | None
    operator: str
    threshold: Any
    evidence_state: EvidenceState = EvidenceState.OBSERVED
    evidence_path: str | None = None

    def evaluate(self) -> bool | None:
        if self.evidence_state != EvidenceState.OBSERVED or self.actual is None:
            return None
        if self.operator == ">=":
            return self.actual >= self.threshold
        if self.operator == "<=":
            return self.actual <= self.threshold
        if self.operator == "==":
            return self.actual == self.threshold
        if self.operator == "between_inclusive":
            low, high = self.threshold
            return low <= self.actual <= high
        if self.operator == "in":
            return self.actual in self.threshold
        raise ValueError(f"unsupported gate operator: {self.operator}")


@dataclass(frozen=True)
class GatePolicy:
    stage_id: str
    confirmatory: bool
    test_outcomes_opened: bool
    allow_scoped_continuation: bool
    go_next_action: str
    scoped_next_action: str
    fail_next_action: str

    def __post_init__(self) -> None:
        if self.confirmatory and self.allow_scoped_continuation:
            raise ValueError("confirmatory gates cannot allow scoped continuation")
        if self.test_outcomes_opened and self.allow_scoped_continuation:
            raise ValueError("scoped continuation must be decided before test outcomes open")


def evaluate_gate(criteria: Iterable[Criterion], policy: GatePolicy) -> dict[str, Any]:
    rows = []
    for criterion in criteria:
        passed = criterion.evaluate()
        row = asdict(criterion)
        row["criterion_class"] = criterion.criterion_class.value
        row["evidence_state"] = criterion.evidence_state.value
        rows.append({**row, "pass": passed})
    if not rows:
        raise ValueError("gate requires at least one criterion")
    hard = [row for row in rows if row["criterion_class"] == CriterionClass.HARD_VALIDITY.value]
    claims = [row for row in rows if row["criterion_class"] == CriterionClass.CLAIM_SCOPE.value]
    if not hard:
        raise ValueError("gate requires at least one hard-validity criterion")

    missing = [row["criterion_id"] for row in rows if row["pass"] is None]
    hard_failures = [row["criterion_id"] for row in hard if row["pass"] is False]
    claim_failures = [row["criterion_id"] for row in claims if row["pass"] is False]
    if missing:
        status = "INCONCLUSIVE_FAIL_CLOSED"
        next_action = policy.fail_next_action
    elif hard_failures:
        status = "NO_GO"
        next_action = policy.fail_next_action
    elif claim_failures and policy.allow_scoped_continuation:
        status = "SCOPED_CONTINUE"
        next_action = policy.scoped_next_action
    elif claim_failures:
        status = "NO_GO"
        next_action = policy.fail_next_action
    else:
        status = "GO"
        next_action = policy.go_next_action

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_machine_gate_decision",
        "stage_id": policy.stage_id,
        "policy": asdict(policy),
        "criteria": rows,
        "hard_failures": hard_failures,
        "claim_scope_failures": claim_failures,
        "missing_evidence": missing,
        "status": status,
        "next_action": next_action,
        "non_compensatory": True,
    }
    payload["decision_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return payload
