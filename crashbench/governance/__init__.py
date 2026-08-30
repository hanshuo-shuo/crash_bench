"""Machine-readable stage gates and prospective claim-scope decisions."""

from .gates import Criterion, GatePolicy, evaluate_gate

__all__ = ["Criterion", "GatePolicy", "evaluate_gate"]
