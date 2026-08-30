"""Exact-state branch capture, restoration, and audit primitives."""

from .state import ExactStateBundle, capture_exact_state, restore_exact_state

__all__ = ["ExactStateBundle", "capture_exact_state", "restore_exact_state"]
