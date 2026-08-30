"""Versioned complete snapshot for exact simulator-policy branching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .policy_state import PolicyContinuation, capture_policy_continuation, restore_policy_continuation
from .rng import RNGSnapshot, capture_rng_state, restore_rng_state


RESTORE_ORDER_VERSION = "model-sim-runtime-policy-rng-observation-audit/v1"


def _hash_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _hash_mapping(mapping: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(mapping):
        digest.update(key.encode())
        digest.update(_hash_array(np.asarray(mapping[key])).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class ExactStateBundle:
    identity: Mapping[str, Any]
    flat_state: np.ndarray
    runtime_state: Mapping[str, np.ndarray]
    model_xml: str
    policy_continuation: PolicyContinuation
    rng: RNGSnapshot
    protocol_version: str = "expansion_v1"
    schema_version: int = 1
    restore_order_version: str = RESTORE_ORDER_VERSION
    provenance: Mapping[str, Any] = field(default_factory=dict)
    component_hashes: Mapping[str, str] = field(default_factory=dict)
    bundle_id: str = ""

    def calculated_component_hashes(self) -> dict[str, str]:
        return {
            "simulator_state_sha256": _hash_array(np.asarray(self.flat_state)),
            "runtime_state_sha256": _hash_mapping(self.runtime_state),
            "model_xml_sha256": hashlib.sha256(self.model_xml.encode()).hexdigest(),
            "policy_continuation_sha256": self.policy_continuation.sha256(),
            "rng_sha256": self.rng.sha256(),
        }

    def calculated_bundle_id(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "restore_order_version": self.restore_order_version,
            "identity": self.identity,
            "provenance": self.provenance,
            "component_hashes": self.calculated_component_hashes(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()

    def assert_integrity(self) -> None:
        actual = self.calculated_component_hashes()
        if dict(self.component_hashes) != actual:
            raise ValueError("ExactStateBundle component hash mismatch")
        if self.bundle_id != self.calculated_bundle_id():
            raise ValueError("ExactStateBundle complete hash mismatch")


def capture_exact_state(
    env: Any,
    policy: Any,
    *,
    identity: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    numpy_generators: Mapping[str, np.random.Generator] | None = None,
    jax_keys: Mapping[str, Any] | None = None,
    declared_branch_seed: int | None = None,
    capture_torch_rng: bool = True,
) -> ExactStateBundle:
    base = ExactStateBundle(
        identity=dict(identity),
        flat_state=np.asarray(env.flat_state()).copy(),
        runtime_state={key: np.asarray(value).copy() for key, value in env.controller_state().items()},
        model_xml=str(env.model_xml()),
        policy_continuation=capture_policy_continuation(policy),
        rng=capture_rng_state(
            numpy_generators=numpy_generators,
            jax_keys=jax_keys,
            declared_branch_seed=declared_branch_seed,
            capture_torch=capture_torch_rng,
        ),
        provenance=dict(provenance or {}),
    )
    hashes = base.calculated_component_hashes()
    bundle = ExactStateBundle(**{**base.__dict__, "component_hashes": hashes})
    bundle = ExactStateBundle(**{**bundle.__dict__, "bundle_id": bundle.calculated_bundle_id()})
    bundle.assert_integrity()
    return bundle


def restore_exact_state(
    bundle: ExactStateBundle,
    env: Any,
    policy: Any,
    *,
    numpy_generators: Mapping[str, np.random.Generator] | None = None,
    jax_key_targets: dict[str, Any] | None = None,
    restore_torch_rng: bool = True,
) -> Any:
    bundle.assert_integrity()
    # Exact XML rebuild + sim state/forward.
    env.reset_to_exact(bundle.flat_state, model_xml=bundle.model_xml)
    # Controller, task manager, sensors, cache, and policy-boundary observation.
    observation = env.restore_controller_state(dict(bundle.runtime_state))
    # Policy state precedes RNG by contract; neither operation may advance time.
    restore_policy_continuation(policy, bundle.policy_continuation)
    restore_rng_state(
        bundle.rng,
        numpy_generators=numpy_generators,
        jax_key_targets=jax_key_targets,
        restore_torch=restore_torch_rng,
    )
    restored = capture_exact_state(
        env,
        policy,
        identity=bundle.identity,
        provenance=bundle.provenance,
        numpy_generators=numpy_generators,
        jax_keys=jax_key_targets,
        declared_branch_seed=bundle.rng.declared_branch_seed,
        capture_torch_rng=restore_torch_rng,
    )
    if restored.component_hashes != bundle.component_hashes or restored.bundle_id != bundle.bundle_id:
        raise RuntimeError("exact-state restore audit failed")
    return observation
