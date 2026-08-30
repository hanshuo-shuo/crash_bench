"""Capture and restore process RNG state used by exact-state branches."""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class RNGSnapshot:
    python_state: object
    numpy_global_state: tuple
    numpy_generators: Mapping[str, dict[str, Any]]
    numpy_random_states: Mapping[str, tuple] = None
    torch_cpu_state: Any | None = None
    torch_cuda_states: tuple[Any, ...] = ()
    jax_keys: Mapping[str, np.ndarray] | None = None
    declared_branch_seed: int | None = None

    def sha256(self) -> str:
        digest = hashlib.sha256()
        _update_digest(digest, self.python_state)
        _update_digest(digest, self.numpy_global_state)
        _update_digest(digest, self.numpy_generators)
        _update_digest(digest, self.numpy_random_states)
        _update_digest(digest, self.torch_cpu_state)
        _update_digest(digest, self.torch_cuda_states)
        _update_digest(digest, self.jax_keys)
        _update_digest(digest, self.declared_branch_seed)
        return digest.hexdigest()


def _update_digest(digest: Any, value: Any) -> None:
    """Hash nested RNG state by semantic value, never pickle object identity."""

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"array\0")
        digest.update(array.dtype.str.encode() + b"\0")
        digest.update(repr(array.shape).encode() + b"\0")
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=str):
            _update_digest(digest, str(key))
            _update_digest(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(b"tuple\0" if isinstance(value, tuple) else b"list\0")
        for item in value:
            _update_digest(digest, item)
    elif value is None:
        digest.update(b"none\0")
    elif isinstance(value, (str, bytes, bool, int, float, np.generic)):
        digest.update(type(value).__name__.encode() + b"\0")
        digest.update(repr(value.item() if isinstance(value, np.generic) else value).encode() + b"\0")
    else:
        raise TypeError(f"unsupported RNG state value for canonical hash: {type(value).__name__}")


def capture_rng_state(
    *,
    numpy_generators: Mapping[str, np.random.Generator] | None = None,
    numpy_random_states: Mapping[str, np.random.RandomState] | None = None,
    jax_keys: Mapping[str, Any] | None = None,
    declared_branch_seed: int | None = None,
    capture_torch: bool = True,
) -> RNGSnapshot:
    torch_cpu = None
    torch_cuda: tuple[Any, ...] = ()
    if capture_torch:
        try:
            import torch

            torch_cpu = torch.random.get_rng_state().cpu().numpy().copy()
            if torch.cuda.is_available():
                torch_cuda = tuple(
                    state.cpu().numpy().copy() for state in torch.cuda.get_rng_state_all()
                )
        except ImportError:
            pass
    return RNGSnapshot(
        python_state=copy.deepcopy(random.getstate()),
        numpy_global_state=copy.deepcopy(np.random.get_state()),
        numpy_generators={
            name: copy.deepcopy(generator.bit_generator.state)
            for name, generator in sorted((numpy_generators or {}).items())
        },
        numpy_random_states={
            name: copy.deepcopy(random_state.get_state())
            for name, random_state in sorted((numpy_random_states or {}).items())
        },
        torch_cpu_state=torch_cpu,
        torch_cuda_states=torch_cuda,
        jax_keys={
            name: np.asarray(key).copy() for name, key in sorted((jax_keys or {}).items())
        },
        declared_branch_seed=declared_branch_seed,
    )


def restore_rng_state(
    snapshot: RNGSnapshot,
    *,
    numpy_generators: Mapping[str, np.random.Generator] | None = None,
    numpy_random_states: Mapping[str, np.random.RandomState] | None = None,
    jax_key_targets: dict[str, Any] | None = None,
    restore_torch: bool = True,
) -> None:
    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_global_state)
    generators = numpy_generators or {}
    missing = set(snapshot.numpy_generators) - set(generators)
    if missing:
        raise KeyError(f"missing named NumPy generators during restore: {sorted(missing)}")
    for name, state in snapshot.numpy_generators.items():
        generators[name].bit_generator.state = copy.deepcopy(state)
    random_states = numpy_random_states or {}
    captured_random_states = snapshot.numpy_random_states or {}
    missing_random_states = set(captured_random_states) - set(random_states)
    if missing_random_states:
        raise KeyError(
            f"missing named NumPy RandomState objects during restore: {sorted(missing_random_states)}"
        )
    for name, state in captured_random_states.items():
        random_states[name].set_state(copy.deepcopy(state))
    if snapshot.jax_keys:
        if jax_key_targets is None:
            raise KeyError("JAX keys were captured but no restore targets were provided")
        missing_jax = set(snapshot.jax_keys) - set(jax_key_targets)
        if missing_jax:
            raise KeyError(f"missing JAX key targets during restore: {sorted(missing_jax)}")
        for name, value in snapshot.jax_keys.items():
            jax_key_targets[name] = np.asarray(value).copy()
    if restore_torch and snapshot.torch_cpu_state is not None:
        import torch

        torch.random.set_rng_state(torch.as_tensor(snapshot.torch_cpu_state, dtype=torch.uint8))
        if snapshot.torch_cuda_states:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA RNG states captured but CUDA is unavailable at restore")
            if torch.cuda.device_count() != len(snapshot.torch_cuda_states):
                raise RuntimeError("CUDA device count differs from captured RNG state")
            torch.cuda.set_rng_state_all(
                [torch.as_tensor(state, dtype=torch.uint8) for state in snapshot.torch_cuda_states]
            )
