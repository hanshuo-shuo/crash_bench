"""Model-agnostic policy factory (Path 3, ROADMAP §2/§4).

Path 3 = run the SAME on/off-path protocol on a second VLA architecture so the
causal collision claim is architecture-independent, not an OpenVLA quirk. The eval
loop (`crashbench.eval.run_episode`) is already model-agnostic — it only needs a
`Policy` (`resize_size` + `act`). The one place that hard-codes OpenVLA is the
entry-point construction. This registry moves that behind a name so every runner /
analysis script can dispatch to any backend with `--policy <name>`.

Design constraints (ROADMAP: "最小,不破坏本地环境"):
  * Backends are imported LAZILY, inside build_policy — importing crashbench.policies
    must never fail just because a second model's deps aren't in this env. The
    working `envs/openvla` conda env carries torch+transformers only; π0 (JAX) and
    OFT live in SEPARATE envs and must never be import-forced here.
  * A known-but-not-installed backend raises a clear, actionable RuntimeError
    (which env / checkpoint to stand up), not an opaque ImportError.

Add a backend by appending to `_BACKENDS` and writing its wrapper module (must
implement the `Policy` protocol). No change to the eval loop or any scenario.
"""

from __future__ import annotations

# name / alias -> (module path, class name, one-line "how to stand it up" hint).
# The hint is surfaced verbatim when the module import fails (deps not installed).
_BACKENDS: dict[str, tuple[str, str, str]] = {
    "openvla": (
        "crashbench.policies.openvla_policy", "OpenVLAPolicy",
        "the working `envs/openvla` conda env (torch+transformers).",
    ),
    "openvla-oft": (
        "crashbench.policies.openvla_oft_policy", "OpenVLAOFTPolicy",
        "OpenVLA-OFT: a SEPARATE env (do NOT install into envs/openvla — it may bump "
        "transformers and break the base policy). See setup/README Path 3.",
    ),
    "pi0": (
        "crashbench.policies.pi0_policy", "Pi0Policy",
        "π0 / openpi: a SEPARATE JAX env (envs/openvla has no jax). See setup/README Path 3.",
    ),
}

# convenience aliases -> canonical name
_ALIASES = {
    "openvla-7b": "openvla",
    "oft": "openvla-oft",
    "openpi": "pi0",
    "pi_0": "pi0",
}


def canonical_name(name: str) -> str:
    n = name.strip().lower()
    n = _ALIASES.get(n, n)
    if n not in _BACKENDS:
        raise ValueError(
            f"unknown policy backend {name!r}. "
            f"known: {sorted(_BACKENDS)} (aliases: {sorted(_ALIASES)})"
        )
    return n


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def resolve_policy_cls(name: str):
    """Import and return the backend's Policy class WITHOUT instantiating it.

    Instantiating loads a multi-GB model (needs a GPU); resolving only touches the
    wrapper module. Used by login-node tests to check dispatch wiring, and by
    build_policy. Raises RuntimeError with the stand-up hint if the module's deps
    aren't installed in the current env.
    """
    canon = canonical_name(name)
    module_path, cls_name, hint = _BACKENDS[canon]
    try:
        import importlib
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise RuntimeError(
            f"policy backend {canon!r} is registered but its deps are not importable "
            f"in this env ({e}). Stand it up in {hint}"
        ) from e
    return getattr(mod, cls_name)


def build_policy(name: str = "openvla", **kwargs):
    """Construct a Policy by backend name. kwargs are forwarded to the wrapper's
    __init__ (e.g. pretrained_checkpoint, unnorm_key, prompt_prefix, capture_hidden).

    Backends accept whatever their own architecture needs; a wrapper should ignore or
    reject kwargs that don't apply to it rather than every caller special-casing.
    """
    cls = resolve_policy_cls(name)
    return cls(**kwargs)
