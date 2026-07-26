"""OpenVLA policy wrapper (PLAN.md §6, baselines in README §7).

Reuses OpenVLA's verified model-loading and action-decoding helpers from
run_libero_eval.py. The output of `act()` is already gripper-normalized and
sign-flipped, i.e. ready to hand straight to LiberoEnv.step().

The `prompt_prefix` hook supports the README §7 "prompted-careful" baseline
(prepend "move slowly, avoid collisions" to the instruction) with no code change.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from crashbench.envs.libero_adapter import add_openvla_to_path


def _resolve_checkpoint(checkpoint: str, revision: str | None) -> tuple[str, dict]:
    """Resolve an immutable HF revision to a cached snapshot before model loading."""
    from pathlib import Path
    requested = checkpoint
    resolved = checkpoint
    if revision is not None and not Path(checkpoint).expanduser().exists():
        from huggingface_hub import snapshot_download
        resolved = snapshot_download(repo_id=checkpoint, revision=revision, local_files_only=True)
    resolved_path = Path(resolved).expanduser()
    resolved_revision = resolved_path.name if resolved_path.parent.name == "snapshots" else None
    if revision is not None and resolved_revision is not None and resolved_revision != revision:
        raise RuntimeError(
            f"checkpoint cache resolved {revision} to unexpected snapshot {resolved_revision}"
        )
    return resolved, {
        "requested": requested,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "resolved_path": str(resolved_path.resolve()) if resolved_path.exists() else resolved,
    }


class OpenVLAPolicy:
    def __init__(
        self,
        pretrained_checkpoint: str = "openvla/openvla-7b-finetuned-libero-spatial",
        unnorm_key: str = "libero_spatial",
        center_crop: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        prompt_prefix: str = "",          # README §7 prompted-careful baseline
        openvla_root: str | None = None,
        capture_hidden: bool = False,     # self-report probe: export the LM's last hidden state
        enable_steering: bool = False,    # Path 1-1b: install a WRITE hook on the final RMSNorm
        checkpoint_revision: str | None = None,
    ):
        add_openvla_to_path(openvla_root) if openvla_root else add_openvla_to_path()
        from experiments.robot.robot_utils import get_model, get_image_resize_size
        from experiments.robot.openvla_utils import get_processor

        pretrained_checkpoint, self.checkpoint_identity = _resolve_checkpoint(
            pretrained_checkpoint, checkpoint_revision)
        # Mirror run_libero_eval.py's GenerateConfig fields that the helpers read.
        self.cfg = SimpleNamespace(
            model_family="openvla",
            pretrained_checkpoint=pretrained_checkpoint,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            center_crop=center_crop,
            unnorm_key=unnorm_key,
        )
        self.model = get_model(self.cfg)
        config_commit = getattr(getattr(self.model, "config", None), "_commit_hash", None)
        self.checkpoint_identity["model_config_commit_hash"] = config_commit
        generation = getattr(self.model, "generation_config", None)
        self.checkpoint_identity["generation_config"] = {
            key: getattr(generation, key, None)
            for key in ("do_sample", "num_beams", "temperature", "top_k", "top_p")
        }
        # resolve unnorm key the same way run_libero_eval does
        if (self.cfg.unnorm_key not in self.model.norm_stats
                and f"{self.cfg.unnorm_key}_no_noops" in self.model.norm_stats):
            self.cfg.unnorm_key = f"{self.cfg.unnorm_key}_no_noops"
        assert self.cfg.unnorm_key in self.model.norm_stats, (
            f"unnorm key {self.cfg.unnorm_key} not in norm_stats")
        self.processor = get_processor(self.cfg)
        self._resize_size = get_image_resize_size(self.cfg)
        self.prompt_prefix = prompt_prefix

        # self-report probe (README §8): capture the LM's final hidden state at the last prompt
        # token of the PREFILL pass — i.e. "having seen the scene, about to emit the action."
        # A forward hook on the final RMSNorm; no third_party edit. last_hidden is set per act().
        self.capture_hidden = capture_hidden
        self.last_hidden: np.ndarray | None = None
        self._cap_seq = -1
        self._cap_vec = None
        if capture_hidden:
            self._install_hidden_hook()

        # Path 1-1b activation steering: subtract alpha * d_unit from the final RMSNorm output
        # at EVERY token position (prefill context + each decoded action token). Because OpenVLA's
        # action = discrete tokens off the LM head applied to this post-norm state, shifting it
        # directly moves the action bins. d_unit is the probe's crash direction, so subtracting it
        # pushes the model away from "I will crash". alpha=0 reproduces the bare policy exactly.
        self._steer_vec_t = None          # torch tensor (hidden,), lazily materialized
        self._steer_alpha = 0.0
        self._steer_np = None             # numpy source vector (set via set_steering)
        if enable_steering:
            self._install_steer_hook()

    def _install_hidden_hook(self):
        # language_model = HF causal LM (Llama); .model.norm = final RMSNorm -> last hidden state.
        try:
            norm = self.model.language_model.model.norm
        except AttributeError as e:  # pragma: no cover - guard against arch drift
            raise RuntimeError(
                "could not find language_model.model.norm for the hidden-state hook; "
                "inspect the model arch and update _install_hidden_hook") from e

        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out          # [B, seq, hidden]
            seq = t.shape[1]
            if seq > self._cap_seq:                                # keep the prefill (largest seq)
                self._cap_seq = seq
                self._cap_vec = t[0, -1].detach().float().cpu().numpy()  # last token

        norm.register_forward_hook(_hook)

    def _install_steer_hook(self):
        import torch
        try:
            norm = self.model.language_model.model.norm
        except AttributeError as e:  # pragma: no cover
            raise RuntimeError("could not find language_model.model.norm for the steering hook") from e

        def _steer(_mod, _inp, out):
            if self._steer_alpha == 0.0 or self._steer_np is None:
                return None                                    # no-op -> identical to bare policy
            t = out[0] if isinstance(out, tuple) else out      # [B, seq, hidden]
            if self._steer_vec_t is None or self._steer_vec_t.device != t.device \
                    or self._steer_vec_t.dtype != t.dtype:
                self._steer_vec_t = torch.as_tensor(self._steer_np, dtype=t.dtype, device=t.device)
            t = t - self._steer_alpha * self._steer_vec_t      # broadcast over [B, seq, hidden]
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t

        norm.register_forward_hook(_steer)

    def set_steering(self, vec: np.ndarray | None, alpha: float) -> None:
        """Set the steering direction (4096-d, e.g. Probe.steer_vector()) and strength alpha.
        alpha=0 (or vec=None) disables steering -> bare policy. Requires enable_steering=True."""
        self._steer_np = None if vec is None else np.asarray(vec, dtype=np.float32)
        self._steer_vec_t = None                               # force re-materialize
        self._steer_alpha = float(alpha)

    @property
    def resize_size(self) -> int:
        return self._resize_size

    def act(self, observation: dict, instruction: str) -> np.ndarray:
        from experiments.robot.robot_utils import (
            get_action, normalize_gripper_action, invert_gripper_action,
        )
        if self.prompt_prefix:
            instruction = f"{self.prompt_prefix} {instruction}"
        if self.capture_hidden:
            self._cap_seq, self._cap_vec = -1, None
        action = get_action(self.cfg, self.model, observation, instruction, processor=self.processor)
        if self.capture_hidden:
            self.last_hidden = self._cap_vec
        # gripper: [0,1] -> [-1,1], then flip sign (OpenVLA convention) — same as run_libero_eval
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        return np.asarray(action)
