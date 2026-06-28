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
    ):
        add_openvla_to_path(openvla_root) if openvla_root else add_openvla_to_path()
        from experiments.robot.robot_utils import get_model, get_image_resize_size
        from experiments.robot.openvla_utils import get_processor

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
