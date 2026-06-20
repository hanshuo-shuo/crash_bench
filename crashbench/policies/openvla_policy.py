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

    @property
    def resize_size(self) -> int:
        return self._resize_size

    def act(self, observation: dict, instruction: str) -> np.ndarray:
        from experiments.robot.robot_utils import (
            get_action, normalize_gripper_action, invert_gripper_action,
        )
        if self.prompt_prefix:
            instruction = f"{self.prompt_prefix} {instruction}"
        action = get_action(self.cfg, self.model, observation, instruction, processor=self.processor)
        # gripper: [0,1] -> [-1,1], then flip sign (OpenVLA convention) — same as run_libero_eval
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        return np.asarray(action)
