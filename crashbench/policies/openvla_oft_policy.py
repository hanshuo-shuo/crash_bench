"""OpenVLA-OFT policy wrapper (Path 3 — cross-architecture reproduction, ROADMAP §2/§4).

Runs the SAME on/off-path collision protocol on OpenVLA-OFT so the causal claim
("VLA collisions are a missing safety policy, not OOD") is shown to be
architecture-independent rather than an OpenVLA quirk.

OFT differs from base OpenVLA (verified against the OFT repo's run_libero_eval.py):
  * continuous action head (L1 regression) + proprio projector, not discrete-token decode;
  * emits an 8-step action CHUNK per query, executed open-loop (num_open_loop_steps=8);
  * two camera views (third-person + wrist), num_images_in_input=2;
  * proprio state normalized (bounds_q99) inside get_vla_action.

To fit the per-step Policy protocol WITHOUT touching the eval loop, this wrapper buffers
the chunk in a deque and returns one action per act() call — exactly OFT's open-loop
execution. eval.run_episode calls policy.reset() at each episode start to clear the buffer.

Env / namespace note (ROADMAP: "不破坏本地环境"): OFT and base OpenVLA both live under the
`experiments.robot` package name and cannot coexist in one process. This wrapper runs in the
SEPARATE `envs/openvla-oft` conda env and points CRASHBENCH_OPENVLA_ROOT at the OFT repo, so
both this policy AND the LiberoEnv built afterward resolve the OFT copy of experiments.robot.
"""

from __future__ import annotations

import os
from collections import deque
from types import SimpleNamespace

import numpy as np

# OFT repo + LIBERO-spatial checkpoint (installed on /projects/p33100, see setup/README).
DEFAULT_OFT_ROOT = os.environ.get("OPENVLA_OFT_ROOT") or \
    "/projects/p33100/siosio/third_party/openvla-oft"
DEFAULT_OFT_CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"
_BASE_OPENVLA_CKPT = "openvla/openvla-7b-finetuned-libero-spatial"   # run_pilot's default (wrong for OFT)


class OpenVLAOFTPolicy:
    def __init__(
        self,
        pretrained_checkpoint: str = DEFAULT_OFT_CHECKPOINT,
        unnorm_key: str = "libero_spatial",
        center_crop: bool = True,
        prompt_prefix: str = "",            # README §7 prompted-careful baseline
        oft_root: str | None = None,
        num_open_loop_steps: int = 8,       # execute the full 8-step chunk (OFT default/best)
        **ignored,                          # tolerate base-only kwargs (capture_hidden, etc.)
    ):
        oft_root = oft_root or DEFAULT_OFT_ROOT
        # Make BOTH this policy and any LiberoEnv constructed later resolve experiments.robot
        # from the OFT repo (single namespace; the two repos can't coexist).
        os.environ["CRASHBENCH_OPENVLA_ROOT"] = oft_root
        from crashbench.envs.libero_adapter import add_openvla_to_path
        add_openvla_to_path(oft_root)

        # run_pilot passes base OpenVLA's checkpoint by default — that won't load under the
        # OFT loaders (they look up per-checkpoint action_head/proprio files). Redirect it.
        if pretrained_checkpoint == _BASE_OPENVLA_CKPT:
            print(f"[OFT] base-openvla default checkpoint seen -> using {DEFAULT_OFT_CHECKPOINT}")
            pretrained_checkpoint = DEFAULT_OFT_CHECKPOINT

        from experiments.robot.robot_utils import get_model, get_image_resize_size
        from experiments.robot.openvla_utils import (
            get_processor, get_action_head, get_proprio_projector,
        )

        # Mirror the GenerateConfig fields the OFT helpers read (run_libero_eval.py:81-107).
        self.cfg = SimpleNamespace(
            model_family="openvla",
            pretrained_checkpoint=pretrained_checkpoint,
            use_l1_regression=True,
            use_diffusion=False,
            num_diffusion_steps_train=50,
            num_diffusion_steps_inference=50,
            use_film=False,
            num_images_in_input=2,          # third-person + wrist
            use_proprio=True,
            center_crop=center_crop,
            num_open_loop_steps=num_open_loop_steps,
            lora_rank=32,
            unnorm_key=unnorm_key,
            load_in_8bit=False,
            load_in_4bit=False,
        )
        self.model = get_model(self.cfg)
        # resolve unnorm key the way run_libero_eval's check_unnorm_key does
        if (self.cfg.unnorm_key not in self.model.norm_stats
                and f"{self.cfg.unnorm_key}_no_noops" in self.model.norm_stats):
            self.cfg.unnorm_key = f"{self.cfg.unnorm_key}_no_noops"
        assert self.cfg.unnorm_key in self.model.norm_stats, (
            f"unnorm key {self.cfg.unnorm_key!r} not in norm_stats {list(self.model.norm_stats)}")

        self.processor = get_processor(self.cfg)
        self.proprio_projector = (
            get_proprio_projector(self.cfg, self.model.llm_dim, proprio_dim=8)
            if self.cfg.use_proprio else None
        )
        self.action_head = (
            get_action_head(self.cfg, self.model.llm_dim)
            if self.cfg.use_l1_regression else None
        )
        self._resize_size = get_image_resize_size(self.cfg)
        self.prompt_prefix = prompt_prefix
        self._queue: deque = deque(maxlen=num_open_loop_steps)

    @property
    def resize_size(self):
        return self._resize_size

    def reset(self) -> None:
        """Clear the open-loop action buffer. Called by eval.run_episode per episode."""
        self._queue.clear()

    def act(self, observation: dict, instruction: str) -> np.ndarray:
        from experiments.robot.robot_utils import (
            get_action, normalize_gripper_action, invert_gripper_action,
        )
        if self.prompt_prefix:
            instruction = f"{self.prompt_prefix} {instruction}"

        # Requery only when the open-loop chunk is exhausted (OFT open-loop execution).
        if not self._queue:
            actions = get_action(
                self.cfg, self.model, observation, instruction,
                processor=self.processor,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                noisy_action_projector=None,
                use_film=self.cfg.use_film,
            )
            self._queue.extend(actions)

        action = self._queue.popleft()
        # same gripper post-processing as OFT's process_action (normalize [0,1]->[-1,1], flip)
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        return np.asarray(action)
