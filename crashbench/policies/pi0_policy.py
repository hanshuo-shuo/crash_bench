"""π0 (openpi) policy wrapper (Path 3 — cross-architecture reproduction, ROADMAP §2/§4).

Third architecture in the Path-3 matrix (after base OpenVLA and OpenVLA-OFT). π0 is a
FLOW-MATCHING VLA on a JAX stack — a genuinely different model family — so reproducing the
same on/off-path collision result here strengthens "VLA collisions are a missing safety
policy, not an OpenVLA quirk" into an architecture-independent claim.

π0 differs from the OpenVLA family (verified against openpi examples/libero/main.py and
src/openpi/policies/libero_policy.py):
  * flow-matching action expert (JAX), not a discrete-token / L1-regression head;
  * emits a multi-step action CHUNK; openpi's LIBERO eval REPLANS every 5 steps
    (num_open_loop_steps), so we execute 5 steps of each chunk then requery;
  * two camera views (third-person + wrist), each 180°-rotated then resize_with_pad→224
    (aspect-preserving pad — NOT the square lanczos resize OpenVLA/OFT use);
  * state = eef_pos(3) + axisangle(eef_quat)(3) + gripper_qpos(2) — IDENTICAL 8-dim vector
    to crashbench's policy_observation (openpi packs the same concatenation);
  * gripper post-processing is handled INSIDE openpi's LiberoOutputs transform, so — unlike
    the OFT wrapper — we do NOT normalize/invert the gripper here; the action is env-ready.

The wrapper runs in the SEPARATE `envs/openpi` JAX venv. crashbench's LiberoEnv takes the
torch-free `model_family="pi0"` path (libero_adapter.py) so openpi (JAX) and LIBERO (mujoco)
coexist in ONE process with NO OpenVLA / torch on the path.
"""

from __future__ import annotations

from collections import deque

import numpy as np

# config name + trained checkpoint (openpi public GCS bucket, anonymous). `pi0_libero` is the
# true π0 (flow-matching) LIBERO checkpoint; `pi05_libero` / `pi0_fast_libero` also exist and
# can be swapped in via kwargs for extra Path-3 columns.
DEFAULT_PI0_CONFIG = "pi0_libero"
DEFAULT_PI0_CHECKPOINT = "gs://openpi-assets/checkpoints/pi0_libero"
_BASE_OPENVLA_CKPT = "openvla/openvla-7b-finetuned-libero-spatial"  # run_pilot's default (wrong for π0)


class Pi0Policy:
    def __init__(
        self,
        pretrained_checkpoint: str = DEFAULT_PI0_CHECKPOINT,
        config_name: str = DEFAULT_PI0_CONFIG,
        resize_size: int = 224,
        num_open_loop_steps: int = 5,       # openpi LIBERO replan_steps (examples/libero/main.py)
        prompt_prefix: str = "",            # README §7 prompted-careful baseline
        capture_hidden: bool = False,       # self-report probe (Path 3): tap the VLM hidden state
        pi0_tap: str = "vlm",               # "vlm" = PaliGemma prefix; "action_expert" = suffix state token
        **ignored,                          # tolerate base/OFT-only kwargs (unnorm_key, center_crop, ...)
    ):
        # run_pilot passes base OpenVLA's checkpoint id by default — meaningless to openpi. Redirect.
        if pretrained_checkpoint == _BASE_OPENVLA_CKPT:
            print(f"[pi0] base-openvla default checkpoint seen -> using {DEFAULT_PI0_CHECKPOINT}")
            pretrained_checkpoint = DEFAULT_PI0_CHECKPOINT

        from openpi.training import config as _config
        from openpi.policies import policy_config

        train_config = _config.get_config(config_name)
        # create_trained_policy runs download.maybe_download on the gs:// dir internally,
        # loads params + norm_stats, and returns a Policy whose .infer applies the LIBERO
        # input/output transforms (tokenize, normalize, un-pad actions to 7-dim).
        self._policy = policy_config.create_trained_policy(train_config, pretrained_checkpoint)

        self._resize_size = int(resize_size)
        self._n_open_loop = int(num_open_loop_steps)
        self.prompt_prefix = prompt_prefix
        self._queue: deque = deque()

        # Self-report probe (Path 3). base/OFT tap the LLM's final post-norm hidden at the last
        # prompt token; the analog for π0 is the PaliGemma (VLM) backbone's FINAL-layer hidden at
        # the last valid PREFIX token — "having seen image+prompt, about to denoise the action."
        # sample_actions computes exactly this prefix forward but DISCARDS its output (keeps only
        # the KV cache, pi0.py:237). We re-run that same prefix forward here (openpi untouched) and
        # keep prefix_out. Real forward happens only on REQUERY; buffered steps set last_hidden=None
        # so the capture logs a hidden state ONLY at query frames.
        self.capture_hidden = capture_hidden
        self.pi0_tap = pi0_tap
        assert pi0_tap in ("vlm", "action_expert"), f"bad pi0_tap {pi0_tap!r}"
        self.last_hidden: np.ndarray | None = None

    def _feature(self, element: dict) -> np.ndarray:
        return self._action_expert_feature(element) if self.pi0_tap == "action_expert" \
            else self._prefix_feature(element)

    def _to_observation(self, element: dict):
        """Replicate Policy.infer's input pipeline: copy -> input_transform -> batch -> Observation."""
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model
        inputs = jax.tree.map(lambda x: x, element)              # copy (transforms may mutate)
        inputs = self._policy._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return _model.Observation.from_dict(inputs)

    def _action_expert_feature(self, element: dict) -> np.ndarray:
        """Action-expert (motor-intent) tap: the suffix STATE-token hidden after one denoising
        forward — the representation the action velocity is read out from (action_out_proj reads
        suffix_out[:, -H:]; the state token at position 0 fuses proprio + prefix vision/language).
        Deterministic probe: time=1.0, noise=zeros. Mirrors sample_actions' prefill + first step
        (openpi untouched). Returns suffix_out[:, 0] as float32 [action_expert_width]."""
        import jax.numpy as jnp
        import einops
        from openpi.models.pi0 import make_attn_mask

        obs = self._to_observation(element)
        model = self._policy._model
        b = obs.state.shape[0]

        # prefill the prefix KV cache (same as sample_actions)
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
        prefix_attn = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_pos = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn, positions=prefix_pos)

        # one suffix forward at t=1 with zero noise (deterministic)
        noise = jnp.zeros((b, model.action_horizon, model.action_dim), dtype=prefix_tokens.dtype)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            obs, noise, jnp.broadcast_to(jnp.asarray(1.0, dtype=prefix_tokens.dtype), b))
        suffix_attn = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn2 = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn = jnp.concatenate([prefix_attn2, suffix_attn], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn, positions=positions,
            kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
        return np.asarray(suffix_out[0, 0], dtype=np.float32)     # state token (position 0)

    def _prefix_feature(self, element: dict) -> np.ndarray:
        """VLM final-layer hidden at the last valid prefix token, for the given raw obs element.
        Mirrors Policy.infer's input pipeline (copy -> input_transform -> batch -> Observation),
        then runs pi0's prefix forward and returns prefix_out[last_valid_token] as float32 [width]."""
        import jax.numpy as jnp
        from openpi.models.pi0 import make_attn_mask

        obs = self._to_observation(element)
        model = self._policy._model
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = model.PaliGemma.llm(
            [prefix_tokens, None], mask=attn_mask, positions=positions)   # [b, prefix_len, width]
        last = jnp.sum(prefix_mask, axis=1) - 1                   # index of last valid prefix token
        feat = prefix_out[jnp.arange(prefix_out.shape[0]), last]  # [b, width]
        return np.asarray(feat[0], dtype=np.float32)

    @property
    def resize_size(self):
        return self._resize_size

    def reset(self) -> None:
        """Clear the open-loop action buffer. Called by eval.run_episode per episode."""
        self._queue.clear()

    @property
    def supports_exact_branching(self) -> bool:
        return True

    def snapshot_continuation(self):
        from crashbench.branching.policy_state import PolicyContinuation

        return PolicyContinuation(
            backend="pi0",
            contract_version=1,
            payload={
                "queue": tuple(np.asarray(action).copy() for action in self._queue),
                "num_open_loop_steps": self._n_open_loop,
                "capture_hidden": bool(self.capture_hidden),
                "tap": self.pi0_tap,
            },
        )

    def restore_continuation(self, snapshot) -> None:
        if snapshot.backend != "pi0" or snapshot.contract_version != 1:
            raise ValueError("incompatible pi0 continuation snapshot")
        payload = snapshot.payload
        if int(payload["num_open_loop_steps"]) != self._n_open_loop:
            raise ValueError("pi0 action-chunk length drift")
        if bool(payload["capture_hidden"]) != bool(self.capture_hidden) or payload["tap"] != self.pi0_tap:
            raise ValueError("pi0 continuation configuration drift")
        self._queue.clear()
        self._queue.extend(np.asarray(action).copy() for action in payload["queue"])
        self.last_hidden = None

    def _prep_image(self, img: np.ndarray) -> np.ndarray:
        # openpi client pipeline: resize_with_pad (aspect-preserving) -> uint8. The 180° rotate
        # already happened in LiberoEnv.policy_observation's pi0 branch (matches train preprocessing).
        from openpi_client import image_tools
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.asarray(img), self._resize_size, self._resize_size)
        )

    def act(self, observation: dict, instruction: str) -> np.ndarray:
        if self.prompt_prefix:
            instruction = f"{self.prompt_prefix} {instruction}"

        # Requery only when the open-loop chunk is exhausted (openpi replan-every-N execution).
        if not self._queue:
            element = {
                "observation/image": self._prep_image(observation["full_image"]),
                "observation/wrist_image": self._prep_image(observation["wrist_image"]),
                "observation/state": np.asarray(observation["state"], dtype=np.float32),
                "prompt": str(instruction),
            }
            if self.capture_hidden:
                self.last_hidden = self._feature(element)          # fresh forward -> real hidden
            chunk = np.asarray(self._policy.infer(element)["actions"])
            assert len(chunk) >= self._n_open_loop, (
                f"π0 emitted {len(chunk)} actions < replan {self._n_open_loop}")
            self._queue.extend(chunk[: self._n_open_loop])
        elif self.capture_hidden:
            self.last_hidden = None                                # buffered step -> no new forward

        return np.asarray(self._queue.popleft())
