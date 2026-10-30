"""OpenVLA-OFT behind serve.py's websocket protocol (RNG reset message, per-request identity diagnostics).

Runs in the openvla-oft environment. Observations carry the raw 180-degree-rotated 256 px agent and wrist images, the
8-d state, the prompt and the LIBERO suite (for the action un-normalisation key); resizing, the model call and the gripper
post-processing follow openvla-oft's experiments/robot/libero/run_libero_eval.py. The L1-regression head is deterministic,
so the RNG reset only restarts the request counter and the reported key is the seed.
"""
import argparse
import hashlib
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from serve import digest


class OFTPolicy:
    def __init__(self, checkpoint, oft_root):
        sys.path.insert(0, oft_root)
        from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, resize_image_for_policy
        from experiments.robot.robot_utils import get_action, get_image_resize_size, get_model, invert_gripper_action, normalize_gripper_action
        # GenerateConfig defaults of run_libero_eval.py for the LIBERO checkpoints (L1 regression, two images, proprio, centre crop).
        self.cfg = SimpleNamespace(model_family='openvla', pretrained_checkpoint=checkpoint, use_l1_regression=True, use_diffusion=False,
                                   num_diffusion_steps_train=50, num_diffusion_steps_inference=50, use_film=False, num_images_in_input=2,
                                   use_proprio=True, center_crop=True, num_open_loop_steps=8, lora_rank=32, unnorm_key='',
                                   load_in_8bit=False, load_in_4bit=False)
        self.model = get_model(self.cfg)
        self.proprio = get_proprio_projector(self.cfg, self.model.llm_dim, proprio_dim=8)
        self.head = get_action_head(self.cfg, self.model.llm_dim)
        self.processor = get_processor(self.cfg); self.size = get_image_resize_size(self.cfg)
        self.resize, self.get_action = resize_image_for_policy, get_action
        self.post = lambda a: invert_gripper_action(normalize_gripper_action(a, binarize=True))
        self.lock = threading.Lock(); self.run = None; self.n = 0; self.seed = None

    def key(self, suite):
        return suite + '_no_noops' if suite + '_no_noops' in self.model.norm_stats else suite

    def infer(self, obs):
        with self.lock:
            if '__paired_reset_rng__' in obs:
                seed = obs['__paired_reset_rng__']
                if type(seed) is not int or not 0 <= seed < 2 ** 32:
                    raise ValueError('Invalid seed')
                self.seed, self.run, self.n = seed, obs['run_id'], 0
                return {'seed': seed, 'run_id': self.run, 'requests': 0, 'key_hex': 'oft-deterministic-%d' % seed}
            ih = digest(obs); self.cfg.unnorm_key = self.key(str(obs['suite']))
            if self.cfg.unnorm_key not in self.model.norm_stats:
                raise RuntimeError('No un-normalisation key for ' + str(obs['suite']))
            view = {'full_image': self.resize(np.asarray(obs['observation/image']), self.size),
                    'wrist_image': self.resize(np.asarray(obs['observation/wrist_image']), self.size), 'state': np.asarray(obs['observation/state'])}
            actions = self.get_action(self.cfg, self.model, view, str(obs['prompt']), processor=self.processor, action_head=self.head,
                                      proprio_projector=self.proprio, noisy_action_projector=None, use_film=False)
            actions = np.stack([self.post(np.asarray(a, dtype=np.float64)) for a in actions]); self.n += 1
            key = 'oft-deterministic-%d' % self.seed
            return {'actions': actions, 'diagnostic': {'input_sha256': ih, 'rng_before': key, 'rng_after': key, 'request_index': self.n,
                                                       'run_id': self.run, 'output_replayed': False, 'native_output_linf_difference': 0.,
                                                       'native_chunk_sha256': hashlib.sha256(actions.tobytes()).hexdigest(),
                                                       'unnorm_key': self.cfg.unnorm_key}}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--port', type=int, required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--oft-root', required=True); p.add_argument('--openpi', required=True); a = p.parse_args()
    sys.path[:0] = [str(Path(a.openpi)/'packages/openpi-client/src'), str(Path(a.openpi)/'src')]
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy = OFTPolicy(a.checkpoint, a.oft_root)
    WebsocketPolicyServer(policy, host='127.0.0.1', port=a.port, metadata={'policy': 'openvla_oft'}).serve_forever()


if __name__ == '__main__':
    main()
