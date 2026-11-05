"""Policy server for the closed-loop causal test of the mechanism pilot (docs/generative_verifier/HIL_MECHANISM_CLOSED_LOOP.md).

It speaks serve.py's protocol to verifier_dataset/hil_suite_run.py (RNG reset per run, the same per-request diagnostic), with
one addition: every reset carries the run's arm, and every request is answered by patching.Patcher.steer. The arm moves
the action expert's image-token keys/values at the chosen layers further from the old prompt A's (amp), or by a random
vector of the same norm (rand), by lam - 1 times their B - A difference; the native arm (lam 1) is the same program with
nothing moved. Every arm draws its noise exactly as openpi's Policy.infer does after the reset, so twins share the noise
of each request. On request 1 of each run the server also runs openpi's own Policy.infer and the steering program at
lam 1 on the same input and noise (the tool check). Every request is logged to LOG_DIR/<run_id>.jsonl.

Usage: serve_steer.py --port P --checkpoint DIR --log-dir DIR (openpi environment, pinned openpi on PYTHONPATH)
"""
import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'feasibility'))
from serve import digest  # noqa: E402

KINDS = ('native', 'amp', 'rand')
ARM_KEYS = {'name', 'kind', 'lam', 'layers', 'prompt_a', 'direction_seed'}


def check_arm(arm, depth):
    """The arm a reset message carries; raises unless it is well formed (native exactly when lam == 1)."""
    if not isinstance(arm, dict) or set(arm) != ARM_KEYS:
        raise ValueError('Arm must have the keys %s' % sorted(ARM_KEYS))
    layers = arm['layers']
    if (arm['kind'] not in KINDS or not isinstance(arm['lam'], float) or arm['lam'] < 1 or (arm['kind'] == 'native') != (arm['lam'] == 1)
            or not isinstance(layers, list) or not layers or layers != sorted(set(layers))
            or not all(type(x) is int and 0 <= x < depth for x in layers)
            or not isinstance(arm['prompt_a'], str) or not arm['prompt_a'].strip()
            or type(arm['direction_seed']) is not int or not 0 <= arm['direction_seed'] < 2 ** 32 or not isinstance(arm['name'], str)):
        raise ValueError('Invalid arm: %r' % (arm,))
    return arm


def chunk_sha(actions):
    return hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()


class SteeredPolicy:
    def __init__(self, policy, patcher, log_dir):
        import jax
        self.policy, self.patcher, self.jax = policy, patcher, jax
        self.directory = Path(log_dir); self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock(); self.run = None; self.arm = None; self.n = 0; self.log = None

    def key(self, rng=None):
        return np.asarray(self.jax.random.key_data(self.policy._rng if rng is None else rng), dtype='<u4').tobytes().hex()

    def reset(self, obs):
        seed = obs['__paired_reset_rng__']
        if type(seed) is not int or not 0 <= seed < 2 ** 32:
            raise ValueError('Invalid seed')
        arm = check_arm(obs.get('__mech_arm__'), self.patcher.depth)
        if self.log is not None:
            self.log.close()
        self.policy._rng = self.jax.random.key(seed); self.run = obs['run_id']; self.n = 0; self.arm = arm
        self.log = (self.directory/(str(self.run) + '.jsonl')).open('x')
        return {'seed': seed, 'run_id': self.run, 'requests': 0, 'key_hex': self.key(), 'arm': arm}

    def infer(self, obs):
        with self.lock:
            if '__paired_reset_rng__' in obs:
                return self.reset(obs)
            if self.arm is None:
                raise RuntimeError('Reset with an arm before the first request')
            if obs.get('prompt') == self.arm['prompt_a']:
                raise RuntimeError('The running prompt equals the old prompt')
            jax = self.jax; arm = self.arm; started = time.monotonic()
            before = self.key(); start = self.policy._rng; ih = digest(obs)
            self.policy._rng, sample = jax.random.split(start)  # as openpi's Policy.infer
            self.n += 1; after = self.key()
            layers = np.isin(np.arange(self.patcher.depth), arm['layers'])
            mode = 1 if arm['kind'] == 'rand' else 0
            direction = jax.random.fold_in(jax.random.key(arm['direction_seed']), self.n)
            old = dict(obs, prompt=arm['prompt_a'])
            actions, stats = self.patcher.steer(dict(obs), old, sample, layers, arm['lam'] - 1., mode, direction)
            entry = dict(request_index=self.n, run_id=self.run, arm=arm['name'], input_sha256=ih, rng_before=before, rng_after=after,
                         delta_k=float(stats[0]), diff_k=float(stats[1]), norm_k=float(stats[2]),
                         delta_v=float(stats[3]), diff_v=float(stats[4]), norm_v=float(stats[5]), chunk_sha256=chunk_sha(actions))
            if self.n == 1:
                plain, _ = self.patcher.steer(dict(obs), old, sample, layers, 0., 0, direction)
                self.policy._rng = start
                native = np.asarray(self.policy.infer(dict(obs))['actions'])
                if self.key() != after:
                    raise RuntimeError('Policy.infer advanced the RNG differently')
                entry.update(fidelity_linf=float(np.abs(plain - native).max()), first_shift_linf=float(np.abs(actions - plain).max()),
                             native_actions=native.tolist(), plain_actions=plain.tolist(), actions=np.asarray(actions).tolist())
            entry['seconds'] = time.monotonic() - started
            self.log.write(json.dumps(entry) + '\n'); self.log.flush()
            return {'actions': np.asarray(actions), 'diagnostic': {
                'input_sha256': ih, 'rng_before': before, 'rng_after': after, 'request_index': self.n, 'run_id': self.run,
                'output_replayed': False, 'native_output_linf_difference': 0., 'native_chunk_sha256': entry['chunk_sha256'], 'arm': arm['name']}}

    def close(self):
        if self.log is not None:
            self.log.close(); self.log = None

    def warm_up(self, example):
        """Compiles the steering program and Policy.infer on an input of the served shapes, then restores the RNG."""
        saved = self.policy._rng
        try:
            layers = np.zeros(self.patcher.depth, bool); layers[0] = True
            key = self.jax.random.key(0)
            self.patcher.steer(dict(example), dict(example, prompt='warm up'), key, layers, 1., 1, key)
            self.policy.infer(dict(example))
        finally:
            self.policy._rng = saved


def example_input():
    return {'observation/image': np.zeros((224, 224, 3), np.uint8), 'observation/wrist_image': np.zeros((224, 224, 3), np.uint8),
            'observation/state': np.zeros(8), 'prompt': 'put the bowl on the plate'}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--port', type=int, required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--log-dir', required=True); a = p.parse_args()
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    import patching as P
    policy = policy_config.create_trained_policy(config.get_config('pi05_libero'), a.checkpoint)
    if policy._is_pytorch_model:
        raise RuntimeError('JAX policy required')
    server = SteeredPolicy(policy, P.Patcher(policy), a.log_dir)
    # Compile before the port opens: the websocket keepalive would drop a client waiting through a first-request compile.
    started = time.monotonic(); server.warm_up(example_input()); print('warm up %.1f s' % (time.monotonic() - started), flush=True)
    WebsocketPolicyServer(server, host='127.0.0.1', port=a.port, metadata=policy.metadata).serve_forever()


if __name__ == '__main__':
    main()
