"""Isolated OpenPI server: explicit reset RPC, otherwise native infer/split."""
import argparse
import hashlib
import json
import threading

class SeededPolicy:
    def __init__(self, policy, random, key_bytes):
        self.policy, self.random, self.key_bytes = policy, random, key_bytes
        self.lock = threading.Lock()
        self.run_id = None
        self.requests = 0

    def infer(self, obs):
        with self.lock:
            if '__paired_reset_rng__' in obs:
                if set(obs) != {'__paired_reset_rng__', 'run_id'}:
                    raise ValueError('Malformed reset')
                seed = obs['__paired_reset_rng__']
                if type(seed) is not int or not 0 <= seed < 2**32:
                    raise ValueError('Seed must be uint32')
                self.policy._rng = self.random.key(seed)
                self.requests = 0
                self.run_id = obs['run_id']
                return {'seed': seed, 'run_id': self.run_id, 'requests': 0,
                        'key_sha256': hashlib.sha256(self.key_bytes(self.policy._rng)).hexdigest()}
            if self.run_id is None:
                raise RuntimeError('Explicit episode reset required')
            output = self.policy.infer(obs)
            self.requests += 1
            output['paired_rng'] = {'run_id': self.run_id, 'request_index': self.requests}
            return output

def main():
    p = argparse.ArgumentParser(); p.add_argument('--port', type=int, required=True); p.add_argument('--checkpoint', required=True)
    args = p.parse_args()
    import jax
    import numpy as np
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy = policy_config.create_trained_policy(config.get_config('pi05_libero'), args.checkpoint)
    if policy._is_pytorch_model:
        raise RuntimeError('This protocol requires native JAX RNG')
    wrapped = SeededPolicy(policy, jax.random, lambda k: np.asarray(jax.random.key_data(k), dtype='<u4').tobytes())
    print(json.dumps({'protocol': 'paired-v1', 'policy': 'pi05_libero', 'reset': 'jax.random.key(seed); native infer split unchanged'}), flush=True)
    WebsocketPolicyServer(wrapped, host='127.0.0.1', port=args.port, metadata=policy.metadata).serve_forever()

if __name__ == '__main__':
    main()
