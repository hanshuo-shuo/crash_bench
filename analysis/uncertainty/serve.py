"""Native production inference with a separate uncertainty RPC and strict numeric gate."""
import argparse
import threading
import time
from pathlib import Path
import numpy as np
from common import atomic_json, config, noise_seed
from sampling import Sampler


class ObserverPolicy:
    def __init__(self, policy, jax, cfg, root, norm_file):
        self.policy, self.jax, self.cfg, self.root = policy, jax, cfg, root
        self.sampler = Sampler(policy, cfg, norm_file)
        self.lock = threading.Lock(); self.run = None; self.n = 0; self.validated = False

    def key(self):
        return np.asarray(self.jax.random.key_data(self.policy._rng), dtype='<u4').tobytes().hex()

    def infer(self, data):
        with self.lock:
            if '__uncertainty_reset__' in data:
                self.run = data['run_id']; self.seed = int(data['seed']); self.scene = data['scene_id']
                self.enabled = bool(data['enabled']); self.n = 0
                self.policy._rng = self.jax.random.key(self.seed)
                return dict(run_id=self.run, seed=self.seed, requests=0, key_hex=self.key(), scales=self.sampler.scales)
            if '__uncertainty_snapshot__' in data:
                return dict(run_id=self.run, seed=self.seed, requests=self.n, key_hex=self.key())
            if self.run is None:
                raise RuntimeError('Run reset required')
            before = self.policy._rng; key_before = self.key(); start = time.monotonic()
            result = self.policy.infer(data)
            native_seconds = time.monotonic()-start; after = self.policy._rng
            self.n += 1
            evidence = dict(run_id=self.run, infer_index=self.n, rng_before=key_before, rng_after=self.key(),
                            native_seconds=native_seconds, diagnostic_seconds=0., disagreement=None)
            if self.enabled:
                seeds = [noise_seed(self.scene, self.seed, self.n, i) for i in range(8)]
                values, metric, diag = self.sampler.infer(data, seeds, validate=not self.validated)
                evidence.update(diag, disagreement=metric, diagnostic_seconds=diag['batch_seconds'])
                if not self.validated:
                    # Native production output brackets the diagnostic at the very same RNG/input.
                    self.policy._rng = before
                    try:
                        again = self.policy.infer(data)
                    finally:
                        self.policy._rng = after
                    evidence['native_action_array_equal'] = bool(np.array_equal(result['actions'], again['actions']))
                    evidence['native_action_max_abs'] = float(np.max(np.abs(result['actions']-again['actions'])))
                    atomic_json(self.root/'SAMPLING_VALIDATION.json', evidence)
                    np.savez_compressed(self.root/'sampling_validation_actions.npz', batch=values,
                                        serial=np.asarray(evidence['serial_actions']), native_before=result['actions'],
                                        native_after=again['actions'])
                    if not evidence['serial_batch_passed'] or not evidence['native_action_array_equal']:
                        raise RuntimeError('Frozen sampling or native-action equality gate failed; inspect SAMPLING_VALIDATION.json')
                    self.validated = True
            if self.key() != evidence['rng_after']:
                raise RuntimeError('Production RNG advanced during diagnostics')
            evidence.pop('serial_actions', None)
            result['uncertainty'] = evidence
            return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--port', type=int, required=True); p.add_argument('--checkpoint', type=Path, required=True)
    a = p.parse_args()
    import jax
    from openpi.policies import policy_config
    from openpi.training import config as training_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    cfg = config(); policy = policy_config.create_trained_policy(training_config.get_config('pi05_libero'), str(a.checkpoint))
    if policy._is_pytorch_model:
        raise RuntimeError('Pinned native JAX model required')
    norm_file = a.checkpoint/'assets/physical-intelligence/libero/norm_stats.json'
    wrapped = ObserverPolicy(policy, jax, cfg, a.root, norm_file)
    atomic_json(a.root/'POLICY_READY.json', dict(checkpoint=str(a.checkpoint),
                model_action_dim=policy._model.action_dim, action_horizon=policy._model.action_horizon,
                jax_version=jax.__version__, devices=[str(x) for x in jax.devices()], scales=wrapped.sampler.scales.tolist()))
    WebsocketPolicyServer(wrapped, host='127.0.0.1', port=a.port, metadata=policy.metadata).serve_forever()


if __name__ == '__main__':
    main()
