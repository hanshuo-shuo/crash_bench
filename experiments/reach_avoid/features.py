"""Frozen common-process sensitivity check; saves tokens, actions and features."""
import argparse
from pathlib import Path
import sys
import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / 'experiments/feasibility_risk'))
from features import prefix_features
sys.path.insert(0, str(BASE / 'experiments/feasibility'))
from serve import ControlledPolicy, digest


class PromptPolicy(ControlledPolicy):
    def __init__(self, policy, jax, directory):
        super().__init__(policy, jax, directory)
        from openpi.shared import nnx_utils
        policy._model.__class__.ra_features = prefix_features
        self.extract = nnx_utils.module_jit(policy._model.ra_features)

    def infer(self, obs):
        if '__ra_prompt_check__' not in obs: return super().infer(obs)
        with self.lock:
            import jax.numpy as jnp
            from openpi.models import model as model_module
            obs = dict(obs); obs.pop('__ra_prompt_check__'); saved = self.policy._rng; before = self.key()
            transformed = self.policy._input_transform(self.jax.tree.map(lambda x: x, obs))
            inputs = self.jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed)
            try:
                features = self.jax.tree.map(lambda x: np.asarray(x[0]),
                    self.extract(model_module.Observation.from_dict(inputs)))
                self.policy._rng = self.jax.random.key(7)
                actions = np.asarray(self.policy.infer(obs)['actions'])
            finally: self.policy._rng = saved
            features.update(actions=actions,
                tokenized_prompt=np.asarray(transformed['tokenized_prompt']),
                tokenized_prompt_mask=np.asarray(transformed['tokenized_prompt_mask']))
            if not all(np.isfinite(v).all() for v in features.values()): raise RuntimeError('Nonfinite prompt diagnostic')
            features['metadata'] = dict(input_sha256=digest(obs), prompt=obs['prompt'],
                rng_before=before, rng_after=self.key(), action_seed=7,
                feature_hook='Frozen prefix final / image-prefix final / projected image embedding',
                weights_frozen=True, execution_actions=0)
            return features


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--cache', required=True)
    args = parser.parse_args()
    import jax
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy = policy_config.create_trained_policy(config.get_config('pi05_libero'), args.checkpoint)
    if policy._is_pytorch_model: raise RuntimeError('Expected pinned JAX policy')
    WebsocketPolicyServer(PromptPolicy(policy, jax, args.cache), host='127.0.0.1', port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == '__main__': main()
