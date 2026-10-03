"""Frozen, initial-only prefix readout hook. No training and no action edits."""
import argparse,hashlib,json,sys,types
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'experiments/feasibility'))
from serve import ControlledPolicy,digest

def prefix_features(self,obs):
    import jax.numpy as jnp
    from openpi.models import model as model_module
    from openpi.models.pi0 import make_attn_mask
    obs=model_module.preprocess_observation(None,obs,train=False)
    tokens,mask,ar=self.embed_prefix(obs)
    (out,_),_=self.PaliGemma.llm([tokens,None],mask=make_attn_mask(mask,ar),positions=jnp.cumsum(mask,axis=1)-1)
    # Image tokens precede text; number derives from actual patch14 inputs.
    n=sum((im.shape[1]//14)*(im.shape[2]//14) for im in obs.images.values())
    def pool(a,m):
        a=a.astype(jnp.float32);m=m.astype(jnp.float32)
        return (a*m[...,None]).sum(1)/jnp.maximum(m.sum(1,keepdims=True),1.)
    return dict(prefix_final=pool(out,mask),image_embedding=pool(tokens[:,:n],mask[:,:n]),image_prefix_final=pool(out[:,:n],mask[:,:n]),valid_tokens=mask.sum(1),valid_image_tokens=mask[:,:n].sum(1))

class FeaturePolicy(ControlledPolicy):
    def __init__(self,policy,jax,directory):
        super().__init__(policy,jax,directory)
        from openpi.shared import nnx_utils
        policy._model.__class__.fr1_prefix_features=prefix_features
        self.extract=nnx_utils.module_jit(policy._model.fr1_prefix_features)
    def infer(self,obs):
        if '__extract_initial_features__' not in obs:return super().infer(obs)
        with self.lock:
            obs=dict(obs);obs.pop('__extract_initial_features__');before=self.key();saved=self.policy._rng
            import jax.numpy as jnp
            from openpi.models import model as model_module
            inputs=self.policy._input_transform(self.jax.tree.map(lambda x:x,obs))
            inputs=self.jax.tree.map(lambda x:jnp.asarray(x)[None,...],inputs)
            observation=model_module.Observation.from_dict(inputs)
            try:
                # The action check brackets extraction at the same policy RNG.
                self.policy._rng=self.jax.random.key(7)
                action_before=np.asarray(self.policy.infer(obs)['actions'])
                a=self.jax.tree.map(lambda x:np.asarray(x[0]),self.extract(observation))
                b=self.jax.tree.map(lambda x:np.asarray(x[0]),self.extract(observation))
                self.policy._rng=self.jax.random.key(7)
                action_after=np.asarray(self.policy.infer(obs)['actions'])
                proposals=[]
                for seed in (101,102,103,104):
                    self.policy._rng=self.jax.random.key(seed);proposals.append(np.asarray(self.policy.infer(obs)['actions']))
                proposals=np.asarray(proposals)
            finally:self.policy._rng=saved
            max_feature=max(float(np.max(np.abs(a[k]-b[k]))) for k in a)
            delta=float(np.max(np.abs(action_before-action_after)))
            if not all(np.isfinite(x).all() for x in list(a.values())+[proposals]):raise RuntimeError('Nonfinite features/proposals')
            if before!=self.key():raise RuntimeError('Feature extraction advanced action RNG')
            a.update(proposal_actions=proposals,action_seed7=action_before,proprio=np.asarray(obs['observation/state']),risk_scores=np.asarray([float(np.linalg.norm(action_before[:5,:6])),float(proposals[:,:5,:6].var(0).mean())]))
            a['metadata']=dict(input_sha256=digest(obs),rng_before=before,rng_after=self.key(),feature_repeat_linf=max_feature,action_before_after_linf=delta,passed=bool(max_feature==0 and delta<=1e-6),weights_frozen=True,training_steps=0,feature_hook='final normalized PaliGemma prefix; valid-token mean',image_baseline='pre-fusion SigLIP image tokens projected to PaliGemma width; valid image-token mean',proposal_seeds=[101,102,103,104],proposal_count=4,no_policy_rng_change=True)
            return a

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--cache',required=True);a=p.parse_args()
    import jax
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),a.checkpoint)
    if policy._is_pytorch_model:raise RuntimeError('JAX only')
    WebsocketPolicyServer(FeaturePolicy(policy,jax,a.cache),host='127.0.0.1',port=a.port,metadata=policy.metadata).serve_forever()
if __name__=='__main__':main()
