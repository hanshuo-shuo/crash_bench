"""Read-only native-weight layer hook, prepared for a later authorized GPU stage.

Not yet validated on the deployed model. It never changes action sampling or
loads another policy. Native and traced final arrays must be saved for audit.
"""
import argparse
from pathlib import Path
import sys
import numpy as np

BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'experiments/feasibility'))
from serve import ControlledPolicy,digest


def gemma_trace(self, embedded, positions, mask, valid_mask, image_mask):
    import jax
    import jax.numpy as jnp
    from openpi.models import gemma
    embedded=jax.tree.map(lambda x:x.astype(self.embed_dtype),embedded)
    mask=jnp.asarray(mask)[:,None,:,:]
    params=self.variables['params']['layers']
    block=gemma.Block(configs=self.configs,dropout=self.dropout,
        dropout_bdims=self.dropout_bdims,parent=None)
    def pool(value,valid):
        m=valid.astype(jnp.float32)
        return (value.astype(jnp.float32)*m[...,None]).sum(1)/jnp.maximum(m.sum(1,keepdims=True),1.)
    def step(xs,parameters):
        output,_=block.apply({'params':parameters},xs,None,positions,mask,[None]*len(self.configs),True)
        return output,(pool(output[0],valid_mask),pool(output[0],image_mask))
    output,stack=jax.lax.scan(step,embedded,params)
    final=self.final_norms[0](output[0],None)[0]
    return dict(layers=jnp.swapaxes(stack[0],0,1),image_layers=jnp.swapaxes(stack[1],0,1),
        trace_final=pool(final,valid_mask))


def feature_hook(self, observation):
    import jax.numpy as jnp
    from openpi.models import model as model_module
    from openpi.models.pi0 import make_attn_mask
    obs=model_module.preprocess_observation(None,observation,train=False)
    tokens,valid,ar=self.embed_prefix(obs)
    n=sum((im.shape[1]//14)*(im.shape[2]//14) for im in obs.images.values())
    image_mask=valid.at[:,n:].set(False)
    result=self.PaliGemma.llm([tokens,None],positions=jnp.cumsum(valid,axis=1)-1,
        mask=make_attn_mask(valid,ar),valid_mask=valid,image_mask=image_mask,method='ra_trace')
    (native,_),_=self.PaliGemma.llm([tokens,None],positions=jnp.cumsum(valid,axis=1)-1,mask=make_attn_mask(valid,ar))
    def pool(value,mask):
        mask=mask.astype(jnp.float32)
        return (value.astype(jnp.float32)*mask[...,None]).sum(1)/jnp.maximum(mask.sum(1,keepdims=True),1.)
    result['native_final']=pool(native,valid)
    result['projected_vision']=pool(tokens,image_mask)
    raw=[];masks=[]
    for name,im in obs.images.items():
        _,aux=self.PaliGemma.img(im,train=False)
        encoded=aux['encoded'];raw.append(encoded)
        masks.append(jnp.broadcast_to(obs.image_masks[name][:,None],encoded.shape[:2]))
    result['own_vision_tower']=pool(jnp.concatenate(raw,axis=1),jnp.concatenate(masks,axis=1))
    return result


class LayerPolicy(ControlledPolicy):
    def __init__(self,policy,jax,directory):
        super().__init__(policy,jax,directory)
        from openpi.models import gemma
        from openpi.shared import nnx_utils
        gemma.Module.ra_trace=gemma_trace
        policy._model.__class__.ra_layer_features=feature_hook
        self.extract=nnx_utils.module_jit(policy._model.ra_layer_features)
    def infer(self,obs):
        if '__ra_layers__' not in obs:return super().infer(obs)
        with self.lock:
            import jax.numpy as jnp
            from openpi.models import model as model_module
            obs=dict(obs);obs.pop('__ra_layers__');saved=self.policy._rng;before=self.key()
            transformed=self.policy._input_transform(self.jax.tree.map(lambda x:x,obs))
            inputs=self.jax.tree.map(lambda x:jnp.asarray(x)[None,...],transformed)
            try:
                result=self.jax.tree.map(lambda x:np.asarray(x[0]),self.extract(model_module.Observation.from_dict(inputs)))
            finally:self.policy._rng=saved
            if not all(np.isfinite(x).all() for x in result.values()):raise RuntimeError('Nonfinite readout')
            result['tokenized_prompt']=np.asarray(transformed['tokenized_prompt'])
            result['tokenized_prompt_mask']=np.asarray(transformed['tokenized_prompt_mask'])
            difference=result['trace_final'].astype(float)-result['native_final'].astype(float)
            result['metadata']=dict(input_sha256=digest(obs),rng_before=before,rng_after=self.key(),
                trace_native_linf=float(np.abs(difference).max()),
                trace_native_relative_l2=float(np.linalg.norm(difference)/max(np.linalg.norm(result['native_final']),1e-12)),
                weights_frozen=True,training_steps=0,action_requests=0,
                scope='Frozen own vision-tower encoded output, projected vision, all18 intermediate native-weight residual layers, final normalized prefix')
            return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--cache',required=True);a=p.parse_args()
    import jax
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),a.checkpoint)
    WebsocketPolicyServer(LayerPolicy(policy,jax,a.cache),host='127.0.0.1',port=a.port,metadata=policy.metadata).serve_forever()


if __name__=='__main__':main()
