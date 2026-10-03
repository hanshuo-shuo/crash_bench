"""Initial action-query to vision-key signal at diffusion t=1, frozen RNG.

KNOWS-inspired K=1 adaptation, not the paper's trajectory-level reproduction.
Native cached-prefix execution and traced suffix velocity are both retained.
"""
import numpy as np

def suffix_trace(self, embedded, positions, mask, adarms_cond, kv_cache, image_mask):
    import jax
    import jax.numpy as jnp
    from openpi.models import gemma,lora
    xs=jax.tree.map(lambda x:x.astype(self.embed_dtype),embedded)
    parameters=self.variables['params']['layers'];mask=jnp.asarray(mask)[:,None,:,:]
    block=gemma.Block(configs=self.configs,dropout=self.dropout,dropout_bdims=self.dropout_bdims,parent=None)
    config=self.configs[1]
    if config.num_kv_heads!=1:raise ValueError('This audited hook expects native single KV head')
    norm=gemma.RMSNorm(parent=None)
    query=lora.Einsum(shape=(config.num_heads,config.width,config.head_dim),
        lora_config=config.lora_configs.get('attn'),parent=None)
    n=image_mask.shape[1]
    def step(values,items):
        p,cache=items
        normalized,_=norm.apply({'params':p['pre_attention_norm_1']},values[1],adarms_cond[1])
        q=query.apply({'params':p['attn']['q_einsum_1']},'BTD,NDH->BTNH',normalized)
        q=gemma._apply_rope(q,positions=positions)*(config.head_dim**-.5)
        keys=cache[0][:,:n,0,:]
        logits=jnp.einsum('BTNH,BSH->BNTS',q,keys,preferred_element_type=jnp.float32)
        logits=jnp.where(image_mask[:,None,None,:],logits,-2.3819763e38)
        attention=jax.nn.softmax(logits,axis=-1).mean(axis=2)
        output,_=block.apply({'params':p},values,cache,positions,mask,adarms_cond,True)
        return output,attention
    output,attention=jax.lax.scan(step,xs,(parameters,kv_cache))
    final=self.final_norms[1](output[1],adarms_cond[1])[0]
    return final,jnp.swapaxes(attention,0,1)


def attention_hook(self,observation,noise):
    import jax.numpy as jnp
    from openpi.models import model as model_module
    from openpi.models.pi0 import make_attn_mask
    obs=model_module.preprocess_observation(None,observation,train=False)
    prefix,pm,pa=self.embed_prefix(obs)
    _,cache=self.PaliGemma.llm([prefix,None],positions=jnp.cumsum(pm,axis=1)-1,mask=make_attn_mask(pm,pa))
    suffix,sm,sa,condition=self.embed_suffix(obs,noise,jnp.ones((noise.shape[0],)))
    mask=jnp.concatenate([jnp.broadcast_to(pm[:,None,:],(sm.shape[0],sm.shape[1],pm.shape[1])),make_attn_mask(sm,sa)],axis=-1)
    positions=jnp.sum(pm,axis=-1)[:,None]+jnp.cumsum(sm,axis=-1)-1
    n=sum((im.shape[1]//14)*(im.shape[2]//14) for im in obs.images.values())
    traced,attention=self.PaliGemma.llm([None,suffix],positions=positions,mask=mask,
        adarms_cond=[None,condition],kv_cache=cache,image_mask=pm[:,:n],method='ra_suffix_trace')
    (_,native),_=self.PaliGemma.llm([None,suffix],positions=positions,mask=mask,adarms_cond=[None,condition],kv_cache=cache)
    return dict(action_vision_attention=attention,trace_velocity=self.action_out_proj(traced[:,-self.action_horizon:]),
        native_velocity=self.action_out_proj(native[:,-self.action_horizon:]),noise=noise)


def initialize(owner):
    from openpi.models import gemma
    from openpi.shared import nnx_utils
    gemma.Module.ra_suffix_trace=suffix_trace
    owner.policy._model.__class__.ra_attention=attention_hook
    owner.attention_extract=nnx_utils.module_jit(owner.policy._model.ra_attention)


def extract(owner,obs):
    import jax.numpy as jnp
    from openpi.models import model as model_module
    from serve import digest
    obs=dict(obs);obs.pop('__ra_attention__');before=owner.key()
    transformed=owner.policy._input_transform(owner.jax.tree.map(lambda x:x,obs))
    inputs=owner.jax.tree.map(lambda x:jnp.asarray(x)[None,...],transformed)
    observation=model_module.Observation.from_dict(inputs)
    model=owner.policy._model
    sample_rng=owner.jax.random.split(owner.policy._rng)[1]
    noise=owner.jax.random.normal(sample_rng,(1,model.action_horizon,model.action_dim))
    result=owner.jax.tree.map(lambda x:np.asarray(x[0]),owner.attention_extract(observation,noise))
    difference=result['trace_velocity'].astype(float)-result['native_velocity'].astype(float)
    result['metadata']=dict(input_sha256=digest(obs),rng_before=before,rng_after=owner.key(),timestep=1.,
        action_queries='All native action-horizon queries, mean after per-query vision-only softmax',
        image_order=list(observation.images),patches_per_image=[int((v.shape[1]//14)*(v.shape[2]//14)) for v in observation.images.values()],
        layers='Zero-based array index 0..17',heads='Zero-based array index 0..7',
        paper_unit_interpretation='One-based layer12/head3 = indices11/2; explicit adaptation, not official reproduction',
        temporal_window_K=1,trace_native_linf=float(np.abs(difference).max()),
        trace_native_relative_l2=float(np.linalg.norm(difference)/max(np.linalg.norm(result['native_velocity']),1e-12)))
    return result
