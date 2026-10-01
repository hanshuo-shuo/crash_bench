"""Native JAX inference plus auditable common-input/common-RNG output replay."""
import argparse
import hashlib
import threading
from pathlib import Path
import numpy as np

def digest(obs):
 h=hashlib.sha256()
 for k,v in sorted(obs.items()):
  h.update(k.encode())
  if isinstance(v,str):h.update(v.encode())
  else:
   a=np.ascontiguousarray(v);h.update(str((a.dtype.str,a.shape)).encode());h.update(a.tobytes())
 return h.hexdigest()

class ControlledPolicy:
 def __init__(self, policy, jax, cache_directory=None):
  self.directory=Path(cache_directory) if cache_directory else None
  if self.directory:self.directory.mkdir(exist_ok=True)
  self.policy,self.jax=policy,jax;self.lock=threading.Lock();self.cache={};self.run=None;self.n=0
 def key(self):
  return np.asarray(self.jax.random.key_data(self.policy._rng),dtype='<u4').tobytes().hex()
 def infer(self,obs):
  with self.lock:
   if '__paired_reset_rng__' in obs:
    seed=obs['__paired_reset_rng__']
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid seed')
    self.policy._rng=self.jax.random.key(seed);self.run=obs['run_id'];self.n=0
    return {'seed':seed,'run_id':self.run,'requests':0,'key_hex':self.key()}
   if '__diagnostic_snapshot__' in obs:return {'key_hex':self.key(),'requests':self.n,'run_id':self.run}
   before=self.key(); ih=digest(obs);cachekey=(before,ih)
   # Native infer always executes, so RNG transitions and numeric discrepancies remain visible.
   raw=self.policy.infer(obs);self.n+=1;after=self.key()
   file=self.directory/(before+'_'+ih+'.npz') if self.directory else None
   if cachekey not in self.cache and file and file.exists():
    saved=np.load(file);self.cache[cachekey]=(saved['actions'],str(saved['rng_after']))
   actions=np.asarray(raw['actions']).copy();difference=0.;replayed=cachekey in self.cache
   if replayed:
    saved,saved_after=self.cache[cachekey]
    if saved_after!=after:raise RuntimeError('Native RNG transition changed')
    difference=float(np.max(np.abs(saved-actions)));raw['actions']=saved.copy()
   else:
    self.cache[cachekey]=(actions,after)
    if file:
     temporary=file.with_suffix('.tmp.npz');np.savez(temporary,actions=actions,rng_after=after);temporary.replace(file)
   raw['diagnostic']={'input_sha256':ih,'rng_before':before,'rng_after':after,'request_index':self.n,
    'run_id':self.run,'output_replayed':replayed,'native_output_linf_difference':difference}
   return raw

def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--cache');a=p.parse_args()
 import jax
 from openpi.policies import policy_config
 from openpi.training import config
 from openpi.serving.websocket_policy_server import WebsocketPolicyServer
 policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),a.checkpoint)
 if policy._is_pytorch_model:raise RuntimeError('JAX policy required')
 WebsocketPolicyServer(ControlledPolicy(policy,jax,a.cache),host='127.0.0.1',port=a.port,metadata=policy.metadata).serve_forever()
if __name__=='__main__':main()
