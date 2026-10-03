"""Frozen native pi0.5, read-only action attention, and matched external vision."""
import argparse,json
from pathlib import Path
import numpy as np
from layer_features import LayerPolicy
import attention_features

class MatrixPolicy(LayerPolicy):
    def __init__(self,policy,jax,directory,vision):
        super().__init__(policy,jax,directory)
        attention_features.initialize(self)
        import torch
        from transformers import AutoModel,AutoImageProcessor
        torch.set_num_threads(1)
        self.torch=torch;self.vision=AutoModel.from_pretrained(vision,local_files_only=True).eval().to('cuda')
        self.vision_processor=AutoImageProcessor.from_pretrained(vision,local_files_only=True)
        self.vision_asset=json.loads((Path(vision)/'ASSET.json').read_text())
    def infer(self,obs):
        if '__ra_attention__' in obs:
            with self.lock:return attention_features.extract(self,obs)
        if '__ra_general_vision__' in obs:
            with self.lock:
                images=[obs['observation/image'],obs['observation/wrist_image']]
                batch=self.vision_processor(images=images,return_tensors='pt',do_resize=False,do_center_crop=False)
                with self.torch.inference_mode():out=self.vision(**{k:v.to('cuda') for k,v in batch.items()}).last_hidden_state
                return dict(dino_patch=out[:,1:].mean((0,1)).detach().cpu().numpy(),
                    dino_views=out[:,1:].mean(1).detach().cpu().numpy().reshape(-1),
                    dino_cls=out[:,0].detach().cpu().numpy().reshape(-1),metadata=self.vision_asset)
        return super().infer(obs)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--cache',required=True);p.add_argument('--vision',required=True);a=p.parse_args()
    import jax
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),a.checkpoint)
    WebsocketPolicyServer(MatrixPolicy(policy,jax,a.cache,a.vision),host='127.0.0.1',port=a.port,metadata=policy.metadata).serve_forever()
