"""Public frozen baseline download only; no model execution on login nodes."""
import hashlib,json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
root=Path('/projects/p33100/siosio/crashbench_safelibero/reach_avoid_assets')
model='facebook/dinov2-small'
revision=HfApi(token=False).model_info(model).sha
path=root/('dinov2-small_'+revision);path.mkdir(parents=True,exist_ok=True)
snapshot_download(model,revision=revision,local_dir=path,token=False,
    allow_patterns=['config.json','preprocessor_config.json','model.safetensors'])
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
receipt=dict(model=model,revision=revision,path=str(path),files=files,api_calls=0,
    input='Both actual 224x224 policy RGB views, original field of view, model normalization only',
    features='Mean final patch tokens and CLS, concatenated across views; frozen backbone')
(path/'ASSET.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
