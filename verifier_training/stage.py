"""Network downloads only on the login node; no model loading or installation."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi, snapshot_download
from .common import *


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    info=HfApi(token=False).model_info(MODEL_ID,files_metadata=True)
    revision=info.sha
    require(len(revision)==40,'Unresolved model revision')
    write(out/'STAGING.json',{'model_id':MODEL_ID,'revision':revision})
    snapshot_download(MODEL_ID,revision=revision,token=False,local_dir=str(out/'model'),
        allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'],max_workers=2)
    files={str(p.relative_to(out/'model')):sha(p) for p in (out/'model').rglob('*') if p.is_file() and '.cache' not in p.parts}
    # Check authoritative Hub LFS hashes when supplied, then pin every local file.
    for s in info.siblings:
        if s.rfilename in files and s.lfs:
            expected=s.lfs['sha256'] if isinstance(s.lfs,dict) else s.lfs.sha256
            require(files[s.rfilename]==expected,'Hub LFS hash mismatch')
    wheels={}
    (out/'wheels').mkdir()
    for package,version in [('peft','0.16.0'),('accelerate','1.8.1')]:
        with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json') as r:meta=json.load(r)
        wheel=next(x for x in meta['urls'] if x['filename'].endswith('py3-none-any.whl'))
        target=out/'wheels'/wheel['filename'];urllib.request.urlretrieve(wheel['url'],target)
        require(sha(target)==wheel['digests']['sha256'],'Wheel hash mismatch')
        wheels[target.name]=sha(target)
    write(out/'STAGED.json',{'model_id':MODEL_ID,'model_revision':revision,'files':files,'wheels':wheels,
        'base_runtime':'/projects/p33100/siosio/envs/openpi/bin/python',
        'base_packages':{'torch':'2.7.1','transformers':'4.53.2','numpy':'1.26.4','pillow':'11.2.1'},
        'overlay_packages':{'peft':'0.16.0','accelerate':'1.8.1'}})
    print(json.dumps({'staged':str(out),'revision':revision,'model_files':len(files)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);main(p.parse_args().out)
