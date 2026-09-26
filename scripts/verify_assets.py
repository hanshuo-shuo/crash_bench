#!/usr/bin/env python3
"""CPU-side complete asset fingerprint before any GPU benchmark result."""
import hashlib
import json
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
assets=Path(os.environ.get('CB_ASSETS','/projects/p33100/siosio/crashbench_safelibero'))
config=json.loads((root/'configs/reproduction.json').read_text())
staging=json.loads((assets/'staging.json').read_text())
assert staging['config']==config
assert set(['checkpoint','tokenizer','bert','groundingdino','python','runtime','torch','oci'])<=set(staging['stages'])
upstream=root/'third_party/vlsa-aegis'
assert (upstream/'.git/HEAD').read_text().strip()==config['upstream_commit']
initials=sorted((upstream/'safelibero/libero/libero/init_files').rglob('*.pruned_init'))
assert len(initials)==32,len(initials)
checkpoint=Path(staging['stages']['checkpoint']['path'])
assert (checkpoint/'params').is_dir()
assert list((checkpoint/'assets').rglob('norm_stats.json')),'Missing normalization statistics'
paths=initials+sorted((upstream/'safelibero/libero/libero/bddl_files').rglob('*.bddl'))
paths += [p for p in sorted(checkpoint.rglob('*')) if p.is_file()]
paths += [Path(staging['stages']['tokenizer']['path']),Path(staging['stages']['groundingdino']['path'])]
paths += [p for p in sorted(Path(staging['stages']['bert']['path']).rglob('*')) if p.is_file()]
manifest=[]
for p in paths:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
job=os.environ.get('SLURM_JOB_ID','manual')
output=assets/('verified_'+job+'.json')
assert not output.exists()
output.write_text(json.dumps({'upstream_commit':config['upstream_commit'],'container_manifest_sha256':config['container_manifest_sha256'],'job':job,'files':manifest},indent=2)+'\n')
link=assets/'VERIFIED.json'
assert not link.exists() and not link.is_symlink()
link.symlink_to(output.name)
print(json.dumps({'verified_files':len(manifest),'bytes':sum(x['bytes'] for x in manifest),'manifest':str(output)}))
