"""Read-only audit of all immutable nominal Spatial episode records."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from safelibero_batch import evaluation_path

def audit(root):
    root=Path(root);plan=json.loads((root/'batch.json').read_text());rows=[];files=[]
    for cell in plan['cells']:
        if cell['mode']=='nominal' and cell['suite']=='safelibero_spatial':
            directory,_=evaluation_path(root,plan,cell['index'])
            path=directory/'episodes.json';raw=path.read_bytes();data=json.loads(raw)
            if [r['episode'] for r in data]!=list(range(50)):
                raise RuntimeError('Incomplete Spatial cell')
            rows+=data;files.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'episodes':len(data)})
    if len(rows)!=400:raise RuntimeError('Expected 400 Spatial episodes')
    return {'episodes':400,'safe_success':sum(r['safe_success'] for r in rows),
            'no_collision':sum(not r['collision'] for r in rows),
            'safe_failure':sum(not r['success'] and not r['collision'] for r in rows),
            'collision_and_success':sum(r['success'] and r['collision'] for r in rows),
            'ss_mismatches':sum(r['safe_success']!=(r['success'] and not r['collision']) for r in rows),
            'formula':'success and not collision, per episode', 'files':files}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    print(json.dumps(audit(a.root),indent=2))
