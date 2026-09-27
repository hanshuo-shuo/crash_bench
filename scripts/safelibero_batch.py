"""Frozen official matrix and partial results; lightweight, safe on login nodes."""
import argparse
import json
from pathlib import Path
import subprocess
import os
import hashlib
from api_budget import atomic_json

SUITES=['safelibero_spatial','safelibero_goal','safelibero_object','safelibero_long']

def matrix():
    return [{'index':i,'mode':mode,'suite':suite,'level':level,'task':task,'episodes':list(range(50))}
            for i,(mode,suite,level,task) in enumerate((m,s,l,t) for m in ['nominal','aegis'] for s in SUITES for l in ['I','II'] for t in range(4))]

def evaluation_path(root,plan,index):
    paths=list((Path(root)/'runs').glob('%02d_*/evaluation/manifest.json'%index))
    inherited=plan.get('inherited_cells',{}).get(str(index))
    if len(paths)>1 or (paths and inherited):raise RuntimeError('Duplicate run root for matrix cell')
    if inherited:
        path=Path(inherited['evaluation_dir'])
        for name in ['manifest.json','episodes.json']:
            if hashlib.sha256((path/name).read_bytes()).hexdigest()!=inherited['sha256'][name]:
                raise RuntimeError('Inherited result changed: '+name)
        return path,inherited['code_commit']
    return (paths[0].parent,plan['code_commit']) if paths else (None,None)

def recovery_selection(root):
    root=Path(root);plan=json.loads((root/'batch.json').read_text())
    if not (root/'STOP.json').exists():raise RuntimeError('Recovery requires a stopped parent batch')
    if plan['cells']!=matrix():raise RuntimeError('Parent matrix differs from official frozen matrix')
    inherited={};partial={};pending=[]
    for cell in plan['cells']:
        index=cell['index'];path,commit=evaluation_path(root,plan,index)
        if path:
            manifest=json.loads((path/'manifest.json').read_text())
            if manifest['code_commit']!=commit:raise RuntimeError('Parent result revision mismatch')
            for key in ['mode','suite','level','task']:
                if manifest[key]!=cell[key]:raise RuntimeError('Parent result identity mismatch')
            rows=json.loads((path/'episodes.json').read_text()) if (path/'episodes.json').exists() else []
            if manifest['status']=='complete':
                if [r['episode'] for r in rows]!=cell['episodes']:raise RuntimeError('Incomplete parent cell claims completion')
                inherited[str(index)]={'evaluation_dir':str(path.resolve()),'code_commit':commit,'sha256':{name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ['manifest.json','episodes.json']}}
                continue
            partial[str(index)]={'evaluation_dir':str(path.resolve()),'retained_but_excluded_episode_indices':[r['episode'] for r in rows]}
        pending.append(index)
    return inherited,partial,pending

def collect(root):
    root=Path(root);plan=json.loads((root/'batch.json').read_text());groups={};complete=[];failed=[];seen=set()
    for cell in plan['cells']:
        path,expected_commit=evaluation_path(root,plan,cell['index'])
        if path is None:continue
        manifest=json.loads((path/'manifest.json').read_text())
        for key in ['mode','suite','level','task']:
            if manifest[key]!=cell[key]:raise RuntimeError('Run disagrees with matrix')
        if manifest['code_commit']!=expected_commit:raise RuntimeError('Mixed code revisions')
        if str(cell['index']) in plan.get('inherited_cells',{}) and manifest['status']!='complete':raise RuntimeError('Inherited cell is not complete')
        if manifest['status']=='failed':failed.append(cell['index'])
        record=path/'episodes.json'
        if not record.exists():continue
        rows=json.loads(record.read_text())
        if manifest['status']=='complete':
            if [r['episode'] for r in rows]!=cell['episodes']:raise RuntimeError('Completed run missing official episodes')
            complete.append(cell['index'])
        for r in rows:
            identity=(cell['index'],r['episode'])
            if r['episode'] not in cell['episodes'] or identity in seen:raise RuntimeError('Duplicate or invalid episode')
            seen.add(identity)
            for group in [cell['mode'],cell['mode']+'/'+cell['suite']+'/'+cell['level']]:
                g=groups.setdefault(group,{'episodes':0,'successes':0,'collisions':0,'safe_successes':0,'actions':0})
                g['episodes']+=1;g['successes']+=r['success'];g['collisions']+=r['collision'];g['safe_successes']+=r['safe_success'];g['actions']+=r['steps']
    for g in groups.values():
        g.update(task_success_rate=g['successes']/g['episodes'],collision_avoidance_rate=1-g['collisions']/g['episodes'],safe_success_rate=g['safe_successes']/g['episodes'],mean_executed_actions=g['actions']/g['episodes'])
    return {'complete_cells':complete,'failed_cells':failed,'completed_episodes':len(seen),'expected_episodes':3200,'groups':groups,'partial':len(complete)!=64}

def run_cell(root,index):
    root=Path(root);plan=json.loads((root/'batch.json').read_text());cell=plan['cells'][index]
    if (root/'STOP.json').exists():raise RuntimeError('Batch is stopped')
    if str(index) in plan.get('inherited_cells',{}):raise RuntimeError('Completed inherited cell must not rerun')
    run=root/'runs'/('%02d_%s_%s'%(index,plan['code_commit'][:12],os.environ['SLURM_JOB_ID']))
    os.environ.update(CB_EXPECTED_COMMIT=plan['code_commit'],CB_RUN_ROOT=str(run),CB_BATCH_ROOT=str(root))
    subprocess.run(['bash','setup/run_smoke.sh',cell['mode'],'--suite',cell['suite'],'--level',cell['level'],'--task',str(cell['task']),'--episode','0','--episode-count','50'],check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','status']);p.add_argument('root',type=Path);p.add_argument('--index',type=int);a=p.parse_args()
    if a.command=='run':run_cell(a.root,a.index)
    else:print(json.dumps(collect(a.root),indent=2))
