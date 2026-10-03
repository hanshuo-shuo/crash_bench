"""Finish only the two fixed external states, retaining all completed parent runs."""
import json
import os
from pathlib import Path
import sys
from run import configure, execute, summarize, sha, write
from contract import PILOT

def recover(root,parent):
    stop=json.loads((parent/'STOP.json').read_text())
    if stop.get('reason')!='Illegal historical command':raise RuntimeError('Unexpected parent failure')
    initial=[s for s in PILOT if 'prefix_steps' not in s]
    inherited=[]
    for state in initial:
        for variant in ['center','side']:
            directory=parent/'runs'/(state['id']+'_'+variant)
            verified=json.loads((directory/'VERIFIED.json').read_text())
            if not verified['passed']:raise RuntimeError('Unverified inherited run')
            for name,value in verified['hashes'].items():
                if sha(directory/name)!=value:raise RuntimeError('Inherited bytes changed')
            inherited.append(dict(path=str(directory),verified_sha256=sha(directory/'VERIFIED.json')))
    for name in ['reference.py','geometry.py','reset_forward.py']:
        here=Path(__file__).resolve().parents[1]/'feasibility'/name
        previous=parent/'source/experiments/feasibility'/name
        if sha(here)!=sha(previous):raise RuntimeError('Witness implementation changed')
    write(root/'INHERITED.json',dict(parent=str(parent),runs=inherited,reason='completed original runs retained without rerun'))
    configure(root)
    for state in PILOT:
        if 'prefix_steps' in state:
            for variant in ['center','side']:execute(root,state,variant,audited_sign_replay=True)
    metrics=summarize(root,PILOT,{s['id']:parent for s in initial})
    write(root/'COMPLETE.json',dict(passed=True,stage='external_recovery',metrics=metrics,
          parent=str(parent),code_commit=os.environ['CB_CODE_COMMIT']))

if __name__=='__main__':
    root,parent=map(Path,sys.argv[1:3])
    try:recover(root,parent)
    except BaseException as e:
        write(root/'STOP.json',dict(type=type(e).__name__,reason=str(e),job=os.environ.get('SLURM_JOB_ID')))
        raise
