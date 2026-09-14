"""Reconstruct terminal-state illustrations from existing C traces, without stepping."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crashbench.envs import LiberoEnv
from scripts.expansion.hash_tree_manifest import resolve_git_head


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def last_state(path):
    final=None
    with gzip.open(path,'rb') as f:
        while True:
            try:row=pickle.load(f)
            except EOFError:break
            final=row
    if final is None or 'state_after' not in final:raise ValueError('missing terminal trace state')
    return final


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Quest Slurm only')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=Path('/projects/p33100/siosio/crashbench_repeat_value/bcaeda762d6c_detour_6240307')
    complete=json.loads((source/'complete.json').read_text())
    if sha(source/'C.json')!=complete['C_sha256']:raise ValueError('historical C changed')
    records=json.loads((source/'C.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    env=LiberoEnv('libero_spatial',0,model_family='pi0',seed=2027)
    plt.rcParams.update({'font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(3,3,figsize=(5.5,6.35))
    evidence=[]
    try:
        for i,(episode,label) in enumerate([
            ('e15','Mixed Base outcomes'),('e18','Benefit in every block'),('e21','Later task completion')]):
            folder=source/episode
            with (folder/'bundle.pkl').open('rb') as f:context=pickle.load(f)
            bundle=context['bundle'];bundle.assert_integrity()
            for j,(repeat,option,title) in enumerate([(4,0,'Base: first repeat'),(5,0,'Base: second repeat'),(4,1,'Detour: first repeat')]):
                record=next(r for r in records if r['episode_id']==episode and r['repeat']==repeat and r['option']==option)
                trace=folder/f'r{repeat}_o{option}.pkl.gz'
                if sha(trace)!=record['trace_sha256']:raise ValueError('historical trace changed')
                terminal=last_state(trace);state=np.asarray(terminal['state_after'])
                obs=env.reset_to_exact(state,model_xml=bundle.model_xml)
                if not np.array_equal(env.flat_state(),state):raise ValueError('reconstruction changed physical state')
                frame=np.asarray(obs['agentview_image'])[::-1,::-1].copy()
                v=record['horizons']['440']
                reason={'success':'Task complete','accident':'Recorded accident','timeout':'Unfinished at deadline'}.get(v['reason'],v['reason'])
                ax=axes[i,j];ax.imshow(frame);ax.set_xticks([]);ax.set_yticks([])
                ax.set_title(title,fontsize=8,pad=5)
                ax.set_xlabel(reason+'\nEpisode action '+str(v['steps']),fontsize=8,
                    color='#1d7467' if v['success'] else '#9e432c',labelpad=4)
                for spine in ax.spines.values():spine.set_visible(False)
                if j==0:ax.set_ylabel(episode+'\n'+label,fontsize=8)
                name=f'{episode}_r{repeat}_o{option}.png';plt.imsave(args.output/name,frame)
                evidence.append(dict(episode=episode,repeat=repeat,option=option,
                    bundle_sha256=record['bundle_sha256'],trace_sha256=record['trace_sha256'],
                    horizon=440,terminal=v,state_equal_after_reconstruction=True,image=name))
    finally:env.env.close()
    fig.tight_layout(pad=.65,w_pad=.7,h_pad=1.3)
    for ext in ('png','svg','pdf'):fig.savefig(args.output/('terminal_examples.'+ext),dpi=230)
    plt.close(fig)
    meta=dict(commit=resolve_git_head(ROOT),job=os.environ['SLURM_JOB_ID'],
        source=str(source),C_sha256=complete['C_sha256'],
        operation='terminal-state reconstruction from saved traces; zero environment step calls',
        new_rollouts=0,new_model_queries=0,examples=evidence)
    (args.output/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in args.output.iterdir() if p.is_file()},indent=2)+'\n')
    print('Nine terminal-state reconstructions complete; no new rollouts',flush=True)


if __name__=='__main__':main()
