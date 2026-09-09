"""Existing-trace diagnostics only; no policy or simulator imports."""
import argparse,csv,gzip,pickle
from pathlib import Path
import numpy as np

def equal(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if a is None or b is None:return a is b
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return np.array_equal(np.asarray(a),np.asarray(b))
def records(path):
    with gzip.open(path,'rb') as f:
        while True:
            try:yield pickle.load(f)
            except EOFError:return

def compare(a,b):
    from itertools import zip_longest
    first={k:None for k in ('action','state_before','state_after','policy_input')};counts={k:0 for k in first};n=0;length_mismatch=False
    for x,y in zip_longest(records(a),records(b)):
        n+=1
        if x is None or y is None:length_mismatch=True;continue
        for k in first:
            if not equal(x[k],y[k]):
                counts[k]+=1
                if first[k] is None:first[k]=int(x['event']['step'])
    return {'steps_compared_or_union':n,'length_mismatch':length_mismatch,**{k+'_first_difference':v for k,v in first.items()},**{k+'_different_steps':v for k,v in counts.items()}}
def main():
    import json
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();rows=[]
    for e in json.loads((a.run/'anchors.json').read_text()):
        if not e['triggered']:continue
        for phase,repeats in [('A',(0,1)),('B',(2,3))]:
            for option in (0,1):
                root=a.run/e['episode_id'];rows.append({'episode':e['episode_id'],'source':e['source'],'condition':e['condition'],'phase':phase,'option':option,**compare(root/f'r{repeats[0]}_o{option}.pkl.gz',root/f'r{repeats[1]}_o{option}.pkl.gz')})
    out=a.run/'analysis/repeat_trace_diagnostics.csv'
    with out.open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['episode','source','condition','phase','option'],lineterminator='\n');w.writeheader();w.writerows(rows)
    print(f'Compared {len(rows)} same-option repeat pairs, no new trajectories.')
if __name__=='__main__':main()
