"""Independent initial-input, prompt, RNG and executed-action mapping checks."""
import argparse,hashlib,json
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

def main(root,out):
    protocol=json.loads((root/'protocol.json').read_text());reports=[]
    for folder in sorted(root.glob('e*/*')):
        if not (folder/'summary.json').exists():continue
        features=json.loads((folder/'FEATURE_AUDIT.json').read_text())
        with np.load(folder/'initial_policy_input.npz') as z:obs={k:z[k] for k in z.files}
        obs['prompt']=protocol['policy_prompt'];ih=digest(obs)
        rows=[json.loads(x) for x in (folder/'policy.jsonl').read_text().splitlines()]
        steps=[json.loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()][1:]
        assert ih==features['input_sha256']==rows[0]['diagnostic']['input_sha256']
        assert features['rng_before']==features['rng_after']==rows[0]['diagnostic']['rng_before']
        assert all(r['prompt']==protocol['policy_prompt'] for r in rows)
        for a,b in zip(rows,rows[1:]):assert a['diagnostic']['rng_after']==b['diagnostic']['rng_before']
        for row in steps:
            p=next(p for p in rows if p['step']<=row['step']<p['step']+5)
            expected=np.asarray(p['actions'][row['step']-p['step']],float);expected[6]=np.clip(expected[6],-1,1)
            assert np.array_equal(expected,np.asarray(row['action']))
        with np.load(folder/'initial_features.npz') as z:
            delta=float(np.max(np.abs(z['action_seed7']-np.asarray(rows[0]['actions']))))
            shapes={k:list(z[k].shape) for k in ('prefix_final','image_embedding','image_prefix_final')}
            assert all(np.isfinite(z[k]).all() for k in ('prefix_final','image_embedding','image_prefix_final','valid_tokens','valid_image_tokens','proposal_actions','action_seed7','proprio','risk_scores'))
            # Websocket server adds a timing dictionary; its object-array envelope
            # is not a scientific feature and is never unpickled.
            assert set(z.files)=={'prefix_final','image_embedding','image_prefix_final','valid_tokens','valid_image_tokens','proposal_actions','action_seed7','proprio','risk_scores','server_timing'}
        reports.append(dict(run=str(folder.relative_to(root)),passed=True,initial_input_sha256=ih,rollout_inferences=len(rows),executed_actions=len(steps),features=shapes,seed7_feature_check_vs_first_rollout_linf=delta,first_rollout_matches_feature_check=delta<=1e-6,read_only_feature_and_proposal_inferences=6,feature_forward_passes=2))
    result=dict(passed=True,runs=reports,rollout_inferences=sum(x['rollout_inferences'] for x in reports),additional_unexecuted_policy_inferences=6*len(reports),feature_forward_passes=2*len(reports),scope='Exact initial inputs preserved. Subsequent inference input digests, output actions, prompt and full executed physics/controller states preserved; later camera tensors were not separately archived.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.root,a.output)
