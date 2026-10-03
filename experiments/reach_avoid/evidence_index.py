"""Join immutable CPU labels to GPU observations/outcomes and audit file identity."""
import argparse,csv,hashlib,json
from pathlib import Path
from validation import verify_certificate,verify_steps
P=json.loads((Path(__file__).parent/'matrix_protocol.json').read_text())

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(cpu,gpu,out):
    out.mkdir(exist_ok=True,parents=True);rows=[];hashes={};brackets=[]
    for i,layout in enumerate(P['layouts']):
        for mechanism in P['mechanisms']:
            numeric=[]
            for variant in P['variants']:
                case=layout['id']+'_'+mechanism+'_'+variant
                label=cpu/('shard_'+str(i//2))/case;policy=gpu/('shard_'+str(i//2))/'output'/case
                ls=read(label/'summary.json');ps=read(policy/'summary.json');inherited=read(policy/'INHERITED.json')
                for name,value in inherited['hashes'].items():
                    if sha(label/name)!=value:raise RuntimeError('Label identity mismatch: '+case+'/'+name)
                for name,expected in read(label/'VERIFIED.json')['hashes'].items():
                    if sha(label/name)!=expected:raise RuntimeError('CPU verified file changed: '+case+'/'+name)
                steps=[json.loads(x) for x in (label/'steps.jsonl').read_text().splitlines()]
                geometry=read(label/'geometry.json')
                cert=verify_certificate(geometry,read(label/'MATERIAL_BALL.json'),read(label/'fixture.json'),steps[0],ls['initial_valid'],read(label/'CERTIFICATE.json'))
                verify_steps(steps,ls,geometry['fixed_goal'],cert,P['execution_horizon_T'])
                if ps['outcome'] in ('safe_completion','safe_timeout','collision'):
                    verified=read(policy/'VERIFIED.json')
                    for name,expected in verified['hashes'].items():
                        if sha(policy/name)!=expected:raise RuntimeError('GPU verified file changed: '+case+'/'+name)
                row=dict(case=case,scene_group=layout['id'],split=layout['split'],constructor=mechanism,variant=variant,
                    independent_label=ls['label'],reference_outcome=ls['outcome'],reference_steps=ls['steps'],
                    policy_outcome=ps['outcome'],policy_steps=ps['steps'],initial_valid=ls['initial_valid'],
                    layer_accepted=ps['layer_accepted'],attention_accepted=ps['attention_accepted'],
                    cpu_job=read(label.parent/'COMPLETE.json')['job'],gpu_job=read(policy.parent/'COMPLETE.json')['job'])
                rows.append(row)
                if variant not in ('open','irrelevant'):numeric.append((float(variant),ls['label']))
                for folder,kind in ((label,'labels'),(policy,'policy')):
                    for name in ('summary.json','CERTIFICATE.json','VERIFIED.json','initial_restore.json','FEATURE_AUDIT.json'):
                        path=folder/name
                        if path.exists():hashes[kind+'/'+case+'/'+name]=sha(path)
            neg=[x for x,l in numeric if l=='infeasible'];pos=[x for x,l in numeric if l=='feasible']
            lo=max(neg) if neg else None;hi=min(pos) if pos else None
            brackets.append(dict(scene_group=layout['id'],constructor=mechanism,largest_sampled_certified_gap=lo,
                smallest_sampled_witnessed_gap=hi,unresolved_bracket_width_m=hi-lo if lo is not None and hi is not None else None,
                sampled_unknown_gaps=[x for x,l in numeric if l=='unknown'],meaning='Sampled sufficient-label bracket, not exact physical boundary'))
    with (out/'state_outcomes.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for name,value in [('STATE_TABLE.json',rows),('UNKNOWN_BRACKETS.json',brackets),('EVIDENCE_HASHES.json',hashes)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    (out/'INDEPENDENT_READBACK_AUDIT.json').write_text(json.dumps(dict(passed=True,states=len(rows),
        checks='Inherited label identity, saved verifier hashes, recomputed actual-geometry certificates, persisted goal/horizon/label predicates',hashes=len(hashes)),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cpu',type=Path);p.add_argument('gpu',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.cpu,a.gpu,a.out)
