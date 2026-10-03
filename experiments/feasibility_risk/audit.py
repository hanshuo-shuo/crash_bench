"""Independent persisted-record audit, standard library only; no execution."""
import argparse,gzip,hashlib,json,math
from pathlib import Path

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def intersects(a,b,lo,hi):
    low,high=0.,1.
    for x,y,l,h in zip(a,b,lo,hi):
        l-=1e-9;h+=1e-9
        if y==x:
            if x<l or x>h:return False
        else:
            u,v=sorted(((l-x)/(y-x),(h-x)/(y-x)))
            low=max(low,u);high=min(high,v)
            if low>high:return False
    return True

def audit_run(path):
    s=read(path/'summary.json');g=read(path/'geometry.json');verified=read(path/'VERIFIED.json')
    for f,h in verified['hashes'].items():
        if sha(path/f)!=h:raise AssertionError('Hash mismatch '+str(path/f))
    actors=set(g['robot']+g['target']);danger=set(g['protected'])|{b['id'] for b in g['boxes']}
    safe=True;previous=None;count=0;first=None
    with gzip.open(path/'samples.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);count+=1;hits=[]
            bad=any(c[2]<=0 and ((c[0] in actors and c[1] in danger) or (c[1] in actors and c[0] in danger)) for c in row['contacts'])
            if row['phase']!='native_cached':
                prev=row['points'] if previous is None else previous
                for i,(a,b) in enumerate(zip(prev,row['points'])):
                    for box in g['boxes']:
                        if intersects(a,b,box['lower'],box['upper']):hits.append([i,box['id']])
                previous=row['points']
            assert hits==row['sweep_hits'],str(path)
            if (bad or hits) and first is None:first=[row['step'],row['substep'],row['phase']]
            safe &= not(bad or hits)
            assert safe==row['safe_history'],str(path)
    rows=[json.loads(x) for x in (path/'steps.jsonl').read_text().splitlines()]
    for row in rows[1:]:assert len(row['action'])==7 and all(math.isfinite(v) and abs(v)<=1 for v in row['action'])
    assert len(rows)-1==s['steps'] and count==s['samples'] and safe==s['safe_history']
    assert rows[-1]['safe_success']==s['safe_success']
    if s['risk_label']=='horizon_safe_30':assert safe and s['steps']==30 and not s['illegal_policy_command']
    if s['risk_label']=='violation_within_30':assert first and 1<=first[0]<=30
    return dict(path=str(path),actions=len(rows)-1,samples=count,hashes=len(verified['hashes']),safe=safe,first_violation=first)

def check_certificate(folder,geometry):
    gate=read(folder/'GATE.json');stored=read(folder/'geometry.json');lo=geometry['lower'];hi=geometry['upper']
    # Independent face coverage: every face of the initial interior must be
    # spanned in its two tangent dimensions and overlap in its normal dimension.
    covered=[]
    for axis in range(3):
        for boundary in (lo[axis],hi[axis]):
            yes=any(b['lower'][axis]-1e-12<=boundary<=b['upper'][axis]+1e-12 and all(b['lower'][j]<=lo[j] and b['upper'][j]>=hi[j] for j in range(3) if j!=axis) for b in stored['boxes'])
            covered.append(yes)
    initial=gate['initial_target_root'];goal=stored['fixed_goal']
    assumptions=dict(faces=all(covered),initial_inside=all(a<x<b for a,x,b in zip(lo,initial,hi)),goal_disjoint=any(goal[1][j]<lo[j] or goal[0][j]>hi[j] for j in range(3)),static=stored['static'],initial_legal=gate['initial_safe'],root_material=gate['root_material_point'],endpoint_compatible=gate['goal_endpoint_compatible'])
    assert all(assumptions.values()),assumptions
    assert gate['certificate']['label']=='infeasible'
    return assumptions

def main(root):
    records=[];pairs=[]
    for p in sorted(root.glob('e*/*/summary.json')):records.append(audit_run(p.parent))
    for layout in sorted(root.glob('e*')):
        if not layout.is_dir() or not (layout/'fixture.json').exists():continue
        geometry=read(layout/'fixture.json')
        if (layout/'LABEL_GATE.json').exists() and read(layout/'LABEL_GATE.json')['passed']:
            labels={v:read(layout/v/'summary.json')['label'] for v in ('open','sealed','parked')}
            assert labels==dict(open='feasible',sealed='infeasible',parked='feasible')
            canonical=read(layout/'baseline/initial_restore.json')
            for v in labels:
                got=read(layout/v/'initial_restore.json')
                for k in ('time','arrays','sim_state','controller','environment','gripper_action'):
                    assert got[k]==canonical[k],(layout,v,k)
                assert sha(layout/v/'official_init.npy')==sha(layout/'baseline/official_init.npy')
            cert=check_certificate(layout/'sealed',geometry)
            lid=geometry['variants']['sealed'][-1];parked=geometry['variants']['parked'][-1]
            assert all(abs((lid['upper'][i]-lid['lower'][i])-(parked['upper'][i]-parked['lower'][i]))<1e-12 for i in range(3))
            pairs.append(dict(layout=layout.name,labels=labels,identical_initial_physics=True,identical_lid_dimensions=True,independent_certificate_assumptions=cert))
    out=dict(passed=True,records=records,valid_triplets=pairs,total_actions=sum(x['actions'] for x in records),total_samples=sum(x['samples'] for x in records),verified_hashes=sum(x['hashes'] for x in records),scope='Independent segment/contact/face-coverage logic on persisted records. Material-root/collider membership also relies on source-audited simulator gate; no empirical failure is an impossibility proof.')
    print(json.dumps(out,indent=2));return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();out=main(a.root)
    if a.output:a.output.write_text(json.dumps(out,indent=2)+'\n')
