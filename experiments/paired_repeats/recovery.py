"""Hash-verified inheritance, exact missing-run selection and merged provenance."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from protocol import SCENARIOS, schedule, scenario_name

EVIDENCE=['row.json','manifest.json','rng_reset.json','settled_qpos.npy','first_action_chunk.npy']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def identity(row):
    return row['scenario'],row['episode'],row['repeat'],row['method']

def scheduled():
    return [dict(scenario=scenario_name(s),suite=s[0],level=s[1],task=s[2],episode=e,
                 repeat=r,method=m,seed=seed,run_id='full_s%d_e%02d_r%d_%s'%(SCENARIOS.index(s),e,r,m))
            for s,e,r,m,seed in schedule()]

def verify_entry(entry):
    directory=Path(entry['directory'])
    for name,expected in entry['sha256'].items():
        if digest(directory/name)!=expected:
            raise RuntimeError('Inherited evidence changed: '+str(directory/name))
    row=json.loads((directory/'row.json').read_text())
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['status']!='complete' or row['run_id']!=entry['run_id']:
        raise RuntimeError('Inherited run is not complete')
    for name,value in row.items():
        if manifest.get(name)!=value:
            raise RuntimeError('Row/manifest disagreement: '+name)
    return row

def select(parent):
    parent=Path(parent);plan=json.loads((parent/'plan.json').read_text())
    if not (parent/'STOP.json').exists():
        raise RuntimeError('Recovery requires a stopped parent; preserve STOP')
    entries=list(plan.get('inherited_records',[]))
    for file in sorted((parent/'full/runs').glob('*/row.json')):
        entries.append({'run_id':file.parent.name,'directory':str(file.parent.resolve()),
                        'sha256':{name:digest(file.parent/name) for name in EVIDENCE}})
    data={};order=scheduled()
    for entry in entries:
        row=verify_entry(entry)
        if row['run_id'] in data:
            raise RuntimeError('Duplicate inherited run')
        if row['phase']!='full' or row['code_commit']!=plan['code_commit']:
            raise RuntimeError('Mixed scientific code or phase')
        if row['collided']!=(row['max_obstacle_l1_m']>.001) or row['modification_threshold']!=1e-6:
            raise RuntimeError('Parent scoring configuration differs')
        data[row['run_id']]=(entry,row)
    # The serialized scheduler must have completed a prefix. Never select by outcome.
    if set(data)!={r['run_id'] for r in order[:len(data)]}:
        raise RuntimeError('Parent completion is not an exact schedule prefix')
    inherited=[]
    for expected in order[:len(data)]:
        entry,row=data[expected['run_id']]
        for name in ['scenario','suite','level','task','episode','repeat','method','seed']:
            if row[name]!=expected[name]:
                raise RuntimeError('Inherited identity/seed differs: '+name)
        inherited.append(entry)
    pending=order[len(data):]
    if not pending:
        raise RuntimeError('No runs remain')
    unfinished=[str(p.parent.resolve()) for p in (parent/'full/runs').glob('*/manifest.json') if not (p.parent/'row.json').exists()]
    return inherited,pending,unfinished

def load_inherited(plan):
    return [verify_entry(entry) for entry in plan['inherited_records']]

def state_references(check_root):
    result={}
    for path in (Path(check_root)/'checks/runs').glob('idle_*/row.json'):
        row=json.loads(path.read_text());key=row['scenario']+'|'+str(row['episode'])
        if key in result:raise RuntimeError('Duplicate state reference')
        result[key]={'qpos_sha256':row['qpos_sha256'],'active_obstacle':row['active_obstacle']}
    if set(result)!={scenario_name(s)+'|'+str(e) for s in SCENARIOS for e in range(20)}:
        raise RuntimeError('Missing one of the 60 checked initial states')
    return result

def first_chunk_audit(rows):
    pairs=defaultdict(list)
    for row in rows:pairs[(row['scenario'],row['episode'],row['repeat'])].append(row)
    mismatches=[]
    for key,rs in pairs.items():
        if len(rs)==2:
            if any(rs[0][k]!=rs[1][k] for k in ['seed','qpos_sha256','policy_reset_key']):
                raise RuntimeError('Seed/RNG/qpos differs between methods')
            if rs[0]['first_action_chunk_sha256']!=rs[1]['first_action_chunk_sha256']:
                mismatches.append({'scenario':key[0],'episode':key[1],'repeat':key[2]})
    return {'paired_count':sum(len(rs)==2 for rs in pairs.values()),
            'first_chunk_mismatch_count':len(mismatches),'first_chunk_mismatches':mismatches,
            'interpretation':'Descriptive outcomes; first-action differences remain a pairing-quality warning.' if mismatches else 'First-action hashes match.'}
