"""Read-only historical capture audit; no simulator, rollout, or model fitting."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CAP = ROOT/'results/counterfactual_router/full_d4751330395e_20260814T152231Z'
RES = ROOT/'results/iclr27/advantage_router_resolver_d4751330395e_20260829T135723Z_job5148751'
def sha(data): return hashlib.sha256(data).hexdigest()
def write(name, data): (OUT/name).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
def table(name, rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)
manifest=json.loads((CAP/'capture_manifest.json').read_text())
resolver=json.loads((RES/'manifest.json').read_text())
assets=[]
for name in ['capture_manifest.json','decision_metadata.json','option_rollouts.jsonl','decision_features.npz']:
    path=CAP/name; actual=sha(path.read_bytes()); expected=resolver['input_sha256']['capture/'+name]
    assets.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=actual, expected_sha256=expected, status='verified' if actual==expected else 'MISMATCH'))
assert all(x['status']=='verified' for x in assets)
commit=manifest['repository']['git_commit']
for name in ['scripts/collect_counterfactual_option_rollouts.py','crashbench/recovery.py','crashbench/policies/openvla_policy.py','crashbench/envs/libero_adapter.py','scripts/collect_glass_recovery_pairs.py','setup/counterfactual_option_rollouts.sbatch','crashbench/counterfactual_router.py']:
    old=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
    assets.append(dict(path=name, bytes=len(old), sha256=sha(old),expected_sha256='',status='historical_git_blob_at_'+commit))
    assets.append(dict(path=name, bytes=(ROOT/name).stat().st_size, sha256=sha((ROOT/name).read_bytes()),expected_sha256='',status='local_current_blob'))
table('assets.csv',assets)
meta=json.loads((CAP/'decision_metadata.json').read_text())
rows=[json.loads(line) for line in (CAP/'option_rollouts.jsonl').read_text().splitlines()]
lookup={}
for r in rows:
    key=(r['decision_id'],r['option']); assert key not in lookup;lookup[key]=r
    assert not (r['crashed'] and r['succeeded'])
assert len(meta)==273 and len(rows)==819 and len({r['source_state_sha256'] for r in rows})==20
pairs=[]
for m in meta:
    b=lookup[m['decision_id'],'base_continue'];d=lookup[m['decision_id'],'detour_complete']
    assert all(lookup[m['decision_id'],o]['source_state_sha256']==m['source_state_sha256'] for o in ('base_continue','detour_complete','retreat_hold'))
    pairs.append(dict(source=m['source_state_sha256'],split=m['split'],condition=m['condition'],placement=m['placement_key'],decision_id=m['decision_id'],anchor_index=m['matched_scan_index'],collision_relative_horizon=m['horizon_actions'],repeats_per_option=1,base_success=int(b['succeeded']),detour_success=int(d['succeeded']),base_accident=int(b['crashed']),detour_accident=int(d['crashed']),delta_success=int(d['succeeded'])-int(b['succeeded']),delta_accident=int(d['crashed'])-int(b['crashed']),success_loss=int(b['succeeded'] and not d['succeeded']),new_accident=int(d['crashed'] and not b['crashed']),base_steps=b['steps'],detour_steps=d['steps']))
table('fixed_detour_pairs.csv',pairs)
terminal=[]
for r in rows:
    reason=r.get('termination') or ('accident' if r['crashed'] else 'success' if r['succeeded'] else 'unrecorded_cap_or_other')
    terminal.append(dict(source=r['source_state_sha256'],split=r['split'],condition=r['condition'],placement=r['placement_key'],decision_id=r['decision_id'],option=r['option'],repeat_count=1,steps=r['steps'],success=r['succeeded'],accident=r['crashed'],termination=reason,controller_final_stage=r.get('controller_final_stage','')))
table('terminal_records.csv',terminal)
source_rows=[]
for source,split in sorted(manifest['source_splits'].items()):
    for condition in ('glass','offpath','noglass'):
        p=[x for x in pairs if x['source']==source and x['condition']==condition]
        source_rows.append(dict(source=source,historical_split=split,condition=condition,decisions=len(p),placements=len({x['placement'] for x in p}),repeats_per_option=1 if p else 0,**{k:sum(x[k] for x in p)/len(p) if p else None for k in ['base_success','detour_success','delta_success','base_accident','detour_accident','delta_accident','success_loss','new_accident']}))
table('source_condition_table.csv',source_rows)
metrics=[]
for family in ('all','glass','offpath','noglass'):
    p=[x for x in pairs if family=='all' or x['condition']==family]
    by=defaultdict(list)
    for x in p: by[x['source']].append(x)
    metrics.append(dict(group=family,sources=len(by),decisions=len(p),**{k:sum(sum(x[k] for x in v)/len(v) for v in by.values())/len(by) for k in ['base_success','detour_success','delta_success','base_accident','detour_accident','delta_accident','success_loss','new_accident']},raw_base_success=sum(x['base_success'] for x in p),raw_detour_success=sum(x['detour_success'] for x in p),raw_success_loss=sum(x['success_loss'] for x in p),raw_new_accident=sum(x['new_accident'] for x in p)))
table('fixed_detour_summary.csv',metrics)
with np.load(CAP/'decision_features.npz',allow_pickle=False) as z:
    feature_info={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype),finite=bool(np.isfinite(z[k]).all())) for k in z.files}
# Algebra sanity check uses synthetic values only, no historical or new label fit.
rng=np.random.default_rng(2027);x=rng.normal(size=(17,5));x=np.column_stack((x,np.ones(17)));w=np.repeat([1/3,1/7,1/7],[3,7,7]);y=rng.normal(size=(17,2));penalty=np.eye(6);penalty[-1,-1]=0
solve=lambda y:np.linalg.solve(x.T@(w[:,None]*x)+penalty,x.T@(w[:,None]*y))
err=float(np.max(np.abs(solve(y[:,1:]-y[:,:1])-(solve(y)[:,1:]-solve(y)[:,:1]))));assert err<1e-12
write('audit_summary.json',dict(status='LOCAL_AUDIT_COMPLETE_REMOTE_UNVERIFIED',local_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),capture_commit=commit,checkpoint=manifest['checkpoint_identity'],protocol=manifest['protocol'],protocol_sha256=manifest['protocol_sha256'],capture_hashes_verified=True,features=feature_info,metrics=metrics,exclusions=manifest['exclusions'],exclusion_counts=manifest['exclusion_counts'],terminal_counts=dict(Counter(x['termination'] for x in terminal)),remote=dict(check='scripts/quest_sync.sh check',result='SSH socket not found: /tmp/quest.sock',commands_after_failed_check=0),full_glass_bundle='not saved by historical collector; not found in local capture; remote unverified',ridge_algebra_max_abs_error=err,new_rollouts=0,new_models_fitted=0))
print(json.dumps(dict(metrics=metrics,features=feature_info,algebra_error=err),indent=2))

# Preserve the separately observed remote evidence when regenerating local summaries.
remote_path=OUT/'quest_asset_evidence.json'
if remote_path.exists():
    summary=json.loads((OUT/'audit_summary.json').read_text())
    summary['status']='LOCAL_AND_QUEST_ASSET_AUDIT_COMPLETE_IMPLEMENTATION_PENDING'
    summary['remote']=json.loads(remote_path.read_text())
    summary['planned_capture_source_count']=len(manifest['source_splits'])
    summary['retained_outcome_source_count']=len({r['source_state_sha256'] for r in rows})
    summary['full_glass_bundle']='historical collector did not persist payload; absent from local and Quest exact capture directory; no proof of independent backup'
    write('audit_summary.json',summary)
