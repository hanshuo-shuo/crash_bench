"""Post-run independent FR-1B evidence checks; no simulator or trained readout."""
import argparse,json
from pathlib import Path
import numpy as np
from audit import audit_run,read,sha
from summarize import cosine_distance

def main(parent_cpu,parent_gpu,cpu,gpu,out):
    canonical=read(parent_cpu/'e10/baseline/initial_restore.json')
    reference=[json.loads(x)['action'] for x in (parent_cpu/'e10/baseline/steps.jsonl').read_text().splitlines()[1:]]
    cross=read(gpu/'CROSS_JOB_COMPARISONS.json')
    repeat=read(gpu/'END_ANCHOR_REPEAT.json')
    assert repeat['passed'] and all(x==0 for x in repeat['max_absolute'].values())
    assert read(gpu/'CHECKPOINT_POST_VERIFY.json')['passed']
    assert read(gpu/'CHECKPOINT_POST_VERIFY.json')['manifest_sha256']==sha(gpu/'CHECKPOINT_SHA256.json')
    for row in cross['rows']:
        p=gpu/'common_inputs'/('e%d'%row['layout'])/row['variant']
        source=parent_gpu/('e%d'%row['layout'])/row['variant']
        fa=read(p/'FEATURE_AUDIT.json');assert fa['passed'] and fa['feature_repeat_linf']==fa['action_before_after_linf']==0
        assert row['source_features_sha256']==sha(source/'initial_features.npz')
        assert row['input_sha256']==read(source/'FEATURE_AUDIT.json')['input_sha256']==fa['input_sha256']
        with np.load(p/'initial_policy_input.npz') as a,np.load(source/'initial_policy_input.npz') as b:
            assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
    fixtures=[read(p/'e10/fixture.json') for p in (cpu,gpu)]
    assert fixtures[0]==fixtures[1] and fixtures[0]['compatible_candidate']
    assert not fixtures[0]['construction_conflicts']
    for f,h in fixtures[0]['source_hashes'].items():assert sha(parent_cpu/'e10'/f)==h
    rows=[];features={};audits=[]
    for name in ('decoy_open','decoy_sealed'):
        cd=cpu/'e10'/name;gd=gpu/'e10'/name
        a=audit_run(cd);audits.append(a)
        s=read(cd/'summary.json');g=read(cd/'GATE.json')
        assert s['safe_success'] and s['label']=='feasible' and s['steps']==226
        assert g['initial_safe'] and g['passed'] and g['target_outside_decoy']
        assert not g['target_collision_bounds_inside']
        assert g['certificate']['label']=='unknown'
        commands=[json.loads(x)['action'] for x in (cd/'steps.jsonl').read_text().splitlines()[1:]]
        assert commands==reference
        for directory in (cd,gd):
            state=read(directory/'initial_restore.json')
            for k in ('time','arrays','sim_state','controller','environment','gripper_action'):
                assert state[k]==canonical[k],(name,k)
            geom=read(directory/'geometry.json');root=g['initial_target_root']
            assert not all(l<x<h for l,x,h in zip(fixtures[0]['lower'],root,fixtures[0]['upper']))
            assert geom['static']
        complete=read(gd/'RENDER_ONLY_COMPLETE.json');fa=read(gd/'FEATURE_AUDIT.json')
        assert complete['environment_actions']==0 and fa['passed'] and fa['weights_frozen'] and fa['training_steps']==0
        assert fa['feature_repeat_linf']==fa['action_before_after_linf']==0 and fa['rng_before']==fa['rng_after']
        steps=[json.loads(x) for x in (gd/'steps.jsonl').read_text().splitlines()]
        assert len(steps)==1 and steps[0]['safe_history']
        visibility=read(gd/'VISIBILITY.json')
        assert all(v['exact_policy_rgb_match'] for v in visibility['views'])
        views={v['camera']:{k:e['pixels_224'] for k,e in v['entities'].items()} for v in visibility['views']}
        rows.append(dict(variant=name,label=s['label'],witness_actions=s['steps'],safe_success=True,views=views,feature_audit=fa))
        with np.load(gd/'initial_features.npz') as f:features[name]={k:f[k].copy() for k in ('prefix_final','image_embedding','image_prefix_final')}
    for name in ('open','sealed','parked'):
        with np.load(gpu/'common_inputs/e10'/name/'initial_features.npz') as f:features[name]={k:f[k].copy() for k in ('prefix_final','image_embedding','image_prefix_final')}
    distances=[]
    for a,b in (('open','sealed'),('decoy_open','decoy_sealed'),('sealed','decoy_sealed'),('open','decoy_open')):
        distances.append(dict(a=a,b=b,**{k:cosine_distance(features[a][k],features[b][k]) for k in features[a]}))
    vectors={k:features['sealed'][k]-features['open'][k] for k in features['open']}
    alignment={}
    for k in vectors:
        distance=cosine_distance(vectors[k],features['decoy_sealed'][k]-features['decoy_open'][k])
        alignment[k]=None if distance is None else 1-distance
    result=dict(passed=True,rows=rows,audits=audits,total_actions=sum(x['actions'] for x in audits),total_samples=sum(x['samples'] for x in audits),verified_hashes=sum(x['hashes'] for x in audits),cross_job_comparisons=cross,end_anchor_repeat=repeat,identical_initial_physics=True,identical_recorded_actions=True,new_independent_layouts=0,new_policy_rollouts=0,new_policy_risk_labels=0,feature_cosine_distances=distances,closure_displacement_cosine=alignment,scope='One closure by relevance construction contrast; descriptive representation changes only, using only common-process features. No trained readout, held-out accuracy, or native understanding claim. Target visibility and enclosure position remain cues. The certificate initial_contact_and_geometry_safe aggregate is false for the decoy because target-inside fails; the separate initial_safe gate passes. Neither decoy is certified infeasible.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('parent_cpu','parent_gpu','cpu','gpu','out'):p.add_argument(key,type=Path)
    a=p.parse_args();main(a.parent_cpu,a.parent_gpu,a.cpu,a.gpu,a.out)
