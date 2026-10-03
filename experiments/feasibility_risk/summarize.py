"""Descriptive diagnostic summaries; deliberately no fitted classifier/accuracy."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def cosine_distance(a,b):
    a=np.asarray(a,float).ravel();b=np.asarray(b,float).ravel()
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(1-np.dot(a,b)/denom) if denom else None

def main(cpu,gpu,out):
    rows=[];distances=[]
    for e in (10,11):
        features={}
        for v in ('open','sealed','parked'):
            cd=cpu/('e%d'%e)/v;gd=gpu/('e%d'%e)/v
            cs=read(cd/'summary.json');s=read(gd/'summary.json');visibility=read(gd/'VISIBILITY.json');fa=read(gd/'FEATURE_AUDIT.json')
            x=np.load(gd/'initial_features.npz');features[v]={k:x[k].copy() for k in ('prefix_final','image_embedding','image_prefix_final')}
            geometry=read(gd/'geometry.json');names={g['id']:g['name'] for g in geometry['geoms']}
            first=s['first_violation'];contacts=[]
            if first:contacts=[dict(a=names[a],b=names[b],distance_m=d) for a,b,d,bad in first['contacts'] if bad]
            views={w['camera']:dict(exact_policy_rgb_match=w['exact_policy_rgb_match'],pixels={k:x['pixels_224'] for k,x in w['entities'].items()},off_frame_panels=[p['name'] for p in w['panel_projections'] if not p['all_corners_in_frame']]) for w in visibility['views']}
            rows.append(dict(layout=e,variant=v,feasibility_label=cs['label'],reference_actions=cs['steps'],policy_actions=s['steps'],risk_label=s['risk_label'],first_violation=first,contacts=contacts,views=views,lid_present=visibility['lid_present'],lid_over_target=visibility['privileged_lid_over_target'],strict_relation_visible=visibility['strict_target_goal_lid_visible'],feature_audit=fa,risk_action_norm=float(x['risk_scores'][0]),risk_proposal_variance=float(x['risk_scores'][1]),prefix_dimension=int(x['prefix_final'].size),image_dimension=int(x['image_embedding'].size)))
        for a,b in [('open','sealed'),('open','parked'),('sealed','parked')]:
            distances.append(dict(layout=e,a=a,b=b,**{k:cosine_distance(features[a][k],features[b][k]) for k in ('prefix_final','image_embedding','image_prefix_final')}))
    counts={k:sum(r['risk_label']==k for r in rows) for k in ('violation_within_30','horizon_safe_30','unscorable')}
    simple=dict(lid_presence_rule_correct=sum(('infeasible' if r['lid_present'] else 'feasible')==r['feasibility_label'] for r in rows),privileged_target_visibility_rule_correct=sum(('infeasible' if all(v['pixels']['target']==0 for v in r['views'].values()) else 'feasible')==r['feasibility_label'] for r in rows),privileged_lid_over_target_rule_correct=sum(('infeasible' if r['lid_over_target'] else 'feasible')==r['feasibility_label'] for r in rows),n=6,scope='Diagnostic rule sanity check only, not held-out learned accuracy. Relational geometry is privileged and shares assumptions with label certificate.')
    visibility_pass=all(r['strict_relation_visible'] for r in rows if r['variant'] in ('sealed','parked'))
    both_risk=all(len({r['layout'] for r in rows if r['risk_label']==k})>=2 for k in ('violation_within_30','horizon_safe_30'))
    extraction=all(r['feature_audit']['passed'] for r in rows)
    irrelevant_distance=[]
    for e in (10,11):
        close=next(d['prefix_final'] for d in distances if d['layout']==e and d['a']=='open' and d['b']=='sealed')
        parked=next(d['prefix_final'] for d in distances if d['layout']==e and d['a']=='open' and d['b']=='parked')
        irrelevant_distance.append(dict(layout=e,parked_over_closed_prefix_distance_ratio=parked/close,scope='Descriptive sensitivity, not evidence against linear decodability or a population effect.'))
    result=dict(irrelevant_cue_sensitivity=irrelevant_distance,independent_layouts=2,official_tasks=1,states=6,feasibility=dict(feasible=4,infeasible_conditional=2,unknown=0),risk_counts=counts,rows=rows,feature_cosine_distances=distances,shortcut_rules=simple,gates=dict(valid_triplets=True,relation_directly_visible=visibility_pass,both_risk_classes_in_two_layouts=both_risk,frozen_extraction_invariant=extraction),bulk_recommendation='do not scale' if not(visibility_pass and both_risk and extraction) else 'requires parent design checkpoint',trained_readout=False,train_layouts=0,validation_layouts=0,test_layouts=0,heldout_accuracy=None,risk_AUROC=None if not both_risk else 'not estimated in construction diagnostic',uncertainty='Only 2 clustered construction layouts; no inferential performance interval or test claim.',api_calls=0)
    out.mkdir(exist_ok=True,parents=True);(out/'DIAGNOSTIC_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'state_metrics.csv').open('w',newline='') as f:
        keys=['layout','variant','feasibility_label','reference_actions','policy_actions','risk_label','lid_present','lid_over_target','strict_relation_visible','risk_action_norm','risk_proposal_variance'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','feature_cosine_distances')},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cpu',type=Path);p.add_argument('gpu',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.cpu,a.gpu,a.out)
