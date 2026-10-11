"""All-row persisted-evidence/observability audit; CPU Slurm only, no simulator."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
from .common import *
from .engineering_contract import DATA_SHA
from .export import check_hashes


def audit(dataset, out):
    out.mkdir(exist_ok=False)
    require(sha(dataset/'dataset.jsonl')==DATA_SHA, 'Dataset changed')
    rows=[json.loads(l) for l in (dataset/'dataset.jsonl').read_text().splitlines()]
    require(len(rows)==120, 'Wrong sample count')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/reach_avoid'))
    from validation import verify_certificate, verify_steps
    images=[]; entries=[]; texts=defaultdict(list); exact=defaultdict(list); coarse=defaultdict(list)
    for row in rows:
        m=row['metadata'];cpu=Path(m['cpu_root']);gpu=Path(m['gpu_root'])
        require(sha(cpu/'VERIFIED.json')==m['cpu_manifest_sha256'], 'CPU manifest changed')
        require(check_hashes(cpu)==m['cpu_verified_hashes'], 'Evidence changed')
        for name,h in m['observation_hashes'].items():require(sha(gpu/name)==h, 'Observation evidence changed')
        summary=read(cpu/'summary.json');cert=read(cpu/'CERTIFICATE.json');label_contract(summary,cert)
        require(summary['label']==row['target']['verdict'], 'Label changed')
        steps=[json.loads(l) for l in (cpu/'steps.jsonl').read_text().splitlines()]
        geom=read(cpu/'geometry.json')
        verified=verify_certificate(geom,read(cpu/'MATERIAL_BALL.json'),read(cpu/'fixture.json'),steps[0],summary['initial_valid'],cert)
        verify_steps(steps,summary,geom['fixed_goal'],verified,300)
        require(steps[0]['step']==0 and steps[0]['action'] is None and not steps[0]['contract_success'],
                'Zero-action diagnostic requires an initially false goal')
        expected,grounding=rationale(m['visibility'])
        require(expected==row['target']['rationale'] and grounding==m['rationale_grounding'], 'Rationale grounding changed')
        arr=[];small=[]
        for im in row['student']['images']:
            p=dataset/im['path'];require(sha(p)==im['sha256'],'Student image changed')
            with Image.open(p) as image:
                require(image.mode=='RGB' and image.size==(224,224),'Bad student image')
                arr.append(np.asarray(image).copy())
                # Fixed coarse screen, never a replacement student input or scientific metric.
                small.append(np.rint(np.asarray(image.resize((28,28),Image.Resampling.BOX),dtype=np.float32)/16).astype(np.uint8))
        pair=np.stack(arr);images.append(pair)
        exact[hashlib.sha256(pair.tobytes()).hexdigest()].append(row['sample_id'])
        coarse[hashlib.sha256(np.stack(small).tobytes()).hexdigest()].append(row['sample_id'])
        texts[(row['student']['task'],row['student']['constraints'])].append(row['sample_id'])
        views=m['visibility']['views'];pixels={k:[v['entities'][k]['pixels_224'] for v in views] for k in ('target','goal','panels')}
        hidden=not any(pixels['target'])
        small_fragments={k:[int(x) for x in p if 0<x<16] for k,p in pixels.items() if any(0<x<16 for x in p)}
        projections=[{p['name']:p for p in v['panel_projections']} for v in views]
        names=set().union(*(set(p) for p in projections))
        not_framed=[n for n in names if not any(n in p and p[n]['all_corners_in_front'] and p[n]['all_corners_in_frame'] for p in projections)]
        entries.append({'sample_id':row['sample_id'],'case':m['case'],'label':row['target']['verdict'],
            'label_basis':m['label_basis'],'label_contract_passed':True,'source_scene':m['source_scene'],
            'milk_external_pixels':pixels['target'][0],'milk_wrist_pixels':pixels['target'][1],
            'basket_external_pixels':pixels['goal'][0],'basket_wrist_pixels':pixels['goal'][1],
            'barrier_external_pixels':pixels['panels'][0],'barrier_wrist_pixels':pixels['panels'][1],
            'milk_hidden_both':hidden,'small_positive_fragments_lt16':small_fragments,
            'barrier_geometries_not_fully_in_either_frame':len(not_framed),
            'barrier_cropped_either_view':any(v['entities']['panels']['touches_image_edge'] for v in views),
            'verdict_identifiability':'unresolved_hidden_target' if hidden else 'not_certified_from_two_RGB_views',
            'important_geometry_missing':True,
            'missing_evidence':'Hidden target pose; occluded enclosure faces; metric material/clearance and dynamics' if hidden else
                'Unseen enclosure faces/interiors; metric material/clearance and dynamics; complete safe route',
            'rationale_evidence_support':'segmentation-supported presence/absence; no metric or causal proof',
            'rationale_recognizability':'review_tiny_fragments' if small_fragments else 'presence_not_identity_certification',
            'causal_verdict_rationale_supported':False,'text_group':'shared_task_constraints_0',
            'text_reveals_individual_label':False,'scientific_supervision_eligible':False,
            'engineering_disposition':'retain_unchanged_in_120_fit_only; report separate observability strata',
            'zero_action_initial_goal_false':True,'safe_witness_steps':summary['steps'] if summary['label']=='feasible' else None,
            'metadata_sha256':hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()})
        print('evidence_checked',row['sample_id'],flush=True)
    arrays=np.stack(images).astype(np.int16);labels=[r['target']['verdict'] for r in rows]
    pair_records=[]
    for i,e in enumerate(entries):
        candidates=[j for j,l in enumerate(labels) if l!=labels[i]]
        d=np.abs(arrays[candidates]-arrays[i]);means=d.mean(axis=(1,2,3,4))
        j=candidates[int(np.argmin(means))];delta=np.abs(arrays[j]-arrays[i])
        mae=float(delta.mean());frac=float(np.mean(np.any(delta>8,axis=-1)))
        e.update(nearest_opposite_id=rows[j]['sample_id'],nearest_opposite_rgb_mae_0_255=mae,
                 nearest_opposite_pixel_fraction_delta_gt8=frac,
                 nearest_opposite_changed_pixel_fraction=float(np.mean(np.any(delta>0,axis=-1))),
                 near_conflict_screen=mae<=1.0 and frac<=.001)
        pair_records.append((mae,min(i,j),max(i,j)))
    by_id={r['sample_id']:r for r in rows}
    def conflicts(groups):
        return [ids for ids in groups.values() if len({by_id[i]['target']['verdict'] for i in ids})>1]
    exact_conflicts=conflicts(exact);coarse_conflicts=conflicts(coarse)
    require(not exact_conflicts,'Exact contradictory inputs require a revised isolation decision')
    # Descriptive in-sample majority baselines, not learned evaluation or generalization.
    associations={}
    for name,key in [('hidden_milk',lambda r:str(not any(v['entities']['target']['pixels_224'] for v in r['metadata']['visibility']['views']))),
                     ('rationale_template',lambda r:r['target']['rationale']),
                     ('privileged_variant_NOT_input',lambda r:r['metadata']['variant'])]:
        groups=defaultdict(Counter)
        for r in rows:groups[key(r)][r['target']['verdict']]+=1
        associations[name]={'counts':{k:dict(v) for k,v in groups.items()},'resubstitution_majority_fraction':sum(max(c.values()) for c in groups.values())/120}
    summary={'passed_for_engineering_only':True,'scientific_observability_passed':False,'rows':120,
        'dataset_sha256':DATA_SHA,'labels':dict(Counter(labels)),'original_sources':1,
        'hidden_both_by_label':dict(Counter(e['label'] for e in entries if e['milk_hidden_both'])),
        'exact_rgb_unique_pairs':len(exact),'exact_rgb_conflicting_groups':exact_conflicts,
        'coarse_conflicting_groups':coarse_conflicts,
        'near_conflict_screen_count':sum(e['near_conflict_screen'] for e in entries),
        'near_conflict_rule':'nearest opposite label pair: mean absolute RGB difference <=1/255 and <=0.1% pixels with any channel difference >8/255; diagnostic screen, not proof of perceptual equality',
        'coarse_rule':'28x28 BOX downsample then round each channel /16; diagnostic only',
        'rows_with_lt16_pixel_fragments':sum(bool(e['small_positive_fragments_lt16']) for e in entries),
        'barrier_cropped_rows':sum(e['barrier_cropped_either_view'] for e in entries),
        'text_groups':len(texts),'text_group_label_counts':dict(Counter(labels)),
        'task_constraint_label_leakage':False,'associations':associations,
        'rationale_templates':len({r['target']['rationale'] for r in rows}),
        'isolation':'All 120 retained unchanged only for engineering fit; none certified as image-identifiable scientific supervision. Hidden32 and near-conflict strata reported separately. Presence via segmentation is weaker than human recognition.',
        'supported_constraint_diagnostics':{
            'horizon_600':'Same labels: each positive safe witness <=300; RA-SEP-1 invariant separator negatives hold at any finite horizon under unchanged guards.',
            'horizon_0':'All infeasible: recorded step0 has no action and contract_success=false; no commands means no transitions. This is a trivial logic diagnostic, not a second physical mechanism.'},
        'code_commit':os.environ['CB_CODE_COMMIT'],'slurm_job':os.environ['SLURM_JOB_ID'],
        'new_simulation':False,'policy_failures_as_labels':False}
    write(out/'AUDIT.json',summary);write(out/'samples.json',entries)
    with (out/'samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader()
        for e in entries:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in e.items()})
    # Close visual pairs shown at native resolution, with amplified differences.
    selected=[];seen=set()
    for mae,i,j in sorted(pair_records):
        if (i,j) not in seen:selected.append((mae,i,j));seen.add((i,j))
        if len(selected)==12:break
    for k,(mae,i,j) in enumerate(selected):
        sheet=Image.new('RGB',(896,3*252),'white');draw=ImageDraw.Draw(sheet)
        for y,index in enumerate((i,j)):
            draw.text((4,y*252+3),f'{rows[index]["sample_id"]} {rows[index]["metadata"]["case"]} {labels[index]}',fill='black')
            for c in range(2):sheet.paste(Image.fromarray(images[index][c]),(c*448,y*252+24))
        draw.text((4,507),f'absolute difference x16; two-view MAE={mae:.8f} on 0..255',fill='black')
        for c in range(2):sheet.paste(Image.fromarray(np.clip(np.abs(arrays[i,c]-arrays[j,c])*16,0,255).astype(np.uint8)),(c*448,528))
        sheet.save(out/f'near_pair_{k:02d}.png')
    print(json.dumps({k:v for k,v in summary.items() if k!='associations'}),flush=True)
    return summary
