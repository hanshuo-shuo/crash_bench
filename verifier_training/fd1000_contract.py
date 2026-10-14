"""Scientific FD1000 gate, separate from the historical 120-row engineering fit."""
from collections import Counter,defaultdict
import json
from pathlib import Path
from .common import read,sha,require
from verifier_dataset.constraint_selection import check_unit


def configuration():
    return dict(version='FD1000-verifier-comparison-v1',seed=713,model_id='Qwen/Qwen2.5-VL-3B-Instruct',
        lora_rank=16,lora_alpha=32,lora_dropout=.05,learning_rate=.0001,epochs=3,
        micro_batch_size=1,effective_batch_size=4,max_length=1536,image_pixels=224*224,
        optimizer='AdamW',weight_decay=0.,gradient_clip=1.,vision_and_projector_frozen=True,
        initialization='Fresh staged base and fresh LoRA for each arm; no historical adapter',
        objective='0.5 decisive binary-token CE + 0.5 mean other assistant-token CE',
        arms={'A':'Verdict: 0 or 1','B':'Grounded rationale followed by Verdict: 0 or 1'},
        inference_inputs_identical=True,tokenization_and_gradient_preflight_required=True,
        evaluation='Free greedy generation; B receives no gold rationale prefix',
        model_selection='Highest validation free-generation verdict accuracy after a complete epoch; tie chooses the earlier epoch',
        test_usage='Evaluate the selected checkpoint once on the locked source test split',
        maximum_new_tokens={'A':12,'B':192},repetition_penalty=1.,hyperparameter_search=False,
        fixed_threshold=.5,scientific_scope='Controlled conditional-feasibility study; no claim that generative verification is necessary')


def validate(root):
    root=Path(root);acceptance=read(root/'ACCEPTANCE.json')
    require(acceptance.get('passed') is True and acceptance.get('scientific_training_allowed') is True,
            'Scientific data audit has not authorized training')
    required={'dataset.jsonl':'dataset_sha256','EVIDENCE_AUDIT.json':'evidence_audit_sha256',
        'VISUAL_REVIEW.json':'visual_review_sha256','SHORTCUT_AUDIT.json':'shortcut_audit_sha256',
        'FROZEN_SOURCES.json':'frozen_sources_sha256'}
    for name,key in required.items():require(sha(root/name)==acceptance[key],'Acceptance evidence changed: '+name)
    require(read(root/'EVIDENCE_AUDIT.json')['passed'],'Evidence audit incomplete')
    require(read(root/'SHORTCUT_AUDIT.json')['passed'],'Specified shortcut screen did not pass')
    review=read(root/'VISUAL_REVIEW.json');require(review['complete'] and not review['unresolved_units'],'Visual premises remain unresolved')
    split={s['source_id']:s for s in read(root/'FROZEN_SOURCES.json')['sources']}
    rows=[json.loads(line) for line in (root/'dataset.jsonl').read_text().splitlines()]
    require(len(rows)==1000 and len({r['sample_id'] for r in rows})==1000,'Exactly 1000 distinct records required')
    require(Counter(r['target']['verdict'] for r in rows)=={'feasible':500,'infeasible':500},'Unbalanced final labels')
    units=defaultdict(list);group_splits=defaultdict(set);images={};source_units=defaultdict(set);full_inputs=set()
    for r in rows:
        m=r['metadata'];source=split[m['source_id']]
        require(r['training_eligible'] is True and r['observability_review']=='passed','Unreviewed row cannot train')
        require(r['split']==source['split'] and m['ancestry_component']==source['ancestry_component'],'Frozen source split changed')
        require(bool(r['target']['rationale']) and r['target']['scope']=='conditional_on_stated_contract','Missing grounded target or conditional scope')
        expected='feasible' if r['evidence']['feasibility']=='feasible' else 'infeasible' if r['evidence']['feasibility']=='conditionally_infeasible' else None
        require(r['target']['verdict']==expected,'Unknown or policy failure is not binary supervision')
        units[m['unit_id']].append(r);source_units[m['source_id']].add(m['unit_id'])
        group_splits[m['ancestry_component']].add(r['split'])
        pair=[]
        for im in r['student']['images']:
            p=Path(im['path']);require(not p.is_absolute() and '..' not in p.parts and p.parts[0]=='images','Images must be packaged beneath dataset/images')
            require(sha(root/p)==im['sha256'],'Final student image changed');pair.append(im['sha256'])
        key=tuple(pair)
        require(key not in images or images[key]==r['split'],'Shared image pair crosses source splits');images[key]=r['split']
        full=(key,r['student']['task'],r['student']['constraints'])
        require(full not in full_inputs,'Repeated complete model input');full_inputs.add(full)
    require(len(units)==250 and len(source_units)>=30 and all(len(v)<=10 for v in source_units.values()),'Source/checkpoint coverage failed')
    require(all(len(v)==1 for v in group_splits.values()),'Ancestry leakage')
    require(len({r['metadata']['task_group'] for r in rows})>=6 and len({r['metadata']['family'] for r in rows})>=3,'Task/family coverage failed')
    for unit in units.values():check_unit(unit)
    from verifier_dataset.constraint_sides import EXIT,ENTRY,VERSION
    if any(r['metadata'].get('construction_version')==VERSION for r in rows):
        require(all(r['metadata'].get('construction_version')==VERSION for r in rows),'Mixed construction versions')
        actual=Counter((u[0]['split'],u[0]['metadata'].get('construction_mode')) for u in units.values())
        expected={(split,mode):n for split,n in (('train',75),('validation',25),('test',25)) for mode in (EXIT,ENTRY)}
        require(actual==expected,'Endpoint-side balance must hold within every frozen source split')
    require(set(units)==set(review['reviewed_units']),'Visual review does not cover the final selected units')
    for name in ('train','validation','test'):
        part=[r for r in rows if r['split']==name]
        require(part and len(part)%4==0,'Incomplete split units')
        require(len({r['metadata']['ancestry_component'] for r in part})>=(15 if name=='train' else 5),'Too few independent split groups')
    return rows,acceptance
