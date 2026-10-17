"""V4 data contracts: the original scientific gate and the 2026-10-16 development mode."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from .common import read,sha,require,MODEL_ID
from verifier_dataset.constraint_selection import check_unit
from .engineering_contract import epoch_orders

ARMS=('A','B','C')
SEEDS=(713,714,715)
KINDS=('direct','facts')
FORCED_PREFIX={'direct':'Verdict: ','facts':'Protected:'}
FACT_KEYS=('protected','target_inside','goal_inside','separated')
DIGIT={'infeasible':'0','feasible':'1'}
DEVELOPMENT_FILES={'dataset.jsonl':'dataset_sha256','EVIDENCE_AUDIT.json':'evidence_audit_sha256',
    'VISUAL_REVIEW.json':'visual_review_sha256','SHORTCUT_AUDIT.json':'shortcut_audit_sha256',
    'FROZEN_SOURCES.json':'frozen_sources_sha256',
    'SHORTCUT_VALIDATION_PREDICTIONS.json':'shortcut_validation_predictions_sha256',
    'SHORTCUT_FEATURES.npz':'shortcut_features_sha256'}


def facts_text(facts):
    c=facts['protected']
    return (f'Protected: {c}\nTarget inside {c}: {facts["target_inside"]}\n'
            f'Goal inside {c}: {facts["goal_inside"]}\nSeparated: {facts["separated"]}')


def rule_verdict(target_inside,goal_inside):
    """Program rule over the two containment bits relative to the protected box (XNOR)."""
    require(target_inside in ('yes','no') and goal_inside in ('yes','no'),'Containment facts must be yes/no')
    return int(target_inside==goal_inside)


def assistant_target(row,kind):
    require(kind in KINDS,'Unknown target kind')
    digit=DIGIT[row['target']['verdict']]
    return 'Verdict: '+digit if kind=='direct' else facts_text(row['target']['facts'])+'\nVerdict: '+digit


def arm_kinds(arm):
    require(arm in ARMS,'Unknown arm')
    return {'A':('direct',),'B':('facts',),'C':('direct','facts')}[arm]


def readings(arm):
    """Greedy selection readings: name -> (target kind, forced prefix or None for free generation)."""
    require(arm in ARMS,'Unknown arm')
    return {'A':{'direct_free':('direct',None)},'B':{'facts_free':('facts',None)},
            'C':{'direct_forced':('direct',FORCED_PREFIX['direct']),'facts_forced':('facts',FORCED_PREFIX['facts'])}}[arm]


def target_kinds(ids,seed,epoch):
    """Arm C: half the samples take the direct target each epoch; epoch 1 flips epoch 0, later epochs redraw."""
    draw=0 if epoch==1 else epoch
    ranked=sorted(ids,key=lambda s:hashlib.sha256(f'v4-c-target/{seed}/{draw}/{s}'.encode()).hexdigest())
    kinds={s:('direct' if i<len(ids)//2 else 'facts') for i,s in enumerate(ranked)}
    if epoch==1:kinds={s:('facts' if k=='direct' else 'direct') for s,k in kinds.items()}
    return kinds


def schedules(train_ids,config):
    """Same sample order for every arm of one seed; only the target kind differs."""
    ids=sorted(train_ids);per_epoch=len(ids)//config['effective_batch_size'];result={}
    for seed in config['seeds']:
        batches=epoch_orders(len(ids),seed,config['epochs'],config['effective_batch_size'])
        require(len(batches)==config['optimizer_updates'],'Update count differs from the epoch schedule')
        kinds=[target_kinds(ids,seed,e) for e in range(config['epochs'])]
        result[str(seed)]={arm:[[[ids[i],arm_kinds(arm)[0] if arm!='C' else kinds[step//per_epoch][ids[i]]] for i in batch]
                               for step,batch in enumerate(batches)] for arm in config['arms']}
    return result


def check_schedule(order,train_ids,arm,config):
    per_epoch=len(train_ids)//config['effective_batch_size']
    require(len(order)==config['optimizer_updates'] and all(len(b)==config['effective_batch_size'] for b in order),'Incomplete update schedule')
    require({s for b in order for s,k in b}<=set(train_ids),'Non-training row in optimizer order')
    for epoch in range(config['epochs']):
        part=[x for b in order[epoch*per_epoch:(epoch+1)*per_epoch] for x in b]
        require(Counter(s for s,k in part)==Counter({s:1 for s in train_ids}),'An epoch repeats or omits training examples')
        counts=Counter(k for s,k in part)
        require(set(counts)<=set(arm_kinds(arm)) and (arm!='C' or counts['direct']==counts['facts']),'Target kinds unbalanced')


def edit_prefix(facts,field):
    """Flip one containment fact and let the model regenerate everything after it."""
    c=facts['protected'];flip=lambda v:'no' if v=='yes' else 'yes'
    if field=='target_inside':return f'Protected: {c}\nTarget inside {c}: {flip(facts["target_inside"])}\n'
    return f'Protected: {c}\nTarget inside {c}: {facts["target_inside"]}\nGoal inside {c}: {flip(facts["goal_inside"])}\n'


def diagnostic_plan(arm):
    if arm=='A':return []
    reading,(kind,prefix)=('facts_free',readings('B')['facts_free']) if arm=='B' else ('facts_forced',readings('C')['facts_forced'])
    return [(reading+'_sampled',kind,prefix,'sampled'),('oracle_facts',kind,None,'oracle'),
            ('edit_target',kind,None,'edit:target_inside'),('edit_goal',kind,None,'edit:goal_inside')]


def configuration(schedule='main'):
    require(schedule in ('main','trial'),'Unknown schedule')
    main=schedule=='main'
    return dict(version='V4-dev-ABC-v1',schedule=schedule,acceptance_mode='development',
        arms=list(ARMS) if main else ['A','B'],seeds=list(SEEDS) if main else [713],
        targets={'A':'Verdict: 0/1','B':'Structured facts, then Verdict: 0/1',
            'C':'Both targets per sample; every epoch assigns half the samples to each target, so the same sample order serves A, B and C'},
        model_id=MODEL_ID,lora_rank=16,lora_alpha=32,lora_dropout=.05,learning_rate=.0001,optimizer='AdamW',
        weight_decay=0.,gradient_clip=1.,micro_batch_size=1,effective_batch_size=4,
        epochs=3 if main else 1,optimizer_updates=384 if main else 128,evaluation_updates=[128,256,384] if main else [128],
        max_length=2048,image_pixels=448*448,vision_and_projector_frozen=True,
        initialization='Fresh staged base and fresh LoRA for every arm and seed; no historical adapter',
        objective='0.5 decisive verdict-token CE + 0.5 mean CE over all other assistant tokens',
        inference_inputs_identical=True,student_text='V4 task and constraint text unchanged',
        evaluation='Greedy generation with no gold rationale prefix; C readings force only Verdict: or Protected:',
        forced_prefixes=dict(FORCED_PREFIX),
        model_selection='A: direct_free accuracy; B: facts_free accuracy; C: mean of direct_forced and facts_forced '
            'accuracy at one shared checkpoint; ties choose the earlier update',
        maximum_new_tokens={'direct':8,'facts':64},repetition_penalty=1.,
        sampling=dict(temperature=.7,top_p=1.,top_k=0,samples=16,k_values=[1,2,4,8,16],seed=1014),
        test_usage='train() never loads test rows; evaluate-test runs once after every run, reading and metric is frozen',
        hyperparameter_search=False,fixed_threshold=.5,
        scientific_scope='Development data with a disclosed failed shortcut gate; tests whether fact supervision helps, not general robot reasoning')


def development_acceptance(root):
    acceptance=read(root/'DEV_ACCEPTANCE.json')
    require(acceptance.get('passed') is True and acceptance.get('acceptance_mode')=='development' and
            acceptance.get('development_training_allowed') is True and acceptance.get('scientific_training_allowed') is False,
            'Development data use has not been recorded')
    gate=acceptance['shortcut_gate']
    require(gate['passed'] is False and gate['reported_as_baseline'] is True and gate['locked_test_scored'] is False,
            'Failed shortcut gate must stay disclosed')
    require(acceptance['user_decision']['date']=='2026-10-16','Missing user decision')
    for name,key in DEVELOPMENT_FILES.items():require(sha(root/name)==acceptance[key],'Development evidence changed: '+name)
    shortcut=read(root/'SHORTCUT_AUDIT.json')
    require(shortcut['locked_test_scored'] is False and shortcut['passed'] is False,'Shortcut audit disclosure differs')
    return acceptance


def scientific_acceptance(root):
    acceptance=read(root/'ACCEPTANCE.json')
    require(acceptance.get('passed') is True and acceptance.get('scientific_training_allowed') is True,
            'Scientific data audit has not authorized training')
    required={'dataset.jsonl':'dataset_sha256','EVIDENCE_AUDIT.json':'evidence_audit_sha256',
        'VISUAL_REVIEW.json':'visual_review_sha256','SHORTCUT_AUDIT.json':'shortcut_audit_sha256',
        'FROZEN_SOURCES.json':'frozen_sources_sha256'}
    for name,key in required.items():require(sha(root/name)==acceptance[key],'Acceptance evidence changed: '+name)
    require(read(root/'SHORTCUT_AUDIT.json')['passed'],'Specified shortcut screen did not pass')
    return acceptance


def check_development_row(r):
    require(r.get('acceptance_mode')=='development','Row lacks development scope')
    facts=r['target'].get('facts',{});require(tuple(facts)==FACT_KEYS,'Structured facts missing')
    m=r['metadata'];protected=r['evidence']['protected_boundary']
    require(facts['protected']==m['appearance_assignment'][protected],'Protected color differs from the assignment')
    require(r['student']['constraints'].count('Only the %s boundary is protected'%facts['protected'])==1,'Rule text names another color')
    verdict=rule_verdict(facts['target_inside'],facts['goal_inside'])
    require(facts['separated']==('no' if verdict else 'yes') and verdict==int(r['target']['verdict']=='feasible'),
            'Facts do not imply the label')
    require(r['target']['rationale']==facts_text(facts),'Rationale is not the structured facts')


def validate(root,mode='scientific',include_test=True):
    root=Path(root);require(mode in ('scientific','development'),'Unknown acceptance mode')
    development=mode=='development'
    acceptance=development_acceptance(root) if development else scientific_acceptance(root)
    audit=read(root/'EVIDENCE_AUDIT.json')
    require(audit['passed'],'Evidence audit incomplete')
    expected_units=250;reduced_quotas=None
    if audit.get('selection_reduction'):
        from verifier_dataset.constraint_curation import selection_settings
        protocol_path=Path(__file__).resolve().parents[1]/'verifier_dataset/constraint_protocol.json'
        expected_units,reduced_quotas=selection_settings(dict(model_predictions_used=False,
            final_selection=audit['selection_reduction']),read(protocol_path),sha(protocol_path))
    review=read(root/'VISUAL_REVIEW.json');require(review['complete'] and not review['unresolved_units'],'Visual premises remain unresolved')
    if development:require(review['prepared_dataset_sha256']==acceptance['prepared_dataset_sha256'],'Visual review bound to other data')
    split={s['source_id']:s for s in read(root/'FROZEN_SOURCES.json')['sources']}
    rows=[json.loads(line) for line in (root/'dataset.jsonl').read_text().splitlines()]
    expected_rows=4*expected_units
    require(len(rows)==expected_rows and len({r['sample_id'] for r in rows})==expected_rows,'Audited distinct record count required')
    require(Counter(r['target']['verdict'] for r in rows)=={'feasible':expected_rows//2,'infeasible':expected_rows//2},'Unbalanced final labels')
    units=defaultdict(list);group_splits=defaultdict(set);images={};source_units=defaultdict(set);full_inputs=set()
    for r in rows:
        m=r['metadata'];source=split[m['source_id']]
        require(r['training_eligible'] is True and r['observability_review']=='passed','Unreviewed row cannot train')
        require(r['split']==source['split'] and m['ancestry_component']==source['ancestry_component'],'Frozen source split changed')
        require(bool(r['target']['rationale']) and r['target']['scope']=='conditional_on_stated_contract','Missing grounded target or conditional scope')
        expected='feasible' if r['evidence']['feasibility']=='feasible' else 'infeasible' if r['evidence']['feasibility']=='conditionally_infeasible' else None
        require(r['target']['verdict']==expected,'Unknown or policy failure is not binary supervision')
        if development:check_development_row(r)
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
    require(len(units)==expected_units and len(source_units)>=30 and all(len(v)<=10 for v in source_units.values()),'Source/checkpoint coverage failed')
    require(all(len(v)==1 for v in group_splits.values()),'Ancestry leakage')
    require(len({r['metadata']['task_group'] for r in rows})>=6 and len({r['metadata']['family'] for r in rows})>=3,'Task/family coverage failed')
    for unit in units.values():check_unit(unit)
    from verifier_dataset.constraint_sides import VERSION,LEGACY_VERSION,selection_quotas
    from verifier_dataset.constraint_selection import selection_cell
    versions={r['metadata'].get('construction_version') for r in rows}
    if versions&{VERSION,LEGACY_VERSION}:
        require(len(versions)==1,'Mixed construction versions');version=next(iter(versions))
        actual=Counter((u[0]['split'],selection_cell(u[0])) for u in units.values())
        expected={(split,cell):n for split,cells in (reduced_quotas or selection_quotas(version)).items() for cell,n in cells.items()}
        require(actual==expected,'Endpoint-side balance must hold within every frozen source split')
        if version==VERSION:
            require(all(im.get('width')==448 and im.get('height')==448 for r in rows for im in r['student']['images']),
                'V4 requires native 448-pixel student images')
    require(set(units)==set(review['reviewed_units']),'Visual review does not cover the final selected units')
    for name in ('train','validation','test'):
        part=[r for r in rows if r['split']==name]
        require(part and len(part)%4==0,'Incomplete split units')
        require(len({r['metadata']['ancestry_component'] for r in part})>=(15 if name=='train' else 5),'Too few independent split groups')
    if not include_test:rows=[r for r in rows if r['split']!='test']
    return rows,acceptance


# Zero-shot prompts (PLAN.md Phase 1), shared by the base-Qwen diagnostics and the API baseline.
VIEWS='View 1 and View 2: two opposing external overview cameras.\n'
ZERO_SHOT_SYSTEM='You are a careful verifier of robot manipulation tasks under explicit safety constraints.'
DECIDE='Decide whether the robot can complete the task while satisfying every stated constraint. '
ZERO_SHOT_INSTRUCTIONS=dict(
    direct=DECIDE+"Reply with exactly one line: 'Verdict: 1' if a solution satisfying all constraints exists, "
        "or 'Verdict: 0' if the constraints make the task impossible.",
    facts=DECIDE+'The white-outlined object is the target and the green outlined region is the goal. '
        'First report these facts, then the verdict, in exactly this format:\n'
        'Protected: <orange or purple>\nTarget inside <protected color>: <yes or no>\nGoal inside <protected color>: <yes or no>\n'
        'Separated: <yes if the protected box contains exactly one of the target and the goal, otherwise no>\n'
        'Verdict: <1 if a solution satisfying all constraints exists, 0 if the constraints make the task impossible>',
    oracle='Facts about this scene: the white-outlined target object is {target} the protected {color} box, and the green '
        'goal region is {goal} the protected {color} box. '+DECIDE+"Reply with exactly one line: 'Verdict: 1' if a solution "
        "satisfying all constraints exists, or 'Verdict: 0' if the constraints make the task impossible.")
PROBE_QUESTIONS={'target':'Is the white-outlined object inside the {color} box? Answer yes or no.',
    'goal':'Is the green outlined destination region inside the {color} box? Answer yes or no.'}


def zero_shot_messages(row,mode):
    require(mode in ZERO_SHOT_INSTRUCTIONS,'Unknown zero-shot mode')
    instruction=ZERO_SHOT_INSTRUCTIONS[mode]
    if mode=='oracle':
        f=row['target']['facts'];side=lambda v:'inside' if v=='yes' else 'outside'
        instruction=instruction.format(target=side(f['target_inside']),goal=side(f['goal_inside']),color=f['protected'])
    text=VIEWS+'Task: '+row['student']['task']+'\nConstraints: '+row['student']['constraints']+'\n\n'+instruction
    return [dict(role='system',content=ZERO_SHOT_SYSTEM),dict(role='user',content=[dict(type='image'),dict(type='image'),dict(type='text',text=text)])]


def probe_messages(entity,color):
    require(entity in PROBE_QUESTIONS and color in ('orange','purple'),'Unknown probe')
    return [dict(role='system',content=ZERO_SHOT_SYSTEM),dict(role='user',content=[dict(type='image'),dict(type='image'),
        dict(type='text',text=VIEWS+PROBE_QUESTIONS[entity].format(color=color))])]


def probe_truth(unit_rows):
    """Four containment bits for one image pair, read from its two rule rows."""
    require(len(unit_rows)==2 and {r['evidence']['protected_boundary'] for r in unit_rows}=={'a','b'},'Probe needs both rules of one image pair')
    bits={}
    for r in unit_rows:
        f=r['target']['facts'];bits[('target',f['protected'])]=f['target_inside']=='yes';bits[('goal',f['protected'])]=f['goal_inside']=='yes'
    require(len(bits)==4,'Probe colors incomplete');return bits
