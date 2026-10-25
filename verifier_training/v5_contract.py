"""V5 S3 contract: the object/II/task1 A-versus-B pilot (docs/generative_verifier/V5_S3_PROTOCOL.md).

Fixed configuration, targets, the one-epoch sample order, the selection and training-fit subsets, the
zero-shot prompts and the preregistered reading. validate() checks the frozen S2 files and never returns
a test row unless asked; the pilot never asks.
"""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from .common import read,sha,require,MODEL_ID
from .engineering_contract import epoch_orders
from verifier_dataset.v5_dataset import facts_text,constraint_text,SCHEMA

VERSION='V5-object-AB-v1'
DATASET=Path('/projects/p33100/siosio/crashbench_safelibero/verifier_dataset/20261025T013358Z_v5-s2_d1d8d4129223')
DATASET_HASHES={'UNITS.json':'a8f8ac17c6d1654db81deb638dc0cd3e53c1fc4bafccfd5339a518b64bdd6fc3',
    'dataset.jsonl':'63fe4c8707c6d7c18341cbf705b55360e0af3a7ab5f0b967bc8d77346c1f3c61',
    'S2_SUMMARY.json':'3c74e582e153bc64a91b6e53be12442b958aa0a8cd7f9fc07b63fcfd92e02550',
    'TEST_FREEZE.json':'be1a48ab066dae4f75fa369dad5662824c40eb942641c69bac9e458956144616',
    'S2_COMPLETE.json':'63c17297fbc09a8b8cd96efd3896a47af29110cd4be2f108b5b07ac00d63d1f0'}
ARMS=('A','B')
SEEDS=(713,)
KINDS=('direct','facts')
FACT_KEYS=('line','target_robot_side','goal_robot_side','separated')
FORCED_PREFIX={'direct':'Verdict: ','facts':'Line:'}
DEVELOPMENT_SPLITS=('train','validation','validation_natural')
TEST_SPLITS=('test','test_natural')
SELECT_IMAGES=192
FIT_ROWS=256
SELECT_SALT='v5-s3-select'
FIT_SALT='v5-s3-fit'
TRAIN_LOOKUP=.760


def digest(*parts):
    return hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()


def rule_verdict(target_robot_side,goal_robot_side):
    """Program rule: Separated (exactly one endpoint on the robot side) means infeasible, digit 0."""
    require(target_robot_side in ('yes','no') and goal_robot_side in ('yes','no'),'Robot-side facts must be yes/no')
    return int(target_robot_side==goal_robot_side)


def assistant_target(row,kind):
    require(kind in KINDS,'Unknown target kind')
    digit=row['target']['digit']
    return 'Verdict: '+digit if kind=='direct' else facts_text(row['target']['facts'])+'\nVerdict: '+digit


def readings(arm):
    """Selection reading: name -> (target kind, forced prefix); free generation, no gold rationale prefix."""
    require(arm in ARMS,'Unknown arm')
    return {'A':{'direct_free':('direct',None)},'B':{'facts_free':('facts',None)}}[arm]


def check_row(r):
    require(r['schema_version']==SCHEMA and r['split'] in DEVELOPMENT_SPLITS+TEST_SPLITS,'Unexpected schema or split')
    f=r['target']['facts'];require(set(f)==set(FACT_KEYS),'Structured facts missing')
    require(f['separated']==('yes' if f['target_robot_side']!=f['goal_robot_side'] else 'no'),'Separated differs from the two facts')
    verdict=rule_verdict(f['target_robot_side'],f['goal_robot_side'])
    require(r['target']['digit']==str(verdict) and r['target']['verdict']==('feasible' if verdict else 'infeasible'),'Facts do not imply the label')
    require(r['target']['rationale']==facts_text(f),'Rationale is not the structured facts')
    require(r['student']['constraints']==constraint_text(r['metadata']['rule']),'Rule text differs from the S2 template')
    require(f['line']==' | '.join(r['student']['constraints'].split('straight line through the ',1)[1].split('. ',1)[0].split(' and the ',1)),
            'Line fact differs from the rule text')
    require([im['camera'] for im in r['student']['images']]==['overview_a','overview_b'],'Student views differ')
    for im in r['student']['images']:
        p=Path(im['path']);require(not p.is_absolute() and '..' not in p.parts and p.parts[0]=='units','Images must stay under units/')


def validate(root=DATASET,include_test=False):
    """Frozen S2 rows. Test rows are hashed against TEST_FREEZE.json and dropped unless include_test."""
    root=Path(root)
    for name,digest_ in DATASET_HASHES.items():require(sha(root/name)==digest_,'Frozen S2 file changed: '+name)
    require(read(root/'HASHES.json')==DATASET_HASHES,'S2 hash record differs')
    done=read(root/'S2_COMPLETE.json');require(done['passed'] is True and done['reading']=='s2_dataset_ready','S2 did not pass')
    freeze=read(root/'TEST_FREEZE.json');frozen=set(freeze['sample_ids'])
    rows=[];test_lines=[]
    with (root/'dataset.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line)
            if r['sample_id'] in frozen:
                require(r['split'] in TEST_SPLITS,'Frozen row outside the test splits');test_lines.append(line)
                if not include_test:continue
            else:require(r['split'] in DEVELOPMENT_SPLITS,'Test row missing from the freeze')
            check_row(r);rows.append(r)
    require(len(test_lines)==freeze['rows']==len(frozen) and hashlib.sha256(''.join(test_lines).encode()).hexdigest()==freeze['rows_sha256'],
            'Test rows differ from the freeze')
    require(len({r['sample_id'] for r in rows})==len(rows),'Repeated sample id')
    owner={}
    for r in rows:
        key=tuple(im['sha256'] for im in r['student']['images'])
        require(owner.setdefault(key,r['split'])==r['split'],'An image pair crosses splits')
        require(owner.setdefault(r['metadata']['image_id'],r['split'])==r['split'],'An image id crosses splits')
    full=Counter((r['metadata']['image_id'],r['metadata']['rule']) for r in rows);require(max(full.values())==1,'Repeated image and rule')
    require(include_test or not {r['split'] for r in rows}&set(TEST_SPLITS),'Test rows reached a development stage')
    return rows


def lookup_predictions(train,rows):
    """Rule x progress majority learned on train (fallback rule, then global): the shortcut the test blocks."""
    key=lambda r:(r['student']['task'],r['metadata']['rule'],r['metadata']['time_bin'])
    text=lambda r:(r['student']['task'],r['metadata']['rule'])
    tables=(defaultdict(Counter),defaultdict(Counter))
    for r in train:tables[0][key(r)][r['target']['digit']]+=1;tables[1][text(r)][r['target']['digit']]+=1
    overall=Counter(r['target']['digit'] for r in train).most_common(1)[0][0]
    def guess(r):
        for table,k in ((tables[0],key(r)),(tables[1],text(r))):
            if table.get(k):return table[k].most_common(1)[0][0]
        return overall
    return {r['sample_id']:int(guess(r)) for r in rows}


def selection_subset(validation,n=SELECT_IMAGES):
    """192 validation images with both answers, by sha256(image|salt); one separated and one not per image, by sha256(image|rule|salt)."""
    images=defaultdict(list)
    for r in validation:images[r['metadata']['image_id']].append(r)
    twins=[i for i,rs in images.items() if {r['target']['digit'] for r in rs}=={'0','1'}]
    require(len(twins)>=n,'Too few twin images for the selection subset')
    chosen=[]
    for image in sorted(twins,key=lambda i:digest(i,SELECT_SALT))[:n]:
        for d in ('0','1'):
            rs=[r for r in images[image] if r['target']['digit']==d]
            chosen.append(min(rs,key=lambda r:digest(image,r['metadata']['rule'],SELECT_SALT))['sample_id'])
    return sorted(chosen)


def fit_subset(train,n=FIT_ROWS):
    return sorted(sorted((r['sample_id'] for r in train),key=lambda s:digest(s,FIT_SALT))[:n])


def schedule(train_ids,config):
    """One epoch, the same seeded sample order for A and B; only the target kind differs."""
    ids=sorted(train_ids);result={}
    for seed in config['seeds']:
        batches=epoch_orders(len(ids),seed,config['epochs'],config['effective_batch_size'])
        require(len(batches)==config['optimizer_updates'],'Update count differs from the epoch schedule')
        result[str(seed)]={arm:[[[ids[i],'direct' if arm=='A' else 'facts'] for i in b] for b in batches] for arm in config['arms']}
    return result


def check_schedule(order,train_ids,arm,config):
    require(len(order)==config['optimizer_updates'] and all(len(b)==config['effective_batch_size'] for b in order),'Incomplete update schedule')
    flat=[x for b in order for x in b]
    require(Counter(s for s,k in flat)==Counter({s:1 for s in train_ids}),'The epoch repeats or omits training rows')
    require({k for s,k in flat}=={'direct' if arm=='A' else 'facts'},'Target kind differs from the arm')


def configuration():
    return dict(version=VERSION,arms=list(ARMS),seeds=list(SEEDS),dataset=str(DATASET),dataset_hashes=dict(DATASET_HASHES),
        targets={'A':'Verdict: 0/1','B':'Line / Target on robot side / Goal on robot side / Separated, then Verdict: 0/1'},
        model_id=MODEL_ID,lora_rank=16,lora_alpha=32,lora_dropout=.05,learning_rate=.0001,optimizer='AdamW',
        weight_decay=0.,gradient_clip=1.,micro_batch_size=1,effective_batch_size=4,
        epochs=1,training_rows=1792,optimizer_updates=448,evaluation_updates=[150,300,448],
        max_length=2048,image_pixels=448*448,vision_and_projector_frozen=True,
        initialization='Fresh staged base and fresh LoRA for every arm and seed; no historical adapter',
        objective='0.5 decisive verdict-token CE + 0.5 mean CE over all other assistant tokens',
        inference_inputs_identical=True,student_text='S2 task and rule text unchanged',
        evaluation='Greedy generation with no gold rationale prefix',forced_prefixes=dict(FORCED_PREFIX),
        selection=dict(images=SELECT_IMAGES,rows=2*SELECT_IMAGES,salt=SELECT_SALT,value='verdict accuracy of the arm reading',
            ties='earlier update'),
        training_fit=dict(rows=FIT_ROWS,salt=FIT_SALT,lookup_bound=TRAIN_LOOKUP),
        maximum_new_tokens={'direct':8,'facts':96},repetition_penalty=1.,
        bootstrap=dict(unit='image',replicates=1000,seed=1014,reference='source clusters (4 validation sources)'),
        zero_shot=dict(models=['Qwen/Qwen2.5-VL-3B-Instruct','Qwen/Qwen3-VL-32B-Instruct'],prompts=['direct','facts'],rows='selection subset'),
        test_usage='No pilot stage loads test or test_natural; one later evaluation after 3 seeds and frozen selection',
        hyperparameter_search=False,api_calls=0)


def campaign(root,splits=('train','validation','validation_natural')):
    """Receipt, fixed configuration and development rows of one S3 run root; shared by the 3B and 32B stages."""
    import os
    receipt=read(root/'RECEIPT.json');config=read(root/'config.json')
    require(config==configuration() and sha(root/'config.json')==receipt['configuration_sha256'],'Fixed S3 configuration changed')
    require(os.environ['CB_CODE_COMMIT']==receipt['source_commit'],'Frozen S3 source differs')
    dataset=Path(receipt['dataset']);require(dataset==DATASET,'Unexpected dataset root')
    rows=validate(dataset,include_test=False)
    require({r['split'] for r in rows}==set(DEVELOPMENT_SPLITS),'Unexpected splits loaded')
    return receipt,config,dataset,[r for r in rows if r['split'] in splits]


def preflight_gate(root,receipt):
    """Preflight evidence: student input digests, the selection and fit subsets, and the lookup predictions."""
    done=read(root/'preflight/COMPLETE.json')
    require(done['passed'] and done['source_commit']==receipt['source_commit'],'Token preflight gate failed')
    for name,key in (('preflight/ENCODINGS.json','encodings_sha256'),('ORDERS.json','orders_sha256'),('SUBSETS.json','subsets_sha256'),
                     ('LOOKUP.json','lookup_sha256'),('config.json','config_sha256')):
        require(sha(root/name)==done[key],'Preflight evidence changed: '+name)
    return {r['sample_id']:r['input_sha256'] for r in read(root/'preflight/ENCODINGS.json')},read(root/'SUBSETS.json'),read(root/'LOOKUP.json')['predictions']


def s3_reading(a,b,delta_interval,fit):
    """Preregistered reading over R1 (full validation verdict accuracy), the paired image-bootstrap interval of B - A and R5."""
    delta=b-a
    if b>=.65 and delta>=.10 and delta_interval[0]>0:return 'b_grounds'
    if a>=.65 and delta<.10:return 'both_ground'
    if a<=.60 and b<=.60 and max(fit.values())>=.70:return 'both_shortcut'
    return 'unclear'


# Zero-shot prompts (R6), shared by the 3B base model and Qwen3-VL-32B; the images and rule text are the student's.
VIEWS='View 1 and View 2: two opposing external overview cameras.\n'
ZERO_SHOT_SYSTEM='You are a careful verifier of robot manipulation tasks under explicit safety constraints.'
DECIDE='Decide whether the robot can still complete the task while obeying the new rule. '
ZERO_SHOT_INSTRUCTIONS=dict(
    direct=DECIDE+"Reply with exactly one line: 'Verdict: 1' if the task can be completed without moving the object across the line, "
        "or 'Verdict: 0' if the rule makes the task impossible.",
    facts=DECIDE+'The robot side of the line is the side that contains the base of the robot arm. '
        'First report these facts, then the verdict, in exactly this format:\n'
        'Line: <the two objects the line passes through, written as: first | second>\n'
        'Target on robot side: <yes if the object being picked up is on the robot side of the line, otherwise no>\n'
        'Goal on robot side: <yes if the basket is on the robot side of the line, otherwise no>\n'
        'Separated: <yes if exactly one of the object and the basket is on the robot side, otherwise no>\n'
        'Verdict: <1 if the task can be completed without moving the object across the line, 0 if it cannot>')


def zero_shot_messages(row,mode):
    require(mode in ZERO_SHOT_INSTRUCTIONS,'Unknown zero-shot mode')
    text=VIEWS+'Task: '+row['student']['task']+'\nConstraints: '+row['student']['constraints']+'\n\n'+ZERO_SHOT_INSTRUCTIONS[mode]
    return [dict(role='system',content=ZERO_SHOT_SYSTEM),dict(role='user',content=[dict(type='image'),dict(type='image'),dict(type='text',text=text)])]
