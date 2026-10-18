"""Step 1 of docs/generative_verifier/NEXT_STEPS.md section 7.2: can the direct arm A fit its training data?

1a scores the saved Phase 2 A adapters on the 512 training rows. 1b trains A and B
from scratch on 16 training rows (one four-row unit per construction cell) with the
Phase 2 hyperparameters. The rules below are written into config.json, whose hash the
campaign receipt records, before the GPU job starts. No test row is ever loaded.
"""
from collections import Counter,defaultdict
import hashlib
from .common import require
from .engineering_contract import epoch_orders
from .fd1000_contract import configuration as phase2_configuration,arm_kinds,readings

SEEDS=(713,714,715)
TRAIN_CHECKPOINTS=(128,384)
FIT_LOSS={'A':'direct_verdict','B':'facts_key_values'}


def configuration():
    base=phase2_configuration('main')
    return dict(version='V4-fit-diagnostic-v1',phase2_configuration=base,
        train_evaluation=dict(arm='A',seeds=list(SEEDS),checkpoints=list(TRAIN_CHECKPOINTS),reading='direct_free',
            rows='All 512 training rows',mode='Greedy generation in eval mode (no dropout); P(1) read at the verdict token'),
        overfit=dict(arms=['A','B'],seeds=list(SEEDS),rows=16,units_per_cell=1,epochs=40,optimizer_updates=160,
            evaluation_updates=[40,80,120,160],effective_batch_size=base['effective_batch_size'],
            selection='Per construction cell, the first training unit in sha256("v4-fit16/" + unit_id) order whose source group is not yet used',
            initialization='Fresh staged base and fresh LoRA for every arm and seed',
            unchanged_from_phase2=['lora_rank','lora_alpha','lora_dropout','learning_rate','optimizer','weight_decay','gradient_clip',
                'micro_batch_size','effective_batch_size','max_length','image_pixels','objective','maximum_new_tokens']),
        rules=dict(fit_loss=dict(FIT_LOSS,meaning='Decisive tokens: the verdict digit for A; the four fact values for B, '
                'whose verdict follows from its own facts under teacher forcing'),
            fits='Final greedy accuracy 16/16 and mean fit loss over the last epoch (4 updates) < 0.1',
            does_not_fit='Final greedy accuracy <= 10/16, or mean fit loss over the last epoch >= 0.6',
            partial='Any other run',arm='Class shared by at least two of the three seeds, otherwise mixed',
            train_at_chance='Every seed at update 384 has training greedy accuracy <= 0.6',
            train_fits='At least two seeds at update 384 have training greedy accuracy >= 0.9',
            train_partial='Any other training-set result',
            decisions={'implementation_suspect':'A does not fit 16 rows while B does: inspect the direct target encoding, masks and loss before step 2',
                'shared_pipeline_suspect':'Neither arm fits 16 rows: inspect the shared training path before step 2',
                'optimization_stall':'A fits 16 rows but sits at chance on 512: step 2 adds one predeclared A training-budget variant',
                'fits_training_data':'A fits 16 and 512 training rows in eval mode, so its validation failure is a generalization failure: step 2 runs A unchanged',
                'inconclusive':'Any other combination: report as measured and discuss before step 2'}),
        thresholds=dict(fit_loss=.1,stuck_loss=.6,stuck_accuracy=10/16,train_chance=.6,train_fit=.9),
        test_usage='No test row is loaded; the Phase 2 locked test is never scored again',
        hyperparameter_search=False,api_calls=0)


def phase2_files():
    """Phase 2 artifacts this diagnostic reads; their hashes are recorded at launch and checked on the GPU node."""
    files=['RECEIPT.json','config.json','preflight/COMPLETE.json','preflight/ENCODINGS.json']
    files+=[f'A_seed{s}/checkpoint_{c}/{name}' for s in SEEDS for c in TRAIN_CHECKPOINTS for name in ('adapter_model.safetensors','adapter_config.json')]
    return files+[f'{arm}_seed{s}/curve.jsonl' for arm in 'ABC' for s in SEEDS]


def overfit_units(train_rows):
    """One whole training unit per construction cell, from four distinct source groups, in fixed hash order."""
    units=defaultdict(list)
    for r in train_rows:
        require(r['split']=='train','Only training rows may be selected');units[r['metadata']['unit_id']].append(r)
    cells=defaultdict(list)
    for uid,members in units.items():
        require(len(members)==4 and len({m['metadata']['construction_cell'] for m in members})==1 and
                len({m['metadata']['ancestry_component'] for m in members})==1,'Incomplete or mixed training unit')
        cells[members[0]['metadata']['construction_cell']].append(uid)
    require(len(cells)==4,'Expected four construction cells')
    chosen=[];groups=set()
    for cell in sorted(cells):
        ranked=sorted(cells[cell],key=lambda u:hashlib.sha256(f'v4-fit16/{u}'.encode()).hexdigest())
        uid=next((u for u in ranked if units[u][0]['metadata']['ancestry_component'] not in groups),None)
        require(uid is not None,'No unused source group in a construction cell')
        chosen.append(uid);groups.add(units[uid][0]['metadata']['ancestry_component'])
    ids=sorted(r['sample_id'] for u in chosen for r in units[u])
    return dict(units=chosen,sample_ids=ids,cells={units[u][0]['metadata']['construction_cell']:u for u in chosen},
        source_groups=sorted(groups),labels=dict(Counter(r['target']['verdict'] for u in chosen for r in units[u])))


def overfit_orders(sample_ids,config):
    """Complete epochs over the 16 rows; A and B of one seed see the same sample order."""
    o=config['overfit'];ids=sorted(sample_ids);require(len(ids)==o['rows'],'Overfit subset size differs')
    result={}
    for seed in o['seeds']:
        batches=epoch_orders(len(ids),seed,o['epochs'],o['effective_batch_size'])
        require(len(batches)==o['optimizer_updates'],'Update count differs from the epoch schedule')
        result[str(seed)]={arm:[[[ids[i],arm_kinds(arm)[0]] for i in batch] for batch in batches] for arm in o['arms']}
    return result


def reading(arm):
    """The single greedy selection reading of arm A or B."""
    names=list(readings(arm));require(len(names)==1,'Arm has more than one selection reading');return names[0]


def last_epoch_loss(curve,arm,config,key=None):
    """Mean fit loss (the arm's FIT_LOSS unless key is given) over the final epoch of training updates."""
    key=key or FIT_LOSS[arm];per_epoch=config['overfit']['rows']//config['overfit']['effective_batch_size'];tail=curve[-per_epoch:]
    require(len(curve)==config['overfit']['optimizer_updates'] and all(key in r for r in tail),'Incomplete training curve')
    return sum(r[key] for r in tail)/len(tail)


def classify_run(final_accuracy,loss,thresholds):
    if final_accuracy==1 and loss<thresholds['fit_loss']:return 'fits'
    if final_accuracy<=thresholds['stuck_accuracy'] or loss>=thresholds['stuck_loss']:return 'does_not_fit'
    return 'partial'


def arm_class(classes):
    require(len(classes)==len(SEEDS),'One class per seed required')
    top,count=Counter(classes).most_common(1)[0];return top if count>=2 else 'mixed'


def train_class(accuracies,thresholds):
    """accuracies: training greedy accuracy of each seed's update-384 adapter."""
    require(len(accuracies)==len(SEEDS),'One accuracy per seed required');values=list(accuracies)
    if all(a<=thresholds['train_chance'] for a in values):return 'at_chance'
    if sum(a>=thresholds['train_fit'] for a in values)>=2:return 'fits'
    return 'partial'


def decision(a16,b16,a512):
    if a16=='does_not_fit' and b16=='fits':return 'implementation_suspect'
    if a16=='does_not_fit' and b16=='does_not_fit':return 'shared_pipeline_suspect'
    if a16=='fits' and a512=='at_chance':return 'optimization_stall'
    if a16=='fits' and a512=='fits':return 'fits_training_data'
    return 'inconclusive'
