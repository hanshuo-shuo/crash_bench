"""V5 fit diagnostic after S3 (docs/generative_verifier/V5_FIT_PROTOCOL.md): can B fit its robot-side facts at all?

B is trained from scratch, with the S3 hyperparameters, on 16 and on 64 training rows drawn from resting
counterfactual twin images, then read on 264 rows of the remaining such training images. The rules below go
into config.json, whose hash the campaign receipt records, before any GPU job starts. Only training rows are
read; validation and test are not scored.
"""
from collections import Counter,defaultdict
from .common import require
from .engineering_contract import epoch_orders
from .v5_contract import configuration as s3_configuration,digest

SIZES=(16,64)
SEEDS=(713,714,715)
SALT='v5-fit'
REST=.05
SIDES=('target_robot_side','goal_robot_side')
S3_FILES=('RECEIPT.json','config.json','preflight/COMPLETE.json','preflight/ENCODINGS.json','LOOKUP.json','S3_READING.json',
    'B_seed713/RESULT.json','B_seed713/selected_adapter/adapter_model.safetensors','B_seed713/selected_adapter/adapter_config.json')


def configuration():
    base=s3_configuration()
    return dict(version='V5-fit-diagnostic-v1',s3_configuration=base,arm='B',seeds=list(SEEDS),
        sizes={'16':dict(rows=16,images=8,per_source=1,epochs=40,optimizer_updates=160,evaluation_updates=[40,80,120,160]),
               '64':dict(rows=64,images=32,per_source=4,epochs=20,optimizer_updates=320,evaluation_updates=[80,160,240,320])},
        pool='Training images from counterfactual layouts that carry both answers and whose rows all have lift < 0.05 m',
        selection='Pool images in sha256(image|v5-fit) order; the 64-row set takes the first 32 with at most 4 per source, the 16-row '
            'set the first 8 of those with at most 1 per source; each image gives one separated and one non-separated row, the '
            'smallest sha256(image|rule|v5-fit) of each answer',
        held_out='Both rows of every other pool image (132 images, 264 rows); no image is shared with a fit set',
        effective_batch_size=base['effective_batch_size'],
        initialization='Fresh staged base and fresh LoRA for every size and seed; no historical adapter',
        unchanged_from_s3=['lora_rank','lora_alpha','lora_dropout','learning_rate','optimizer','weight_decay','gradient_clip',
            'micro_batch_size','effective_batch_size','max_length','image_pixels','objective','maximum_new_tokens'],
        readouts=dict(fit='Greedy generation on the fit set after each evaluation update: verdict, both robot-side facts right, each fact',
            fit_loss='Teacher-forced cross-entropy of the two robot-side value tokens on the fit set after the last update (eval mode)',
            transfer='Teacher-forced robot-side facts on the held-out rows after the last update: P(yes) vs P(no) at each value token '
                'with the gold prefix; compared with the training rule x progress lookup for that fact on the same rows',
            reference='The S3 B adapter (seed 713, selected update) read the same way on the held-out rows; it saw them once in training'),
        rules=dict(fits='Final greedy verdict and both-sides accuracy 1.0 on the fit set, and fit loss < 0.1',
            does_not_fit='Final greedy both-sides accuracy <= 0.6, or fit loss >= 0.5',partial='Any other run',
            size='Class shared by at least two of the three seeds, otherwise mixed',
            transfer='gain = mean over the two facts of (teacher-forced accuracy - lookup accuracy) on the held-out rows, averaged over '
                'the three 64-row runs; seen if gain >= 0.10, otherwise not seen',
            decisions={'cannot_fit':'B cannot fit 16 rows: inspect the V5 target encoding, masks and loss; if they hold, the 3B model '
                    'cannot separate these facts in the two views and the facts need another form (option 3)',
                'fits_16_only':'B fits 16 rows but not 64: the facts are hard to learn from these views; leans to option 3',
                'fits_and_transfers':'B fits 64 rows and the facts carry over to unseen training images: S3 was undertrained; next is '
                    'option 2 (three epochs)',
                'fits_without_transfer':'B fits 64 rows but the facts do not carry over: it memorises images rather than reading the '
                    'line; option 2 is still the direct test, but option 3 becomes more likely',
                'inconclusive':'Any other combination: report as measured and discuss'}),
        thresholds=dict(fit_loss=.1,stuck_loss=.5,stuck_both=.6,transfer_gain=.10),
        test_usage='No validation or test row is scored; the S3 test freeze stays untouched',hyperparameter_search=False,api_calls=0)


def pool_images(train):
    images=defaultdict(list)
    for r in train:
        require(r['split']=='train','Only training rows may be selected');images[r['metadata']['image_id']].append(r)
    return {i:rs for i,rs in images.items() if rs[0]['metadata']['layout_kind']=='counterfactual'
            and {r['target']['digit'] for r in rs}=={'0','1'} and all(r['metadata']['lift']<REST for r in rs)}


def pair(image,rows):
    """One separated and one non-separated row of an image, smallest digest of each."""
    return [min((r for r in rows if r['target']['digit']==d),key=lambda r:digest(image,r['metadata']['rule'],SALT))['sample_id'] for d in ('0','1')]


def subsets(train,config):
    """Nested 16- and 64-row fit sets and the disjoint held-out rows, all from the same pool of training images."""
    pool=pool_images(train);ranked=sorted(pool,key=lambda i:digest(i,SALT));source=lambda i:pool[i][0]['metadata']['source_id']
    def take(candidates,n,cap):
        chosen,count=[],Counter()
        for i in candidates:
            if len(chosen)<n and count[source(i)]<cap:chosen.append(i);count[source(i)]+=1
        require(len(chosen)==n,'Too few pool images for a fit set');return chosen
    s=config['sizes'];big=take(ranked,s['64']['images'],s['64']['per_source']);small=take(big,s['16']['images'],s['16']['per_source'])
    held=[i for i in ranked if i not in set(big)]
    result={}
    for name,images in (('16',small),('64',big),('held_out',held)):
        ids=[x for i in images for x in (pair(i,pool[i]) if name!='held_out' else sorted(r['sample_id'] for r in pool[i]))]
        result[name]=dict(images=sorted(images),sample_ids=sorted(ids),sources=len({source(i) for i in images}))
    by_id={r['sample_id']:r for r in train}
    for name,part in result.items():
        rows=[by_id[x] for x in part['sample_ids']]
        part.update(labels=dict(Counter(r['target']['digit'] for r in rows)),
            sides=dict(Counter('T%s G%s'%(r['target']['facts']['target_robot_side'],r['target']['facts']['goal_robot_side']) for r in rows)))
    return result


def orders(sample_ids,size,config):
    """Complete epochs over one fit set, one order per seed."""
    s=config['sizes'][size];ids=sorted(sample_ids);require(len(ids)==s['rows'],'Fit set size differs');result={}
    for seed in config['seeds']:
        batches=epoch_orders(len(ids),seed,s['epochs'],config['effective_batch_size'])
        require(len(batches)==s['optimizer_updates'],'Update count differs from the epoch schedule')
        result[str(seed)]=[[[ids[i],'facts'] for i in b] for b in batches]
    return result


def classify_run(verdict,both,loss,t):
    if verdict==1 and both==1 and loss<t['fit_loss']:return 'fits'
    if both<=t['stuck_both'] or loss>=t['stuck_loss']:return 'does_not_fit'
    return 'partial'


def size_class(classes):
    require(len(classes)==len(SEEDS),'One class per seed required')
    top,count=Counter(classes).most_common(1)[0];return top if count>=2 else 'mixed'


def decision(fit16,fit64,gain,t):
    if fit16=='does_not_fit':return 'cannot_fit'
    if fit16=='fits' and fit64=='does_not_fit':return 'fits_16_only'
    if fit16=='fits' and fit64=='fits':return 'fits_and_transfers' if gain>=t['transfer_gain'] else 'fits_without_transfer'
    return 'inconclusive'


def fact_lookup(train,field):
    """Rule x progress majority of one robot-side fact on the training rows (fallback rule, then global)."""
    key=lambda r:(r['metadata']['rule'],r['metadata']['time_bin'])
    tables=(defaultdict(Counter),defaultdict(Counter))
    for r in train:tables[0][key(r)][r['target']['facts'][field]]+=1;tables[1][r['metadata']['rule']][r['target']['facts'][field]]+=1
    overall=Counter(r['target']['facts'][field] for r in train).most_common(1)[0][0]
    def guess(r):
        for table,k in ((tables[0],key(r)),(tables[1],r['metadata']['rule'])):
            if table.get(k):return table[k].most_common(1)[0][0]
        return overall
    return guess
