"""Single-pathway label controls after step 1 (docs/generative_verifier/CONTROLS_PROTOCOL.md).

Same 16 training rows, sample order, direct encoding path and Phase 2 hyperparameters as step 1;
only the training target changes:
  t  : Verdict: 1 when the rule protects the orange box (text only)
  s  : Verdict: 1 when the separating box is orange (image only); the verdict is s XOR t
  AP : Protected: <colour> then Verdict: <verdict> (restate the protected colour, then judge)
The rules below are written into config.json, whose hash the receipt records, before the GPU job
starts. No test row is loaded. The gradient measurement is deferred by the user's 2026-10-18 scope.
"""
from collections import Counter,defaultdict
import copy
from .common import require
from .engineering_contract import epoch_orders
from .fit_contract import configuration as step1_configuration

ARMS=('t','s','AP')
KIND={'t':'direct','s':'direct','AP':'facts'}
OTHER={'orange':'purple','purple':'orange'}
PHASE2_FILES=('RECEIPT.json','config.json','preflight/COMPLETE.json','preflight/ENCODINGS.json')
STEP1_FILES=('RECEIPT.json','config.json','SUBSET.json','ORDERS.json','COMPLETE.json','REPORT.json')
PATHWAY={('fits','fits'):'composition_bottleneck',('fits','does_not_fit'):'image_relation_bottleneck',
    ('does_not_fit','fits'):'text_pathway_problem',('does_not_fit','does_not_fit'):'direct_path_problem'}
AP_READING={'fits':'restatement_restores_fit','does_not_fit':'restatement_insufficient'}


def text_bit(row):
    """t: 1 when the rule protects the orange box."""
    colour=row['target']['facts']['protected'];require(colour in OTHER,'Unknown protected colour')
    return int(colour=='orange')


def separating_colour(row):
    """Colour of the box that holds exactly one of target and goal; by construction it is boundary a."""
    f=row['target']['facts'];inside=(f['target_inside'],f['goal_inside'])
    require(set(inside)<={'yes','no'},'Containment facts must be yes/no')
    separated=inside[0]!=inside[1];require(f['separated']==('yes' if separated else 'no'),'Separated fact inconsistent')
    colour=f['protected'] if separated else OTHER[f['protected']]
    require(colour==row['metadata']['appearance_assignment']['a'],'Separating box is not boundary a')
    return colour


def image_bit(row):
    """s: 1 when the separating box is orange."""
    return int(separating_colour(row)=='orange')


def label(row,arm):
    require(arm in ARMS,'Unknown control arm')
    if arm=='t':return text_bit(row)
    if arm=='s':return image_bit(row)
    return int(row['target']['verdict']=='feasible')


def answer(row,arm):
    """Assistant target text of a control arm."""
    if arm=='AP':return f'Protected: {row["target"]["facts"]["protected"]}\nVerdict: {label(row,arm)}'
    return f'Verdict: {label(row,arm)}'


def relabel(row,arm):
    """Copy of a row whose scored truth is the control label; inputs, facts and metadata are unchanged."""
    out=copy.deepcopy(row);out['target']['verdict']='feasible' if label(row,arm) else 'infeasible';out['target']['control']=arm
    return out


def check_labels(rows):
    """verdict = s XOR t on every row; s depends only on the images, t only on the text; each input group is 50/50 on the verdict."""
    groups={'image':defaultdict(list),'text':defaultdict(list)}
    for r in rows:
        t,s=text_bit(r),image_bit(r);verdict=int(r['target']['verdict']=='feasible')
        require(verdict==s^t,'Verdict is not s XOR t')
        groups['image'][tuple(i['sha256'] for i in r['student']['images'])].append((s,verdict))
        groups['text'][(r['student']['task'],r['student']['constraints'])].append((t,verdict))
    for name,by in groups.items():
        require(all(len({b for b,v in g})==1 for g in by.values()),f'The {name} label depends on more than the {name}')
        require(all(Counter(v for b,v in g)[0]==Counter(v for b,v in g)[1] for g in by.values()),f'Verdict is not 50/50 within a {name} group')
    balance={arm:Counter(label(r,arm) for r in rows) for arm in ('t','s')}
    require(all(c[0]==c[1] for c in balance.values()),'Control labels unbalanced')
    return dict(rows=len(rows),image_groups=len(groups['image']),text_groups=len(groups['text']),verdict_is_s_xor_t=True,
        image_label_constant_per_image_pair=True,text_label_constant_per_text=True,verdict_half_feasible_in_every_group=True,
        balance={arm:{str(k):v for k,v in sorted(c.items())} for arm,c in balance.items()})


def orders(sample_ids,config):
    """Complete epochs over the 16 rows; for each seed every arm sees the step 1 sample sequence."""
    o=config['overfit'];ids=sorted(sample_ids);require(len(ids)==o['rows'],'Subset size differs')
    result={}
    for seed in o['seeds']:
        batches=epoch_orders(len(ids),seed,o['epochs'],o['effective_batch_size'])
        require(len(batches)==o['optimizer_updates'],'Update count differs from the epoch schedule')
        result[str(seed)]={arm:[[[ids[i],KIND[arm]] for i in batch] for batch in batches] for arm in o['arms']}
    return result


def pathway_reading(t,s):
    return PATHWAY.get((t,s),'inconclusive')


def ap_reading(ap):
    return AP_READING.get(ap,'inconclusive')


def configuration():
    step1=step1_configuration();rules=step1['rules']
    return dict(version='V4-fit-controls-v1',phase2_configuration=step1['phase2_configuration'],step1_configuration_version=step1['version'],
        overfit=dict(step1['overfit'],arms=list(ARMS),selection='The step 1 subset (SUBSET.json), checked again on the GPU node'),
        controls=dict(
            t=dict(target='Verdict: 1 when the rule protects the orange box, else Verdict: 0',kind='direct',reading='t_free',loss='direct_verdict',
                pathway='Text only: one label for both image assignments of a rule text'),
            s=dict(target='Verdict: 1 when the separating box (holding exactly one of target and goal) is orange, else Verdict: 0',
                kind='direct',reading='s_free',loss='direct_verdict',pathway='Image only: one label for both rules of an image pair'),
            AP=dict(target='Protected: <colour>, a newline, then Verdict: <0/1 verdict>',kind='facts',reading='AP_free',loss='facts_verdict',
                key_fields=['protected'],pathway='Both; restates the protected colour before the verdict, which also changes the output structure')),
        label_checks='verdict = s XOR t; s constant per image pair, t constant per rule text; verdict 50/50 within every image and text group; '
            'checked on the 16 rows and on all 512 training rows before training',
        rules=dict(fits=rules['fits'],does_not_fit=rules['does_not_fit'],partial=rules['partial'],arm=rules['arm'],
            fit_loss='Cross-entropy of the digit after Verdict: (direct_verdict for t and s, facts_verdict for AP), mean over the last epoch',
            evaluation='Greedy free generation on the 16 rows after updates 40, 80, 120 and 160; accuracy against the arm label',
            pathway_readings={
                'composition_bottleneck':'t and s fit while the step 1 verdict did not: the difficulty lies in combining the two pathways; '
                    'not by itself a gradient-mechanism claim',
                'image_relation_bottleneck':'t fits, s does not: the direct path does not learn which box separates target and goal',
                'text_pathway_problem':'s fits, t does not: a text-only label is not learned; check implementation, optimization and input distinguishability',
                'direct_path_problem':'Neither label fits: check the direct path implementation and optimization; not proof of a bug',
                'inconclusive':'A partial or mixed arm: report as measured and discuss before any next step'},
            ap_readings={
                'restatement_restores_fit':'AP fits: restating the colour, together with the output-structure change, restores fitting; '
                    'this job cannot separate the two',
                'restatement_insufficient':'AP does not fit: a colour restatement is not enough; the containment relations of B are needed',
                'inconclusive':'AP partial or mixed: report as measured'},
            reference='Step 1 on the same rows, orders and seeds: A does_not_fit, B fits'),
        thresholds=step1['thresholds'],
        deferred=['Initial per-row gradient-cancellation measurement (user scope 2026-10-18: can come later)',
            'A 4x update budget or learning-rate sweep'],
        test_usage='No test row is loaded; the Phase 2 locked test is never scored again',
        hyperparameter_search=False,api_calls=0)
