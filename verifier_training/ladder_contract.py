"""Step 2 ladder: which part of B's output makes the verdict learnable, and does it carry to held-out sources?
(docs/generative_verifier/LADDER_PROTOCOL.md)

Each rung adds one thing before the same verdict digit, on the same inputs:
  A  : Verdict: d                                  (Phase 2 runs, reused)
  AF : Protected: box, newline, Verdict: d          (four constant tokens; no input information)
  AP : Protected: <colour>, newline, Verdict: d     (the protected colour restated)
  B  : four structured facts, then Verdict: d       (Phase 2 runs, reused)
Part 1 fits AF on the step 1 16 rows; AP's 16-row fit is in the controls campaign. Part 2 trains AF and AP on
the 512 training rows with the Phase 2 schedule and orders, and scores them on the 192 validation rows only;
A and B are read from the Phase 2 validation records at the same updates. No test row is loaded.
"""
from .common import require
from .engineering_contract import epoch_orders
from .fd1000_contract import SEEDS
from .control_contract import configuration as controls_configuration

FILLER='box'
ARMS16=('AF',)
ARMS512=('AF','AP')
RUNGS=('A','AF','AP','B')
COMPARISONS=(('AP','B'),('AP','A'),('AF','A'),('AP','AF'))
PRIMARY_UPDATE=384
PHASE2_FILES=('RECEIPT.json','config.json','ORDERS.json','preflight/COMPLETE.json','preflight/ENCODINGS.json')+tuple(
    f'{arm}_seed{s}/{name}' for arm in ('A','B') for s in SEEDS
    for name in ['VALIDATION_SCORES.json','curve.jsonl']+[f'validation_step_{u}.jsonl' for u in (128,256,384)])
STEP1_FILES=('RECEIPT.json','config.json','SUBSET.json','ORDERS.json','COMPLETE.json')
CONTROLS_FILES=('RECEIPT.json','config.json','COMPLETE.json','REPORT.json')


def answer(row,arm):
    """Assistant target of a new rung; the verdict digit is the original label."""
    digit=int(row['target']['verdict']=='feasible')
    if arm=='AF':return f'Protected: {FILLER}\nVerdict: {digit}'
    require(arm=='AP','Unknown ladder rung');return f'Protected: {row["target"]["facts"]["protected"]}\nVerdict: {digit}'


def fit16_orders(sample_ids,config):
    """Complete epochs over the 16 rows; the step 1 sample sequence for each seed."""
    o=config['fit16'];ids=sorted(sample_ids);require(len(ids)==o['rows'],'Subset size differs')
    result={}
    for seed in o['seeds']:
        batches=epoch_orders(len(ids),seed,o['epochs'],o['effective_batch_size'])
        require(len(batches)==o['optimizer_updates'],'Update count differs from the epoch schedule')
        result[str(seed)]={arm:[[[ids[i],'facts'] for i in batch] for batch in batches] for arm in o['arms']}
    return result


def comparison_reading(interval):
    """Sign of a source-clustered 95% interval of treatment minus control."""
    low,high=interval
    return 'treatment_higher' if low>0 else 'treatment_lower' if high<0 else 'no_detected_difference'


def fit16_reading(af):
    return {'fits':'extra_tokens_restore_fit','does_not_fit':'colour_needed_for_fit'}.get(af,'inconclusive')


def configuration():
    controls=controls_configuration();base=controls['phase2_configuration']
    target=lambda text:dict(target=text,kind='facts',loss='facts_verdict')
    return dict(version='V4-ladder-v1',phase2_configuration=base,controls_configuration_version=controls['version'],
        fit16=dict(controls['overfit'],arms=list(ARMS16),selection='The step 1 subset (SUBSET.json), checked again on the GPU node'),
        train512=dict(arms=list(ARMS512),seeds=list(SEEDS),rows=512,epochs=base['epochs'],optimizer_updates=base['optimizer_updates'],
            evaluation_updates=base['evaluation_updates'],effective_batch_size=base['effective_batch_size'],
            order='Phase 2 ORDERS.json, rung B of each seed (the A/B/C sample sequence with the facts target kind)',
            evaluation='Greedy free generation on the 192 validation rows after updates 128, 256 and 384; checkpoint saved at each'),
        rungs=dict(
            A=dict(target='Verdict: d',source='Phase 2 A runs, reading direct_free'),
            AF=dict(target(f'Protected: {FILLER}, a newline, then Verdict: d'),reading='AF_free',key_fields=[],
                adds='Four constant output tokens before the verdict; same token positions as AP, no input information'),
            AP=dict(target('Protected: <colour>, a newline, then Verdict: d'),reading='AP_free',key_fields=['protected'],
                adds='The protected colour, restated as an output token right before the verdict'),
            B=dict(target='Protected, target-inside, goal-inside and separated facts, then Verdict: d',source='Phase 2 B runs, reading facts_free')),
        comparisons=[list(c) for c in COMPARISONS],primary_update=PRIMARY_UPDATE,
        statistics=dict(method='fd1000_metrics.clustered_difference: row correctness averaged over the three seeds, '
                'treatment minus control, 14 validation source groups, bootstrap 1000 replicates with seed 1014, exact sign-flip test',
            reading='CI lower bound > 0: treatment_higher; upper bound < 0: treatment_lower; otherwise no_detected_difference, '
                'which is not equivalence: report the interval width',
            scope='Exploratory development-data validation; the Phase 2 locked test is never scored again'),
        rules=dict(fit16=dict(fits=controls['rules']['fits'],does_not_fit=controls['rules']['does_not_fit'],partial=controls['rules']['partial'],
                arm=controls['rules']['arm'],readings={
                'extra_tokens_restore_fit':'AF fits: four constant tokens before the verdict are enough to fit the 16 rows, so the colour content is not needed for fitting',
                'colour_needed_for_fit':'AF does not fit: the restated colour, not the extra tokens, is what restores fitting',
                'inconclusive':'AF partial or mixed: report as measured'}),
            comparisons={
                'AP-B':{'treatment_lower':'The containment facts add held-out accuracy beyond restating the colour',
                    'no_detected_difference':'No detected validation gap between restating the colour and the full facts',
                    'treatment_higher':'Restating the colour scores above the full facts'},
                'AP-A':{'treatment_higher':'Restating the colour improves held-out accuracy over the direct verdict',
                    'no_detected_difference':'No detected validation gain from restating the colour','treatment_lower':'AP below A'},
                'AF-A':{'treatment_higher':'Constant extra tokens alone improve held-out accuracy',
                    'no_detected_difference':'No detected validation gain from constant extra tokens','treatment_lower':'AF below A'},
                'AP-AF':{'treatment_higher':'The colour content helps held-out accuracy beyond the extra tokens',
                    'no_detected_difference':'No detected validation difference between the colour and the constant tokens',
                    'treatment_lower':'AF above AP'}}),
        thresholds=controls['thresholds'],
        deferred=['R (two relations and the program rule): B already equals its own relations under the program rule row by row, '
                'and R has no verdict token for the current loss','Initial per-row gradient-cancellation measurement',
            'Confirmatory comparison on new locked sources (needs a separate data-collection approval)'],
        test_usage='No test row is loaded; the Phase 2 locked test is never scored again',
        hyperparameter_search=False,api_calls=0)
