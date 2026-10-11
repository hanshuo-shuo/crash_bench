"""Fixed, predeclared engineering protocol. No tuning or new simulation."""
import copy
import random
import re
from .common import require, rationale, student_messages

DATA_SHA = '1ab19294758acd2c59cda91077b15fd28938417685a9470d5e72c43a7d41c7b0'
CONDITIONS = ('original', 'shuffled_pairs', 'text_only', 'horizon_600', 'horizon_0')


def epoch_orders(n, seed, epochs, batch_size):
    require(n % batch_size == 0, 'Complete epoch batches required')
    rng = random.Random(seed)
    batches = []
    for epoch in range(epochs):
        order = list(range(n)); rng.shuffle(order)
        batches.extend(order[j:j+batch_size] for j in range(0, n, batch_size))
    return batches


def execution_config(proposal):
    c = copy.deepcopy(proposal)
    expected = {'seed':713, 'lora_rank':16, 'lora_alpha':32, 'lora_dropout':.05,
                'learning_rate':.0001, 'max_steps':300, 'micro_batch_size':1,
                'effective_batch_size':4, 'epochs':10, 'training_samples':120,
                'max_length':1536, 'image_pixels':50176, 'dataset_sha256':DATA_SHA}
    for k,v in expected.items(): require(c[k] == v, 'Fixed proposal changed: '+k)
    c.update(mode='authorized_full_dataset_engineering', approved_for_launch=True,
             authorization='User explicitly requested fixed A/B fit on all 120 after observability isolation',
             evaluate_every=30, generate_every=150, early_stopping=False)
    return c


def shuffled_donors(n, seed=714):
    # Seeded single-cycle permutation: every coherent pair moves, without labels.
    cycle=list(range(n)); random.Random(seed).shuffle(cycle)
    donors=[None]*n
    for i,j in zip(cycle,cycle[1:]+cycle[:1]): donors[i]=j
    return donors


def diagnostic_row(row, condition, donor=None):
    r=copy.deepcopy(row)
    if condition=='shuffled_pairs':
        require(donor is not None, 'Missing donor')
        r['student']['images']=copy.deepcopy(donor['student']['images'])
    elif condition.startswith('horizon_'):
        horizon=int(condition.split('_')[1])
        old='Use at most 300 original 7D OSC/gripper commands'
        require(old in r['student']['constraints'], 'Unexpected horizon contract')
        r['student']['constraints']=r['student']['constraints'].replace(old,f'Use at most {horizon} original 7D OSC/gripper commands',1)
    return r


def diagnostic_messages(row, variant, condition):
    messages=student_messages(row,variant)
    if condition=='text_only':
        messages[1]['content']=[x for x in messages[1]['content'] if x['type']!='image']
        messages[1]['content'][0]['text']=messages[1]['content'][0]['text'].replace(
            'View 1: external camera. View 2: wrist camera.\n', 'No images are supplied.\n',1)
    return messages


def parse_verdict(text, variant):
    pattern=r'Verdict: (feasible|infeasible)' if variant=='A' else r'Rationale: .+\nVerdict: (feasible|infeasible)'
    m=re.fullmatch(pattern,text,flags=re.S)
    return m.group(1) if m else None


def score_rationale(text, row, condition):
    """Conservative finite-grammar checker; unknown prose always needs review."""
    if not text.startswith('Rationale: ') or '\nVerdict: ' not in text:
        return {'status':'invalid_format','contradictions':[], 'unsupported':[], 'unparsed':[text]}
    body=text[len('Rationale: '):text.rfind('\nVerdict: ')]
    sentences=[s.strip()+'.' for s in body.split('.') if s.strip()]
    expected,_=rationale(row['metadata']['visibility'])
    expected_sentences=[s.strip()+'.' for s in expected.split('.') if s.strip()]
    visibility=expected_sentences[:3]
    caveats=expected_sentences[3:]
    known=[]
    for name in ('The milk is','The basket is','Red barriers are'):
        known.extend(name+' '+ending+'.' for ending in ('visible in both views','visible only in the external view',
                      'visible only in the wrist view','not visible in either view'))
    contradictions=[s for s in sentences if s in known and s not in visibility and condition!='text_only']
    unsupported=[s for s in sentences if s in known and condition=='text_only']
    unparsed=[s for s in sentences if s not in known+caveats]
    missing=[s for s in expected_sentences if s not in sentences]
    status=('contradiction' if contradictions else 'unsupported_no_images' if unsupported else
            'needs_manual_review' if unparsed else 'supported_complete' if not missing else 'supported_incomplete')
    return {'status':status,'contradictions':contradictions,'unsupported':unsupported,'unparsed':unparsed,
            'missing_expected':missing,'exact_observability_template':body==expected,
            'physical_verdict_justification_established':False}
