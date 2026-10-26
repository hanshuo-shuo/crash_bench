"""V5 test evaluation: the six selected adapters scored once on the frozen test split (docs/generative_verifier/V5_TEST_PROTOCOL.md).

A keeps the frozen S3 settings and P the frozen S4 settings; the adapters are the ones the S3, S4 and seed runs selected on
validation, frozen at launch with their hashes. Before any test row, each job reloads its adapter and must reproduce its
recorded validation outputs on a fixed set of validation rows. The reading repeats the seed campaign's on the test split: row
means over the three seeds, the paired image bootstrap and the same thresholds.
"""
from pathlib import Path
from .common import read,sha,require
from .v5_contract import DATASET,TEST_SPLITS,digest
from .v5_seeds_contract import configuration as seeds_configuration,ARMS,ALL_SEEDS,REFERENCE_SEED,THRESHOLDS
from .v5_metrics import REPLICATES,SEED

VERSION='V5-object-test-v1'
TARGETS=tuple(f'{arm}_seed{seed}' for arm in ARMS for seed in ALL_SEEDS)
LANES=(('P_seed713','A_seed713','A_seed714','A_seed715'),('P_seed714','P_seed715'))
SMOKE_SPLITS=('validation_natural',)
RELOAD_ROWS=32
RELOAD_SALT='v5-test-reload'
RELOAD_TEXT_MISMATCHES=0
S3_FILES=('RECEIPT.json','config.json','preflight/ENCODINGS.json','LOOKUP.json','S3_READING.json')
S4_FILES=('RECEIPT.json','config.json','POINTS.json','S4_READING.json')
SEEDS_FILES=('RECEIPT.json','config.json','SEEDS_READING.json')
RUN_FILES=('RESULT.json','validation.jsonl','validation_natural.jsonl','selected_adapter/adapter_model.safetensors',
    'selected_adapter/adapter_config.json')


def split_target(target):
    require(target in TARGETS,'Unknown target: '+target)
    arm,seed=target.split('_seed');return arm,int(seed)


def target_root(target,roots):
    """Seed 713 lives in the S3 (A) and S4 (P) runs; seeds 714 and 715 in the seed campaign."""
    arm,seed=split_target(target)
    if seed==REFERENCE_SEED:return Path(roots['s3' if arm=='A' else 's4'])
    return Path(roots['seeds'])


def evidence_files(roots):
    """Every prior file the test campaign reads, by run: campaign files plus each target's run files."""
    files={'s3':list(S3_FILES),'s4':list(S4_FILES),'seeds':list(SEEDS_FILES)}
    for target in TARGETS:
        key={Path(roots[k]):k for k in ('s3','s4','seeds')}[target_root(target,roots)]
        files[key]+=[f'{target}/{name}' for name in RUN_FILES]
    return files


def evidence(roots):
    return {k:{name:sha(Path(roots[k])/name) for name in names} for k,names in evidence_files(roots).items()}


def check_evidence(recorded,roots):
    require({k:sorted(v) for k,v in recorded.items()}=={k:sorted(v) for k,v in evidence_files(roots).items()},'Evidence list differs')
    for key,files in recorded.items():
        for name,digest_ in files.items():require(sha(Path(roots[key])/name)==digest_,'Evidence changed: %s %s'%(key,name))


def freeze(roots):
    """The six selected adapters as the S3, S4 and seed runs left them; nothing is reselected."""
    result={}
    for target in TARGETS:
        arm,seed=split_target(target);folder=target_root(target,roots)/target;done=read(folder/'RESULT.json')
        require(done['passed'] and done['arm']==arm and done['seed']==seed and done['test_rows_loaded']==0,'Run incomplete: '+target)
        adapter=sha(folder/'selected_adapter/adapter_model.safetensors');require(adapter==done['adapter_sha256'],'Adapter differs from RESULT: '+target)
        result[target]=dict(arm=arm,seed=seed,run_dir=str(folder),selected_step=done['selected_step'],adapter_sha256=adapter,
            result_sha256=sha(folder/'RESULT.json'),validation_accuracy=done['R1_validation_accuracy' if arm=='A' else 'R1_program_accuracy'])
    return result


def reload_subset(validation_ids,n=RELOAD_ROWS):
    """Fixed validation rows on which a reloaded adapter must reproduce its recorded outputs."""
    ids=sorted(set(validation_ids));require(len(ids)>=n,'Too few validation rows for the reload check')
    return sorted(sorted(ids,key=lambda s:digest(s,RELOAD_SALT))[:n])


def reload_agreement(new,old,arm,allowed=RELOAD_TEXT_MISMATCHES):
    """Reloaded outputs against the recorded validation outputs of the same adapter: every decision equal, texts within allowed."""
    old={r['sample_id']:r for r in old};require(len(new)==len({r['sample_id'] for r in new}) and all(r['sample_id'] in old for r in new),
        'Reload rows are not recorded validation rows')
    decision='predicted' if arm=='A' else 'program'
    same_decision=[r['sample_id'] for r in new if r[decision]==old[r['sample_id']][decision]]
    same_text=[r['sample_id'] for r in new if r['generated']==old[r['sample_id']]['generated']]
    same_input=all(r['student_input_sha256']==old[r['sample_id']]['student_input_sha256'] for r in new)
    passed=bool(new) and same_input and len(same_decision)==len(new) and len(new)-len(same_text)<=allowed
    return dict(passed=passed,rows=len(new),decision=decision,decisions_equal=len(same_decision),texts_equal=len(same_text),
        inputs_equal=same_input,allowed_text_mismatches=allowed,
        mismatches=[dict(sample_id=r['sample_id'],new=r['generated'],old=old[r['sample_id']]['generated']) for r in new
            if r['generated']!=old[r['sample_id']]['generated']])


def configuration():
    return dict(version=VERSION,seeds_configuration=seeds_configuration(),dataset=str(DATASET),targets=list(TARGETS),lanes=[list(l) for l in LANES],
        evaluated_splits=list(TEST_SPLITS),primary_split='test',
        adapters='The selected_adapter of each run (A: S3 A_seed713 and seed-campaign A_seed714/715; P: S4 P_seed713 and seed-campaign '
            'P_seed714/715), frozen at launch in FREEZE.json with their hashes; no reselection, no training',
        loading='Staged base model in bf16 plus the adapter with PeftModel.from_pretrained (is_trainable=False); TF32 matmuls on as in the '
            'training-time evaluation; greedy decoding exactly as on validation',
        arm_settings={'A':'S3 arm A: v5_run.item and v5_run.evaluate, reading direct_free, at most 8 new tokens',
            'P':'S4 arm P: v5_pos_run.p_item and v5_pos_run.evaluate, program verdict primary, at most 128 new tokens'},
        reload_check=dict(rows=RELOAD_ROWS,salt=RELOAD_SALT,split='validation',allowed_text_mismatches=RELOAD_TEXT_MISMATCHES,
            rule='Before any test row: the reloaded adapter regenerates these validation rows; every decision (A verdict, P program verdict) and '
                'every input digest must equal the recorded validation record, and at most allowed_text_mismatches generated texts may differ'),
        preflight='Test rows hashed against TEST_FREEZE.json; unit files against their HASHES.json; A and P inputs equal per row; '
            'the program on the projected true points agrees with every test label; rule x progress lookup from train equal to the S2 readout',
        aggregation=dict(row='For each test row and arm, mean correctness over the three seeds (A verdict, P program verdict)',
            difference='Mean over rows of (P row mean - A row mean)',
            bootstrap=dict(unit='image',replicates=REPLICATES,seed=SEED,reference='source clusters (4 test sources), reference only')),
        rules=dict(test_confirms='Every seed P >= 0.65 and P - A >= 0.10 on test, and the image-bootstrap 95% interval of the seed-mean P - A above 0',
            test_fails='At least one seed P <= 0.60 on test',unclear='Any other result',
            descriptive='test_natural, twin-image all-correct, point errors, P own verdict, validation-to-test change; reported, not used'),
        thresholds=dict(THRESHOLDS),evaluations_per_adapter=1,
        rerun='Only after an infrastructure failure before COMPLETE.json, with the same code and adapter, recorded beside the failed attempt',
        hyperparameter_search=False,api_calls=0)


def test_reading(p_by_seed,delta_by_seed,interval,t=THRESHOLDS):
    """Preregistered reading over the three seeds' test P (program) accuracies, their P - A, and the seed-mean P - A interval."""
    require(sorted(p_by_seed)==sorted(delta_by_seed)==sorted(str(s) for s in ALL_SEEDS),'Reading needs all three seeds')
    if min(p_by_seed.values())>=t['p_floor'] and min(delta_by_seed.values())>=t['delta'] and interval[0]>0:return 'test_confirms'
    if min(p_by_seed.values())<=t['p_fail']:return 'test_fails'
    return 'unclear'
