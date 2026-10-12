"""Fixed rank64 comparison contract; no search and no changes to old records."""
from copy import deepcopy
from pathlib import Path
from .common import read, require, sha

PROFILE = 'rank64_alpha128'
LOW_LR_PROFILE = 'rank64_alpha128_lr_quarter'
RANK64_REFERENCE_NAME = '20261012T034246Z_rank64_sft_0e0ebfacf97f'
BASELINE_NAME = '20261011T042240Z_binary_correction_706fd1409f15'
BASELINE_REVIEW_NAME = '20261011T054114Z_binary_review_77d5f6a03e8f'
BASELINE_FILES = (
    'RECEIPT.json', 'smoke_config.json', 'full_config.json',
    'preflight/COMPLETE.json', 'preflight/ENCODINGS.json',
    'smoke/RESULT.json', 'full/RESULT.json', 'full/PROVENANCE.json',
    'full/TRAINING_ORDER.json', 'full/generations_final.json',
    'full/generations_horizon_0.json', 'full/curve.jsonl',
    'full/loss_before.json', 'full/loss_step_0300.json',
    'full/adapter/adapter_config.json', 'full/adapter/adapter_model.safetensors',
)
RANK64_REFERENCE_FILES = (
    'RECEIPT.json', 'smoke_config.json', 'full_config.json',
    'preflight/COMPLETE.json', 'preflight/ENCODINGS.json',
    'smoke/RESULT.json', 'smoke/PROVENANCE.json', 'smoke/GRADIENT_PROBE.json',
    'smoke/generations_final.json', 'smoke/generations_horizon_0.json',
    'smoke/curve.jsonl', 'smoke/loss_before.json', 'smoke/loss_step_0240.json',
    'smoke/adapter/adapter_config.json', 'smoke/adapter/adapter_model.safetensors',
    'STOP_smoke.json',
)


def apply_profile(config, profile=None):
    result = deepcopy(config)
    if profile is None:
        return result
    require(profile in (PROFILE,LOW_LR_PROFILE), 'Only authorized fixed rank64 profiles are allowed')
    require((result['lora_rank'], result['lora_alpha']) == (16, 32), 'Baseline rank changed')
    result.update(lora_rank=64, lora_alpha=128)
    if profile==LOW_LR_PROFILE:
        require(result['learning_rate'] in (2e-4,1e-4),'Unexpected baseline learning rate')
        result['learning_rate']*=.25
    return result


def config_difference(baseline, candidate):
    changed = {key: {'rank16': baseline.get(key), 'rank64': candidate.get(key)}
               for key in baseline.keys() | candidate.keys()
               if baseline.get(key) != candidate.get(key)}
    require(changed == {'lora_rank': {'rank16': 16, 'rank64': 64},
                        'lora_alpha': {'rank16': 32, 'rank64': 128}},
            'Only rank and proportional alpha may differ from the frozen baseline')
    return changed


def validate_parent(receipt):
    require(receipt['comparison_profile'] in (PROFILE,LOW_LR_PROFILE), 'Wrong comparison profile')
    parent = Path(receipt['rank16_baseline'])
    require(parent.name == BASELINE_NAME, 'Wrong rank16 baseline')
    require(set(receipt['baseline_sha256']) == set(BASELINE_FILES), 'Incomplete baseline manifest')
    for name, digest in receipt['baseline_sha256'].items():
        require(sha(parent/name) == digest, 'Rank16 artifact changed: '+name)
    old = read(parent/'RECEIPT.json')
    for key in ('campaign', 'assets', 'diagnostic', 'prefix_diagnostic',
                'diagnostic_complete_sha256', 'prefix_complete_sha256'):
        require(old[key] == receipt[key], 'Baseline context changed: '+key)
    require(read(parent/'full/RESULT.json')['passed'], 'Rank16 execution incomplete')
    require(read(parent/'smoke/RESULT.json')['passed'], 'Rank16 smoke incomplete')
    require(read(parent/'full/PROVENANCE.json')['trainable_parameters'] == 29933568,
            'Unexpected baseline trainable count')
    if receipt['comparison_profile']==LOW_LR_PROFILE:
        reference=Path(receipt['rank64_reference'])
        require(reference.name==RANK64_REFERENCE_NAME,'Wrong high-LR rank64 reference')
        require(set(receipt['rank64_reference_sha256'])==set(RANK64_REFERENCE_FILES),'Incomplete rank64 manifest')
        for name,digest in receipt['rank64_reference_sha256'].items():
            require(sha(reference/name)==digest,'Rank64 reference changed: '+name)
        old64=read(reference/'RECEIPT.json')
        require(old64['comparison_profile']==PROFILE and old64['baseline_sha256']==receipt['baseline_sha256'],
                'Rank64 reference has different ancestry')
        for key in ('campaign','assets','diagnostic','prefix_diagnostic'):
            require(old64[key]==receipt[key],'Rank64 context changed: '+key)
        require(read(reference/'smoke/RESULT.json')['passed'] is False,'Reference smoke result changed')
        require(not (reference/'full').exists(),'Reference unexpectedly has a full fit')
    return parent


def lr_difference(baseline, candidate):
    require((baseline['lora_rank'],baseline['lora_alpha'])==(64,128),'Expected rank64/alpha128 reference')
    require(baseline['learning_rate'] in (2e-4,1e-4),'Unexpected reference learning rate')
    changed={key:{'original_lr':baseline.get(key),'quarter_lr':candidate.get(key)}
             for key in baseline.keys() | candidate.keys() if baseline.get(key)!=candidate.get(key)}
    require(changed=={'learning_rate':{'original_lr':baseline['learning_rate'],
                                     'quarter_lr':baseline['learning_rate']*.25}},
            'Only the predeclared quarter learning rate may change')
    return changed
