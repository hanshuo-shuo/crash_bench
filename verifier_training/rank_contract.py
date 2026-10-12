"""Fixed rank64 comparison contract; no search and no changes to old records."""
from copy import deepcopy
from pathlib import Path
from .common import read, require, sha

PROFILE = 'rank64_alpha128'
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


def apply_profile(config, profile=None):
    result = deepcopy(config)
    if profile is None:
        return result
    require(profile == PROFILE, 'Only the authorized rank64 comparison is allowed')
    require((result['lora_rank'], result['lora_alpha']) == (16, 32), 'Baseline rank changed')
    result.update(lora_rank=64, lora_alpha=128)
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
    require(receipt['comparison_profile'] == PROFILE, 'Wrong comparison profile')
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
    return parent
