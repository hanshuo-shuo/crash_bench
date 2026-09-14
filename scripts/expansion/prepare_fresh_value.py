"""Build a metadata-only, pre-execution contract; no outcome analysis or simulator."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'docs/audits/20260913/fresh_value/contract.json'


def main():
    base = json.loads((ROOT/'configs/detour_benefit/development_v1.json').read_text())
    inputs = [
        'configs/detour_benefit/development_v1.json',
        'results/expansion/governance/exposure_registry.json',
        'results/expansion/governance/exposure_attempts.jsonl',
        'results/expansion/governance/split_manifest_v1_1.json',
        'configs/expansion/source_sampling_v1.yaml',
        'docs/audits/20260909/detour_benefit_gate/evidence/freeze.json',
        'docs/audits/20260909/detour_benefit_gate/evidence/checkpoint_hashes.json',
        base['risk_path'], str(Path(base['risk_path']).with_suffix('.npz')),
        'crashbench/recovery.py', 'crashbench/detour_benefit.py',
        'crashbench/repeat_value.py', 'crashbench/fresh_value.py',
        'scripts/expansion/run_detour_benefit.py',
        'scripts/expansion/run_fresh_value.py',
        'scripts/expansion/analyze_fresh_value.py',
        'scripts/prepare_glass_recovery_placements.py',
    ]
    for rel in ('screen', 'formal_nominal'):
        p = json.loads((ROOT/'configs/expansion/source_sampling_v1.yaml').read_text())[rel]['plan']
        inputs.append(p)
    # Only identity fields from the existing registry, attempts and split manifest.
    # In particular, no D8 outcome paths are opened.
    excluded = {k: set() for k in ('source_state_sha256', 'reset_seed', 'scene_fingerprint')}
    def identities(value):
        if isinstance(value, list):
            for v in value: identities(v)
        elif isinstance(value, dict):
            for k, v in value.items():
                if k in excluded and v is not None and not isinstance(v, (dict, list)):
                    excluded[k].add(str(v))
                elif k == 'identifier_type' and v in excluded:
                    excluded[v].add(str(value['value']))
                elif k.startswith('observed_') and k[9:].removesuffix('s') in excluded:
                    excluded[k[9:].removesuffix('s')].update(map(str, v))
                elif isinstance(v, (dict, list)):
                    identities(v)
    for rel in inputs:
        if rel.endswith('.json') and ('governance/' in rel or 'sources/' in rel):
            identities(json.loads((ROOT/rel).read_text()))
        elif rel.endswith('.jsonl'):
            for line in (ROOT/rel).read_text().splitlines():
                if line.strip(): identities(json.loads(line))
    excluded['source_state_sha256'].update(r['source'] for r in base['panel'])
    config = {k: base[k] for k in ('seed', 'H', 'long_H', 'settle_steps',
        'candidate_last_action', 'risk_path', 'risk_threshold', 'checkpoint',
        'checkpoint_revision', 'detour')}
    config['policy_seed_override'] = {str(i): None for i in range(6)}
    attempts = [dict(source_id=f's{i:02d}', reset_seed=2027091400+i,
        requested_fraction=[.48, .54, .60, .64][i % 4],
        offpath_side=1 if i % 2 == 0 else -1) for i in range(12)]
    if any(str(a['reset_seed']) in excluded['reset_seed'] for a in attempts):
        raise ValueError('reset seed already exposed')
    payload = dict(schema_version=1, kind='fresh_reset_repeat_value_development',
        authorized_request='2026-09-13 user: continue along Pro direction and this round findings',
        status='FROZEN_BEFORE_NEW_SOURCES_AND_OUTCOMES', source_attempts=attempts,
        suite='libero_spatial', task_id=0, source_count=12,
        conditions=['glass', 'offpath', 'noglass'], max_prefixes=36,
        max_authoring_rollouts=12, authoring_horizon=220, max_branches=432,
        phase_repeats={'A':[0,1], 'B':[2,3], 'C':[4,5]}, config=config,
        geometry=dict(size=[.028,.060], density=400., min_target_clearance=.12,
            bowl_rest_offset=.005, offpath_offset=.20,
            recipe='one nominal EEF arclength placement; backward clearance clamp; no hazard or recovery-success screen'),
        source_population='Fresh seeded LIBERO task-0 resets; no nominal-success filtering; all attempts retained; not the historical success/hazard-screened population',
        invalid_source_action='record authoring failure and stop before any paired branch; no replacement or implicit rerun',
        primary='real_full, A selection at 440, A/B/C evaluation at absolute episode 440',
        secondary=['absolute episode 220', 'real_one', 'pseudo_one', 'pseudo_one_swapped',
                   'fixed Base/AlwaysDetour/RiskDetour/BenefitGate', 'all condition and source contributions'],
        practical_difference_pp=5, bootstrap=dict(draws=5000, seed=20270913, unit='physical_source'),
        isolation='freeze A before B process starts; freeze A/B forecasts before C process starts',
        execution='fresh process per A/B/C; exact bundle and RNG restore; no added sampling; greedy policy stays greedy',
        order='A ascending, B descending, C rotate by 12 episodes; options alternate by episode index plus repeat',
        no_new_confirmation_or_d8=True, no_benchmark_expansion=True,
        excluded_identities={k: sorted(v) for k, v in excluded.items()},
        input_sha256={rel: hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in inputs})
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with DEST.open('x') as f: json.dump(payload, f, indent=2, sort_keys=True); f.write('\n')
    print('Frozen 12 reset attempts, 36 episodes, at most 432 paired branches')


if __name__ == '__main__': main()
