"""V5 detour HIL summary; the GPU runs are v5_hil_run.py shards driven by the plan's three conditions.

Protocol: docs/generative_verifier/V5_DETOUR_PROTOCOL.md; contract in v5_detour.py. ROOT summarizes a
finished detour batch: per state, the C0, C1 and D outcomes recomputed from the step traces, the
twin divergences C0-C1 and C1-D, and the readouts and reading of section 8.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
from .inventory import read, sha, write
from . import v5_detour as D
from . import v5_hil as H
from .v5_hil_run import load


def summarize(root):
    campaign = read(root/'CAMPAIGN.json'); plan = read(root/'PLAN.json'); states = plan['states']
    if tuple(plan['conditions']) != D.CONDITIONS:
        raise RuntimeError('Not a detour plan')
    for k in range(campaign['shards']):
        if not read(root/('SHARD_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Shard %d incomplete' % k)
    outcomes, divergences, runs = {}, {}, []
    for index, spec in enumerate(states):
        if not read(root/('state_%03d' % index)/'COMPLETE.json')['passed']:
            raise RuntimeError('State %d incomplete' % index)
        loaded = {c: load(root, index, c, D.outcome) for c in D.CONDITIONS}
        outcomes[spec['state']] = {c: row['outcome'] for c, (row, _) in loaded.items()}
        divergences[spec['state']] = dict(C0_C1=H.divergence(loaded['C0'][1], loaded['C1'][1]),
                                          C1_D=H.divergence(loaded['C1'][1], loaded['D'][1]))
        runs += [dict(state=spec['state'], condition=c, base_source=spec['base_source'], split=spec['split'], anchor=spec['anchor'],
                      layout=spec['layout'], time_bin=spec['time_bin'], checkpoint=spec['checkpoint'], elapsed_seconds=row['elapsed_seconds'],
                      policy_requests=row['policy_requests'], replayed_outputs=row['replayed_outputs'],
                      maximum_native_difference=row['maximum_native_difference'], actions_outside_spec=row['actions_outside_spec'],
                      **{k: v for k, v in row['outcome'].items() if k != 'entry'}) for c, (row, _) in loaded.items()]
    with gzip.open(root/'RUNS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in runs)
    reading = D.readouts(states, outcomes, divergences)
    reading.update(mode=campaign['mode'], seconds_per_run=H.distribution(r['elapsed_seconds'] for r in runs),
                   total_run_hours=sum(r['elapsed_seconds'] for r in runs) / 3600., prompt_tokens=plan['prompt_tokens'],
                   replayed_outputs=sum(r['replayed_outputs'] for r in runs), actions_outside_spec=sum(r['actions_outside_spec'] for r in runs),
                   maximum_native_difference=max(r['maximum_native_difference'] for r in runs))
    write(root/'DETOUR_READING.json', reading)
    write(root/'DETOUR_COMPLETE.json', dict(passed=True, mode=campaign['mode'], reading=reading['reading'], veers=reading['veers'],
                                            states=len(states), runs=len(runs), api_calls=0, code_commit=os.environ.get('CB_CODE_COMMIT'),
                                            slurm_job=os.environ.get('SLURM_JOB_ID')))
    write(root/'HASHES.json', {n: sha(root/n) for n in ('PLAN.json', 'RUNS.jsonl.gz', 'DETOUR_READING.json', 'DETOUR_COMPLETE.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    a = p.parse_args()
    try:
        summarize(a.root)
    except BaseException as error:
        write(a.root/'STOP_summarize.json', dict(error=type(error).__name__, reason=str(error)))
        raise
