"""CPU-only gate: admit exactly the frozen formal plan after a complete, same-code smoke."""
import argparse
import os
from pathlib import Path
from .inventory import ASSETS, read, sha, write
from . import language_ablation as A


def validate(smoke, formal):
    parent = ASSETS/'language_ablation'
    if smoke == formal or any(parent not in p.parents for p in (smoke, formal)):
        raise RuntimeError('Gate roots must be distinct language ablation runs')
    if list(smoke.glob('STOP_*.json')) or list(formal.glob('STOP_*.json')):
        raise RuntimeError('A pipeline stage failed')
    hashes = read(smoke/'HASHES.json')
    if not {'PLAN.json', 'SUMMARY.json', 'COMPLETE.json', 'RUNS.jsonl.gz'} <= set(hashes):
        raise RuntimeError('Incomplete smoke hashes')
    for name, digest in hashes.items():
        if sha(smoke/name) != digest:
            raise RuntimeError('Smoke output changed: ' + name)
    complete = read(smoke/'COMPLETE.json'); sm = read(smoke/'CAMPAIGN.json'); fm = read(formal/'CAMPAIGN.json')
    if (not complete['passed'] or complete['mode'] != 'smoke' or complete['runs'] != 16 or complete['states'] != 4
            or sm['policy'] != 'pi05_libero' or fm['policy'] != 'pi05_libero'
            or any(sm[k] != fm[k] for k in ('code_commit', 'protocol_sha256', 'upstream_commit', 'libero_commit', 'policy_checkpoint'))):
        raise RuntimeError('Smoke is not a complete same-code same-policy integrity check')
    names = read(formal/'NAMES.json')
    if (read(smoke/'NAMES.json') != names or read(smoke/'PLAN.json')['mode'] != 'smoke'
            or read(formal/'PLAN.json')['mode'] != 'formal'
            or read(smoke/'PLAN.json')['units'] != A.plan(names, 'smoke')
            or read(formal/'PLAN.json')['units'] != A.plan(names, 'formal')
            or read(formal/'PLAN.json')['smoke_root'] != str(smoke)):
        raise RuntimeError('Plan differs from the frozen contract')
    write(formal/'GATE.json', dict(passed=True, smoke_root=str(smoke), smoke_hashes=hashes,
                                  formal_plan_sha256=sha(formal/'PLAN.json'), code_commit=fm['code_commit'], slurm_job=os.environ.get('SLURM_JOB_ID')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('smoke', type=Path); p.add_argument('formal', type=Path); args = p.parse_args()
    try:
        validate(args.smoke, args.formal)
    except BaseException as error:
        write(args.formal/'STOP_gate.json', dict(reason=str(error)))
        raise
