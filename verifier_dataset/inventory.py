"""Read historical evidence on a CPU allocation; never infer negative labels."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import statistics

ASSETS = Path('/projects/p33100/siosio/crashbench_safelibero')
INSTRUMENTED = '20261009T005828Z_full_7e37d51e5c93'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def identity(scene):
    return '{suite}/{level}/task{task}/episode{episode}'.format(**scene)


def candidate(result, directory):
    """An execution outcome is routing metadata, never an existence label."""
    spec = result['spec']
    scene = spec['scene']
    complete = result.get('status') == 'complete' and not result.get('exited')
    success = complete and result.get('success') is True and result.get('collision_step') is None
    tape = directory / 'steps.jsonl'
    settled = directory / 'settled_state.npz'
    return dict(source_id=identity(scene), task_group='{suite}/{level}/task{task}'.format(**scene),
                family=scene['suite'], scene=scene, method=spec['method'], repeat=spec['repeat'],
                root=str(directory), seed=result['seed'], actions=result['actions'],
                perception_variant=spec.get('perception_variant','native'),
                source_outcome=result['outcome'], infrastructure_complete=complete,
                proxy_safe_success=success, feasibility_label='unknown',
                needs_contact_replay=success, has_action_tape=tape.is_file(),
                has_settled_state=settled.is_file(),
                result_sha256=sha(directory/'RESULT.json'),
                tape_sha256=sha(tape) if tape.is_file() else None,
                settled_sha256=sha(settled) if settled.is_file() else None,
                inference_files=len(list(directory.glob('infer_*.npz'))),
                snapshot_files=len(list((directory/'snapshots').glob('*_physics.npz'))),
                source_code_commit=read(directory/'manifest.json')['code_commit'])


def smoke_selection(rows, limit=12):
    """Round-robin family/task/state coverage; fixed order, no new outcome search."""
    eligible = [r for r in rows if r['proxy_safe_success'] and r['has_action_tape'] and r['has_settled_state']]
    chosen, used = [], set()
    buckets = {}
    for row in sorted(eligible, key=lambda r: (r['source_id'], r['repeat'], r['method'])):
        buckets.setdefault(row['task_group'], []).append(row)
    while len(chosen) < limit:
        changed = False
        for task in sorted(buckets):
            while buckets[task] and buckets[task][0]['source_id'] in used:
                buckets[task].pop(0)
            if buckets[task] and len(chosen) < limit:
                row = buckets[task].pop(0); chosen.append(row); used.add(row['source_id']); changed = True
        if not changed:
            break
    return chosen


def main(root):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Inventory belongs on a Quest CPU allocation')
    out = root/'inventory'; out.mkdir(exist_ok=False)
    rows, errors = [], []
    source = ASSETS/'uncertainty'/INSTRUMENTED
    for shard in (0, 1):
        for path in sorted((source/'shards'/str(shard)/'runs').glob('*/RESULT.json')):
            try:
                rows.append(candidate(read(path), path.parent))
            except (KeyError, ValueError, OSError) as error:
                errors.append(dict(path=str(path), error=type(error).__name__, reason=str(error)))
    write(out/'CANDIDATES.json', rows)
    write(out/'REPLAY_SMOKE.json', smoke_selection(rows))
    official = ASSETS/'batches/20260927T204911Z_d22035816647/COMPLETE.json'
    report = read(official)
    costs = []
    for name in ('20260926T233314Z_af6ac874d168', '20260927T204911Z_d22035816647'):
        ledger = read(ASSETS/'batches'/name/'budget.json')
        costs.extend(float(Decimal(v['cost_usd'])) for v in ledger['calls'].values() if v.get('status') == 'settled')
    successes = [r for r in rows if r['proxy_safe_success']]
    summary = dict(passed=len(rows) == 1200 and not errors, inventory_only=True, accepted_dataset_rows=0,
        total_rollouts=len(rows), proxy_safe_successes=len(successes),
        original_states=len({r['source_id'] for r in rows}),
        successful_original_states=len({r['source_id'] for r in successes}),
        families=dict(Counter(r['family'] for r in rows)),
        successful_by_task=dict(Counter(r['task_group'] for r in successes)),
        successful_by_method=dict(Counter(r['method'] for r in successes)),
        contact_replay_ready=sum(r['has_action_tape'] and r['has_settled_state'] for r in successes),
        official_batch=dict(path=str(official), sha256=sha(official), episodes=report['completed_episodes'],
                            nominal=report['groups']['nominal'], aegis=report['groups']['aegis']),
        historical_api_cost=dict(settled_attempts=len(costs), mean_usd=statistics.mean(costs),
                                 max_usd=max(costs), total_of_these_ledgers_usd=sum(costs),
                                 caveat='Historical perception only; not a current price or rationale-generation estimate'),
        new_api_calls=0, campaign_api_limit_usd='20.00', errors=errors,
        code_commit=(root/'SOURCE_COMMIT').read_text().strip(), slurm_job=os.environ['SLURM_JOB_ID'], seed=1014)
    write(out/'SUMMARY.json', summary)
    write(out/'HASHES.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(summary, indent=2), flush=True)
    if not summary['passed']:
        raise RuntimeError('Historical inventory completeness failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    main(p.parse_args().root)
