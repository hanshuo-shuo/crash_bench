"""V5 return HIL summary; the GPU runs are v5_hil_run.py shards driven by the plan's three conditions.

Protocol: docs/generative_verifier/V5_RETURN_PROTOCOL.md; contract in v5_return.py.
ROOT summarizes a finished return batch: each state's starting position comes from its runs' ROW.json,
and the C0 replication from the HIL batch the plan was copied from.
--natural HIL_ROOT OUT_JSON checks the metrics on the HIL C0 records (section 2). HIL recorded no
starting position, so a state takes its source's end-effector position at a t = 0 HIL state where one
exists, else the mean of those positions; the check also reports the widest 10-step dwell gap.
"""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
from .inventory import read, sha, write
from . import v5_hil as H
from . import v5_return as R
from .v5_hil_run import load


def summarize(root):
    campaign = read(root/'CAMPAIGN.json'); plan = read(root/'PLAN.json'); states = plan['states']; hil = Path(plan['hil_root'])
    if tuple(plan['conditions']) != R.CONDITIONS:
        raise RuntimeError('Not a return plan')
    if sha(hil/'PLAN.json') != plan['hil_plan_sha256']:
        raise RuntimeError('HIL plan changed')
    for k in range(campaign['shards']):
        if not read(root/('SHARD_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Shard %d incomplete' % k)
    m, divergences, replication, runs = {}, {}, {}, []
    for index, spec in enumerate(states):
        if not read(root/('state_%03d' % index)/'COMPLETE.json')['passed']:
            raise RuntimeError('State %d incomplete' % index)
        if read(hil/('state_%03d' % index)/'STATE.json')['image'] != spec['image']:
            raise RuntimeError('HIL state differs: %d' % index)
        loaded = {c: load(root, index, c) for c in R.CONDITIONS}; start = loaded['C0'][0]['start_eef']
        _, earlier = load(hil, index, 'C0')
        m[spec['state']] = {c: R.metrics(trace, start) for c, (_, trace) in loaded.items()}
        divergences[spec['state']] = dict({c: H.divergence(loaded['C0'][1], loaded[c][1]) for c in R.NEW},
                                          hil_C0=H.divergence(earlier, loaded['C0'][1]))
        replication[spec['state']] = m[spec['state']]['C0']['completed'] == bool(earlier[-1]['done'])
        runs += [dict(state=spec['state'], condition=c, base_source=spec['base_source'], split=spec['split'], time_bin=spec['time_bin'],
                      phase=spec['phase'], checkpoint=spec['checkpoint'], start_eef=start, elapsed_seconds=row['elapsed_seconds'],
                      policy_requests=row['policy_requests'], steps=row['outcome']['steps'], replayed_outputs=row['replayed_outputs'],
                      maximum_native_difference=row['maximum_native_difference'], actions_outside_spec=row['actions_outside_spec'],
                      **m[spec['state']][c]) for c, (row, _) in loaded.items()]
    with gzip.open(root/'RUNS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in runs)
    reading = R.readouts(states, m, divergences, replication)
    reading.update(mode=campaign['mode'], seconds_per_run=H.distribution(r['elapsed_seconds'] for r in runs),
                   total_run_hours=sum(r['elapsed_seconds'] for r in runs) / 3600., prompt_tokens=plan['prompt_tokens'],
                   replayed_outputs=sum(r['replayed_outputs'] for r in runs), actions_outside_spec=sum(r['actions_outside_spec'] for r in runs),
                   maximum_native_difference=max(r['maximum_native_difference'] for r in runs),
                   hil_C0_divergence=dict(H.distribution(divergences[s['state']]['hil_C0'] for s in states),
                                          never=sum(divergences[s['state']]['hil_C0'] is None for s in states)))
    write(root/'RETURN_READING.json', reading)
    write(root/'RETURN_COMPLETE.json', dict(passed=True, mode=campaign['mode'], readings=reading['readings'], heads_home=reading['heads_home'],
                                            states=len(states), runs=len(runs), api_calls=0, code_commit=os.environ.get('CB_CODE_COMMIT'),
                                            slurm_job=os.environ.get('SLURM_JOB_ID')))
    write(root/'HASHES.json', {n: sha(root/n) for n in ('PLAN.json', 'RUNS.jsonl.gz', 'RETURN_READING.json', 'RETURN_COMPLETE.json')})


def dwell_gap(trace, start):
    """Smallest, over every DWELL consecutive steps after t, of the largest distance to the start."""
    d = R.gaps(trace, start)[1:]
    return min(max(d[i:i + R.DWELL]) for i in range(len(d) - R.DWELL + 1)) if len(d) >= R.DWELL else None


def natural(hil):
    """Section 2 on the HIL C0 records: the main analysis set and how close C0 comes to the start."""
    plan = read(hil/'PLAN.json'); traces = {}
    for index, s in enumerate(plan['states']):
        with gzip.open(hil/('state_%03d' % index)/'C0/STEPS.jsonl.gz', 'rt') as stream:
            traces[s['state']] = [json.loads(x) for x in stream]
    known = {}
    for s in plan['states']:
        if s['checkpoint'] == 0:
            known.setdefault(s['base_source'], traces[s['state']][0]['eef'])
    mean = [sum(v[i] for v in known.values()) / len(known) for i in range(3)]
    m, rows = {}, []
    for s in plan['states']:
        start = known.get(s['base_source'], mean); trace = traces[s['state']]; x = R.metrics(trace, start)
        m[s['state']] = dict(C0=x)
        rows.append(dict(state=s['state'], base_source=s['base_source'], split=s['split'], checkpoint=s['checkpoint'], phase=s['phase'],
                         exact_start=s['base_source'] in known, dwell_gap=dwell_gap(trace, start), **x))
    main = {s['state'] for s in R.analysis(plan['states'], m)}; inside = [r for r in rows if r['state'] in main]
    return dict(hil_root=str(hil), states=len(rows), sources_with_exact_start=len(known),
                start_spread=max(math.dist(v, mean) for v in known.values()), mean_start=mean,
                analysis_states=len(inside), analysis_by_split={k: sum(r['split'] == k for r in inside) for k in H.SPLITS},
                analysis_sources=len({r['base_source'] for r in inside}), excluded_checkpoints=sorted(r['checkpoint'] for r in rows if r['state'] not in main),
                home=sum(r['home'] for r in inside), completed=sum(r['completed'] for r in inside), stuck=sum(r['stuck'] for r in inside),
                closest=H.distribution(r['closest'] for r in inside), final_gap=H.distribution(r['final_gap'] for r in inside),
                dwell_gap=H.distribution(r['dwell_gap'] for r in inside),
                dwell_gap_exact_start=H.distribution(r['dwell_gap'] for r in inside if r['exact_start']), rows=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    p.add_argument('--natural', type=Path, metavar='OUT_JSON')
    a = p.parse_args()
    if a.natural:
        out = natural(a.root); a.natural.write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
    else:
        try:
            summarize(a.root)
        except BaseException as error:
            write(a.root/'STOP_summarize.json', dict(error=type(error).__name__, reason=str(error)))
            raise
