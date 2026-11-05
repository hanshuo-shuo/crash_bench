"""P4 closed loop (docs/generative_verifier/HIL_MECHANISM_NEURONS.md 4.2, 5, 6): plan and summary.

Reruns of HIL suite stage 1 switch branches (the closed-loop test's cells) under serve_steer.py's arms: native, amp2 (keys and
values, as the closed-loop test) and the neuron arms (kind nrn) on each pair's set S of k* neurons from the formal open-loop
batch, its matched random set R, a random direction on S, and every hook-layer unit.

  neuron_loop.py plan BRANCH_ROOT NEURON_ROOT OUT_DIR [--smoke] [--shards N]   PLAN.json and NEURON_SETS.json (read-only inputs)
  neuron_loop.py summarize ROOT [--require-tool-ok]                           tool check, N3-N5, tables -> ROOT/SUMMARY.json
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import closed_loop as C  # noqa: E402
import neuron_readings as NR  # noqa: E402

LAYERS = C.LAYERS                     # amp2's key/value window
FAILURE, SUCCESS, START = C.FAILURE, C.SUCCESS, C.START
ARMS = dict(native=('native', 1., None, None), amp2=('amp', 2., None, None),
            nrn2=('nrn', 2., 'top', 'own'), nrnrand2=('nrn', 2., 'random', 'own'), nrndir2=('nrn', 2., 'top', 'random'),
            mlpall2=('nrn', 2., 'all', 'own'),
            nrnout=('nrn', 0., 'top', 'own'), nrnoutrand=('nrn', 0., 'random', 'own'), mlpallout=('nrn', 0., 'all', 'own'))
CELL_ARMS = dict(failure=('native', 'amp2', 'nrn2', 'nrnrand2', 'nrndir2', 'mlpall2'),
                 success=('native', 'nrn2', 'nrnout', 'nrnoutrand', 'mlpallout'),
                 start=('native', 'nrn2', 'nrnout', 'nrnoutrand', 'mlpallout'))
SMOKE_INITS, FORMAL_INITS = C.SMOKE_INITS, C.FORMAL_INITS
GAIN, HARM = C.GAIN, C.HARM            # .15 and .10, as R4 and R5
SCALE_TOLERANCE = C.SCALE_TOLERANCE    # 1e-3 relative


def direction_seed(sw):
    return int(hashlib.sha256(('mechanism-neurons-closed:' + sw).encode()).hexdigest()[:8], 16)


def neuron_sets(neuron_root):
    """Each pair's set S (the first k* of its formal ranking, k* from the formal summary's N1, or 10,000) and R (same per-layer
    counts, other units, its own seed); refuses a root whose summary is not tool_ok."""
    neuron_root = Path(neuron_root); summary = C.read(neuron_root/'SUMMARY.json')
    if summary['tool'] != 'tool_ok' or summary['mode'] != 'formal':
        raise RuntimeError('The neuron sets need the formal open-loop root with tool_ok')
    if C.sha(neuron_root/'NEURONS.json') != C.read(neuron_root/'NEURONS.sha256.json')['sha256']:
        raise RuntimeError('NEURONS.json changed')
    neurons = C.read(neuron_root/'NEURONS.json'); sets = {}
    for pair, info in neurons['pairs'].items():
        k = summary['k_used'][pair]; top = [tuple(x[:2]) for x in info['top']]
        sets[pair] = dict(k=k, top=[list(x) for x in top[:k]],
                          random=[list(x) for x in NR.random_set(info['top'], k, pair, neurons['width'], salt='neurons-closed-random')])
    return dict(sets=sets, hooks=neurons['hooks'], width=neurons['width'], neuron_root=str(neuron_root),
                neurons_sha256=C.sha(neuron_root/'NEURONS.json'), summary_sha256=C.sha(neuron_root/'SUMMARY.json'),
                readings=summary['readings'], k_used=summary['k_used'])


def units(branch_plan, runs, inits, hooks):
    done = {r['id']: bool(r['b_done']) for r in runs}
    prompts = {p['id']: p['prompt'] for p in branch_plan['prefix_order']}
    out = []
    for u in branch_plan['branches']:
        if u['group'] != '1a' or u['condition'] != 'SW' or u['init'] not in inits:
            continue
        pair = '%d->%d' % (u['task'], u['other']); cell = C.cell_of(pair, u['phase'])
        if cell is None:
            continue
        if prompts[u['prefix']] == u['prompt']:
            raise ValueError('B equals A for %s' % u['id'])
        for name in CELL_ARMS[cell]:
            kind, lam, neurons, direction = ARMS[name]
            arm = dict(name=name, kind=kind, lam=lam, layers=list(hooks) if kind == 'nrn' else list(LAYERS), prompt_a=prompts[u['prefix']],
                       direction_seed=direction_seed(u['id']))
            if kind == 'nrn':
                arm.update(neuron_set=pair, neurons=neurons, direction=direction)
            out.append(dict(u, id='%s__%s' % (u['id'], name), sw=u['id'], cell=cell, pair=pair, arm=arm, stage1_b_done=done[u['id']]))
    return out


def plan(branch_root, sets, mode, shards):
    branch_root = Path(branch_root); branch_plan = C.read(branch_root/'PLAN.json')
    with gzip.open(branch_root/'RUNS.jsonl.gz', 'rt') as stream:
        runs = [json.loads(x) for x in stream]
    chosen = units(branch_plan, runs, SMOKE_INITS if mode == 'smoke' else FORMAL_INITS, sets['hooks'])
    used = {u['prefix'] for u in chosen}
    order = sorted((p for p in branch_plan['prefix_order'] if p['id'] in used), key=lambda p: (p['task'], p['init']))
    rank = {p['id']: k for k, p in enumerate(order)}; arms = list(ARMS)
    chosen.sort(key=lambda u: (rank[u['prefix']], u['phase'], arms.index(u['arm']['name'])))
    return dict(mode=mode, experiment='mechanism-neurons-closed', shards=shards, prefix_root=branch_plan['prefix_root'],
                prefix_hashes=branch_plan['prefix_hashes'], prefix_order=order, branches=chosen, excluded=[],
                stage1_root=str(branch_root), stage1_hashes_sha256=C.sha(branch_root/'HASHES.json'), layers=LAYERS,
                arms={k: list(v) for k, v in ARMS.items()}, cell_arms={k: list(v) for k, v in CELL_ARMS.items()},
                k_used=sets['k_used'], neuron_root=sets['neuron_root'], inits=list(SMOKE_INITS if mode == 'smoke' else FORMAL_INITS))


# ---------------------------------------------------------------- readings (pure)
def sufficiency(runs, inits):
    """N3 on the failure cells."""
    gain = C.paired(runs, 'nrn2', 'native', FAILURE, inits)
    specs = [C.paired(runs, 'nrn2', y, FAILURE, inits) for y in ('nrnrand2', 'nrndir2')]
    if gain is None or any(s is None for s in specs):
        reading = 'insufficient'
    elif gain['lo'] > 0:
        reading = ('restores_switch' if gain['mean'] >= GAIN else 'weakly_restores') if all(s['lo'] > 0 for s in specs) else 'nonspecific'
    elif gain['hi'] < 0:
        reading = 'harms'
    else:
        reading = 'no_restore'
    return dict(reading=reading, gain=gain, specificity_set=specs[0], specificity_direction=specs[1])


def selectivity(runs, inits):
    """N4 on the success and start cells (R5's rule)."""
    d = C.paired(runs, 'nrn2', 'native', SUCCESS + START, inits)
    if d is None:
        reading = 'insufficient'
    elif d['lo'] >= -HARM:
        reading = 'selective'
    elif d['hi'] < 0 and d['mean'] <= -HARM:
        reading = 'damages'
    else:
        reading = 'unclear'
    return dict(reading=reading, change=d)


def necessity(runs, inits):
    """N5 on the success and start cells: native vs S set back to A's values, and vs the matched random set set back."""
    drop = C.paired(runs, 'native', 'nrnout', SUCCESS + START, inits); spec = C.paired(runs, 'nrnoutrand', 'nrnout', SUCCESS + START, inits)
    if drop is None or spec is None:
        reading = 'insufficient'
    elif drop['lo'] > 0:
        reading = ('necessary' if drop['mean'] >= GAIN else 'weakly_necessary') if spec['lo'] > 0 else 'nonspecific'
    else:
        reading = 'not_necessary'
    return dict(reading=reading, drop=drop, specificity=spec)


def tables(runs, inits):
    out = {}
    for key in sorted({(r['pair'], r['phase']) for r in runs}):
        cell = dict(cls=C.cell_of(*key))
        for arm in ARMS:
            rs = [r for r in runs if (r['pair'], r['phase']) == key and r['arm'] == arm and r['init'] in inits]
            if rs:
                failed = [r for r in rs if not r['b_done']]; after = [r['b_after'] for r in rs if r['b_done']]
                cell[arm] = dict(n=len(rs), b_done=sum(r['b_done'] for r in rs), a_done_when_failed=sum(r['a_done'] for r in failed),
                                 b_after_median=float(np.median(after)) if after else None,
                                 first_shift_median=float(np.median([r['first_shift_linf'] for r in rs])))
        out['%s f=%s' % key] = cell
    return out


def readings(runs, inits):
    gain = lambda x: C.paired(runs, x, 'native', FAILURE, inits)
    n3 = sufficiency(runs, inits); amp, mlp = gain('amp2'), gain('mlpall2')
    share = lambda g, ref: g['mean'] / ref['mean'] if g and ref and ref['mean'] > 0 else None
    return dict(N3=n3, N4=selectivity(runs, inits), N5=necessity(runs, inits),
                reported=dict(amp2_vs_native=amp, mlpall2_vs_native=mlp, nrn2_share_of_amp2=share(n3['gain'], amp),
                              nrn2_share_of_mlpall2=share(n3['gain'], mlp),
                              native_vs_mlpallout=C.paired(runs, 'native', 'mlpallout', SUCCESS + START, inits),
                              nrn2_vs_native_success=C.paired(runs, 'nrn2', 'native', SUCCESS, inits),
                              nrn2_vs_native_start=C.paired(runs, 'nrn2', 'native', START, inits)),
                tables=tables(runs, inits))


def tool_reading(runs):
    """Closed-loop tool check (section 5)."""
    fidelity = [r['fidelity_linf'] for r in runs]
    detail = dict(runs=len(runs), physics_equal=sum(r['physics_equal'] for r in runs),
                  fidelity_median=float(np.median(fidelity)), fidelity_max=float(np.max(fidelity)),
                  native_moved=sum(r['max_delta'] > 0 or r['first_shift_linf'] > 0 for r in runs if r['kind'] == 'native'),
                  scale_error_max=max([r['scale_error'] for r in runs if r['kind'] != 'native'], default=0.),
                  steered_unmoved=sum(r['first_shift_linf'] == 0 or r['max_delta'] == 0 for r in runs if r['kind'] != 'native'),
                  prompt_errors=sum(not r['prompt_ok'] for r in runs))
    ok = (detail['physics_equal'] == len(runs) and detail['fidelity_median'] <= C.FIDELITY_MEDIAN and detail['fidelity_max'] <= C.FIDELITY_MAX
          and detail['native_moved'] == 0 and detail['scale_error_max'] <= SCALE_TOLERANCE and detail['steered_unmoved'] == 0
          and detail['prompt_errors'] == 0)
    return ('tool_ok' if ok else 'tool_failed'), detail


# ---------------------------------------------------------------- summary (CPU)
def scale_error(log, kind, lam):
    """Largest relative mismatch of the logged change: keys/values against (lam - 1)|B - A| (amp), or each layer's neuron
    change against its target (nrn; layers with a zero target must not move)."""
    if kind == 'amp':
        ratio = lambda e, n: e['delta_' + n] / ((lam - 1) * e['diff_' + n]) if e['diff_' + n] > 0 else (1. if e['delta_' + n] == 0 else np.inf)
        return max(abs(ratio(e, n) - 1) for e in log for n in 'kv')
    if kind == 'nrn':
        out = 0.
        for e in log:
            for t, d in zip(e['target'], e['delta']):
                out = max(out, abs(d - t) / t if t > 0 else (0. if d == 0 else np.inf))
        return out
    return 0.


def run_row(root, u, stage1_root):
    from verifier_dataset import hil_suite as S
    from verifier_dataset import hil_suite_run as H
    row, trace = H.load(Path(root)/'branch'/u['id'])
    if not C.same_outcome(S.outcome(trace, u, row['start_eef']), row['outcome']):
        raise RuntimeError('Recomputed outcome differs: ' + u['id'])
    if row['rng_reset'].get('arm') != u['arm']:
        raise RuntimeError('Arm not acknowledged: ' + u['id'])
    with open(Path(root)/'branch'/u['id']/'POLICY.jsonl') as stream:
        client = [json.loads(x) for x in stream]
    log = C.server_log(root, row['rng_reset']['run_id'], client)
    s1 = C.read(Path(stage1_root)/'branch'/u['sw']/'ROW.json')
    kind = u['arm']['kind']; o = row['outcome']
    moved = (max(max(e['delta_k'], e['delta_v']) for e in log) if kind != 'nrn' else max(max(e['delta']) for e in log))
    return dict(id=u['id'], sw=u['sw'], cell=u['cell'], pair=u['pair'], phase=u['phase'], init=u['init'], arm=u['arm']['name'], kind=kind,
                b_done=o['b_done'], a_done=o['a_done'], b_after=o['b_after'], steps=o['steps'], stage1_b_done=u['stage1_b_done'],
                physics_equal=row['physics_at_checkpoint'] == s1['physics_at_checkpoint'],
                first_input_equal=row['first_policy_input_sha256'] == s1['first_policy_input_sha256'],
                prompt_ok=row['prompt'] == u['prompt'], fidelity_linf=log[0]['fidelity_linf'], first_shift_linf=log[0]['first_shift_linf'],
                max_delta=moved, scale_error=scale_error(log, kind, u['arm']['lam']),
                neuron_change_first=float(np.sqrt(np.sum(np.square(log[0]['delta'])))) if kind == 'nrn' else None,
                written_error=max((abs(w - d) / d for e in log for w, d in zip(e['written'], e['delta']) if d > 0), default=0.) if kind == 'nrn' else None,
                requests=len(log), seconds_per_request=float(np.median([e['seconds'] for e in log[1:]])) if len(log) > 1 else None,
                actions_outside_spec=row['actions_outside_spec'], elapsed_seconds=row['elapsed_seconds'])


def summarize(root, require_tool_ok=False):
    root = Path(root); plan_ = C.read(root/'PLAN.json')
    for k in range(plan_['shards']):
        if not C.read(root/('SHARD_branch_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Shard %d incomplete' % k)
    if C.sha(Path(plan_['stage1_root'])/'HASHES.json') != plan_['stage1_hashes_sha256']:
        raise RuntimeError('Stage 1 run changed')
    runs = [run_row(root, u, plan_['stage1_root']) for u in plan_['branches']]
    tool, detail = tool_reading(runs)
    inits = tuple(plan_['inits'])
    native = [r for r in runs if r['arm'] == 'native']
    logs = sorted((root/'server').glob('*.jsonl'))
    per_arm = defaultdict(list)
    for r in runs:
        per_arm[r['arm']].append(r)
    summary = dict(
        mode=plan_['mode'], tool=tool, tool_detail=detail, runs=len(runs), k_used=plan_['k_used'],
        readings=readings(runs, inits) if tool == 'tool_ok' else None, reading_inits=list(inits), descriptive_only=plan_['mode'] == 'smoke',
        native_vs_stage1=dict(runs=len(native), same_b_done=sum(r['b_done'] == r['stage1_b_done'] for r in native),
                              stage1_b_done=sum(r['stage1_b_done'] for r in native), native_b_done=sum(r['b_done'] for r in native)),
        first_shift_median={a: float(np.median([r['first_shift_linf'] for r in rs])) for a, rs in per_arm.items()},
        neuron_change_first_median={a: float(np.median([r['neuron_change_first'] for r in rs])) for a, rs in per_arm.items() if rs[0]['kind'] == 'nrn'},
        written_error_max=max([r['written_error'] for r in runs if r['kind'] == 'nrn'], default=None),
        first_input_equal=sum(r['first_input_equal'] for r in runs),
        seconds_per_run=float(np.median([r['elapsed_seconds'] for r in runs])),
        seconds_per_request={a: float(np.median([r['seconds_per_request'] for r in rs if r['seconds_per_request'] is not None])) for a, rs in per_arm.items()},
        outside_spec_fraction={a: sum(r['actions_outside_spec'] for r in rs) / max(1, sum(r['steps'] for r in rs)) for a, rs in per_arm.items()},
        total_run_hours=sum(r['elapsed_seconds'] for r in runs) / 3600.,
        server_logs_sha256=hashlib.sha256(''.join(p.name + C.sha(p) for p in logs).encode()).hexdigest(), server_logs=len(logs),
        api_calls=0, code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID'))
    with gzip.open(root/'RUNS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in runs)
    C.write(root/'SUMMARY.json', summary)
    C.write(root/'HASHES.json', {n: C.sha(root/n) for n in ('PLAN.json', 'NEURON_SETS.json', 'RUNS.jsonl.gz', 'SUMMARY.json')})
    print(json.dumps({k: summary[k] for k in ('mode', 'tool', 'tool_detail', 'native_vs_stage1', 'seconds_per_run')}, indent=1))
    if summary['readings']:
        r = summary['readings']
        print(json.dumps(dict(N3=r['N3']['reading'], N4=r['N4']['reading'], N5=r['N5']['reading'])))
    if require_tool_ok and (tool != 'tool_ok' or plan_['mode'] != 'smoke'):
        raise SystemExit('Gate: the smoke tool check is %s; the formal batch does not start' % tool)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['plan', 'summarize']); parser.add_argument('paths', nargs='+')
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--require-tool-ok', action='store_true')
    parser.add_argument('--shards', type=int, default=2); args = parser.parse_args()
    if args.action == 'plan':
        sets = neuron_sets(args.paths[1]); out = Path(args.paths[2])
        C.write(out/'NEURON_SETS.json', sets); C.write(out/'PLAN.json', plan(args.paths[0], sets, 'smoke' if args.smoke else 'formal', args.shards))
    else:
        summarize(args.paths[0], args.require_tool_ok)
