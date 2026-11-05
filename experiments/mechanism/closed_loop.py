"""Closed-loop causal test of the mechanism pilot (docs/generative_verifier/HIL_MECHANISM_CLOSED_LOOP.md): plan and summary.

A unit is one HIL suite stage 1 switch run (an SW branch: the prefix of task A replayed to the switch step, then prompt B
with the prefix seed) rerun under one arm. The arm changes only what pi0.5's action expert reads (serve_steer.py): at
LAYERS, on the image tokens, the keys/values under prompt B move further from those under A on the same observation by
lam - 1 times their difference (amp), or by a random vector of the same norm (rand); native moves nothing. The runner is
verifier_dataset/hil_suite_run.py's branch phase, unchanged apart from passing the arm on the RNG reset.

  closed_loop.py plan BRANCH_ROOT OUT_JSON [--smoke]            the units (read-only on the stage 1 run)
  closed_loop.py summarize ROOT                                 tool check, readings and tables -> ROOT/SUMMARY.json
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
import metrics as M  # noqa: E402

LAYERS = [12, 13, 14]  # the best image-only window at f = 0 across the three pairs (HIL_MECHANISM_REPORT.md, P2)
ARMS = dict(native=('native', 1.), amp2=('amp', 2.), rand2=('rand', 2.), amp4=('amp', 4.), rand4=('rand', 4.))
DOSES = (2, 4)
FAILURE = M.FAILURE_CELLS
SUCCESS = M.SUCCESS_CELLS
START = (('1->8', 0), ('7->8', 0), ('2->9', 0))
CELL_ARMS = dict(failure=('native', 'amp2', 'rand2', 'amp4', 'rand4'), success=('native', 'amp2', 'amp4'), start=('native', 'amp2', 'amp4'))
SMOKE_INITS = (0, 1)
FORMAL_INITS = M.READING_INITS
GAIN = .15      # R4: a restoring dose raises the failure cells' switch rate by >= .15 over native
HARM = .10      # R5: a selective dose lowers the success and start cells' rate by less than .10
FIDELITY_MEDIAN, FIDELITY_MAX = .01, .05   # C1: request 1, steering program at lam 1 vs openpi's Policy.infer (L-inf)
SCALE_TOLERANCE = 1e-3
DRAWS, SEED = 2000, 0


def read(path):
    with open(path) as stream:
        return json.load(stream)


def write(path, data):
    temporary = Path(str(path) + '.tmp'); temporary.write_text(json.dumps(data, indent=1)); temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell_of(pair, phase):
    key = (pair, phase)
    return 'failure' if key in FAILURE else 'success' if key in SUCCESS else 'start' if key in START else None


def direction_seed(sw):
    """Seed of the random directions of a state's rand arms (shared by rand2 and rand4, so rand4 is rand2 scaled by 3)."""
    return int(hashlib.sha256(('mechanism-closed-loop:' + sw).encode()).hexdigest()[:8], 16)


def units(branch_plan, runs, inits):
    """Every 1a SW unit of the chosen cells and inits, once per arm of its cell (nothing selected on outcome)."""
    done = {r['id']: bool(r['b_done']) for r in runs}
    prompts = {p['id']: p['prompt'] for p in branch_plan['prefix_order']}
    out = []
    for u in branch_plan['branches']:
        if u['group'] != '1a' or u['condition'] != 'SW' or u['init'] not in inits:
            continue
        pair = '%d->%d' % (u['task'], u['other']); cell = cell_of(pair, u['phase'])
        if cell is None:
            continue
        if prompts[u['prefix']] == u['prompt']:
            raise ValueError('B equals A for %s' % u['id'])
        for name in CELL_ARMS[cell]:
            kind, lam = ARMS[name]
            arm = dict(name=name, kind=kind, lam=lam, layers=list(LAYERS), prompt_a=prompts[u['prefix']], direction_seed=direction_seed(u['id']))
            out.append(dict(u, id='%s__%s' % (u['id'], name), sw=u['id'], cell=cell, pair=pair, arm=arm, stage1_b_done=done[u['id']]))
    return out


def plan(branch_root, mode):
    """The closed-loop PLAN.json body: units ordered by prefix (task, init), the prefixes as the runner's shard order."""
    branch_root = Path(branch_root); branch_plan = read(branch_root/'PLAN.json')
    with gzip.open(branch_root/'RUNS.jsonl.gz', 'rt') as stream:
        runs = [json.loads(x) for x in stream]
    chosen = units(branch_plan, runs, SMOKE_INITS if mode == 'smoke' else FORMAL_INITS)
    used = {u['prefix'] for u in chosen}
    order = sorted((p for p in branch_plan['prefix_order'] if p['id'] in used), key=lambda p: (p['task'], p['init']))
    rank = {p['id']: k for k, p in enumerate(order)}; arms = list(ARMS)
    chosen.sort(key=lambda u: (rank[u['prefix']], u['phase'], arms.index(u['arm']['name'])))
    return dict(mode=mode, experiment='mechanism-closed-loop', shards=2, prefix_root=branch_plan['prefix_root'],
                prefix_hashes=branch_plan['prefix_hashes'], prefix_order=order, branches=chosen, excluded=[],
                stage1_root=str(branch_root), stage1_hashes_sha256=sha(branch_root/'HASHES.json'), layers=LAYERS,
                arms={k: list(v) for k, v in ARMS.items()}, cell_arms={k: list(v) for k, v in CELL_ARMS.items()},
                inits=list(SMOKE_INITS if mode == 'smoke' else FORMAL_INITS))


# ---------------------------------------------------------------- readings (pure)
def paired(runs, x, y, cells, inits, draws=DRAWS, seed=SEED):
    """Mean over states of b_done[x] - b_done[y] (states paired by their stage 1 SW unit), with a percentile bootstrap 95%
    interval that resamples states within each (pair, phase) cell."""
    by = {(r['sw'], r['arm']): int(r['b_done']) for r in runs}
    groups = defaultdict(list)
    for r in runs:
        if r['arm'] == x and (r['pair'], r['phase']) in cells and r['init'] in inits and (r['sw'], y) in by:
            groups[(r['pair'], r['phase'])].append((by[(r['sw'], x)], by[(r['sw'], y)]))
    pairs = [p for g in groups.values() for p in g]
    if not pairs:
        return None
    diffs = [np.asarray([a - b for a, b in g], float) for g in groups.values()]
    rng = np.random.default_rng(seed)
    boot = [np.concatenate([d[rng.integers(0, len(d), len(d))] for d in diffs]).mean() for _ in range(draws)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=len(pairs), mean=float(np.mean([a - b for a, b in pairs])), lo=float(lo), hi=float(hi),
                x_rate=float(np.mean([a for a, _ in pairs])), y_rate=float(np.mean([b for _, b in pairs])),
                x_only=sum(a > b for a, b in pairs), y_only=sum(b > a for a, b in pairs))


def restoration(runs, dose, inits):
    """R4 at one dose: failure cells, amp vs native (gain) and amp vs rand at the same norm (specificity)."""
    gain = paired(runs, 'amp%d' % dose, 'native', FAILURE, inits); spec = paired(runs, 'amp%d' % dose, 'rand%d' % dose, FAILURE, inits)
    if gain is None or spec is None:
        reading = 'insufficient'
    elif gain['lo'] > 0:
        reading = ('restores_switch' if gain['mean'] >= GAIN else 'weakly_restores') if spec['lo'] > 0 else 'nonspecific'
    elif gain['hi'] < 0:
        reading = 'harms'
    else:
        reading = 'no_restore'
    return dict(reading=reading, gain=gain, specificity=spec, random_vs_native=paired(runs, 'rand%d' % dose, 'native', FAILURE, inits))


def selectivity(runs, dose, inits):
    """R5 at one dose: success and start cells, amp vs native."""
    d = paired(runs, 'amp%d' % dose, 'native', SUCCESS + START, inits)
    if d is None:
        reading = 'insufficient'
    elif d['lo'] >= -HARM:
        reading = 'selective'
    elif d['hi'] < 0 and d['mean'] <= -HARM:
        reading = 'damages'
    else:
        reading = 'unclear'
    return dict(reading=reading, change=d, success_cells=paired(runs, 'amp%d' % dose, 'native', SUCCESS, inits),
                start_cells=paired(runs, 'amp%d' % dose, 'native', START, inits))


def tool_reading(runs):
    """C1: (reading, detail). Completion, replay and log consistency are checked while loading (they raise)."""
    fidelity = [r['fidelity_linf'] for r in runs]
    native = [r for r in runs if r['arm'] == 'native']; steered = [r for r in runs if r['arm'] != 'native']
    detail = dict(runs=len(runs), physics_equal=sum(r['physics_equal'] for r in runs),
                  fidelity_median=float(np.median(fidelity)), fidelity_max=float(np.max(fidelity)),
                  native_moved=sum(r['max_delta'] > 0 or r['first_shift_linf'] > 0 for r in native),
                  scale_error_max=max([r['scale_error'] for r in steered], default=0.),
                  steered_unmoved=sum(r['first_shift_linf'] == 0 for r in steered))
    ok = (detail['physics_equal'] == len(runs) and detail['fidelity_median'] <= FIDELITY_MEDIAN and detail['fidelity_max'] <= FIDELITY_MAX
          and detail['native_moved'] == 0 and detail['scale_error_max'] <= SCALE_TOLERANCE and detail['steered_unmoved'] == 0)
    return ('tool_ok' if ok else 'tool_failed'), detail


def tables(runs, inits):
    """Per cell and arm: switch rate, A completions among the failed switches, median steps to B, the median first-chunk
    change against lam 1 (L-inf), and the image-route gap at LAYERS on request 1 and its median over the run
    (|V_B - V_A| / |V_B|)."""
    out = {}
    for key in sorted({(r['pair'], r['phase']) for r in runs}):
        cell = {}
        for arm in ARMS:
            rs = [r for r in runs if (r['pair'], r['phase']) == key and r['arm'] == arm and r['init'] in inits]
            if rs:
                failed = [r for r in rs if not r['b_done']]; after = [r['b_after'] for r in rs if r['b_done']]
                cell[arm] = dict(n=len(rs), b_done=sum(r['b_done'] for r in rs), a_done_when_failed=sum(r['a_done'] for r in failed),
                                 b_after_median=float(np.median(after)) if after else None,
                                 first_shift_median=float(np.median([r['first_shift_linf'] for r in rs])),
                                 gap_first=float(np.median([r['gap_first'] for r in rs])), gap_run=float(np.median([r['gap_run'] for r in rs])))
        out['%s f=%s' % key] = cell
    return out


def readings(runs, inits):
    # Reported, not a gate (added after the smoke, before freezing): the rand arms match amp's key/value change, but move
    # the first chunk far less (smoke median L-inf .018 rand2, .085 rand4, .18 amp2); rand4 is the closer match in actions.
    return dict(restoration={'lam%d' % d: restoration(runs, d, inits) for d in DOSES},
                amp2_vs_rand4=paired(runs, 'amp2', 'rand4', FAILURE, inits),
                selectivity={'lam%d' % d: selectivity(runs, d, inits) for d in DOSES}, tables=tables(runs, inits))


# ---------------------------------------------------------------- summary (CPU, reads the run root)
def server_log(root, run_id, client_requests):
    """The server's log of one run, checked against the client's POLICY.jsonl request by request."""
    with open(Path(root)/'server'/(run_id + '.jsonl')) as stream:
        entries = [json.loads(x) for x in stream]
    if [e['request_index'] for e in entries] != list(range(1, len(client_requests) + 1)):
        raise RuntimeError('Server log and client requests differ in count: ' + run_id)
    for e, c in zip(entries, client_requests):
        if (e['input_sha256'], e['rng_before'], e['rng_after'], e['chunk_sha256']) != (c['input_sha256'], c['rng_before'], c['rng_after'], c['native_chunk_sha256']):
            raise RuntimeError('Server log and client request differ: %s request %d' % (run_id, e['request_index']))
    return entries


def same_outcome(x, y):
    """Recomputed vs recorded outcome: flags and step counts exactly, distances to 1e-9 (the runner's Python 3.8 and the
    summary's Python 3.11 round math.dist differently in the last bit)."""
    import math
    if set(x) != set(y):
        return False
    for k in x:
        a, b = x[k], y[k]
        if isinstance(a, float) and isinstance(b, float) and not isinstance(a, bool):
            if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
                return False
        elif type(a) is not type(b) or a != b:
            return False
    return True


def run_row(root, u, stage1_root):
    from verifier_dataset import hil_suite as S
    from verifier_dataset import hil_suite_run as H
    row, trace = H.load(Path(root)/'branch'/u['id'])
    if not same_outcome(S.outcome(trace, u, row['start_eef']), row['outcome']):
        raise RuntimeError('Recomputed outcome differs: ' + u['id'])
    if row['rng_reset'].get('arm') != u['arm']:
        raise RuntimeError('Arm not acknowledged: ' + u['id'])
    with open(Path(root)/'branch'/u['id']/'POLICY.jsonl') as stream:
        client = [json.loads(x) for x in stream]
    log = server_log(root, row['rng_reset']['run_id'], client)
    s1 = read(Path(stage1_root)/'branch'/u['sw']/'ROW.json')
    scale = u['arm']['lam'] - 1
    ratio = lambda e, n: e['delta_' + n] / (scale * e['diff_' + n]) if e['diff_' + n] > 0 else (1. if e['delta_' + n] == 0 else np.inf)
    errors = [abs(ratio(e, n) - 1) for e in log for n in 'kv'] if scale else [0.]
    gaps = [e['diff_v'] / e['norm_v'] for e in log]
    o = row['outcome']
    return dict(id=u['id'], sw=u['sw'], cell=u['cell'], pair=u['pair'], phase=u['phase'], init=u['init'], arm=u['arm']['name'],
                b_done=o['b_done'], a_done=o['a_done'], b_after=o['b_after'], steps=o['steps'], stage1_b_done=u['stage1_b_done'],
                physics_equal=row['physics_at_checkpoint'] == s1['physics_at_checkpoint'],
                first_input_equal=row['first_policy_input_sha256'] == s1['first_policy_input_sha256'],
                fidelity_linf=log[0]['fidelity_linf'], first_shift_linf=log[0]['first_shift_linf'],
                max_delta=max(max(e['delta_k'], e['delta_v']) for e in log), scale_error=max(errors),
                gap_first=gaps[0], gap_run=float(np.median(gaps)), requests=len(log),
                seconds_per_request=float(np.median([e['seconds'] for e in log[1:]])) if len(log) > 1 else None,
                actions_outside_spec=row['actions_outside_spec'], elapsed_seconds=row['elapsed_seconds'])


def summarize(root):
    root = Path(root); plan_ = read(root/'PLAN.json')
    for k in range(plan_['shards']):
        if not read(root/('SHARD_branch_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Shard %d incomplete' % k)
    if sha(Path(plan_['stage1_root'])/'HASHES.json') != plan_['stage1_hashes_sha256']:
        raise RuntimeError('Stage 1 run changed')
    runs = [run_row(root, u, plan_['stage1_root']) for u in plan_['branches']]
    tool, detail = tool_reading(runs)
    inits = tuple(plan_['inits'])
    native = [r for r in runs if r['arm'] == 'native']
    logs = sorted((root/'server').glob('*.jsonl'))
    summary = dict(
        mode=plan_['mode'], tool=tool, tool_detail=detail, runs=len(runs), layers=plan_['layers'],
        readings=readings(runs, inits) if tool == 'tool_ok' else None,
        reading_inits=list(inits), descriptive_only=plan_['mode'] == 'smoke',
        native_vs_stage1=dict(runs=len(native), same_b_done=sum(r['b_done'] == r['stage1_b_done'] for r in native),
                              stage1_b_done=sum(r['stage1_b_done'] for r in native), native_b_done=sum(r['b_done'] for r in native)),
        first_input_equal=sum(r['first_input_equal'] for r in runs),
        seconds_per_run=float(np.median([r['elapsed_seconds'] for r in runs])),
        seconds_per_request=float(np.median([r['seconds_per_request'] for r in runs if r['seconds_per_request'] is not None])),
        actions_outside_spec=sum(r['actions_outside_spec'] for r in runs), total_run_hours=sum(r['elapsed_seconds'] for r in runs) / 3600.,
        server_logs_sha256=hashlib.sha256(''.join(p.name + sha(p) for p in logs).encode()).hexdigest(), server_logs=len(logs),
        api_calls=0, code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID'))
    with gzip.open(root/'RUNS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in runs)
    write(root/'SUMMARY.json', summary)
    write(root/'HASHES.json', {n: sha(root/n) for n in ('PLAN.json', 'RUNS.jsonl.gz', 'SUMMARY.json')})
    print(json.dumps({k: summary[k] for k in ('tool', 'tool_detail', 'native_vs_stage1', 'seconds_per_run')}, indent=1))
    if summary['readings']:
        print(json.dumps({k: {d: v['reading'] for d, v in summary['readings'][k].items()} for k in ('restoration', 'selectivity')}))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['plan', 'summarize']); parser.add_argument('paths', nargs='+')
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    if args.action == 'plan':
        body = plan(args.paths[0], 'smoke' if args.smoke else 'formal'); write(args.paths[1], body)
        print(len(body['branches']), 'units', len(body['prefix_order']), 'prefixes')
    else:
        summarize(args.paths[0])
