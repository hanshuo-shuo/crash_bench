"""Dose batch of the mechanism pilot (docs/generative_verifier/HIL_MECHANISM_DOSE.md): plan and summary.

The closed-loop test and the repair batch showed that amplifying the action expert's image-route conditioning (layers 12-14,
image tokens, B - A at lam 2) restores failed mid-task switches. This batch asks the reverse, on states where pi0.5 switches
natively: does weakening the same route to the level seen in natural failures (lam 0.5; failure cells fade to about 0.55 of
their start) make the switch fail, and does removing it (lam 0) block it? Each run reruns a HIL suite stage 1 switch branch
(SW unit: A's prefix replayed to the switch step, then prompt B with the prefix seed) under one arm of serve_steer.py:
  native               nothing moved
  wk75 wk50 wk25 wk0   amp at lam .75/.5/.25/0: B's keys/values at LAYERS on the image tokens move toward A's (lam 0 = A's)
  wk0all               amp at lam 0 on every layer of the action expert
  rnd0                 rand at lam 0: a random direction of the size of wk0's change (per vector)
  act0                 actrand at lam 0: the native chunk pushed, per row and action group, by the size of wk0's change
                       along a direction fixed for the run
Cells are stage 1 cells where B succeeds natively (frozen from the stage 1 classes before any rerun).

  dose.py plan BRANCH_ROOT OUT_JSON [--smoke] [--shards N]      the units (read-only on the stage 1 run)
  dose.py summarize ROOT [--require-tool-ok]                     tool check, readings and tables -> ROOT/SUMMARY.json
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
import metrics as M  # noqa: E402
import repair as R  # noqa: E402

LAYERS = C.LAYERS
ALL_LAYERS = R.ALL_LAYERS
ARMS = dict(native=('native', 1., LAYERS), wk75=('amp', .75, LAYERS), wk50=('amp', .5, LAYERS), wk25=('amp', .25, LAYERS),
            wk0=('amp', 0., LAYERS), wk0all=('amp', 0., ALL_LAYERS), rnd0=('rand', 0., LAYERS), act0=('actrand', 0., LAYERS))
CURVE = ('native', 'wk75', 'wk50', 'wk25', 'wk0')
G = R.GOAL
PILOT_SUCCESS = tuple((R.label(G, p), f) for p, f in M.SUCCESS_CELLS)          # 1->8 at .25 / .5 / .75
PILOT_MIDDLE = ((R.label(G, '7->8'), .25),)                                     # stage 1: 15/20
START = tuple((R.label(G, p), 0) for p in ('1->8', '7->8', '2->9'))
HELD_SUCCESS = R.HELD['success']                                                # Goal 1->4, 4->8, 8->1 at .25-.75; 9->2 at .5
CELLS = dict(pilot_success=PILOT_SUCCESS, pilot_middle=PILOT_MIDDLE, heldout_success=HELD_SUCCESS, start=START)
SHORT = ('native', 'wk50', 'wk0', 'act0')
CELL_ARMS = dict(pilot_success=tuple(ARMS), pilot_middle=tuple(ARMS), heldout_success=SHORT, start=SHORT)
SUCCESS = PILOT_SUCCESS + HELD_SUCCESS     # D1-D3
PILOT = PILOT_SUCCESS + PILOT_MIDDLE       # D4, D6
SMOKE_INITS = (0, 1)
FORMAL_INITS = C.FORMAL_INITS              # 2-19
BLOCK = .15             # D1/D2: a drop of at least .15 in the switch rate
NECESSARY = .5          # D2: removal loses at least half the switches
COMMIT_SHARE = .5       # D3: induced failures finish A at least half the time
MIN_INDUCED = 10        # D3: induced failures needed for a reading
SUCCESS_AT = .9         # stage 1 rate that makes a cell a success cell (repair.classify's SUCCESS_AT)


def cell_class(key):
    return next((k for k, v in CELLS.items() if key in v), None)


def stage1_check(cells):
    """The frozen cells against stage 1 (all inits): the held-out success cells are repair.classify's, the pilot success
    cells switch at SUCCESS_AT or more, and 7->8 .25 lies between the failure and success thresholds."""
    derived = R.classify(cells)
    if set(derived.get('success', ())) != set(HELD_SUCCESS):
        raise RuntimeError('Stage 1 success cells differ from the frozen held-out cells: %r' % (derived.get('success'),))
    rate = lambda key: float(np.mean([b for b, _ in cells[key]]))
    if any(rate(k) < SUCCESS_AT for k in PILOT_SUCCESS) or not R.FAILURE_AT < rate(PILOT_MIDDLE[0]) < SUCCESS_AT:
        raise RuntimeError('Stage 1 pilot rates differ from the frozen classes')


def units(branch_plan, runs, inits):
    """Every SW unit of the chosen cells and inits, once per arm of its class (nothing selected on outcome)."""
    done = {r['id']: bool(r['b_done']) for r in runs}
    prompts = {p['id']: p['prompt'] for p in branch_plan['prefix_order']}
    out = []
    for u in branch_plan['branches']:
        if u['condition'] != 'SW' or u['init'] not in inits or u['id'] not in done:
            continue
        pair = R.label(u['suite'], '%d->%d' % (u['task'], u['other'])); cls = cell_class((pair, u['phase']))
        if cls is None:
            continue
        a = prompts[u['prefix']]
        if a == u['prompt']:
            raise ValueError('B equals A for %s' % u['id'])
        for name in CELL_ARMS[cls]:
            kind, lam, layers = ARMS[name]
            seed = R.push_seed(u['id']) if kind == 'actrand' else C.direction_seed(u['id'])
            arm = dict(name=name, kind=kind, lam=lam, layers=list(layers), prompt_a=a, direction_seed=seed)
            out.append(dict(u, id='%s__%s' % (u['id'], name), sw=u['id'], cell=cls, pair=pair, arm=arm, stage1_b_done=done[u['id']],
                            prompt_b=u['prompt']))
    return out


def plan(branch_root, mode, shards):
    branch_root = Path(branch_root); branch_plan = C.read(branch_root/'PLAN.json')
    with gzip.open(branch_root/'RUNS.jsonl.gz', 'rt') as stream:
        runs = [json.loads(x) for x in stream]
    stage1_check(R.stage1_cells(branch_plan, runs))
    inits = SMOKE_INITS if mode == 'smoke' else FORMAL_INITS
    chosen = units(branch_plan, runs, inits)
    used = {u['prefix'] for u in chosen}
    order = sorted((p for p in branch_plan['prefix_order'] if p['id'] in used), key=lambda p: (p['suite'], p['task'], p['init']))
    rank = {p['id']: k for k, p in enumerate(order)}; arms = list(ARMS)
    chosen.sort(key=lambda u: (rank[u['prefix']], u['phase'], arms.index(u['arm']['name'])))
    return dict(mode=mode, experiment='mechanism-dose', shards=shards, prefix_root=branch_plan['prefix_root'],
                prefix_hashes=branch_plan['prefix_hashes'], prefix_order=order, branches=chosen, excluded=[],
                stage1_root=str(branch_root), stage1_hashes_sha256=C.sha(branch_root/'HASHES.json'), layers=LAYERS,
                arms={k: [v[0], v[1], list(v[2])] for k, v in ARMS.items()}, cell_arms={k: list(v) for k, v in CELL_ARMS.items()},
                cells={k: [list(c) for c in v] for k, v in CELLS.items()}, inits=list(inits))


# ---------------------------------------------------------------- readings (pure)
def weakening(runs, arm, cells, inits, necessary):
    """D1 (wk50, necessary=False) / D2 (wk0, necessary=True): drop = native - arm, specificity = act0 - arm."""
    drop = C.paired(runs, 'native', arm, cells, inits); spec = C.paired(runs, 'act0', arm, cells, inits)
    if drop is None or spec is None:
        reading = 'insufficient'
    elif drop['lo'] > 0 and spec['lo'] <= 0:
        reading = 'nonspecific'
    elif drop['lo'] > 0:
        if necessary:
            reading = 'necessary' if drop['mean'] >= NECESSARY else 'contributes' if drop['mean'] >= BLOCK else 'weakly_contributes'
        else:
            reading = 'fade_blocks' if drop['mean'] >= BLOCK else 'fade_weakens'
    elif drop['hi'] < BLOCK:
        reading = 'not_necessary' if necessary else 'fade_tolerated'
    else:
        reading = 'unclear'
    return dict(reading=reading, drop=drop, specificity=spec,
                pilot_vs_rnd0=C.paired(runs, 'rnd0', arm, tuple(c for c in cells if c in PILOT_SUCCESS), inits),
                act0_vs_native=C.paired(runs, 'act0', 'native', cells, inits))


def failure_mode(runs, arm, cells, inits):
    """D3: among states that switch natively but not under the arm, the share that finished A instead."""
    by = {(r['sw'], r['arm']): r for r in runs}
    induced = [by[(r['sw'], arm)] for r in runs if r['arm'] == 'native' and r['b_done'] and (r['pair'], r['phase']) in cells
               and r['init'] in inits and (r['sw'], arm) in by and not by[(r['sw'], arm)]['b_done']]
    share = float(np.mean([r['a_done'] for r in induced])) if induced else None
    if len(induced) < MIN_INDUCED:
        reading = 'insufficient'
    else:
        reading = 'commits_like_natural' if share >= COMMIT_SHARE else 'derails'
    return dict(reading=reading, induced=len(induced), finished_a=sum(r['a_done'] for r in induced), share=share)


def curve(runs, cells, inits, arms=CURVE):
    """Switch rate per arm on the given cells (states present under every listed arm)."""
    by = defaultdict(dict)
    for r in runs:
        if (r['pair'], r['phase']) in cells and r['init'] in inits:
            by[r['sw']][r['arm']] = r['b_done']
    full = [v for v in by.values() if all(a in v for a in arms)]
    return dict(states=len(full), **{a: dict(b_done=sum(v[a] for v in full), rate=float(np.mean([v[a] for v in full])) if full else None) for a in arms})


def tables(runs, inits):
    """Per cell and arm: switches, A completions among the failed switches, median steps to B, the first chunk's median change
    against lam 1 (L-inf) and the image-route gap on request 1."""
    out = {}
    for key in sorted({(r['pair'], r['phase']) for r in runs}):
        cell = dict(cls=cell_class(key))
        for arm in ARMS:
            rs = [r for r in runs if (r['pair'], r['phase']) == key and r['arm'] == arm and r['init'] in inits]
            if rs:
                failed = [r for r in rs if not r['b_done']]; after = [r['b_after'] for r in rs if r['b_done']]
                cell[arm] = dict(n=len(rs), b_done=sum(r['b_done'] for r in rs), a_done_when_failed=sum(r['a_done'] for r in failed),
                                 b_after_median=float(np.median(after)) if after else None,
                                 first_shift_median=float(np.median([r['first_shift_linf'] for r in rs])),
                                 gap_first=float(np.median([r['gap_first'] for r in rs])))
        out['%s f=%s' % key] = cell
    return out


def readings(runs, inits):
    return dict(
        fade=weakening(runs, 'wk50', SUCCESS, inits, necessary=False),                      # D1
        removal=weakening(runs, 'wk0', SUCCESS, inits, necessary=True),                     # D2
        failure_mode=failure_mode(runs, 'wk0', SUCCESS, inits),                             # D3
        reported=dict(
            failure_mode_wk50=failure_mode(runs, 'wk50', SUCCESS, inits),
            fade_pilot=weakening(runs, 'wk50', PILOT_SUCCESS, inits, necessary=False),
            fade_heldout=weakening(runs, 'wk50', HELD_SUCCESS, inits, necessary=False),
            removal_pilot=weakening(runs, 'wk0', PILOT_SUCCESS, inits, necessary=True),
            removal_heldout=weakening(runs, 'wk0', HELD_SUCCESS, inits, necessary=True),
            curve_pilot=curve(runs, PILOT, inits),                                          # D4
            curve_pilot_success=curve(runs, PILOT_SUCCESS, inits),
            curve_middle=curve(runs, PILOT_MIDDLE, inits),
            start=dict(wk0=C.paired(runs, 'native', 'wk0', START, inits), wk50=C.paired(runs, 'native', 'wk50', START, inits),
                       act0=C.paired(runs, 'native', 'act0', START, inits)),                 # D5
            all_layers=dict(wk0all_drop=C.paired(runs, 'native', 'wk0all', PILOT, inits),
                            wk0all_vs_wk0=C.paired(runs, 'wk0', 'wk0all', PILOT, inits)),    # D6
            rnd0_drop_pilot=C.paired(runs, 'native', 'rnd0', PILOT, inits)),
        tables=tables(runs, inits))


def tool_reading(runs):
    """C1: (reading, detail). Completion, replay, outcome recomputation and log consistency are checked while loading."""
    fidelity = [r['fidelity_linf'] for r in runs]
    native = [r for r in runs if r['kind'] == 'native']; moved = [r for r in runs if r['kind'] != 'native']
    detail = dict(runs=len(runs), physics_equal=sum(r['physics_equal'] for r in runs),
                  fidelity_median=float(np.median(fidelity)), fidelity_max=float(np.max(fidelity)),
                  native_moved=sum(r['max_delta'] > 0 or r['first_shift_linf'] > 0 for r in native),
                  scale_error_max=max([r['scale_error'] for r in moved], default=0.),
                  push_match_error_max=max([r['push_match_error'] for r in runs if r['kind'] == 'actrand'], default=0.),
                  steered_unmoved=sum(r['first_shift_linf'] == 0 for r in moved),
                  prompt_errors=sum(not r['prompt_ok'] for r in runs))
    ok = (detail['physics_equal'] == len(runs) and detail['fidelity_median'] <= C.FIDELITY_MEDIAN and detail['fidelity_max'] <= C.FIDELITY_MAX
          and detail['native_moved'] == 0 and detail['scale_error_max'] <= C.SCALE_TOLERANCE
          and detail['push_match_error_max'] <= R.PUSH_TOLERANCE and detail['steered_unmoved'] == 0 and detail['prompt_errors'] == 0)
    return ('tool_ok' if ok else 'tool_failed'), detail


# ---------------------------------------------------------------- summary (CPU, reads the run root)
def scale_errors(log, kind, lam):
    """|applied key/value change / (|lam - 1| * |B - A|) - 1| over every request (the change of amp's program for actrand)."""
    if kind == 'native':
        return [0.]
    scale = abs(lam - 1)
    ratio = lambda e, n: e['delta_' + n] / (scale * e['diff_' + n]) if e['diff_' + n] > 0 else (1. if e['delta_' + n] == 0 else np.inf)
    return [abs(ratio(e, n) - 1) for e in log for n in 'kv']


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
    return dict(id=u['id'], sw=u['sw'], cell=u['cell'], pair=u['pair'], phase=u['phase'], init=u['init'], arm=u['arm']['name'], kind=kind,
                lam=u['arm']['lam'], b_done=o['b_done'], a_done=o['a_done'], b_after=o['b_after'], steps=o['steps'],
                stage1_b_done=u['stage1_b_done'], physics_equal=row['physics_at_checkpoint'] == s1['physics_at_checkpoint'],
                first_input_equal=row['first_policy_input_sha256'] == s1['first_policy_input_sha256'],
                prompt_ok=row['prompt'] == u['prompt_b'] == u['prompt'],
                fidelity_linf=log[0]['fidelity_linf'], first_shift_linf=log[0]['first_shift_linf'],
                max_delta=max(max(e['delta_k'], e['delta_v']) for e in log), scale_error=max(scale_errors(log, kind, u['arm']['lam'])),
                push_match_error=max(e['push_match_error'] for e in log) if kind == 'actrand' else 0.,
                push_l2_median=float(np.median([e['push_l2'] for e in log])) if kind == 'actrand' else None,
                gap_first=log[0]['diff_v'] / log[0]['norm_v'], requests=len(log),
                seconds_per_request=float(np.median([e['seconds'] for e in log[1:]])) if len(log) > 1 else None,
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
        mode=plan_['mode'], tool=tool, tool_detail=detail, runs=len(runs), layers=plan_['layers'],
        readings=readings(runs, inits) if tool == 'tool_ok' else None,
        reading_inits=list(inits), descriptive_only=plan_['mode'] == 'smoke',
        native_vs_stage1=dict(runs=len(native), same_b_done=sum(r['b_done'] == r['stage1_b_done'] for r in native),
                              stage1_b_done=sum(r['stage1_b_done'] for r in native), native_b_done=sum(r['b_done'] for r in native)),
        first_shift_median={a: float(np.median([r['first_shift_linf'] for r in rs])) for a, rs in per_arm.items()},
        act0_push_l2_median=float(np.median([r['push_l2_median'] for r in per_arm['act0']])) if per_arm['act0'] else None,
        first_input_equal=sum(r['first_input_equal'] for r in runs),
        seconds_per_run=float(np.median([r['elapsed_seconds'] for r in runs])),
        seconds_per_request={a: float(np.median([r['seconds_per_request'] for r in rs if r['seconds_per_request'] is not None]))
                             for a, rs in per_arm.items()},
        actions_outside_spec={a: sum(r['actions_outside_spec'] for r in rs) for a, rs in per_arm.items()},
        total_run_hours=sum(r['elapsed_seconds'] for r in runs) / 3600.,
        server_logs_sha256=hashlib.sha256(''.join(p.name + C.sha(p) for p in logs).encode()).hexdigest(), server_logs=len(logs),
        api_calls=0, code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID'))
    with gzip.open(root/'RUNS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in runs)
    C.write(root/'SUMMARY.json', summary)
    C.write(root/'HASHES.json', {n: C.sha(root/n) for n in ('PLAN.json', 'RUNS.jsonl.gz', 'SUMMARY.json')})
    print(json.dumps({k: summary[k] for k in ('mode', 'tool', 'tool_detail', 'native_vs_stage1', 'seconds_per_run')}, indent=1))
    if summary['readings']:
        r = summary['readings']
        print(json.dumps(dict(D1=r['fade']['reading'], D2=r['removal']['reading'], D3=r['failure_mode']['reading'])))
    if require_tool_ok and (tool != 'tool_ok' or plan_['mode'] != 'smoke'):
        raise SystemExit('Gate: the smoke tool check is %s; the formal batch does not start' % tool)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['plan', 'summarize']); parser.add_argument('paths', nargs='+')
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--require-tool-ok', action='store_true')
    parser.add_argument('--shards', type=int, default=2); args = parser.parse_args()
    if args.action == 'plan':
        body = plan(args.paths[0], 'smoke' if args.smoke else 'formal', args.shards); C.write(args.paths[1], body)
        print(len(body['branches']), 'units', len(body['prefix_order']), 'prefixes')
    else:
        summarize(args.paths[0], args.require_tool_ok)
