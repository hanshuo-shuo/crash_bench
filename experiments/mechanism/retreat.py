"""Retreat and redo (docs/generative_verifier/HIL_RETREAT_PROTOCOL.md): plan and summary.

On the mid-task states of the finished task-pair census (every ordered pair of the ten LIBERO-Goal tasks, initial states 0-9,
switch at half and three quarters of task A), a branch replays A's census prefix to the switch step, runs a scripted retreat
(hil_suite.Retreat, run by verifier_dataset/hil_suite_run.py) and then lets native pi0.5 continue under prompt B:
  home     open the gripper, rise, return to the episode's start pose;
  release  open the gripper and rise 10 cm;
  native   no script (the census condition again), on one initial state in ten, to report how well reruns agree.
The census runs of the same states (native, amp2) are the paired controls and are not rerun. The policy server is
serve_steer.py with the native arm, as in the census, so the request-1 image-token gap is logged after the script.

  retreat.py plan CENSUS_ROOT OUT_ROOT smoke|formal|dry [--settings JSON]   writes OUT_ROOT/PLAN.json (read-only on the census)
  retreat.py summarize ROOT [--out DIR] [--require-tool-ok]                tool check, readings, tables -> SUMMARY.json
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import closed_loop as C  # noqa: E402
import census as K  # noqa: E402
import census_reach_posthoc as W  # noqa: E402
from verifier_dataset import hil_suite as S  # noqa: E402

ARMS = dict(home='HOME', release='RELEASE', native='SW')
RECHECK_INIT = 5                      # formal: the native recheck runs on this initial state only
SHARDS = dict(smoke=2, formal=6, dry=1)
# Readings (protocol section 5).
HIGH, MODERATE = .80, .50             # U1
POSE = .15                            # U2
COST = .10                            # U4
REACH = .90                           # C1: share of home runs that arrive within both tolerances
NEAR = .08                            # reported: a failed run "went to B" when it came this close to B's work point
MOVED = .01                           # reported: an object counts as moved by the script beyond this many metres
DRAWS, SEED = K.DRAWS, K.SEED


# ---------------------------------------------------------------- plan
def finished_census(census_root):
    """The census formal run this batch stands on: complete, unchanged, tool_ok."""
    census_root = Path(census_root)
    for n, digest in C.read(census_root/'HASHES.json').items():
        if C.sha(census_root/n) != digest:
            raise RuntimeError('Census run output changed: ' + n)
    summary = C.read(census_root/'SUMMARY.json'); plan_ = C.read(census_root/'PLAN.json')
    if summary['mode'] != 'formal' or summary['tool'] != 'tool_ok' or plan_.get('experiment') != 'mechanism-census':
        raise RuntimeError('The retreat batch needs the finished formal census run')
    return plan_


def units(census_plan, mode):
    """One unit per arm on each mid-task census state. smoke and dry: initial state 0 with B = (A + 3) mod 10; dry has no
    native arm (it runs no policy); formal: every state, the native recheck on initial state RECHECK_INIT only."""
    out = []
    for u in census_plan['branches']:
        if u['arm']['name'] != 'native' or not u['phase']:
            continue
        if mode != 'formal' and not (u['init'] == K.SMOKE_INIT and u['other'] == (u['task'] + K.SMOKE_SHIFT) % 10):
            continue
        for arm, condition in ARMS.items():
            if arm == 'native' and (mode == 'dry' or (mode == 'formal' and u['init'] != RECHECK_INIT)):
                continue
            out.append(dict(u, id=u['sw'] + '__' + arm, condition=condition, retreat_arm=arm))
    return out


def plan(census_root, mode, settings=None):
    census_root = Path(census_root); census_plan = finished_census(census_root)
    chosen = units(census_plan, mode); used = {u['prefix'] for u in chosen}
    body = dict(mode=mode, experiment='mechanism-retreat', shards=SHARDS[mode], prefix_root=census_plan['prefix_root'],
                prefix_hashes=census_plan['prefix_hashes'], prefix_order=[p for p in census_plan['prefix_order'] if p['id'] in used],
                branches=chosen, excluded=census_plan['excluded'], census_root=str(census_root),
                census_hashes_sha256=C.sha(census_root/'HASHES.json'), params=dict(S.RETREAT_PARAMS), arms=dict(ARMS),
                recheck_init=RECHECK_INIT, inits=sorted({u['init'] for u in chosen}))
    if mode == 'dry':
        body['settings'] = settings if settings is not None else dict(default={})
    return body


def write_plan(census_root, out_root, mode, settings=None):
    out_root = Path(out_root)
    if (out_root/'PLAN.json').exists():
        raise RuntimeError('Plan target already has a plan: %s' % out_root)
    body = plan(census_root, mode, settings); C.write(out_root/'PLAN.json', body)
    print(mode, len(body['branches']), 'units', Counter(u['retreat_arm'] for u in body['branches']), len(body['prefix_order']), 'prefixes')
    return body


# ---------------------------------------------------------------- statistics (pure)
def rate(states, x, cells=None, draws=None, seed=SEED):
    """Mean of states' x (0 / 1; None is left out) with a 95% interval from the census's two-level bootstrap: task pairs with
    replacement, then states within each (pair, phase) cell."""
    draws = DRAWS if draws is None else draws
    rows = [dict(pair=s['pair'], cell=(s['pair'], s['phase']), v=int(s[x])) for s in states
            if s.get(x) is not None and (cells is None or (s['pair'], s['phase']) in cells)]
    if not rows:
        return None
    groups = K.by_pair(rows, lambda r: r['cell']); rng = np.random.default_rng(seed)
    boot = [np.mean([r['v'] for g in K.resample(groups, rng) for v in g.values() for r in v]) for _ in range(draws)]
    lo, hi = K.interval(boot)
    return dict(n=len(rows), cells=len({r['cell'] for r in rows}), pairs=len(groups), done=sum(r['v'] for r in rows),
                mean=float(np.mean([r['v'] for r in rows])), lo=lo, hi=hi)


def cell_median(states, x, cells, draws=None, seed=SEED):
    """Median over cells of each cell's median of x (None left out), with a pair-level bootstrap interval."""
    draws = DRAWS if draws is None else draws
    rows = [dict(pair=s['pair'], cell=(s['pair'], s['phase']), v=float(s[x])) for s in states
            if s.get(x) is not None and (s['pair'], s['phase']) in cells]
    if not rows:
        return None
    def stat(rs):
        by = defaultdict(list)
        for r in rs:
            by[r['cell']].append(r['v'])
        return float(np.median([np.median(v) for v in by.values()]))
    groups = K.by_pair(rows, lambda r: r['cell']); rng = np.random.default_rng(seed)
    boot = [stat([dict(r, cell=(j, r['cell'])) for j, g in enumerate(K.resample(groups, rng)) for v in g.values() for r in v]) for _ in range(draws)]
    lo, hi = K.interval(boot)
    return dict(n=len(rows), cells=len({r['cell'] for r in rows}), median=stat(rows), lo=lo, hi=hi)


# ---------------------------------------------------------------- readings (pure, on per-state rows)
def upper_bound(states):
    """U1: home's B success over every mid-task state."""
    r = rate(states, 'home')
    reading = 'insufficient' if r is None else 'high' if r['mean'] >= HIGH else 'moderate' if r['mean'] >= MODERATE else 'low'
    return dict(reading=reading, home=r, vs_native=K.paired_diff(states, 'home', 'native', all_cells(states)),
                vs_amp2=K.paired_diff(states, 'home', 'amp2', all_cells(states)))


def all_cells(states):
    return {(s['pair'], s['phase']) for s in states}


def release_enough(states, failure):
    """U2 (failure cells): does the start pose matter beyond letting go and rising?"""
    d = K.paired_diff(states, 'home', 'release', failure); r = K.paired_diff(states, 'release', 'native', failure)
    if d is None or r is None:
        reading = 'insufficient'
    elif d['mean'] >= POSE and d['lo'] > 0:
        reading = 'pose_matters'
    elif d['hi'] < POSE and r['mean'] >= POSE and r['lo'] > 0:
        reading = 'release_enough'
    else:
        reading = 'unclear'
    return dict(reading=reading, home_minus_release=d, release_minus_native=r)


def cost(states, success):
    """U4 (census success cells): does retreating hurt switches that already work?"""
    d = K.paired_diff(states, 'home', 'native', success)
    if d is None:
        reading = 'insufficient'
    elif d['lo'] >= -COST:
        reading = 'harmless'
    elif d['hi'] < 0 and d['mean'] <= -COST:
        reading = 'costs'
    else:
        reading = 'unclear'
    return dict(reading=reading, change=d)


def state_rows(runs, census_states, classes):
    """One row per mid-task census state: the census's native and amp2 outcomes and gamma, the cell class, and this batch's
    arms. An arm whose script ended with B already true is None (left out of numerators and denominators) and counted."""
    out = {s['sw']: dict(sw=s['sw'], pair=s['pair'], task=s['task'], other=s['other'], init=s['init'], phase=s['phase'], covariates=s['covariates'],
                         native=int(s['native']), amp2=int(s['amp2']) if 'amp2' in s else None, native_a=bool(s['a_done']),
                         gamma_switch=s['gamma'], start_gap=s['start_gap'], cls=classes.get((s['pair'], s['phase'])))
           for s in census_states if s['phase']}
    for r in runs:
        s = out[r['sw']]; arm = r['arm']
        if arm == 'native':
            s['recheck'] = int(r['b_done']); s['recheck_a'] = bool(r['a_done']); continue
        s[arm] = None if r['b_at_handover'] else int(r['b_done'])
        s[arm + '_a'] = bool(r['a_done']); s[arm + '_handover_b'] = bool(r['b_at_handover']); s[arm + '_reached'] = r['reached']
        s[arm + '_steps'] = r['scripted_steps']; s[arm + '_within'] = r['within_budget']; s[arm + '_near_b'] = r['near_b']
        s[arm + '_moved_b'] = r['moved_b']
        if arm == 'home':
            s['gamma_post'] = r['gap_first'] / s['start_gap'] if s['start_gap'] and r['reached'] else None
    return [s for s in out.values() if 'home' in s or 'release' in s or 'recheck' in s]


def readings(states):
    classes = {(s['pair'], s['phase']): s['cls'] for s in states}
    pick = lambda kind: tuple(k for k, c in classes.items() if c == kind)
    failure, success = pick('failure'), pick('success')
    home = [s for s in states if s.get('home') is not None]

    def share(rows, key):
        rows = [s for s in rows if s.get(key) is not None]
        return dict(n=len(rows), done=sum(int(s[key]) for s in rows), rate=float(np.mean([int(s[key]) for s in rows])) if rows else None)

    def split(arm, group):
        by = defaultdict(list)
        for s in states:
            if s.get(arm) is not None and group(s) is not None:
                by[str(group(s))].append(s)
        return {k: share(v, arm) for k, v in sorted(by.items())}

    def gamma_auc(rows):
        rows = [s for s in rows if s.get('gamma_post') is not None]
        return K.auc([s['gamma_post'] for s in rows], [bool(s['home']) for s in rows]) if rows else None

    kind = lambda b, a: 'B' if b else 'A' if a else 'neither'
    flow = Counter((kind(s['native'], s['native_a']), kind(s['home'], s['home_a'])) for s in home)
    pairs = defaultdict(list)
    for s in home:
        pairs[s['pair']].append(s['home'])
    rates = sorted(float(np.mean(v)) for v in pairs.values())
    failed = lambda arm: [s for s in states if s.get(arm) == 0]
    reported = dict(
        classes={c: sum(v == c for v in classes.values()) for c in ('failure', 'middle', 'success')},
        by_class={c: {arm: rate(states, arm, pick(c)) for arm in ('home', 'release', 'native', 'amp2')}
                  for c in ('failure', 'middle', 'success')},
        release=rate(states, 'release'), release_vs_native=K.paired_diff(states, 'release', 'native', all_cells(states)),
        home_vs_release=K.paired_diff(states, 'home', 'release', all_cells(states)),
        gamma_post=cell_median(states, 'gamma_post', failure), gamma_switch=cell_median([s for s in states if s.get('gamma_post') is not None], 'gamma_switch', failure),
        gamma_post_auc=dict(all=gamma_auc(home), same_object=gamma_auc([s for s in home if s['covariates']['same_object']]),
                            different_object=gamma_auc([s for s in home if not s['covariates']['same_object']])),
        home_failures=dict(n=len(failed('home')), finished_a=sum(s['home_a'] for s in failed('home')),
                           neither=sum(not s['home_a'] for s in failed('home'))),
        went_to_b={arm: share(failed(arm), arm + '_near_b') for arm in ('home', 'release')},
        flow_native_to_home={'%s->%s' % k: v for k, v in sorted(flow.items())},
        home_by={name: split('home', group) for name, group in (
            ('same_object', lambda s: s['covariates']['same_object']), ('gripper_closed', lambda s: s['covariates']['gripper_closed']),
            ('phase', lambda s: s['phase']), ('b_object_moved', lambda s: None if s.get('home_moved_b') is None else s['home_moved_b'] > MOVED),
            ('old_task', lambda s: s['task']), ('new_task', lambda s: s['other']), ('reached', lambda s: s.get('home_reached')))},
        release_by={name: split('release', group) for name, group in (
            ('same_object', lambda s: s['covariates']['same_object']), ('gripper_closed', lambda s: s['covariates']['gripper_closed']),
            ('phase', lambda s: s['phase']))},
        pairs=dict(count=len(rates), zero_success=sum(r == 0 for r in rates), at_least_80=sum(r >= .8 for r in rates),
                   median=float(np.median(rates)) if rates else None, lowest=sorted(pairs, key=lambda p: np.mean(pairs[p]))[:10]),
        handover_b={arm: sum(bool(s.get(arm + '_handover_b')) for s in states) for arm in ('home', 'release')},
        scripted_steps={arm: S.H.distribution(s[arm + '_steps'] for s in states if arm + '_steps' in s) for arm in ('home', 'release')},
        reached={arm: share(states, arm + '_reached') for arm in ('home', 'release')},
        within_budget={arm: share([s for s in states if s.get(arm) is not None], arm + '_within') for arm in ('home', 'release')},
        recheck=dict(states=sum('recheck' in s for s in states), same_b=sum(s['recheck'] == s['native'] for s in states if 'recheck' in s),
                     same_outcome=sum(kind(s['recheck'], s['recheck_a']) == kind(s['native'], s['native_a']) for s in states if 'recheck' in s),
                     recheck_b=sum(s['recheck'] for s in states if 'recheck' in s), census_b=sum(s['native'] for s in states if 'recheck' in s)))
    return dict(upper_bound=upper_bound(states),                 # U1
                release_enough=release_enough(states, failure),  # U2
                cost=cost(states, success),                      # U4
                reported=reported)                               # U3 (gamma_post) and the rest: reported only


def tool_reading(runs, state_check, params):
    """C1: (reading, detail). Completion, hashes, outcome recomputation and the server log are checked while loading."""
    home = [r for r in runs if r['arm'] == 'home']; scripted = [r for r in runs if r['arm'] != 'native']
    fidelity = [r['fidelity_linf'] for r in runs]
    caps = dict(home=params['home_steps'], release=params['release_steps'])
    detail = dict(runs=len(runs), physics_equal=sum(r['physics_equal'] for r in runs),
                  script_out_of_range=sum(not r['script_in_range'] for r in scripted), script_misplaced=sum(not r['script_contiguous'] for r in runs),
                  script_over_cap=sum(r['scripted_steps'] > caps[r['arm']] for r in scripted), unscripted=sum(r['scripted_steps'] == 0 for r in scripted),
                  home_runs=len(home), home_reached=sum(bool(r['reached']) for r in home),
                  home_reach_rate=float(np.mean([bool(r['reached']) for r in home])) if home else None,
                  fidelity_median=float(np.median(fidelity)), fidelity_max=float(np.max(fidelity)),
                  steered=sum(r['max_delta'] > 0 or r['first_shift_linf'] > 0 for r in runs),
                  prompt_errors=sum(not r['prompt_ok'] for r in runs), b_at_switch=sum(bool(r['b_at_switch']) for r in runs),
                  params_current=params == dict(S.RETREAT_PARAMS),
                  state_check=dict(passed=state_check.get('passed'), runs=state_check.get('runs'), cross_checks=state_check.get('cross_checks')))
    ok = (detail['physics_equal'] == len(runs) and detail['script_out_of_range'] == 0 and detail['script_misplaced'] == 0
          and detail['script_over_cap'] == 0 and detail['unscripted'] == 0 and bool(home) and detail['home_reach_rate'] >= REACH
          and detail['fidelity_median'] <= C.FIDELITY_MEDIAN and detail['fidelity_max'] <= C.FIDELITY_MAX and detail['steered'] == 0
          and detail['prompt_errors'] == 0 and detail['b_at_switch'] == 0 and detail['params_current']
          and state_check.get('passed') is True and (state_check.get('cross_checks') or 0) > 0)
    return ('tool_ok' if ok else 'tool_failed'), detail


# ---------------------------------------------------------------- summary (CPU, reads the run root)
def run_row(root, u, tape, points):
    from verifier_dataset import hil_suite_run as H
    row, trace = H.load(Path(root)/'branch'/u['id']); retreating = u['condition'] in S.RETREAT
    recomputed = S.retreat_outcome(trace, u, row['start_eef']) if retreating else S.outcome(trace, u, row['start_eef'])
    if not C.same_outcome(recomputed, row['outcome']):
        raise RuntimeError('Recomputed outcome differs: ' + u['id'])
    if row['rng_reset'].get('arm') != u['arm']:
        raise RuntimeError('Arm not acknowledged: ' + u['id'])
    with open(Path(root)/'branch'/u['id']/'POLICY.jsonl') as stream:
        client = [json.loads(x) for x in stream]
    log = C.server_log(root, row['rng_reset']['run_id'], client)
    o = row['outcome']; n = sum(bool(x.get('scripted')) for x in trace); script = row.get('retreat') or {}
    point = points.get((u['pair'], u['init'])); label = K.MOVED[u['other']]
    moved = [v for k, v in (script.get('moved') or {}).items() if label in k]
    return dict(id=u['id'], sw=u['sw'], pair=u['pair'], task=u['task'], other=u['other'], init=u['init'], phase=u['phase'], arm=u['retreat_arm'],
                b_done=o['b_done'], a_done=o['a_done'], b_after=o['b_after'], steps=o['steps'], b_at_switch=bool(trace[0]['others'][str(u['other'])]),
                b_at_handover=bool(o['b_at_checkpoint']) if retreating else False, b_in_script=o.get('b_in_script', False),
                a_at_handover=o.get('a_at_handover', False), scripted_steps=n,
                script_contiguous=all(bool(x.get('scripted')) == (0 < i <= n) for i, x in enumerate(trace)) and (retreating or n == 0),
                script_in_range=all(abs(v) <= 1 for x in trace[1:n + 1] for v in x['action']),
                reached=script.get('reached'), status=script.get('status'), position_error=script.get('position_error'),
                orientation_error=script.get('orientation_error'), moved_b=max(moved) if moved else None,
                within_budget=bool(o['b_done']) and n + o['b_after'] <= u['budget'],
                near_b=None if point is None else W.closest(trace[n:], point) <= NEAR, covariates=u['covariates'],
                physics_equal=row['physics_at_checkpoint'] == tape[u['checkpoint']]['physics'], prompt_ok=row['prompt'] == u['prompt'],
                fidelity_linf=log[0]['fidelity_linf'], first_shift_linf=log[0]['first_shift_linf'],
                max_delta=max(max(e['delta_k'], e['delta_v']) for e in log), gap_first=log[0]['diff_v'] / log[0]['norm_v'], requests=len(log),
                actions_outside_spec=row['actions_outside_spec'], elapsed_seconds=row['elapsed_seconds'])


def census_view(census_root):
    """(per-state census rows with gamma and the f = 0 gap, cell classes, B's work point per (pair, init))."""
    census_root = Path(census_root)
    with gzip.open(census_root/'RUNS.jsonl.gz', 'rt') as stream:
        runs = [json.loads(x) for x in stream]
    rows = K.states(runs); _, classes = K.classes(K.cell_table(rows))
    start = {(s['pair'], s['init']): s.get('gap') for s in rows if s['phase'] == 0}
    for s in rows:
        s['start_gap'] = start.get((s['pair'], s['init']))
    points = {}
    for r in runs:
        if r['phase'] == 0 and r['b_done']:
            point = W.work_point(W.read(census_root, r['id']))
            if point is not None:
                points[(r['pair'], r['init'])] = point
    return rows, classes, points


def summarize(root, out=None, require_tool_ok=False):
    from verifier_dataset import hil_suite_run as H
    root = Path(root); out = root if out is None else Path(out); plan_ = C.read(root/'PLAN.json')
    for k in range(plan_['shards']):
        if not C.read(root/('SHARD_branch_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Shard %d incomplete' % k)
    census_root = Path(plan_['census_root'])
    if C.sha(census_root/'HASHES.json') != plan_['census_hashes_sha256']:
        raise RuntimeError('Census run changed')
    finished_census(census_root)
    prefix_root = Path(plan_['prefix_root'])
    for n, digest in plan_['prefix_hashes'].items():
        if C.sha(prefix_root/n) != digest:
            raise RuntimeError('Prefix run output changed: ' + n)
    tapes = {p['id']: H.load(prefix_root/'prefix'/p['id'])[1] for p in plan_['prefix_order']}
    census_states, classes, points = census_view(census_root)
    runs = [run_row(root, u, tapes[u['prefix']], points) for u in plan_['branches']]
    check = C.read(root/'STATE_CHECK.json') if (root/'STATE_CHECK.json').exists() else {}
    tool, detail = tool_reading(runs, check, plan_['params'])
    logs = sorted((root/'server').glob('*.jsonl')); per_arm = defaultdict(list)
    for r in runs:
        per_arm[r['arm']].append(r)
    summary = dict(
        mode=plan_['mode'], tool=tool, tool_detail=detail, runs=len(runs), runs_per_arm={a: len(rs) for a, rs in per_arm.items()},
        params=plan_['params'], inits=plan_['inits'], readings=readings(state_rows(runs, census_states, classes)) if tool == 'tool_ok' else None,
        descriptive_only=plan_['mode'] == 'smoke', seconds_per_run=float(np.median([r['elapsed_seconds'] for r in runs])),
        actions_outside_spec={a: sum(r['actions_outside_spec'] for r in rs) for a, rs in per_arm.items()},
        total_run_hours=sum(r['elapsed_seconds'] for r in runs) / 3600., summarized_root=str(root),
        server_logs_sha256=hashlib.sha256(''.join(p.name + C.sha(p) for p in logs).encode()).hexdigest(), server_logs=len(logs),
        api_calls=0, code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID'))
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out/'RUNS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in runs)
    C.write(out/'SUMMARY.json', summary)
    C.write(out/'HASHES.json', {n: C.sha(p) for n, p in (('PLAN.json', root/'PLAN.json'), ('RUNS.jsonl.gz', out/'RUNS.jsonl.gz'),
                                                         ('SUMMARY.json', out/'SUMMARY.json'))})
    print(json.dumps({k: summary[k] for k in ('mode', 'tool', 'tool_detail', 'runs_per_arm', 'seconds_per_run')}, indent=1))
    if summary['readings']:
        r = summary['readings']
        print(json.dumps(dict(U1=r['upper_bound']['reading'], U1_home=r['upper_bound']['home'], U2=r['release_enough']['reading'], U4=r['cost']['reading'])))
    if require_tool_ok and (tool != 'tool_ok' or plan_['mode'] != 'smoke'):
        raise SystemExit('Gate: the smoke tool check is %s; the formal batch does not start' % tool)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['plan', 'summarize']); parser.add_argument('paths', nargs='+')
    parser.add_argument('--settings'); parser.add_argument('--out'); parser.add_argument('--require-tool-ok', action='store_true')
    args = parser.parse_args()
    if args.action == 'plan':
        census_root, out_root, mode = args.paths
        write_plan(os.path.abspath(census_root), os.path.abspath(out_root), mode, None if args.settings is None else C.read(args.settings))
    else:
        summarize(args.paths[0], args.out, args.require_tool_ok)
