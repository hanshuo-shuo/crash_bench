"""Repair batch of the mechanism pilot (docs/generative_verifier/HIL_MECHANISM_REPAIR.md): plan and summary.

One batch, two questions, on reruns of HIL suite stage 1 switch branches (SW units: task A's prefix replayed to the switch
step, then prompt B with the prefix seed) under the arms of serve_steer.py:
  1. the action-matched control: in the five failure cells of the closed-loop test, does amp2 beat a push of the same
     size in action space along a random direction (actrand)?
  2. P5 on held-out pairs: the closed-loop repair (amp2: layers 12-14, image tokens, lam 2, nothing tuned) on stage 1
     pairs and suites that were not used to choose the window, against native, the key/value-matched random direction
     (rand2), and three repairs that need no localization: the same amplification on every layer (ampall), extrapolating
     the output chunk from A to B (actdiff) and a rephrased prompt that names the switch (rephrase).
The cells of the held-out pairs are classed from stage 1 alone (all its inits), before any rerun; see classify().

  repair.py plan BRANCH_ROOT OUT_JSON [--smoke]                  the units (read-only on the stage 1 run)
  repair.py summarize ROOT [--require-tool-ok]                   tool check, readings and tables -> ROOT/SUMMARY.json
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

LAYERS = C.LAYERS                     # 12-14, chosen on the open-loop data of the three pilot pairs
ALL_LAYERS = list(range(18))          # every layer of the action expert
ARMS = dict(native=('native', 1., LAYERS), amp2=('amp', 2., LAYERS), rand2=('rand', 2., LAYERS), actrand=('actrand', 2., LAYERS),
            actdiff=('actdiff', 2., LAYERS), rephrase=('native', 1., LAYERS), ampall=('amp', 2., ALL_LAYERS))
ALTERNATIVES = ('actdiff', 'rephrase', 'ampall')
GOAL, TEN, OBJECT = 'libero_goal', 'libero_10', 'libero_object'
PILOT_PAIRS = ((GOAL, '1->8'), (GOAL, '7->8'), (GOAL, '2->9'))


def label(suite, pair):
    return '%s %s' % (suite, pair)


IN_FAILURE = tuple((label(GOAL, p), f) for p, f in (('7->8', .5), ('7->8', .75), ('2->9', .25), ('2->9', .5), ('2->9', .75)))
# Held-out classes as classify() derives them from stage 1 (inits 0-19); plan() refuses a stage 1 run that disagrees.
HELD = dict(
    commits=((label(TEN, '0->1'), .75), (label(TEN, '1->0'), .75), (label(OBJECT, '1->0'), .5)),
    derails=((label(GOAL, '8->7'), .75), (label(TEN, '0->1'), .25), (label(TEN, '0->1'), .5), (label(TEN, '1->0'), .25), (label(TEN, '1->0'), .5)),
    success=tuple((label(GOAL, p), f) for p in ('1->4', '4->8', '8->1') for f in (.25, .5, .75)) + ((label(GOAL, '9->2'), .5),),
    middle=((label(GOAL, '8->7'), .25), (label(GOAL, '8->7'), .5), (label(GOAL, '9->2'), .25), (label(GOAL, '9->2'), .75)),
    start=tuple((label(s, p), 0) for s, p in ((GOAL, '1->4'), (GOAL, '4->8'), (GOAL, '8->1'), (GOAL, '8->7'), (GOAL, '9->2'),
                                              (TEN, '0->1'), (TEN, '1->0'), (OBJECT, '1->0'))))
CELL_ARMS = dict(in_failure=tuple(ARMS), commits=tuple(ARMS), derails=('native', 'amp2', 'rand2'),
                 success=('native', 'amp2'), middle=('native', 'amp2'), start=('native', 'amp2'))
FAILURE_AT, SUCCESS_AT, CEILING_AT, COMMIT_AT = .5, .9, .5, .5   # classify(): stage 1 rates over all inits
SMOKE_INITS = (0, 1)
FORMAL_INITS = C.FORMAL_INITS
GAIN, HARM = C.GAIN, C.HARM
PUSH_TOLERANCE = 1e-5                 # C1: actrand's push norm vs amp's change, per row and group (absolute)


def rephrase(a, b):
    """The rephrase arm's running prompt: names the switch in plain words (fixed before any rerun)."""
    return 'do not %s. %s instead' % (a, b)


def push_seed(sw):
    """Seed of actrand's fixed push directions; independent of the rand arms' key/value directions."""
    return int(hashlib.sha256(('mechanism-repair-push:' + sw).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- plan (read-only on stage 1)
def stage1_cells(branch_plan, runs):
    """(pair label, phase) -> list of (b_done, a_done) over every stage 1 SW unit with a run."""
    by = {r['id']: r for r in runs}; out = defaultdict(list)
    for u in branch_plan['branches']:
        if u['condition'] == 'SW' and u['id'] in by:
            out[(label(u['suite'], '%d->%d' % (u['task'], u['other'])), u['phase'])].append((bool(by[u['id']]['b_done']), bool(by[u['id']]['a_done'])))
    return out


def classify(cells):
    """Held-out classes from stage 1 rates (all inits). A pair is held out when it is not a pilot pair and B from the start
    (f = 0) succeeds at least CEILING_AT. Its f = 0 cell is a start cell; a later cell fails when B's rate is at most
    FAILURE_AT (commits when at least COMMIT_AT of its failed switches finished A, else derails), succeeds at SUCCESS_AT
    or more, and is middle otherwise."""
    pilot = {label(*p) for p in PILOT_PAIRS}; out = defaultdict(list)
    for pair in sorted({p for p, _ in cells} - pilot):
        if np.mean([b for b, _ in cells[(pair, 0)]]) < CEILING_AT:
            continue
        for phase in sorted(f for p, f in cells if p == pair):
            rate = np.mean([b for b, _ in cells[(pair, phase)]]); failed = [a for b, a in cells[(pair, phase)] if not b]
            if phase == 0:
                kind = 'start'
            elif rate <= FAILURE_AT:
                kind = 'commits' if np.mean(failed) >= COMMIT_AT else 'derails'
            else:
                kind = 'success' if rate >= SUCCESS_AT else 'middle'
            out[kind].append((pair, phase))
    return {k: tuple(v) for k, v in out.items()}


def cell_class(key):
    if key in IN_FAILURE:
        return 'in_failure'
    return next((k for k, v in HELD.items() if key in v), None)


def units(branch_plan, runs, inits):
    """Every SW unit of the chosen cells and inits, once per arm of its class (nothing selected on outcome)."""
    done = {r['id']: bool(r['b_done']) for r in runs}
    prompts = {p['id']: p['prompt'] for p in branch_plan['prefix_order']}
    out = []
    for u in branch_plan['branches']:
        if u['condition'] != 'SW' or u['init'] not in inits or u['id'] not in done:
            continue
        pair = label(u['suite'], '%d->%d' % (u['task'], u['other'])); cls = cell_class((pair, u['phase']))
        if cls is None:
            continue
        a = prompts[u['prefix']]
        if a == u['prompt']:
            raise ValueError('B equals A for %s' % u['id'])
        for name in CELL_ARMS[cls]:
            kind, lam, layers = ARMS[name]
            seed = push_seed(u['id']) if kind == 'actrand' else C.direction_seed(u['id'])
            arm = dict(name=name, kind=kind, lam=lam, layers=list(layers), prompt_a=a, direction_seed=seed)
            out.append(dict(u, id='%s__%s' % (u['id'], name), sw=u['id'], cell=cls, pair=pair, arm=arm, stage1_b_done=done[u['id']],
                            prompt=rephrase(a, u['prompt']) if name == 'rephrase' else u['prompt'], prompt_b=u['prompt']))
    return out


def plan(branch_root, mode, shards):
    branch_root = Path(branch_root); branch_plan = C.read(branch_root/'PLAN.json')
    with gzip.open(branch_root/'RUNS.jsonl.gz', 'rt') as stream:
        runs = [json.loads(x) for x in stream]
    derived = classify(stage1_cells(branch_plan, runs))
    if {k: set(v) for k, v in derived.items()} != {k: set(v) for k, v in HELD.items()}:
        raise RuntimeError('Stage 1 classes differ from the frozen held-out cells: %r' % (derived,))
    chosen = units(branch_plan, runs, SMOKE_INITS if mode == 'smoke' else FORMAL_INITS)
    used = {u['prefix'] for u in chosen}
    order = sorted((p for p in branch_plan['prefix_order'] if p['id'] in used), key=lambda p: (p['suite'], p['task'], p['init']))
    rank = {p['id']: k for k, p in enumerate(order)}; arms = list(ARMS)
    chosen.sort(key=lambda u: (rank[u['prefix']], u['phase'], arms.index(u['arm']['name'])))
    return dict(mode=mode, experiment='mechanism-repair', shards=shards, prefix_root=branch_plan['prefix_root'],
                prefix_hashes=branch_plan['prefix_hashes'], prefix_order=order, branches=chosen, excluded=[],
                stage1_root=str(branch_root), stage1_hashes_sha256=C.sha(branch_root/'HASHES.json'), layers=LAYERS,
                arms={k: [v[0], v[1], list(v[2])] for k, v in ARMS.items()}, cell_arms={k: list(v) for k, v in CELL_ARMS.items()},
                in_failure=[list(c) for c in IN_FAILURE], held={k: [list(c) for c in v] for k, v in HELD.items()},
                inits=list(SMOKE_INITS if mode == 'smoke' else FORMAL_INITS))


# ---------------------------------------------------------------- readings (pure)
def restoration(runs, cells, inits):
    """R4's rule on the given failure cells: amp2 vs native (gain) and vs rand2 (specificity); amp2 vs actrand reported."""
    gain = C.paired(runs, 'amp2', 'native', cells, inits); spec = C.paired(runs, 'amp2', 'rand2', cells, inits)
    if gain is None or spec is None:
        reading = 'insufficient'
    elif gain['lo'] > 0:
        reading = ('restores_switch' if gain['mean'] >= GAIN else 'weakly_restores') if spec['lo'] > 0 else 'nonspecific'
    elif gain['hi'] < 0:
        reading = 'harms'
    else:
        reading = 'no_restore'
    return dict(reading=reading, gain=gain, specificity=spec, rand2_vs_native=C.paired(runs, 'rand2', 'native', cells, inits),
                amp2_vs_actrand=C.paired(runs, 'amp2', 'actrand', cells, inits))


def action_control(runs, cells, inits):
    """R6: amp2 vs a push of the same size in action space (actrand), and actrand vs native."""
    spec = C.paired(runs, 'amp2', 'actrand', cells, inits); push = C.paired(runs, 'actrand', 'native', cells, inits)
    if spec is None or push is None:
        reading = 'insufficient'
    elif spec['lo'] > 0:
        reading = 'direction_specific'
    elif push['lo'] > 0:
        reading = 'push_suffices'
    else:
        reading = 'unclear'
    return dict(reading=reading, amp2_vs_actrand=spec, actrand_vs_native=push, amp2_vs_native=C.paired(runs, 'amp2', 'native', cells, inits))


def selectivity(runs, cells, inits):
    """R5's rule: amp2 vs native where native switches (held-out success cells and start cells)."""
    d = C.paired(runs, 'amp2', 'native', cells, inits)
    if d is None:
        reading = 'insufficient'
    elif d['lo'] >= -HARM:
        reading = 'selective'
    elif d['hi'] < 0 and d['mean'] <= -HARM:
        reading = 'damages'
    else:
        reading = 'unclear'
    return dict(reading=reading, change=d)


def alternative(runs, other, cells, inits):
    """R9 for one repair that needs no localization: amp2 vs it, and it vs native, on the commits-type failure cells."""
    better = C.paired(runs, 'amp2', other, cells, inits); own = C.paired(runs, other, 'native', cells, inits)
    if better is None or own is None:
        reading = 'insufficient'
    elif better['lo'] > 0:
        reading = 'amp_better'
    elif own['lo'] > 0 and own['mean'] >= GAIN:
        reading = 'x_comparable'
    else:
        reading = 'unclear'
    return dict(reading=reading, amp2_vs_x=better, x_vs_native=own)


def tables(runs, inits):
    """Per cell and arm: switch count, A completions among the failed switches, median steps to B, and the first chunk's
    median change against lam 1 (L-inf)."""
    out = {}
    for key in sorted({(r['pair'], r['phase']) for r in runs}):
        cell = dict(cls=cell_class(key))
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
    commits = IN_FAILURE + HELD['commits']
    return dict(
        action_control=action_control(runs, IN_FAILURE, inits),                                     # R6
        heldout_restoration={k: restoration(runs, HELD[k], inits) for k in ('commits', 'derails')},  # R7
        heldout_selectivity=selectivity(runs, HELD['success'] + HELD['start'], inits),               # R8
        alternatives={x: alternative(runs, x, commits, inits) for x in ALTERNATIVES},               # R9
        reported=dict(
            in_failure_replication=C.paired(runs, 'amp2', 'native', IN_FAILURE, inits),
            action_control_heldout_commits=action_control(runs, HELD['commits'], inits),
            heldout_failure_pooled=restoration(runs, HELD['commits'] + HELD['derails'], inits),
            heldout_success=C.paired(runs, 'amp2', 'native', HELD['success'], inits),
            heldout_start=C.paired(runs, 'amp2', 'native', HELD['start'], inits),
            heldout_middle=C.paired(runs, 'amp2', 'native', HELD['middle'], inits),
            alternatives_in_failure={x: alternative(runs, x, IN_FAILURE, inits) for x in ALTERNATIVES},
            alternatives_heldout_commits={x: alternative(runs, x, HELD['commits'], inits) for x in ALTERNATIVES}),
        tables=tables(runs, inits))


def tool_reading(runs):
    """C1: (reading, detail). Completion, replay, outcome recomputation and log consistency are checked while loading."""
    fidelity = [r['fidelity_linf'] for r in runs]
    still = [r for r in runs if r['kind'] in ('native', 'actdiff')]; scaled = [r for r in runs if r['kind'] in ('amp', 'rand', 'actrand')]
    moved = [r for r in runs if r['arm'] not in ('native', 'rephrase')]
    detail = dict(runs=len(runs), physics_equal=sum(r['physics_equal'] for r in runs),
                  fidelity_median=float(np.median(fidelity)), fidelity_max=float(np.max(fidelity)),
                  unscaled_moved=sum(r['max_delta'] > 0 for r in still),
                  native_shifted=sum(r['first_shift_linf'] > 0 for r in runs if r['kind'] == 'native'),
                  scale_error_max=max([r['scale_error'] for r in scaled], default=0.),
                  push_match_error_max=max([r['push_match_error'] for r in runs if r['kind'] == 'actrand'], default=0.),
                  steered_unmoved=sum(r['first_shift_linf'] == 0 for r in moved),
                  prompt_errors=sum(not r['prompt_ok'] for r in runs))
    ok = (detail['physics_equal'] == len(runs) and detail['fidelity_median'] <= C.FIDELITY_MEDIAN and detail['fidelity_max'] <= C.FIDELITY_MAX
          and detail['unscaled_moved'] == 0 and detail['native_shifted'] == 0 and detail['scale_error_max'] <= C.SCALE_TOLERANCE
          and detail['push_match_error_max'] <= PUSH_TOLERANCE and detail['steered_unmoved'] == 0 and detail['prompt_errors'] == 0)
    return ('tool_ok' if ok else 'tool_failed'), detail


# ---------------------------------------------------------------- summary (CPU, reads the run root)
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
    kind = u['arm']['kind']; scale = u['arm']['lam'] - 1
    ratio = lambda e, n: e['delta_' + n] / (scale * e['diff_' + n]) if e['diff_' + n] > 0 else (1. if e['delta_' + n] == 0 else np.inf)
    errors = [abs(ratio(e, n) - 1) for e in log for n in 'kv'] if kind in ('amp', 'rand', 'actrand') else [0.]
    want = rephrase(u['arm']['prompt_a'], u['prompt_b']) if u['arm']['name'] == 'rephrase' else u['prompt_b']
    o = row['outcome']
    return dict(id=u['id'], sw=u['sw'], cell=u['cell'], pair=u['pair'], phase=u['phase'], init=u['init'], arm=u['arm']['name'], kind=kind,
                b_done=o['b_done'], a_done=o['a_done'], b_after=o['b_after'], steps=o['steps'], stage1_b_done=u['stage1_b_done'],
                physics_equal=row['physics_at_checkpoint'] == s1['physics_at_checkpoint'],
                first_input_equal=row['first_policy_input_sha256'] == s1['first_policy_input_sha256'],
                prompt_ok=row['prompt'] == want == u['prompt'],
                fidelity_linf=log[0]['fidelity_linf'], first_shift_linf=log[0]['first_shift_linf'],
                max_delta=max(max(e['delta_k'], e['delta_v']) for e in log), scale_error=max(errors),
                push_match_error=max(e['push_match_error'] for e in log) if kind == 'actrand' else 0.,
                push_l2_median=float(np.median([e['push_l2'] for e in log])) if kind == 'actrand' else None,
                ab_action_l2_median=float(np.median([e['ab_action_l2'] for e in log])) if kind == 'actdiff' else None,
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
        actrand_push_l2_median=float(np.median([r['push_l2_median'] for r in per_arm['actrand']])) if per_arm['actrand'] else None,
        actdiff_ab_l2_median=float(np.median([r['ab_action_l2_median'] for r in per_arm['actdiff']])) if per_arm['actdiff'] else None,
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
        print(json.dumps(dict(R6=r['action_control']['reading'], R7={k: v['reading'] for k, v in r['heldout_restoration'].items()},
                              R8=r['heldout_selectivity']['reading'], R9={k: v['reading'] for k, v in r['alternatives'].items()})))
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
