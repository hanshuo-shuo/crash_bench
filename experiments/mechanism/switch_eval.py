"""Evaluation of a fine-tuned pi0.5 checkpoint on census states (docs/generative_verifier/HIL_TRAIN_PROTOCOL.md, section
2.4): plan and summary.

The runner and the policy server are the census's, unchanged; only the served checkpoint differs. Every unit carries the
census's native arm (the server moves nothing), so what runs is the fine-tuned model itself. Three kinds of unit on the
states of one split (teacher.split):
  switch  a census mid-task state: prompt B from the switch step on; stop when B holds. Compared with the stored runs of the
          same state: the untouched pi05_libero (native), the scripted retreat (home), release and amp2.
  start   a census f = 0 state: prompt B from the end of the wait steps. Compared with the stored untouched run.
  keep    a mid-task state with the prompt left at A, once per (old task, initial layout, switch step); stop when A holds.
          Compared with the same units run on the base checkpoint (plan --tag base, kinds keep only).

A model taught the other mid-task instructions (protocol section 8) is also run on three more kinds, each once per (old task,
initial layout, switch step), with the suite's prompt and rule:
  stop, return, cancel   the prompt becomes STOP, RETURN or CANCEL at the switch step; the run did what was asked when it is
          stopped, home, or has not finished A (hil_suite.follow). Compared with the same units on the base checkpoint and
          with the script on the same state (--script-root, a finished teacher.py baseline).

  switch_eval.py plan CENSUS_ROOT RETREAT_ROOT OUT_ROOT SPLIT TAG CHECKPOINT [--kinds ...] [--base-root ROOT] [--script-root ROOT]
  switch_eval.py summarize ROOT          tool check, readings and tables -> ROOT/SUMMARY.json
"""
import argparse
from collections import Counter, defaultdict
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
import census as K  # noqa: E402
import teacher as T  # noqa: E402
from verifier_dataset import hil_suite as S  # noqa: E402

KINDS = ('switch', 'start', 'keep')
INSTRUCT = {'stop': 'STOP', 'return': 'RETURN', 'cancel': 'CANCEL'}   # unit kind -> the suite's condition
PER_STATE = ('keep',) + tuple(INSTRUCT)                               # kinds run once per (old task, initial layout, switch step)
SHARDS = 2
# Readings (protocol section 5).
LEARNS = .15            # E1: fine-tuned minus untouched on the switch states
MATCH, BELOW = -.10, -.20   # E2: fine-tuned minus the scripted retreat
KEEPS_START, KEEPS_A = -.05, -.10   # E3: from the start; continuing the old task
OBEYS, IGNORES = S.OBEYS, .5         # E4: share of runs that did what the instruction asks
DRAWS, SEED = K.DRAWS, K.SEED


# ---------------------------------------------------------------- plan
def units(branches, split, kinds, tag):
    """The units of one split. branches: the census plan's units."""
    native = [u for u in branches if u['arm']['name'] == 'native' and T.split(u['task'], u['init']) == split]
    prompts = {u['task']: u['arm']['prompt_a'] for u in branches}
    out = []
    for u in native:
        kind = 'switch' if u['phase'] else 'start'
        if kind in kinds:
            out.append(dict(u, id='%s__%s' % (u['sw'], tag), kind=kind))
    states = {}
    for u in native:
        if u['phase']:
            states.setdefault((u['prefix'], u['checkpoint']), u)
    for kind in PER_STATE:
        if kind not in kinds:
            continue
        condition = INSTRUCT.get(kind, 'C0')
        for key, u in sorted(states.items()):
            sw = '%s_s%03d_%s' % (key + ('KEEP' if kind == 'keep' else condition,)); a = prompts[u['task']]
            # The server refuses a running prompt equal to the arm's old prompt; the native arm never uses the old prompt.
            arm = dict(u['arm'], prompt_a=prompts[(u['task'] + 1) % len(prompts)])
            out.append(dict(id='%s__%s' % (sw, tag), sw=sw, kind=kind, group=u['group'], stages=u['stages'], suite=u['suite'], task=u['task'],
                            other=None, init=u['init'], phase=u['phase'], checkpoint=u['checkpoint'], condition=condition,
                            prompt=dict(C0=a, STOP=S.STOP, RETURN=S.RETURN, CANCEL=S.CANCEL % a)[condition],
                            seed=u['seed'], budget=u['budget'], stop_on='A', record=[], prefix=u['prefix'],
                            pair='%d->%s' % (u['task'], u['task'] if kind == 'keep' else condition), arm=arm, covariates=u['covariates']))
    return out


def plan(census_root, retreat_root, split, tag, checkpoint, kinds=KINDS, base_root=None, shards=SHARDS, script_root=None):
    import retreat as R
    census_root = Path(census_root); retreat_root = Path(retreat_root); census_plan = R.finished_census(census_root)
    if split not in ('dev', 'test') or not set(kinds) <= set(KINDS) | set(INSTRUCT) or not tag.isalnum():
        raise RuntimeError('Unknown split, kind or tag')
    chosen = units(census_plan['branches'], split, kinds, tag); used = {u['prefix'] for u in chosen}
    return dict(mode=split, experiment='mechanism-switch-eval', tag=tag, kinds=list(kinds), checkpoint=str(checkpoint), shards=shards,
                held_out=list(T.HELD_OUT), dev_init=T.DEV_INIT, prefix_root=census_plan['prefix_root'], prefix_hashes=census_plan['prefix_hashes'],
                prefix_order=[p for p in census_plan['prefix_order'] if p['id'] in used], branches=chosen,
                census_root=str(census_root), census_hashes_sha256=C.sha(census_root/'HASHES.json'), retreat_root=str(retreat_root),
                retreat_hashes_sha256=C.sha(retreat_root/'HASHES.json'), base_root=None if base_root is None else str(base_root),
                script_root=None if script_root is None else str(script_root), units_per_kind=dict(Counter(u['kind'] for u in chosen)))


def write_plan(census_root, retreat_root, out_root, split, tag, checkpoint, kinds=KINDS, base_root=None, shards=SHARDS, script_root=None):
    out_root = Path(out_root)
    if (out_root/'PLAN.json').exists():
        raise RuntimeError('Plan target already has a plan: %s' % out_root)
    body = plan(census_root, retreat_root, split, tag, checkpoint, kinds, base_root, shards, script_root); C.write(out_root/'PLAN.json', body)
    print(split, tag, body['units_per_kind'], len(body['prefix_order']), 'prefixes')
    return body


# ---------------------------------------------------------------- statistics (pure)
def cells(rows):
    return {(s['pair'], s['phase']) for s in rows}


def flat_diff(rows, x, y, draws=None, seed=SEED):
    """Mean of x - y over rows with a 95% interval from resampling rows within each (old task, phase): for the keep units,
    which have too few old tasks for the pair-level bootstrap."""
    draws = DRAWS if draws is None else draws
    rows = [r for r in rows if r.get(x) is not None and r.get(y) is not None]
    if not rows:
        return None
    groups = defaultdict(list)
    for r in rows:
        groups[(r['task'], r['phase'])].append(int(r[x]) - int(r[y]))
    rng = np.random.default_rng(seed); keys = sorted(groups)
    boot = [np.mean([d for k in keys for d in rng.choice(groups[k], len(groups[k]))]) for _ in range(draws)]
    lo, hi = K.interval(boot)
    return dict(n=len(rows), mean=float(np.mean([int(r[x]) - int(r[y]) for r in rows])), lo=lo, hi=hi,
                x_rate=float(np.mean([int(r[x]) for r in rows])), y_rate=float(np.mean([int(r[y]) for r in rows])))


def learns(switch):
    """E1: does the fine-tuned model switch where the untouched one does not?"""
    d = K.paired_diff(switch, 'tuned', 'native', cells(switch))
    reading = ('insufficient' if d is None else 'learns' if d['mean'] >= LEARNS and d['lo'] > 0 else
               'does_not_learn' if d['hi'] < LEARNS else 'unclear')
    return dict(reading=reading, tuned_minus_native=d)


def against_script(switch):
    """E2: how far below the scripted retreat?"""
    d = K.paired_diff(switch, 'tuned', 'home', cells(switch))
    reading = ('insufficient' if d is None else 'matches_script' if d['lo'] >= MATCH else
               'below_script' if d['mean'] <= BELOW and d['hi'] < 0 else 'between')
    return dict(reading=reading, tuned_minus_home=d)


def keeps(start, keep):
    """E3: the abilities the model had: a task from the start, and carrying on with the old task when nothing changed."""
    s = K.paired_diff(start, 'tuned', 'native', cells(start)); k = flat_diff(keep, 'tuned', 'base')
    if s is None or k is None:
        reading = 'insufficient'
    else:
        reading = 'keeps' if s['lo'] >= KEEPS_START and k['lo'] >= KEEPS_A else 'forgets'
    return dict(reading=reading, start=s, keep=k)


def obeys(rows):
    """E4, per instruction: the share of runs that did what was asked, beside the base checkpoint and the script."""
    out = {}
    for kind in INSTRUCT:
        rs = [r for r in rows if r['kind'] == kind]
        if not rs:
            continue
        rate = float(np.mean([bool(r['tuned']) for r in rs]))
        out[kind] = dict(reading='obeys' if rate >= OBEYS else 'does_not_obey' if rate < IGNORES else 'partly',
                         tuned=count(rs, 'tuned'), base=count(rs, 'base'), script=count(rs, 'script'), tuned_minus_base=flat_diff(rs, 'tuned', 'base'),
                         released_and_still=count(rs, 'put_down'), old_task_finished=count(rs, 'tuned_a'),
                         by_old_task={str(t): dict(tuned=count([r for r in rs if r['task'] == t], 'tuned'), base=count([r for r in rs if r['task'] == t], 'base'))
                                      for t in sorted({r['task'] for r in rs})})
    return out


def count(rows, key):
    rows = [r for r in rows if r.get(key) is not None]
    return dict(done=sum(bool(r[key]) for r in rows), n=len(rows))


ARMS = ('tuned', 'native', 'home', 'release', 'amp2')


def median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def tables(switch, start, keep):
    def arms(rows):
        return {a: count(rows, a) for a in ARMS}
    both = [s for s in switch if s['tuned'] and s['home']]
    return dict(
        switch=dict(all=arms(switch), by_phase={str(f): arms([s for s in switch if s['phase'] == f]) for f in sorted({s['phase'] for s in switch})},
                    by_old_task={str(t): arms([s for s in switch if s['task'] == t]) for t in sorted({s['task'] for s in switch})},
                    by_new_task={str(t): arms([s for s in switch if s['other'] == t]) for t in sorted({s['other'] for s in switch})},
                    same_object=arms([s for s in switch if s['same_object']]), different_object=arms([s for s in switch if not s['same_object']]),
                    by_pair={p: arms([s for s in switch if s['pair'] == p]) for p in sorted({s['pair'] for s in switch})},
                    outcome=dict(Counter('B' if s['tuned'] else 'A' if s['tuned_a'] else 'neither' for s in switch)),
                    flow=dict(Counter('%s->%s' % ('B' if s['native'] else 'not', 'B' if s['tuned'] else 'not') for s in switch)),
                    script_fails_tuned_succeeds=sum(bool(s['tuned']) and not s['home'] for s in switch),
                    script_succeeds_tuned_fails=sum(bool(s['home']) and not s['tuned'] for s in switch),
                    came_home=count(switch, 'tuned_home'),
                    steps=dict(tuned=median(s['tuned_steps'] for s in switch if s['tuned']), home=median(s['home_steps'] for s in switch if s['home']),
                               both=dict(n=len(both), tuned=median(s['tuned_steps'] for s in both), home=median(s['home_steps'] for s in both)))),
        start=dict(tuned=count(start, 'tuned'), native=count(start, 'native'),
                   by_new_task={str(t): dict(tuned=count([s for s in start if s['other'] == t], 'tuned'), native=count([s for s in start if s['other'] == t], 'native'))
                                for t in sorted({s['other'] for s in start})}),
        keep=dict(tuned=count(keep, 'tuned'), base=count(keep, 'base'),
                  by_old_task={str(t): dict(tuned=count([s for s in keep if s['task'] == t], 'tuned'), base=count([s for s in keep if s['task'] == t], 'base'))
                               for t in sorted({s['task'] for s in keep})}))


def readings(switch, start, keep, instruct=()):
    out = dict(learns=learns(switch), against_script=against_script(switch), keeps=keeps(start, keep), tables=tables(switch, start, keep))
    if instruct:
        out['obeys'] = obeys(instruct)
    return out


# ---------------------------------------------------------------- summary (CPU, reads the run roots)
def lines(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(x) for x in stream]


def state_rows(runs, census_runs, retreat_runs, base_runs):
    """(switch, start, keep): one row per unit with the stored results of the same state."""
    census = {(r['sw'], r['arm']): r for r in census_runs}; retreat = {(r['sw'], r['arm']): r for r in retreat_runs}
    base = {r['sw']: r for r in base_runs}
    switch, start, keep = [], [], []
    for r in runs:
        row = dict(id=r['id'], sw=r['sw'], pair=r['pair'], task=r['task'], other=r['other'], init=r['init'], phase=r['phase'],
                   same_object=r['covariates']['same_object'])
        if r['unit_kind'] == 'keep':
            keep.append(dict(row, tuned=r['a_done'], base=base[r['sw']]['a_done'] if r['sw'] in base else None))
        elif r['unit_kind'] == 'start':
            start.append(dict(row, tuned=r['b_done'], native=census[(r['sw'], 'native')]['b_done']))
        elif r['unit_kind'] == 'switch':
            home = retreat[(r['sw'], 'home')]
            switch.append(dict(row, tuned=r['b_done'], tuned_a=r['a_done'], tuned_steps=r['b_after'], tuned_home=r['home'],
                               native=census[(r['sw'], 'native')]['b_done'], amp2=census[(r['sw'], 'amp2')]['b_done'], home=home['b_done'],
                               release=retreat[(r['sw'], 'release')]['b_done'],
                               home_steps=None if home['b_after'] is None else home['b_after'] + home['scripted_steps']))
    return switch, start, keep


def instruct_rows(runs, base_runs, script):
    """One row per stop, return or cancel unit: whether the fine-tuned model, the base checkpoint and the script (script[sw],
    None when not run) did what was asked. put_down: at the end the gripper is open and the arm at rest."""
    base = {r['sw']: r for r in base_runs}
    follow = lambda r: bool(S.follow(r['condition'], r))
    return [dict(id=r['id'], sw=r['sw'], kind=r['unit_kind'], condition=r['condition'], task=r['task'], init=r['init'], phase=r['phase'],
                 tuned=follow(r), base=follow(base[r['sw']]) if r['sw'] in base else None, script=script.get(r['sw']),
                 tuned_a=r['a_done'], put_down=r['released'] and r['still'])
            for r in runs if r['unit_kind'] in INSTRUCT]


def run_rows(root, plan_, tapes):
    from verifier_dataset import hil_suite_run as H
    out = []; w = S.INSTRUCT_PARAMS['still_window']
    for u in plan_['branches']:
        row = K.run_row(root, u, tapes[u['prefix']]); full, trace = H.load(Path(root)/'branch'/u['id']); o = full['outcome']; last = trace[-1]
        out.append(dict(row, unit_kind=u['kind'], condition=u['condition'], home=o['home'], closest_home=o['closest_home'], stopped=o['stopped'],
                        released=last['gripper'][0] - last['gripper'][1] >= K.CLOSED,
                        still=S.path_length(trace, last['step'] - w, last['step']) < S.INSTRUCT_PARAMS['still_path']))
    return out


def summarize(root):
    from verifier_dataset import hil_suite_run as H
    import retreat as R
    root = Path(root); plan_ = C.read(root/'PLAN.json')
    for k in range(plan_['shards']):
        if not C.read(root/('SHARD_branch_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Shard %d incomplete' % k)
    census_root = Path(plan_['census_root']); retreat_root = Path(plan_['retreat_root'])
    if C.sha(census_root/'HASHES.json') != plan_['census_hashes_sha256'] or C.sha(retreat_root/'HASHES.json') != plan_['retreat_hashes_sha256']:
        raise RuntimeError('A stored batch changed')
    R.finished_census(census_root)
    prefix_root = Path(plan_['prefix_root'])
    for n, digest in plan_['prefix_hashes'].items():
        if C.sha(prefix_root/n) != digest:
            raise RuntimeError('Prefix run output changed: ' + n)
    tapes = {p['id']: H.load(prefix_root/'prefix'/p['id'])[1] for p in plan_['prefix_order']}
    runs = run_rows(root, plan_, tapes)
    check = C.read(root/'STATE_CHECK.json') if (root/'STATE_CHECK.json').exists() else {}
    tool, detail = K.tool_reading(runs, check)
    base_runs = []
    if plan_['base_root'] is not None:
        base_root = Path(plan_['base_root']); base_plan = C.read(base_root/'PLAN.json')
        if base_plan['mode'] != plan_['mode'] or base_plan['kinds'] != [k for k in plan_['kinds'] if k in PER_STATE]:
            raise RuntimeError('The base root is not the per-state units of this split')
        for k in range(base_plan['shards']):
            if not C.read(base_root/('SHARD_branch_%d_COMPLETE.json' % k))['passed']:
                raise RuntimeError('Base shard %d incomplete' % k)
        base_runs = run_rows(base_root, base_plan, {p['id']: H.load(prefix_root/'prefix'/p['id'])[1] for p in base_plan['prefix_order']})
        base_tool = (all(r['physics_equal'] and r['prompt_ok'] for r in base_runs) and max(r['fidelity_linf'] for r in base_runs) <= C.FIDELITY_MAX)
        detail['base'] = dict(runs=len(base_runs), ok=base_tool)
        if not base_tool:
            tool = 'tool_failed'
    switch, start, keep = state_rows(runs, lines(census_root/'RUNS.jsonl.gz'), lines(retreat_root/'RUNS.jsonl.gz'), base_runs)
    script = {}
    if plan_.get('script_root') is not None:
        done = C.read(Path(plan_['script_root'])/'BASELINE.json')
        if not done['passed']:
            raise RuntimeError('The script baseline did not pass')
        script = {r['id']: r['valid'] for r in done['rows']}
    instruct = instruct_rows(runs, base_runs, script)
    logs = sorted((root/'server').glob('*.jsonl'))
    summary = dict(mode=plan_['mode'], tag=plan_['tag'], checkpoint=plan_['checkpoint'], tool=tool, tool_detail=detail, runs=len(runs),
                   units_per_kind=plan_['units_per_kind'], readings=readings(switch, start, keep, instruct) if tool == 'tool_ok' else None,
                   descriptive_only=plan_['mode'] != 'test', seconds_per_run=float(np.median([r['elapsed_seconds'] for r in runs])),
                   total_run_hours=sum(r['elapsed_seconds'] for r in runs) / 3600.,
                   server_logs_sha256=hashlib.sha256(''.join(p.name + C.sha(p) for p in logs).encode()).hexdigest(), server_logs=len(logs),
                   api_calls=0, code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID'))
    with gzip.open(root/'RUNS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in runs + [dict(r, base=True) for r in base_runs])
    C.write(root/'SUMMARY.json', summary)
    C.write(root/'HASHES.json', {n: C.sha(root/n) for n in ('PLAN.json', 'RUNS.jsonl.gz', 'SUMMARY.json')})
    print(json.dumps({k: summary[k] for k in ('mode', 'tag', 'tool', 'tool_detail', 'units_per_kind', 'seconds_per_run')}, indent=1))
    if summary['readings']:
        r = summary['readings']
        print(json.dumps(dict(E1=r['learns'], E2=r['against_script'], E3=r['keeps'], switch=r['tables']['switch']['all'],
                              start=r['tables']['start']['tuned'], keep=r['tables']['keep']), indent=1))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('plan')
    for name in ('census_root', 'retreat_root', 'out_root', 'split', 'tag', 'checkpoint'):
        a.add_argument(name)
    a.add_argument('--kinds', nargs='+', default=list(KINDS)); a.add_argument('--base-root'); a.add_argument('--script-root')
    a = sub.add_parser('summarize'); a.add_argument('root')
    args = p.parse_args()
    if args.command == 'plan':
        write_plan(args.census_root, args.retreat_root, args.out_root, args.split, args.tag, args.checkpoint, tuple(args.kinds), args.base_root,
                   script_root=args.script_root)
    else:
        summarize(args.root)
