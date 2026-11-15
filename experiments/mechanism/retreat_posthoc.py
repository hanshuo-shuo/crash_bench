"""Post-hoc read of the retreat-and-redo batch: where do the failures of the `home` arm sit, and what does the policy do there?

Read-only, CPU, standard library only. Written after the batch's results were seen (2026-11-15), so everything here is
descriptive and must be labelled post hoc wherever it is quoted; the frozen readings are retreat.py summarize -> SUMMARY.json.

Why: the summary's by-old-task table shows one old task (3, open the top drawer and put the bowl inside) at 30% while the other
nine are at 87% or above. This script reads RUNS.jsonl.gz of the retreat batch and of the census, and the stored step traces
of the old-task-3 runs.

Tables:
- by_old_task, by_pair of the exception: B success per arm (home, release; census native, amp2; census start = native at f = 0).
- without: every rate again with the exception left out, split by same / different object.
- worse: states that switch untouched but not after the retreat, by pair.
- behaviour: in the exception's failed home runs, did the policy finish A, never close the gripper (no stretch of GRASP
  consecutive steps with the aperture below CLOSED after the hand-over), or close it and finish neither.
- gamma_auc: AUROC of the mark after the retreat (request 1's gap over the census f = 0 gap of the same pair and init) for B
  success, home runs that reached the start pose; all / same object / different object, with and without the exception.
- state_check: recorded-versus-re-evaluated differences of STATE_CHECK.json by task (reported, not failed).

Usage: python retreat_posthoc.py RETREAT_FORMAL_ROOT CENSUS_FORMAL_ROOT [OLD_TASK]        prints JSON
"""
import collections
import gzip
import json
import sys
from pathlib import Path

CLOSED = .05            # same threshold as census.CLOSED
GRASP = 5               # consecutive closed steps that count as a grasp attempt
EXCEPTION = 3           # the old task read as the exception


def lines(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(x) for x in stream]


def aperture(step):
    return abs(step['gripper'][0] - step['gripper'][1])


def longest_closed(steps):
    """Longest stretch of consecutive steps with the gripper closed."""
    run = best = 0
    for s in steps:
        run = run + 1 if aperture(s) < CLOSED else 0; best = max(best, run)
    return best


def auc(positive, negative):
    """Probability that a positive scores above a negative (ties count half); None without both classes."""
    if not positive or not negative:
        return None
    return sum((p > n) + .5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def count(rows, key='b_done'):
    rows = list(rows)
    return dict(done=sum(bool(r[key]) for r in rows), n=len(rows))


def behaviour(run, steps):
    if run['a_done']:
        return 'finished_a'
    return 'never_closed' if longest_closed(steps[run['scripted_steps'] + 1:]) < GRASP else 'closed_neither'


def tables(runs, census, load, old=EXCEPTION):
    """runs: retreat RUNS rows; census: census RUNS rows; load(run id) -> the run's step records."""
    native = {r['sw']: r for r in census if r['arm'] == 'native' and r['phase'] > 0}
    amp = {r['sw']: r for r in census if r['arm'] == 'amp2'}
    start = collections.defaultdict(list); gap0 = {}
    for r in census:
        if r['arm'] == 'native' and r['phase'] == 0:
            start[r['pair']].append(r); gap0[(r['pair'], r['init'])] = r['gap_first']
    home = [r for r in runs if r['arm'] == 'home']; release = {r['sw']: r for r in runs if r['arm'] == 'release'}

    def arms(rows):
        rows = list(rows)
        return dict(home=count(rows), release=count(release[r['sw']] for r in rows), native=count(native[r['sw']] for r in rows),
                    amp2=count(amp[r['sw']] for r in rows))

    odd = [r for r in home if r['task'] == old]; rest = [r for r in home if r['task'] != old]
    by_pair = {}
    for pair in sorted({r['pair'] for r in odd}):
        by_pair[pair] = dict(arms(r for r in odd if r['pair'] == pair), start=count(start[pair]))
    worse = collections.Counter(r['pair'] for r in home if native[r['sw']]['b_done'] and not r['b_done'])
    acts = collections.Counter(behaviour(r, load(r['id'])) for r in odd if not r['b_done'])
    reached = [dict(r, gamma=r['gap_first'] / gap0[(r['pair'], r['init'])]) for r in home
               if r['reached'] and gap0.get((r['pair'], r['init']))]

    def separation(rows):
        rows = list(rows)
        return dict(n=len(rows), failures=sum(not r['b_done'] for r in rows),
                    auc=auc([r['gamma'] for r in rows if r['b_done']], [r['gamma'] for r in rows if not r['b_done']]))

    def three(rows):
        rows = list(rows)
        return dict(all=separation(rows), same_object=separation(r for r in rows if r['covariates']['same_object']),
                    different_object=separation(r for r in rows if not r['covariates']['same_object']))

    return dict(old_task=old, by_old_task={t: arms(r for r in home if r['task'] == t) for t in sorted({r['task'] for r in home})},
                exception=dict(arms(odd), by_pair=by_pair, by_phase={str(f): count(r for r in odd if r['phase'] == f) for f in (.5, .75)},
                               failures=sum(not r['b_done'] for r in odd), behaviour=dict(acts),
                               b_object_moved=sum((r['moved_b'] or 0) > .01 for r in odd if not r['b_done'])),
                without=dict(arms(rest), same_object=arms(r for r in rest if r['covariates']['same_object']),
                             different_object=arms(r for r in rest if not r['covariates']['same_object']),
                             failures_by_pair=collections.Counter(r['pair'] for r in rest if not r['b_done']).most_common(8)),
                same_object=arms(r for r in home if r['covariates']['same_object']),
                different_object=dict(arms(r for r in home if not r['covariates']['same_object']),
                                      pairs=len({r['pair'] for r in home if not r['covariates']['same_object']})),
                home_failures=dict(n=sum(not r['b_done'] for r in home), exception=sum(not r['b_done'] for r in odd)),
                worse=dict(n=sum(worse.values()), by_pair=worse.most_common(6)),
                outcomes={a: dict(collections.Counter('B' if r['b_done'] else 'A' if r['a_done'] else 'neither'
                                                      for r in runs if r['arm'] == a)) for a in ('home', 'release')},
                gamma_auc=dict(three(reached), without=three(r for r in reached if r['task'] != old)))


def state_check(root):
    rows = json.loads((Path(root)/'STATE_CHECK.json').read_text())['rows']
    differs = [r for r in rows if r['recorded'] != r['reevaluated']]
    return dict(recorded_true=sum(bool(r['recorded']) for r in rows if r['which'] != 'A'), differs=len(differs),
                by_goal=dict(collections.Counter(r['which'] for r in differs)),
                by_arm=dict(collections.Counter(r['unit'].rsplit('__', 1)[-1] for r in differs)))


def main(root, census_root, old=EXCEPTION):
    root = Path(root); census_root = Path(census_root)
    out = tables(lines(root/'RUNS.jsonl.gz'), lines(census_root/'RUNS.jsonl.gz'),
                 lambda run_id: lines(root/'branch'/run_id/'STEPS.jsonl.gz'), int(old))
    out['state_check'] = dict(retreat=state_check(root), census=state_check(census_root))
    return out


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    print(json.dumps(main(*sys.argv[1:]), indent=1, ensure_ascii=False))
