"""Post-hoc tables of the task-pair census (docs/generative_verifier/HIL_METHOD_PLAN.md section 2.1). Read-only, CPU, standard
library only. These are descriptive re-reads of the stored runs, written after the results were seen: they are not the frozen
readings (those are census.py summarize -> SUMMARY.json) and must be labelled post hoc wherever they are quoted.

A state is (pair, init, phase) at phase .5 or .75; each has a native run and an amp2 run from the same replayed prefix and seed.
Outcome of a run: B (the new task's goal became true), A (not B, the old task was completed) or neither.

Usage: python census_posthoc.py CENSUS_FORMAL_ROOT        prints JSON
"""
import collections
import gzip
import json
import statistics
import sys
from pathlib import Path


def outcome(run):
    return 'B' if run['b_done'] else ('A' if run['a_done'] else 'neither')


def auc(positive, negative):
    """Probability that a positive value exceeds a negative one (ties count one half); None when a side is empty."""
    if not positive or not negative:
        return None
    return sum((p > n) + .5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def tables(rows):
    twins = collections.defaultdict(dict); start_gap = {}; start = [0, 0]
    for r in rows:
        if r['phase'] == 0:
            start_gap[(r['pair'], r['init'])] = r['gap_first']; start[0] += 1; start[1] += bool(r['b_done'])
        else:
            twins[(r['pair'], r['phase'], r['init'])][r['arm']] = r
    states = {k: v for k, v in twins.items() if set(v) >= {'native', 'amp2'}}
    names = ('A', 'neither', 'B')
    flow = collections.Counter((outcome(v['native']), outcome(v['amp2'])) for v in states.values())
    totals = dict(native={n: sum(flow[(n, m)] for m in names) for n in names}, amp2={m: sum(flow[(n, m)] for n in names) for m in names})

    def tally(key):
        out = collections.defaultdict(lambda: collections.Counter())
        for k, v in states.items():
            c = out[key(k, v)]; c['states'] += 1
            for arm in ('native', 'amp2'):
                c[arm + '_' + outcome(v[arm])] += 1
        keys = ['states'] + [arm + '_' + n for arm in ('native', 'amp2') for n in names]
        return {str(k): {x: c[x] for x in keys} for k, c in sorted(out.items(), key=lambda x: str(x[0]))}

    pairs = collections.defaultdict(lambda: [0, 0, 0])
    for (pair, _, _), v in states.items():
        p = pairs[pair]; p[0] += 1; p[1] += bool(v['native']['b_done']); p[2] += bool(v['amp2']['b_done'])
    # Integer comparisons: a gain of at least .20 is 5 * (amp2 - native) >= n; a change under .10 is 10 * |amp2 - native| < n.
    up = sorted(k for k, (n, a, b) in pairs.items() if 5 * (b - a) >= n)
    down = sorted(k for k, (n, a, b) in pairs.items() if 5 * (a - b) >= n)
    flat = sorted(k for k, (n, a, b) in pairs.items() if 10 * abs(b - a) < n)

    def gammas(select):
        won, lost = [], []
        for (pair, _, init), v in states.items():
            base = start_gap.get((pair, init))
            if base and select(v['native']):
                (won if v['native']['b_done'] else lost).append(v['native']['gap_first'] / base)
        return won, lost

    cells = collections.defaultdict(list)
    for (pair, phase, init), v in states.items():
        cells[(pair, phase)].append((v, start_gap.get((pair, init))))
    failing = []
    for members in cells.values():
        if 2 * sum(bool(v['native']['b_done']) for v, _ in members) <= len(members):   # native rate <= .5, the census failure cell
            g = [v['native']['gap_first'] / b for v, b in members if b]
            if g:
                failing.append((statistics.median(g), len(members), sum(bool(v['native']['b_done']) for v, _ in members),
                                sum(bool(v['amp2']['b_done']) for v, _ in members)))
    failing.sort(); third = len(failing) // 3
    bands = {name: dict(gamma_from=round(part[0][0], 3), gamma_to=round(part[-1][0], 3), states=sum(x[1] for x in part),
                        native_b=sum(x[2] for x in part), amp2_b=sum(x[3] for x in part))
             for name, part in (('faintest', failing[:third]), ('middle', failing[third:2 * third]), ('least_faint', failing[2 * third:]))}
    return dict(
        mid_task_states=len(states), start=dict(states=start[0], b_done=start[1]),
        outcome_totals=totals, flow_native_to_amp2={'%s->%s' % k: flow[k] for k in sorted(flow)},
        by_old_task=tally(lambda k, v: v['native']['task']), by_new_task=tally(lambda k, v: v['native']['other']),
        by_same_object=tally(lambda k, v: 'same_object' if v['native']['covariates']['same_object'] else 'different_object'),
        pairs=dict(count=len(pairs), gain_at_least_20_points=up, loss_at_least_20_points=down, change_under_10_points=len(flat),
                   never_switch_native=sum(a == 0 for _, a, _ in pairs.values()),
                   counts={k: dict(states=n, native_b=a, amp2_b=b) for k, (n, a, b) in sorted(pairs.items())}),
        gamma_auc=dict(all=auc(*gammas(lambda r: True)), same_object=auc(*gammas(lambda r: r['covariates']['same_object'])),
                       different_object=auc(*gammas(lambda r: not r['covariates']['same_object']))),
        failure_cells_by_gamma_third=bands)


def main(root):
    with gzip.open(Path(root)/'RUNS.jsonl.gz', 'rt') as stream:
        rows = [json.loads(x) for x in stream]
    print(json.dumps(tables(rows), indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
