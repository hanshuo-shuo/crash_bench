"""Post-hoc read of the task-pair census: after the instruction is switched, does the arm go to where the new task B starts its work?

Read-only, CPU, standard library only. Written after the census results were seen (2026-11-14), so everything here is descriptive
and must be labelled post hoc wherever it is quoted; the frozen readings are census.py summarize -> SUMMARY.json.

Why: amp2 turns 286 "finishes the old task" states into "finishes neither". That was read as "the instruction was heard, but B
cannot be done from this state". The alternative is that the amplification only disrupts A. The runs store the end-effector
position at every step, so the two can be told apart by where the arm goes.

Definitions (fixed before the numbers were looked at):
- B's work point of a state (pair A->B, init) is the end-effector position at the first step with the gripper closed (aperture below
  census.CLOSED) in the f = 0 native run of the same pair and init, when that run completed B. It is the same layout, with B given
  from the start. States without such a reference are dropped and counted.
- A run reaches B when, at any step after the switch, the end-effector is within RADIUS of B's work point.
- A run is stalled when the end-effector path over its last TAIL steps is shorter than STILL.
- Only different-object pairs are read (for same-object pairs A and B work at the same place), and only states whose switch-step
  position is farther than FAR from B's work point.

Also counts, over all runs, the steps whose action leaves [-1, 1], split into arm dimensions (0-5) and the gripper dimension (6).

Usage: python census_reach_posthoc.py CENSUS_FORMAL_ROOT        prints JSON
"""
import collections
import gzip
import json
import math
import sys
from pathlib import Path

CLOSED = .05            # same threshold as census.CLOSED
RADIUS = .05            # metres; RADII are reported side by side
RADII = (.03, .05, .08)
FAR = .10
TAIL = 50
STILL = .01


def aperture(step):
    return step['gripper'][0] - step['gripper'][1]


def work_point(steps):
    """End-effector position at the first closed-gripper step, or None."""
    for s in steps:
        if aperture(s) < CLOSED:
            return s['eef']
    return None


def closest(steps, point):
    return min(math.dist(s['eef'], point) for s in steps)


def stalled(steps):
    tail = steps[-(TAIL + 1):]
    return len(tail) > TAIL and sum(math.dist(a['eef'], b['eef']) for a, b in zip(tail, tail[1:])) < STILL


def outside(steps):
    """(policy steps, steps with an arm dimension outside [-1, 1], steps with only the gripper dimension outside, largest |value|)."""
    n = arm = grip = 0; peak = 0.
    for s in steps:
        a = s['action']
        if a is None:
            continue
        n += 1; peak = max(peak, max(abs(x) for x in a))
        if any(abs(x) > 1 for x in a[:6]):
            arm += 1
        elif abs(a[6]) > 1:
            grip += 1
    return n, arm, grip, peak


def outcome(run):
    return 'B' if run['b_done'] else ('A' if run['a_done'] else 'neither')


def read(root, run_id):
    with gzip.open(Path(root)/'branch'/run_id/'STEPS.jsonl.gz', 'rt') as stream:
        return [json.loads(x) for x in stream]


def tables(rows, load):
    """rows: RUNS.jsonl rows; load(run_id) -> list of step records (the first is the switch step)."""
    twins = collections.defaultdict(dict); points = {}; dropped = collections.Counter(); spec = collections.Counter(); peak = 0.
    traces = {}
    for r in rows:
        steps = load(r['id']); n, arm, grip, top = outside(steps)
        spec['steps'] += n; spec['arm_outside'] += arm; spec['gripper_only_outside'] += grip; peak = max(peak, top)
        if r['phase'] == 0:
            if r['b_done']:
                point = work_point(steps)
                if point is None:
                    dropped['reference never closes the gripper'] += 1
                else:
                    points[(r['pair'], r['init'])] = point
        else:
            twins[(r['pair'], r['phase'], r['init'])][r['arm']] = r; traces[r['id']] = steps
    cells = collections.defaultdict(lambda: collections.Counter()); read_states = 0
    for (pair, phase, init), v in twins.items():
        if set(v) < {'native', 'amp2'}:
            continue
        if v['native']['covariates']['same_object']:
            dropped['same-object pair'] += 1; continue
        point = points.get((pair, init))
        if point is None:
            dropped['no B reference from the start'] += 1; continue
        if math.dist(traces[v['native']['id']][0]['eef'], point) <= FAR:
            dropped['switch step already near B work point'] += 1; continue
        read_states += 1
        flow = '%s->%s' % (outcome(v['native']), outcome(v['amp2']))
        for arm, key in (('native', 'native ' + outcome(v['native'])), ('amp2', 'amp2 ' + flow)):
            steps = traces[v[arm]['id']]; c = cells[key]; near = closest(steps, point)
            c['runs'] += 1; c['stalled'] += stalled(steps)
            for radius in RADII:
                c['reach_%d' % round(radius * 100)] += near <= radius
            c['reach_and_stalled'] += near <= RADIUS and stalled(steps)
    keys = ['runs'] + ['reach_%d' % round(x * 100) for x in RADII] + ['stalled', 'reach_and_stalled']
    return dict(states_read=read_states, dropped=dict(dropped), radius_cm=[round(x * 100) for x in RADII], far_cm=round(FAR * 100),
                tail_steps=TAIL, still_cm=STILL * 100, by_outcome={k: {x: c[x] for x in keys} for k, c in sorted(cells.items())},
                actions_outside_spec=dict(spec, largest_abs=peak))


def main(root):
    with gzip.open(Path(root)/'RUNS.jsonl.gz', 'rt') as stream:
        rows = [json.loads(x) for x in stream]
    print(json.dumps(tables(rows, lambda run_id: read(root, run_id)), indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
