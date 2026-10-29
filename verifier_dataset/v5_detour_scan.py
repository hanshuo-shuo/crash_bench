"""V5 detour HIL, CPU scan: can the existing HIL records supply keep-out-zone units without new layouts?

Read-only (docs/generative_verifier/V5_DETOUR_PROTOCOL.md, section 2). For each HIL state, scene
object and radius, the zone is the object's xy box at the checkpoint t expanded by the radius (the
v5_geometry zone template). A cell is usable when the pudding at t and the basket box lie outside the
zone with margin, the HIL C0 run (no rule) carries the pudding centre into it, and the source tape's
own continuation from t in the same S1a layout (a valid, successful replay) stays outside: that
continuation is the feasibility witness. No execution, API call or test-source read.

python -m verifier_dataset.v5_detour_scan HIL_ROOT OUT_JSON
"""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
from . import v5_geometry as g

RADII = (.03, .05, .075, .10, .125, .15)
CONDITIONS = ('C0', 'C1', 'C2', 'C3')
ENTRY = .01
START_GAP = .03
GOAL_GAP = .03
WITNESS_GAP = .01
DEEP = .03


def keep_out(obj, r):
    """The v5_geometry zone template: the object's xy box grown by r on every side."""
    return ([obj['lo'][0] - r, obj['lo'][1] - r], [obj['hi'][0] + r, obj['hi'][1] + r])


def depth(p, zone):
    """> 0 inside the zone (distance to its edge), < 0 outside (the gap)."""
    return g.zone_signed(g.point(p), zone)


def entry(trace, zone, t):
    """First step after t where the pudding centre is more than ENTRY inside the zone."""
    deep = max(depth(x['material'], zone) for x in trace[1:])
    first = next((x for x in trace[1:] if depth(x['material'], zone) > ENTRY), None)
    return dict(max_depth=deep, entered=first is not None, completed=bool(trace[-1]['done']),
                after=None if first is None else first['step'] - t,
                held=None if first is None else bool(first['command'] is not None and first['command'] > 0 and first['lift'] > .02))


def cells(root):
    plan = json.loads((root/'PLAN.json').read_text()); cache = {}; out = []
    for index, s in enumerate(plan['states']):
        case = Path(s['layout_case'])
        if case not in cache:
            with gzip.open(case/'TRACE.json.gz', 'rt') as stream:
                cache[case] = (json.load(stream), json.loads((case/'LAYOUT.json').read_text()))
        trace, layout = cache[case]; t = s['checkpoint']
        if not (layout['valid'] and layout['success']):
            raise RuntimeError('HIL state on an invalid layout')
        snap = [c for c in trace['checkpoints'] if c['step'] == t]
        if len(snap) != 1:
            raise RuntimeError('No layout snapshot at the checkpoint')
        objects, spec = snap[0]['objects'], s['spec']
        excluded = {spec['target']} | g.goal_objects(objects, spec['goal'][2])
        anchors = set(g.anchors(objects, spec['student_task'], excluded))
        runs = {}
        for c in CONDITIONS:
            with gzip.open(root/('state_%03d' % index)/c/'STEPS.jsonl.gz', 'rt') as stream:
                runs[c] = [json.loads(x) for x in stream]
        start = runs['C0'][0]['material']
        if max(abs(a - b) for a, b in zip(start, snap[0]['material'])) > 1e-6:
            raise RuntimeError('C0 start differs from the layout snapshot')
        basket = (objects['basket_1']['lo'], objects['basket_1']['hi'])
        witness = [p[1:] for p in trace['path'] if p[0] > t]
        for name in sorted(n for n, o in objects.items() if o['active'] and n not in excluded):
            for r in RADII:
                zone = keep_out(objects[name], r)
                out.append(dict(state=s['state'], split=s['split'], base_source=s['base_source'], checkpoint=t,
                                phase=s['phase'], time_bin=s['time_bin'], object=name, anchor=name in anchors,
                                moved=name in layout['moved'], radius=r, start_gap=-depth(start, zone),
                                basket_gap=-g.zone_signed(basket, zone), witness_gap=min(-depth(p, zone) for p in witness),
                                runs={c: entry(runs[c], zone, t) for c in CONDITIONS}))
    return plan, out


def usable(c):
    return (c['runs']['C0']['entered'] and c['start_gap'] >= START_GAP and c['basket_gap'] >= GOAL_GAP
            and c['witness_gap'] >= WITNESS_GAP)


def summary(plan, rows):
    out = dict(states=len(plan['states']), by_radius={})
    for r in RADII:
        rs = [c for c in rows if c['radius'] == r]
        steps = [('C0_enters', lambda c: c['runs']['C0']['entered']),
                 ('start_outside', lambda c: c['start_gap'] >= START_GAP),
                 ('basket_outside', lambda c: c['basket_gap'] >= GOAL_GAP),
                 ('witness_outside', lambda c: c['witness_gap'] >= WITNESS_GAP),
                 ('deep_entry', lambda c: c['runs']['C0']['max_depth'] >= DEEP)]
        funnel = {}
        for scope, keep in (('anchors', lambda c: c['anchor']), ('all_objects', lambda c: True)):
            kept = [c for c in rs if keep(c)]; block = {}
            for name, test in steps:
                kept = [c for c in kept if test(c)]
                block[name] = dict(cells=len(kept), states=len({c['state'] for c in kept}))
            funnel[scope] = block
        entered = [c for c in rs if c['anchor'] and c['runs']['C0']['entered']]
        funnel['C0_entered_cells'] = [dict(state=c['state'], object=c['object'], moved=c['moved'], depth=round(c['runs']['C0']['max_depth'], 3),
                                           start_gap=round(c['start_gap'], 3), basket_gap=round(c['basket_gap'], 3),
                                           witness_gap=round(c['witness_gap'], 3), held=c['runs']['C0']['held'],
                                           other_runs_entered=sum(c['runs'][k]['entered'] for k in CONDITIONS[1:]))
                                      for c in entered] if r in (.05, .10) else len(entered)
        out['by_radius']['%.3f' % r] = funnel
    best = max(out['by_radius'].values(), key=lambda f: f['all_objects']['witness_outside']['states'])
    out['most_usable_states'] = best['all_objects']['witness_outside']['states']
    out['objects'] = dict(Counter(c['object'] for c in rows if c['radius'] == RADII[0]))
    return out


if __name__ == '__main__':
    root, target = Path(sys.argv[1]), Path(sys.argv[2])
    plan, rows = cells(root)
    target.write_text(json.dumps(dict(hil_root=str(root), radii=RADII, margins=dict(entry=ENTRY, start=START_GAP, goal=GOAL_GAP,
                                      witness=WITNESS_GAP, deep=DEEP), summary=summary(plan, rows), cells=rows)))
    print(json.dumps(summary(plan, rows)['by_radius']['0.100']['anchors'], indent=1))
