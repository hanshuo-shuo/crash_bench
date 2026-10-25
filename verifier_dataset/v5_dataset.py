"""V5 S2: the object/II/task1 dataset for the A-versus-B pilot.

Protocol: docs/generative_verifier/V5_S2_PROTOCOL.md. plan() picks the images and rule rows
before any rendering: training images from the natural and counterfactual layouts of training
sources in their natural rule mix; validation and test from strict counterfactual twins balanced
within rule and progress bin; small natural-layout diagnostics for validation and test. The
render run (v5_render, two overviews) is followed by build(), which checks every label and fact
against the render-time geometry, writes dataset.jsonl and freezes the test split.

ROOT --build runs the CPU build after all render shards finish.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path
from .inventory import read, sha, write
from . import v5_geometry as g
from . import v5_layouts as L
from .v5_trace import rows_from_trace

SCHEMA = 'V5-object-v1'
VIEWS = ('overview_a', 'overview_b')
STEPS_PER_BIN = 2
RULES_PER_IMAGE = 2
MIN_TWIN_IMAGES = 150
MIN_ROWS = 300
DIGIT = {1: '0', 0: '1'}
VERDICT = {1: 'infeasible', 0: 'feasible'}
SPLITS = ('train', 'validation', 'validation_natural', 'test', 'test_natural')


def kept(row):
    """Usable for B's facts, and a feasible row needs its witness (the rest of the tape never crosses)."""
    return L.usable(row) and (row['label'] == 1 or row.get('witnessed', False))


def natural_rows(trace, robot):
    """C0 natural-layout rows with B's robot-side facts; the robot base is fixed, so its S1a position applies."""
    for snap in trace['checkpoints']:
        snap['robot'] = list(robot)
    return rows_from_trace(trace)


def constraint_text(key):
    a, b = key.split(':', 1)[1].split('|')
    return ('A new rule is added now, part-way through the task: do not move the object being picked up across the '
            'straight line through the %s and the %s. The line extends past both objects in both directions. '
            'The robot arm itself may pass over the line.' % (g.phrase(a), g.phrase(b)))


def facts(row):
    a, b = row['rule'].split(':', 1)[1].split('|')
    yn = lambda v: 'yes' if v else 'no'
    return dict(line='%s | %s' % (g.phrase(a), g.phrase(b)), target_robot_side=yn(row['target_robot_side']),
                goal_robot_side=yn(row['goal_robot_side']), separated=yn(row['label']))


def facts_text(f):
    return ('Line: %s\nTarget on robot side: %s\nGoal on robot side: %s\nSeparated: %s'
            % (f['line'], f['target_robot_side'], f['goal_robot_side'], f['separated']))


def sample_natural(rows, salt):
    """Natural rule mix: per image source and progress bin, STEPS_PER_BIN checkpoints, then RULES_PER_IMAGE rules, all by digest."""
    images = defaultdict(list)
    for r in rows:
        images[(r['source'], r['unit'])].append(r)
    bins = defaultdict(list)
    for (source, step), rs in images.items():
        bins[(source, rs[0]['time_bin'])].append(step)
    chosen = []
    for (source, _), steps in sorted(bins.items()):
        for step in sorted(steps, key=lambda s: g.digest(source, s, salt))[:STEPS_PER_BIN]:
            rs = sorted(images[(source, step)], key=lambda r: g.digest(source, step, r['rule'], salt))
            chosen.append(((source, step), rs[:RULES_PER_IMAGE]))
    return chosen


def balanced_twins(rows, salt):
    """Strict resting twins, then within each (task, rule, progress-bin) group keep equal numbers of each label by digest."""
    pool = [r for twin in L.strict_twins(rows).values() for r in twin]
    groups = defaultdict(lambda: ([], []))
    for r in pool:
        groups[g.time_key(r)][r['label']].append(r)
    out = []
    for zero, one in groups.values():
        n = min(len(zero), len(one))
        order = lambda r: g.digest(r['source'], r['unit'], r['rule'], salt)
        out += sorted(zero, key=order)[:n] + sorted(one, key=order)[:n]
    images = defaultdict(list)
    for r in out:
        images[(r['source'], r['unit'])].append(r)
    return sorted(images.items())


def plan(s1_root, c0_root):
    """Images to render (renderer units) and the rows each carries; nothing here depends on rendering."""
    split = L.frozen_split()
    entries = read(s1_root/'LAYOUTS.json')
    layout_case = {'%s#%d' % (e['spec']['source'], e['plan']['layout']): str(s1_root/('layout_%03d' % i)) for i, e in enumerate(entries)}
    spec = {e['spec']['source']: e['spec'] for e in entries}
    robots = {tuple(read(Path(c)/'LAYOUT.json')['robot']) for c in layout_case.values()}
    if len(robots) != 1:
        raise RuntimeError('Robot base differs across layouts')
    robot = next(iter(robots))
    with gzip.open(s1_root/'LAYOUT_ROWS.jsonl.gz', 'rt') as stream:
        counterfactual = [r for r in map(json.loads, stream) if kept(r)]
    c0 = {c['source']: i for i, c in enumerate(read(c0_root/'CASES.json'))}
    natural = []
    for source in sorted(spec):
        with gzip.open(c0_root/('case_%03d' % c0[source])/'TRACE.json.gz', 'rt') as stream:
            rows = natural_rows(json.load(stream), robot)
        with gzip.open(c0_root/('case_%03d' % c0[source])/'ROWS.jsonl.gz', 'rt') as stream:
            old = {(r['unit'], r['rule']): r['label'] for r in map(json.loads, stream)}
        if any(old[(r['unit'], r['rule'])] != r['label'] for r in rows):
            raise RuntimeError('Natural labels differ from C0: ' + source)
        natural += [dict(r, base_source=source, layout=None) for r in rows if kept(r)]
    picks = []
    for part in ('train', 'validation', 'test'):
        nat = [r for r in natural if split[r['base_source']] == part]
        if part == 'train':
            cf = [r for r in counterfactual if split[r['base_source']] == part]
            picks += [('train', k, rs) for k, rs in sample_natural(nat + cf, 'v5-s2-train')]
        else:
            cf = [r for r in counterfactual if split[r['base_source']] == part]
            picks += [(part, k, rs) for k, rs in balanced_twins(cf, 'v5-s2-' + part)]
            picks += [(part + '_natural', k, rs) for k, rs in sample_natural(nat, 'v5-s2-' + part + '-natural')]
    candidates = defaultdict(set)
    for r in natural + counterfactual:
        candidates[(r['source'], r['unit'])].add(r['rule'])
    units = []
    for part, (source, step), rs in sorted(picks, key=lambda p: (SPLITS.index(p[0]), p[1])):
        base = rs[0]['base_source']
        unit = dict(review_id='i%04d' % len(units), split=part, source=source, base_source=base, layout=rs[0]['layout'],
                    unit=step, task=rs[0]['task'], v4_case=spec[base]['v4_case'], student_task=spec[base]['student_task'],
                    views=list(VIEWS), diagnostic=False, candidate_rules=sorted(candidates[(source, step)]),
                    rules=[dict(key=r['rule'], label=r['label'], clearance=r['clearance'], lift=r['lift'], time_bin=r['time_bin'],
                                target_robot_side=r['target_robot_side'], goal_robot_side=r['goal_robot_side'],
                                robot_clearance=r['robot_clearance']) for r in rs])
        if source in layout_case:
            unit['layout_case'] = layout_case[source]
        units.append(unit)
    return units


def lookup(train, test, field, key=g.time_key):
    """Best rule-and-progress lookup learned on train rows, scored on test rows (fallback: rule, then global majority)."""
    tables = [defaultdict(Counter), defaultdict(Counter)]
    for r in train:
        tables[0][key(r)][r[field]] += 1; tables[1][g.text_key(r)][r[field]] += 1
    overall = Counter(r[field] for r in train).most_common(1)[0][0]
    def guess(r):
        for table, k in ((tables[0], key(r)), (tables[1], g.text_key(r))):
            if table.get(k):
                return table[k].most_common(1)[0][0]
        return overall
    return sum(guess(r) == r[field] for r in test) / len(test) if test else None


def readouts(rows):
    """Preregistered S2 readouts over flat rows (task, rule, time_bin, label, facts, split, image)."""
    out, train = {}, [r for r in rows if r['split'] == 'train']
    for part in SPLITS:
        rs = [r for r in rows if r['split'] == part]
        images = defaultdict(set)
        for r in rs:
            images[r['image']].add(r['label'])
        out[part] = dict(rows=len(rs), images=len(images), twin_images=sum(v == {0, 1} for v in images.values()),
                         sources=len({r['base_source'] for r in rs}),
                         separated_fraction=sum(r['label'] for r in rs) / len(rs) if rs else None,
                         target_robot_side_fraction=sum(r['target_robot_side'] for r in rs) / len(rs) if rs else None,
                         goal_robot_side_fraction=sum(r['goal_robot_side'] for r in rs) / len(rs) if rs else None,
                         counterfactual_fraction=sum(r['layout'] is not None for r in rs) / len(rs) if rs else None,
                         lookup_from_train=dict((f, lookup(train, rs, f)) for f in ('label', 'target_robot_side', 'goal_robot_side'))
                         if rs and train else None)
    out['train']['text_time_bound'] = g.text_bound(train, g.time_key)
    ok = all(out[p]['twin_images'] >= MIN_TWIN_IMAGES and out[p]['rows'] >= MIN_ROWS and out[p]['sources'] >= L.S1_MIN_SOURCES
             for p in ('validation', 'test'))
    ok = ok and all(abs(out[p]['lookup_from_train']['label'] - .5) < 1e-9 for p in ('validation', 'test'))
    out['reading'] = 's2_dataset_ready' if ok else 's2_dataset_insufficient'
    return out


def build(root):
    """Check each planned row against render-time geometry, write dataset.jsonl, freeze test."""
    campaign = read(root/'CAMPAIGN.json')
    for k in range(campaign['shards']):
        if not read(root/('SHARD_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Render shard %d incomplete' % k)
    units = read(root/'UNITS.json'); rows, flat, dropped = [], [], []
    for u in units:
        folder = root/'units'/u['review_id']
        geometry = read(folder/'RENDER_GEOMETRY.json'); at = {x['key']: x for x in geometry['labels']}
        images = [dict(camera=v, path='units/%s/%s.png' % (u['review_id'], v), sha256=sha(folder/(v + '.png')), width=448, height=448)
                  for v in u['views']]
        for r in u['rules']:
            x = at[r['key']]
            if not (x['agrees'] and x['facts_agree']):
                dropped.append(dict(image=u['review_id'], rule=r['key'])); continue
            row = dict(task=u['task'], rule=r['key'], time_bin=r['time_bin'], label=r['label'], target_robot_side=r['target_robot_side'],
                       goal_robot_side=r['goal_robot_side'], split=u['split'], image=u['review_id'], base_source=u['base_source'],
                       layout=u['layout'])
            flat.append(row); f = facts(dict(r, rule=r['key']))
            rows.append(dict(schema_version=SCHEMA, sample_id='v5o_%05d' % len(rows), split=u['split'],
                student=dict(images=images, task=u['student_task'], constraints=constraint_text(r['key'])),
                target=dict(verdict=VERDICT[r['label']], digit=DIGIT[r['label']], facts=f, rationale=facts_text(f),
                            scope='conditional_on_stated_rule'),
                metadata=dict(source_id=u['base_source'], image_source=u['source'], layout=u['layout'], checkpoint=u['unit'],
                              image_id=u['review_id'], rule=r['key'], clearance=r['clearance'], lift=r['lift'], time_bin=r['time_bin'],
                              robot_clearance=r['robot_clearance'], render_clearance=x['clearance_at_render'],
                              pixel_prescreen=geometry['pixel_prescreen'], layout_kind='natural' if u['layout'] is None else 'counterfactual')))
    with (root/'dataset.jsonl').open('w') as stream:
        stream.writelines(json.dumps(r, sort_keys=True) + '\n' for r in rows)
    summary = dict(readouts(flat), dropped=dropped, schema=SCHEMA, views=list(VIEWS))
    write(root/'S2_SUMMARY.json', summary)
    test = [r for r in rows if r['split'] in ('test', 'test_natural')]
    write(root/'TEST_FREEZE.json', dict(rows=len(test), sample_ids=[r['sample_id'] for r in test],
        rows_sha256=sha_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in test)),
        images={i['path']: i['sha256'] for r in test for i in r['student']['images']},
        rule='Scored once per selected checkpoint after all training and selection are frozen; never used for selection'))
    write(root/'S2_COMPLETE.json', dict(passed=True, reading=summary['reading'], rows=len(rows), dropped=len(dropped),
        api_calls=0, code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID')))
    write(root/'HASHES.json', {n: sha(root/n) for n in ('UNITS.json', 'dataset.jsonl', 'S2_SUMMARY.json', 'TEST_FREEZE.json', 'S2_COMPLETE.json')})


def sha_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); p.add_argument('--build', action='store_true')
    a = p.parse_args()
    try:
        build(a.root)
    except BaseException as error:
        write(a.root/'STOP_build.json', dict(error=type(error).__name__, reason=str(error)))
        raise
