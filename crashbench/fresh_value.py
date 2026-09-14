"""Frozen choices and source-level scoring for the prospective fresh-reset panel."""
from collections import defaultdict

import numpy as np

from crashbench.repeat_value import (
    freeze_choice, measure_cell, source_average, terminal_sample,
)

PHASE_REPEATS = {'A': (0, 1), 'B': (2, 3), 'C': (4, 5)}
MODES = ('real_full', 'real_one', 'pseudo_one', 'pseudo_one_swapped')
HORIZONS = (220, 440)


def validate_anchors(anchors, expected_sources=12):
    if len(anchors) != expected_sources * 3:
        raise ValueError('incomplete source/condition denominator')
    if len({a['episode_id'] for a in anchors}) != len(anchors):
        raise ValueError('duplicate episode identity')
    groups = defaultdict(list)
    for a in anchors:
        if a['role'] != 'fresh_development':
            raise ValueError('fresh panel crossed a historical split')
        groups[a['source']].append(a['condition'])
    if len(groups) != expected_sources or any(
        sorted(v) != ['glass', 'noglass', 'offpath'] for v in groups.values()
    ):
        raise ValueError('source identity or condition coverage changed')


def validate_records(anchors, records, phase):
    expected = {(a['episode_id'], r, o) for a in anchors if a['triggered']
                for r in PHASE_REPEATS[phase] for o in (0, 1)}
    actual = [(r['episode_id'], r['repeat'], r['option']) for r in records]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('incomplete, duplicate, or crossed execution repeats')
    lookup = {a['episode_id']: a for a in anchors}
    for r in records:
        a = lookup[r['episode_id']]
        if r['phase'] != phase or any(r[k] != a[k] for k in (
            'source', 'condition', 'role', 'bundle_sha256', 'bundle_id'
        )):
            raise ValueError('execution identity or bundle changed')


def cells_from_records(anchors, blocks):
    cells = {}
    for a in anchors:
        c = dict(id=a['episode_id'], source=a['source'], condition=a['condition'],
                 role=a['role'], samples=defaultdict(list))
        if not a['triggered']:
            v = a['horizons']['440']
            c['shared'] = terminal_sample(v['success'], v['accident'], v['steps'], -1)
        cells[c['id']] = c
    for phase, records in blocks.items():
        validate_records(anchors, records, phase)
        for r in records:
            v = r['horizons']['440']
            cells[r['episode_id']]['samples'][phase, r['option']].append(
                terminal_sample(v['success'], v['accident'], v['steps'], r['repeat']))
    for c in cells.values():
        for values in c['samples'].values():
            values.sort(key=lambda s: s['repeat'])
    return list(cells.values())


def freeze_a(anchors, records, historical_info):
    """Only accepts complete A records; never fits a new deployable gate."""
    validate_records(anchors, records, 'A')
    cells = cells_from_records(anchors, {'A': records})
    if historical_info['BenefitGate_threshold'] is not None:
        raise ValueError('expected historical all-Base gate; do not invent predictions')
    fixed = {
        'Base': {a['episode_id']: 0 for a in anchors},
        'AlwaysDetour': {a['episode_id']: int(a['triggered']) for a in anchors},
        'BenefitGate': {a['episode_id']: 0 for a in anchors},
        'RiskDetour': {a['episode_id']: int(a['triggered'] and
            a['risk'] > historical_info['RiskDetour_threshold']) for a in anchors},
    }
    choices = []
    for h in HORIZONS:
        for mode in MODES:
            choices.append(dict(method=mode, mode=mode, selection_horizon=h,
                                choices=freeze_choice(cells, mode, h)))
        for method, mapping in fixed.items():
            choices.append(dict(method=method, mode='real_full', selection_horizon=h,
                                choices=mapping))
    return choices


def score_choices(anchors, blocks, choices):
    cells = cells_from_records(anchors, blocks)
    rows = []
    for f in choices:
        if set(f['choices']) != {c['id'] for c in cells}:
            raise ValueError('frozen choices do not cover the complete panel')
        for h in HORIZONS:
            for c in cells:
                row = {k: c[k] for k in ('id', 'source', 'condition', 'role')}
                row.update(method=f['method'], selection_horizon=f['selection_horizon'],
                           evaluation_horizon=h, shared='shared' in c,
                           choice=f['choices'][c['id']])
                for phase in blocks:
                    row.update({phase+'_'+k: v for k, v in measure_cell(
                        c, phase, f['mode'], h, row['choice']).items()})
                rows.append(row)
    return rows


def summarize_ab(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['method'], r['selection_horizon'], r['evaluation_horizon']].append(r)
    result = []
    for (method, sh, eh), group in groups.items():
        keys = [k for k in group[0] if k.startswith(('A_', 'B_'))]
        sources = source_average(group, keys)
        result.append(dict(method=method, selection_horizon=sh, evaluation_horizon=eh,
            per_source=sources, **{k: float(np.mean([s[k] for s in sources])) for k in keys}))
    return result
