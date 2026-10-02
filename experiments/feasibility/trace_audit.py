"""Post-run common-trajectory audits for budget and unchanged-AEGIS controls."""
import json
from collections import defaultdict
from pathlib import Path

FIELDS = ['step', 'original', 'output', 'qp_status', 'delta_linf',
          'eef_pos', 'target_pos', 'reference_phase', 'obstacle_l1_m', 'success']


def compare_common_trace(shorter, longer, label):
    if len(longer) < len(shorter):
        raise RuntimeError('Extended execution stopped before common trajectory: ' + label)
    for index, (a, b) in enumerate(zip(shorter, longer)):
        for field in FIELDS:
            if a[field] != b[field]:
                raise RuntimeError('Common trajectory differs: %s action=%d field=%s' %
                                   (label, index + 1, field))
    return len(shorter)


def audit(root, rows):
    root = Path(root)
    cache = {}

    def trace(row):
        name = row['run_id']
        if name not in cache:
            cache[name] = [json.loads(line) for line in
                           (root / 'runs' / name / 'steps.jsonl').read_text().splitlines()]
        return cache[name]

    groups = defaultdict(dict)
    initial = {}
    for row in rows:
        if row['branch_step'] is None and row['condition'] == 'identity_geometry':
            initial[(row['state'], row['repeat'])] = row
        if row['branch_step'] is not None:
            groups[(row['state'], row['repeat'], row['branch_step'], row['condition'])][row['extra_budget']] = row
    budgets = []; controls = []
    for (state, repeat, step, candidate), group in sorted(groups.items()):
        if step:
            if set(group) != {0, step}:
                raise RuntimeError('Budget pair coverage missing')
            remaining, extended = group[0], group[step]
            length = compare_common_trace(trace(remaining), trace(extended),
                                          remaining['run_id'] + ' vs ' + extended['run_id'])
            budgets.append({'remaining': remaining['run_id'], 'extended': extended['run_id'],
                            'verified_common_actions': length})
        if candidate == 'aegis':
            base = initial[(state, repeat)]
            for row in group.values():
                a, b = trace(base), trace(row)
                length = compare_common_trace(a, b, base['run_id'] + ' vs ' + row['run_id'])
                if row['extra_budget'] == 0 and (len(a) != len(b) or
                        (base['success'], base['collided']) != (row['success'], row['collided'])):
                    raise RuntimeError('Unchanged AEGIS control outcome differs')
                controls.append({'base': base['run_id'], 'continuation': row['run_id'],
                                 'verified_common_actions': length})
    return {'status': 'passed', 'budget_pairs': budgets, 'unchanged_aegis_controls': controls,
            'scope': 'all stored execution fields match through the common trajectory; longer budget only extends it'}
