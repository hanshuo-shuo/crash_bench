"""Repeat-disjoint value diagnostics on frozen terminal records; no simulator or learner."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def terminal_sample(success, accident, step, repeat):
    if success and accident:
        raise ValueError('competing terminal labels cannot both be true')
    return dict(success_step=float(step) if success else None,
                accident_step=float(step) if accident else None, repeat=int(repeat))


def outcomes(samples, horizon):
    return np.array([[float(s['success_step'] is not None and s['success_step'] <= horizon),
                      float(s['accident_step'] is not None and s['accident_step'] <= horizon)]
                     for s in samples], dtype=float)


def load_panels(root):
    root = Path(root)
    paths = []
    panels = []
    detour = root / 'docs/audits/20260909/detour_benefit_gate/evidence'
    records_path = detour / 'all_terminal_records.csv'
    paths.extend([records_path, detour / 'shared_prefix_outcomes.json', detour / 'freeze.json'])
    cells = {}
    for r in csv.DictReader(records_path.open()):
        if int(r['horizon']) != 440:
            continue
        c = cells.setdefault(r['episode_id'], dict(id=r['episode_id'], source=r['source'],
                            condition=r['condition'], role=r['role'], samples=defaultdict(list)))
        c['samples'][r['phase'], int(r['option'])].append(terminal_sample(
            int(r['success']), int(r['accident']), int(r['steps']), r['repeat']))
    for r in read_json(detour / 'shared_prefix_outcomes.json'):
        v = r['horizons']['440']
        cells[r['episode_id']] = dict(id=r['episode_id'], source=r['source'],
            condition=r['condition'], role=r['role'], samples={},
            shared=terminal_sample(v['success'], v['accident'], v['steps'], -1))
    frozen = read_json(detour / 'freeze.json')['choices']
    panels.append(dict(name='detour', cells=sorted(cells.values(), key=lambda c: c['id']),
                       horizons=[220, 440], repeats=2, treatment='glass',
                       historical={h: frozen for h in [220, 440]}))
    for name, directory, horizons in [
        ('candidate_refresh', 'docs/audits/20260908/candidate_refresh/verified_evidence', [100, 200]),
        ('selection_retest', 'docs/audits/20260907/selection_retest/evidence', [100]),
    ]:
        directory = root / directory
        anchors_path = directory / ('anchors.json' if name == 'candidate_refresh' else 'panel.json')
        paths.extend([anchors_path, directory / 'A.json', directory / 'B.json', directory / 'freeze.json'])
        cells = {r['panel_id']: dict(id=r['panel_id'], source=r['physical_source_id'],
                 condition=r['condition'], role=r['role'], samples=defaultdict(list))
                 for r in read_json(anchors_path)}
        for phase in ['A', 'B']:
            for r in read_json(directory / (phase + '.json')):
                if r['phase'] != phase:
                    raise ValueError('phase mismatch')
                v = r.get('horizons', {}).get(str(max(horizons)), r)
                cells[r['panel_id']]['samples'][phase, int(r['option'])].append(terminal_sample(
                    v['task_success'], v['catastrophe'], v['steps'], r['repeat']))
        frozen = read_json(directory / 'freeze.json')
        history = {}
        for h in horizons:
            f = frozen[str(h)] if name == 'candidate_refresh' else frozen
            history[h] = dict(f['rules'])
            history[h]['A_reference'] = f['reference']
        panels.append(dict(name=name, cells=sorted(cells.values(), key=lambda c: c['id']),
                           horizons=horizons, repeats=4, treatment='stale', historical=history))
    for panel in panels:
        validate_panel(panel)
    return panels, {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_panel(panel):
    identities = set()
    roles = {}
    for c in panel['cells']:
        if c['id'] in identities:
            raise ValueError('duplicate episode')
        identities.add(c['id'])
        if roles.setdefault(c['source'], c['role']) != c['role']:
            raise ValueError('source crosses roles')
        if 'shared' in c:
            if c['samples']:
                raise ValueError('shared outcome must not have invented repeats')
            continue
        if set(c['samples']) != {('A', 0), ('A', 1), ('B', 0), ('B', 1)}:
            raise ValueError('missing phase or arm')
        for phase in ['A', 'B']:
            repeats = []
            for option in [0, 1]:
                samples = c['samples'][phase, option]
                samples.sort(key=lambda s: s['repeat'])
                ids = [s['repeat'] for s in samples]
                if len(ids) != panel['repeats'] or len(set(ids)) != len(ids):
                    raise ValueError('incomplete or duplicate repeats')
                repeats.append(ids)
            if repeats[0] != repeats[1]:
                raise ValueError('unmatched repeat identities')
        if set(s['repeat'] for s in c['samples']['A', 0]) & set(s['repeat'] for s in c['samples']['B', 0]):
            raise ValueError('A/B repetitions overlap')
    for mapping in panel['historical'].values():
        for choice in mapping.values():
            if set(choice) != identities or set(choice.values()) - {0, 1}:
                raise ValueError('historical choice incomplete or invalid')


def arms(cell, phase, mode, horizon):
    if 'shared' in cell:
        raise ValueError('shared outcome is a scalar, not a repeated arm')
    base = cell['samples'][phase, 0]
    real = cell['samples'][phase, 1]
    if mode == 'real_full':
        pair = base, real
    elif mode == 'real_one':
        pair = base[:1], real[:1]
    elif mode == 'pseudo_one':
        pair = base[:1], base[1:2]
    elif mode == 'pseudo_one_swapped':
        pair = base[1:2], base[:1]
    elif mode == 'real_two':
        pair = base[:2], real[:2]
    elif mode == 'pseudo_two':
        if len(base) != 4:
            raise ValueError('two-replicate pseudo arms require four Base repetitions')
        pair = base[:2], base[2:4]
    else:
        raise ValueError(mode)
    return tuple(outcomes(p, horizon) for p in pair)


def freeze_choice(cells, mode, horizon):
    choice = {}
    for c in cells:
        if 'shared' in c:
            choice[c['id']] = 0
        else:
            base, other = arms(c, 'A', mode, horizon)
            choice[c['id']] = int(other[:, 0].mean() > base[:, 0].mean())
    return choice


def measure_cell(c, phase, mode, horizon, take):
    if 'shared' in c:
        if take:
            raise ValueError('cannot intervene after shared prefix termination')
        s, a = outcomes([c['shared']], horizon)[0]
        return dict(base_success=s, selected_success=s, gain=0., base_accident=a,
                    selected_accident=a, accident_delta=0., loss=0., new_accident=0., intervention=0.)
    base, other = arms(c, phase, mode, horizon)
    selected = other if take else base
    s, a = selected.mean(axis=0)
    bs, ba = base.mean(axis=0)
    return dict(base_success=bs, selected_success=s, gain=s-bs, base_accident=ba,
                selected_accident=a, accident_delta=a-ba,
                loss=float(np.mean(base[:, 0] * (1-selected[:, 0]))),
                new_accident=float(np.mean((1-base[:, 1]) * selected[:, 1])), intervention=float(take))


def source_average(rows, keys):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['source']].append(r)
    return [dict(source=s, **{k: float(np.mean([r[k] for r in group])) for k in keys})
            for s, group in sorted(grouped.items())]


def summarize(rows, bootstrap=5000, seed=20270913):
    keys = [k for k in rows[0] if k.startswith(('A_', 'B_'))]
    sources = source_average(rows, keys)
    values = np.array([[r[k] for k in keys] for r in sources])
    samples = np.random.default_rng(seed).integers(0, len(sources), (bootstrap, len(sources)))
    boot = values[samples].mean(axis=1)
    result = dict(n_sources=len(sources), n_cells=len(rows))
    for i, k in enumerate(keys):
        result[k] = float(values[:, i].mean())
        result[k+'_lo'], result[k+'_hi'] = map(float, np.quantile(boot[:, i], [.025, .975]))
    ai, bi = keys.index('A_gain'), keys.index('B_gain')
    gap = boot[:, ai]-boot[:, bi]
    result.update(reuse_gap=float((values[:, ai]-values[:, bi]).mean()),
                  reuse_gap_lo=float(np.quantile(gap, .025)), reuse_gap_hi=float(np.quantile(gap, .975)),
                  sources_any_A_positive=len({r['source'] for r in rows if r['A_gain'] > 0}),
                  sources_any_B_positive=len({r['source'] for r in rows if r['B_gain'] > 0}),
                  sources_net_B_positive=sum(r['B_gain'] > 0 for r in sources),
                  sources_net_B_negative=sum(r['B_gain'] < 0 for r in sources),
                  sources_with_A_B_gain_change=sum(r['A_gain'] != r['B_gain'] for r in sources))
    return result, sources


def evaluate(panel, mode, selection_horizon, evaluation_horizon, choices=None):
    if choices is None:
        choices = freeze_choice(panel['cells'], mode, selection_horizon)
    rows = []
    for c in panel['cells']:
        row = {k: c[k] for k in ['id', 'source', 'condition', 'role']}
        row['shared'] = 'shared' in c
        row['choice'] = choices[c['id']]
        for phase in ['A', 'B']:
            row.update({phase+'_'+k: v for k, v in measure_cell(
                c, phase, mode, evaluation_horizon, choices[c['id']]).items()})
        rows.append(row)
    return rows


def subsets(rows, treatment):
    yield 'all', rows
    for condition in sorted({r['condition'] for r in rows}):
        yield 'condition:'+condition, [r for r in rows if r['condition'] == condition]
    controls = [r for r in rows if r['condition'] != treatment]
    if controls:
        yield 'controls', controls
    for role in sorted({r['role'] for r in rows}):
        yield 'role:'+role, [r for r in rows if r['role'] == role]


def analyze(panels, bootstrap=5000):
    overall, per_source, per_cell, curves, choices_log = [], [], [], [], []
    for panel in panels:
        modes = ['real_full', 'real_one', 'pseudo_one', 'pseudo_one_swapped']
        if panel['repeats'] == 4:
            modes += ['real_two', 'pseudo_two']
        for sh in panel['horizons']:
            runs = [(mode, mode, freeze_choice(panel['cells'], mode, sh)) for mode in modes]
            runs += [('historical:'+name, 'real_full', choice)
                     for name, choice in panel['historical'][sh].items()]
            for name, mode, choice in runs:
                choices_log.append(dict(panel=panel['name'], method=name, selection_horizon=sh, choices=choice))
                for eh in panel['horizons']:
                    meta = dict(panel=panel['name'], method=name, selection_horizon=sh, evaluation_horizon=eh)
                    rows = evaluate(panel, mode, sh, eh, choice)
                    per_cell += [dict(**meta, **r) for r in rows]
                    for subset, group in subsets(rows, panel['treatment']):
                        summary, sources = summarize(group, bootstrap)
                        overall.append(dict(**meta, subset=subset, **summary))
                        per_source += [dict(**meta, subset=subset, **r) for r in sources]
            # All available deadlines, only the previously selected full-repeat reference.
            choice = freeze_choice(panel['cells'], 'real_full', sh)
            for h in range(max(panel['horizons'])+1):
                rows = evaluate(panel, 'real_full', sh, h, choice)
                for subset, group in subsets(rows, panel['treatment']):
                    if subset.startswith('role:'):
                        continue
                    keys = [k for k in group[0] if k.startswith(('A_', 'B_'))]
                    src = source_average(group, keys)
                    curves.append(dict(panel=panel['name'], selection_horizon=sh, deadline=h, subset=subset,
                                       **{k: float(np.mean([r[k] for r in src])) for k in keys}))
    return dict(overall=overall, per_source=per_source, per_cell=per_cell, curves=curves, choices=choices_log)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def forecast_comparison(source_rows, bootstrap=5000, seed=20270913):
    """Paired source-level forecast errors against a noisy future execution block."""
    x = np.array([[r[k] for k in ['A_gain', 'B_gain', 'C_gain']] for r in source_rows])
    d = (x[:, 0]-x[:, 2])**2 - (x[:, 1]-x[:, 2])**2
    draws = np.random.default_rng(seed).integers(0, len(x), (bootstrap, len(x)))
    result = dict(A_gain=float(x[:, 0].mean()), B_gain=float(x[:, 1].mean()),
                  C_gain=float(x[:, 2].mean()),
                  A_source_rmse=float(np.sqrt(np.mean((x[:, 0]-x[:, 2])**2))),
                  B_source_rmse=float(np.sqrt(np.mean((x[:, 1]-x[:, 2])**2))))
    for name, values in [('A_minus_C', x[:, 0]-x[:, 2]), ('B_minus_C', x[:, 1]-x[:, 2]),
                         ('error_improvement', d), ('C_gain', x[:, 2])]:
        low, high = np.quantile(values[draws].mean(axis=1), [.025, .975])
        result.update({name: float(values.mean()), name+'_lo': float(low), name+'_hi': float(high)})
    return result
