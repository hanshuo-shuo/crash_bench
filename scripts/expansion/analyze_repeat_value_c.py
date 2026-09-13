"""Score a completed prospective C block against choices and forecasts locked before C."""
import argparse
import json
import os
from pathlib import Path
import sys
import hashlib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crashbench.repeat_value import (load_panels, terminal_sample, evaluate, measure_cell,
    subsets, source_average, forecast_comparison, write_csv)
from scripts.expansion.run_repeat_value_c import read, check_records


def render(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keep = [r for r in rows if r['subset'] == 'all' and r['selection_horizon'] == r['evaluation_horizon']
            and r['method'] in ['real_one', 'real_full', 'pseudo_one', 'pseudo_one_swapped']]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for phase, dx, color, label in [('A', -.24, '#bb6a3b', 'A: reused outcomes'),
                                    ('B', 0, '#237f80', 'B: prior estimate'),
                                    ('C', .24, '#50578a', 'C: new execution')]:
        ax.bar(np.arange(len(keep))+dx, [100*r[phase+'_gain'] for r in keep], width=.23, color=color, label=label)
    ax.set_xticks(range(len(keep)), [r['method'].replace('_', ' ')+'\nH='+str(r['evaluation_horizon']) for r in keep], fontsize=8)
    ax.set_ylabel('Safe task success gain (percentage points)')
    ax.axhline(0, color='#9ba8ad', linewidth=.8)
    ax.legend(frameon=False)
    ax.set_title('Locked A selections evaluated across three execution blocks', loc='left', pad=16)
    fig.text(.07, .02, 'Same previously exposed physical sources; C is a new execution block. Source intervals and all frozen references are in the accompanying tables.', fontsize=9, color='#53636b')
    fig.tight_layout(rect=[0, .06, 1, 1])
    for ext in ['png', 'svg']:
        fig.savefig(output/('prospective_c.'+ext), dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Research analysis runs on Quest')
    run = args.run_dir
    complete = read(run/'complete.json')
    if complete['status'] != 'COMPLETE' or hashlib.sha256((run/'C.json').read_bytes()).hexdigest() != complete['C_sha256']:
        raise ValueError('C block not complete or changed')
    contract = read(run/'contract.json')
    contract_path = ROOT/'docs/audits/20260913/repeat_value/c_contract.json'
    if hashlib.sha256(contract_path.read_bytes()).hexdigest() != complete['contract_sha256']:
        raise ValueError('pre-C contract changed')
    for rel, digest in contract['input_sha256'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen forecast input changed')
    name = read(run/'provenance.json')['panel']
    panels, _ = load_panels(ROOT)
    panel = next(p for p in panels if p['name'] == name)
    lookup = {c['id']: c for c in panel['cells']}
    records = read(run/'C.json')
    check_records(read(run/'anchors.json'), records, contract['panels'][name]['repeats'], name)
    for c in panel['cells']:
        if 'shared' not in c:
            c['samples']['C', 0], c['samples']['C', 1] = [], []
    for r in records:
        key = r.get('episode_id', r.get('panel_id'))
        v = r['horizons'][str(max(panel['horizons']))]
        sample = terminal_sample(v.get('success', v.get('task_success')),
            v.get('accident', v.get('catastrophe')), v['steps'], r['repeat'])
        lookup[key]['samples']['C', r['option']].append(sample)
    for c in panel['cells']:
        for option in [0, 1]:
            if 'shared' not in c:
                c['samples']['C', option].sort(key=lambda s: s['repeat'])
    frozen = [f for f in read(ROOT/contract['forecast_choices']) if f['panel'] == name]
    overall, all_sources, all_cells = [], [], []
    for f in frozen:
        mode = 'real_full' if f['method'].startswith('historical:') else f['method']
        for eh in panel['horizons']:
            meta = dict(panel=name, method=f['method'], selection_horizon=f['selection_horizon'], evaluation_horizon=eh)
            rows = evaluate(panel, mode, f['selection_horizon'], eh, f['choices'])
            for row in rows:
                row.update({'C_'+k: v for k, v in measure_cell(lookup[row['id']], 'C', mode, eh, row['choice']).items()})
            all_cells += [dict(**meta, **r) for r in rows]
            for subset, group in subsets(rows, panel['treatment']):
                keys = [k for k in group[0] if k.startswith(('A_', 'B_', 'C_'))]
                source = source_average(group, keys)
                summary = forecast_comparison(source)
                summary.update({k: float(np.mean([r[k] for r in source])) for k in keys if k not in summary})
                overall.append(dict(**meta, subset=subset, n_sources=len(source), n_cells=len(group),
                    C_positive_sources=sum(r['C_gain'] > 0 for r in source),
                    C_negative_sources=sum(r['C_gain'] < 0 for r in source), **summary))
                all_sources += [dict(**meta, subset=subset, **r) for r in source]
    output = run/'analysis'
    output.mkdir(exist_ok=False)
    for filename, rows in [('overall.csv', overall), ('per_source.csv', all_sources), ('per_cell.csv', all_cells)]:
        write_csv(output/filename, rows)
    render(overall, output)
    lines = ['# Prospective C execution results', '',
        '| Method | Selection / evaluation H | A gain | B gain | C gain | A-C [95% CI] | Source RMSE A / B |',
        '|---|---|---:|---:|---:|---|---|']
    for r in overall:
        if r['subset'] != 'all' or r['selection_horizon'] != r['evaluation_horizon']:
            continue
        if r['method'].startswith('historical:'):
            continue
        lines.append(f"| {r['method']} | {r['selection_horizon']}/{r['evaluation_horizon']} | {100*r['A_gain']:.2f} | {100*r['B_gain']:.2f} | {100*r['C_gain']:.2f} | {100*r['A_minus_C']:.2f} [{100*r['A_minus_C_lo']:.2f}, {100*r['A_minus_C_hi']:.2f}] | {100*r['A_source_rmse']:.2f}/{100*r['B_source_rmse']:.2f} |")
    lines += ['', 'All displayed values are percentage points. C is a noisy new execution block on existing sources.',
        'Separate-execution forecasts are not assumed to improve; signed source error contrasts and all controls are retained in overall.csv.', '']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
