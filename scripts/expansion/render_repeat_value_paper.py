"""Compose the paper's fixed A/B/C comparisons from completed Quest result tables."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def records(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def one(rows, panel, method, h):
    selected = [r for r in rows if r['panel'] == panel and r['method'] == method and
                r['selection_horizon'] == str(h) and r['evaluation_horizon'] == str(h) and r['subset'] == 'all']
    if len(selected) != 1:
        raise ValueError('paper cell missing or duplicated')
    return selected[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detour-analysis', type=Path, required=True)
    parser.add_argument('--refresh-analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Scientific paper figures are generated on Quest')
    args.output.mkdir(parents=True, exist_ok=False)
    old_path = ROOT/'docs/audits/20260913/repeat_value/evidence/overall.csv'
    paths = [old_path, args.detour_analysis/'overall.csv', args.refresh_analysis/'overall.csv']
    old = records(old_path)
    new = records(paths[1])+records(paths[2])
    cells = [('candidate_refresh', 100, 'Refresh\n100 actions'), ('candidate_refresh', 200, 'Refresh\n200 actions'),
             ('detour', 220, 'Detour\n220 actions'), ('detour', 440, 'Detour\n440 actions')]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
    table = []
    for ax, method, title in zip(axes, ['real_one', 'real_full'],
                                ['One-shot selection and evaluation', 'Full-repeat selection and evaluation']):
        for panel, h, label in cells:
            a, c = one(old, panel, method, h), one(new, panel, method, h)
            for k in ['A_gain', 'B_gain']:
                if abs(float(a[k])-float(c[k])) > 1e-12:
                    raise ValueError('C scorer did not preserve the prior forecasts')
            table.append(dict(panel=panel, horizon=h, method=method, n_sources=c['n_sources'],
                **{k:c[k] for k in ['A_gain','B_gain','C_gain','C_gain_lo','C_gain_hi',
                     'A_source_rmse','B_source_rmse','error_improvement','error_improvement_lo','error_improvement_hi',
                     'C_loss','C_new_accident']}))
        for phase, dx, color, label in [('A', -.24, '#bb6a3b', 'A: reused outcomes'),
                                      ('B', 0, '#237f80', 'B: prior estimate'),
                                      ('C', .24, '#50578a', 'C: new execution')]:
            rows = [one(new if phase == 'C' else old, p, method, h) for p,h,_ in cells]
            mean = np.array([float(r[phase+'_gain'])*100 for r in rows])
            low = np.array([float(r[phase+'_gain_lo'])*100 for r in rows])
            high = np.array([float(r[phase+'_gain_hi'])*100 for r in rows])
            ax.bar(np.arange(4)+dx, mean, width=.23, color=color, label=label)
            ax.errorbar(np.arange(4)+dx, mean, yerr=[np.maximum(mean-low,0), np.maximum(high-mean,0)],
                        fmt='none', ecolor='#253641', elinewidth=.9, capsize=2)
        ax.axhline(0, color='#9ba8ad', linewidth=.8)
        ax.set_xticks(range(4), [label for _,_,label in cells])
        ax.set_title(title, loc='left', fontsize=11, pad=14)
    axes[0].set_ylabel('Selected safe task success gain (pp)')
    axes[1].legend(frameon=False, fontsize=9, loc='upper right')
    fig.text(.06, .015, 'Fixed A choices; source-weighted means and descriptive 95% source intervals. Refresh n=8, Detour n=16. C contains new executions of existing states.', fontsize=9, color='#53636b')
    fig.tight_layout(rect=[0,.07,1,1])
    for ext in ['png','svg']:
        fig.savefig(args.output/('abc_value.'+ext),dpi=190)
    plt.close(fig)
    with (args.output/'paper_table.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (args.output/'provenance.json').write_text(json.dumps(dict(job=os.environ['SLURM_JOB_ID'],
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        operation='presentation of precomputed, predeclared comparison cells; no refit or new rollout'),indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.iterdir() if p.is_file()},indent=2)+'\n')
    print('Paper figure and table:',args.output)


if __name__ == '__main__':
    main()
