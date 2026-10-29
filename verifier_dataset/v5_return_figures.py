"""Figures for V5_RETURN_REPORT.md from a finished return run (read-only).

python -m verifier_dataset.v5_return_figures RUN_ROOT OUT_DIR writes V5_RETURN_OUTCOMES.png
(returned home, completed the task and stuck under each prompt on the main analysis set, with
source-clustered 95% intervals) and V5_RETURN_DISTANCE.png (end-effector distance to the starting
position over the steps after t: median and interquartile band of the runs still going, drawn while
at least MIN_RUNS remain).
"""
import argparse
import gzip
import json
from pathlib import Path
from .inventory import read
from . import v5_hil as H
from . import v5_return as R
from .v5_hil_figures import GRID, INK, INK_2, SERIES, SURFACE, style

NAMES = dict(C0='C0 original task', R='R instruction replaced', RA='RA instruction appended')
MIN_RUNS = 20


def runs(root):
    states = read(root/'PLAN.json')['states']; out = {s['state']: {} for s in states}
    with gzip.open(root/'RUNS.jsonl.gz', 'rt') as stream:
        for r in map(json.loads, stream):
            out[r['state']][r['condition']] = r
    return [s for s in states if out[s['state']]['C0']['start_gap'] >= R.AWAY], out


def outcome_figure(root, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    states, out = runs(root)
    measures = (('home', 'returned to the start (10 steps within 5 cm)', SERIES[2]), ('completed', 'completed the original task', SERIES[0]),
                ('stuck', 'neither (stopped away from the start)', SERIES[3]))
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200); fig.patch.set_facecolor(SURFACE); style(ax)
    width = .25
    for j, (field, label, color) in enumerate(measures):
        for i, c in enumerate(R.CONDITIONS):
            stat = lambda ss, c=c, field=field: H.mean(out[s['state']][c][field] for s in ss)
            value = stat(states); low, high = H.bootstrap(states, stat, seed=R.BOOTSTRAP_SEED)
            x = i + (j - 1) * (width + .03)
            ax.bar(x, value, width, color=color, label=label if i == 0 else None, zorder=2)
            ax.plot([x, x], [low, high], color=INK, linewidth=1, zorder=3)
            ax.text(x, max(value, high) + .03, '%.2f' % value, ha='center', va='bottom', fontsize=8, color=INK_2)
    ax.set_xticks(range(3)); ax.set_xticklabels([NAMES[c] for c in R.CONDITIONS], fontsize=9, color=INK)
    ax.set_ylim(0, 1.15); ax.set_yticks([0, .25, .5, .75, 1]); ax.grid(axis='x', visible=False)
    ax.set_ylabel('share of %d states' % len(states), fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3, labelcolor=INK)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE, bbox_inches='tight'); plt.close(fig)


def distance_figure(root, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    states, _ = runs(root); index = {s['state']: i for i, s in enumerate(read(root/'PLAN.json')['states'])}
    curves = {c: [] for c in R.CONDITIONS}
    for s in states:
        for c in R.CONDITIONS:
            folder = root/('state_%03d' % index[s['state']])/c
            with gzip.open(folder/'STEPS.jsonl.gz', 'rt') as stream:
                trace = [json.loads(x) for x in stream]
            curves[c].append(R.gaps(trace, read(folder/'ROW.json')['start_eef']))
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200); fig.patch.set_facecolor(SURFACE); style(ax)
    at = lambda xs, q: xs[int(q * (len(xs) - 1))]
    for (c, ds), color in zip(curves.items(), (SERIES[0], SERIES[1], SERIES[2])):
        steps, mid, low, high = [], [], [], []
        for k in range(max(map(len, ds))):
            xs = sorted(d[k] for d in ds if k < len(d))
            if len(xs) < MIN_RUNS:
                break
            steps.append(k); mid.append(100 * at(xs, .5)); low.append(100 * at(xs, .25)); high.append(100 * at(xs, .75))
        ax.fill_between(steps, low, high, color=color, alpha=.15, linewidth=0)
        ax.plot(steps, mid, color=color, linewidth=1.6, label=NAMES[c])
    ax.axhline(100 * R.HOME, color=INK_2, linewidth=.8, linestyle='--')
    ax.text(2, 100 * R.HOME + 1, 'home radius 5 cm', fontsize=8, color=INK_2)
    ax.set_xlabel('steps after the instruction changes (t)', fontsize=9, color=INK_2)
    ax.set_ylabel('end effector to start (cm)', fontsize=9, color=INK_2); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(.5, 1.), ncol=3, labelcolor=INK)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); p.add_argument('out', type=Path)
    a = p.parse_args()
    outcome_figure(a.root, a.out/'V5_RETURN_OUTCOMES.png')
    distance_figure(a.root, a.out/'V5_RETURN_DISTANCE.png')
