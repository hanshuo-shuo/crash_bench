"""Figures for V5_HIL_REPORT.md from a finished HIL run (read-only).

python -m verifier_dataset.v5_hil_figures RUN_ROOT OUT_DIR writes
V5_HIL_OUTCOMES.png (completion and crossing of the infeasible line under each prompt, with
source-clustered 95% intervals) and V5_HIL_PATHS.png (the object's top-down path under the four
prompts for the first state of each sampling stratum, in plan order; no selection on outcome).
"""
import argparse
import gzip
import json
from pathlib import Path
from .inventory import read
from . import v5_hil as H

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e4e3df'
SERIES = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')
NAMES = dict(C0='C0 no rule', C1='C1 neutral note', C2='C2 feasible rule', C3='C3 infeasible rule')
STRATA = (('resting', 'train', 0), ('resting', 'train', 1), ('resting', 'validation', 0), ('resting', 'validation', 1),
          ('lifted', 'train', 2), ('lifted', 'train', 3))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=.6); ax.set_axisbelow(True)


def outcomes(root):
    states = read(root/'PLAN.json')['states']; out = {s['state']: {} for s in states}
    with gzip.open(root/'RUNS.jsonl.gz', 'rt') as stream:
        for r in map(json.loads, stream):
            out[r['state']][r['condition']] = r
    return states, out


def outcome_figure(root, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    states, runs = outcomes(root)
    measures = (('completed', 'completed the task', SERIES[0]), ('crossed_C3', 'moved the object across the infeasible (C3) line', SERIES[1]))
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200); fig.patch.set_facecolor(SURFACE); style(ax)
    width = .36
    for j, (field, label, color) in enumerate(measures):
        for i, c in enumerate(H.CONDITIONS):
            stat = lambda ss, c=c, field=field: H.mean(runs[s['state']][c][field] for s in ss)
            value = stat(states); low, high = H.bootstrap(states, stat)
            x = i + (j - .5) * (width + .04)
            ax.bar(x, value, width, color=color, label=label if i == 0 else None, zorder=2)
            ax.plot([x, x], [low, high], color=INK, linewidth=1, zorder=3)
            ax.text(x, max(value, high) + .03, '%.2f' % value, ha='center', va='bottom', fontsize=8, color=INK_2)
    ax.set_xticks(range(4)); ax.set_xticklabels([NAMES[c] for c in H.CONDITIONS], fontsize=9, color=INK)
    ax.set_ylim(0, 1.15); ax.set_yticks([0, .25, .5, .75, 1]); ax.grid(axis='x', visible=False)
    ax.set_ylabel('fraction of %d states' % len(states), fontsize=9, color=INK_2)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)
    ax.set_title('Same state, same seed, only the sentence changes', fontsize=11, color=INK, loc='left')
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE); plt.close(fig)


def path_figure(root, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    states = read(root/'PLAN.json')['states']
    chosen = [next(i for i, s in enumerate(states) if (s['phase'], s['split'], s['time_bin']) == k) for k in STRATA
              if any((s['phase'], s['split'], s['time_bin']) == k for s in states)]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.4), dpi=200); fig.patch.set_facecolor(SURFACE)
    for ax, index in zip(axes.flat, chosen):
        style(ax); spec = states[index]; case = root/('state_%03d' % index)
        row = read(case/'C0'/'ROW.json')
        # Top-down with the robot below: horizontal = -y, vertical = x (as the S1 top view).
        for k, dash in (('C3', '-'), ('C2', '--')):
            (ax_, ay), (bx, by) = row['lines'][k]['ends']
            dx, dy = bx - ax_, by - ay
            xs = [ax_ - 3 * dx, bx + 3 * dx]; ys = [ay - 3 * dy, by + 3 * dy]
            ax.plot([-y for y in ys], xs, dash, color=INK_2 if k == 'C3' else '#9a9893', linewidth=1.2, zorder=1)
            ax.plot([-ay, -by], [ax_, bx], 's', color=INK_2, markersize=5, zorder=4)
        for c, color in zip(H.CONDITIONS, SERIES):
            with gzip.open(case/c/'STEPS.jsonl.gz', 'rt') as stream:
                trace = [json.loads(x) for x in stream]
            ax.plot([-x['material'][1] for x in trace], [x['material'][0] for x in trace], color=color, linewidth=1.6,
                    label=NAMES[c], zorder=2)
            ax.plot([-trace[-1]['material'][1]], [trace[-1]['material'][0]], 'o', color=color, markersize=4, zorder=3)
        start = trace[0]['material']
        ax.plot([-start[1]], [start[0]], 'o', markerfacecolor=SURFACE, markeredgecolor=INK, markersize=6, zorder=5)
        ax.set_xlim(-.45, .45); ax.set_ylim(-.35, .45); ax.set_aspect('equal')
        ax.set_title('%s  %s, t=%d (%s, bin %d)' % (spec['state'], spec['image_source'].split('/')[-1], spec['checkpoint'],
                     dict(validation='val').get(spec['split'], spec['split']), spec['time_bin']), fontsize=8.5, color=INK, loc='left')
    for ax in list(axes.flat)[len(chosen):]:
        ax.axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle('Object path (metres, top-down, robot base below) from the rule\'s arrival (open circle) to the end (dot)\n'
                 'under the four prompts; solid line = C3 rule, dashed = C2 rule, squares = anchors', fontsize=8.5, color=INK_2, x=.02, ha='left')
    fig.tight_layout(rect=(0, .05, 1, .96)); fig.savefig(path, facecolor=SURFACE); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); p.add_argument('out', type=Path)
    a = p.parse_args()
    outcome_figure(a.root, a.out/'V5_HIL_OUTCOMES.png'); path_figure(a.root, a.out/'V5_HIL_PATHS.png')
