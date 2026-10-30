"""Figures for V5_DETOUR_REPORT.md from a finished detour run (read-only).

python -m verifier_dataset.v5_detour_figures RUN_ROOT OUT_DIR writes V5_DETOUR_OUTCOMES.png (entered the
zone, completed the task and obeyed the rule under each prompt, with source-clustered 95% intervals) and
V5_DETOUR_PATHS.png (the pudding's top-down path under the three prompts and the scripted witness's
detour, for the first state of each anchor and checkpoint in plan order; no selection on outcome).
"""
import argparse
import gzip
import json
from pathlib import Path
from .inventory import read
from . import v5_detour as D
from . import v5_geometry as g
from . import v5_hil as H
from .v5_hil_figures import GRID, INK, INK_2, SERIES, SURFACE, style

NAMES = dict(C0='C0 task only', C1='C1 neutral note', D='D keep-out rule')


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
    states, out = outcomes(root)
    measures = (('entered', 'pudding entered the 10 cm zone', SERIES[1]), ('completed', 'completed the task', SERIES[0]),
                ('obeyed', 'completed without entering', SERIES[2]))
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200); fig.patch.set_facecolor(SURFACE); style(ax)
    width = .25
    for j, (field, label, color) in enumerate(measures):
        for i, c in enumerate(D.CONDITIONS):
            stat = lambda ss, c=c, field=field: H.mean(out[s['state']][c][field] for s in ss)
            value = stat(states); low, high = H.bootstrap(states, stat, seed=D.BOOTSTRAP_SEED)
            x = i + (j - 1) * (width + .03)
            ax.bar(x, value, width, color=color, label=label if i == 0 else None, zorder=2)
            ax.plot([x, x], [low, high], color=INK, linewidth=1, zorder=3)
            ax.text(x, max(value, high) + .03, '%.2f' % value, ha='center', va='bottom', fontsize=8, color=INK_2)
    ax.set_xticks(range(3)); ax.set_xticklabels([NAMES[c] for c in D.CONDITIONS], fontsize=9, color=INK)
    ax.set_ylim(0, 1.15); ax.set_yticks([0, .25, .5, .75, 1]); ax.grid(axis='x', visible=False)
    ax.set_ylabel('share of %d states' % len(states), fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3, labelcolor=INK)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE, bbox_inches='tight'); plt.close(fig)


def rectangle(ax, lo, hi, **kw):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((100 * lo[0], 100 * lo[1]), 100 * (hi[0] - lo[0]), 100 * (hi[1] - lo[1]), **kw))


def paths_figure(root, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan = read(root/'PLAN.json'); states = plan['states']; out = outcomes(root)[1]; chosen = []
    for anchor in D.ANCHORS:
        for when in ('early', 'late'):
            pick = [i for i, s in enumerate(states) if s['anchor'] == anchor and s['time_bin'] == when]
            if pick:
                chosen.append(pick[0])
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 6.4), dpi=200, sharex=True, sharey=True); fig.patch.set_facecolor(SURFACE)
    colors = dict(C0=SERIES[0], C1=SERIES[3], D=SERIES[1])
    for ax, i in zip(axes.T.flat, chosen):
        style(ax); s = states[i]; case = Path(s['layout_case']); z = s['zone']
        with gzip.open(case/'TRACE.json.gz', 'rt') as stream:
            snap = [c for c in json.load(stream)['checkpoints'] if c['step'] == s['checkpoint']][0]
        rectangle(ax, *z['zone'], facecolor=SERIES[1], alpha=.10, edgecolor=SERIES[1], linewidth=.8, linestyle='--')
        anchor = snap['objects'][s['anchor']]; basket = snap['objects'][D.BASKET]
        rectangle(ax, anchor['lo'], anchor['hi'], facecolor=INK_2, alpha=.6, linewidth=0)
        rectangle(ax, basket['lo'], basket['hi'], facecolor='none', edgecolor=INK_2, linewidth=.8)
        side = read(case/'COMPLETE.json')['witness']
        with gzip.open(case/('WITNESS_%s.jsonl.gz' % side), 'rt') as stream:
            witness = [json.loads(x)['material'] for x in stream]
        ax.plot([100 * p[0] for p in witness], [100 * p[1] for p in witness], color=SERIES[2], linewidth=1.2, linestyle=':',
                label='scripted witness' if i == chosen[0] else None)
        for c in D.CONDITIONS:
            with gzip.open(root/('state_%03d' % i)/c/'STEPS.jsonl.gz', 'rt') as stream:
                trace = [json.loads(x) for x in stream]
            ax.plot([100 * x['material'][0] for x in trace], [100 * x['material'][1] for x in trace], color=colors[c], linewidth=1.2,
                    label=NAMES[c] if i == chosen[0] else None)
            if out[s['state']][c]['completed']:
                ax.plot(100 * trace[-1]['material'][0], 100 * trace[-1]['material'][1], 'o', color=colors[c], markersize=3)
        ax.set_title('%s, t = %d (%s)' % (g.phrase(s['anchor']), s['checkpoint'], s['split']), fontsize=8, color=INK)
        ax.set_aspect('equal')
    for ax in axes[-1]:
        ax.set_xlabel('x (cm)', fontsize=8, color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel('y (cm)', fontsize=8, color=INK_2)
    fig.legend(frameon=False, fontsize=8, loc='lower center', ncol=4, labelcolor=INK, bbox_to_anchor=(.5, -.01))
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(path, facecolor=SURFACE, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); p.add_argument('out', type=Path)
    a = p.parse_args()
    outcome_figure(a.root, a.out/'V5_DETOUR_OUTCOMES.png')
    paths_figure(a.root, a.out/'V5_DETOUR_PATHS.png')
