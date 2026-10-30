"""Figures for HIL_SUITE_REPORT.md from finished HIL suite runs (read-only; every number comes from SUITE_READING.json).

python -m verifier_dataset.hil_suite_figures OUT_DIR --switch POLICY=BRANCH_ROOT ... [--physical ROOT] [--probe ROOT] writes
HIL_SUITE_SWITCH.png (stage 1: B success after a mid-task switch to another trained instruction, by switch time, with
the t = 0 ceiling, per group and policy), HIL_SUITE_COMMANDS.png (stage 2: share of runs that did what the untrained
command asked, and A completion relative to C0, per suite), HIL_SUITE_PHYSICAL.png (stage 3 recovery) and
HIL_SUITE_PROBE.png (stage 5 prompt sensitivity by progress).
"""
import argparse
from pathlib import Path
from .inventory import read
from . import hil_suite as S
from .v5_hil_figures import GRID, INK, INK_2, SERIES, SURFACE, style

GROUPS = {'1a': 'LIBERO-Goal (8 switch pairs)', '1b': 'LIBERO-10 scene 2 (2 pairs)', '1c': 'LIBERO-Object control (2 pairs)'}
POLICY = dict(pi05_libero='π0.5', pi0_libero='π0', openvla_oft='OpenVLA-OFT')
SUITE = dict(libero_spatial='Spatial', libero_object='Object', libero_goal='Goal', libero_10='LIBERO-10')


def plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as p
    return p


def interval(ax, x, est, color):
    if est['interval'] is not None:
        ax.plot([x, x], est['interval'], color=color, linewidth=1, alpha=.8, zorder=3)


def switch_figure(runs, path):
    p = plt(); readings = {k: read(Path(v)/'SUITE_READING.json')['stage1'] for k, v in runs}
    groups = [g for g in GROUPS if any(g in r for r in readings.values())]
    fig, axes = p.subplots(1, len(groups), figsize=(3.6 * len(groups), 3.4), dpi=200, squeeze=False); fig.patch.set_facecolor(SURFACE)
    for ax, g in zip(axes[0], groups):
        style(ax)
        for j, (policy, r) in enumerate(readings.items()):
            if g not in r:
                continue
            phases = [0.] + [float(f) for f in r[g]['phases']]; color = SERIES[j % len(SERIES)]
            b = [r[g]['ceiling']] + [r[g]['phases'][f]['sw_b'] for f in r[g]['phases']]
            a = [r[g]['phases'][f]['persist'] for f in r[g]['phases']]
            ax.plot(phases, [e['value'] for e in b], marker='o', color=color, label='%s: did B' % POLICY.get(policy, policy), zorder=4)
            ax.plot(phases[1:], [e['value'] for e in a], marker='s', linestyle='--', color=color, alpha=.7,
                    label='%s: finished A' % POLICY.get(policy, policy), zorder=4)
            for x, e in zip(phases, b):
                interval(ax, x + .008 * j, e, color)
        ax.set_title(GROUPS[g], fontsize=9, color=INK); ax.set_xticks([0, .25, .5, .75]); ax.set_xticklabels(['0 (ceiling)', '25%', '50%', '75%'], fontsize=8)
        ax.set_xlabel('switch time (share of A done)', fontsize=8, color=INK_2); ax.set_ylim(-.03, 1.05)
    axes[0][0].set_ylabel('share of (pair, state) rows', fontsize=8, color=INK_2)
    axes[0][-1].legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1), labelcolor=INK)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE, bbox_inches='tight'); p.close(fig)


def commands_figure(runs, path):
    p = plt(); readings = {k: read(Path(v)/'SUITE_READING.json')['stage2'] for k, v in runs}
    fig, axes = p.subplots(1, len(S.INTERVENTIONS), figsize=(11, 3.3), dpi=200); fig.patch.set_facecolor(SURFACE)
    width = .8 / max(1, len(readings))
    for ax, c in zip(axes, S.INTERVENTIONS):
        style(ax)
        for j, (policy, r) in enumerate(readings.items()):
            suites = [s for s in S.SUITES if s in r[c]['suites']]
            xs = [i + (j - (len(readings) - 1) / 2) * width for i in range(len(suites))]
            ax.bar(xs, [r[c]['suites'][s]['follow'] for s in suites], width * .9, color=SERIES[j % len(SERIES)], label=POLICY.get(policy, policy), zorder=2)
            ax.scatter(xs, [r[c]['suites'][s]['persist'] for s in suites], marker='_', s=120, color=INK, zorder=3,
                       label='finished A ÷ C0' if j == 0 else None)
        ax.set_xticks(range(len(S.SUITES))); ax.set_xticklabels([SUITE[s] for s in S.SUITES], fontsize=8); ax.set_ylim(0, 1.15)
        ax.set_title('%s: "%s"' % (c, dict(STOP=S.STOP, RETURN='stop and move … back to its starting position', CANCEL='do not <task>, stop doing it now')[c]),
                     fontsize=8, color=INK); ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('did what was asked (bars)', fontsize=8, color=INK_2)
    axes[-1].legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1), labelcolor=INK)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE, bbox_inches='tight'); p.close(fig)


def physical_figure(root, path):
    p = plt(); r = read(Path(root)/'SUITE_READING.json')['stage3']
    fig, ax = p.subplots(figsize=(6.4, 3.2), dpi=200); fig.patch.set_facecolor(SURFACE); style(ax)
    names = dict(OBJ='shift the resting target 6 cm', DROP='knock the lifted object out', PUSH='push the arm 10 cm', GOAL='shift the receptacle 6 cm')
    for i, c in enumerate(S.PERTURB):
        e = r[c]['rho']
        if e['value'] is None:
            continue
        ax.bar(i, e['value'], .6, color=SERIES[i % len(SERIES)], zorder=2); interval(ax, i, e, INK)
        ax.text(i, (e['interval'] or [e['value']] * 2)[1] + .03, '%.2f\nn=%d' % (e['value'], r[c]['applied']), ha='center', fontsize=7, color=INK_2)
    for y in (S.RECOVERS, S.FRAGILE):
        ax.axhline(y, color=GRID, linewidth=1, linestyle=':')
    ax.set_xticks(range(len(S.PERTURB))); ax.set_xticklabels([names[c] for c in S.PERTURB], fontsize=7); ax.set_ylim(0, 1.3)
    ax.set_ylabel('A finished ÷ C0 (ρ_phys)', fontsize=8, color=INK_2); ax.grid(axis='x', visible=False)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE, bbox_inches='tight'); p.close(fig)


def probe_figure(root, path):
    p = plt(); r = read(Path(root)/'SUITE_READING.json')['groups']
    fig, ax = p.subplots(figsize=(6.4, 3.2), dpi=200); fig.patch.set_facecolor(SURFACE); style(ax)
    for j, (g, v) in enumerate(sorted(r.items())):
        bins = sorted(v['by_progress'], key=int); xs = [(int(b) + .5) / S.PROBE_BINS for b in bins]
        ax.plot(xs, [v['by_progress'][b]['B_over_noise'] for b in bins], marker='o', color=SERIES[j], label='%s: switch prompt B' % GROUPS[g].split(' (')[0])
        ax.plot(xs, [v['by_progress'][b]['empty_over_noise'] for b in bins], marker='s', linestyle='--', color=SERIES[j], alpha=.7,
                label='%s: empty prompt' % GROUPS[g].split(' (')[0])
    ax.axhline(1, color=GRID, linewidth=1, linestyle=':'); ax.set_yscale('log')
    ax.set_xlabel('progress of A when probed', fontsize=8, color=INK_2); ax.set_ylabel('first-chunk change ÷ seed noise (median)', fontsize=8, color=INK_2)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE, bbox_inches='tight'); p.close(fig)


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('out', type=Path); a.add_argument('--switch', nargs='+', default=[])
    a.add_argument('--physical'); a.add_argument('--probe'); x = a.parse_args()
    runs = [tuple(s.split('=', 1)) for s in x.switch]
    if runs:
        switch_figure(runs, x.out/'HIL_SUITE_SWITCH.png'); commands_figure(runs, x.out/'HIL_SUITE_COMMANDS.png')
    if x.physical:
        physical_figure(x.physical, x.out/'HIL_SUITE_PHYSICAL.png')
    if x.probe:
        probe_figure(x.probe, x.out/'HIL_SUITE_PROBE.png')
