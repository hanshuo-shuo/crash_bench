"""Run the fixed existing-record repeat-value study on a Quest CPU allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crashbench.repeat_value import analyze, load_panels, write_csv


def render(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'figure.facecolor': 'white'})
    panels = [('detour', 440, 'Detour / 440'), ('candidate_refresh', 100, 'Refresh candidates / 100'),
              ('candidate_refresh', 200, 'Refresh candidates / 200'), ('selection_retest', 100, 'Refresh panel / 100')]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
    for ax, method, title in zip(axes, ['real_one', 'pseudo_one'],
                               ['Real intervention: one observation per arm', 'Pseudo-intervention: Base versus Base']):
        for phase, dx, color, label in [('A', -.17, '#bb6a3b', 'A: outcomes reused'),
                                       ('B', .17, '#237f80', 'B: separate executions')]:
            rows = [next(r for r in result['overall'] if r['panel'] == p and r['method'] == method
                         and r['selection_horizon'] == h and r['evaluation_horizon'] == h and r['subset'] == 'all')
                    for p, h, _ in panels]
            means = np.array([r[phase+'_gain'] for r in rows])*100
            low = np.array([r[phase+'_gain_lo'] for r in rows])*100
            high = np.array([r[phase+'_gain_hi'] for r in rows])*100
            ax.bar(np.arange(4)+dx, means, width=.32, color=color, label=label)
            ax.errorbar(np.arange(4)+dx, means, yerr=[means-low, high-means], fmt='none', ecolor='#253641', capsize=3)
        ax.axhline(0, color='#9ba8ad', linewidth=.8)
        ax.set_xticks(range(4), [p[2].replace(' / ', '\n') for p in panels])
        ax.set_title(title, fontsize=11, loc='left', pad=15)
    axes[0].set_ylabel('Selected safe task success minus reference (pp)')
    axes[1].legend(frameon=False, loc='upper right', fontsize=9)
    fig.text(.06, .015, 'Source-weighted development estimates; 95% source bootstrap. Pseudo-options are execution indices, not deployable actions.', fontsize=9, color='#53636b')
    fig.tight_layout(rect=[0, .065, 1, 1])
    for ext in ['png', 'svg']:
        fig.savefig(output / ('real_and_pseudo.'+ext), dpi=190)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    for ax, panel, short, long, treatment, title in [
        (axes[0], 'candidate_refresh', 100, 200, 'stale', 'Refresh: success under a changing deadline'),
        (axes[1], 'detour', 220, 440, 'glass', 'Detour: retained and late recovery'),
    ]:
        for sh, style in [(short, '-'), (long, '--')]:
            rows = [r for r in result['curves'] if r['panel'] == panel and r['selection_horizon'] == sh and r['subset'] == 'condition:'+treatment]
            x = [r['deadline'] for r in rows]
            if sh == short:
                ax.plot(x, [100*r['B_base_success'] for r in rows], color='#677781', label='Base (B)')
            ax.plot(x, [100*r['B_selected_success'] for r in rows], style, color='#237f80' if sh == short else '#bb6a3b', label=f'A choice at {sh}, scored on B')
        ax.axvline(short, color='#aab4b8', linewidth=.8, linestyle=':')
        ax.set(xlabel='Recorded action deadline', ylabel='Safe task success (%)', ylim=(-2, 102), xlim=(0, long))
        ax.set_title(title, loc='left', fontsize=11)
        ax.legend(frameon=False, fontsize=9)
    fig.text(.06, .015, 'Treatment subsets; full source tables retain controls. Selection stays fixed along each curve. The two studies use different action clocks.', fontsize=9, color='#53636b')
    fig.tight_layout(rect=[0, .06, 1, 1])
    for ext in ['png', 'svg']:
        fig.savefig(output / ('deadline_curves.'+ext), dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Scientific analysis must run in the requested Quest Slurm allocation')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    panels, hashes = load_panels(ROOT)
    result = analyze(panels)
    for name in ['overall', 'per_source', 'per_cell', 'curves']:
        write_csv(args.output_dir / (name+'.csv'), result[name])
    (args.output_dir / 'choices.json').write_text(json.dumps(result['choices'], indent=2)+'\n')
    provenance = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        job=os.environ['SLURM_JOB_ID'], host=platform.node(), python=sys.version, input_sha256=hashes,
        scope='existing exposed development records; no new rollouts, model fitting, or D8',
        bootstrap_draws=5000, bootstrap_seed=20270913,
        panels=[dict(name=p['name'], cells=len(p['cells']), sources=len({c['source'] for c in p['cells']}),
                     shared_prefix_outcomes=sum('shared' in c for c in p['cells']),
                     actual_branches=sum(len(v) for c in p['cells'] for v in c['samples'].values())) for p in panels])
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    render(result, args.output_dir)
    lines = ['# Existing-data repeat-value results', '',
             'Development diagnostics. All gains are safe-task-success percentage points, physical-source weighted.', '',
             '| Panel | Horizon | Method | A reused gain | B separate gain | A minus B [95% descriptive CI] | Positive B sources |',
             '|---|---:|---|---:|---:|---|---:|']
    for r in result['overall']:
        if r['subset'] != 'all' or r['selection_horizon'] != r['evaluation_horizon']:
            continue
        if r['method'] not in ['real_full', 'real_one', 'pseudo_one', 'pseudo_one_swapped', 'real_two', 'pseudo_two']:
            continue
        lines.append(f"| {r['panel']} | {r['evaluation_horizon']} | {r['method']} | {100*r['A_gain']:.2f} | {100*r['B_gain']:.2f} | {100*r['reuse_gap']:.2f} [{100*r['reuse_gap_lo']:.2f}, {100*r['reuse_gap_hi']:.2f}] | {r['sources_any_B_positive']}/{r['n_sources']} |")
    lines += ['', 'A/B are execution blocks on the same sources, not fresh-source generalization.',
              'The pseudo-option columns must not be subtracted from real intervention columns as a noise correction.',
              'Zero intervals reflect the recorded source contributions, not population safety certification.', '']
    (args.output_dir / 'RESULTS.md').write_text('\n'.join(lines))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output_dir.iterdir() if p.is_file()}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('\n'.join(lines), flush=True)
    print('Output:', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
