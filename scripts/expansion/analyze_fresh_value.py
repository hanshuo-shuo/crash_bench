"""Evaluate the previously locked references after completion of all fresh blocks."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crashbench.fresh_value import (
    HORIZONS, cells_from_records, score_choices, summarize_ab, validate_anchors,
)
from crashbench.repeat_value import (
    forecast_comparison, outcomes, source_average, subsets, write_csv,
)
from scripts.expansion.hash_tree_manifest import file_sha256
from scripts.expansion.run_fresh_value import read, verify_run, write


def render(overall, support, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
        'axes.spines.right':False, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), gridspec_kw={'width_ratios':[1.1,1]})
    keep = [r for r in overall if r['subset']=='all' and
        r['selection_horizon']==440 and r['evaluation_horizon']==440 and
        r['method'] in ('real_full','real_one','pseudo_one','pseudo_one_swapped')]
    for phase, shift, color in [('A',-.24,'#bb6a3b'),('B',0,'#237f80'),('C',.24,'#50578a')]:
        axes[0].bar(np.arange(len(keep))+shift, [100*r[phase+'_gain'] for r in keep],
            width=.23, color=color, label=phase)
    axes[0].set_xticks(range(len(keep)), [r['method'].replace('_','\n') for r in keep])
    axes[0].set_ylabel('Safe success gain (percentage points)')
    axes[0].set_title('Frozen A choices on fresh source states', loc='left')
    axes[0].legend(frameon=False, ncol=3)
    positive = [r for r in support if r['horizon']==440 and r['condition']=='glass']
    x = np.arange(len(positive))
    for phase, marker, color in [('A','o','#bb6a3b'),('B','s','#237f80'),('C','^','#50578a')]:
        axes[1].plot(x, [100*r[phase+'_gain'] for r in positive], marker=marker,
            color=color, label=phase, linewidth=1, alpha=.85)
    axes[1].set_xticks(x, [r['source_id'] for r in positive], rotation=45)
    axes[1].set_ylabel('Detour minus Base (percentage points)')
    axes[1].set_title('All on-path sources, including unselected ones', loc='left')
    for ax in axes: ax.axhline(0, color='#9ba8ad', linewidth=.8)
    fig.text(.07,.02,'12 fresh resets; all three conditions retained. Absolute episode H=440. Same-state execution blocks are distinct from physical sources.',
             fontsize=9, color='#53636b')
    fig.tight_layout(rect=[0,.06,1,1])
    for ext in ('png','svg'): fig.savefig(output/('fresh_value.'+ext), dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Research analysis runs on Quest')
    run = args.run_dir
    contract = verify_run(run)
    anchors = read(run/'anchors.json'); validate_anchors(anchors)
    blocks = {p:read(run/(p+'.json')) for p in ('A','B','C')}
    for phase in blocks:
        if read(run/(phase+'_complete.json'))['records_sha256'] != file_sha256(run/(phase+'.json')):
            raise ValueError('phase changed after completion')
    frozen, forecasts = read(run/'freeze_A.json'), read(run/'forecast_AB.json')
    for rel, digest in forecasts['input_sha256'].items():
        if file_sha256(run/rel) != digest: raise ValueError('pre-C forecast input changed')
    ab = score_choices(anchors, {p:blocks[p] for p in ('A','B')}, frozen['choices'])
    if summarize_ab(ab) != forecasts['forecasts']:
        raise ValueError('A/B estimates changed after seeing C')
    runtime = {p:read(run/(p+'_provenance.json')) for p in ('A','B','C')}
    if len({r['pid'] for r in runtime.values()}) != 3:
        raise ValueError('A/B/C must use separate processes')
    for key in ('host','gpu','versions','commit'):
        if any(runtime[p][key] != runtime['A'][key] for p in ('B','C')):
            raise ValueError('execution contract differs between blocks: '+key)
    if not (frozen['frozen_unix'] < runtime['B']['started_unix'] and
            forecasts['frozen_unix'] < runtime['C']['started_unix']):
        raise ValueError('choice/forecast locks did not precede future blocks')
    all_cells = score_choices(anchors, blocks, frozen['choices'])
    groups = defaultdict(list)
    for r in all_cells:
        groups[r['method'],r['selection_horizon'],r['evaluation_horizon']].append(r)
    overall, all_sources = [], []
    for (method,sh,eh), group in groups.items():
        meta = dict(method=method, selection_horizon=sh, evaluation_horizon=eh)
        for subset, selected in subsets(group,'glass'):
            keys = [k for k in selected[0] if k.startswith(('A_','B_','C_'))]
            source = source_average(selected,keys)
            summary = forecast_comparison(source, **dict(
                bootstrap=contract['bootstrap']['draws'], seed=contract['bootstrap']['seed']))
            summary.update({k:float(np.mean([s[k] for s in source])) for k in keys if k not in summary})
            overall.append(dict(**meta, subset=subset, n_sources=len(source), n_cells=len(selected),
                C_positive_sources=sum(s['C_gain']>0 for s in source),
                C_negative_sources=sum(s['C_gain']<0 for s in source), **summary))
            all_sources += [dict(**meta, subset=subset, **s) for s in source]
    # Show both arms for every source/cell, irrespective of the A choice.
    cells = cells_from_records(anchors,blocks)
    lookup = {a['episode_id']:a for a in anchors}
    support = []
    for c in cells:
        for h in HORIZONS:
            row = dict(id=c['id'], source=c['source'], source_id=lookup[c['id']]['source_id'],
                       condition=c['condition'], horizon=h, shared='shared' in c)
            for phase in blocks:
                for opt, label in ((0,'Base'),(1,'Detour')):
                    values = [c['shared']] if 'shared' in c else c['samples'][phase,opt]
                    success, accident = outcomes(values,h).mean(axis=0)
                    row[phase+'_'+label+'_success'] = float(success)
                    row[phase+'_'+label+'_accident'] = float(accident)
                row[phase+'_gain'] = row[phase+'_Detour_success']-row[phase+'_Base_success']
            support.append(row)
    out = run/'analysis'; out.mkdir(exist_ok=False)
    for name, rows in [('overall',overall),('per_source',all_sources),('per_cell',all_cells),('arm_support',support)]:
        write_csv(out/(name+'.csv'),rows)
    render(overall,support,out)
    lines = ['# Fresh-source prospective execution results','',
        'Twelve prespecified fresh reset attempts; no nominal-success or recovery-success selection.',
        'This population differs from the historical success/hazard-screened sources. Absolute episode actions include prefixes and recovery.', '',
        '| Choice | H | A gain | B gain | C gain [95% source CI] | Source RMSE A / B |',
        '|---|---:|---:|---:|---:|---:|']
    for r in overall:
        if r['subset']!='all' or r['selection_horizon']!=r['evaluation_horizon']: continue
        lines.append(f"| {r['method']} | {r['evaluation_horizon']} | {100*r['A_gain']:.2f} | {100*r['B_gain']:.2f} | {100*r['C_gain']:.2f} [{100*r['C_gain_lo']:.2f}, {100*r['C_gain_hi']:.2f}] | {100*r['A_source_rmse']:.2f} / {100*r['B_source_rmse']:.2f} |")
    lines += ['', 'Gains, intervals and RMSE are percentage points. The privileged references know A outcomes at each state; they are not deployable gates.',
        'A/B/C are separate processes with identical software, checkpoint, GPU and restored bundle/RNG contracts; C remains a noisy execution block.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    manifest = {p.name:file_sha256(p) for p in out.iterdir() if p.is_file()}
    write(out/'manifest.json',manifest)
    count = sum(map(len,blocks.values()))
    if count > contract['max_branches']: raise ValueError('branch budget exceeded')
    write(run/'complete.json',dict(status='COMPLETE', sources=12, prefixes=len(anchors),
        candidates=sum(a['triggered'] for a in anchors), branches=count,
        contract_sha256=file_sha256(run/'contract.json'),
        stage_provenance={p:runtime[p] for p in runtime},
        artifacts={p:file_sha256(run/p) for p in ('A.json','B.json','C.json','freeze_A.json','forecast_AB.json','analysis/manifest.json')}))
    print('\n'.join(lines),flush=True)


if __name__ == '__main__': main()
