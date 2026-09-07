#!/usr/bin/env python3
"""B-only paired evaluation with source-cluster, task-stratified uncertainty."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.expansion.selection_retest import validate_records, write_json, file_sha256


def evaluate(anchors, records, selection, bootstrap=10000, excluded_pairs=()):
    validate_records(anchors, records, 'B')
    excluded_pairs = set(excluded_pairs)
    valid_pairs = {(a['panel_id'], repeat) for a in anchors for repeat in range(4, 8)}
    if not excluded_pairs.issubset(valid_pairs):
        raise ValueError('excluded pair not in B panel')
    pairs = [(anchor, repeat) for anchor in anchors for repeat in range(4, 8)
             if (anchor['panel_id'], repeat) not in excluded_pairs]
    if any(not any(a['panel_id'] == anchor['panel_id'] for a, _ in pairs) for anchor in anchors):
        raise ValueError('diagnostic may not remove an entire anchor')
    lookup = {(r['panel_id'], r['repeat'], r['option']): r for r in records}
    maps = dict(selection['rules'])
    maps['A_reference'] = selection['reference']
    maps['A_best_fixed'] = maps[selection['best_fixed']]
    maps['A_best_simple'] = maps[selection['best_simple']]
    names = list(maps) + ['B_posthoc_single_winner']
    values = []
    for name in names:
        rows = []
        for anchor, repeat in pairs:
            key = anchor['panel_id']
            base = lookup[key, repeat, 0]
            if name == 'B_posthoc_single_winner':
                option = max((0, 1), key=lambda o: (lookup[key, repeat, o]['task_success'],
                             -lookup[key, repeat, o]['catastrophe'], -o))
            else:
                option = maps[name][key]
            row = lookup[key, repeat, option]
            control = anchor['condition'] == 'matched_buffer_control'
            rows.append([row['task_success'], row['catastrophe'], option, row['steps'], row['path_length_m'],
                         row['task_success']-base['task_success'], row['catastrophe']-base['catastrophe'],
                         int(base['task_success'] == 0 and row['task_success'] == 1),
                         int(base['task_success'] == 1 and row['task_success'] == 0),
                         int(control and base['task_success'] == 1 and row['task_success'] == 1),
                         int(control and base['task_success'] == 1)])
        values.append(rows)
    x = np.asarray(values, dtype=float)
    # Resample the same sources for every method; preserve all anchors and repeats.
    sources = sorted({a['physical_source_id'] for a in anchors})
    task_sources = {}
    indices = {}
    for source in sources:
        indices[source] = [i for i, (a, _) in enumerate(pairs) if a['physical_source_id'] == source]
        task = next(a['task_id'] for a in anchors if a['physical_source_id'] == source)
        task_sources.setdefault(task, []).append(source)
    rng = np.random.default_rng(20260907)
    samples = []
    for _ in range(bootstrap):
        picked = [s for group in task_sources.values() for s in rng.choice(group, len(group), replace=True)]
        idx = [i for s in picked for i in indices[s]]
        samples.append(x[:, idx, :].mean(axis=1))
    samples = np.asarray(samples)
    estimates = x.mean(axis=1)
    table = []
    labels = ['success', 'accident', 'refresh', 'steps', 'path_m', 'success_delta', 'accident_delta']
    for i, name in enumerate(names):
        row = {'method': name, 'n_B_pairs': x.shape[1]}
        for j, label in enumerate(labels):
            row[label] = float(estimates[i, j])
            row[label+'_ci95'] = np.quantile(samples[:, i, j], [.025, .975]).tolist()
        row.update(paired_rescues=int(x[i, :, 7].sum()), paired_harms=int(x[i, :, 8].sum()),
                   control_success_retained=int(x[i, :, 9].sum()), control_base_successes=int(x[i, :, 10].sum()))
        table.append(row)
    ref, simple = names.index('A_reference'), names.index('A_best_simple')
    contrast = {'success_delta': float(estimates[ref, 0]-estimates[simple, 0]),
                'ci95': np.quantile(samples[:, ref, 0]-samples[:, simple, 0], [.025, .975]).tolist()}
    return {'table': table, 'reference_minus_best_simple': contrast,
            'sources_with_valid_anchors': len(sources), 'source_counts_by_task': {k: len(v) for k, v in task_sources.items()},
            'bootstrap_seed': 20260907, 'bootstrap_replicates': bootstrap,
            'excluded_pairs': sorted(excluded_pairs),
            'limits': ['training panel only', 'finite-repeat reference is not true oracle',
                       'Risk->BestFixed and DirectQ not run', 'residual runtime distribution with fixed continuation',
                       'A/B temporal drift not excluded', 'single-pair rescues/harms depend on declared repeat pairing']}


def render(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = {r['method']: r for r in result['table']}
    names = ['B_posthoc_single_winner', 'A_reference', 'A_best_fixed', 'A_best_simple']
    labels = ['Same-result winner\n(privileged)', 'A select / B evaluate', 'A best fixed / B', 'A best simple / B']
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    for i, name in enumerate(names):
        row = rows[name]
        lo, hi = np.asarray(row['success_delta_ci95'])*100
        ax.vlines(i, lo, hi, color='#245970', linewidth=2)
        ax.scatter(i, row['success_delta']*100, color='#245970', s=55)
    ax.axhline(0, color='gray', linewidth=1)
    ax.set_xticks(range(len(names)), labels)
    ax.set_ylabel('Task success difference vs Base (percentage points)')
    ax.set_title('Training-source selection retest: common B evaluation')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(out/'main_figure.png', dpi=180)
    fig.savefig(out/'main_figure.svg')
    plt.close(fig)
    lines = ['# 训练来源选择收益复验', '', '所有方法使用相同 B 重复。区间按任务分层、来源配对重采样。', '',
             '| 方法 | 成功率 | 相对 Base 差 [95% CI] | 事故率 | 控制成功保持 | 配对救回/损害 | Refresh 次数 | 平均步数 |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in result['table']:
        ci = r['success_delta_ci95']
        lines.append(f"| {r['method']} | {r['success']:.2%} | {r['success_delta']*100:.2f} [{ci[0]*100:.2f}, {ci[1]*100:.2f}] pp | {r['accident']:.2%} | {r['control_success_retained']}/{r['control_base_successes']} | {r['paired_rescues']}/{r['paired_harms']} | {round(r['refresh']*r['n_B_pairs'])} | {r['steps']:.1f} |")
    c = result['reference_minus_best_simple']
    lines += ['', f"A 参考相对 A 所选简单规则的 B 成功率差：{c['success_delta']*100:.2f} pp，95% CI [{c['ci95'][0]*100:.2f}, {c['ci95'][1]*100:.2f}]。", '',
              '不构成可部署方法或新来源泛化证据；未运行 Risk→BestFixed、DirectQ。宽区间不能解释为无效。',
              '事故为 75 N 协议代理。控制保持及救回/损害依赖相同重复编号的配对；主图报告边际成功率差。',
              '中性控制的分母为 B Base 成功重复数，0/0 时无可评价的保持率。', '']
    if 'cross_job_pair_diagnostic' in result:
        contrast = result['cross_job_pair_diagnostic']['reference_minus_best_simple']
        lines += ['存储中断后 B 分布于两个作业；b17/repeat5 的两选项跨越作业边界。',
                  f"预声明诊断（仅去掉这对重复）：参考相对简单规则差 {contrast['success_delta']*100:.2f} pp，"
                  f"95% CI [{contrast['ci95'][0]*100:.2f}, {contrast['ci95'][1]*100:.2f}]。", '']
    (out/'RESULTS_ZH.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = args.run_dir
    if json.loads((root/'complete.json').read_text())['status'] != 'COMPLETE':
        raise ValueError('incomplete run')
    selection = json.loads((root/'freeze.json').read_text())
    if file_sha256(root/'freeze.json') != (root/'freeze.sha256').read_text().strip():
        raise ValueError('freeze hash mismatch')
    for filename, key in [('A.json', 'A_sha256'), ('anchors.json', 'anchors_sha256')]:
        if file_sha256(root/filename) != selection[key]:
            raise ValueError('frozen selection input changed')
    anchors = json.loads((root/'anchors.json').read_text())
    records = json.loads((root/'B.json').read_text())
    result = evaluate(anchors, records, selection)
    provenance = json.loads((root/'provenance.json').read_text())
    if provenance.get('kind') == 'storage_failure_recovery':
        result['cross_job_pair_diagnostic'] = evaluate(anchors, records, selection,
                                                      excluded_pairs=[('b17', 5)])
    out = args.output_dir or root/'analysis'
    out.mkdir(exist_ok=False)
    write_json(out/'metrics.json', result)
    render(result, out)


if __name__ == '__main__':
    main()
