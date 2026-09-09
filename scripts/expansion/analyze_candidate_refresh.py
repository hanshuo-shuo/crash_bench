#!/usr/bin/env python3
"""Repeat-aware, horizon-specific descriptive analysis of selected train anchors."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.expansion.candidate_refresh import at_horizon, freeze_horizons
from scripts.expansion.selection_retest import validate_records, file_sha256, write_json
from scripts.expansion.analyze_selection_retest import evaluate


def summarize(anchors, a, b, frozen, bootstrap=10000):
    for phase, records in [('A', a), ('B', b)]:
        validate_records(anchors, records, phase)
    rebuilt = freeze_horizons(anchors, a)
    if any(rebuilt[str(h)] != frozen[str(h)] for h in (100, 200)):
        raise ValueError('A decisions differ from freeze')
    result = {'groups': {}, 'anchors': [], 'limits': [
        'Historical-success-selected training configurations; not population success estimates.',
        'Eight physical sources; repeats are not independent sources.',
        'New anchors, not exact replay of historical D5 states.',
        '200-step noncompletion does not establish permanent failure.',
        'Matched controls have their own freshly generated neutral-condition anchors.',
        'Pair transitions depend on declared repeat pairing; marginal rates are primary.',
        'A/B evaluate selection on the same anchors, not new-source generalization.',
        'No learned method or DirectQ fitted in this diagnostic.']}
    for h in (100, 200):
        records = at_horizon(b, h)
        result['groups'][str(h)] = {}
        for condition in ('stale', 'matched_buffer_control'):
            selected = [x for x in anchors if x['condition'] == condition]
            ids = {x['panel_id'] for x in selected}
            if selected:
                result['groups'][str(h)][condition] = evaluate(selected,
                    [r for r in records if r['panel_id'] in ids], frozen[str(h)], bootstrap=bootstrap)
    for anchor in anchors:
        key = anchor['panel_id']
        entry = dict(anchor, repeats={}, paired_B={})
        for phase, records in [('A', a), ('B', b), ('pooled_descriptive', a+b)]:
            entry['repeats'][phase] = {}
            for h in (100, 200):
                entry['repeats'][phase][str(h)] = {}
                for o in (0, 1):
                    rows = [r for r in at_horizon(records,h) if r['panel_id']==key and r['option']==o]
                    counts = Counter((r['task_success'],r['catastrophe'],r['termination_reason']) for r in rows)
                    entry['repeats'][phase][str(h)][str(o)] = {
                        'n':len(rows),'successes':sum(r['task_success'] for r in rows),
                        'accidents':sum(r['catastrophe'] for r in rows),
                        'success_steps':[r['steps'] for r in rows if r['task_success']],
                        'mean_inference_calls':sum(r['inference_calls'] for r in rows)/len(rows),
                        'mean_steps':sum(r['steps'] for r in rows)/len(rows),
                        'terminal_disagreement_pairs':sum(x!=y for i,x in enumerate([(r['task_success'],r['catastrophe']) for r in rows]) for y in [(r['task_success'],r['catastrophe']) for r in rows][i+1:]),
                        'terminal_counts':[dict(success=x[0],accident=x[1],reason=x[2],n=n) for x,n in sorted(counts.items())]}
        lookup={(r['repeat'],r['option']):r['horizons'] for r in b if r['panel_id']==key}
        counts=Counter()
        for repeat in range(4,8):
            base, refresh = lookup[repeat,0],lookup[repeat,1]
            for h in ('100','200'):
                counts['rescue_'+h] += int(not base[h]['task_success'] and refresh[h]['task_success'])
                counts['harm_'+h] += int(base[h]['task_success'] and not refresh[h]['task_success'])
            if not base['100']['task_success'] and refresh['100']['task_success']:
                counts['base_catches_up_by_200'] += int(base['200']['task_success'])
                counts['rescue_persists_at_200'] += int(not base['200']['task_success'])
            counts['accident_to_success_200'] += int(base['200']['catastrophe'] and refresh['200']['task_success'])
        entry['paired_B']=dict(counts)
        result['anchors'].append(entry)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(); root=args.run_dir
    read=lambda name:json.loads((root/name).read_text())
    if read('complete.json')['status']!='COMPLETE': raise ValueError('incomplete run')
    frozen=read('freeze.json')
    if file_sha256(root/'freeze.json')!=(root/'freeze.sha256').read_text().strip(): raise ValueError('freeze changed')
    for name,key in [('A.json','A_sha256'),('anchors.json','anchors_sha256')]:
        if file_sha256(root/name)!=frozen[key]: raise ValueError('frozen input changed')
    anchors,a,b=read('anchors.json'),read('A.json'),read('B.json')
    for anchor in anchors:
        if file_sha256(root/anchor['panel_id']/'bundle.pkl')!=anchor['bundle_sha256']: raise ValueError('bundle changed')
    for r in a+b:
        trace=root/r['panel_id']/f"{r['phase']}_r{r['repeat']}_o{r['option']}.pkl.gz"
        if file_sha256(trace)!=r['trace_sha256']: raise ValueError('trace changed')
    result=summarize(anchors,a,b,frozen)
    out=root/'analysis';out.mkdir(exist_ok=False)
    write_json(out/'metrics.json',result)
    lines=['# 历史候选 Refresh 收益诊断','',
        'B 重复为主要评价；A 用于固定选项。100/200 步来自同一条轨迹。',
        '历史成功筛选的训练配置，不用于估计总体成功率；200 步仍失败不代表永久失败。','',
        '| 时域 | 条件 | Base 成功 | Refresh 成功 | 差值及来源 bootstrap 95% CI | Base / Refresh 事故 |',
        '|---|---|---:|---:|---|---|']
    for h,groups in result['groups'].items():
        for condition,g in groups.items():
            rows={r['method']:r for r in g['table']};base,refresh=rows['Base'],rows['AlwaysRefresh'];ci=refresh['success_delta_ci95']
            lines.append(f"| {h} | {condition} | {base['success']:.1%} | {refresh['success']:.1%} | {refresh['success_delta']:.1%} [{ci[0]:.1%}, {ci[1]:.1%}] | {base['accident']:.1%} / {refresh['accident']:.1%} |")
    lines += ['','逐锚点 A/B 重复、完成步数、同选项波动、推理调用成本及配对转换见 metrics.json；完整输入、动作、状态及 continuation 见原始 bundle 和轨迹。','']
    (out/'RESULTS_ZH.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
