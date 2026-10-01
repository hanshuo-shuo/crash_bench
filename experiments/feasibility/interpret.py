"""Final, count-based decision evidence; separate expert and finite-library witnesses."""
import csv
import hashlib
import json
import os
from pathlib import Path
from protocol import STATES, CONDITIONS, BRANCH_CONDITIONS, INTERVENTION_CANDIDATES, CHECKPOINTS
from analysis import validate_rows, validate_branch_coverage, continuation_comparisons


def outcome_counts(rows):
    return {'n': len(rows),
            'safe_complete': sum(r['success'] and not r['collided'] for r in rows),
            'safe_incomplete': sum(not r['success'] and not r['collided'] for r in rows),
            'unsafe_complete': sum(r['success'] and r['collided'] for r in rows),
            'unsafe_incomplete': sum(not r['success'] and r['collided'] for r in rows),
            'method_exits': sum(bool(r.get('exited')) for r in rows)}


def finite_library(rows, state, step, extra):
    selected = [r for r in rows if r['state'] == state and r['branch_step'] == step
                and r['extra_budget'] == extra]
    by_repeat = {}
    for row in selected:
        by_repeat.setdefault(row['repeat'], {})[row['condition']] = row
    complete = {r: g for r, g in by_repeat.items() if set(g) == set(BRANCH_CONDITIONS)}
    counts = {c: sum(g[c]['safe_success'] for g in complete.values()) for c in BRANCH_CONDITIONS}
    ordinary = {c: counts[c] for c in INTERVENTION_CANDIDATES}
    return {'n': len(complete), 'repeats': sorted(complete), 'counts': counts,
            'ordinary_library_any': sum(any(g[c]['safe_success'] for c in INTERVENTION_CANDIDATES)
                                        for g in complete.values()),
            'best_fixed_candidate_count': max(ordinary.values(), default=0),
            'best_fixed_candidates': [c for c, n in ordinary.items() if n and n == max(ordinary.values())],
            'scope': 'existence in prespecified library is an oracle, not a deployed selector'}


def evidence(rows):
    result = []
    for state in STATES:
        sid = state['id']
        initial = [r for r in rows if r['state'] == sid and r['branch_step'] is None]
        conditions = {c: outcome_counts([r for r in initial if r['condition'] == c]) for c in CONDITIONS}
        reference = [r for r in initial if r['condition'] == 'reference' and r['validation']]
        forks = []
        for step in CHECKPOINTS:
            for extra in sorted({0, step}):
                item = finite_library(rows, sid, step, extra)
                if item['n']:
                    forks.append(dict(item, step=step, extra=extra,
                                      suffix_budget=300 + extra - step))
        identity_rows = [r for r in initial if r['condition'] == 'identity']
        result.append({'state': sid, 'role': state['role'], 'conditions': conditions,
                       'identity_changed_runs': sum(r['identity_changed'] for r in identity_rows),
                       'reference_validation': outcome_counts(reference),
                       'reference_variant': reference[0]['variant'] if reference else None,
                       'forks': forks, 'time_budget': continuation_comparisons(rows, sid)})
    return result


def generate(target, output, supplement_file=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    target, output = Path(target).resolve(), Path(output).resolve()
    if not (target / 'COMPLETE.json').is_file() or (target / 'STOP.json').exists():
        raise RuntimeError('Complete scientific root required')
    rows = json.loads((target / 'rows.json').read_text())
    quality = validate_rows(rows, require_initial_complete=True)
    approved = json.loads((target / 'BRANCH_GATE.json').read_text())['eligible_states']
    coverage = {s: validate_branch_coverage(target, rows, s) for s in approved}
    data = evidence(rows)
    contact_rows = []; alternatives = []; supplement_hashes = {}
    if supplement_file is not None:
        supplemental = json.loads(Path(supplement_file).read_text())
        for path in supplemental.get('contact_roots', []):
            folder = Path(path)
            if not (folder / 'CONTACT_SUBSTEPS_COMPLETE.json').is_file():
                raise RuntimeError('Contact audit incomplete: ' + str(folder))
            file = folder / 'CONTACT_SUBSTEPS.json'
            supplement_hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
            for item in json.loads(file.read_text()):
                if not item['replay_verified'] or 0 not in item['exact_physics_verified_checkpoints']:
                    raise RuntimeError('Contact replay unverified')
                source = Path(item['source_root']) / 'runs' / item['run_id'] / 'row.json'
                if hashlib.sha256(source.read_bytes()).hexdigest() != item['source_row_sha256']:
                    raise RuntimeError('Contact source row changed')
                robot = [e for e in item['events'] if e['category'] == 'robot']
                target_events = [e for e in item['events'] if e['category'] == 'target']
                contact_rows.append({k: item[k] for k in ['run_id', 'state', 'source_safe_success',
                    'source_success', 'source_proxy_max_l1_m', 'exact_physics_verified_checkpoints', 'slurm_job']})
                contact_rows[-1].update(robot_contact_actions=sorted(set(e['action'] for e in robot)),
                    target_contact_actions=sorted(set(e['action'] for e in target_events)),
                    robot_contact_samples=len(robot), target_contact_samples=len(target_events),
                    completed_without_robot_target_protected_contact=item['source_success'] and not robot and not target_events,
                    robot_body_pairs=sorted(set(tuple(e['body_pair']) for e in robot)))
        for path in supplemental.get('alternative_roots', []):
            folder = Path(path)
            if not (folder / 'ALTERNATIVE_SCREEN_COMPLETE.json').is_file():
                raise RuntimeError('Alternative screen incomplete')
            file = folder / 'rows.json'
            supplement_hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
            alternatives.extend(json.loads(file.read_text()))
    output.mkdir(exist_ok=False)
    (output / 'decision_evidence.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    (output / 'contact_evidence.json').write_text(json.dumps({'contact_replays': contact_rows,
        'post_contact_fixed_alternative': alternatives}, ensure_ascii=False, indent=2) + '\n')
    summary = []
    table = ['# 安全未完成诊断：最终决策证据', '',
             '八个预先指定的暴露状态，六个诊断、两个历史安全对照。计数只描述本诊断集；不估计总体性能或认证成功概率。',
             '参考新验证是十次fresh环境执行，但物理初始状态固定、参考确定性。新策略种子不构成参考的物理扰动；重复成功证明该执行可复现，不能认证高概率。',
             '原始VLM回答冻结，身份修正与碰撞几何修正分开；六个诊断原回答均正确，身份修正是no-op，不能用零增益排除完整系统的身份误认风险。',
             '全部受控首次观察、空队列、控制器、物理/力状态和RNG配对；RGB原生变体和原生推理数值差异保留。中途分叉重执行相同前缀，完整状态核验，不直接写qpos。', '',
             '|状态|参考新验证安全完成|原AEGIS安全完成／安全未完成|身份+几何安全完成／安全未完成|普通候选续接证据|',
             '|---|---:|---|---|---|']
    for x in data:
        ref, raw, corrected = x['reference_validation'], x['conditions']['raw'], x['conditions']['identity_geometry']
        equal = [f for f in x['forks'] if f['suffix_budget'] == 300]
        best = max(equal, key=lambda f: (f['ordinary_library_any'] / f['n'], f['ordinary_library_any']), default=None)
        if best:
            description = 't=%d：固定库至少一个 %d/%d；最佳单一候选 %s %d/%d' % (
                best['step'], best['ordinary_library_any'], best['n'], ','.join(best['best_fixed_candidates']) or '无',
                best['best_fixed_candidate_count'], best['n'])
        else:
            description = '未分叉（无资格）；不等于候选全失败'
        table.append('|%s|%d/%d|%d/%d；%d/%d|%d/%d；%d/%d|%s|' % (
            x['state'], ref['safe_complete'], ref['n'], raw['safe_complete'], raw['n'], raw['safe_incomplete'], raw['n'],
            corrected['safe_complete'], corrected['n'], corrected['safe_incomplete'], corrected['n'], description))
        for c, values in x['conditions'].items():
            summary.append(dict(state=x['state'], role=x['role'], condition=c, **values))
        summary.append(dict(state=x['state'], role=x['role'], condition='reference_validation', **ref))
    with (output / 'outcome_categories.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    table += ['', '本文主矩阵的“安全”沿用官方障碍L1位移代理；接触核验是额外机制证据，未改变1mm边界。',
              '上表省略的不安全完成与不安全未完成均在outcome_categories.csv及下图单独报告。',
              '固定库“至少一个”是事后存在性见证，需要oracle选中；不是已实现选择器的成功率。最佳单一候选同时报告，避免把候选库上界当作部署能力。', '',
              '|状态|后期检查点|共同幸存种子|参考早→晚（同300续接步）|普通库早→晚（同300续接步）|只加预算增益／损失|',
              '|---|---:|---:|---:|---:|---:|']
    for x in data:
        for v in x['time_budget']:
            table.append('|%s|%d|%d|%d→%d|%d→%d|%d／%d|' % (
                x['state'], v['checkpoint'], len(v['same_surviving_prefix_repeats']),
                v['reference_early_witness'], v['reference_late_witness'],
                v['early_any_candidate_witness'], v['late_any_candidate_witness'],
                v['extra_budget_gains'], v['extra_budget_losses']))
    table += ['', '早晚只比较同一重复且后期尚未碰撞的前缀。不存在的检查点不记失败；碰撞后缺失说明安全窗口已关闭，不能当作“未碰撞但不可恢复”。',
              '相同预算下固定参考失败仍不能证明真实不可行；专家抓取/运输能力不足与物理不可行尚须分开。普通库没有成功，也不能由特权参考成功推出该库选择学习值得做。', '',
              '## 四种结果的研究判断约束', '',
              '|实际证据模式|允许的研究判断|', '|---|---|',
              '|单独身份或几何修正后多数安全完成且有配对增益|优先感知／代理表示，注明是哪一个因子及交互|',
              '|独立安全见证；受控AEGIS仍失败；普通固定候选在相同条件下重复有效|方向二候选证据；按单一候选和oracle库差距、时间变化，选择“选择”或“介入时机”之一|',
              '|早期有见证；同预算、同幸存种子下后期见证减少|方向一证据；限定于固定参考／有限候选能力，不能声称真实不可行|',
              '|找不到独立见证；或现有候选库没有有效续接|未知或候选库不足；不训练无解分类器，不堆选择模块|', '',
              '本表是证据约束，具体研究建议须结合各状态计数与轨迹审查。不得把四种结果都包装成同一成功论文。']
    if contact_rows:
        table += ['', '## 官方代理与实际受保护障碍接触分开', '',
                  '只读回放所有保存动作；在每个原MuJoCo积分步后读取接触，不增加积分或改写状态。初始及已存检查点的qpos/qvel/ctrl逐字节核验，官方最大位移和任务结果复现。',
                  '下表动作数是存在接触的不同动作数；多个几何对或子步不计成多次独立碰撞。与桌面的支撑接触不在机器人／目标接触计数中。', '',
                  '|轨迹|官方代理安全完成|最大位移/mm|机器人接触动作数|目标物体接触动作数|完成且无上述受保护接触|',
                  '|---|---|---:|---:|---:|---|']
        for x in contact_rows:
            table.append('|%s|%s|%.6f|%d|%d|%s|' % (x['run_id'], x['source_safe_success'],
                x['source_proxy_max_l1_m'] * 1000, len(x['robot_contact_actions']),
                len(x['target_contact_actions']), x['completed_without_robot_target_protected_contact']))
        table += ['', 'Spatial9/15的rim参考通过官方位移代理，但记录到了robot0_link5与酒瓶接触；不能据此声称无受保护接触的安全可行性，更不能用这类早期见证声称强安全的真实可行性衰退。普通续接计数也仍仅是官方代理下的结果，未逐条完成接触审计。']
        screened = [x for x in contact_rows if '_screen_' in x['run_id']]
        if screened:
            values = np.array([[x['source_success'], x['source_safe_success'],
                                x['completed_without_robot_target_protected_contact']] for x in screened], dtype=float)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.imshow(values, vmin=0, vmax=1, cmap=ListedColormap(['#f7ecd0', '#b4d9c7']), aspect='auto')
            ax.set_xticks(range(3)); ax.set_xticklabels(['Task completed', 'Official displacement\nproxy safe completion',
                                                       'Completed without robot/target\ncontact with protected obstacle'], fontsize=9)
            ax.set_yticks(range(len(screened))); ax.set_yticklabels([x['state'] for x in screened])
            for i in range(len(screened)):
                for j in range(3): ax.text(j, i, 'Yes' if values[i,j] else 'No', ha='center', va='center', color='#17382d')
            ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(screened), 1), minor=True)
            ax.grid(which='minor', color='white', linewidth=1.5)
            ax.tick_params(which='minor', bottom=False, left=False)
            ax.set_title('Independent reference evidence: score and physical contacts differ')
            fig.tight_layout(); fig.savefig(output / 'protected_contact_evidence.png', dpi=180); plt.close(fig)
    if alternatives:
        table += ['', '## 接触发现后的单一固定补试', '',
                  'rim_south25只把规划中的盘内放置waypoint向负Y偏25mm，未移动真实盘、障碍或对象，仍经原7D动作。三个Spatial状态均未完成且有机器人-酒瓶接触；没有再调参数或评分边界。此失败不证明不存在安全路径。', '',
                  '|状态|完成|官方代理安全完成|最大位移/mm|接触样本数|', '|---|---|---|---:|---:|']
        for x in alternatives:
            table.append('|%s|%s|%s|%.6f|%d|' % (x['state'], x['success'], x['official_safe_success'],
                         x['max_obstacle_l1_m'] * 1000, x['contact_samples']))
    (output / 'DECISION_EVIDENCE.md').write_text('\n'.join(table) + '\n')

    categories = ['safe_complete', 'safe_incomplete', 'unsafe_complete', 'unsafe_incomplete']
    labels = ['Safe complete', 'Safe incomplete', 'Unsafe complete', 'Unsafe incomplete']
    colors = ['#299268', '#83b8db', '#d8753d', '#ab3653']
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    for ax, x in zip(axes.flat, data):
        groups = [x['conditions'][c] for c in CONDITIONS] + [x['reference_validation']]
        lower = np.zeros(6)
        for c, label, color in zip(categories, labels, colors):
            values = np.array([g[c] / g['n'] if g['n'] else np.nan for g in groups])
            ax.bar(np.arange(6), values, bottom=lower, color=color, label=label, width=.72)
            lower += values
        ax.set_xticks(range(6)); ax.set_xticklabels(['Pi', 'AEGIS', 'ID', 'Geom', 'ID+G', 'Ref'], rotation=30, ha='right')
        ax.set_ylim(0, 1.12); ax.set_title(x['state'], fontsize=10)
        for i, g in enumerate(groups): ax.text(i, 1.03, 'n=%d' % g['n'], ha='center', fontsize=7)
        ax.set_ylabel('Observed fraction')
    handles, legend = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='lower center', ncol=4, frameon=False)
    fig.suptitle('Fixed diagnostic states: safety and completion remain separate', fontsize=13)
    fig.tight_layout(rect=(0,.07,1,.95)); fig.savefig(output / 'outcome_categories.png', dpi=180); plt.close(fig)

    eligible = [x for x in data if x['forks']]
    if eligible:
        fig, axes = plt.subplots(len(eligible), 1, figsize=(10, 3.2 * len(eligible)), squeeze=False)
        for ax, x in zip(axes.flat, eligible):
            equal = [f for f in x['forks'] if f['suffix_budget'] == 300]
            remaining = [f for f in x['forks'] if f['extra'] == 0]
            ax.plot([f['step'] for f in equal], [f['counts']['reference'] / f['n'] for f in equal], 'o-', color='#855ba2', label='Privileged reference, 300 suffix')
            ax.plot([f['step'] for f in equal], [f['ordinary_library_any'] / f['n'] for f in equal], 'o-', color='#26866f', label='Ordinary library existence, 300 suffix (oracle)')
            ax.plot([f['step'] for f in equal], [f['best_fixed_candidate_count'] / f['n'] for f in equal], 's--', color='#4374ae', label='Best single candidate per checkpoint, 300 suffix')
            ax.plot([f['step'] for f in remaining], [f['ordinary_library_any'] / f['n'] for f in remaining], 'x:', color='#b77030', label='Ordinary library existence, remaining budget')
            for f in equal: ax.text(f['step'], 1.055, 'n=%d' % f['n'], ha='center', fontsize=8)
            ax.set_ylim(-.06, 1.16); ax.set_xticks(CHECKPOINTS); ax.set_xlabel('Uncollided common-prefix checkpoint / actions')
            ax.set_ylabel('Observed safe completion'); ax.set_title(x['state'] + ' (conditional on surviving prefixes)')
        axes.flat[0].legend(loc='lower left', fontsize=8)
        fig.tight_layout(); fig.savefig(output / 'time_and_budget.png', dpi=180); plt.close(fig)

    provenance = {'target_root': str(target), 'scientific_commit': (target / 'SOURCE_COMMIT').read_text().strip(),
                  'analysis_commit': os.environ.get('CB_ANALYSIS_COMMIT'), 'slurm_job': os.environ.get('SLURM_JOB_ID'),
                  'input_rows_sha256': hashlib.sha256((target / 'rows.json').read_bytes()).hexdigest(),
                  'quality': quality, 'coverage': coverage, 'supplement_files_sha256': supplement_hashes,
                  'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}}
    (output / 'INTERPRET_COMPLETE.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    import sys
    generate(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
