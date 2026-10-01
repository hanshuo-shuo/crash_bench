"""Preliminary figure and table from completed legal-action contact replays."""
import csv
import hashlib
import json
import os
from pathlib import Path


def summarize(item):
    if not item['replay_verified'] or 0 not in item['exact_physics_verified_checkpoints']:
        raise RuntimeError('Unverified physical replay')
    robot = [e for e in item['events'] if e['category'] == 'robot']
    target = [e for e in item['events'] if e['category'] == 'target']
    return {'run': item['run_id'], 'state': item['state'], 'task_complete': item['source_success'],
            'official_proxy_safe_complete': item['source_safe_success'],
            'complete_without_robot_target_protected_contact': item['source_success'] and not robot and not target,
            'max_l1_mm': item['source_proxy_max_l1_m'] * 1000,
            'robot_contact_actions': len(set(e['action'] for e in robot)),
            'target_contact_actions': len(set(e['action'] for e in target)),
            'min_robot_contact_distance_mm': min([e['distance_m'] * 1000 for e in robot] or [0.]),
            'robot_body_pairs': sorted(set(tuple(e['body_pair']) for e in robot)),
            'replay_job': item['slurm_job']}


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    root = Path(root)
    config = json.loads((root / 'inputs.json').read_text())
    rows = []; hashes = {}
    for name in config['contact_roots']:
        folder = Path(name)
        if not (folder / 'CONTACT_SUBSTEPS_COMPLETE.json').is_file():
            raise RuntimeError('Incomplete contact audit')
        file = folder / 'CONTACT_SUBSTEPS.json'
        hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        for item in json.loads(file.read_text()):
            source = Path(item['source_root']) / 'runs' / item['run_id'] / 'row.json'
            if hashlib.sha256(source.read_bytes()).hexdigest() != item['source_row_sha256']:
                raise RuntimeError('Source row changed')
            rows.append(summarize(item))
    out = root / 'report'; out.mkdir()
    (out / 'contact_evidence.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (out / 'contact_evidence.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    screen = [r for r in rows if '_screen_' in r['run']]
    values = np.asarray([[r['task_complete'], r['official_proxy_safe_complete'],
                          r['complete_without_robot_target_protected_contact']] for r in screen], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.imshow(values, vmin=0, vmax=1, cmap='YlGn')
    ax.set_xticks(range(3)); ax.set_xticklabels(['Task completed', 'Official displacement\nproxy safe completion',
                                               'Completed without robot/target\ncontact with protected obstacle'], fontsize=9)
    ax.set_yticks(range(len(screen))); ax.set_yticklabels([r['state'] for r in screen])
    for i in range(len(screen)):
        for j in range(3): ax.text(j, i, 'Yes' if values[i,j] else 'No', ha='center', va='center')
    ax.set_title('Screening reference trajectories (one execution per state)')
    fig.tight_layout(); fig.savefig(out / 'contact_witnesses.png', dpi=180); plt.close(fig)
    text = ['# 阶段性接触核验：不能把位移代理通过等同于无受保护接触', '',
            '这是六个诊断状态固定参考的单次筛查及三条几何修正AEGIS的机制回放。完整304次初始阶段与受控分叉尚未完成。不是成功概率认证或总体性能评估。',
            '每条保存的合法动作在相同官方初始状态重执行，qpos/qvel/ctrl在初始与已保存检查点逐字节匹配，最大障碍位移和任务结果复现。每个原MuJoCo积分步后只读接触；不额外积分、移动对象或写qpos。',
            '“无上述接触”只指机器人／目标物体与受保护障碍之间；支撑接触与正常抓取接触不属于这一列。主结果仍使用原1mm位移代理。', '',
            '|参考状态（各一次）|任务完成|官方代理安全完成|最大L1/mm|机器人接触动作数|完成且无上述接触|',
            '|---|---|---|---:|---:|---|']
    for r in screen:
        text.append('|%s|%s|%s|%.6f|%d|%s|' % (r['state'], r['task_complete'],
            r['official_proxy_safe_complete'], r['max_l1_mm'], r['robot_contact_actions'],
            r['complete_without_robot_target_protected_contact']))
    text += ['', 'Spatial9/15的参考在官方位移评分下安全完成，但link5仍接触了酒瓶。Object0/2/5的参考完成且没有观察到机器人／目标-酒瓶接触，新重复验证仍在进行。',
             '几何修正AEGIS的Spatial9、Spatial15及Object0回放都捕捉到link5接触。Object0的短暂接触只在物理子步中出现，动作端点检查会漏掉。',
             '这与末端单椭球代理无法覆盖全部真实机器人碰撞几何的机制一致；还没有单独排除离散时间控制和跟踪误差，不能归因于唯一原因。',
             '后续研究判断必须分清官方代理下的有效续接与更强的受保护接触安全。已有特权参考见证也不能单独支持普通候选选择学习。']
    (out / 'CONTACT_REPORT.md').write_text('\n'.join(text) + '\n')
    (out / 'REPORT_COMPLETE.json').write_text(json.dumps({
        'input_files_sha256': hashes, 'code_commit': (root / 'SOURCE_COMMIT').read_text().strip(),
        'slurm_job': os.environ['SLURM_JOB_ID'], 'scope': 'preliminary contact evidence only',
        'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}, indent=2) + '\n')


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
