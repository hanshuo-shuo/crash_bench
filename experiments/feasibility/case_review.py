"""Compute-side review of recorded Object5 reference actions; no new execution."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys


def review(target, interpreted, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    target, interpreted, output = map(lambda p: Path(p).resolve(), (target, interpreted, output))
    if not (target / 'COMPLETE.json').is_file() or (target / 'STOP.json').exists():
        raise RuntimeError('Complete scientific root required')
    audit = json.loads((interpreted / 'COMMON_TRAJECTORY_AUDIT.json').read_text())
    if audit['status'] != 'passed':
        raise RuntimeError('Passed full common-trajectory audit required')
    rows = json.loads((target / 'rows.json').read_text())
    selected = [r for r in rows if r['state'] == 'object_05' and r['condition'] == 'reference'
                and r['branch_step'] is not None and r['extra_budget'] == r['branch_step']]
    expected = {(r, t) for r in range(5) for t in [0, 27, 77, 152, 252]}
    if len(selected) != 25 or {(r['repeat'], r['branch_step']) for r in selected} != expected:
        raise RuntimeError('All Object5 equal-budget reference executions required')
    inputs, traces, facts, phases = {}, {}, [], []
    for row in selected:
        folder = target / 'runs' / row['run_id']
        for name in ['row.json', 'steps.jsonl']:
            file = folder / name
            inputs[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        trace = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]
        if len(trace) != row['end_step'] or [x['step'] for x in trace] != list(range(1, len(trace) + 1)):
            raise RuntimeError('Recorded action coverage differs: ' + row['run_id'])
        suffix = [x for x in trace if x['step'] > row['branch_step']]
        if not suffix or len(suffix) > 300:
            raise RuntimeError('Incorrect suffix budget')
        traces[row['run_id']] = suffix
        target_positions = np.asarray([x['target_pos'] for x in suffix])
        eef_positions = np.asarray([x['eef_pos'] for x in suffix])
        record = {k: row[k] for k in ['run_id', 'repeat', 'branch_step', 'success', 'safe_success', 'end_step']}
        record.update(suffix_actions=len(suffix), final_logged_phase=suffix[-1]['reference_phase'],
                      first_logged_target_xyz_m=target_positions[0].tolist(), last_logged_target_xyz_m=target_positions[-1].tolist(),
                      last_logged_eef_xyz_m=eef_positions[-1].tolist(),
                      target_z_min_m=float(target_positions[:, 2].min()),
                      target_z_max_m=float(target_positions[:, 2].max()),
                      scope='observed recorded motion, no grasp or infeasibility classifier')
        facts.append(record)
        counts = {}
        for item in suffix:
            phase = item['reference_phase']
            counts[phase] = counts.get(phase, 0) + 1
        phases += [dict(run_id=row['run_id'], repeat=row['repeat'], checkpoint=row['branch_step'],
                        logged_phase=phase, suffix_action_count=count) for phase, count in counts.items()]
    output.mkdir(exist_ok=False)
    for filename, data in [('object05_reference_motion.csv', facts), ('object05_logged_phases.csv', phases)]:
        with (output / filename).open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    (output / 'object05_reference_motion.json').write_text(json.dumps(facts, indent=2) + '\n')
    example_steps = [0, 77, 152, 252]
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), squeeze=False)
    for pair, checkpoint in zip(axes, example_steps):
        name = 'object_05_r00_t%03d_b%03d_reference' % (checkpoint, checkpoint)
        row = next(r for r in selected if r['run_id'] == name)
        suffix = traces[name]
        target_positions = np.asarray([x['target_pos'] for x in suffix])
        eef_positions = np.asarray([x['eef_pos'] for x in suffix])
        elapsed = np.arange(1, len(suffix) + 1)
        left, right = pair
        left.plot(target_positions[:, 0], target_positions[:, 1], color='#c35b3d', label='Target carton')
        left.plot(eef_positions[:, 0], eef_positions[:, 1], color='#236b88', label='End effector')
        left.scatter(*target_positions[0, :2], color='#c35b3d', marker='o', s=30)
        left.scatter(*target_positions[-1, :2], color='#c35b3d', marker='x', s=45)
        left.set_xlim(-.28, .15); left.set_ylim(-.56, .36)
        left.set_xlabel('World X / m'); left.set_ylabel('World Y / m')
        left.set_title('Fork at t=%d: task complete=%s' % (checkpoint, row['success']))
        right.plot(elapsed, target_positions[:, 2], color='#c35b3d', label='Target carton')
        right.plot(elapsed, eef_positions[:, 2], color='#236b88', label='End effector')
        last = None
        for index, item in enumerate(suffix):
            if item['reference_phase'] != last:
                right.axvline(index + 1, color='#bbbbbb', lw=.6, alpha=.7)
                right.text(index + 1, .97, item['reference_phase'], rotation=90, fontsize=7,
                           transform=right.get_xaxis_transform(), va='top')
                last = item['reference_phase']
        right.set_xlim(0, 305); right.set_ylim(0, .55)
        right.set_xlabel('Executed suffix action (300 maximum)'); right.set_ylabel('World height / m')
        right.set_title('Logged phases; prefix excluded')
        left.grid(alpha=.15); right.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8, loc='upper left'); axes[0, 1].legend(fontsize=8, loc='lower right')
    fig.suptitle('Object5 recorded reference motion: repeat 0 illustration, not a feasibility certificate', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(output / 'object05_reference_motion.png', dpi=180); plt.close(fig)
    frames = [('object_05_r00_t000_b000_reference', 'frame_227.jpg', 'Initial reference: completed at action 228'),
              ('object_05_r00_identity_geometry', 'frame_149.jpg', 'AEGIS prefix: action 150, before fork 152'),
              ('object_05_r00_t152_b152_reference', 'frame_451.jpg', 'Fork 152 + 300 actions: task incomplete')]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5))
    for ax, (name, image_name, title) in zip(axes, frames):
        file = target / 'runs' / name / image_name
        inputs[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        ax.imshow(plt.imread(file)); ax.axis('off'); ax.set_title(title, fontsize=9)
    fig.suptitle('Object5 actual execution frames (same repeat); illustrative post hoc case', fontsize=12)
    fig.tight_layout(); fig.savefig(output / 'object05_execution_case.png', dpi=180); plt.close(fig)
    text = ['# Object5 固定参考的具体失败机制', '',
            '这是完整实验结束后的轨迹复核，没有增加仿真、策略或控制权限。全部25条同300步参考续接的运动摘要和阶段计数均导出；图中repeat0是事后选定的案例示意，不替代五次重复的正式计数。', '',
            '![实际执行帧](object05_execution_case.png)', '',
            '三个画面分别对应从初始状态的参考完成、AEGIS前缀第150动作（尚未到第152检查点）、以及第152检查点续接300动作后的未完成状态。第一个完成帧仍可能保持抓持，采用原官方任务谓词。', '',
            '![末端与目标的记录运动](object05_reference_motion.png)', '',
            '曲线只展示真正执行的续接部分，AEGIS前缀不计入参考阶段。位置来自每个动作前的原日志观察；任务完成取动作后的原官方判定。阶段名称取自原执行记录，不能将“lift/transit/place”标签当作已经抓住或搬动了目标的证据。',
            '若末端走完运输流程而目标没有跟随，固定参考抓取／保持能力不足是应进一步核查的解释。这里没有通过新专家或完整可达性证明排除真实不可行，也没有训练无解分类器。', '',
            '全部计数须结合最终接触条件下、同一幸存重复及同300动作预算的结果解释。末状态低高度或画面侧倒本身不证明无解。']
    (output / 'CASE_REVIEW.md').write_text('\n'.join(text) + '\n')
    outputs = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}
    receipt = {'status': 'complete', 'slurm_job': os.environ['SLURM_JOB_ID'],
               'source_commit': os.environ['CB_REVIEW_COMMIT'], 'target': str(target),
               'common_trajectory_audit_sha256': hashlib.sha256((interpreted / 'COMMON_TRAJECTORY_AUDIT.json').read_bytes()).hexdigest(),
               'source_inputs_sha256': inputs, 'outputs_sha256': outputs,
               'scope': 'recorded-motion review only; no new scientific execution'}
    (output / 'CASE_REVIEW_COMPLETE.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    review(*sys.argv[1:])
