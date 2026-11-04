"""Run from-start language ablations through the already validated LIBERO/policy adapter."""
import argparse
from collections import defaultdict
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from .inventory import read, sha, write
from . import hil_suite as S
from . import hil_suite_run as R
from . import language_ablation as A


def run_one(root, index, port):
    plan = read(root/'PLAN.json')
    if plan['mode'] == 'formal':
        gate = read(root/'GATE.json')
        if not gate['passed'] or gate['formal_plan_sha256'] != sha(root/'PLAN.json'):
            raise RuntimeError('Formal inference requires a passed integrity gate for this exact plan')
    unit = plan['units'][index]
    folder = root/'run'/unit['id']; folder.mkdir(parents=True, exist_ok=False)
    physical_hash = R.configure(root); started = time.monotonic(); env = client = None
    try:
        env, task, obs = R.make_env(unit['suite'], unit['task'], unit['init'])
        if task.name != unit['name'] or task.language != unit['original_prompt']:
            raise RuntimeError('Task identity differs from pinned LIBERO')
        trace = [R.record(env, 0, obs, None, False, physical_hash, {})]
        for step in range(1, S.WAIT + 1):
            obs, _, done, _ = env.step(S.DUMMY)
            if done:
                raise RuntimeError('Task already succeeds during settling')
            trace.append(R.record(env, step, obs, S.DUMMY, done, physical_hash, {}))
        at = physical_hash(env)
        client = R.Client(port, unit['seed'], '%s_language_%s' % (os.environ['SLURM_JOB_ID'], unit['id']))
        tail, first, outside = R.policy_loop(env, obs, client, unit['prompt'], unit['suite'], S.WAIT, S.WAIT + unit['max_steps'],
                                             lambda x: x['done'], physical_hash, {}, folder)
        trace += tail
        row = dict(unit, success=bool(trace[-1]['done']), steps=trace[-1]['step'], physics_at_start=at,
                   final_state=[float(x) for x in env.sim.get_state().flatten()],
                   first_policy_input_sha256=first['with_prompt'], policy_input_state_sha256=first['state'],
                   rng_reset=client.ack, policy_requests=len(client.requests),
                   replayed_outputs=sum(r['output_replayed'] for r in client.requests),
                   maximum_native_difference=max(r['native_output_linf_difference'] for r in client.requests),
                   actions_outside_spec=outside, elapsed_seconds=time.monotonic() - started, **R.provenance())
        R.save(folder, trace, row)
    finally:
        if client is not None:
            client.close()
        if env is not None:
            env.close()


def run(root, shard, shards, port):
    units = read(root/'PLAN.json')['units']; done = 0
    for index in A.shard_indices(units, shard, shards):
        if list(root.glob('STOP_*.json')):
            raise RuntimeError('Another shard failed; stop further inference')
        (root/'logs').mkdir(exist_ok=True)
        with (root/'logs'/(units[index]['id'] + '.log')).open('w') as log:
            subprocess.run([sys.executable, '-u', '-m', 'verifier_dataset.language_ablation_run', str(root),
                            '--index', str(index), '--port', str(port)], stdout=log, stderr=subprocess.STDOUT, check=True)
        done += 1
        write(root/('PROGRESS_%d.json' % shard), dict(done=done, last=units[index]['id'], **R.provenance()))
    write(root/('SHARD_%d_COMPLETE.json' % shard), dict(passed=True, units=done, **R.provenance()))


def check_twins(root, units, rows):
    import numpy as np
    groups = defaultdict(list); noise = dict(groups_with_difference=0, max_levels=0, max_values=0)
    for u in units:
        groups[u['group']].append(rows[u['id']])
    for group, rs in groups.items():
        if len({r['physics_at_start'] for r in rs}) != 1 or len({r['rng_reset']['key_hex'] for r in rs}) != 1:
            raise RuntimeError('Paired physics or policy RNG differs: ' + group)
        if len({r['first_policy_input_sha256'] for r in rs if r['condition'] != 'C0R'}) != 3:
            raise RuntimeError('Prompt intervention did not change policy input: ' + group)
        with np.load(root/'run'/rs[0]['id']/'first_policy_input.npz') as first:
            seen = False
            for row in rs[1:]:
                with np.load(root/'run'/row['id']/'first_policy_input.npz') as other:
                    if not np.array_equal(first['observation/state'], other['observation/state']):
                        raise RuntimeError('Paired proprioception differs: ' + group)
                    for name in ('observation/image', 'observation/wrist_image'):
                        ok, levels, values = S.images_match(first[name], other[name])
                        if not ok:
                            raise RuntimeError('Paired images differ beyond frozen rendering tolerance: ' + group)
                        seen |= values > 0; noise['max_levels'] = max(noise['max_levels'], levels); noise['max_values'] = max(noise['max_values'], values)
            noise['groups_with_difference'] += seen
    return dict(noise, groups=len(groups))


def report(root, summary):
    """Write the self-contained result beside immutable raw outputs, without editing the live source checkout."""
    campaign = read(root/'CAMPAIGN.json'); receipt = read(root/'RECEIPT.json')
    def pct(x):
        return '未定义' if x is None else '%.1f%%' % (100 * x)
    def difference(x):
        v = x['value']; ci = x['ci']
        return '%+.1f [%+.1f, %+.1f] pp' % (100 * v, 100 * ci[0], 100 * ci[1])
    lines = ['# π0.5 从头语言消融', '', '模式：`%s`。%d 个状态，%d 次执行。' % (summary['mode'], summary['states'], summary['runs']), '',
             '| 套件 | 状态 | 原指令 | 空指令 | 随机字符 | 空 − 原（95% CI） | 随机 − 原（95% CI） |',
             '|---|---:|---:|---:|---:|---|---|']
    for suite, stats in list(summary['suites'].items()) + [('总体（等任务权重）', summary['overall'])]:
        lines.append('| %s | %d | %s | %s | %s | %s | %s |' % (suite, stats['states'],
                     *(pct(stats['success'][c]['value']) for c in A.CONDITIONS),
                     *(difference(stats['contrasts'][c]['difference']) for c in A.CONDITIONS[1:])))
    lines += ['', '三组使用相同场景、官方初始状态和策略种子，从头给对应文字。随机银行为 12 个均匀 A–F / + 符号，',
              '同一初始状态编号的字符串在所有任务共用。原指令失败的状态照样计入；重复原指令只作噪声诊断。', '',
              '区间按每个固定任务内的初始状态配对 bootstrap，10,000 次，seed 1103，任务等权。',
              '区间不覆盖随机文本银行和任务集合变化的不确定性。随机 / 空文字本身也分布外，',
              '因此性能下降不能单独区分缺少任务信息和输入分布变化；本结果不能直接推出中途指令服从能力。', '',
              '## 完整性与运行', '',
              '- 全部计划单元和轨迹核对通过；起始物理 / 本体 / 策略 RNG 配对通过。',
              '- %d 个配对组；图像最大差 %d 级、%d 个值。' % (summary['twins']['groups'], summary['twins']['max_levels'], summary['twins']['max_values']),
              '- 原指令重复：%d / %d 成功判定一致，%d / %d 逐步哈希一致。' % (summary['repeats']['same_success'], summary['repeats']['n'],
                        summary['repeats']['same_trace'], summary['repeats']['n']),
              '- 执行总耗时 %.2f GPU h（不含服务启动 / 汇总）；重放输出 %d，动作越界 %d 步。' %
                        (summary['total_run_hours'], summary['replayed_outputs'], summary['actions_outside_spec']),
              '- 代码 `%s`；上游 `%s`；LIBERO `%s`。' % (campaign['code_commit'], campaign['upstream_commit'], campaign['libero_commit']),
              '- GPU 作业 `%s`；CPU 汇总 `%s`。' % (receipt['job'], receipt.get('summarize_job', 'smoke 内汇总')),
              '- 根目录：`%s`；配置 / 种子 / 全部文字见 `CAMPAIGN.json`、`PLAN.json`。' % root,
              '- 无 API、无 verifier、无训练，不读 SafeLIBERO 或 V4/V5 锁定测试。', '',
              '## 逐任务结果', '', '| 套件 | 任务 | 原句 | C0 | EMPTY | RANDOM |', '|---|---:|---|---:|---:|---:|']
    names = read(root/'NAMES.json')
    for suite, tasks in summary['tasks'].items():
        for task, stats in tasks.items():
            cells = ['%d/%d' % (stats['success'][c]['count'], stats['success'][c]['n']) for c in A.CONDITIONS]
            lines.append('| %s | %s | %s | %s | %s | %s |' % (suite, task, S.language(names[suite][int(task)]), *cells))
    (root/'REPORT.md').write_text('\n'.join(lines) + '\n')


def summarize(root):
    if list(root.glob('STOP_*.json')):
        raise RuntimeError('Run contains a failure marker')
    plan = read(root/'PLAN.json'); units = plan['units']; rows = {}; traces = {}; repeats = []
    for shard in range(plan['shards']):
        if not read(root/('SHARD_%d_COMPLETE.json' % shard))['passed']:
            raise RuntimeError('Shard incomplete')
    expected = {u['id'] for u in units}
    if {p.name for p in (root/'run').iterdir()} != expected:
        raise RuntimeError('Output folders differ from the exact run plan')
    for u in units:
        row, trace = R.load(root/'run'/u['id'])
        if ([x['step'] for x in trace] != list(range(len(trace))) or row['success'] != trace[-1]['done']
                or row['success'] != any(x['done'] for x in trace[S.WAIT + 1:])
                or len(trace) > S.WAIT + u['max_steps'] + 1 or row['steps'] != trace[-1]['step']
                or trace[S.WAIT]['physics'] != row['physics_at_start']
                or (not row['success'] and row['steps'] != S.WAIT + u['max_steps'])):
            raise RuntimeError('Trace or outcome differs from plan: ' + u['id'])
        rows[u['id']], traces[u['id']] = row, trace
    pairs = A.paired_rows(units, rows); twins = check_twins(root, units, rows)
    for u in units:
        if u['condition'] == 'C0R':
            base = u['group'] + '_C0'; a = traces[u['id']]; b = traces[base]
            repeats.append(dict(group=u['group'], same_success=rows[u['id']]['success'] == rows[base]['success'],
                                same_trace=[x['physics'] for x in a] == [x['physics'] for x in b],
                                max_eef_gap=max(math.dist(x['eef'], y['eef']) for x, y in zip(a, b))))
    summary = dict(mode=plan['mode'], runs=len(rows), states=len(pairs), overall=A.statistics(pairs),
                   suites={s: A.statistics([p for p in pairs if p['suite'] == s]) for s in S.SUITES},
                   tasks={s: {str(t): A.statistics([p for p in pairs if p['suite'] == s and p['task'] == t])
                              for t in sorted({p['task'] for p in pairs if p['suite'] == s})} for s in S.SUITES},
                   twins=twins, repeats=dict(n=len(repeats), same_success=sum(p['same_success'] for p in repeats),
                                           same_trace=sum(p['same_trace'] for p in repeats), rows=repeats),
                   total_run_hours=sum(r['elapsed_seconds'] for r in rows.values()) / 3600.,
                   seconds_per_run=S.H.distribution(r['elapsed_seconds'] for r in rows.values()),
                   replayed_outputs=sum(r['replayed_outputs'] for r in rows.values()),
                   actions_outside_spec=sum(r['actions_outside_spec'] for r in rows.values()), **R.provenance())
    with gzip.open(root/'RUNS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(rows[u['id']]) + '\n' for u in units)
    write(root/'PAIRS.json', pairs); write(root/'SUMMARY.json', summary)
    report(root, summary)
    write(root/'COMPLETE.json', dict(passed=True, mode=plan['mode'], runs=len(rows), states=len(pairs), **R.provenance()))
    write(root/'HASHES.json', {n: sha(root/n) for n in ('PLAN.json', 'RUNS.jsonl.gz', 'PAIRS.json', 'SUMMARY.json', 'REPORT.md', 'COMPLETE.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    parser.add_argument('--phase', choices=['ablation']); parser.add_argument('--shard', type=int); parser.add_argument('--shards', type=int)
    parser.add_argument('--index', type=int); parser.add_argument('--port', type=int)
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    try:
        if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CB_POLICY_CONFIG') != 'pi05_libero':
            raise RuntimeError('This experiment requires Slurm and pi05_libero')
        if args.summarize:
            summarize(args.root)
        elif args.smoke:
            run(args.root, 0, 1, args.port); summarize(args.root)
        elif args.index is not None:
            run_one(args.root, args.index, args.port)
        else:
            run(args.root, args.shard, args.shards, args.port)
    except BaseException as error:
        write(args.root/('STOP_%s.json' % ('summary' if args.summarize else 'index_%s' % args.index if args.index is not None else 'shard_%s' % args.shard)),
              dict(error=type(error).__name__, reason=str(error), **R.provenance()))
        raise
