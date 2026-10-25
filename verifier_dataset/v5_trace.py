"""V5 trace: replay witnessed source tapes on CPU and record checkpoint-time rule geometry.

A rule arrives at a checkpoint. Its line is fixed by the anchors' positions at that
moment; the label compares the carried object's material point with the goal region
at that moment; the source suffix witnesses a feasible row only if the material point
never crosses that fixed line afterwards, at any integration substep. Every replayed
action must reproduce the source physics hash. Protocol: V5_COLLECTION_PROTOCOL.md.

ROOT runs all cases in order; ROOT --shard K runs every fourth case; ROOT --summarize
aggregates; ROOT --case-index I traces one case in a fresh process.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .inventory import read, sha, write
from . import v5_geometry as g

SHARDS = 4


def rows_from_trace(trace):
    """Checkpoint snapshots plus the substep material path give one row per line rule."""
    rows, actions = [], trace['actions']
    for snap in trace['checkpoints']:
        t, objects = snap['step'], snap['objects']
        excluded = {trace['target']} | g.goal_objects(objects, trace['receiver'])
        names = g.anchors(objects, trace['student_task'], excluded)
        start, goal_box = g.point(snap['material']), (snap['goal']['lo'], snap['goal']['hi'])
        suffix = [g.point(p[1:]) for p in trace['path'] if p[0] > t]
        lift = max(0., snap['material'][2] - snap['surface'])
        for rule in g.rules(objects, names):
            if rule['template'] != 'line':
                continue
            separated, clearance = g.label(rule, [(start, goal_box)])
            crossed = g.crosses(rule, start, suffix)
            row = dict(source=trace['source'], unit=t, family=trace['family'],
                       task=trace['task'] + '|' + trace['student_task'], rule=rule['key'], template='line',
                       anchors=rule['anchors'], label=separated, clearance=clearance, lift=lift,
                       time_bin=min(g.TIME_BINS - 1, int(g.TIME_BINS * t / actions)), suffix_crosses=crossed)
            if not separated:
                row['witnessed'] = not crossed
            if snap.get('robot') is not None:
                row.update(g.robot_sides(rule, start, goal_box, snap['robot']))
            rows.append(row)
    return rows


def trace_case(root, index):
    import numpy as np
    from .paired_scene import make_env
    from .replay import assert_trace
    from .v5_scan import goal_region, scene_objects
    case = root/('case_%03d' % index); spec = read(case/'TRACE_INPUT.json'); folder = case/'execution'; folder.mkdir()
    env = original = None
    try:
        env, physical_hash, update_marker, source, context = make_env(case, folder, render=False)
        from geometry import object_points
        if context['step'] != 0:
            raise RuntimeError('Replay must start from the canonical initial state')
        m, d = env.sim.model, env.sim.data; geom = m.geom_name2id(spec['material_geom'])
        tape_path = Path(source['root'])/'steps.jsonl'
        if sha(tape_path) != source['tape_sha256']:
            raise RuntimeError('Original source tape changed')
        tape = [json.loads(x) for x in tape_path.read_text().splitlines()]
        assert_trace(tape, source['actions'])
        wanted = set(range(0, max(0, len(tape) - 10), 5)); path, snaps, current = [], [], [0]
        def snapshot(step):
            before = physical_hash(env)
            objects, surface = scene_objects(env, object_points)
            snaps.append(dict(step=step, objects=objects, surface=surface, material=np.asarray(d.geom_xpos[geom]).tolist(),
                              goal=goal_region(env, objects, spec['goal'][2])))
            if physical_hash(env) != before:
                raise RuntimeError('Snapshot mutated physics')
        original = env.sim.step
        def monitored(*args, **kwargs):
            result = original(*args, **kwargs)
            path.append([current[0]] + np.asarray(d.geom_xpos[geom]).tolist())
            return result
        snapshot(0); env.sim.step = monitored
        low, high = map(np.asarray, env.env.action_spec)
        for item in tape:
            current[0] = item['step']
            obs, _, done, _ = env.step(np.clip(item['applied'], low, high).tolist())
            if physical_hash(env) != item['physics_after_sha256'] or bool(done) != item['success']:
                raise RuntimeError('Replay differs from the source at step %d' % item['step'])
            update_marker(obs)
            if item['step'] in wanted:
                snapshot(item['step'])
        trace = dict(spec, actions=len(tape), checkpoints=snaps, path=path, source_physics_equal=True, final_success=bool(done))
        with gzip.open(case/'TRACE.json.gz', 'wt') as stream:
            json.dump(trace, stream)
        rows = rows_from_trace(trace)
        with gzip.open(case/'ROWS.jsonl.gz', 'wt') as stream:
            stream.writelines(json.dumps(r) + '\n' for r in rows)
        write(case/'COMPLETE.json', dict(passed=True, rows=len(rows), checkpoints=len(snaps), actions=len(tape),
            final_success=bool(done), slurm_job=os.environ.get('SLURM_JOB_ID')))
    finally:
        if env is not None:
            if original is not None:
                delattr(env.sim, 'step')
            env.close()


def run(root, shard=None):
    cases = read(root/'CASES.json')
    for index, spec in enumerate(cases):
        if shard is not None and index % SHARDS != shard:
            continue
        case = root/('case_%03d' % index); case.mkdir()
        row = read(Path(spec['v4_case'])/'INPUT.json'); write(case/'INPUT.json', row); write(case/'TRACE_INPUT.json', spec)
        checkpoint = Path(row['evidence']['witness_root'])/'checkpoint_000'
        if row['evidence']['checkpoint_step'] != 0 or sha(checkpoint/'context.json') != row['evidence']['checkpoint_context_sha256']:
            raise RuntimeError('Original canonical checkpoint changed')
        shutil.copyfile(checkpoint/'context.json', case/'CANONICAL_CONTEXT.json')
        shutil.copyfile(checkpoint/'physics.npz', case/'CANONICAL_PHYSICS.npz')
        with (case/'worker.log').open('w') as log:
            subprocess.run([sys.executable, '-u', '-m', 'verifier_dataset.v5_trace', str(root), '--case-index', str(index)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    write(root/('SHARD_%s_COMPLETE.json' % shard), dict(passed=True, shard=shard, slurm_job=os.environ.get('SLURM_JOB_ID')))


def summarize(root):
    cases = read(root/'CASES.json'); rows = []
    for index in range(len(cases)):
        case = root/('case_%03d' % index)
        if not read(case/'COMPLETE.json')['passed']:
            raise RuntimeError('Case %d incomplete' % index)
        with gzip.open(case/'ROWS.jsonl.gz', 'rt') as stream:
            rows += [json.loads(x) for x in stream]
    with gzip.open(root/'TRACE_ROWS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in rows)
    primary = [r for r in rows if r['clearance'] >= g.required_clearance(g.CLEARANCES[0], r['lift'])]
    summary = dict(trace=g.summarize(rows, timed=True), cases=len(cases), rows=len(rows),
                   sanity=dict(separated_without_crossing=sum(r['label'] == 1 and not r['suffix_crosses'] for r in primary),
                               separated_rows=sum(r['label'] == 1 for r in primary)),
                   selected_twins=len(g.select_units(rows)), strict_twins=sum(u['strict'] for u in g.select_units(rows, n=10**6, strict_n=10**6)))
    write(root/'TRACE_SUMMARY.json', summary)
    write(root/'TRACE_COMPLETE.json', dict(passed=True, rows=len(rows), cases=len(cases), api_calls=0, dataset_rows=0,
        code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID')))
    write(root/'HASHES.json', {n: sha(root/n) for n in ('CASES.json', 'TRACE_ROWS.jsonl.gz', 'TRACE_SUMMARY.json', 'TRACE_COMPLETE.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    p.add_argument('--case-index', type=int); p.add_argument('--shard', type=int); p.add_argument('--summarize', action='store_true')
    a = p.parse_args()
    try:
        if a.case_index is not None:
            trace_case(a.root, a.case_index)
        elif a.summarize:
            summarize(a.root)
        else:
            run(a.root, a.shard)
    except BaseException as error:
        name = 'STOP_case_%s.json' % a.case_index if a.case_index is not None else ('STOP_summarize.json' if a.summarize else 'STOP_shard_%s.json' % a.shard)
        write(a.root/name, dict(error=type(error).__name__, reason=str(error)))
        raise
