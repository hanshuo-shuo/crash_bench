"""V5 detour layouts: move one anchor onto the source route, replay the tape, and witness a detour.

Protocol: docs/generative_verifier/V5_DETOUR_PROTOCOL.md, section 4. The placements are sampled at plan
time (v5_detour.candidates, from the C0 retrace's step-0 objects and route). Each candidate runs in fresh
processes from the canonical initial state:
  replay: place the anchor and replay the whole source tape. Valid as in S1a: the goal is reached, the
    moved anchor touches only the table and drifts at most 1 cm, and neither the robot nor the pudding
    touches the SafeLIBERO obstacle. The physics hash every five steps goes to LAYOUT.json (S1a format,
    so the HIL runner rebuilds states from it).
  witness-SIDE: place, replay to the lift-off step (hashes checked), then a privileged waypoint controller
    rises to the carry height, passes two corners of the grown zone on SIDE, lowers over the basket and
    releases (ordinary 7D OSC and gripper actions in [-1, 1]). It holds when the goal is reached before
    step 300, the pudding centre stays WITNESS_GAP outside the zone from step 0, and the robot and the
    pudding touch neither the moved anchor nor the obstacle.
A candidate is accepted when the replay is valid and the near-side witness holds, else the far-side one.

ROOT --shard K --shards N runs every N-th candidate; ROOT --candidate-index I --part P runs one part;
ROOT --summarize aggregates.
"""
import argparse
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .inventory import read, sha, write
from . import v5_detour as D
from . import v5_geometry as g
from . import v5_hil as H
from . import v5_layouts as L

SIDES = ('near', 'far')
TOLERANCE = .015
LIMITS = dict(rise=45, corner=60, over=60, lower=45)
RELEASE_STEPS = 15
RELEASE_ABOVE = .07
RETREAT = .10
GAIN = .05
TURN = .5


def run_part(root, index, part):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from .paired_scene import make_env
    from .replay import assert_trace
    from .v5_scan import goal_region, scene_objects
    from .v5_render import body_geoms
    case = root/('candidate_%03d' % index); entry = read(case/'CANDIDATE.json'); spec = entry['spec']; anchor = entry['anchor']
    folder = case/part; folder.mkdir(); env = original = None
    try:
        env, physical_hash, update_marker, source, context = make_env(case, folder, render=False)
        from geometry import object_points
        m, d = env.sim.model, env.sim.data; geom = m.geom_name2id(spec['material_geom'])
        objects, surface = scene_objects(env, object_points)
        for name, o in entry['objects'].items():
            if max(abs(a - b) for k in ('lo', 'hi') for a, b in zip(o[k], objects[name][k])) > 1e-9:
                raise RuntimeError('Step-0 objects differ from the retrace the placement was sampled on: ' + name)
        support = {x for x in range(m.ngeom) if int(m.geom_bodyid[x]) == 0 or any(
            k in (m.body_id2name(int(m.geom_bodyid[x])) or '').lower() for k in ('table', 'floor'))}
        owner = {}
        for name, body in env.env.obj_body_id.items():
            for x in body_geoms(m, int(body)):
                owner[x] = name
        robot = {x for x in range(m.ngeom) if (m.body_id2name(int(m.geom_bodyid[x])) or '').startswith(('robot0_', 'gripper0_'))}
        def contacts():
            return [(int(c.geom1), int(c.geom2)) for c in d.contact[:d.ncon] if c.dist <= 0]
        fixed = {spec['target']} | g.goal_objects(objects, spec['goal'][2])
        stacks = L.movable_stacks(objects, contacts(), owner, support, robot, free_joint=lambda n: (n + '_joint0') in m.joint_names
                                  and int(m.jnt_type[m.joint_name2id(n + '_joint0')]) == 0, anchors=[anchor], fixed=fixed)
        if stacks != {anchor: [anchor]}:
            raise RuntimeError('The anchor is not a free single object resting on the table: ' + anchor)
        placement = entry['placement']
        layout = dict(moved={anchor: placement['center']}, shifts={anchor: placement['shift']}, stacks={anchor: [anchor]})
        L.place(env, layout)
        placed_hash = physical_hash(env)
        body = int(env.env.obj_body_id[anchor]); placed = np.asarray(d.body_xpos[body]).copy()
        after, _ = scene_objects(env, object_points)
        zone = D.zones(after[anchor]['lo'], after[anchor]['hi'])
        obstacle = [n for n in objects if 'obstacle' in n and objects[n]['active']]
        protected = {x for x, n in owner.items() if n in obstacle}
        target_geoms = {x for x, n in owner.items() if n == spec['target']}
        violations, path, current, drift = [], [], [0], [0.]
        def check():
            for a, b in contacts():
                for x, y in ((a, b), (b, a)):
                    if owner.get(x) == anchor and y not in support and owner.get(y) != anchor:
                        violations.append(dict(step=current[0], kind='moved_anchor_contact', other=m.geom_id2name(y)))
                    if x in protected and (y in robot or y in target_geoms):
                        violations.append(dict(step=current[0], kind='obstacle_contact', other=m.geom_id2name(y)))
            shift = float(np.linalg.norm(np.asarray(d.body_xpos[body]) - placed))
            drift[0] = max(drift[0], shift)
            if shift > L.MAX_DRIFT:
                violations.append(dict(step=current[0], kind='anchor_drift', metres=shift))
        original = env.sim.step
        def monitored(*args, **kwargs):
            result = original(*args, **kwargs)
            path.append([current[0]] + np.asarray(d.geom_xpos[geom]).tolist()); check()
            return result
        tape_path = Path(source['root'])/'steps.jsonl'
        if sha(tape_path) != source['tape_sha256']:
            raise RuntimeError('Original source tape changed')
        tape = [json.loads(x) for x in tape_path.read_text().splitlines()]
        assert_trace(tape, source['actions'])
        low, high = map(np.asarray, env.env.action_spec)
        check(); env.sim.step = monitored
        if part == 'replay':
            snaps = []
            def snapshot(step):
                objs, surf = scene_objects(env, object_points)
                snaps.append(dict(step=step, objects=objs, surface=surf, material=np.asarray(d.geom_xpos[geom]).tolist(),
                                  goal=goal_region(env, objs, spec['goal'][2]), physics_sha256=physical_hash(env)))
            wanted = set(range(0, max(0, len(tape) - 10), 5)); snapshot(0)
            for item in tape:
                current[0] = item['step']
                obs, _, done, _ = env.step(np.clip(item['applied'], low, high).tolist())
                update_marker(obs)
                if item['step'] in wanted:
                    snapshot(item['step'])
                if len(violations) > 20:
                    break
            success = bool(env.env._eval_predicate(spec['goal']))
            with gzip.open(Path(entry['c0_case'])/'TRACE.json.gz', 'rt') as stream:
                source_path = json.load(stream)['path']
            deviation = max((math.dist(a[1:], b[1:]) for a, b in zip(path, source_path)), default=None)
            lift_step = next(p[0] for p in path if p[3] - surface >= D.LIFTED)
            result = dict(layout, anchor=anchor, candidate=entry['candidate'], placement=placement, placed_physics_sha256=placed_hash,
                          movable=[anchor], valid=success and not violations, success=success, violations=violations[:20],
                          maximum_anchor_drift=drift[0], maximum_path_deviation=deviation, obstacle=obstacle, zone=zone,
                          start_gap=-D.depth(objects[spec['target']]['center'], zone['zone']),
                          route_depth=max(D.depth(p[1:3], zone['zone']) for p in path), lift_step=lift_step,
                          checkpoint_physics={str(s['step']): s['physics_sha256'] for s in snaps})
            write(case/'LAYOUT.json', result)
            with gzip.open(case/'TRACE.json.gz', 'wt') as stream:
                json.dump(dict(spec, source='%s#d%d' % (spec['source'], entry['candidate']), actions=len(tape), checkpoints=snaps, path=path), stream)
            return
        side = part.split('-', 1)[1]
        layout_record = read(case/'LAYOUT.json')
        if placed_hash != layout_record['placed_physics_sha256']:
            raise RuntimeError('Placed layout differs from the replay run')
        lift_step = layout_record['lift_step']; checked = 0
        for item in tape[:lift_step]:
            current[0] = item['step']
            obs, _, done, _ = env.step(np.clip(item['applied'], low, high).tolist())
            update_marker(obs)
            if str(item['step']) in layout_record['checkpoint_physics']:
                if physical_hash(env) != layout_record['checkpoint_physics'][str(item['step'])]:
                    raise RuntimeError('Witness replay differs from the replay run at step %d' % item['step'])
                checked += 1
        now, _ = scene_objects(env, object_points)
        p0 = np.asarray(d.geom_xpos[geom]).copy(); goal = goal_region(env, now, spec['receiver'])
        site = np.asarray(goal['lo']); carry = surface + D.carry_height(now, surface, spec['target'], float(p0[2]) - surface)
        ways = D.corners(zone['zone'], side, p0[:2].tolist(), site[:2].tolist())
        targets = ([('rise', [p0[0], p0[1], carry])] + [('corner', [x, y, carry]) for x, y in ways]
                   + [('over', [site[0], site[1], carry]), ('lower', [site[0], site[1], site[2] + RELEASE_ABOVE])])
        orientation = Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
        phase = phase_steps = released = 0; hold = None; step = lift_step; switches = []; rows = []
        def reached(name, err):
            return abs(err[2]) < TOLERANCE if name == 'rise' else (
                float(np.linalg.norm(err[:2])) < TOLERANCE if name in ('corner', 'over') else float(np.linalg.norm(err)) < TOLERANCE)
        while step < H.HORIZON and not done:
            p = np.asarray(d.geom_xpos[geom]); eef = np.asarray(obs['robot0_eef_pos'])
            while phase < len(targets) and (reached(targets[phase][0], np.asarray(targets[phase][1]) - p) or phase_steps >= LIMITS[targets[phase][0]]):
                phase += 1; phase_steps = 0; switches.append(dict(step=step, phase=targets[phase][0] if phase < len(targets) else 'release'))
            if phase < len(targets):
                move, grip = np.clip((np.asarray(targets[phase][1]) - p) / GAIN, -1, 1), 1.
            else:
                if hold is None:
                    hold = eef.copy()
                aim = hold if released < RELEASE_STEPS else hold + [0., 0., RETREAT]
                move, grip = np.clip((aim - eef) / GAIN, -1, 1), -1.; released += 1
            turn = Rotation.from_matrix(orientation @ Rotation.from_quat(obs['robot0_eef_quat']).as_matrix().T).as_rotvec()
            action = np.r_[move, np.clip(turn / TURN, -1, 1), grip]
            current[0] = step + 1
            obs, _, done, _ = env.step(action.tolist())
            update_marker(obs); step += 1; phase_steps += 1
            rows.append(dict(step=step, material=np.asarray(d.geom_xpos[geom]).tolist(), eef=np.asarray(obs['robot0_eef_pos']).tolist(),
                             phase=targets[phase][0] if phase < len(targets) else 'release', done=bool(done)))
        gap = -max(D.depth(p[1:3], zone['zone']) for p in path)
        holds = bool(done) and gap >= D.WITNESS_GAP and not violations
        reasons = [k for k, bad in (('goal_not_reached', not done), ('entered_zone', gap < D.WITNESS_GAP), ('contact', bool(violations))) if bad]
        write(case/('WITNESS_%s.json' % side), dict(side=side, holds=holds, reasons=reasons, done_step=step if done else None,
              minimum_gap=gap, violations=violations[:20], maximum_anchor_drift=drift[0], corners=ways, carry_height=carry - surface,
              lift_step=lift_step, checked_hashes=checked, switches=switches, slurm_job=os.environ.get('SLURM_JOB_ID')))
        with gzip.open(case/('WITNESS_%s.jsonl.gz' % side), 'wt') as stream:
            stream.writelines(json.dumps(r) + '\n' for r in rows)
    finally:
        if env is not None:
            if original is not None:
                delattr(env.sim, 'step')
            env.close()


def run(root, shard, shards):
    for index, entry in enumerate(read(root/'CANDIDATES.json')):
        if index % shards != shard:
            continue
        case = root/('candidate_%03d' % index); case.mkdir()
        if entry['placement'] is None:
            write(case/'COMPLETE.json', dict(passed=True, accepted=False, valid=None, witness=None, reason='no_placement'))
            continue
        row = read(Path(entry['spec']['v4_case'])/'INPUT.json'); write(case/'INPUT.json', row); write(case/'CANDIDATE.json', entry)
        checkpoint = Path(row['evidence']['witness_root'])/'checkpoint_000'
        if row['evidence']['checkpoint_step'] != 0 or sha(checkpoint/'context.json') != row['evidence']['checkpoint_context_sha256']:
            raise RuntimeError('Original canonical checkpoint changed')
        shutil.copyfile(checkpoint/'context.json', case/'CANONICAL_CONTEXT.json')
        shutil.copyfile(checkpoint/'physics.npz', case/'CANONICAL_PHYSICS.npz')
        def part(name):
            with (case/(name + '.log')).open('w') as log:
                subprocess.run([sys.executable, '-u', '-m', 'verifier_dataset.v5_detour_layouts', str(root), '--candidate-index', str(index),
                                '--part', name], stdout=log, stderr=subprocess.STDOUT, check=True)
        part('replay'); valid = read(case/'LAYOUT.json')['valid']; witness = None
        for side in SIDES if valid else ():
            part('witness-' + side)
            if read(case/('WITNESS_%s.json' % side))['holds']:
                witness = side
                break
        write(case/'COMPLETE.json', dict(passed=True, accepted=bool(valid and witness), valid=valid, witness=witness,
                                         reason=None if valid and witness else ('invalid_replay' if not valid else 'no_witness'),
                                         slurm_job=os.environ.get('SLURM_JOB_ID')))
    write(root/('SHARD_%d_COMPLETE.json' % shard), dict(passed=True, shard=shard, slurm_job=os.environ.get('SLURM_JOB_ID')))


def summarize(root):
    campaign = read(root/'CAMPAIGN.json'); entries = read(root/'CANDIDATES.json'); split = L.frozen_split(); rows = []
    for k in range(campaign['shards']):
        if not read(root/('SHARD_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Shard %d incomplete' % k)
    for index, entry in enumerate(entries):
        case = root/('candidate_%03d' % index); done = read(case/'COMPLETE.json')
        layout = read(case/'LAYOUT.json') if (case/'LAYOUT.json').exists() else None
        witnesses = {s: read(case/('WITNESS_%s.json' % s)) for s in SIDES if (case/('WITNESS_%s.json' % s)).exists()}
        rows.append(dict(index=index, source=entry['source'], split=split[entry['source']], candidate=entry['candidate'], anchor=entry['anchor'],
                         lift_step=layout['lift_step'] if layout else entry['lift_step'], accepted=done['accepted'], valid=done['valid'],
                         witness=done['witness'], reason=done['reason'], placement=entry['placement'],
                         violations=[v['kind'] for v in layout['violations'][:1]] if layout else [],
                         route_depth=layout['route_depth'] if layout else None, start_gap=layout['start_gap'] if layout else None,
                         witnesses={s: dict(holds=w['holds'], reasons=w['reasons'], done_step=w['done_step'], minimum_gap=w['minimum_gap'],
                                            carry_height=w['carry_height']) for s, w in witnesses.items()}))
    summary = D.layout_reading(rows)
    summary.update(candidates=len(rows), placed=sum(r['placement'] is not None for r in rows), valid=sum(bool(r['valid']) for r in rows),
                   accepted=sum(r['accepted'] for r in rows), reasons=dict(Counter(r['reason'] for r in rows if not r['accepted'])),
                   invalid_kinds=dict(Counter(k for r in rows for k in r['violations'])),
                   witness_sides=dict(Counter(r['witness'] for r in rows if r['accepted'])),
                   witness_failures=dict(Counter(x for r in rows for w in r['witnesses'].values() for x in w['reasons'])),
                   by_source={s: dict(split=split[s], placed=sum(r['source'] == s and r['placement'] is not None for r in rows),
                                      accepted=sum(r['source'] == s and r['accepted'] for r in rows)) for s in sorted({r['source'] for r in rows})},
                   witness_done_step=H.distribution(w['done_step'] for r in rows for w in r['witnesses'].values() if w['holds']),
                   witness_minimum_gap=H.distribution(w['minimum_gap'] for r in rows for w in r['witnesses'].values() if w['holds']),
                   route_depth=H.distribution(r['route_depth'] for r in rows if r['accepted']),
                   lift_step=H.distribution(r['lift_step'] for r in rows if r['accepted']))
    write(root/'CANDIDATE_ROWS.json', rows); write(root/'DETOUR_LAYOUT_SUMMARY.json', summary)
    write(root/'DETOUR_LAYOUT_COMPLETE.json', dict(passed=True, mode=campaign['mode'], reading=summary['reading'], api_calls=0,
                                                   code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID')))
    write(root/'HASHES.json', {n: sha(root/n) for n in ('CANDIDATES.json', 'CANDIDATE_ROWS.json', 'DETOUR_LAYOUT_SUMMARY.json',
                                                        'DETOUR_LAYOUT_COMPLETE.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    p.add_argument('--shard', type=int); p.add_argument('--shards', type=int)
    p.add_argument('--candidate-index', type=int); p.add_argument('--part', choices=('replay',) + tuple('witness-' + s for s in SIDES))
    p.add_argument('--summarize', action='store_true')
    a = p.parse_args()
    try:
        if a.candidate_index is not None:
            run_part(a.root, a.candidate_index, a.part)
        elif a.summarize:
            summarize(a.root)
        else:
            run(a.root, a.shard, a.shards)
    except BaseException as error:
        name = ('STOP_candidate_%s_%s.json' % (a.candidate_index, a.part) if a.candidate_index is not None else
                ('STOP_summarize.json' if a.summarize else 'STOP_shard_%s.json' % a.shard))
        write(a.root/name, dict(error=type(error).__name__, reason=str(error)))
        raise
