"""V5 step 2: replay V4 source tapes to selected checkpoints and render two plain overviews.

Protocol: docs/generative_verifier/V5_STEP2_PROTOCOL.md. Student images carry no
annotation. Diagnostic images draw each selected rule's line on the table and are
never student input. Every replayed action must reproduce the source physics hash.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from .inventory import read, sha, write
from . import v5_geometry as g

SIZE = 448
LINE_COLORS = ('1 .1 .1 1', '.1 .4 1 1')
LINE_HALF_LENGTH = .6


def add_lines(xml):
    tree = ET.fromstring(xml); world = tree.find('worldbody')
    for index, color in enumerate(LINE_COLORS):
        ET.SubElement(world, 'geom', dict(name='v5_line_%d' % index, type='box', pos='99 99 99',
                      size='%g .003 .0015' % LINE_HALF_LENGTH, rgba=color, contype='0', conaffinity='0', group='1'))
    return ET.tostring(tree, encoding='unicode')


def poses(regions, seed, fovy):
    """Two opposing 55-degree overviews at a common distance, seeded azimuth, all regions in frame."""
    import numpy as np
    from .constraint_visuals import overview_pose
    value = hashlib.sha256((seed + '|v5-overviews').encode()).digest()
    azimuth = 2 * np.pi * int.from_bytes(value[:8], 'big') / 2**64; elevation = np.radians(55.)
    corners = np.asarray([c for bounds in regions for c in itertools.product(*zip(*bounds))], dtype=float)
    center = corners.mean(0); rotations = []
    for angle in (azimuth, azimuth + np.pi):
        right = np.asarray([-np.sin(angle), np.cos(angle), 0.])
        backward = np.asarray([np.cos(angle) * np.cos(elevation), np.sin(angle) * np.cos(elevation), np.sin(elevation)])
        rotations.append(np.column_stack((right, np.cross(backward, right), backward)))
    fitted = [overview_pose(R, regions, fovy, center=center) for R in rotations]
    distance = max(float(np.linalg.norm(np.asarray(p['position']) - center)) for p in fitted)
    out = []
    for R in rotations:
        local = (corners - center) @ R
        normalized = local[:, :2] / (distance - local[:, 2, None]) / np.tan(np.radians(fovy) / 2)
        if np.max(np.abs(normalized)) >= .8:
            raise RuntimeError('Overview clipping margin failed')
        out.append(dict(position=(center + distance * R[:, 2]).tolist(), rotation=R.reshape(-1).tolist(),
                        fovy=float(fovy), focus=center.tolist(), distance=distance, azimuth=float(np.degrees(azimuth)),
                        maximum_projected_abs=float(np.max(np.abs(normalized)))))
    return out


def body_geoms(m, root):
    out = []
    for geom in range(m.ngeom):
        body = int(m.geom_bodyid[geom])
        while body and body != root:
            body = int(m.body_parentid[body])
        if body == root:
            out.append(geom)
    return out


def render(env, camera):
    import numpy as np
    from openpi_client import image_tools
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
    import robosuite.macros as macros
    rgb = np.asarray(env.sim.render(width=1024, height=1024, camera_name=camera))
    rgb = rgb[::IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]][::-1, ::-1]
    segmentation = np.asarray(env.sim.render(width=1024, height=1024, camera_name=camera, segmentation=True))
    segmentation = segmentation[::IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]][::-1, ::-1]
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(rgb), SIZE, SIZE)), segmentation


def scene(env, unit, geometry, object_points):
    """Current geometry of the target, goal and every candidate-rule anchor; labels recomputed here."""
    import numpy as np
    m, d = env.sim.model, env.sim.data
    goal = geometry['goal_definition']['goal']; names = sorted(env.env.obj_body_id)
    def box(name):
        points, _ = object_points(env, name)
        return points.min(0).tolist(), points.max(0).tolist()
    by_category = {}
    for key in unit['candidate_rules']:
        for cat in key.split(':', 1)[1].split('|'):
            match = [n for n in names if g.category(n) == cat]
            if len(match) != 1:
                raise RuntimeError('Anchor category is not unique: ' + cat)
            by_category[cat] = match[0]
    anchors = {cat: box(name) for cat, name in by_category.items()}
    receiver = goal[2]
    if receiver in env.env.obj_body_id:
        goal_box, goal_body = box(receiver), receiver
    else:
        position = d.site_xpos[m.site_name2id(receiver)].tolist()
        goal_box = ([position[0] - .03, position[1] - .03, position[2] - .03], [position[0] + .03, position[1] + .03, position[2] + .03])
        goal_body = max((n for n in names if receiver.startswith(n + '_')), key=len)
    material = np.asarray(d.geom_xpos[m.geom_name2id(geometry['material']['geom_name'])]).tolist()
    labels = []
    for rule in unit['rules']:
        a, b = rule['key'].split(':', 1)[1].split('|')
        center = lambda bounds: [(lo + hi) / 2 for lo, hi in zip(*bounds)]
        spec = dict(template='line', params=[center(anchors[a])[:2], center(anchors[b])[:2]])
        separated, clearance = g.label(spec, [(g.point(material), goal_box)])
        labels.append(dict(key=rule['key'], label_at_render=separated, clearance_at_render=clearance,
                           step1_label=rule['label'], agrees=separated == rule['label'], anchors=[a, b],
                           ends=spec['params']))
    return dict(target=goal[1], target_box=box(goal[1]), material=material, goal_body=goal_body, goal_box=goal_box,
                anchors=by_category, anchor_boxes=anchors, labels=labels)


def capture(env, folder, unit, geometry, physical_hash):
    import numpy as np
    from PIL import Image
    from OpenGL import GL
    import mujoco
    from geometry import object_points
    m, d = env.sim.model, env.sim.data; before = physical_hash(env)
    folder.mkdir(parents=True)
    state = scene(env, unit, geometry, object_points)
    regions = [state['target_box'], state['goal_box']] + list(state['anchor_boxes'].values())
    cid = m.camera_name2id('agentview')
    keep = np.asarray(d.cam_xpos[cid]).copy(), np.asarray(d.cam_xmat[cid]).copy()
    views = poses(regions, unit['source'] + '|' + str(unit['unit']), float(m.cam_fovy[cid]))
    lines = [m.geom_name2id('v5_line_%d' % i) for i in range(len(LINE_COLORS))]
    if any(m.geom_contype[x] or m.geom_conaffinity[x] or int(m.geom_bodyid[x]) != 0 for x in lines):
        raise RuntimeError('Diagnostic line affects the physical scene')
    entities = dict(target=state['target'], goal=state['goal_body'], **state['anchors'])
    geoms = {k: body_geoms(m, int(env.env.obj_body_id[v])) for k, v in entities.items()}
    floor = min(b[0][2] for b in state['anchor_boxes'].values())
    visibility = {}
    for mode in ('student', 'diagnostic'):
        for geom, rule in zip(lines, state['labels']):
            (ax, ay), (bx, by) = rule['ends']
            yaw = np.arctan2(by - ay, bx - ax)
            center = [(ax + bx) / 2, (ay + by) / 2, floor + .004] if mode == 'diagnostic' else [99., 99., 99.]
            m.geom_pos[geom] = center; d.geom_xpos[geom] = center
            d.geom_xmat[geom] = [np.cos(yaw), -np.sin(yaw), 0, np.sin(yaw), np.cos(yaw), 0, 0, 0, 1]
        for name, pose in zip(('overview_a', 'overview_b'), views):
            d.cam_xpos[cid] = pose['position']; d.cam_xmat[cid] = pose['rotation']
            rgb, segmentation = render(env, 'agentview')
            vendor = GL.glGetString(GL.GL_VENDOR).decode()
            if 'NVIDIA' not in vendor.upper():
                raise RuntimeError('Requested NVIDIA rendering is not active')
            prefix = '' if mode == 'student' else 'diagnostic_'
            Image.fromarray(rgb).save(folder/(prefix + name + '.png'))
            if mode == 'student':
                is_geom = segmentation[:, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
                counts = {}
                for entity, ids in geoms.items():
                    mask = is_geom & np.isin(segmentation[:, :, 1], ids)
                    small = np.asarray(Image.fromarray(mask.astype('uint8')).resize((SIZE, SIZE), Image.NEAREST))
                    counts[entity] = int(small.sum())
                visibility[name] = counts
    d.cam_xpos[cid], d.cam_xmat[cid] = keep
    for geom in lines:
        m.geom_pos[geom] = [99., 99., 99.]; d.geom_xpos[geom] = [99., 99., 99.]
    if physical_hash(env) != before:
        raise RuntimeError('Rendering altered physical arrays')
    best = {k: max(v[k] for v in visibility.values()) for k in entities}
    prescreen = best['target'] >= 96 and all(best[k] >= 48 for k in entities if k != 'target')
    write(folder/'CAMERA.json', dict(views=views, regions=regions, rule_independent_framing=True))
    write(folder/'RENDER_GEOMETRY.json', dict(state, visibility=visibility, maximum_pixels=best,
                                              pixel_prescreen=prescreen, physical_sha256=before))
    write(folder/'HASHES.json', {p.name: sha(p) for p in sorted(folder.iterdir()) if p.name != 'HASHES.json'})


def render_case(root, index):
    import numpy as np
    from .paired_scene import make_env
    from .replay import assert_trace
    case = root/('case_%03d' % index); units = read(case/'UNITS.json'); folder = case/'execution'; folder.mkdir()
    geometry = read(Path(units[0]['v4_case'])/'execution/GEOMETRY.json')
    env = None
    try:
        env, physical_hash, update_marker, source, context = make_env(case, folder, render=True, visual_transform=add_lines)
        if context['step'] != 0:
            raise RuntimeError('Replay must start from the canonical initial state')
        tape_path = Path(source['root'])/'steps.jsonl'
        if sha(tape_path) != source['tape_sha256']:
            raise RuntimeError('Original source tape changed')
        tape = [json.loads(x) for x in tape_path.read_text().splitlines()]
        assert_trace(tape, source['actions'])
        wanted = {u['unit']: u for u in units}; last = max(wanted)
        if 0 in wanted:
            capture(env, root/'units'/wanted[0]['review_id'], wanted[0], geometry, physical_hash)
        low, high = map(np.asarray, env.env.action_spec); checked = 0
        for item in tape:
            if item['step'] > last:
                break
            obs, _, done, _ = env.step(np.clip(item['applied'], low, high).tolist())
            if physical_hash(env) != item['physics_after_sha256'] or bool(done) != item['success']:
                raise RuntimeError('Replay differs from the source at step %d' % item['step'])
            checked += 1; update_marker(obs)
            if item['step'] in wanted:
                capture(env, root/'units'/wanted[item['step']]['review_id'], wanted[item['step']], geometry, physical_hash)
        write(case/'COMPLETE.json', dict(passed=True, units=[u['review_id'] for u in units], replayed_actions=checked,
                                         source_physics_equal=True, slurm_job=os.environ.get('SLURM_JOB_ID')))
    finally:
        if env is not None:
            env.close()


def main(root, case_index=None):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Quest compute allocation required')
    if case_index is not None:
        return render_case(root, case_index)
    device = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip()
    write(root/'GPU.json', dict(devices=device, scope='OpenGL rendering only', slurm_job=os.environ['SLURM_JOB_ID']))
    if 'A100' not in device:
        raise RuntimeError('Exact source replay is qualified on A100 nodes only')
    units = read(root/'UNITS.json'); cases = sorted({u['v4_case'] for u in units})
    for index, v4 in enumerate(cases):
        case = root/('case_%03d' % index); case.mkdir()
        row = read(Path(v4)/'INPUT.json'); write(case/'INPUT.json', row)
        checkpoint = Path(row['evidence']['witness_root'])/'checkpoint_000'
        if row['evidence']['checkpoint_step'] != 0 or sha(checkpoint/'context.json') != row['evidence']['checkpoint_context_sha256']:
            raise RuntimeError('Original canonical checkpoint changed')
        shutil.copyfile(checkpoint/'context.json', case/'CANONICAL_CONTEXT.json')
        shutil.copyfile(checkpoint/'physics.npz', case/'CANONICAL_PHYSICS.npz')
        write(case/'UNITS.json', [u for u in units if u['v4_case'] == v4])
        with (case/'worker.log').open('w') as log:
            subprocess.run([sys.executable, '-u', '-m', 'verifier_dataset.v5_render', str(root), '--case-index', str(index)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    rendered = {u['review_id']: read(root/'units'/u['review_id']/'RENDER_GEOMETRY.json') for u in units}
    labels = [l for r in rendered.values() for l in r['labels']]
    write(root/'RENDER_COMPLETE.json', dict(passed=True, units=len(rendered), cases=len(cases),
        labels_agree=sum(l['agrees'] for l in labels), labels=len(labels),
        pixel_prescreen=sum(r['pixel_prescreen'] for r in rendered.values()), api_calls=0, dataset_rows=0,
        code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ['SLURM_JOB_ID']))
    write(root/'HASHES.json', {n: sha(root/n) for n in ('UNITS.json', 'REVIEW_SHEET.json', 'RENDER_COMPLETE.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); p.add_argument('--case-index', type=int)
    a = p.parse_args()
    try:
        main(a.root, a.case_index)
    except BaseException as error:
        write(a.root/('STOP_case_%s.json' % a.case_index if a.case_index is not None else 'STOP_render.json'),
              dict(error=type(error).__name__, reason=str(error)))
        raise
