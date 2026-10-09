"""Allocated CPU-only provenance and static geometry audit; no policy/API calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from common import atomic_json


def stream_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprints(root):
    assets = Path(os.environ['CB_ASSETS'])
    upstream = Path(os.environ['CB_LIVE_UPSTREAM'])
    commit = '2457feed5968ae803926e178c8ce8243b9ecdcf9'
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=upstream)
    if git('rev-parse', 'HEAD').decode().strip() != commit or git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('Pinned upstream tracked tree is not clean')
    frozen = root/'upstream'; frozen.mkdir()
    archive = subprocess.Popen(['git', 'archive', commit], cwd=upstream, stdout=subprocess.PIPE)
    subprocess.run(['tar', '-x', '-C', str(frozen)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    if archive.wait(): raise RuntimeError('Upstream archive failed')
    entries = []
    for name in git('ls-tree', '-r', '--name-only', commit).decode().splitlines():
        expected = stream_sha(frozen/name)
        if stream_sha(upstream/name) != expected:
            raise RuntimeError('Upstream working bytes differ from Git archive: '+name)
        entries.append(dict(path=name, bytes=(frozen/name).stat().st_size, sha256=expected))
    manifest = json.loads((assets/'VERIFIED.json').read_text())
    verified = []
    for item in manifest['files']:
        p = Path(item['path'])
        if p.stat().st_size != item['bytes'] or stream_sha(p) != item['sha256']:
            raise RuntimeError('Staged asset differs: '+str(p))
        verified.append(item)
    container = assets/'containers/aegis-py38.sif'
    container_hash = stream_sha(container)
    checksum = (assets/'containers/aegis-py38.sif.sha256').read_text().split()[0]
    if container_hash != checksum: raise RuntimeError('Container hash differs')
    atomic_json(root/'FINGERPRINTS.json', dict(passed=True, upstream_commit=commit,
        upstream=str(frozen), upstream_files=entries, assets=verified,
        verified_manifest_sha256=stream_sha(assets/'VERIFIED.json'),
        container=dict(path=str(container),sha256=container_hash,bytes=container.stat().st_size),
        code_commit=(root/'SOURCE_COMMIT').read_text().strip(), slurm_job=os.environ['SLURM_JOB_ID']))


def geometry(root):
    import numpy as np
    import mujoco
    upstream = root/'upstream'
    cfg = root/'libero_config'; cfg.mkdir()
    benchmark = upstream/'safelibero/libero/libero'
    atomic_json(cfg/'config.yaml', dict(benchmark_root=str(benchmark),
        bddl_files=str(benchmark/'bddl_files'), init_states=str(benchmark/'init_files'),
        assets=str(benchmark/'assets'), datasets=str(upstream/'safelibero/libero/datasets')))
    os.environ['LIBERO_CONFIG_PATH'] = str(cfg)
    sys.path.insert(0, str(upstream/'safelibero'))
    from libero.libero import benchmark as suites
    from libero.libero.envs.env_wrapper import ControlEnv
    from geometry import ARRAYS, Distance, box_distance
    suite = suites.get_benchmark_dict()['safelibero_object'](safety_level='II')
    task = suite.get_task(1)
    env = ControlEnv(bddl_file_name=str(benchmark/'bddl_files'/task.problem_folder/task.bddl_file),
                     use_camera_obs=False, has_offscreen_renderer=False, camera_depths=False)
    try:
        env.seed(7); env.reset()
        snapshot = Path(os.environ['CB_SMOKE_ROOT'])/'runs/smoke_s1_r0_nominal/snapshots/100_physics.npz'
        saved = np.load(snapshot)
        m = env.sim.model._model; data = mujoco.MjData(m)
        for k in ARRAYS: getattr(data,k)[:] = saved[k]
        data.time = float(saved['sim_state'][0])
        m.body_pos[:] = saved['marker_body_pos']; m.body_quat[:] = saved['marker_body_quat']
        mujoco.mj_forward(m,data)
        steps = [json.loads(x) for x in (snapshot.parents[1]/'steps.jsonl').read_text().splitlines()]
        expected = next(x for x in steps if x['step']==101)
        distance = Distance(env, 'red_coffee_mug_obstacle_1', 10.)
        # Static snapshot inspection only: no claim of controller/RNG continuation restore.
        pairs = sorted({tuple(x['closest_geom_pair']) for x in steps if x['min_dist'] < 0})
        records = []
        for a,b in pairs:
            def geom(g):
                return dict(id=g,name=env.sim.model.geom_id2name(g),type=int(m.geom_type[g]),
                    size=m.geom_size[g].tolist(), position=data.geom_xpos[g].tolist(),
                    matrix=data.geom_xmat[g].reshape(3,3).tolist(),margin=float(m.geom_margin[g]),
                    contype=int(m.geom_contype[g]),conaffinity=int(m.geom_conaffinity[g]))
            values=[]
            for cutoff in [0., .001, .01, .05, .1, .5, 1., 10.]:
                witness=np.zeros(6)
                value=float(mujoco.mj_geomDistance(m,data,a,b,cutoff,witness))
                values.append(dict(distmax=cutoff,distance=value,points=witness.tolist()))
            # Independent OBB separating-axis bound for box-box pairs; positive proves separation.
            sat=None
            exact=None;exact_points=None
            if int(m.geom_type[a])==6 and int(m.geom_type[b])==6:
                ra=data.geom_xmat[a].reshape(3,3); rb=data.geom_xmat[b].reshape(3,3)
                delta=data.geom_xpos[b]-data.geom_xpos[a]
                axes=[ra[:,i] for i in range(3)]+[rb[:,i] for i in range(3)]
                axes += [np.cross(ra[:,i],rb[:,j]) for i in range(3) for j in range(3)]
                gaps=[]
                for axis in axes:
                    n=np.linalg.norm(axis)
                    if n<1e-12: continue
                    axis=axis/n
                    gaps.append(float(abs(np.dot(delta,axis))-np.dot(m.geom_size[a],abs(ra.T@axis))-np.dot(m.geom_size[b],abs(rb.T@axis))))
                sat=max(gaps)
                exact,exact_points=box_distance(data.geom_xpos[a],data.geom_xmat[a],m.geom_size[a],
                                               data.geom_xpos[b],data.geom_xmat[b],m.geom_size[b])
                if exact<sat-1e-12:raise RuntimeError('Exact separation violates independent SAT bound')
                from scipy.optimize import lsq_linear
                A=np.concatenate((ra,-rb),axis=1)
                solved=lsq_linear(A,delta,bounds=(-np.r_[m.geom_size[a],m.geom_size[b]],np.r_[m.geom_size[a],m.geom_size[b]]),tol=1e-13,max_iter=1000)
                independent=float(np.linalg.norm(A@solved.x-delta))
                if not solved.success or abs(independent-max(0.,exact))>1e-8:
                    raise RuntimeError('Exact OBB separation differs from independent bounded least squares')
            contacts=[dict(geom1=int(c.geom1),geom2=int(c.geom2),distance=float(c.dist))
                      for c in data.contact[:data.ncon] if {int(c.geom1),int(c.geom2)}=={a,b}]
            records.append(dict(robot=geom(a),protected=geom(b),queries=values,
                                obb_max_separating_gap_m=sat,exact_obb_distance_m=exact,
                                exact_obb_points=None if exact_points is None else exact_points.tolist(),
                                independent_bounded_least_squares_distance_m=independent if exact is not None else None,
                                engine_contacts=contacts))
        matched=next(r for r in records if [r['robot']['id'],r['protected']['id']]==expected['closest_geom_pair'])
        reproduced=matched['queries'][-1]['distance']
        if abs(reproduced-expected['min_dist'])>1e-6:
            raise RuntimeError('Static snapshot geometry did not reproduce pre-action101 distance')
        atomic_json(root/'GEOMETRY_AUDIT.json',dict(mujoco=mujoco.__version__, snapshot=str(snapshot),
            snapshot_sha256=stream_sha(snapshot), action_predictor_step=101,
            expected_distance=expected['min_dist'],reproduced_distance=reproduced,
            official_obstacle_l1_m=expected['obstacle_l1_m'], official_outcome='safe_success',
            pairs=records, geometry_selection=distance.details(),
            scope='Static physics/model-pose reconstruction only; no policy/API/render/rollout or continuation restoration'))
        if exact is not None:
            eye=np.eye(3);unit=np.ones(3)
            cases=[(np.array([3.,0,0]),1.),(np.array([3.,3.,0]),np.sqrt(2.)),
                   (np.array([3.,3.,3.]),np.sqrt(3.)),(np.array([1.5,0,0]),-.5)]
            for center,expected_value in cases:
                value,_=box_distance(np.zeros(3),eye,unit,center,eye,unit)
                if abs(value-expected_value)>1e-12:raise RuntimeError('Analytic OBB sanity check failed')
            original=matched['queries'][-1]['distance'];fixed=matched['exact_obb_distance_m']
            if not (original<0 and fixed>0 and matched['obb_max_separating_gap_m']>0 and not matched['engine_contacts']):
                raise RuntimeError('Expected separated-box defect not confirmed')
            atomic_json(root/'GEOMETRY_RESOLUTION.json',dict(passed=True,
                old_native_distance_m=original,corrected_obb_distance_m=fixed,
                independently_positive_sat_gap_m=matched['obb_max_separating_gap_m'],
                fix='Exact Euclidean OBB closest features / signed SAT depth only for box-box diagnostic measurement',
                original_official_crash_success_policy_controller_unchanged=True,
                analytic_cases=len(cases),old_smoke_min_dist_not_validated_for_statistical_fitting=True))
    finally: env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['fingerprints','geometry']); p.add_argument('root',type=Path)
    args=p.parse_args()
    (fingerprints if args.mode=='fingerprints' else geometry)(args.root)
