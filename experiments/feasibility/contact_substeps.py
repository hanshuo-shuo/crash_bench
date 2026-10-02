"""Read-only contact sampling after every original MuJoCo integration step."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
from protocol import STATES
from geometry import active_obstacle, object_points


def audit(root, case_file):
    root = Path(root)
    config = json.loads(Path(case_file).read_text())
    target = Path(config['source_root']).resolve()
    assets = Path('/projects/p33100/siosio/crashbench_safelibero/feasibility').resolve()
    if target.parent != assets or root.resolve().parent != assets:
        raise RuntimeError('Wrong project root')
    upstream = Path(os.environ['CB_UPSTREAM'])
    benchmark_root = upstream / 'safelibero/libero/libero'
    libero_config = root / 'libero_config'
    libero_config.mkdir()
    (libero_config / 'config.yaml').write_text(json.dumps({
        'benchmark_root': str(benchmark_root), 'bddl_files': str(benchmark_root / 'bddl_files'),
        'init_states': str(benchmark_root / 'init_files'), 'assets': str(benchmark_root / 'assets'),
        'datasets': str(upstream / 'safelibero/libero/datasets')}))
    os.environ['LIBERO_CONFIG_PATH'] = str(libero_config)
    sys.path[:0] = [str(upstream / 'main'), str(upstream / 'safelibero')]
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    reset_forward = {'calls': 0, 'scope': 'legacy headless initialization'}
    if config.get('replay_renderer_reset_forward', False):
        from robosuite.environments.base import MujocoEnv
        from reset_forward import replay_renderer_reset_forward
        reset_forward = replay_renderer_reset_forward(MujocoEnv)
    results = []
    for name in config['run_ids']:
        reset_calls_before = reset_forward['calls']
        directory = target / 'runs' / name
        expected = json.loads((directory / 'row.json').read_text())
        state = next(s for s in STATES if s['id'] == expected['state'])
        steps = [json.loads(line) for line in (directory / 'steps.jsonl').read_text().splitlines()]
        suite = benchmark.get_benchmark_dict()[state['suite']](safety_level=state['level'])
        task = suite.get_task(state['task'])
        random.seed(7); np.random.seed(7)
        env = ControlEnv(bddl_file_name=Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file,
                         use_camera_obs=False, has_offscreen_renderer=False, camera_depths=False)
        try:
            env.seed(7); env.reset()
            obs = env.set_init_state(suite.get_task_init_states(state['task'])[state['episode']])
            for _ in range(20): obs, _, _, _ = env.step([0.] * 6 + [-1.])
            obstacle = active_obstacle(env, obs)
            initial = obs[obstacle + '_pos'].copy()
            _, info = object_points(env, obstacle)
            protected = {g['geom_id'] for g in info}
            _, info = object_points(env, state['target'])
            target_geoms = {g['geom_id'] for g in info}
            model, sim_data = env.sim.model, env.sim.data
            robot_root = model.body_name2id(env.robots[0].robot_model.root_body)
            robot_geoms = set()
            for geom in range(model.ngeom):
                body = int(model.geom_bodyid[geom])
                while body:
                    if body == robot_root:
                        robot_geoms.add(geom); break
                    body = int(model.body_parentid[body])
            events = []
            cursor = {'action': None, 'substep': 0, 'total': 0}
            original_step = env.sim.step

            def step_and_read(*args, **kwargs):
                result = original_step(*args, **kwargs)
                cursor['substep'] += 1; cursor['total'] += 1
                for contact in sim_data.contact[:sim_data.ncon]:
                    pair = [int(contact.geom1), int(contact.geom2)]
                    if not any(g in protected for g in pair) or contact.dist > 0:
                        continue
                    other = pair[1] if pair[0] in protected else pair[0]
                    body = int(model.geom_bodyid[other])
                    category = ('robot' if other in robot_geoms else
                                ('target' if other in target_geoms else 'other_dynamic'))
                    if category == 'other_dynamic':
                        ancestor = body; moving = False
                        while ancestor:
                            moving = moving or int(model.body_dofnum[ancestor]) > 0
                            ancestor = int(model.body_parentid[ancestor])
                        if not moving:
                            continue  # Static support bodies may have positive mass but no DOFs.
                    events.append({'action': cursor['action'], 'substep': cursor['substep'],
                                   'simulation_time': float(sim_data.time), 'category': category,
                                   'geom_pair': [model.geom_id2name(g) for g in pair],
                                   'body_pair': [model.body_id2name(int(model.geom_bodyid[g])) for g in pair],
                                   'distance_m': float(contact.dist)})
                return result

            env.sim.step = step_and_read
            maximum = 0.; done = False; verified_checkpoints = []
            for item in steps:
                checkpoint_step = item['step'] - 1
                snapshot = directory / ('checkpoint_%03d.npz' % checkpoint_step)
                if snapshot.exists():
                    expected_physics = np.load(snapshot)
                    for key in ['qpos', 'qvel', 'ctrl']:
                        if not np.array_equal(expected_physics[key], getattr(sim_data, key)):
                            a, b = expected_physics[key], np.asarray(getattr(sim_data, key)).copy()
                            mismatch = {'run_id': name, 'checkpoint': checkpoint_step, 'field': key,
                                        'expected': a.tolist(), 'actual': b.tolist(),
                                        'expected_shape': list(a.shape), 'actual_shape': list(b.shape),
                                        'max_abs_difference': float(np.max(np.abs(a-b))) if a.shape == b.shape else None,
                                        'joint_names': list(model.joint_names),
                                        'joint_qpos_addresses': model.jnt_qposadr.tolist(),
                                        'slurm_job': os.environ['SLURM_JOB_ID']}
                            (root / 'PHYSICS_MISMATCH.json').write_text(json.dumps(mismatch, indent=2) + '\n')
                            raise RuntimeError('Contact replay physics mismatch at %s/%d/%s' %
                                               (name, checkpoint_step, key))
                    verified_checkpoints.append(checkpoint_step)
                cursor.update(action=item['step'], substep=0)
                obs, _, done, _ = env.step(item['output'])
                maximum = max(maximum, float(np.abs(obs[obstacle + '_pos'] - initial).sum()))
            verified = abs(maximum - expected['max_obstacle_l1_m']) < 1e-9 and bool(done) == expected['success']
            counts = {c: sum(e['category'] == c for e in events) for c in ['robot', 'target', 'other_dynamic']}
            result = {'run_id': name, 'state': state['id'], 'source_root': str(target),
                      'source_row_sha256': hashlib.sha256((directory / 'row.json').read_bytes()).hexdigest(),
                      'commands': len(steps), 'integration_steps': cursor['total'],
                      'exact_physics_verified_checkpoints': verified_checkpoints,
                      'source_proxy_max_l1_m': expected['max_obstacle_l1_m'], 'replay_proxy_max_l1_m': maximum,
                      'source_safe_success': expected['safe_success'], 'source_success': expected['success'],
                      'replay_success': bool(done), 'replay_verified': verified, 'contact_sample_counts': counts,
                      'body_pairs': sorted(set(tuple(e['body_pair']) for e in events)),
                      'events': events, 'slurm_job': os.environ['SLURM_JOB_ID'],
                      'renderer_reset_forward_replayed': bool(config.get('replay_renderer_reset_forward', False)),
                      'initialization_forward_calls': reset_forward['calls'] - reset_calls_before,
                      'scope': 'auxiliary integration-step contact audit; official action-end displacement scoring unchanged'}
            results.append(result)
            (root / 'CONTACT_SUBSTEPS.json').write_text(json.dumps(results, indent=2) + '\n')
            if not verified:
                raise RuntimeError('Read-only contact replay differs from source: ' + name)
        finally:
            env.close()
    (root / 'CONTACT_SUBSTEPS_COMPLETE.json').write_text(json.dumps({
        'runs': len(results), 'all_replays_verified': True,
        'source_root': str(target), 'slurm_job': os.environ['SLURM_JOB_ID']}, indent=2) + '\n')


if __name__ == '__main__':
    audit(sys.argv[1], sys.argv[2])
