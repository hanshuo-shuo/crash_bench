"""One fixed legal placement alternative motivated by observed forearm contacts."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
from protocol import STATES
from reference import Reference
from geometry import active_obstacle, object_points


class SouthPlacement(Reference):
    def __init__(self, state):
        super().__init__(state, 'rim')

    def step(self, env, obs):
        planning = dict(obs)
        planning[self.state['goal'] + '_pos'] = np.asarray(obs[self.state['goal'] + '_pos']).copy() + [0., -.025, 0.]
        # Only a local waypoint changes. Actual objects, task predicates and physics are untouched.
        return super().step(env, planning)


def main(root, target):
    root, target = Path(root), Path(target)
    upstream = Path(os.environ['CB_UPSTREAM']); benchmark_root = upstream / 'safelibero/libero/libero'
    config = root / 'libero_config'; config.mkdir()
    (config / 'config.yaml').write_text(json.dumps({
        'benchmark_root': str(benchmark_root), 'bddl_files': str(benchmark_root / 'bddl_files'),
        'init_states': str(benchmark_root / 'init_files'), 'assets': str(benchmark_root / 'assets'),
        'datasets': str(upstream / 'safelibero/libero/datasets')}))
    os.environ['LIBERO_CONFIG_PATH'] = str(config)
    sys.path[:0] = [str(upstream / 'main'), str(upstream / 'safelibero')]
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    rows = []
    for state in STATES[:3]:
        sid = state['id']; directory = root / sid; directory.mkdir()
        suite = benchmark.get_benchmark_dict()[state['suite']](safety_level=state['level'])
        task = suite.get_task(state['task'])
        random.seed(7); np.random.seed(7)
        env = ControlEnv(bddl_file_name=Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file,
                         use_camera_obs=False, has_offscreen_renderer=False, camera_depths=False)
        try:
            env.seed(7); env.reset()
            obs = env.set_init_state(suite.get_task_init_states(state['task'])[state['episode']])
            for _ in range(20): obs, _, _, _ = env.step([0.] * 6 + [-1.])
            initial = np.load(target / 'runs' / (sid + '_screen_rim') / 'checkpoint_000.npz')
            for key in ['qpos', 'qvel', 'ctrl']:
                if not np.array_equal(initial[key], getattr(env.sim.data, key)):
                    raise RuntimeError('Alternative initial physics differs: ' + sid + '/' + key)
            obstacle = active_obstacle(env, obs); initial_position = obs[obstacle + '_pos'].copy()
            _, geoms = object_points(env, obstacle); protected = {g['geom_id'] for g in geoms}
            _, geoms = object_points(env, state['target']); target_geoms = {g['geom_id'] for g in geoms}
            model, data = env.sim.model, env.sim.data
            robot_root = model.body_name2id(env.robots[0].robot_model.root_body); robot_geoms = set()
            for geom in range(model.ngeom):
                body = int(model.geom_bodyid[geom])
                while body:
                    if body == robot_root: robot_geoms.add(geom); break
                    body = int(model.body_parentid[body])
            events = []; original_step = env.sim.step; cursor = {'action': 0}

            def read_contacts(*args, **kwargs):
                result = original_step(*args, **kwargs)
                for c in data.contact[:data.ncon]:
                    pair = [int(c.geom1), int(c.geom2)]
                    if c.dist > 0 or not any(g in protected for g in pair): continue
                    other = pair[1] if pair[0] in protected else pair[0]
                    if other not in robot_geoms and other not in target_geoms: continue
                    events.append({'action': cursor['action'], 'category': 'robot' if other in robot_geoms else 'target',
                                   'bodies': [model.body_id2name(int(model.geom_bodyid[g])) for g in pair],
                                   'distance_m': float(c.dist), 'time': float(data.time)})
                return result

            env.sim.step = read_contacts
            planner = SouthPlacement(state); actions = []; trace = []; maximum = 0.; done = False
            lower, upper = map(np.asarray, env.env.action_spec)
            for step in range(300):
                cursor['action'] = step + 1
                action = planner.step(env, obs)
                if not np.isfinite(action).all() or (action < lower).any() or (action > upper).any():
                    raise RuntimeError('Alternative action exceeds native capability')
                obs, _, done, _ = env.step(action.tolist()); actions.append(action.tolist())
                displacement = float(np.abs(obs[obstacle + '_pos'] - initial_position).sum()); maximum = max(maximum, displacement)
                trace.append({'action': step + 1, 'phase': planner.phase, 'displacement_m': displacement,
                              'success': bool(done), 'output': action.tolist()})
                if done: break
            tape = np.asarray(actions); np.save(directory / 'actions.npy', tape)
            row = {'state': sid, 'variant': 'rim_south25', 'placement_waypoint_offset_m': [0., -.025, 0.],
                   'success': bool(done), 'official_safe_success': bool(done) and maximum <= .001,
                   'no_robot_target_contact_safe_success': bool(done) and maximum <= .001 and not events,
                   'max_obstacle_l1_m': maximum, 'actions': len(actions), 'contact_samples': len(events),
                   'contact_body_pairs': sorted(set(tuple(e['bodies']) for e in events)),
                   'initial_physics_exact': True, 'action_tape_sha256': hashlib.sha256((directory / 'actions.npy').read_bytes()).hexdigest(),
                   'environment_seed': 7, 'code_commit': (root / 'SOURCE_COMMIT').read_text().strip(),
                   'slurm_job': os.environ['SLURM_JOB_ID'], 'scope': 'single post-contact diagnostic screen, not confirmatory or probability certification'}
            (directory / 'trace.json').write_text(json.dumps(trace, indent=2) + '\n')
            (directory / 'contacts.json').write_text(json.dumps(events, indent=2) + '\n')
            (directory / 'row.json').write_text(json.dumps(row, indent=2) + '\n'); rows.append(row)
            (root / 'rows.json').write_text(json.dumps(rows, indent=2) + '\n')
        finally: env.close()
    (root / 'ALTERNATIVE_SCREEN_COMPLETE.json').write_text(json.dumps({'states': len(rows), 'api_calls': 0}, indent=2) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
