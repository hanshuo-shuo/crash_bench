"""One baseline timing plus eight instrumented original-loop smoke rollouts."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
import types
import numpy as np
from common import (BASE, atomic_json, config, scene_id, seed_for, schedule, sha,
                    norm, churn, outcome, time_to_crash)
from adapter import adapt
from geometry import Distance, ARRAYS, physical_hash

sys.path.insert(0, str(BASE/'scripts'))
from run_safelibero import link_offline_bert


def clean(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, (str, int, float, bool, type(None))): return value
    return {'unserialized_type': type(value).__name__}


class Failure(RuntimeError):
    pass


class Observer:
    def __init__(self, runner, spec):
        self.runner, self.spec, self.cfg = runner, spec, runner.cfg
        self.directory = runner.root/'runs'/spec['name']; self.directory.mkdir(parents=True)
        self.scene, self.seed = scene_id(spec['scene']), seed_for(spec['scene'], spec['repeat'])
        self.rows, self.steps, self.inferences = [], [], []
        self.collide = None; self.success = False; self.previous = None; self.latest = None
        self.executed = 0; self.ctx = None; self.exited = None
        self.trace = (self.directory/'steps.jsonl').open('w')
        self.infer_file = (self.directory/'inferences.jsonl').open('w')
        atomic_json(self.directory/'manifest.json', dict(status='started', spec=spec, seed=self.seed,
                    code_commit=os.environ['CB_CODE_COMMIT'], upstream_commit=self.cfg['upstream_commit'],
                    slurm_job=os.environ['SLURM_JOB_ID'], configuration=self.cfg))

    def client(self, host, port):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        observer = self
        self.ws = WebsocketClientPolicy(host, port)
        class Client:
            def infer(_, data):
                try:
                    result = observer.ws.infer(data)
                except BaseException as error:
                    raise Failure('Policy service failure') from error
                info = result['uncertainty']
                if info['run_id'] != observer.spec['name'] or info['infer_index'] != len(observer.inferences)+1:
                    raise Failure('Policy stream identity mismatch')
                chunk = np.asarray(result['actions'])
                if chunk.shape != (10, 7): raise Failure('Invalid production action chunk')
                if observer.spec['diagnostics'] and not info.get('native_action_array_equal'):
                    raise Failure('Missing per-infer native action invariance evidence')
                info.update(step=observer.ctx['t']+1, churn=churn(chunk, observer.previous, observer.scales))
                observer.previous = chunk.copy(); observer.latest = info
                observer.inferences.append(info); observer.infer_file.write(json.dumps(info)+'\n'); observer.infer_file.flush()
                diagnostic = {}
                if observer.cfg.get('save_diagnostic_samples'):
                    samples = np.asarray(result.pop('uncertainty_samples'))
                    if samples.shape != (8,10,7): raise Failure('Missing complete diagnostic action samples')
                    diagnostic = dict(diagnostic_actions=samples, sample_seeds=np.asarray(info['sample_seeds'],dtype='<u4'),
                                      normalization_scales=np.asarray(info['normalization_scales']))
                np.savez_compressed(observer.directory/('infer_%03d.npz'%info['infer_index']), actions=chunk,
                    image=data['observation/image'], wrist_image=data['observation/wrist_image'], state=data['observation/state'], prompt=data['prompt'], **diagnostic)
                return result
        return Client()

    def begin(self, client):
        ack = self.ws.infer({'__uncertainty_reset__': True, 'run_id': self.spec['name'],
                            'scene_id': self.scene, 'seed': self.seed, 'enabled': self.spec['diagnostics']})
        if ack['seed'] != self.seed or ack['requests'] != 0: raise Failure('Policy reset mismatch')
        self.scales = np.asarray(ack['scales']).tolist()
        atomic_json(self.directory/'rng_reset.json', clean(ack))

    def settled(self, env, obs, t):
        if t != 20: raise Failure('Incomplete settling')
        self.env = env; self.settled_hash = physical_hash(env)
        self.qpos_hash = hashlib.sha256(np.asarray(env.sim.data.qpos,dtype='<f8').tobytes()).hexdigest()
        if self.spec['scene'].get('expected_qpos_sha256',self.qpos_hash) != self.qpos_hash:
            raise Failure('Settled qpos differs from frozen sixty-state manifest')
        np.savez_compressed(self.directory/'settled_state.npz', **{k: np.asarray(getattr(env.sim.data, k)).copy() for k in ARRAYS})

    def ready(self, env, obs, obstacle, enabled):
        self.obstacle = obstacle; self.initial = np.asarray(obs[obstacle+'_pos']).copy()
        if self.spec['scene'].get('expected_obstacle',obstacle) != obstacle:
            raise Failure('Active obstacle differs from frozen sixty-state manifest')
        self.distance = Distance(env, obstacle, self.cfg['distance_max_m'])
        atomic_json(self.directory/'geometry.json', dict(obstacle=obstacle, robot_geoms=self.distance.robot,
            protected_geoms=self.distance.protected, filter_enabled=bool(enabled), distance_units='m', signed=True,
            live_mutation_check='physics arrays + time + model body poses at every read'))
        atomic_json(self.directory/'geometry_details.json', self.distance.details())

    def before(self, ctx):
        self.ctx = ctx
        self.min_dist, self.pair = self.distance.read()
        if self.spec['method'] == 'nominal' and ctx['t'] and ctx['t'] % 20 == 0:
            self.snapshot(ctx['t'], ctx)
        self.physics_before = physical_hash(self.env)

    def snapshot(self, step, ctx):
        path = self.directory/'snapshots'; path.mkdir(exist_ok=True)
        if (path/('%03d.json'%step)).exists(): return
        env = self.env; robot = env.robots[0]
        physics = {k: np.asarray(getattr(env.sim.data, k)).copy() for k in ARRAYS}
        physics.update(sim_state=env.get_sim_state().copy(), marker_body_pos=env.sim.model.body_pos.copy(),
                       marker_body_quat=env.sim.model.body_quat.copy())
        np.savez_compressed(path/('%03d_physics.npz'%step), **physics)
        obs = ctx.get('obs', {})
        np.savez_compressed(path/('%03d_observation.npz'%step), **{k: np.asarray(v).copy() for k, v in obs.items() if isinstance(v, np.ndarray)})
        controller = {k: clean(v) for k, v in vars(robot.controller).items()
                      if isinstance(v, (np.ndarray, np.generic, str, float, int, bool, list, tuple, type(None)))}
        payload = dict(step=step, physics_sha256=sha(path/('%03d_physics.npz'%step)),
            observation_sha256=sha(path/('%03d_observation.npz'%step)), controller=controller,
            action_queue=clean(list(ctx['action_plan'])), python_rng=clean(random.getstate()),
            numpy_rng=clean(np.random.get_state()), policy_rng=clean(self.ws.infer({'__uncertainty_snapshot__': True})),
            diagnostic_infer_count=len(self.inferences), previous_chunk=clean(self.previous),
            gripper_action=clean(robot.gripper.current_action),
            environment=dict(timestep=env.env.timestep, cur_time=env.env.cur_time, done=env.env.done),
            aegis={k: clean(ctx.get(k)) for k in ['p1','R1','Q1_diag','p2','R2','Q2_diag','z_fixed','flag_safety_control']},
            collision_step=self.collide, restore_verified=False,
            scope='save only; arbitrary mid-rollout physical/controller/observable/queue/RNG restoration has not been tested')
        atomic_json(path/('%03d.json'%step), payload)

    def candidate(self, raw, applied, t, status):
        if not np.isfinite(raw).all() or not np.isfinite(applied).all(): raise Failure('Nonfinite execution command')
        queue_equal = bool(np.array_equal(np.asarray(raw), self.previous[t % self.cfg['replan_steps']]))
        physical_equal = self.physics_before == physical_hash(self.env)
        if not queue_equal or not physical_equal:
            atomic_json(self.directory/'ACTION_INVARIANCE_FAILURE.json', dict(step=t+1,
                        queue_array_equal=queue_equal, physical_before_action_unchanged=physical_equal))
            raise Failure('Actual queue command or live physics changed before execution')
        applied = np.asarray(applied); controller = self.env.robots[0].controller
        clipped = applied.copy(); clipped[:6] = np.clip(applied[:6], controller.input_min, controller.input_max)
        self.pending = dict(step=t+1, raw=np.asarray(raw).tolist(), proposed=applied.tolist(),
                            applied=applied.tolist(), controller_clipped=clipped.tolist(), qp_status=status,
                            act_norm6=norm(applied, self.scales, 6), act_norm7=norm(applied, self.scales),
                            controller_clipped_norm7=norm(clipped, self.scales), min_dist=self.min_dist,
                            distance_witness=self.distance.witness, distance_native_max_abs=self.distance.native_max_abs,
                            closest_geom_pair=self.pair, infer_index=self.latest['infer_index'],
                            queue_array_equal=queue_equal, physical_before_action_unchanged=physical_equal,
                            native_seconds=self.latest['native_seconds'], diagnostic_seconds=self.latest['diagnostic_seconds'])

    def after(self, obs, done, t):
        displacement = float(np.abs(np.asarray(obs[self.obstacle+'_pos'])-self.initial).sum())
        if displacement > .001 and self.collide is None: self.collide = t+1
        self.executed = t+1; self.success = bool(done)
        self.pending.update(obstacle_l1_m=displacement, success=bool(done), physics_after_sha256=physical_hash(self.env))
        self.steps.append(dict(self.pending)); self.trace.write(json.dumps(self.pending)+'\n'); self.trace.flush()
        self.rows.append(dict(scene_id=self.scene, task_id='%s/%s/task%d'%(self.spec['scene']['suite'], self.spec['scene']['level'], self.spec['scene']['task']),
            capability=self.spec['method'], seed=self.seed, step=t+1, disagreement=self.latest['disagreement'],
            churn=self.latest['churn'], act_norm=self.pending['act_norm7'], min_dist=self.min_dist,
            crashed=self.collide is not None, act_norm6=self.pending['act_norm6'], infer_index=self.latest['infer_index'],
            infer_boundary=self.latest['step']==t+1))

    def caught(self, error, t):
        if isinstance(error, NameError) and str(error) == "name 'a' is not defined":
            self.exited = 'upstream_qp_infeasible_undefined_a'
            return
        raise Failure('Evaluation exception at action %d: %s'%(t+1, type(error).__name__)) from error

    def finish(self, done, collided, video, ctx):
        if done != self.success or collided != (self.collide is not None): raise Failure('Scoring disagreement')
        if self.spec['method'] == 'nominal' and self.executed and self.executed % 20 == 0:
            self.snapshot(self.executed, ctx)
        self.video = video

    def load_detector(self, loader, cfg, checkpoint):
        if self.runner.detector is None: self.runner.detector = loader(cfg, checkpoint)
        return self.runner.detector

    def perception(self, image, instruction, suite):
        from openrouter_perception import export_request, read_response
        import matplotlib.pyplot as plt
        buf = io.BytesIO(); plt.imsave(buf, image, format='png')
        # Fresh run-scoped request directory preserves the original paired AEGIS VLM semantics.
        request = export_request(buf.getvalue(), instruction, suite, self.runner.root/'requests'/self.spec['name'], budgeted=True)
        deadline = time.monotonic()+300
        while not (request/'response.json').exists():
            if (self.runner.root/'STOP.json').exists(): raise Failure('API worker stopped')
            if time.monotonic() > deadline: raise Failure('Original pre-action VLM boundary timed out')
            time.sleep(2)
        return read_response(request)


class Runner:
    def __init__(self, root, port, cfg=None):
        self.root, self.port, self.cfg = root, port, cfg or config(); self.detector = None
        self.upstream = Path(os.environ['CB_UPSTREAM']); self.assets = Path(os.environ['CB_ASSETS'])
        path = root/'libero_config'; path.mkdir()
        benchmark = self.upstream/'safelibero/libero/libero'
        atomic_json(path/'config.yaml', dict(benchmark_root=str(benchmark), bddl_files=str(benchmark/'bddl_files'),
            init_states=str(benchmark/'init_files'), assets=str(benchmark/'assets'), datasets=str(self.upstream/'safelibero/libero/datasets')))
        os.environ['LIBERO_CONFIG_PATH'] = str(path)
        sys.path[:0] = [str(self.upstream/'main'), str(self.upstream/'safelibero'), str(self.upstream/'openpi/packages/openpi-client/src')]
        self.source = (self.upstream/'main/main_aegis.py').read_text()

    def run(self, spec):
        observer = Observer(self, spec); previous = Path.cwd(); os.chdir(observer.directory)
        started = time.monotonic(); status = 'failed'; envs = []; module_name = None
        try:
            if spec['method'] == 'aegis':
                (observer.directory/'GroundingDINO').symlink_to(self.assets/'GroundingDINO', target_is_directory=True)
                link_offline_bert(observer.directory, self.assets)
            import utils
            utils.obstacle_detection = observer.perception
            patched = adapt(self.source, spec['method'])
            (observer.directory/'adapted_main_aegis.py').write_text(patched)
            module = types.ModuleType('uncertainty_upstream_'+spec['name']); module.__file__ = str(self.upstream/'main/main_aegis.py')
            module_name = module.__name__
            sys.modules[module.__name__] = module; ns = vars(module); ns['observer'] = observer
            exec(compile(patched, module.__file__, 'exec'), ns)
            original = ns['_get_libero_env']
            def get_env(*args):
                random.seed(7); np.random.seed(7)
                env, description = original(*args); envs.append(env); return env, description
            ns['_get_libero_env'] = get_env
            s = spec['scene']
            ns['eval_libero'](ns['Args'](host='127.0.0.1', port=self.port, task_suite_name=s['suite'], safety_level=s['level'],
                task_index=[s['task']], episode_index=[s['episode']], video_out_path=str(observer.directory/'videos'),
                seed=7, replan_steps=5, num_steps_wait=20))
            if not observer.executed: raise Failure('No action execution evidence')
            for row in observer.rows:
                row.update(time_to_crash=time_to_crash(row['step'], observer.collide), outcome=outcome(observer.success, observer.collide))
            with (observer.directory/'rows.jsonl').open('w') as f:
                for row in observer.rows: f.write(json.dumps(row)+'\n')
            if spec['name'] == 'smoke_s0_r0_nominal':
                baseline = [json.loads(x) for x in (self.root/'runs/timing/steps.jsonl').read_text().splitlines()]
                equal = len(baseline)==len(observer.steps) and all(a[k]==b[k] for a,b in zip(baseline,observer.steps)
                    for k in ['raw','proposed','applied','controller_clipped','physics_after_sha256','success','obstacle_l1_m'])
                proof = dict(passed=equal, baseline='timing', diagnostic=spec['name'], baseline_actions=len(baseline),
                             diagnostic_actions=len(observer.steps), acceptance_gate=False,
                             equality='exact command values and live physical bytes at every action',
                             first_mismatch_step=next((b['step'] for a,b in zip(baseline,observer.steps)
                                if any(a[k]!=b[k] for k in ['raw','applied','physics_after_sha256'])),None),
                             warning=None if equal else 'Fresh environments may differ in RGB rendering; per-infer same-input/RNG invariance is the approved acceptance gate.')
                atomic_json(self.root/'ROLLOUT_EQUALITY.json', proof)
            conditional = dict(passed=all(x.get('native_action_array_equal',False) for x in observer.inferences)
                               and all(x['queue_array_equal'] and x['physical_before_action_unchanged'] for x in observer.steps),
                               inferences=len(observer.inferences), actions=observer.executed,
                               acceptance='strict same-input/RNG native before/after each infer; actual queue commands exact; live physics unchanged before every action')
            if spec['diagnostics']:
                atomic_json(observer.directory/'CONDITIONAL_ACTION_EQUALITY.json', conditional)
                if not conditional['passed']: raise Failure('Conditional action invariance failed')
            status = 'complete'
            result = dict(spec=spec, seed=observer.seed, status=status, actions=observer.executed,
                outcome=outcome(observer.success, observer.collide), collision_step=observer.collide, success=observer.success,
                elapsed_seconds=time.monotonic()-started, inferences=len(observer.inferences), exited=observer.exited,
                native_seconds=sum(x['native_seconds'] for x in observer.inferences),
                diagnostic_seconds=sum(x['diagnostic_seconds'] for x in observer.inferences),
                reference_check_seconds=sum(x['reference_check_seconds'] for x in observer.inferences),
                snapshots=len(list((observer.directory/'snapshots').glob('*.json'))) if (observer.directory/'snapshots').exists() else 0,
                video=getattr(observer, 'video', None), settled_physics_sha256=observer.settled_hash,
                settled_qpos_sha256=observer.qpos_hash)
            atomic_json(observer.directory/'RESULT.json', result)
            return result
        finally:
            observer.trace.close(); observer.infer_file.close()
            for env in envs: env.close()
            if hasattr(observer, 'ws'): observer.ws._ws.close()
            if module_name: sys.modules.pop(module_name,None)
            observer.ctx = None
            if 'utils' in sys.modules:
                sys.modules['utils'].obstacle_detection = None
            import gc
            gc.collect()
            os.chdir(previous)
            atomic_json(observer.directory/'manifest.json', dict(status=status, spec=spec, seed=observer.seed,
                elapsed_seconds=time.monotonic()-started, executed_actions=observer.executed,
                code_commit=os.environ['CB_CODE_COMMIT'], upstream_commit=self.cfg['upstream_commit'],
                slurm_job=os.environ['SLURM_JOB_ID'], configuration=self.cfg))


def main(root, port):
    runner = Runner(root, port); results = []
    try:
        for spec in schedule(runner.cfg):
            if (root/'STOP.json').exists(): raise Failure('External worker stop')
            result = runner.run(spec); results.append(result)
            atomic_json(root/'progress.json', dict(completed=len(results), expected=9, results=results))
            if spec['name']=='timing': atomic_json(root/'TIMING.json', result)
        atomic_json(root/'COMPUTE_COMPLETE.json', dict(completed=9, results=results, slurm_job=os.environ['SLURM_JOB_ID']))
        proofs=[json.loads((root/'runs'/x['spec']['name']/'CONDITIONAL_ACTION_EQUALITY.json').read_text())
                for x in results if x['spec']['diagnostics']]
        atomic_json(root/'CONDITIONAL_ACTION_EQUALITY.json', dict(passed=len(proofs)==8 and all(x['passed'] for x in proofs),
                    runs=8, inferences=sum(x['inferences'] for x in proofs), actions=sum(x['actions'] for x in proofs),
                    acceptance=proofs[0]['acceptance']))
    except BaseException as error:
        atomic_json(root/'STOP.json', dict(stage='compute', error=type(error).__name__, reason=str(error),
            slurm_job=os.environ.get('SLURM_JOB_ID'), completed=len(results)))
        raise


if __name__=='__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); p.add_argument('--port', type=int, required=True)
    a = p.parse_args(); main(a.root, a.port)
