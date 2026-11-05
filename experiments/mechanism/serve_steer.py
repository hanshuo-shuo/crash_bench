"""Policy server for the closed-loop causal test of the mechanism pilot (docs/generative_verifier/HIL_MECHANISM_CLOSED_LOOP.md).

It speaks serve.py's protocol to verifier_dataset/hil_suite_run.py (RNG reset per run, the same per-request diagnostic), with
one addition: every reset carries the run's arm, and every request is answered by patching.Patcher.steer. The arm moves
the action expert's image-token keys/values at the chosen layers further from the old prompt A's (amp), or by a random
vector of the same norm (rand), by lam - 1 times their B - A difference; the native arm (lam 1) is the same program with
nothing moved. Every arm draws its noise exactly as openpi's Policy.infer does after the reset, so twins share the noise
of each request. On request 1 of each run the server also runs openpi's own Policy.infer and the steering program at
lam 1 on the same input and noise (the tool check). Every request is logged to LOG_DIR/<run_id>.jsonl.

Two action-space arms of the repair batch (HIL_MECHANISM_REPAIR.md) reuse the same program and touch no key/value:
  actrand  each request computes amp's chunk and the native chunk (same noise); the output is the native chunk plus, on every
           row and action group (translation, rotation, gripper), a push of the same norm as amp's change of that row and
           group, along one random direction per group fixed for the whole run (from direction_seed).
  actdiff  the native chunks under B and under the old prompt A (same noise); the output extrapolates B away from A,
           a_B + (lam - 1)(a_B - a_A), like classifier-free guidance on the output chunk.

The P4 neuron arms (HIL_MECHANISM_NEURONS.md 4.2, kind nrn) change the VLM's GeGLU hidden units on the image rows instead
(neurons.NeuronPatcher.neuron_steer): on the arm's neurons ('top' = the pair's set S, 'random' = its matched random set R, 'all'
= every hook-layer unit) the running prompt's hidden units become h_B + delta, with the per-layer norm of (lam - 1) times
|h_B - h_A| on the reference set (S, or all for 'all'), along h_B - h_A of the applied units ('own') or a random direction
('random'). lam 2 amplifies; lam 0 sets them back to the old prompt's values. The sets come from --neurons FILE.

Usage: serve_steer.py --port P --checkpoint DIR --log-dir DIR [--neurons FILE] (openpi environment, pinned openpi on PYTHONPATH)
"""
import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'feasibility'))
from serve import digest  # noqa: E402

KINDS = ('native', 'amp', 'rand', 'actrand', 'actdiff', 'nrn')
ACTION_GROUPS = ((0, 3), (3, 6), (6, 7))   # environment action: translation, rotation, gripper
ARM_KEYS = {'name', 'kind', 'lam', 'layers', 'prompt_a', 'direction_seed'}
NEURON_KEYS = {'neuron_set', 'neurons', 'direction'}


def check_arm(arm, depth, neuron_sets=None):
    """The arm a reset message carries; raises unless it is well formed (native exactly when lam == 1; a neuron arm names a
    loaded set and has lam 0 or > 1)."""
    if isinstance(arm, dict) and arm.get('kind') == 'nrn':
        if set(arm) != ARM_KEYS | NEURON_KEYS or neuron_sets is None or arm['neuron_set'] not in neuron_sets['sets'] \
                or arm['neurons'] not in ('top', 'random', 'all') or arm['direction'] not in ('own', 'random') \
                or not isinstance(arm['lam'], float) or not (arm['lam'] == 0 or arm['lam'] > 1):
            raise ValueError('Invalid neuron arm: %r' % (arm,))
        return check_arm({k: v for k, v in arm.items() if k not in NEURON_KEYS} | dict(kind='amp', lam=2.), depth) and arm
    if not isinstance(arm, dict) or set(arm) != ARM_KEYS:
        raise ValueError('Arm must have the keys %s' % sorted(ARM_KEYS))
    layers = arm['layers']
    if (arm['kind'] not in KINDS or arm['kind'] == 'nrn' or not isinstance(arm['lam'], float) or arm['lam'] < 1 or (arm['kind'] == 'native') != (arm['lam'] == 1)
            or not isinstance(layers, list) or not layers or layers != sorted(set(layers))
            or not all(type(x) is int and 0 <= x < depth for x in layers)
            or not isinstance(arm['prompt_a'], str) or not arm['prompt_a'].strip()
            or type(arm['direction_seed']) is not int or not 0 <= arm['direction_seed'] < 2 ** 32 or not isinstance(arm['name'], str)):
        raise ValueError('Invalid arm: %r' % (arm,))
    return arm


def push_directions(seed):
    """actrand: one random unit direction per action group, fixed for a run (a sign for the one-dimensional gripper)."""
    rng = np.random.default_rng(seed)
    return [u / np.linalg.norm(u) for u in (rng.standard_normal(hi - lo) for lo, hi in ACTION_GROUPS)]


def action_push(plain, delta, directions):
    """plain + a push that has, on every row and action group, the norm of delta's row group, along the group's direction.
    Returns (actions in plain's dtype, the largest absolute norm mismatch of the returned push)."""
    plain = np.asarray(plain); delta = np.asarray(delta, np.float64)
    push = np.zeros(plain.shape, np.float64)
    for (lo, hi), u in zip(ACTION_GROUPS, directions):
        push[:, lo:hi] = np.linalg.norm(delta[:, lo:hi], axis=1, keepdims=True) * u
    actions = (plain.astype(np.float64) + push).astype(plain.dtype)
    moved = actions.astype(np.float64) - plain.astype(np.float64)
    error = max(float(np.abs(np.linalg.norm(moved[:, lo:hi], axis=1) - np.linalg.norm(delta[:, lo:hi], axis=1)).max())
                for lo, hi in ACTION_GROUPS)
    return actions, error


def chunk_sha(actions):
    return hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()


def neuron_masks(sets, arm, depth, width):
    """(reference mask, applied mask, random direction flag) of a neuron arm, as [depth, width] booleans."""
    def mask(pairs):
        out = np.zeros((depth, width), bool)
        if len(pairs):
            idx = np.asarray(pairs, int); out[idx[:, 0], idx[:, 1]] = True
        return out
    every = np.zeros((depth, width), bool); every[list(sets['hooks'])] = True
    s = sets['sets'][arm['neuron_set']]
    top, rand = mask(s['top']), mask(s['random'])
    ref = every if arm['neurons'] == 'all' else top
    apply = dict(top=top, random=rand, all=every)[arm['neurons']]
    return ref, apply, int(arm['direction'] == 'random')


class SteeredPolicy:
    def __init__(self, policy, patcher, log_dir, neuron_sets=None):
        import jax
        self.policy, self.patcher, self.jax = policy, patcher, jax
        self.directory = Path(log_dir); self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock(); self.run = None; self.arm = None; self.n = 0; self.log = None; self.push = None
        self.neuron_sets = neuron_sets; self.masks = None

    def key(self, rng=None):
        return np.asarray(self.jax.random.key_data(self.policy._rng if rng is None else rng), dtype='<u4').tobytes().hex()

    def reset(self, obs):
        seed = obs['__paired_reset_rng__']
        if type(seed) is not int or not 0 <= seed < 2 ** 32:
            raise ValueError('Invalid seed')
        arm = check_arm(obs.get('__mech_arm__'), self.patcher.depth, self.neuron_sets)
        if self.log is not None:
            self.log.close()
        self.policy._rng = self.jax.random.key(seed); self.run = obs['run_id']; self.n = 0; self.arm = arm
        self.push = push_directions(arm['direction_seed']) if arm['kind'] == 'actrand' else None
        self.masks = neuron_masks(self.neuron_sets, arm, self.patcher.depth, self.patcher.mlp) if arm['kind'] == 'nrn' else None
        self.log = (self.directory/(str(self.run) + '.jsonl')).open('x')
        return {'seed': seed, 'run_id': self.run, 'requests': 0, 'key_hex': self.key(), 'arm': arm}

    def infer(self, obs):
        with self.lock:
            if '__paired_reset_rng__' in obs:
                return self.reset(obs)
            if self.arm is None:
                raise RuntimeError('Reset with an arm before the first request')
            if obs.get('prompt') == self.arm['prompt_a']:
                raise RuntimeError('The running prompt equals the old prompt')
            jax = self.jax; arm = self.arm; started = time.monotonic()
            before = self.key(); start = self.policy._rng; ih = digest(obs)
            self.policy._rng, sample = jax.random.split(start)  # as openpi's Policy.infer
            self.n += 1; after = self.key()
            layers = np.isin(np.arange(self.patcher.depth), arm['layers'])
            mode = 1 if arm['kind'] == 'rand' else 0
            direction = jax.random.fold_in(jax.random.key(arm['direction_seed']), self.n)
            old = dict(obs, prompt=arm['prompt_a']); scale = arm['lam'] - 1.; extra = {}
            if arm['kind'] == 'nrn':
                ref, apply, random_dir = self.masks
                actions, nstats = self.patcher.neuron_steer(dict(obs), old, sample, ref, apply, scale, random_dir, direction)
                stats = np.zeros(6)
                extra = dict(target=nstats[:, 0].tolist(), delta=nstats[:, 1].tolist(), written=nstats[:, 2].tolist())
            elif arm['kind'] in ('native', 'amp', 'rand'):
                actions, stats = self.patcher.steer(dict(obs), old, sample, layers, scale, mode, direction)
            elif arm['kind'] == 'actrand':
                amp, stats = self.patcher.steer(dict(obs), old, sample, layers, scale, 0, direction)
                plain, _ = self.patcher.steer(dict(obs), old, sample, layers, 0., 0, direction)
                actions, error = action_push(plain, np.asarray(amp, np.float64) - np.asarray(plain, np.float64), self.push)
                extra = dict(amp_shift_l2=float(np.linalg.norm(np.asarray(amp, np.float64) - plain)),
                             push_l2=float(np.linalg.norm(actions.astype(np.float64) - plain)), push_match_error=error)
            else:  # actdiff: nothing moved under B, and the same program under A for the native chunk of the old prompt
                plain, stats = self.patcher.steer(dict(obs), old, sample, layers, 0., 0, direction)
                under_a, _ = self.patcher.steer(dict(old), dict(obs), sample, layers, 0., 0, direction)
                gap = np.asarray(plain, np.float64) - np.asarray(under_a, np.float64)
                actions = (np.asarray(plain, np.float64) + scale * gap).astype(np.asarray(plain).dtype)
                extra = dict(ab_action_l2=float(np.linalg.norm(gap)), ab_action_linf=float(np.abs(gap).max()))
            entry = dict(request_index=self.n, run_id=self.run, arm=arm['name'], input_sha256=ih, rng_before=before, rng_after=after,
                         delta_k=float(stats[0]), diff_k=float(stats[1]), norm_k=float(stats[2]),
                         delta_v=float(stats[3]), diff_v=float(stats[4]), norm_v=float(stats[5]), chunk_sha256=chunk_sha(actions), **extra)
            if self.n == 1:
                if arm['kind'] == 'nrn':
                    plain, _ = self.patcher.neuron_steer(dict(obs), old, sample, self.masks[0], self.masks[1], 0., 0, direction)
                else:
                    plain, _ = self.patcher.steer(dict(obs), old, sample, layers, 0., 0, direction)
                self.policy._rng = start
                native = np.asarray(self.policy.infer(dict(obs))['actions'])
                if self.key() != after:
                    raise RuntimeError('Policy.infer advanced the RNG differently')
                entry.update(fidelity_linf=float(np.abs(plain - native).max()), first_shift_linf=float(np.abs(actions - plain).max()),
                             native_actions=native.tolist(), plain_actions=plain.tolist(), actions=np.asarray(actions).tolist())
            entry['seconds'] = time.monotonic() - started
            self.log.write(json.dumps(entry) + '\n'); self.log.flush()
            return {'actions': np.asarray(actions), 'diagnostic': {
                'input_sha256': ih, 'rng_before': before, 'rng_after': after, 'request_index': self.n, 'run_id': self.run,
                'output_replayed': False, 'native_output_linf_difference': 0., 'native_chunk_sha256': entry['chunk_sha256'], 'arm': arm['name']}}

    def close(self):
        if self.log is not None:
            self.log.close(); self.log = None

    def warm_up(self, example):
        """Compiles the steering program and Policy.infer on an input of the served shapes, then restores the RNG."""
        saved = self.policy._rng
        try:
            layers = np.zeros(self.patcher.depth, bool); layers[0] = True
            key = self.jax.random.key(0)
            self.patcher.steer(dict(example), dict(example, prompt='warm up'), key, layers, 1., 1, key)
            if self.neuron_sets is not None:
                mask = np.zeros((self.patcher.depth, self.patcher.mlp), bool); mask[0, 0] = True
                self.patcher.neuron_steer(dict(example), dict(example, prompt='warm up'), key, mask, mask, 1., 0, key)
            self.policy.infer(dict(example))
        finally:
            self.policy._rng = saved


def example_input():
    return {'observation/image': np.zeros((224, 224, 3), np.uint8), 'observation/wrist_image': np.zeros((224, 224, 3), np.uint8),
            'observation/state': np.zeros(8), 'prompt': 'put the bowl on the plate'}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--port', type=int, required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--log-dir', required=True); p.add_argument('--neurons'); a = p.parse_args()
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    import patching as P
    policy = policy_config.create_trained_policy(config.get_config('pi05_libero'), a.checkpoint)
    if policy._is_pytorch_model:
        raise RuntimeError('JAX policy required')
    if a.neurons:
        import neurons as N
        with open(a.neurons) as stream:
            sets = json.load(stream)
        server = SteeredPolicy(policy, N.NeuronPatcher(policy), a.log_dir, sets)
    else:
        server = SteeredPolicy(policy, P.Patcher(policy), a.log_dir)
    # Compile before the port opens: the websocket keepalive would drop a client waiting through a first-request compile.
    started = time.monotonic(); server.warm_up(example_input()); print('warm up %.1f s' % (time.monotonic() - started), flush=True)
    WebsocketPolicyServer(server, host='127.0.0.1', port=a.port, metadata=policy.metadata).serve_forever()


if __name__ == '__main__':
    main()
